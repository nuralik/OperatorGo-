"""
Fast GPU self-play training — no MCTS.

Plays games_per_iter games simultaneously in one GPU-batched loop,
trains on outcomes, repeats. Designed to saturate T4/A100 overnight.

Usage:
    python -m training.fast_train                    # fresh start
    python -m training.fast_train --resume           # continue
"""

import argparse
import os
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.deeponet import DeepONetGo
from training.fast_self_play import play_games_batched
from training.self_play_train import (
    ReplayBuffer, train_on_buffer,
    save_checkpoint, save_best, load_checkpoint,
    print_header, print_row,
    CHECKPOINT_PATH, BEST_PATH,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',         action='store_true')
    parser.add_argument('--board_size',     type=int,   default=9)
    parser.add_argument('--games_per_iter', type=int,   default=64,
                        help='Games per iteration — all run in one GPU batch')
    parser.add_argument('--train_steps',    type=int,   default=400)
    parser.add_argument('--batch_size',     type=int,   default=512)
    parser.add_argument('--lr',             type=float, default=2e-4)
    parser.add_argument('--buffer_size',    type=int,   default=500_000)
    parser.add_argument('--filters',        type=int,   default=64)
    parser.add_argument('--res_blocks',     type=int,   default=5)
    parser.add_argument('--latent_dim',     type=int,   default=256)
    parser.add_argument('--max_iters',      type=int,   default=5000)
    parser.add_argument('--temp_moves',     type=int,   default=20,
                        help='Moves with temperature=1 before going greedy')
    args = parser.parse_args()

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )

    print(f"Device      : {device}")
    print(f"Games/iter  : {args.games_per_iter}  (all batched on GPU)")
    print(f"Train steps : {args.train_steps}")
    print(f"Batch size  : {args.batch_size}")

    model = DeepONetGo(
        board_size=args.board_size,
        filters=args.filters,
        n_res_blocks=args.res_blocks,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer    = ReplayBuffer(max_size=args.buffer_size)

    start_iter = 1
    best_loss  = float('inf')

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        start_iter, best_loss = load_checkpoint(model, optimizer, buffer, device)
        start_iter += 1
        print(f"Resumed from iteration {start_iter - 1}  (buffer: {len(buffer)} samples)")
    elif os.path.exists('checkpoints/best_deeponet.pt'):
        ckpt = torch.load('checkpoints/best_deeponet.pt', map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print("Loaded supervised DeepONet weights as starting point.")
    else:
        print("Starting from random weights.")

    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}\n")

    stop_requested = False
    def _handler(sig, frame):
        nonlocal stop_requested
        print("\nCtrl+C — finishing current iteration then saving...")
        stop_requested = True
    signal.signal(signal.SIGINT, _handler)

    total_games = 0
    print_header()

    for iteration in range(start_iter, start_iter + args.max_iters):
        if stop_requested:
            break

        t0 = time.time()

        # ── Self-play phase — fully batched on GPU ────────────────────────
        model.eval()
        samples = play_games_batched(
            model, device,
            n_games=args.games_per_iter,
            board_size=args.board_size,
            temp_moves=args.temp_moves,
        )
        buffer.add(samples)
        total_games += args.games_per_iter

        # Count outcomes (last history entry per game reflects final value)
        # Every 8 samples (D8 aug) from same position share same value
        outcomes = [s[2] for s in samples[::8]]   # one per position, stride aug
        wins   = sum(1 for v in outcomes if v > 0)
        losses = sum(1 for v in outcomes if v < 0)
        draws  = sum(1 for v in outcomes if v == 0)
        # Rough game-level W/D/L (not exact but indicative)
        game_wins   = wins   * args.games_per_iter // max(wins + losses + draws, 1)
        game_losses = losses * args.games_per_iter // max(wins + losses + draws, 1)
        game_draws  = args.games_per_iter - game_wins - game_losses

        # ── Training phase ────────────────────────────────────────────────
        if len(buffer) >= args.batch_size:
            metrics = train_on_buffer(model, buffer, optimizer,
                                      args.batch_size, args.train_steps, device)
        else:
            metrics = {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

        elapsed = time.time() - t0
        print_row(iteration, total_games, len(buffer),
                  metrics, (game_wins, game_draws, game_losses), elapsed)

        save_checkpoint(model, optimizer, buffer, iteration, best_loss, args)
        if metrics['loss'] > 0 and metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_best(model, iteration, args)

    print(f"\nDone. Total games: {total_games}")


if __name__ == '__main__':
    main()
