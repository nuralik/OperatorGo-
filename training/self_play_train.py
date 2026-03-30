"""
Self-play training loop for DeepONet Go AI.

Usage:
    # Start fresh (uses supervised checkpoint as starting point if available)
    python -m training.self_play_train

    # Resume from a self-play checkpoint
    python -m training.self_play_train --resume

    # Custom settings
    python -m training.self_play_train --simulations 100 --games_per_iter 20

Press Ctrl+C to stop cleanly — progress is saved after every iteration.
"""

import argparse
import collections
import os
import signal
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F

from models.deeponet import DeepONetGo
from training.self_play import play_game


# ── Parallel worker ───────────────────────────────────────────────────────────

WORKER_WEIGHTS_PATH = 'checkpoints/_worker_weights.pt'

def _play_game_worker(args_tuple):
    """
    Runs in a separate process.
    Loads model weights from a shared temp file (avoids passing large
    state_dict over IPC pipes on every call).
    """
    board_size, simulations, filters, res_blocks, latent_dim = args_tuple
    from mcts.search import MCTS
    model = DeepONetGo(board_size=board_size, filters=filters,
                       n_res_blocks=res_blocks, latent_dim=latent_dim)
    model.load_state_dict(
        torch.load(WORKER_WEIGHTS_PATH, map_location='cpu', weights_only=False)
    )
    mcts = MCTS(model, device='cpu', n_simulations=simulations)
    return play_game(mcts, board_size=board_size)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer using deque — O(1) append and automatic eviction,
    no memory reallocation on trim.
    Samples batches by random index selection to avoid converting the
    full buffer to tensors every iteration.
    """

    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.boards   = collections.deque(maxlen=max_size)
        self.policies = collections.deque(maxlen=max_size)
        self.values   = collections.deque(maxlen=max_size)

    def add(self, samples: list):
        for b, p, v in samples:
            self.boards.append(b)
            self.policies.append(p)
            self.values.append(v)

    def __len__(self):
        return len(self.boards)

    def sample_batch(self, batch_size: int):
        """Sample a random batch directly — no full-buffer copy."""
        n = len(self.boards)
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        boards   = np.array([self.boards[i]   for i in idx], dtype=np.float32)
        policies = np.array([self.policies[i] for i in idx], dtype=np.float32)
        values   = np.array([self.values[i]   for i in idx], dtype=np.float32)
        return (
            torch.from_numpy(boards),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )

    def state_dict(self):
        return {
            'boards':   list(self.boards),
            'policies': list(self.policies),
            'values':   list(self.values),
        }

    def load_state_dict(self, d):
        self.boards   = collections.deque(d['boards'],   maxlen=self.max_size)
        self.policies = collections.deque(d['policies'], maxlen=self.max_size)
        self.values   = collections.deque(d['values'],   maxlen=self.max_size)


# ── Training step ─────────────────────────────────────────────────────────────

def train_on_buffer(model, buffer: ReplayBuffer, optimizer,
                    batch_size: int, n_steps: int, device: str) -> dict:
    model.train()
    total_loss = total_pl = total_vl = 0.0

    for _ in range(n_steps):
        b, p, v = buffer.sample_batch(batch_size)
        b, p, v = b.to(device), p.to(device), v.to(device)

        policy_logits, value_pred = model(b)

        log_probs   = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(p * log_probs).sum(dim=-1).mean()
        value_loss  = F.mse_loss(value_pred, v)
        loss        = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pl   += policy_loss.item()
        total_vl   += value_loss.item()

    return {
        'loss':        total_loss / n_steps,
        'policy_loss': total_pl   / n_steps,
        'value_loss':  total_vl   / n_steps,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

CHECKPOINT_PATH = 'checkpoints/self_play_latest.pt'
CHECKPOINT_TMP  = 'checkpoints/self_play_latest.tmp.pt'
BEST_PATH       = 'checkpoints/self_play_best.pt'
BEST_TMP        = 'checkpoints/self_play_best.tmp.pt'


def save_checkpoint(model, optimizer, buffer, iteration, best_loss, args):
    """Atomic save: write to .tmp then os.replace() — safe against crashes."""
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'iteration':   iteration,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'buffer':      buffer.state_dict(),
        'best_loss':   best_loss,
        'args':        vars(args),
    }, CHECKPOINT_TMP)
    os.replace(CHECKPOINT_TMP, CHECKPOINT_PATH)


def save_best(model, iteration, args):
    torch.save({
        'iteration':   iteration,
        'model_state': model.state_dict(),
        'args':        vars(args),
    }, BEST_TMP)
    os.replace(BEST_TMP, BEST_PATH)


def load_checkpoint(model, optimizer, buffer, device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    buffer.load_state_dict(ckpt['buffer'])
    return ckpt['iteration'], ckpt.get('best_loss', float('inf'))


# ── Progress display ──────────────────────────────────────────────────────────

def print_header():
    print(f"\n{'Iter':>5}  {'Games':>6}  {'Buf':>7}  "
          f"{'Loss':>8}  {'Pol Loss':>8}  {'Val Loss':>8}  "
          f"{'W/D/L':>9}  {'Time':>6}")
    print('─' * 75)


def print_row(it, total_games, buf_size, metrics, wdl, elapsed):
    w, d, l = wdl
    print(f"{it:>5}  {total_games:>6}  {buf_size:>7}  "
          f"{metrics['loss']:>8.4f}  {metrics['policy_loss']:>8.4f}  "
          f"{metrics['value_loss']:>8.4f}  "
          f"{w:>3}/{d:>2}/{l:>3}  {elapsed:>5.1f}s",
          flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',         action='store_true')
    parser.add_argument('--board_size',     type=int,   default=9)
    parser.add_argument('--simulations',    type=int,   default=100)
    parser.add_argument('--games_per_iter', type=int,   default=20)
    parser.add_argument('--train_steps',    type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--lr',             type=float, default=2e-4)
    parser.add_argument('--buffer_size',    type=int,   default=200_000)
    parser.add_argument('--filters',        type=int,   default=64)
    parser.add_argument('--res_blocks',     type=int,   default=5)
    parser.add_argument('--latent_dim',     type=int,   default=256)
    parser.add_argument('--max_iters',      type=int,   default=1000)
    parser.add_argument('--workers',        type=int,   default=8)
    args = parser.parse_args()

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    n_workers = min(args.workers, 8)

    print(f"Device     : {device}")
    print(f"Simulations: {args.simulations} per move")
    print(f"Games/iter : {args.games_per_iter}")
    print(f"Train steps: {args.train_steps}")
    print(f"Workers    : {n_workers} parallel games")

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

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}\n")

    # Graceful Ctrl+C
    stop_requested = False
    def _handler(sig, frame):
        nonlocal stop_requested
        print("\n\nCtrl+C received — will stop after current games finish...")
        stop_requested = True
    signal.signal(signal.SIGINT, _handler)

    # Worker args template (no state_dict — workers load from file)
    worker_arg = (args.board_size, args.simulations,
                  args.filters, args.res_blocks, args.latent_dim)

    total_games = 0
    print_header()

    for iteration in range(start_iter, start_iter + args.max_iters):
        if stop_requested:
            print(f"Stopped before iteration {iteration}.")
            break

        t0 = time.time()
        wins = draws = losses = 0

        # ── Write weights to shared file for workers ──────────────────────
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.cpu().state_dict(), WORKER_WEIGHTS_PATH)
        model.to(device)

        # ── Self-play phase — non-blocking so Ctrl+C is responsive ────────
        model.eval()
        with mp.Pool(processes=n_workers) as pool:
            async_result = pool.map_async(_play_game_worker,
                                          [worker_arg] * args.games_per_iter)
            while not async_result.ready():
                if stop_requested:
                    pool.terminate()
                    break
                async_result.wait(timeout=2.0)

            if stop_requested:
                print(f"\nSaved checkpoint at iteration {iteration - 1}.")
                print("Resume with: python -m training.self_play_train --resume")
                break

            results = async_result.get()

        for samples in results:
            buffer.add(samples)
            total_games += 1
            if samples and samples[-1][2] == 1.0:
                wins += 1
            elif samples and samples[-1][2] == -1.0:
                losses += 1
            else:
                draws += 1

        # ── Training phase ────────────────────────────────────────────────
        if len(buffer) >= args.batch_size:
            metrics = train_on_buffer(model, buffer, optimizer,
                                      args.batch_size, args.train_steps, device)
        else:
            metrics = {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

        elapsed = time.time() - t0
        print_row(iteration, total_games, len(buffer),
                  metrics, (wins, draws, losses), elapsed)

        # ── Atomic checkpoint save ────────────────────────────────────────
        save_checkpoint(model, optimizer, buffer, iteration, best_loss, args)
        if metrics['loss'] > 0 and metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_best(model, iteration, args)

    print(f"\nDone. Total games played: {total_games}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
