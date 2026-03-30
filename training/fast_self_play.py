"""
GPU-efficient synchronized self-play — no MCTS.

All games advance one move at a time together.
Every step = one GPU batch evaluation for all active games.
With 64 games: 80 steps × batch_size=64 → GPU stays busy the whole time.

Policy target: one-hot of the sampled move (outcome-weighted REINFORCE).
Value target:  game outcome +1 / -1.
"""

import numpy as np
import torch

from go_env.board import GoBoard, BLACK, WHITE
from training.augment import _transform_board, transform_policy_vec


def play_games_batched(model, device: str, n_games: int = 64,
                       board_size: int = 9,
                       temp_moves: int = 20,
                       augment: bool = True) -> list:
    """
    Play n_games simultaneously in a single GPU-batched loop.

    Returns list of (board_tensor, policy_target, value) samples.
        board_tensor  : (3, N, N) float32
        policy_target : (N*N+1,) float32  one-hot of move played
        value         : float  +1 win / -1 loss / 0 draw
    """
    boards   = [GoBoard(board_size) for _ in range(n_games)]
    colors   = [BLACK] * n_games
    histories = [[] for _ in range(n_games)]   # (board_t, move_idx, color)
    done     = [False] * n_games
    steps    = [0] * n_games
    max_moves = board_size * board_size * 3

    model.eval()

    while not all(done):
        active = [i for i in range(n_games) if not done[i]]

        # ── Build batch ───────────────────────────────────────────────────
        batch = []
        for i in active:
            t = boards[i].to_tensor()
            if colors[i] == WHITE:
                t = np.stack([t[1], t[0], t[2]], axis=0)
            batch.append(t)

        with torch.no_grad():
            t_gpu  = torch.from_numpy(np.stack(batch)).to(device)
            logits, values = model(t_gpu)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            vals   = values.cpu().numpy()

        # ── Each game picks a move ────────────────────────────────────────
        for j, i in enumerate(active):
            legal  = boards[i].legal_moves(colors[i])
            n_pos  = board_size * board_size

            # Build legal mask
            mask = np.zeros(n_pos + 1, dtype=np.float32)
            for move in legal:
                mask[n_pos if move is None else move[0] * board_size + move[1]] = 1.0

            # Mask and renormalise
            p = probs[j] * mask
            s = p.sum()
            p = p / s if s > 1e-8 else mask / mask.sum()

            # Temperature: explore early, greedy later
            temperature = 1.0 if steps[i] < temp_moves else 0.05
            if temperature < 0.1:
                move_idx = int(np.argmax(p))
            else:
                p_temp = p ** (1.0 / temperature)
                p_temp /= p_temp.sum()
                move_idx = int(np.random.choice(len(p_temp), p=p_temp))

            # Record position
            board_t = boards[i].to_tensor()
            if colors[i] == WHITE:
                board_t = np.stack([board_t[1], board_t[0], board_t[2]], axis=0)
            histories[i].append((board_t, move_idx, colors[i]))

            # Play
            move = None if move_idx == n_pos else (move_idx // board_size,
                                                    move_idx % board_size)
            game_over = boards[i].play(colors[i], move)
            steps[i] += 1
            colors[i] = -colors[i]

            if game_over or steps[i] >= max_moves:
                done[i] = True

    # ── Assign outcomes + build samples ──────────────────────────────────
    all_samples = []
    for i in range(n_games):
        winner = boards[i].winner()
        for board_t, move_idx, color in histories[i]:
            value = 0.0 if winner == 0 else (1.0 if winner == color else -1.0)

            policy_target = np.zeros(board_size * board_size + 1, dtype=np.float32)
            policy_target[move_idx] = 1.0

            if augment:
                for k in range(4):
                    for flip in (False, True):
                        all_samples.append((
                            _transform_board(board_t, k, flip),
                            transform_policy_vec(policy_target, board_size, k, flip),
                            value,
                        ))
            else:
                all_samples.append((board_t, policy_target, value))

    return all_samples
