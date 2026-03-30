"""
Self-play game generation.
Plays one full game using MCTS + DeepONet and returns training samples.
"""

import numpy as np
from go_env.board import GoBoard, BLACK, WHITE
from mcts.search import MCTS
from training.augment import _transform_board, transform_policy_vec


def play_game(mcts: MCTS, board_size: int = 9,
              temp_threshold: int = 15) -> list:
    """
    Play one self-play game to completion.

    Returns list of (board_tensor, policy_target, value) tuples.
        board_tensor  : (3, N, N) float32
        policy_target : (N*N+1,) float32 — visit count distribution
        value         : float — +1 if this player won, -1 if lost
    """
    board   = GoBoard(board_size)
    color   = BLACK
    history = []   # (board_tensor, policy_vec, color)
    move_num = 0

    while True:
        # Temperature: explore early, greedy later
        temp = 1.0 if move_num < temp_threshold else 0.0

        move_probs, _ = mcts.get_policy(board, color, temperature=temp)

        # Build full policy vector (N*N+1)
        policy_vec = np.zeros(board_size * board_size + 1, dtype=np.float32)
        for move, prob in move_probs.items():
            idx = board_size * board_size if move is None else move[0] * board_size + move[1]
            policy_vec[idx] = prob

        # Record state from current player's perspective
        t = board.to_tensor()
        if color == WHITE:
            t = np.stack([t[1], t[0], t[2]], axis=0)
        history.append((t, policy_vec, color))

        # Sample a move
        moves  = list(move_probs.keys())
        probs  = np.array([move_probs[m] for m in moves], dtype=np.float64)
        probs /= probs.sum()
        chosen = moves[np.random.choice(len(moves), p=probs)]

        done = board.play(color, chosen)
        move_num += 1
        color = -color

        if done or move_num > board_size * board_size * 4:
            break

    # Determine winner and assign values, apply D8 augmentation to every sample
    winner = board.winner()
    samples = []
    for board_t, policy_vec, player_color in history:
        value = 0.0 if winner == 0 else (1.0 if winner == player_color else -1.0)
        for k in range(4):
            for flip in (False, True):
                aug_board  = _transform_board(board_t, k, flip)
                aug_policy = transform_policy_vec(policy_vec, board_size, k, flip)
                samples.append((aug_board, aug_policy, value))

    return samples
