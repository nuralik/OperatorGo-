"""
SGF game loader using sgfmill.
Yields (board_tensor, policy_target, value_target) training samples
from a collection of SGF files.

policy_target : (N*N + 1,) one-hot  — the move played (last index = pass)
value_target  : float  — +1 if the player to move won, -1 if lost
"""

import os
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple

import sgfmill.sgf as sgf
import sgfmill.boards as sgfboards

from go_env.board import GoBoard, BLACK, WHITE, EMPTY


def _sgf_color_to_int(c: str) -> int:
    return BLACK if c == 'b' else WHITE


def _coord_to_rowcol(coord, size: int):
    """sgfmill uses (row, col) with row=0 at top."""
    if coord is None:
        return None
    return coord  # already (row, col)


def load_sgf_game(path: str) -> list:
    """
    Parse one SGF file and return a list of training samples:
        (board_np, policy_idx, value)
    board_np  : (3, N, N) float32
    policy_idx: int  (0..N*N-1 for board move, N*N for pass)
    value     : float (+1 or -1 from the current player's perspective)
    """
    with open(path, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    size = game.get_size()
    if size not in (9, 13, 19):
        return []

    # Determine winner
    result = game.get_root().get('RE')
    if result is None:
        return []
    result = result.upper()
    if result.startswith('B'):
        game_winner = BLACK
    elif result.startswith('W'):
        game_winner = WHITE
    else:
        return []  # jigo or unknown

    board = GoBoard(size)
    samples = []

    main_seq = game.get_main_sequence()
    for node in main_seq[1:]:  # skip root
        color_str, move_coord = node.get_move()
        if color_str is None:
            continue

        color = _sgf_color_to_int(color_str)
        board_snapshot = board.to_tensor()  # (3, N, N)

        # Policy target
        if move_coord is None:
            policy_idx = size * size  # pass
        else:
            r, c = move_coord
            policy_idx = r * size + c

        # Value target: +1 if current player wins
        value = 1.0 if color == game_winner else -1.0

        samples.append((board_snapshot, policy_idx, value))

        # Play the move on the board
        try:
            move = None if move_coord is None else (move_coord[0], move_coord[1])
            board.play(color, move)
        except AssertionError:
            break  # illegal move in record — skip rest

    return samples


def iter_sgf_directory(
    directory: str,
    size_filter: int = 9,
) -> Iterator[Tuple[np.ndarray, int, float]]:
    """
    Recursively yield (board_np, policy_idx, value) from all SGF files
    in `directory` that match `size_filter`.
    """
    root = Path(directory)
    for path in root.rglob('*.sgf'):
        try:
            for sample in load_sgf_game(str(path)):
                board_np, policy_idx, value = sample
                if board_np.shape[1] == size_filter:
                    yield board_np, policy_idx, value
        except Exception:
            continue  # skip malformed files


def build_dataset(
    directory: str,
    size_filter: int = 9,
    max_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all SGF samples into numpy arrays.
    Returns:
        boards   : (N_samples, 3, size, size)
        policies : (N_samples,) int indices
        values   : (N_samples,) float
    """
    boards, policies, values = [], [], []
    for i, (b, p, v) in enumerate(iter_sgf_directory(directory, size_filter)):
        boards.append(b)
        policies.append(p)
        values.append(v)
        if max_samples and i + 1 >= max_samples:
            break

    return (
        np.array(boards, dtype=np.float32),
        np.array(policies, dtype=np.int64),
        np.array(values, dtype=np.float32),
    )
