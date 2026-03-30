"""
D4 symmetry augmentation for Go boards.

A square board has 8 symmetries: 4 rotations x 2 reflections.
We apply the same transform to both the board tensor and the policy target.

Policy index layout (row-major):
    idx = row * size + col   for board moves
    idx = size*size          for pass (never transformed)
"""

import numpy as np


def _transform_board(board: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """
    board : (3, N, N)
    k     : number of 90-degree CCW rotations (0-3)
    flip  : whether to flip horizontally after rotation
    """
    b = np.rot90(board, k=k, axes=(1, 2))
    if flip:
        b = np.flip(b, axis=2).copy()
    return b


def _transform_policy(idx: int, size: int, k: int, flip: bool) -> int:
    """Transform a flat policy index under the same D4 transform."""
    if idx == size * size:
        return idx  # pass index unchanged

    row, col = divmod(idx, size)

    # Apply k CCW rotations
    for _ in range(k):
        row, col = col, size - 1 - row

    # Apply horizontal flip
    if flip:
        col = size - 1 - col

    return row * size + col


def augment_sample(board: np.ndarray, policy_idx: int, value: float, size: int):
    """
    Yield all 8 symmetry variants of a single (board, policy, value) sample.
    board : (3, N, N) float32
    """
    for k in range(4):
        for flip in (False, True):
            aug_board  = _transform_board(board, k, flip)
            aug_policy = _transform_policy(policy_idx, size, k, flip)
            yield aug_board, aug_policy, value


def transform_policy_vec(policy_vec: np.ndarray, size: int,
                         k: int, flip: bool) -> np.ndarray:
    """
    Apply a D4 transform to a soft policy distribution (N*N+1,).
    Used for self-play samples where targets are visit-count distributions,
    not hard indices.
    """
    new_policy = np.zeros_like(policy_vec)
    for idx in range(size * size):
        new_idx = _transform_policy(idx, size, k, flip)
        new_policy[new_idx] = policy_vec[idx]
    new_policy[size * size] = policy_vec[size * size]  # pass unchanged
    return new_policy


def augment_dataset(
    boards: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> tuple:
    """
    Expand a dataset by all 8 D4 symmetries.
    Input  shape: (N, 3, size, size)
    Output shape: (8N, 3, size, size)
    """
    if boards.ndim < 3 or len(boards) == 0:
        raise ValueError(f"augment_dataset received empty or malformed boards array: shape={boards.shape}. "
                         f"Check that SGF data exists and was loaded correctly.")
    size = boards.shape[2]
    out_boards, out_policies, out_values = [], [], []

    for b, p, v in zip(boards, policies, values):
        for ab, ap, av in augment_sample(b, int(p), float(v), size):
            out_boards.append(ab)
            out_policies.append(ap)
            out_values.append(av)

    return (
        np.array(out_boards,   dtype=np.float32),
        np.array(out_policies, dtype=np.int64),
        np.array(out_values,   dtype=np.float32),
    )
