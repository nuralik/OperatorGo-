"""Tests for D4 augmentation correctness."""

import sys
sys.path.insert(0, '/Users/nurali/Documents/OperatorGo')

import numpy as np
import pytest
from training.augment import augment_sample, augment_dataset, _transform_policy


def test_eight_variants():
    board = np.random.rand(3, 9, 9).astype(np.float32)
    variants = list(augment_sample(board, 5, 1.0, 9))
    assert len(variants) == 8


def test_pass_index_unchanged():
    board = np.zeros((3, 9, 9), dtype=np.float32)
    for _, p, _ in augment_sample(board, 81, 1.0, 9):
        assert p == 81  # pass always stays pass


def test_policy_index_roundtrip():
    # After 4 CCW rotations we should get back to the original index
    size = 9
    for idx in [0, 5, 40, 80]:
        result = idx
        for _ in range(4):
            result = _transform_policy(result, size, k=1, flip=False)
        assert result == idx, f"Rotation roundtrip failed for idx={idx}"


def test_board_shape_preserved():
    board = np.random.rand(3, 9, 9).astype(np.float32)
    for ab, _, _ in augment_sample(board, 0, 1.0, 9):
        assert ab.shape == (3, 9, 9)


def test_augment_dataset_size():
    boards   = np.random.rand(10, 3, 9, 9).astype(np.float32)
    policies = np.zeros(10, dtype=np.int64)
    values   = np.ones(10, dtype=np.float32)
    ab, ap, av = augment_dataset(boards, policies, values)
    assert len(ab) == 80  # 10 * 8


def test_corner_stone_rotation():
    # Stone at top-left corner (row=0,col=0) → after 1 CCW rotation → (row=0, col=8)
    size = 9
    idx = 0 * size + 0  # (0,0) → flat index 0
    rotated = _transform_policy(idx, size, k=1, flip=False)
    row, col = divmod(rotated, size)
    assert (row, col) == (0, 8), f"Expected (0,8) got ({row},{col})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
