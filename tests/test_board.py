"""Tests for the GoBoard rules engine."""

import sys
sys.path.insert(0, '/Users/nurali/Documents/OperatorGo')

import numpy as np
import pytest
from go_env.board import GoBoard, BLACK, WHITE, EMPTY


def test_initial_board():
    b = GoBoard(9)
    assert b.board.shape == (9, 9)
    assert np.all(b.board == EMPTY)


def test_place_stone():
    b = GoBoard(9)
    b.play(BLACK, (4, 4))
    assert b.board[4, 4] == BLACK


def test_capture_single():
    b = GoBoard(9)
    # Surround a white stone
    b.play(WHITE, (0, 1))
    b.play(BLACK, (0, 0))
    b.play(BLACK, (0, 2))
    b.play(BLACK, (1, 1))
    # White at (0,1) should be captured
    assert b.board[0, 1] == EMPTY
    assert b.captured[BLACK] == 1


def test_ko_rule():
    b = GoBoard(9)
    # Ko requires: capturing stone has exactly 1 liberty (the just-captured point).
    # Setup: white at (2,2), black surrounds 3 sides: (1,2),(2,1),(2,3)
    #        white blocks below-left/right/below: (3,1),(3,3),(4,2)
    # Black plays (3,2) → captures white (2,2) → black (3,2) has only 1 liberty at (2,2)
    b.play(WHITE, (2, 2))
    b.play(BLACK, (1, 2))
    b.play(BLACK, (2, 1))
    b.play(BLACK, (2, 3))
    b.play(WHITE, (3, 1))
    b.play(WHITE, (3, 3))
    b.play(WHITE, (4, 2))
    b.play(BLACK, (3, 2))   # captures white at (2,2)
    assert b.board[2, 2] == EMPTY       # white was captured
    assert b.ko_point == (2, 2)         # ko point set
    assert not b.is_legal(WHITE, 2, 2)  # immediate recapture illegal


def test_suicide_illegal():
    b = GoBoard(9)
    # Corner with no liberties
    b.play(BLACK, (0, 1))
    b.play(BLACK, (1, 0))
    # White cannot play at (0,0) — suicide
    assert not b.is_legal(WHITE, 0, 0)


def test_suicide_capture_is_legal():
    b = GoBoard(9)
    # White fills in, but captures black — net positive liberties
    b.play(BLACK, (0, 0))
    b.play(WHITE, (0, 1))
    b.play(WHITE, (1, 0))
    # Black at (0,0) is surrounded; White already played those.
    # Now black is captured; confirm it's empty
    assert b.board[0, 0] == EMPTY


def test_pass():
    b = GoBoard(9)
    done = b.play(BLACK, None)
    assert not done
    done = b.play(WHITE, None)
    assert done  # two passes → game over


def test_legal_moves_includes_pass():
    b = GoBoard(9)
    moves = b.legal_moves(BLACK)
    assert None in moves


def test_copy_independence():
    b = GoBoard(9)
    b.play(BLACK, (4, 4))
    c = b.copy()
    c.play(WHITE, (4, 5))
    assert b.board[4, 5] == EMPTY  # original unaffected


def test_to_tensor():
    b = GoBoard(9)
    b.play(BLACK, (0, 0))
    b.play(WHITE, (8, 8))
    t = b.to_tensor()
    assert t.shape == (3, 9, 9)
    assert t[0, 0, 0] == 1.0   # black channel
    assert t[1, 8, 8] == 1.0   # white channel
    assert t[2, 4, 4] == 1.0   # empty channel


def test_tromp_taylor_score_empty():
    b = GoBoard(9)
    # Empty board: no stones, all territory neutral
    s = b.score()
    assert s['BLACK'] == 0
    assert s['WHITE'] == 0


def test_tromp_taylor_score_one_stone():
    b = GoBoard(5)
    # Black stone in center of 5x5
    b.play(BLACK, (2, 2))
    s = b.score()
    # Black has 1 stone + all 24 empty (all reachable only by black)
    assert s['BLACK'] == 25
    assert s['WHITE'] == 0


def test_to_field():
    b = GoBoard(9)
    b.play(BLACK, (0, 0))
    b.play(WHITE, (8, 8))
    f = b.to_field()
    assert f.shape == (9, 9)
    assert f[0, 0] == 1.0
    assert f[8, 8] == -1.0
    assert f[4, 4] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
