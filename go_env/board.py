"""
Minimal Go rules engine for NxN boards (default 9x9).
Supports: legal moves, ko rule, captures, Tromp-Taylor scoring.

Color convention:
    BLACK = 1, WHITE = -1, EMPTY = 0
"""

import numpy as np
from typing import Optional

BLACK = 1
WHITE = -1
EMPTY = 0


class GoBoard:
    def __init__(self, size: int = 9):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.ko_point: Optional[tuple] = None          # (row, col) or None
        self.captured = {BLACK: 0, WHITE: 0}
        self.move_history: list = []                   # list of (color, move)
        self.pass_count = 0

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _on_board(self, r, c) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _neighbors(self, r, c):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if self._on_board(nr, nc):
                yield nr, nc

    def _group(self, r, c):
        """BFS — returns (group_cells, liberties) as sets."""
        color = self.board[r, c]
        group, liberties = set(), set()
        stack = [(r, c)]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.add(cur)
            for nr, nc in self._neighbors(*cur):
                if self.board[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return group, liberties

    def _remove_group(self, r, c) -> int:
        """Remove a group and return number of stones captured."""
        group, _ = self._group(r, c)
        for gr, gc in group:
            self.board[gr, gc] = EMPTY
        return len(group)

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def is_legal(self, color: int, r: int, c: int) -> bool:
        """Check if placing `color` at (r, c) is legal."""
        if not self._on_board(r, c):
            return False
        if self.board[r, c] != EMPTY:
            return False
        if self.ko_point == (r, c):
            return False

        # Simulate placement to check suicide / ko
        self.board[r, c] = color
        opponent = -color

        # Capture any opponent groups with no liberties
        captured_points = []
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == opponent:
                _, libs = self._group(nr, nc)
                if not libs:
                    captured_points.append((nr, nc))

        # Check suicide: after captures, does placed stone have liberties?
        _, my_libs = self._group(r, c)
        is_suicide = len(my_libs) == 0 and not captured_points

        # Undo simulation
        self.board[r, c] = EMPTY

        return not is_suicide

    def legal_moves(self, color: int) -> list:
        """Returns list of legal (r, c) moves including pass (None)."""
        moves = [None]  # pass is always legal
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(color, r, c):
                    moves.append((r, c))
        return moves

    def play(self, color: int, move: Optional[tuple]) -> bool:
        """
        Play a move. move=None is a pass.
        Returns True if game over (two consecutive passes).
        """
        if move is None:
            self.pass_count += 1
            self.ko_point = None
            self.move_history.append((color, None))
            return self.pass_count >= 2

        r, c = move
        assert self.is_legal(color, r, c), f"Illegal move: {color} at ({r},{c})"
        self.pass_count = 0

        self.board[r, c] = color
        opponent = -color

        # Capture opponent groups with no liberties
        total_captured = 0
        captured_groups = []
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == opponent:
                _, libs = self._group(nr, nc)
                if not libs:
                    captured_groups.append((nr, nc))

        potential_ko = None
        for gr, gc in captured_groups:
            n = self._remove_group(gr, gc)
            total_captured += n
            if n == 1 and len(captured_groups) == 1:
                potential_ko = (gr, gc)

        self.captured[color] += total_captured

        # Ko: only set if exactly one stone captured and the played group
        # has exactly one liberty (the just-captured point)
        if potential_ko is not None:
            _, my_libs = self._group(r, c)
            self.ko_point = potential_ko if len(my_libs) == 1 else None
        else:
            self.ko_point = None

        self.move_history.append((color, move))
        return False

    # ------------------------------------------------------------------
    # Scoring (Tromp-Taylor)
    # ------------------------------------------------------------------

    def score(self) -> dict:
        """
        Tromp-Taylor scoring. Returns {'BLACK': float, 'WHITE': float}.
        Komi is NOT applied here — caller adds 6.5 to WHITE.
        """
        black_territory = 0
        white_territory = 0

        visited = np.zeros((self.size, self.size), dtype=bool)

        for r in range(self.size):
            for c in range(self.size):
                if visited[r, c] or self.board[r, c] != EMPTY:
                    visited[r, c] = True
                    continue

                # BFS over empty region
                region = []
                touches = set()
                stack = [(r, c)]
                while stack:
                    cur = stack.pop()
                    if visited[cur[0], cur[1]]:
                        continue
                    visited[cur[0], cur[1]] = True
                    region.append(cur)
                    for nr, nc in self._neighbors(*cur):
                        if self.board[nr, nc] == EMPTY and not visited[nr, nc]:
                            stack.append((nr, nc))
                        elif self.board[nr, nc] != EMPTY:
                            touches.add(self.board[nr, nc])

                if touches == {BLACK}:
                    black_territory += len(region)
                elif touches == {WHITE}:
                    white_territory += len(region)
                # neutral — not counted

        black_stones = int(np.sum(self.board == BLACK))
        white_stones = int(np.sum(self.board == WHITE))

        return {
            'BLACK': black_stones + black_territory,
            'WHITE': white_stones + white_territory,
        }

    def winner(self, komi: float = 6.5) -> int:
        """Returns BLACK or WHITE (or 0 for draw)."""
        s = self.score()
        diff = s['BLACK'] - (s['WHITE'] + komi)
        if diff > 0:
            return BLACK
        elif diff < 0:
            return WHITE
        return 0

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def to_tensor(self) -> np.ndarray:
        """
        Returns (3, N, N) float32 array:
            channel 0: black stones
            channel 1: white stones
            channel 2: empty
        """
        t = np.zeros((3, self.size, self.size), dtype=np.float32)
        t[0] = (self.board == BLACK).astype(np.float32)
        t[1] = (self.board == WHITE).astype(np.float32)
        t[2] = (self.board == EMPTY).astype(np.float32)
        return t

    def to_field(self) -> np.ndarray:
        """
        Returns (N, N) float32 with values in {-1, 0, 1}.
        Used as the scalar field input for DeepONet branch net.
        """
        return self.board.astype(np.float32)

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        b.board = self.board.copy()
        b.ko_point = self.ko_point
        b.captured = self.captured.copy()
        b.move_history = self.move_history.copy()
        b.pass_count = self.pass_count
        return b

    def __repr__(self):
        symbols = {BLACK: '●', WHITE: '○', EMPTY: '·'}
        rows = []
        for r in range(self.size):
            row = ' '.join(symbols[self.board[r, c]] for c in range(self.size))
            rows.append(f"{self.size - r:2d} {row}")
        cols = '   ' + ' '.join('ABCDEFGHJKLMNOPQRST'[:self.size])
        return '\n'.join(rows) + '\n' + cols
