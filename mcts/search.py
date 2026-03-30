"""
MCTS with PUCT selection — AlphaGo Zero style.

Each node stores:
    N  : visit count
    W  : total value
    Q  : mean value  (W/N)
    P  : prior probability from policy network

Selection uses PUCT:
    U(s,a) = C_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
    a* = argmax [ Q(s,a) + U(s,a) ]
"""

import math
import numpy as np
from typing import Optional

from go_env.board import GoBoard, BLACK, WHITE


C_PUCT = 1.5


class MCTSNode:
    __slots__ = ('board', 'color', 'move', 'parent',
                 'children', 'N', 'W', 'P', '_expanded')

    def __init__(self, board: GoBoard, color: int,
                 move=None, parent=None, prior: float = 0.0):
        self.board    = board
        self.color    = color      # color to play at this node
        self.move     = move       # move that led here
        self.parent   = parent
        self.children: dict = {}   # move → MCTSNode
        self.N  = 0
        self.W  = 0.0
        self.P  = prior            # prior from policy net
        self._expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_leaf(self) -> bool:
        return not self._expanded

    def expand(self, legal_moves: list, priors: np.ndarray, board_size: int):
        """
        legal_moves : list of (r,c) or None (pass), length L
        priors      : (N*N+1,) policy output from network
        """
        for move in legal_moves:
            idx = board_size * board_size if move is None else move[0] * board_size + move[1]
            child_board = self.board.copy()
            child_board.play(self.color, move)
            self.children[move] = MCTSNode(
                board=child_board,
                color=-self.color,
                move=move,
                parent=self,
                prior=float(priors[idx]),
            )
        self._expanded = True

    def select_child(self) -> 'MCTSNode':
        """PUCT selection."""
        sqrt_N = math.sqrt(sum(c.N for c in self.children.values()) + 1)
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            score = child.Q + C_PUCT * child.P * sqrt_N / (1 + child.N)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def backup(self, value: float):
        """Propagate value up the tree. Value is from current player's perspective."""
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            value = -value          # flip perspective as we go up
            node = node.parent


class MCTS:
    def __init__(self, model, device='cpu', n_simulations: int = 100,
                 dirichlet_alpha: float = 0.03, dirichlet_eps: float = 0.25):
        self.model        = model
        self.device       = device
        self.n_simulations = n_simulations
        self.dir_alpha    = dirichlet_alpha
        self.dir_eps      = dirichlet_eps

    def _evaluate(self, board: GoBoard, color: int):
        """Run network on board, return (priors, value) from `color`'s perspective."""
        import torch
        t = board.to_tensor()                           # (3, N, N)
        if color == WHITE:
            # Flip channels so the network always sees itself as "black"
            t = np.stack([t[1], t[0], t[2]], axis=0)
        probs, value = self.model.predict(t, device=self.device)
        return probs, value

    def get_policy(self, board: GoBoard, color: int,
                   temperature: float = 1.0) -> tuple:
        """
        Run MCTS from `board` with `color` to play.
        Returns:
            move_probs : dict {move: probability}
            root       : MCTSNode (for debugging)
        """
        import numpy as np

        root = MCTSNode(board.copy(), color)
        priors, value = self._evaluate(board, color)

        legal = board.legal_moves(color)

        # Add Dirichlet noise at root for exploration
        noise = np.random.dirichlet([self.dir_alpha] * len(legal))
        noisy_priors = priors.copy()
        for i, move in enumerate(legal):
            idx = board.size ** 2 if move is None else move[0] * board.size + move[1]
            noisy_priors[idx] = ((1 - self.dir_eps) * priors[idx]
                                 + self.dir_eps * noise[i])

        root.expand(legal, noisy_priors, board.size)
        root.backup(value)

        # Simulations
        for _ in range(self.n_simulations - 1):
            node = root

            # Selection: walk down to a leaf
            while not node.is_leaf() and node.children:
                node = node.select_child()

            # Check for terminal
            if node.board.pass_count >= 2:
                winner = node.board.winner()
                leaf_value = 1.0 if winner == node.color else (
                    -1.0 if winner == -node.color else 0.0)
                node.backup(leaf_value)
                continue

            # Expansion + evaluation
            leaf_priors, leaf_value = self._evaluate(node.board, node.color)
            leaf_legal = node.board.legal_moves(node.color)
            node.expand(leaf_legal, leaf_priors, node.board.size)
            node.backup(leaf_value)

        # Build move probability distribution from visit counts
        counts = {move: child.N for move, child in root.children.items()}
        total  = sum(counts.values())

        if temperature == 0:
            # Greedy
            best = max(counts, key=counts.get)
            move_probs = {m: (1.0 if m == best else 0.0) for m in counts}
        else:
            # Softmax over counts^(1/T)
            raw = {m: n ** (1.0 / temperature) for m, n in counts.items()}
            s   = sum(raw.values())
            move_probs = {m: v / s for m, v in raw.items()}

        return move_probs, root
