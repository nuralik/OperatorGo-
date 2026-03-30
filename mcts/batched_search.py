"""
Batched MCTS with GPU evaluation.

Architecture:
  - EvalBatcher    : dedicated thread that collects board eval requests
                     from all parallel game threads, batches them, sends
                     to GPU, dispatches results back.
  - BatchedMCTS    : same PUCT logic as MCTS but submits evals to batcher
                     instead of calling CPU model directly.
  - play_games_batched : runs N games as threads simultaneously, all
                         sharing one EvalBatcher → GPU stays busy.

GPU utilization:
  Old: 2000 single-sample CPU evals per iter  (GPU idle)
  New: 2000/batch_size GPU batched evals       (GPU busy)
"""

import math
import queue
import threading
import numpy as np
import torch

from go_env.board import GoBoard, BLACK, WHITE
from mcts.search import MCTSNode, C_PUCT
from training.self_play import play_game as _play_game_sequential
from training.augment import _transform_board, transform_policy_vec


# ── Eval Batcher ─────────────────────────────────────────────────────────────

class EvalBatcher:
    """
    Runs in a dedicated thread.
    Collects (board_tensor, result_queue) requests, batches up to
    `max_batch_size` at a time, evaluates on GPU, sends results back.
    """

    def __init__(self, model, device: str,
                 max_batch_size: int = 64,
                 timeout_s: float = 0.005):
        self.model          = model
        self.device         = device
        self.max_batch_size = max_batch_size
        self.timeout_s      = timeout_s
        self._queue         = queue.Queue()
        self._thread        = None
        self._running       = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self):
        self.model.eval()
        while self._running:
            batch = []

            # Block waiting for first request
            try:
                batch.append(self._queue.get(timeout=self.timeout_s))
            except queue.Empty:
                continue

            # Drain as many more as available (non-blocking)
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            # Batch eval on GPU
            boards = np.stack([b for b, _ in batch])
            with torch.no_grad():
                t      = torch.from_numpy(boards).to(self.device)
                logits, values = self.model(t)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                vals   = values.cpu().numpy()

            # Dispatch results back to waiting threads
            for i, (_, result_q) in enumerate(batch):
                result_q.put((probs[i], float(vals[i])))

    def evaluate(self, board_tensor: np.ndarray):
        """Submit one board. Blocks until batcher returns (probs, value)."""
        result_q = queue.SimpleQueue()
        self._queue.put((board_tensor, result_q))
        return result_q.get()


# ── Batched MCTS ──────────────────────────────────────────────────────────────

class BatchedMCTS:
    """
    PUCT MCTS that uses EvalBatcher for GPU-batched evaluations.
    Drop-in replacement for mcts.search.MCTS.
    """

    def __init__(self, batcher: EvalBatcher, n_simulations: int = 100,
                 dirichlet_alpha: float = 0.03, dirichlet_eps: float = 0.25):
        self.batcher       = batcher
        self.n_simulations = n_simulations
        self.dir_alpha     = dirichlet_alpha
        self.dir_eps       = dirichlet_eps

    def _evaluate(self, board: GoBoard, color: int):
        t = board.to_tensor()
        if color == WHITE:
            t = np.stack([t[1], t[0], t[2]], axis=0)
        return self.batcher.evaluate(t)

    def get_policy(self, board: GoBoard, color: int,
                   temperature: float = 1.0) -> tuple:
        root   = MCTSNode(board.copy(), color)
        priors, value = self._evaluate(board, color)
        legal  = board.legal_moves(color)

        # Dirichlet noise at root
        noise = np.random.dirichlet([self.dir_alpha] * len(legal))
        noisy = priors.copy()
        for i, move in enumerate(legal):
            idx = board.size ** 2 if move is None else move[0] * board.size + move[1]
            noisy[idx] = ((1 - self.dir_eps) * priors[idx]
                          + self.dir_eps * noise[i])

        root.expand(legal, noisy, board.size)
        root.backup(value)

        for _ in range(self.n_simulations - 1):
            node = root
            while not node.is_leaf() and node.children:
                node = node.select_child()

            if node.board.pass_count >= 2:
                winner = node.board.winner()
                v = 1.0 if winner == node.color else (-1.0 if winner == -node.color else 0.0)
                node.backup(v)
                continue

            leaf_priors, leaf_value = self._evaluate(node.board, node.color)
            node.expand(node.board.legal_moves(node.color), leaf_priors, node.board.size)
            node.backup(leaf_value)

        counts = {move: child.N for move, child in root.children.items()}
        total  = sum(counts.values())

        if temperature == 0:
            best       = max(counts, key=counts.get)
            move_probs = {m: (1.0 if m == best else 0.0) for m in counts}
        else:
            raw        = {m: n ** (1.0 / temperature) for m, n in counts.items()}
            s          = sum(raw.values())
            move_probs = {m: v / s for m, v in raw.items()}

        return move_probs, root


# ── Parallel game runner ──────────────────────────────────────────────────────

def _play_one_game(batcher, board_size, n_simulations, temp_threshold, result_list, idx):
    """Thread target — plays one game using the shared batcher."""
    mcts = BatchedMCTS(batcher, n_simulations=n_simulations)

    board   = GoBoard(board_size)
    color   = BLACK
    history = []
    move_num = 0

    while True:
        temp = 1.0 if move_num < temp_threshold else 0.0
        move_probs, _ = mcts.get_policy(board, color, temperature=temp)

        policy_vec = np.zeros(board_size * board_size + 1, dtype=np.float32)
        for move, prob in move_probs.items():
            pidx = board_size * board_size if move is None else move[0] * board_size + move[1]
            policy_vec[pidx] = prob

        t = board.to_tensor()
        if color == WHITE:
            t = np.stack([t[1], t[0], t[2]], axis=0)
        history.append((t, policy_vec, color))

        moves  = list(move_probs.keys())
        probs  = np.array([move_probs[m] for m in moves], dtype=np.float64)
        probs /= probs.sum()
        chosen = moves[np.random.choice(len(moves), p=probs)]

        done = board.play(color, chosen)
        move_num += 1
        color = -color

        if done or move_num > board_size * board_size * 4:
            break

    winner  = board.winner()
    samples = []
    for board_t, policy_vec, player_color in history:
        value = 0.0 if winner == 0 else (1.0 if winner == player_color else -1.0)
        # D8 augmentation
        for k in range(4):
            for flip in (False, True):
                samples.append((
                    _transform_board(board_t, k, flip),
                    transform_policy_vec(policy_vec, board_size, k, flip),
                    value,
                ))

    result_list[idx] = samples


def play_games_batched(model, device: str, n_games: int,
                       board_size: int = 9, n_simulations: int = 100,
                       temp_threshold: int = 15,
                       max_batch_size: int = 64) -> list:
    """
    Run n_games games in parallel using threads.
    All games share one EvalBatcher → GPU processes evals in batches.

    Returns list of sample lists, one per game.
    """
    batcher = EvalBatcher(model, device,
                          max_batch_size=max_batch_size,
                          timeout_s=0.002)
    batcher.start()

    results = [None] * n_games
    threads = [
        threading.Thread(
            target=_play_one_game,
            args=(batcher, board_size, n_simulations, temp_threshold, results, i),
            daemon=True,
        )
        for i in range(n_games)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    batcher.stop()
    return results
