"""
DeepONet-style policy+value network for Go.

Architecture:
  - Branch net  : CNN residual tower encodes board state → latent vector (p,)
  - Trunk net   : Fourier encoding + MLP on (x,y) → basis vector (p,)
  - Policy      : dot(branch, trunk) evaluated at every board position → (N*N+1,)
  - Value head  : separate MLP on branch output → scalar in [-1,1]

Fourier positional encoding (NeRF-style) transforms raw (x,y) coordinates into
a high-frequency embedding, giving the trunk immediate awareness of corners,
edges, and distances without having to learn them from scratch.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_net import ResBlock


def fourier_encode(coords: torch.Tensor, num_frequencies: int = 6) -> torch.Tensor:
    """
    NeRF-style Fourier positional encoding.
    coords : (Q, 2) in [0, 1]²
    returns: (Q, 2 + 4 * num_frequencies)
        — original coords + sin/cos at frequencies 2^0 … 2^(L-1) for each axis
    """
    encoded = [coords]
    for i in range(num_frequencies):
        freq = 2.0 ** i * math.pi
        encoded.append(torch.sin(freq * coords))
        encoded.append(torch.cos(freq * coords))
    return torch.cat(encoded, dim=-1)   # (Q, 2 + 4*L)


class TrunkNet(nn.Module):
    """
    Maps (x, y) ∈ [0,1]² → basis vector of dimension `latent_dim`.
    Applies Fourier positional encoding before the MLP so the trunk
    immediately understands corners, edges and centre geometry.
    """

    def __init__(self, latent_dim: int, hidden: int = 128, depth: int = 3,
                 num_frequencies: int = 6):
        super().__init__()
        self.num_frequencies = num_frequencies
        in_dim = 2 + 4 * num_frequencies   # Fourier encoding output size

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords : (Q, 2)  — Q query points, each in [0,1]²
        returns: (Q, latent_dim)
        """
        encoded = fourier_encode(coords, self.num_frequencies)
        return self.net(encoded)


class BranchNet(nn.Module):
    """CNN residual tower that encodes (3, N, N) board → latent vector (latent_dim,)."""

    def __init__(self, board_size: int, in_channels: int = 3,
                 filters: int = 64, n_res_blocks: int = 5, latent_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(n_res_blocks)])
        self.pool  = nn.AdaptiveAvgPool2d(1)   # (B, filters, 1, 1) → (B, filters)
        self.proj  = nn.Linear(filters, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, N, N)
        returns: (B, latent_dim)
        """
        x = self.stem(x)
        x = self.tower(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class DeepONetGo(nn.Module):
    def __init__(self, board_size: int = 9, in_channels: int = 3,
                 filters: int = 64, n_res_blocks: int = 5,
                 latent_dim: int = 128, trunk_hidden: int = 128, trunk_depth: int = 3):
        super().__init__()
        self.board_size = board_size
        self.latent_dim = latent_dim

        self.branch = BranchNet(board_size, in_channels, filters, n_res_blocks, latent_dim)
        self.trunk  = TrunkNet(latent_dim, trunk_hidden, trunk_depth)

        # Learnable bias for the output function (standard in DeepONet)
        self.output_bias = nn.Parameter(torch.zeros(1))

        # Pass-move score: a separate scalar learned from branch output only
        self.pass_head = nn.Linear(latent_dim, 1)

        # Value head: MLP on branch output
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, filters),
            nn.ReLU(),
            nn.Linear(filters, 1),
            nn.Tanh(),
        )

        # Pre-compute and cache query coordinates for the default board size
        self._register_query_coords(board_size)

    def _register_query_coords(self, size: int):
        """
        Build (N*N, 2) coordinate grid normalised to [0,1]².
        Row-major: index i → (row=i//size, col=i%size).
        Registered as a buffer so it moves with .to(device).
        """
        coords = []
        for r in range(size):
            for c in range(size):
                coords.append([r / (size - 1), c / (size - 1)])
        self.register_buffer('query_coords', torch.tensor(coords, dtype=torch.float32))

    def forward(self, x: torch.Tensor,
                query_coords: torch.Tensor = None) -> tuple:
        """
        x            : (B, 3, N, N)
        query_coords : (Q, 2) or None (uses cached grid for board_size)

        Returns:
            policy_logits : (B, Q+1)  — Q board moves + pass
            value         : (B,)
        """
        B = x.shape[0]

        # Branch: encode board state
        branch_out = self.branch(x)              # (B, latent_dim)

        # Trunk: encode query coordinates
        if query_coords is None:
            query_coords = self.query_coords     # (N*N, 2)
        trunk_out = self.trunk(query_coords)     # (Q, latent_dim)

        # Operator output: inner product at each query point
        # branch_out: (B, D), trunk_out: (Q, D)
        # → policy_board: (B, Q)
        policy_board = branch_out @ trunk_out.T + self.output_bias  # (B, Q)

        # Pass move score
        pass_score = self.pass_head(branch_out)  # (B, 1)

        policy_logits = torch.cat([policy_board, pass_score], dim=1)  # (B, Q+1)

        value = self.value_head(branch_out).squeeze(-1)               # (B,)

        return policy_logits, value

    def predict(self, board_tensor: np.ndarray, device='cpu'):
        """
        Convenience: single (3, N, N) numpy array → (policy_probs, value).
        """
        t = torch.from_numpy(board_tensor).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            logits, value = self(t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, value.item()
