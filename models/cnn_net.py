"""
Baseline CNN policy+value network — AlphaGo Zero style (simplified).

Input : (batch, 3, N, N)  — black / white / empty channels
Output:
    policy_logits : (batch, N*N + 1)  — move probabilities (last = pass)
    value         : (batch,)          — win probability in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class GoNet(nn.Module):
    def __init__(self, board_size: int = 9, in_channels: int = 3,
                 filters: int = 64, n_res_blocks: int = 5):
        super().__init__()
        self.board_size = board_size
        n_moves = board_size * board_size + 1  # +1 for pass

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(n_res_blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, n_moves),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, filters),
            nn.ReLU(),
            nn.Linear(filters, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        x : (B, 3, N, N)
        Returns policy_logits (B, N*N+1), value (B,)
        """
        x = self.stem(x)
        x = self.tower(x)
        policy = self.policy_head(x)
        value  = self.value_head(x).squeeze(-1)
        return policy, value

    def predict(self, board_tensor: 'np.ndarray', device='cpu'):
        """
        Convenience: takes a single (3, N, N) numpy array,
        returns (policy_probs, value) as numpy.
        """
        import numpy as np
        t = torch.from_numpy(board_tensor).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            logits, value = self(t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, value.item()
