"""
PyTorch Dataset wrapper around the SGF samples.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class GoDataset(Dataset):
    def __init__(self, boards: np.ndarray, policies: np.ndarray, values: np.ndarray):
        """
        boards   : (N, 3, size, size) float32
        policies : (N,)              int64   — move index
        values   : (N,)              float32 — +1 / -1
        """
        self.boards   = torch.from_numpy(boards)
        self.policies = torch.from_numpy(policies)
        self.values   = torch.from_numpy(values)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.values[idx]

    @staticmethod
    def from_sgf_dir(directory: str, size_filter: int = 9, max_samples: int = None,
                     augment: bool = True):
        from data.sgf_loader import build_dataset
        from training.augment import augment_dataset
        boards, policies, values = build_dataset(directory, size_filter, max_samples)
        if augment:
            boards, policies, values = augment_dataset(boards, policies, values)
        return GoDataset(boards, policies, values)
