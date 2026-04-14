"""Observation preprocessors (SRP: one job — raw obs -> tensor)."""
from __future__ import annotations
from typing import Protocol
import numpy as np
import torch


class Preprocessor(Protocol):
    out_size: int
    def __call__(self, obs, device: torch.device) -> torch.Tensor: ...


class MiniGridObjectIdPreprocessor:
    """Extract the object-id channel of a MiniGrid ImgObsWrapper observation,
    flatten and normalise to [0,1]."""

    def __init__(self, rows: int = 7, cols: int = 7, divisor: float = 10.0):
        self.rows = rows
        self.cols = cols
        self.divisor = divisor
        self.out_size = rows * cols

    def __call__(self, obs: np.ndarray, device: torch.device) -> torch.Tensor:
        # obs: (rows, cols, channels); channel 0 = object id
        obj = obs[..., 0].astype(np.float32) / self.divisor
        return torch.from_numpy(obj.flatten()).unsqueeze(0).to(device)
