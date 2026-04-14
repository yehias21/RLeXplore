"""Bootstrapped DQN (Osband et al., 2016): K parallel Q-heads on a shared torso."""
from __future__ import annotations
from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import MODELS


@MODELS.register("bootstrapped_mlp")
class BootstrappedMLPQNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int,
                 hidden_sizes: Sequence[int] = (128, 128), num_heads: int = 10):
        super().__init__()
        sizes = [input_size, *hidden_sizes]
        self.torso = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.heads = nn.ModuleList(nn.Linear(sizes[-1], num_actions) for _ in range(num_heads))
        self.num_heads = num_heads
        self.num_actions = num_actions
        self._active: Optional[int] = None

    def set_active_head(self, k: Optional[int]):
        self._active = k

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.torso:
            x = F.relu(layer(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """If a head is active returns its Q-values; else mean over heads (for eval)."""
        feat = self._features(x)
        if self._active is not None:
            return self.heads[self._active](feat)
        return torch.stack([h(feat) for h in self.heads], dim=0).mean(dim=0)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Shape (K, batch, A). Used for per-head TD targets."""
        feat = self._features(x)
        return torch.stack([h(feat) for h in self.heads], dim=0)
