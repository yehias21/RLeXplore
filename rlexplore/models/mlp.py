"""MLP Q-network."""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import MODELS


@MODELS.register("mlp")
class MLPQNet(nn.Module):
    """Simple MLP for discrete-action Q-learning.

    Kept intentionally small: architecture concerns live here, nowhere else (SRP).
    """

    def __init__(self, input_size: int, num_actions: int,
                 hidden_sizes: Sequence[int] = (128, 128), dropout: float = 0.0):
        super().__init__()
        sizes = [input_size, *hidden_sizes]
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(a, b))
        self.hidden = nn.ModuleList(layers)
        self.head = nn.Linear(sizes[-1], num_actions)
        self.dropout = dropout
        for m in [*self.hidden, self.head]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = F.relu(layer(x))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)
