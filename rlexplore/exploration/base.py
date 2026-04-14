"""Base class for exploration strategies.

LSP: every strategy can substitute `ExplorationStrategy` for `select_action`.
Mixins (`IntrinsicRewardSource`, `Updatable`) are opt-in — ISP.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch


class ExplorationStrategy(ABC):
    """Minimum contract: pick an action."""

    def __init__(self, num_actions: int, device: torch.device):
        self.num_actions = num_actions
        self.device = device

    @abstractmethod
    def select_action(self, state: torch.Tensor, q_net, step: int) -> torch.Tensor:
        """Return a (1,1) long tensor holding the chosen action index."""

    # Optional hooks — default no-op; strategies override only what they need.
    def observe(self, state, action, next_state, reward, done) -> None:
        return None

    def bonus(self, state, action, next_state, extrinsic) -> float:
        return 0.0

    def on_episode_start(self) -> None:
        return None
