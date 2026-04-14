"""Probability matching / posterior sampling (survey §8).

- NoisyNetExploration : stochasticity lives in the Q-net weights (Fortunato 2018).
- BootstrappedDQNExploration : sample one Q-head per episode (Osband 2016a).
"""
from __future__ import annotations
import random
import torch

from ..core.registry import STRATEGIES
from .base import ExplorationStrategy


@STRATEGIES.register("noisy_net")
class NoisyNetExploration(ExplorationStrategy):
    """Assumes the Q-net has `reset_noise()`. Call it on each selection.
    Action is argmax over the noisy Q."""
    def select_action(self, state, q_net, step):
        if hasattr(q_net, "reset_noise"):
            q_net.reset_noise()
        with torch.no_grad():
            return q_net(state).max(1)[1].unsqueeze(0)


@STRATEGIES.register("bootstrapped_dqn")
class BootstrappedDQNExploration(ExplorationStrategy):
    """Sample a head uniformly at each episode start; act greedily wrt it."""
    def __init__(self, num_actions, device, num_heads: int = 10):
        super().__init__(num_actions, device)
        self.num_heads = num_heads
        self._head = 0

    def on_episode_start(self):
        self._head = random.randrange(self.num_heads)

    def active_head(self) -> int:
        return self._head

    def select_action(self, state, q_net, step):
        if hasattr(q_net, "set_active_head"):
            q_net.set_active_head(self._head)
        with torch.no_grad():
            out = q_net(state)
        if hasattr(q_net, "set_active_head"):
            q_net.set_active_head(None)
        return out.max(1)[1].unsqueeze(0)
