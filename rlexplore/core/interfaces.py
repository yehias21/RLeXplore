"""Protocols defining the contracts between components (DIP + ISP).

Small, focused interfaces so strategies only implement what they need.
"""
from __future__ import annotations
from typing import Protocol, Optional, Any, runtime_checkable
import torch


@runtime_checkable
class Environment(Protocol):
    num_actions: int
    observation_size: int

    def reset(self) -> tuple[torch.Tensor, dict]: ...
    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool, dict]: ...
    def close(self) -> None: ...


@runtime_checkable
class QNetwork(Protocol):
    def __call__(self, state: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class ActionSelector(Protocol):
    """Picks an action given a state and a Q-network."""
    def select_action(self, state: torch.Tensor, q_net: QNetwork, step: int) -> torch.Tensor: ...


@runtime_checkable
class IntrinsicRewardSource(Protocol):
    """Yields a per-transition intrinsic reward bonus."""
    def bonus(self, state: torch.Tensor, action: int,
              next_state: Optional[torch.Tensor], extrinsic: float) -> float: ...


@runtime_checkable
class Updatable(Protocol):
    """Strategies with internal learners (RND/ICM/counts) see every transition."""
    def observe(self, state: torch.Tensor, action: int,
                next_state: Optional[torch.Tensor], reward: float, done: bool) -> None: ...


@runtime_checkable
class Logger(Protocol):
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def close(self) -> None: ...
