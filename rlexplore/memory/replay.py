"""Experience replay buffer."""
from __future__ import annotations
from collections import deque, namedtuple
from typing import Optional
import random
import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    """Standard uniform experience replay.

    SRP: stores and samples transitions. Nothing more.
    """

    def __init__(self, capacity: int):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: torch.Tensor,
             next_state: Optional[torch.Tensor], reward: torch.Tensor) -> None:
        self._buf.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)
