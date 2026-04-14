"""MiniGrid environment adapter."""
from __future__ import annotations
from typing import Optional
import torch
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

from ..core.registry import ENVS
from .preprocessors import MiniGridObjectIdPreprocessor, Preprocessor


@ENVS.register("minigrid")
class MiniGridEnvironment:
    """Adapter: owns a gym env + a preprocessor, exposes the Environment Protocol.

    SRP: does *not* do preprocessing itself — delegates. Swap preprocessors freely (DIP).
    """

    def __init__(
        self,
        grid_type: str = "MiniGrid-Empty-16x16-v0",
        max_steps: int = 1000,
        num_actions: int = 3,
        preprocessor: Optional[Preprocessor] = None,
        device: Optional[torch.device] = None,
    ):
        self.grid_type = grid_type
        self.max_steps = max_steps
        self.num_actions = num_actions
        self.device = device or torch.device("cpu")
        raw = gym.make(grid_type, render_mode=None, max_steps=max_steps)
        self._env = ImgObsWrapper(raw)
        if preprocessor is None:
            obs, _ = self._env.reset()
            rows, cols = obs.shape[0], obs.shape[1]
            preprocessor = MiniGridObjectIdPreprocessor(rows=rows, cols=cols)
        self._pre = preprocessor
        self.observation_size = self._pre.out_size

    def reset(self):
        obs, info = self._env.reset()
        return self._pre(obs, self.device), info

    def step(self, action: int):
        obs, reward, done, trunc, info = self._env.step(int(action))
        state = self._pre(obs, self.device) if not (done or trunc) else None
        return state, float(reward), bool(done), bool(trunc), info

    def close(self):
        self._env.close()

    @property
    def step_count(self) -> int:
        return getattr(self._env, "step_count", 0)
