"""Greedy-policy evaluator."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..core.interfaces import Environment


@dataclass
class EvalConfig:
    episodes: int = 100
    max_steps: int = 500


class Evaluator:
    def __init__(self, env: Environment, policy_net: torch.nn.Module,
                 device: torch.device, cfg: EvalConfig):
        self.env = env
        self.policy_net = policy_net.to(device).eval()
        self.device = device
        self.cfg = cfg

    @torch.no_grad()
    def run(self) -> dict:
        finished = 0
        total_reward = 0.0
        total_steps = 0
        for _ in range(self.cfg.episodes):
            state, _ = self.env.reset()
            for step in range(1, self.cfg.max_steps + 1):
                action = int(self.policy_net(state).max(1)[1].item())
                state, reward, done, trunc, _ = self.env.step(action)
                if done or trunc:
                    total_reward += reward
                    total_steps += step
                    if done:
                        finished += 1
                    break
        n = self.cfg.episodes
        return {
            "completion_rate": finished / n,
            "average_reward": total_reward / n,
            "average_steps": total_steps / n,
        }
