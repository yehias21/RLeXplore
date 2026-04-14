"""DQN agent: composes a Q-net, a target net, an optimiser, and a memory.

DIP: accepts any `nn.Module` Q-network (MLP, NoisyMLP, Bootstrapped) and any
exploration strategy. Strategy interface handled via duck-typing on the hooks
defined in `ExplorationStrategy`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from ..memory import ReplayMemory
from ..exploration.base import ExplorationStrategy
from ..exploration.posterior import BootstrappedDQNExploration


@dataclass
class DQNConfig:
    gamma: float = 0.99
    batch_size: int = 128
    learning_rate: float = 5e-4
    memory_size: int = 200_000
    target_update: int = 1000
    grad_clip: float = 1.0
    weight_decay: float = 1e-5


class DQNAgent:
    def __init__(self, q_net: nn.Module, strategy: ExplorationStrategy,
                 device: torch.device, cfg: DQNConfig):
        self.cfg = cfg
        self.device = device
        self.policy_net = q_net.to(device)
        self.target_net = copy.deepcopy(q_net).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optim = optim.Adam(self.policy_net.parameters(),
                                lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.memory = ReplayMemory(cfg.memory_size)
        self.strategy = strategy
        self._step = 0

    # ---------- acting ----------
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.strategy.select_action(state, self.policy_net, self._step)

    def observe_transition(self, s, a, s_next, r_ext, done):
        self.strategy.observe(s, int(a.item()) if torch.is_tensor(a) else int(a),
                              s_next, r_ext, done)

    def bonus(self, s, a, s_next, r_ext) -> float:
        return self.strategy.bonus(s, int(a.item()) if torch.is_tensor(a) else int(a),
                                   s_next, r_ext)

    # ---------- learning ----------
    def push(self, s, a, s_next, r):
        self.memory.push(s, a, s_next, r)

    def _maybe_reset_noise(self):
        if hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()
        if hasattr(self.target_net, "reset_noise"):
            self.target_net.reset_noise()

    def optimize(self) -> Optional[float]:
        if len(self.memory) < self.cfg.batch_size:
            return None
        from ..memory.replay import Transition
        batch = Transition(*zip(*self.memory.sample(self.cfg.batch_size)))

        s = torch.cat(batch.state).to(self.device)
        a = torch.cat(batch.action).to(self.device)
        r = torch.cat(batch.reward).to(self.device).float()

        mask = torch.tensor([ns is not None for ns in batch.next_state],
                            device=self.device, dtype=torch.bool)
        ns_tensors = [ns for ns in batch.next_state if ns is not None]

        self._maybe_reset_noise()

        # Bootstrapped DQN: train the currently active head only.
        active_head = None
        if isinstance(self.strategy, BootstrappedDQNExploration) \
                and hasattr(self.policy_net, "set_active_head"):
            active_head = self.strategy.active_head()
            self.policy_net.set_active_head(active_head)
            self.target_net.set_active_head(active_head)

        q_sa = self.policy_net(s).gather(1, a)
        next_v = torch.zeros(self.cfg.batch_size, device=self.device)
        if ns_tensors:
            ns = torch.cat(ns_tensors).to(self.device)
            with torch.no_grad():
                next_v[mask] = self.target_net(ns).max(1)[0]
        target = (next_v * self.cfg.gamma) + r
        loss = nn.functional.mse_loss(q_sa, target.unsqueeze(1))

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip)
        self.optim.step()

        if active_head is not None:
            self.policy_net.set_active_head(None)
            self.target_net.set_active_head(None)

        self._step += 1
        if self._step % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
