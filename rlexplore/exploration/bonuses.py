"""Optimism / count / prediction-error bonus strategies (survey §6).

Action selection is greedy over Q; exploration happens via an intrinsic reward
bonus B(s,a) added to the extrinsic reward (eq. 24).

- CountBased        : B = beta / sqrt(N(s))
- HashPseudoCount   : Tang et al. 2017 — SimHash-based count for continuous obs
- UCBQ              : Jin et al. 2018 — B = c * sqrt(log step / N(s,a))
- RND               : Burda et al. 2018b — prediction error as bonus
- ICM               : Pathak et al. 2017 — forward-model prediction error in feature space
"""
from __future__ import annotations
from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import STRATEGIES
from .base import ExplorationStrategy


def _greedy(state, q_net, device):
    with torch.no_grad():
        return q_net(state).max(1)[1].unsqueeze(0)


@STRATEGIES.register("count_based")
class CountBased(ExplorationStrategy):
    """Tabular count-based bonus: beta / sqrt(N(s)).

    Action selection is greedy on Q. The bonus is added to the reward
    inside the trainer (eq. 24) — NOT to the Q-values at selection time.
    """
    def __init__(self, num_actions, device, beta: float = 0.1):
        super().__init__(num_actions, device)
        self.beta = beta
        self._counts: dict = {}

    def _key(self, state):
        return tuple(state.detach().cpu().flatten().tolist())

    def observe(self, state, action, next_state, reward, done):
        if next_state is None:
            return
        k = self._key(next_state)
        self._counts[k] = self._counts.get(k, 0) + 1

    def bonus(self, state, action, next_state, extrinsic):
        if next_state is None:
            return 0.0
        n = max(1, self._counts.get(self._key(next_state), 0))
        return self.beta / math.sqrt(n)

    def select_action(self, state, q_net, step):
        return _greedy(state, q_net, self.device)


@STRATEGIES.register("hash_count")
class HashPseudoCount(CountBased):
    """Tang et al. 2017: SimHash to map continuous states to discrete codes,
    then count collisions."""
    def __init__(self, num_actions, device, beta: float = 0.1,
                 input_size: int = 49, hash_bits: int = 16, seed: int = 0):
        super().__init__(num_actions, device, beta=beta)
        rng = np.random.RandomState(seed)
        self._A = rng.randn(hash_bits, input_size).astype(np.float32)

    def _key(self, state):
        v = state.detach().cpu().numpy().flatten()
        return tuple((self._A @ v >= 0).astype(np.int8).tolist())


@STRATEGIES.register("ucb_q")
class UCBQ(ExplorationStrategy):
    """Jin et al. 2018: B(s,a) = c * sqrt(log(step+1) / N(s,a)).

    Action selected by argmax over Q(s,a) + B(s,a). Bonus is also added to reward
    so that the learned Q internalises the exploration pressure (common practical
    variant of Hoeffding-style UCB-H)."""
    def __init__(self, num_actions, device, c: float = 1.0):
        super().__init__(num_actions, device)
        self.c = c
        self._counts: dict = {}

    def _key(self, state):
        return tuple(state.detach().cpu().flatten().tolist())

    def _ucb(self, state, step):
        counts = self._counts.get(self._key(state))
        log_t = math.log(step + 2)
        if counts is None:
            return torch.full((1, self.num_actions), 1e6, device=self.device)
        b = [self.c * math.sqrt(log_t / max(1, counts[a])) for a in range(self.num_actions)]
        return torch.tensor([b], device=self.device, dtype=torch.float32)

    def observe(self, state, action, next_state, reward, done):
        k = self._key(state)
        c = self._counts.setdefault(k, [0] * self.num_actions)
        c[int(action)] += 1

    def bonus(self, state, action, next_state, extrinsic):
        counts = self._counts.get(self._key(state))
        if counts is None:
            return 0.0
        step = sum(counts)
        return self.c * math.sqrt(math.log(step + 2) / max(1, counts[int(action)]))

    def select_action(self, state, q_net, step):
        with torch.no_grad():
            q = q_net(state)
        q_plus = q + self._ucb(state, step)
        return q_plus.max(1)[1].unsqueeze(0)


class _RandomFeatureNet(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_size),
        )
    def forward(self, x): return self.net(x)


@STRATEGIES.register("rnd")
class RND(ExplorationStrategy):
    """Random Network Distillation — bonus = MSE(predictor(s), target(s)).
    Target net is frozen-random; predictor is trained on visited states."""
    def __init__(self, num_actions, device, input_size: int,
                 hidden: int = 128, out_size: int = 128,
                 beta: float = 0.1, lr: float = 1e-4):
        super().__init__(num_actions, device)
        self.beta = beta
        self.target = _RandomFeatureNet(input_size, hidden, out_size).to(device)
        self.predictor = _RandomFeatureNet(input_size, hidden, out_size).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def _err(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tgt = self.target(state)
        pred = self.predictor(state)
        return F.mse_loss(pred, tgt, reduction="none").mean(dim=1)

    def observe(self, state, action, next_state, reward, done):
        if next_state is None:
            return
        err = self._err(next_state).mean()
        self.opt.zero_grad()
        err.backward()
        self.opt.step()

    def bonus(self, state, action, next_state, extrinsic):
        if next_state is None:
            return 0.0
        with torch.no_grad():
            return float(self.beta * self._err(next_state).item())

    def select_action(self, state, q_net, step):
        return _greedy(state, q_net, self.device)


class _ICMNets(nn.Module):
    def __init__(self, input_size, feat_size, num_actions, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_size),
        )
        # forward model: phi(s), a -> phi(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feat_size + num_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_size),
        )
        # inverse model: phi(s), phi(s') -> a (logits)
        self.inverse_model = nn.Sequential(
            nn.Linear(feat_size * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.num_actions = num_actions
        self.feat_size = feat_size


@STRATEGIES.register("icm")
class ICM(ExplorationStrategy):
    """Pathak et al. 2017: forward-model prediction error in a feature space
    learned via an inverse-dynamics auxiliary task."""
    def __init__(self, num_actions, device, input_size: int,
                 feat_size: int = 64, hidden: int = 128,
                 beta: float = 0.01, lr: float = 1e-4,
                 forward_weight: float = 0.2):
        super().__init__(num_actions, device)
        self.nets = _ICMNets(input_size, feat_size, num_actions, hidden).to(device)
        self.beta = beta
        self.forward_weight = forward_weight
        self.opt = torch.optim.Adam(self.nets.parameters(), lr=lr)

    def _one_hot(self, a: int) -> torch.Tensor:
        v = torch.zeros(1, self.num_actions, device=self.device)
        v[0, a] = 1.0
        return v

    def _forward_error(self, state, action, next_state) -> torch.Tensor:
        phi_s = self.nets.encoder(state)
        phi_ns = self.nets.encoder(next_state)
        a = self._one_hot(int(action))
        pred = self.nets.forward_model(torch.cat([phi_s, a], dim=1))
        return 0.5 * (pred - phi_ns).pow(2).sum(dim=1), phi_s, phi_ns

    def observe(self, state, action, next_state, reward, done):
        if next_state is None:
            return
        fwd_err, phi_s, phi_ns = self._forward_error(state, action, next_state)
        inv_logits = self.nets.inverse_model(torch.cat([phi_s, phi_ns], dim=1))
        inv_loss = F.cross_entropy(inv_logits, torch.tensor([int(action)], device=self.device))
        loss = (1 - self.forward_weight) * inv_loss + self.forward_weight * fwd_err.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def bonus(self, state, action, next_state, extrinsic):
        if next_state is None:
            return 0.0
        with torch.no_grad():
            err, _, _ = self._forward_error(state, action, next_state)
        return float(self.beta * err.item())

    def select_action(self, state, q_net, step):
        return _greedy(state, q_net, self.device)
