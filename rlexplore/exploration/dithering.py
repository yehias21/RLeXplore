"""Blind / randomised-action-selection strategies (survey §4.1, §5.1).

- EpsilonGreedy           : eq. (14), exponential decay
- EpsilonFirst            : pure random for first T steps, then greedy
- DecayingEpsilon         : linear decay
- EzGreedy                : temporally-extended eps-greedy (Dabney et al., 2020)
- Boltzmann               : eq. (20), softmax(Q/T) with temperature decay
- MaxBoltzmann            : Wiering (1999) — eps-greedy that falls back to Boltzmann
- VDBE / VDBESoftmax      : Tokic (2010, 2011) — state-dependent epsilon from TD error
- Pursuit                 : Thathachar & Sastry (1984)
"""
from __future__ import annotations
import math
import random
from typing import Optional
import torch
import torch.nn.functional as F

from ..core.registry import STRATEGIES
from .base import ExplorationStrategy


def _argmax_action(state: torch.Tensor, q_net) -> torch.Tensor:
    with torch.no_grad():
        return q_net(state).max(1)[1].unsqueeze(0)


def _random_action(num_actions: int, device) -> torch.Tensor:
    return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)


# ---------- epsilon schedules ----------

def _exp_decay(start: float, end: float, decay: float, step: int) -> float:
    return end + (start - end) * math.exp(-step / max(decay, 1e-9))


def _linear_decay(start: float, end: float, horizon: int, step: int) -> float:
    frac = min(1.0, step / max(horizon, 1))
    return start + frac * (end - start)


@STRATEGIES.register("epsilon_greedy")
class EpsilonGreedy(ExplorationStrategy):
    """eq. (14). Exponential decay of epsilon."""
    def __init__(self, num_actions, device,
                 start_epsilon: float = 1.0, stop_epsilon: float = 0.01,
                 decay_rate: float = 3000):
        super().__init__(num_actions, device)
        self.start, self.end, self.decay = start_epsilon, stop_epsilon, decay_rate

    def epsilon(self, step: int) -> float:
        return _exp_decay(self.start, self.end, self.decay, step)

    def select_action(self, state, q_net, step):
        if random.random() > self.epsilon(step):
            return _argmax_action(state, q_net)
        return _random_action(self.num_actions, self.device)


@STRATEGIES.register("epsilon_first")
class EpsilonFirst(ExplorationStrategy):
    """Tran-Thanh et al. 2010: random for first `explore_steps`, greedy after."""
    def __init__(self, num_actions, device, explore_steps: int = 5000):
        super().__init__(num_actions, device)
        self.explore_steps = explore_steps

    def select_action(self, state, q_net, step):
        if step < self.explore_steps:
            return _random_action(self.num_actions, self.device)
        return _argmax_action(state, q_net)


@STRATEGIES.register("decaying_epsilon")
class DecayingEpsilon(ExplorationStrategy):
    """Linearly decaying epsilon (Caelen & Bontempi, 2007, bandit analogue)."""
    def __init__(self, num_actions, device,
                 start_epsilon: float = 1.0, stop_epsilon: float = 0.05,
                 horizon: int = 50000):
        super().__init__(num_actions, device)
        self.start, self.end, self.horizon = start_epsilon, stop_epsilon, horizon

    def epsilon(self, step): return _linear_decay(self.start, self.end, self.horizon, step)

    def select_action(self, state, q_net, step):
        if random.random() > self.epsilon(step):
            return _argmax_action(state, q_net)
        return _random_action(self.num_actions, self.device)


@STRATEGIES.register("ez_greedy")
class EzGreedy(ExplorationStrategy):
    """Dabney et al. 2020: when exploring, repeat the same random action
    for n~zeta steps. Uses a zeta(mu) distribution truncated at n_max."""
    def __init__(self, num_actions, device,
                 start_epsilon: float = 1.0, stop_epsilon: float = 0.01,
                 decay_rate: float = 3000, mu: float = 2.0, n_max: int = 8):
        super().__init__(num_actions, device)
        self.start, self.end, self.decay = start_epsilon, stop_epsilon, decay_rate
        self.mu, self.n_max = mu, n_max
        self._action: Optional[int] = None
        self._remaining = 0

    def on_episode_start(self):
        self._action = None
        self._remaining = 0

    def _sample_duration(self) -> int:
        # Pareto-ish: approximate zeta by sampling n in {1..n_max} with p(n) ~ n^-mu.
        weights = [1.0 / (k ** self.mu) for k in range(1, self.n_max + 1)]
        Z = sum(weights)
        r = random.random() * Z
        c = 0.0
        for k, w in enumerate(weights, start=1):
            c += w
            if r <= c:
                return k
        return self.n_max

    def epsilon(self, step): return _exp_decay(self.start, self.end, self.decay, step)

    def select_action(self, state, q_net, step):
        if self._remaining > 0 and self._action is not None:
            self._remaining -= 1
            return torch.tensor([[self._action]], device=self.device, dtype=torch.long)
        if random.random() > self.epsilon(step):
            return _argmax_action(state, q_net)
        self._action = random.randrange(self.num_actions)
        self._remaining = self._sample_duration() - 1
        return torch.tensor([[self._action]], device=self.device, dtype=torch.long)


@STRATEGIES.register("boltzmann")
class Boltzmann(ExplorationStrategy):
    """eq. (20): pi(a|s) = softmax(Q(s,a)/T). T decays exponentially."""
    def __init__(self, num_actions, device,
                 start_temp: float = 1.5, min_temp: float = 0.1, temp_decay: float = 3000):
        super().__init__(num_actions, device)
        self.start, self.end, self.decay = start_temp, min_temp, temp_decay

    def temperature(self, step): return _exp_decay(self.start, self.end, self.decay, step)

    def select_action(self, state, q_net, step):
        T = self.temperature(step)
        with torch.no_grad():
            q = q_net(state)
        probs = F.softmax(q / T, dim=1)
        return torch.multinomial(probs, 1)


@STRATEGIES.register("max_boltzmann")
class MaxBoltzmann(ExplorationStrategy):
    """Wiering 1999: with prob 1-eps pick argmax; otherwise sample from Boltzmann."""
    def __init__(self, num_actions, device,
                 start_epsilon: float = 0.5, stop_epsilon: float = 0.05,
                 decay_rate: float = 3000, temperature: float = 1.0):
        super().__init__(num_actions, device)
        self.start, self.end, self.decay = start_epsilon, stop_epsilon, decay_rate
        self.T = temperature

    def epsilon(self, step): return _exp_decay(self.start, self.end, self.decay, step)

    def select_action(self, state, q_net, step):
        if random.random() > self.epsilon(step):
            return _argmax_action(state, q_net)
        with torch.no_grad():
            q = q_net(state)
        probs = F.softmax(q / self.T, dim=1)
        return torch.multinomial(probs, 1)


@STRATEGIES.register("vdbe")
class VDBE(ExplorationStrategy):
    """Tokic 2010: state-dependent epsilon driven by TD error magnitude.
    eps_{t+1}(s) = delta * f(|TD|) + (1-delta) * eps_t(s),
    where f(x) = (1 - exp(-x/sigma)) / (1 + exp(-x/sigma))."""
    def __init__(self, num_actions, device, sigma: float = 1.0, delta: Optional[float] = None):
        super().__init__(num_actions, device)
        self.sigma = sigma
        self.delta = delta if delta is not None else 1.0 / num_actions
        self._eps: dict = {}

    @staticmethod
    def _state_key(state: torch.Tensor):
        return tuple(state.detach().cpu().flatten().tolist())

    def _f(self, td_error: float) -> float:
        x = math.exp(-abs(td_error) / max(self.sigma, 1e-9))
        return (1.0 - x) / (1.0 + x)

    def observe(self, state, action, next_state, reward, done):
        # TD error is injected by the trainer via set_last_td. If unavailable, skip.
        td = getattr(self, "_last_td", None)
        if td is None:
            return
        key = self._state_key(state)
        prev = self._eps.get(key, 1.0)
        self._eps[key] = self.delta * self._f(td) + (1 - self.delta) * prev

    def set_last_td(self, td_error: float):
        self._last_td = td_error

    def epsilon_for(self, state):
        return self._eps.get(self._state_key(state), 1.0)

    def select_action(self, state, q_net, step):
        if random.random() > self.epsilon_for(state):
            return _argmax_action(state, q_net)
        return _random_action(self.num_actions, self.device)


@STRATEGIES.register("vdbe_softmax")
class VDBESoftmax(VDBE):
    """VDBE where exploration falls back to Boltzmann (Tokic & Palm 2011)."""
    def __init__(self, num_actions, device, sigma: float = 1.0,
                 delta: Optional[float] = None, temperature: float = 1.0):
        super().__init__(num_actions, device, sigma, delta)
        self.T = temperature

    def select_action(self, state, q_net, step):
        if random.random() > self.epsilon_for(state):
            return _argmax_action(state, q_net)
        with torch.no_grad():
            q = q_net(state)
        probs = F.softmax(q / self.T, dim=1)
        return torch.multinomial(probs, 1)


@STRATEGIES.register("pursuit")
class Pursuit(ExplorationStrategy):
    """Thathachar & Sastry 1984 (RL form, eq. 23).
    Maintains pi(s,a); greedy action's probability is pulled towards 1."""
    def __init__(self, num_actions, device, alpha: float = 0.01):
        super().__init__(num_actions, device)
        self.alpha = alpha
        self._pi: dict = {}

    def _key(self, state):
        return tuple(state.detach().cpu().flatten().tolist())

    def _probs(self, key):
        p = self._pi.get(key)
        if p is None:
            p = [1.0 / self.num_actions] * self.num_actions
            self._pi[key] = p
        return p

    def select_action(self, state, q_net, step):
        key = self._key(state)
        with torch.no_grad():
            q = q_net(state)
        greedy = int(q.argmax(1).item())
        p = self._probs(key)
        for i in range(self.num_actions):
            target = 1.0 if i == greedy else 0.0
            p[i] = p[i] + self.alpha * (target - p[i])
        # sample
        r = random.random()
        c = 0.0
        for i, pi in enumerate(p):
            c += pi
            if r <= c:
                return torch.tensor([[i]], device=self.device, dtype=torch.long)
        return torch.tensor([[self.num_actions - 1]], device=self.device, dtype=torch.long)
