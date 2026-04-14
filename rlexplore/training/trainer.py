"""Training loop. Orchestrates env, agent, strategy hooks, logger."""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
import torch

from ..core.interfaces import Environment, Logger
from ..agents import DQNAgent
from ..exploration.dithering import EpsilonGreedy, Boltzmann, VDBE


@dataclass
class TrainConfig:
    episodes: int = 2000
    max_steps_per_episode: int = 1000
    save_path: str = "models/dqn.pth"


class Trainer:
    """Single responsibility: run episodes, call agent hooks, log metrics."""
    def __init__(self, env: Environment, agent: DQNAgent,
                 logger: Logger, cfg: TrainConfig):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.cfg = cfg

    def run(self) -> str:
        os.makedirs(os.path.dirname(self.cfg.save_path) or ".", exist_ok=True)
        device = self.agent.device

        for episode in range(self.cfg.episodes):
            state, _ = self.env.reset()
            self.agent.strategy.on_episode_start()
            total_reward = 0.0
            total_loss = 0.0
            loss_count = 0
            steps = 0

            for steps in range(1, self.cfg.max_steps_per_episode + 1):
                action = self.agent.select_action(state)
                next_state, reward, done, trunc, _ = self.env.step(action.item())

                self.agent.observe_transition(state, action, next_state, reward, done)
                bonus = self.agent.bonus(state, action, next_state, reward)
                total_reward += reward
                augmented = reward + bonus

                r_t = torch.tensor([augmented], device=device, dtype=torch.float32)
                self.agent.push(state, action, next_state, r_t)

                loss = self.agent.optimize()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
                    # VDBE needs the TD-error signal we just produced (approx via loss).
                    if isinstance(self.agent.strategy, VDBE):
                        self.agent.strategy.set_last_td(math.sqrt(loss))

                if done or trunc:
                    break
                state = next_state

            metrics = {
                "reward": total_reward,
                "steps": steps,
                "loss": total_loss / max(1, loss_count),
            }
            if isinstance(self.agent.strategy, EpsilonGreedy):
                metrics["epsilon"] = self.agent.strategy.epsilon(self.agent._step)
            if isinstance(self.agent.strategy, Boltzmann):
                metrics["temperature"] = self.agent.strategy.temperature(self.agent._step)
            self.logger.log_metrics(metrics, step=episode)

        self.agent.save(self.cfg.save_path)
        return self.cfg.save_path
