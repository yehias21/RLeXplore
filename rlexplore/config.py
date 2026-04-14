"""Declarative config + factory.

OCP: to add a new strategy/env/model, register it; configs reference it by name.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import yaml
import torch

from .core.registry import ENVS, MODELS, STRATEGIES
# Registration side-effects:
from . import envs  # noqa: F401
from . import models  # noqa: F401
from . import exploration  # noqa: F401

from .agents import DQNAgent, DQNConfig
from .training import TrainConfig, EvalConfig
from .logging_ import make_logger


@dataclass
class Block:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    environment: Block
    model: Block
    exploration: Block
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    logger: Block = field(default_factory=lambda: Block(type="stdout"))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            environment=Block(**d["environment"]),
            model=Block(**d["model"]),
            exploration=Block(**d["exploration"]),
            dqn=DQNConfig(**d.get("dqn", {})),
            training=TrainConfig(**d.get("training", {})),
            evaluation=EvalConfig(**d.get("evaluation", {})),
            logger=Block(**d.get("logger", {"type": "stdout"})),
            device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            seed=d.get("seed", 0),
        )

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)
        return cls.from_dict(data)


def build(cfg: ExperimentConfig):
    device = torch.device(cfg.device)
    env = ENVS.get(cfg.environment.type)(device=device, **cfg.environment.params)

    model_cls = MODELS.get(cfg.model.type)
    model_params = dict(cfg.model.params)
    model_params.setdefault("input_size", env.observation_size)
    model_params.setdefault("num_actions", env.num_actions)
    q_net = model_cls(**model_params)

    strat_cls = STRATEGIES.get(cfg.exploration.type)
    strat_params = dict(cfg.exploration.params)
    # Inject common fields strategies expect:
    strat_params.setdefault("num_actions", env.num_actions)
    strat_params.setdefault("device", device)
    if strat_cls.__name__ in {"RND", "ICM", "HashPseudoCount"}:
        strat_params.setdefault("input_size", env.observation_size)
    strategy = strat_cls(**strat_params)

    agent = DQNAgent(q_net=q_net, strategy=strategy, device=device, cfg=cfg.dqn)
    logger = make_logger(cfg.logger.type, **cfg.logger.params)
    return env, agent, logger
