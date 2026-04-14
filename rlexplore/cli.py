"""Command-line entry point."""
from __future__ import annotations
import argparse
import random
import numpy as np
import torch

from .config import ExperimentConfig, build
from .training import Trainer, Evaluator


def _seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(prog="rlexplore")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    p.add_argument("--model-path", default=None)
    args = p.parse_args()

    cfg = ExperimentConfig.load(args.config)
    _seed(cfg.seed)
    env, agent, logger = build(cfg)
    logger.log_params({
        "strategy": cfg.exploration.type,
        "model": cfg.model.type,
        "env": cfg.environment.type,
        **cfg.exploration.params,
    })

    model_path = args.model_path
    if args.mode in ("train", "both"):
        trainer = Trainer(env, agent, logger, cfg.training)
        model_path = trainer.run()

    if args.mode in ("eval", "both"):
        if model_path is None:
            raise SystemExit("--model-path required for eval-only mode")
        agent.load(model_path)
        evaluator = Evaluator(env, agent.policy_net, agent.device, cfg.evaluation)
        results = evaluator.run()
        logger.log_metrics(results, step=cfg.training.episodes)

    logger.close()
    env.close()


if __name__ == "__main__":
    main()
