"""Logger implementations behind a common Protocol (DIP).

Default is `StdoutLogger` (no external deps). Comet is opt-in."""
from __future__ import annotations
from typing import Any, Optional


class StdoutLogger:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def log_params(self, params: dict[str, Any]) -> None:
        print(f"{self.prefix}params: {params}")

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        kv = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())
        print(f"{self.prefix}step={step} {kv}")

    def close(self) -> None:
        return None


class CometLogger:
    """Lazy import so comet_ml is an optional dependency."""
    def __init__(self, api_key: Optional[str] = None,
                 project_name: str = "rlexplore", workspace: Optional[str] = None):
        import comet_ml  # noqa: F401
        self._exp = comet_ml.start(
            api_key=api_key, project_name=project_name, workspace=workspace,
        )

    def log_params(self, params): self._exp.log_parameters(params)
    def log_metrics(self, metrics, step): self._exp.log_metrics(metrics, step=step)
    def close(self): self._exp.end()


def make_logger(kind: str = "stdout", **kwargs):
    if kind == "stdout":
        return StdoutLogger(**kwargs)
    if kind == "comet":
        return CometLogger(**kwargs)
    raise ValueError(f"unknown logger: {kind}")
