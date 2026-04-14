"""Registry pattern for plugins (OCP): add new strategies/envs/models
without modifying the trainer."""
from __future__ import annotations
from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable] = {}

    def register(self, key: str):
        def deco(cls):
            if key in self._items:
                raise ValueError(f"{self.name}: '{key}' already registered")
            self._items[key] = cls
            return cls
        return deco

    def get(self, key: str) -> Callable:
        if key not in self._items:
            raise KeyError(f"{self.name}: unknown '{key}'. Known: {list(self._items)}")
        return self._items[key]

    def keys(self) -> list[str]:
        return list(self._items)


ENVS = Registry("env")
MODELS = Registry("model")
STRATEGIES = Registry("strategy")
