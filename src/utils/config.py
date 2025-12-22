"""Utility helpers to load YAML configs into namespace-style objects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(item) for item in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a YAML file and return nested SimpleNamespace objects."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _to_namespace(data)


def namespace_to_dict(ns: Any) -> Any:
    """Recursively convert a SimpleNamespace back into primitive types."""
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in ns.__dict__.items()}
    if isinstance(ns, list):
        return [namespace_to_dict(item) for item in ns]
    return ns
