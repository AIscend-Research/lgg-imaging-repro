"""Determinism, config loading, small shared helpers (R7)."""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed random / numpy / torch and (where feasible) enable deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device(prefer_cuda: bool = True):
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def deep_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
