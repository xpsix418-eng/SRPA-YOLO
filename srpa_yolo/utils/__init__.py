"""Path and reproducibility utilities."""

import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


__all__ = ("ROOT", "set_reproducible_seed")
