"""Utility helpers for data preparation and reproducibility."""
from __future__ import annotations

import random
from typing import Generator, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch


T = TypeVar("T")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sliding_window(sequence: Sequence[T], window_size: int) -> Generator[Tuple[Optional[T], Optional[T]], None, None]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    for idx in range(len(sequence) - 1):
        parent_idx = max(0, idx - (window_size - 2))
        yield sequence[parent_idx], sequence[idx + 1]
