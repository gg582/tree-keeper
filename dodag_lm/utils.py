"""Utility helpers for data preparation, reproducibility, and IO."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Generator, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch

from .config import TrainingConfig
from .model import DodagLanguageModel
from .vocab import Vocabulary


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


def load_checkpoint(model_path: Path | str, device: Optional[str] = None) -> tuple[DodagLanguageModel, Vocabulary, TrainingConfig]:
    """Load a serialized checkpoint into memory.

    Parameters
    ----------
    model_path:
        Path to the ``.pt`` checkpoint produced by :mod:`train.py`.
    device:
        Optional device override.  Defaults to ``"cuda"`` when available
        otherwise ``"cpu"``.
    """

    resolved = Path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Model checkpoint {resolved} does not exist")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(resolved, map_location=device)
    tokens = payload["vocab"]
    config_dict = payload["config"]

    vocab = Vocabulary()
    vocab.idx_to_token = tokens
    vocab.token_to_idx = {token: idx for idx, token in enumerate(tokens)}

    config = TrainingConfig(**config_dict)
    model = DodagLanguageModel(len(vocab), config.embedding_dim, config.hidden_dim)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, vocab, config
