"""CUDA accelerated routines for the DoDAG language model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_extension: Optional[torch.types.ModuleType] = None


def _load_extension() -> None:
    global _extension
    if _extension is not None:
        return

    if not torch.cuda.is_available():
        logger.info("CUDA not available; using PyTorch implementation.")
        _extension = None
        return

    try:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        sources = [
            str(root / "csrc" / "embedding_kernel.cpp"),
            str(root / "csrc" / "embedding_kernel.cu"),
        ]
        _extension = load(
            name="dodag_cuda",
            sources=sources,
            verbose=False,
            extra_cuda_cflags=["-O2"],
        )
        logger.info("Loaded custom CUDA extension for bilinear scoring.")
    except Exception as exc:  # pragma: no cover - extension build is optional
        logger.warning("Falling back to PyTorch implementation: %s", exc)
        _extension = None


def bilinear_score(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
    """Compute the bilinear score for parent/child embeddings."""

    if parent.shape != child.shape:
        raise ValueError("parent and child tensors must share the same shape")

    original_shape = parent.shape[:-1]
    parent_flat = parent.reshape(-1, parent.shape[-1])
    child_flat = child.reshape(-1, child.shape[-1])

    if _extension is None:
        _load_extension()

    if _extension is not None:
        scores = _extension.dodag_bilinear_forward(parent_flat, child_flat)
    else:
        scores = (parent_flat * child_flat).sum(dim=-1)

    return scores.view(original_shape)
