"""Configuration dataclasses for the DoDAG language model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters controlling model training."""

    embedding_dim: int = 96
    hidden_dim: int = 128
    negative_samples: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 8
    gradient_clip: float = 1.0
    seed: int = 7

    def to_dict(self) -> dict[str, float | int]:
        """Return a dictionary representation for logging."""

        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "negative_samples": self.negative_samples,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "gradient_clip": self.gradient_clip,
            "seed": self.seed,
        }
