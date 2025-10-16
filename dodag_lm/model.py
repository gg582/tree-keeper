"""Model architecture for the DoDAG language model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from . import cuda_backend


class DodagLanguageModel(nn.Module):
    """A parent->child bilinear language model."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.parent_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.child_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        parent_indices: torch.Tensor,
        child_indices: torch.Tensor,
        negative_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        parent_embed = self.embeddings(parent_indices)
        child_embed = self.embeddings(child_indices)
        parent_embed = self.layer_norm(self.parent_projection(parent_embed))
        child_embed = self.layer_norm(self.child_projection(child_embed))

        positive_scores = cuda_backend.bilinear_score(parent_embed, child_embed)

        negative_scores = None
        if negative_indices is not None:
            neg_embed = self.embeddings(negative_indices)
            parent_expanded = parent_embed.unsqueeze(1).expand_as(neg_embed)
            neg_embed = self.layer_norm(self.child_projection(neg_embed))
            negative_scores = cuda_backend.bilinear_score(parent_expanded, neg_embed)
        return positive_scores, negative_scores

    def predict_child(self, parent_indices: torch.Tensor) -> torch.Tensor:
        parent_embed = self.embeddings(parent_indices)
        parent_embed = self.layer_norm(self.parent_projection(parent_embed))
        child_embeds = self.layer_norm(self.child_projection(self.embeddings.weight))
        logits = parent_embed @ child_embeds.t()
        return logits
