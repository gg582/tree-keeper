"""Model architecture for the DoDAG language model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from . import cuda_backend


class _DodagUnit(nn.Module):
    """Projection block for a single DoDAG mixture component."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
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

    def project_parent(self, parent_embed: torch.Tensor) -> torch.Tensor:
        return self.parent_projection(parent_embed)

    def project_child(self, child_embed: torch.Tensor) -> torch.Tensor:
        return self.child_projection(child_embed)


class DodagLanguageModel(nn.Module):
    """A mixture-based parent/child bilinear language model.

    The original DoDAG model used a single projection tower for parent and child
    embeddings.  That design makes it difficult for the model to specialise on
    fragmented relational structure.  We instead maintain multiple projection
    units and learn a soft gating function that assigns every edge to a
    component-specific DoDAG.  This keeps the overall DoDAG intuition while
    allowing the network to pick unit-specific representations when different
    token relations conflict with one another.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        mixture_components: int,
    ) -> None:
        super().__init__()
        if mixture_components < 1:
            raise ValueError("mixture_components must be >= 1")

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.units = nn.ModuleList(
            [_DodagUnit(embedding_dim, hidden_dim) for _ in range(mixture_components)]
        )
        self.parent_norm = nn.LayerNorm(embedding_dim)
        self.child_norm = nn.LayerNorm(embedding_dim)
        self.gating = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mixture_components),
        )

    @property
    def mixture_components(self) -> int:
        return len(self.units)

    def _project_parents(self, parent_embed: torch.Tensor) -> torch.Tensor:
        projections = [
            self.parent_norm(unit.project_parent(parent_embed)) for unit in self.units
        ]
        return torch.stack(projections, dim=1)

    def _project_children(self, child_embed: torch.Tensor) -> torch.Tensor:
        projections = [
            self.child_norm(unit.project_child(child_embed)) for unit in self.units
        ]
        return torch.stack(projections, dim=1)

    def _gating_weights(self, parent_embed: torch.Tensor) -> torch.Tensor:
        logits = self.gating(parent_embed)
        return logits.softmax(dim=-1)

    def forward(
        self,
        parent_indices: torch.Tensor,
        child_indices: torch.Tensor,
        negative_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        parent_embed = self.embeddings(parent_indices)
        child_embed = self.embeddings(child_indices)

        parent_proj = self._project_parents(parent_embed)
        child_proj = self._project_children(child_embed)
        gates = self._gating_weights(parent_embed)

        unit_scores = cuda_backend.bilinear_score(parent_proj, child_proj)
        positive_scores = (gates * unit_scores).sum(dim=-1)

        negative_scores = None
        if negative_indices is not None:
            neg_embed = self.embeddings(negative_indices)
            neg_proj = self._project_children(neg_embed)
            parent_proj_expanded = parent_proj.unsqueeze(2).expand_as(neg_proj)
            unit_neg_scores = cuda_backend.bilinear_score(parent_proj_expanded, neg_proj)
            negative_scores = (gates.unsqueeze(-1) * unit_neg_scores).sum(dim=1)
        return positive_scores, negative_scores

    def predict_child(self, parent_indices: torch.Tensor) -> torch.Tensor:
        parent_embed = self.embeddings(parent_indices)
        gates = self._gating_weights(parent_embed)

        logits_per_unit = []
        for unit in self.units:
            parent_proj = self.parent_norm(unit.project_parent(parent_embed))
            child_proj = self.child_norm(unit.project_child(self.embeddings.weight))
            logits_per_unit.append(parent_proj @ child_proj.t())

        logits_stack = torch.stack(logits_per_unit, dim=-1)
        logits = (gates.unsqueeze(1) * logits_stack).sum(dim=-1)
        return logits
