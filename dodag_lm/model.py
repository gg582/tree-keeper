"""Model architecture for the DoDAG language model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from . import cuda_backend


class HebbianGroupMemory(nn.Module):
    """Neuro-inspired associative memory for grouping related tokens."""

    def __init__(
        self,
        embedding_dim: int,
        groups: int,
        decay: float = 0.97,
        temperature: float = 0.7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be >= 1")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.temperature = max(temperature, eps)
        self.eps = eps
        patterns = torch.randn(groups, embedding_dim)
        patterns = F.normalize(patterns, dim=-1)
        self.register_buffer("patterns", patterns)
        self.register_buffer("usage", torch.ones(groups))

    @property
    def groups(self) -> int:
        return self.patterns.size(0)

    def assign(self, pair_embed: torch.Tensor) -> torch.Tensor:
        """Return soft assignments of pairs to group prototypes."""

        if pair_embed.numel() == 0:
            return torch.empty(
                *pair_embed.shape[:-1],
                self.groups,
                device=pair_embed.device,
                dtype=pair_embed.dtype,
            )

        flat = pair_embed.reshape(-1, pair_embed.size(-1))
        queries = F.normalize(flat, dim=-1)
        patterns = F.normalize(self.patterns, dim=-1)
        logits = queries @ patterns.t()
        homeostasis = torch.log(self.usage.clamp_min(self.eps))
        logits = logits - homeostasis.unsqueeze(0)
        logits = logits / self.temperature
        weights = torch.softmax(logits, dim=-1)
        return weights.view(*pair_embed.shape[:-1], self.groups)

    def update(self, pair_embed: torch.Tensor, assignments: torch.Tensor) -> None:
        """Hebbian-style update of group prototypes."""

        if not self.training:
            return
        if pair_embed.numel() == 0:
            return

        flat_embed = pair_embed.reshape(-1, pair_embed.size(-1))
        flat_assign = assignments.reshape(-1, assignments.size(-1))
        if flat_assign.numel() == 0:
            return

        weights_sum = flat_assign.sum(dim=0)
        valid = weights_sum > 0
        if not torch.any(valid):
            return

        with torch.no_grad():
            self.usage.mul_(self.decay)
            usage_updates = weights_sum[valid]
            self.usage[valid] = self.usage[valid] + (1 - self.decay) * usage_updates

            updates = torch.zeros_like(self.patterns)
            updates[valid] = flat_assign[:, valid].t() @ flat_embed
            updates[valid] = updates[valid] / (weights_sum[valid].unsqueeze(-1) + self.eps)

            current = self.patterns[valid]
            blended = current * self.decay + (1 - self.decay) * updates[valid]
            blended = F.normalize(blended, dim=-1)
            self.patterns[valid] = blended


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
    """Parent/child bilinear model with grouped DoDAG dynamics.

    The network replaces a conventional recurrent cell with a single DoDAG that
    is internally organised into a handful of associative groups. Each group
    acts like a bundle of tensors reserved for semantically related words. A
    Hebbian memory module encourages tokens that frequently co-occur to share a
    group, mirroring prior neurobiological models of cell assemblies.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        mixture_components: int,
        hebbian_decay: float = 0.97,
        hebbian_temperature: float = 0.7,
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
        self.memory = HebbianGroupMemory(
            embedding_dim,
            mixture_components,
            decay=hebbian_decay,
            temperature=hebbian_temperature,
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

    def _group_assignments(self, pair_embed: torch.Tensor) -> torch.Tensor:
        return self.memory.assign(pair_embed)

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
        pair_embed = 0.5 * (parent_embed + child_embed)
        gates = self._group_assignments(pair_embed)
        if self.training:
            self.memory.update(pair_embed.detach(), gates.detach())

        unit_scores = cuda_backend.bilinear_score(parent_proj, child_proj)
        positive_scores = (gates * unit_scores).sum(dim=-1)

        negative_scores = None
        if negative_indices is not None:
            neg_embed = self.embeddings(negative_indices)
            neg_proj = self._project_children(neg_embed)
            parent_proj_expanded = parent_proj.unsqueeze(2).expand_as(neg_proj)
            unit_neg_scores = cuda_backend.bilinear_score(parent_proj_expanded, neg_proj)
            neg_pair_embed = 0.5 * (parent_embed.unsqueeze(1) + neg_embed)
            neg_gates = self._group_assignments(neg_pair_embed).transpose(1, 2)
            negative_scores = (neg_gates * unit_neg_scores).sum(dim=1)
        return positive_scores, negative_scores

    def predict_child(self, parent_indices: torch.Tensor) -> torch.Tensor:
        parent_embed = self.embeddings(parent_indices)
        parent_proj = self._project_parents(parent_embed)

        logits_per_unit = []
        child_weight = self.embeddings.weight
        for idx, unit in enumerate(self.units):
            parent_proj_unit = parent_proj[:, idx, :]
            child_proj_unit = self.child_norm(unit.project_child(child_weight))
            logits_per_unit.append(parent_proj_unit @ child_proj_unit.t())

        logits_stack = torch.stack(logits_per_unit, dim=1)
        pair_embed = 0.5 * (parent_embed.unsqueeze(1) + child_weight.unsqueeze(0))
        gates = self._group_assignments(pair_embed).permute(0, 2, 1)
        logits = (gates * logits_stack).sum(dim=1)
        return logits
