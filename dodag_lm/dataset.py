"""Dataset helpers for DoDAG-based language modeling."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .graph import DodagGraph
from .vocab import Vocabulary
from .utils import sliding_window


class DodagGraphDataset(Dataset[Tuple[int, int, torch.Tensor]]):
    """A dataset of parent/child indices and sampled negatives."""

    def __init__(
        self,
        edges: Sequence[Tuple[int, int]],
        num_tokens: int,
        negative_samples: int,
        seed: int | None = None,
    ) -> None:
        self.edges = list(edges)
        self.num_tokens = num_tokens
        self.negative_samples = negative_samples
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Tuple[int, int, torch.Tensor]:
        parent, child = self.edges[idx]
        negatives = self._sample_negatives(child)
        return parent, child, negatives

    def _sample_negatives(self, positive_child: int) -> torch.Tensor:
        samples = []
        while len(samples) < self.negative_samples:
            candidate = self.random.randint(0, self.num_tokens - 1)
            if candidate != positive_child:
                samples.append(candidate)
        return torch.tensor(samples, dtype=torch.long)


def corpus_from_file(path: str | Path) -> List[List[str]]:
    """Read a simple whitespace-tokenised corpus from disk."""

    path = Path(path)
    sentences: List[List[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sentences.append(line.split())
    return sentences


def build_dodag_from_corpus(
    corpus: Iterable[List[str]],
    vocab: Vocabulary | None = None,
    window: int = 2,
) -> tuple[DodagGraph, Vocabulary]:
    """Construct a DoDAG graph from a sentence corpus.

    Each sentence becomes a chain of parent->child relationships anchored at
    a special root token. A sliding window is used so that every token is
    connected both to the preceding token and, when available, to tokens that
    appear within the configurable window.
    """

    if vocab is None:
        vocab = Vocabulary()
        vocab.build_from_corpus(corpus)

    root_id = vocab.encode(Vocabulary.root_token)
    graph = DodagGraph(root_id)

    for sentence in corpus:
        encoded = [vocab.encode(tok) for tok in sentence]
        for parent_idx, child_idx in sliding_window([root_id] + encoded, window_size=window + 1):
            if parent_idx is None or child_idx is None:
                continue
            graph.add_edge(parent_idx, child_idx)

    return graph, vocab


def split_edges(
    edges: Sequence[Tuple[int, int]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int | None = None,
) -> tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split edges into train/validation/test partitions."""

    assert math.isclose(sum(ratios), 1.0, rel_tol=1e-6)
    rng = random.Random(seed)
    shuffled = list(edges)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * ratios[0])
    valid_end = train_end + int(total * ratios[1])
    train = shuffled[:train_end]
    valid = shuffled[train_end:valid_end]
    test = shuffled[valid_end:]
    return train, valid, test
