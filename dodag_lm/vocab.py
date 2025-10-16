"""Vocabulary helpers for mapping between tokens and indices."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


class Vocabulary:
    """A minimal vocabulary wrapper with <root> and <unk> tokens."""

    root_token: str = "<root>"
    unk_token: str = "<unk>"

    def __init__(self) -> None:
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: List[str] = []
        self.add_token(self.root_token)
        self.add_token(self.unk_token)

    def add_token(self, token: str) -> int:
        if token not in self.token_to_idx:
            idx = len(self.idx_to_token)
            self.token_to_idx[token] = idx
            self.idx_to_token.append(token)
        return self.token_to_idx[token]

    def build_from_corpus(self, corpus: Iterable[List[str]], min_freq: int = 2) -> None:
        counter: Counter[str] = Counter()
        for sentence in corpus:
            counter.update(sentence)
        for token, freq in counter.items():
            if freq >= min_freq:
                self.add_token(token)

    def encode(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def decode(self, idx: int) -> str:
        return self.idx_to_token[idx]

    def __len__(self) -> int:
        return len(self.idx_to_token)
