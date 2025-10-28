"""Text processing helpers that leverage spaCy for tokenisation and parsing."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterator, List, Sequence

try:  # pragma: no cover - imported lazily in environments without spaCy
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc
except Exception:  # pragma: no cover - spaCy is optional at runtime
    spacy = None  # type: ignore[assignment]
    Language = Doc = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class DependencyTree:
    """Container describing a dependency parse over a single sentence."""

    tokens: List[str]
    heads: List[int]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.heads):  # pragma: no cover - sanity guard
            raise ValueError("tokens and heads must be the same length")


def load_spacy_pipeline(model: str | None = "en_core_web_sm") -> Language | None:
    """Load a spaCy pipeline, falling back to a blank pipeline when unavailable."""

    if spacy is None:
        logger.warning("spaCy is not installed; dependency parsing will be disabled")
        return None

    if model is None:
        pipeline = spacy.blank("en")
    else:
        try:
            pipeline = spacy.load(model)
        except Exception as exc:  # pragma: no cover - depends on runtime env
            logger.warning(
                "Unable to load spaCy model '%s' (%s); falling back to blank pipeline",
                model,
                exc,
            )
            pipeline = spacy.blank("en")

    if "sentencizer" not in pipeline.pipe_names:
        pipeline.add_pipe("sentencizer")

    if "parser" not in pipeline.pipe_names:
        logger.warning(
            "spaCy pipeline has no dependency parser; falling back to sequential edges"
        )

    return pipeline


def read_text_corpus(path: str | Path) -> List[str]:
    """Read a corpus treating each non-empty line as a raw sentence."""

    path = Path(path)
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _parse_with_spacy(doc: Doc) -> Iterator[DependencyTree]:
    """Extract dependency parses from a processed spaCy document."""

    has_parser = doc.has_annotation("DEP") and doc.has_annotation("HEAD")
    for sent in doc.sents:
        tokens = [token.text for token in sent]
        if has_parser:
            heads = [
                token.head.i - sent[0].i if token.head != token else -1
                for token in sent
            ]
        else:
            # Sequential fallback connects tokens in order.
            heads = [-1] + list(range(len(sent) - 1))
        yield DependencyTree(tokens=tokens, heads=heads)


def dependency_trees_from_texts(
    texts: Sequence[str],
    pipeline: Language | None,
    batch_size: int = 32,
) -> List[DependencyTree]:
    """Generate dependency trees for a set of sentences."""

    if pipeline is None:
        fallback_trees: List[DependencyTree] = []
        for text in texts:
            tokens = text.split()
            if not tokens:
                continue
            heads = [-1] + list(range(len(tokens) - 1))
            fallback_trees.append(DependencyTree(tokens=tokens, heads=heads))
        return fallback_trees

    trees: List[DependencyTree] = []
    for doc in pipeline.pipe(texts, batch_size=batch_size):
        trees.extend(_parse_with_spacy(doc))
    return trees


__all__ = [
    "DependencyTree",
    "dependency_trees_from_texts",
    "load_spacy_pipeline",
    "read_text_corpus",
]
