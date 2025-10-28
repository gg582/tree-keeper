"""DoDAG-based language modeling utilities."""

from .config import TrainingConfig
from .model import DodagLanguageModel
from .train_loop import train, evaluate
from .graph import DodagGraph, DodagNode
from .dataset import (
    DodagGraphDataset,
    build_dodag_from_corpus,
    build_dodag_from_dependency_trees,
    corpus_from_file,
)
from .text_processing import (
    DependencyTree,
    dependency_trees_from_texts,
    load_spacy_pipeline,
    read_text_corpus,
)

__all__ = [
    "TrainingConfig",
    "DodagLanguageModel",
    "train",
    "evaluate",
    "DodagGraph",
    "DodagNode",
    "DodagGraphDataset",
    "build_dodag_from_corpus",
    "build_dodag_from_dependency_trees",
    "corpus_from_file",
    "DependencyTree",
    "dependency_trees_from_texts",
    "load_spacy_pipeline",
    "read_text_corpus",
]
