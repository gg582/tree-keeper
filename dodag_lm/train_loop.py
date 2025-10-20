"""Training and evaluation loops for the DoDAG language model."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .dataset import DodagGraphDataset, build_dodag_from_corpus, corpus_from_file, split_edges
from .model import DodagLanguageModel
from .utils import set_seed
from .vocab import Vocabulary

logger = logging.getLogger(__name__)


def _prepare_dataloader(dataset: DodagGraphDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _loss_function(pos_scores: torch.Tensor, neg_scores: torch.Tensor | None) -> torch.Tensor:
    positive_loss = torch.nn.functional.softplus(-pos_scores).mean()
    if neg_scores is not None:
        negative_loss = torch.nn.functional.softplus(neg_scores).mean()
    else:
        negative_loss = torch.tensor(0.0, device=pos_scores.device)
    return positive_loss + negative_loss


def train(
    config: TrainingConfig,
    dataset: DodagGraphDataset,
    valid_dataset: DodagGraphDataset,
    vocab_size: int,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> DodagLanguageModel:
    """Train a DoDAG language model and return the fitted module."""

    set_seed(config.seed)
    model = DodagLanguageModel(
        vocab_size,
        config.embedding_dim,
        config.hidden_dim,
        config.mixture_components,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = _prepare_dataloader(dataset, config.batch_size)
    valid_loader = _prepare_dataloader(valid_dataset, config.batch_size, shuffle=False)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for parents, children, negatives in train_loader:
            parents = parents.to(device)
            children = children.to(device)
            negatives = negatives.to(device)
            optimizer.zero_grad()
            pos_scores, neg_scores = model(parents, children, negatives)
            loss = _loss_function(pos_scores, neg_scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            running_loss += loss.item() * parents.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        valid_metrics = evaluate(model, valid_loader, device)
        logger.info(
            "Epoch %d | train_loss=%.4f | valid_loss=%.4f | valid_accuracy=%.3f",
            epoch,
            epoch_loss,
            valid_metrics["loss"],
            valid_metrics["accuracy"],
        )

    return model


def evaluate(
    model: DodagLanguageModel,
    dataloader: DataLoader,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for parents, children, negatives in dataloader:
            parents = parents.to(device)
            children = children.to(device)
            negatives = negatives.to(device)
            pos_scores, neg_scores = model(parents, children, negatives)
            loss = _loss_function(pos_scores, neg_scores)
            total_loss += loss.item() * parents.size(0)
            total_samples += parents.size(0)

            logits = model.predict_child(parents)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == children).sum().item()

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": correct / max(total_samples, 1),
    }


def from_corpus_file(
    corpus_path: str,
    config: TrainingConfig,
    window: int = 2,
) -> Tuple[DodagLanguageModel, Vocabulary, Dict[str, float]]:
    """Train a model directly from a corpus file."""

    corpus = corpus_from_file(corpus_path)
    graph, vocab = build_dodag_from_corpus(corpus, window=window)
    train_edges, valid_edges, test_edges = split_edges(graph.edge_list, seed=config.seed)

    train_dataset = DodagGraphDataset(train_edges, len(vocab), config.negative_samples, seed=config.seed)
    valid_dataset = DodagGraphDataset(valid_edges, len(vocab), config.negative_samples, seed=config.seed + 1)
    test_dataset = DodagGraphDataset(test_edges, len(vocab), config.negative_samples, seed=config.seed + 2)

    model = train(config, train_dataset, valid_dataset, len(vocab))
    test_loader = _prepare_dataloader(test_dataset, config.batch_size, shuffle=False)
    metrics = evaluate(model, test_loader)
    return model, vocab, metrics
