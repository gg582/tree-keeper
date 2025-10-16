"""Command line entry point for training the DoDAG language model."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from dodag_lm.config import TrainingConfig
from dodag_lm.dataset import DodagGraphDataset, build_dodag_from_corpus, corpus_from_file, split_edges
from dodag_lm.train_loop import evaluate, train
from dodag_lm.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DoDAG language model")
    parser.add_argument("corpus", type=Path, help="Path to a whitespace-tokenised corpus")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--negative-samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("model.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = TrainingConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        negative_samples=args.negative_samples,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    corpus = corpus_from_file(args.corpus)
    graph, vocab = build_dodag_from_corpus(corpus, window=args.window)
    train_edges, valid_edges, test_edges = split_edges(graph.edge_list, seed=config.seed)

    train_dataset = DodagGraphDataset(train_edges, len(vocab), config.negative_samples, seed=config.seed)
    valid_dataset = DodagGraphDataset(valid_edges, len(vocab), config.negative_samples, seed=config.seed + 1)
    test_dataset = DodagGraphDataset(test_edges, len(vocab), config.negative_samples, seed=config.seed + 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    model = train(config, train_dataset, valid_dataset, len(vocab), device=device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    metrics = evaluate(model, test_loader, device=device)

    logger.info("Test metrics: %s", json.dumps(metrics, indent=2))

    payload = {
        "model_state": model.state_dict(),
        "vocab": vocab.idx_to_token,
        "config": config.to_dict(),
    }
    torch.save(payload, args.output)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
