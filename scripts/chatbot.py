"""Interactive chatbot loop for the DoDAG language model."""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

import torch

from dodag_lm.utils import load_checkpoint
from dodag_lm.vocab import Vocabulary


def sample_next_token(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    logits = logits.squeeze(0)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        scores, indices = torch.topk(logits, top_k)
        probs = torch.softmax(scores, dim=-1)
        choice = torch.multinomial(probs, 1).item()
        return indices[choice].item()

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_response(
    parent_token: str,
    max_tokens: int,
    top_k: int,
    temperature: float,
    device: str,
    model,
    vocab: Vocabulary,
) -> List[str]:
    response_tokens: List[str] = []
    parent_idx = torch.tensor([vocab.encode(parent_token)], device=device)

    for _ in range(max_tokens):
        logits = model.predict_child(parent_idx)
        next_idx = sample_next_token(logits, top_k, temperature)
        next_token = vocab.decode(next_idx)

        if next_token == Vocabulary.root_token:
            break
        if response_tokens and response_tokens[-1] == next_token:
            break

        response_tokens.append(next_token)
        parent_idx = torch.tensor([next_idx], device=device)
    return response_tokens


def normalise_user_message(message: str) -> Iterable[str]:
    message = message.replace("\n", " ")
    return [token for token in message.strip().split(" ") if token]


def interactive_chat(args: argparse.Namespace) -> None:
    device = args.device
    model, vocab, _ = load_checkpoint(args.model_path, device)

    context: List[str] = [Vocabulary.root_token]
    print("Loaded DoDAG model. Type 'quit' to exit.")

    while True:
        try:
            user_message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()  # newline for clean exit
            break

        if user_message.strip().lower() in {"quit", "exit"}:
            break

        user_tokens = list(normalise_user_message(user_message)) or [Vocabulary.unk_token]
        context.extend(user_tokens)

        parent_token = context[-1]
        bot_tokens = generate_response(
            parent_token=parent_token,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            device=device,
            model=model,
            vocab=vocab,
        )

        if not bot_tokens:
            print("Bot: (no confident response)")
            continue

        context.extend(bot_tokens)
        print("Bot:", " ".join(bot_tokens))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive DoDAG chatbot")
    parser.add_argument("model_path", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tokens", type=int, default=30, dest="max_tokens")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    interactive_chat(args)


if __name__ == "__main__":
    main(sys.argv[1:])
