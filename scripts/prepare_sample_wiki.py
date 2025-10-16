"""Download a small Wikipedia extract for experimentation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import requests

API_URL = "https://en.wikipedia.org/w/api.php"


def fetch_sentences(topic: str, sentences: int) -> Iterable[str]:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": topic,
        "format": "json",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", {})
    text = next(iter(pages.values())).get("extract", "")
    return [sent.strip() for sent in text.split(".\n") if sent.strip()][:sentences]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small Wikipedia corpus")
    parser.add_argument("topic", help="Page title to download")
    parser.add_argument("--sentences", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("data/wiki_sample.txt"))
    args = parser.parse_args()

    extracted = fetch_sentences(args.topic, args.sentences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sentence in extracted:
            tokens = sentence.replace("\n", " ").split()
            if not tokens:
                continue
            handle.write(" ".join(tokens) + "\n")
    print(f"Wrote {len(extracted)} sentences to {args.output}")


if __name__ == "__main__":
    main()
