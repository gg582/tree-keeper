"""Download a small Wikipedia extract for experimentation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError, RedirectError


def fetch_sentences(topic: str, sentences: int) -> Iterable[str]:
    """Return up to ``sentences`` cleaned sentences for ``topic``.

    The :mod:`wikipedia` package provides a compliant wrapper around the
    MediaWiki API which automatically sets an appropriate user agent.  We
    attempt to resolve simple disambiguation pages by picking the first option
    and surface any other errors with a clear message so callers can handle
    them.
    """

    wikipedia.set_lang("en")
    try:
        page = wikipedia.page(topic, auto_suggest=False, preload=False)
    except DisambiguationError as exc:
        page = wikipedia.page(exc.options[0], auto_suggest=False, preload=False)
    except RedirectError as exc:
        page = wikipedia.page(exc.title, auto_suggest=False, preload=False)
    except PageError as exc:
        raise RuntimeError(f"Topic '{topic}' could not be resolved: {exc}") from exc

    sentences_iter = [segment.strip() for segment in page.content.split(".\n") if segment.strip()]
    return sentences_iter[:sentences]


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
