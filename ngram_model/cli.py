"""Command-line interface for the n-gram language model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .model import NGramLanguageModel


def positive_integer(value: str) -> int:
    #Parse a command-line value as an integer greater than zero.
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an n-gram model from a text file and interactively predict "
            "the next word."
        )
    )
    parser.add_argument("corpus", type=Path, help="UTF-8 .txt training file")
    parser.add_argument(
        "-n",
        "--context-size",
        type=positive_integer,
        default=2,
        help="number of preceding words used for prediction (default: 2)",
    )
    parser.add_argument(
        "--min-count",
        type=positive_integer,
        default=1,
        help="words seen fewer times become <UNK> (default: 1)",
    )
    parser.add_argument(
        "--top-k",
        type=positive_integer,
        default=5,
        help="number of next-word suggestions to display (default: 5)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="training file encoding (default: utf-8)",
    )
    return parser


def run_interactive(model: NGramLanguageModel, top_k: int) -> None:
    print(
        f"\nEnter at least {model.context_size} words. The last "
        f"{model.context_size} words will be used to predict the next word."
    )
    print("Type :quit to exit.\n")

    while True:
        try:
            sentence = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if sentence.lower() in {":quit", ":q", "quit", "exit"}:
            print("Goodbye.")
            return

        tokens = model.tokenize(sentence)
        if len(tokens) < model.context_size:
            print(f"Please enter at least {model.context_size} words.\n")
            continue

        predictions = model.predict_next(tokens, top_k=top_k)
        print("Next-word predictions:")
        for rank, prediction in enumerate(predictions, start=1):
            print(f"  {rank}. {prediction.word:<20} {prediction.probability:.6f}")
        print(f"Sentence perplexity: {model.perplexity(tokens):.6f}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.corpus.suffix.lower() != ".txt":
        print("Error: the training corpus must be a .txt file.")
        return 2
    if not args.corpus.is_file():
        print(f"Error: file not found: {args.corpus}")
        return 2

    model = NGramLanguageModel(
        context_size=args.context_size,
        min_count=args.min_count,
    )
    try:
        model.fit_file(args.corpus, encoding=args.encoding)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: could not train model: {error}")
        return 1

    print(
        "Training complete: "
        f"{model.training_token_count} tokens, "
        f"{len(model.vocabulary)} known words, "
        f"context size {model.context_size}."
    )
    run_interactive(model, args.top_k)
    return 0
