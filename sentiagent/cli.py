"""
cli.py
------
Command-line interface for SentiAgent.

Usage:
    python -m sentiagent.cli --text "The phone camera is excellent, but the battery drains quickly."
    python -m sentiagent.cli --file reviews.txt --output results.json
    echo "Great sound, terrible battery life." | python -m sentiagent.cli --stdin
"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import SentiAgentPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SentiAgent multi-agent sentiment analysis pipeline."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="A single review to analyze.")
    input_group.add_argument(
        "--file", type=str, help="Path to a text file with one review per line."
    )
    input_group.add_argument(
        "--stdin", action="store_true", help="Read a single review from stdin."
    )

    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-6", help="LLM model identifier."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider backend.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to write JSON results."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print pipeline stage progress."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.text:
        reviews = [args.text]
    elif args.stdin:
        reviews = [sys.stdin.read().strip()]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            reviews = [line.strip() for line in f if line.strip()]

    pipeline = SentiAgentPipeline(
        model=args.model, backend=args.backend, verbose=args.verbose
    )

    results = [pipeline.run(review).to_dict() for review in reviews]

    output_json = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Wrote {len(results)} result(s) to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
