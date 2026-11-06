"""Command-line entry point for localsec."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from localsec import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``localsec`` command."""
    parser = argparse.ArgumentParser(
        prog="localsec",
        description="An agentic system.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``localsec`` command."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0
