"""Smoke tests for the package skeleton."""

from __future__ import annotations

import subprocess
import sys

from localsec import __version__
from localsec.cli import build_parser, main


def test_version_is_exposed() -> None:
    assert __version__


def test_parser_is_built() -> None:
    parser = build_parser()

    assert parser.prog == "localsec"


def test_main_runs_without_arguments() -> None:
    assert main([]) == 0


def test_the_patch_command_is_offered() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["patch", "a bug", "--workspace", "/tmp/repo", "--yes"],
    )

    assert args.command == "patch"
    assert args.description == "a bug"
    assert args.yes is True


def test_module_entry_point_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "localsec", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert __version__ in result.stdout
