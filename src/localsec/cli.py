"""Command-line entry point for localsec."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from localsec import __version__
from localsec.agents.patching.orchestrator import PatchingRun, RunOutcome
from localsec.agents.patching.workspace import RepoDeps, Workspace
from localsec.runtime.approvals import ApprovalGate, Decision, console_approver
from localsec.runtime.audit import AuditLog
from localsec.runtime.provider import ModelSession
from localsec.runtime.sandbox import DockerSandbox


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``localsec`` command."""
    parser = argparse.ArgumentParser(
        prog="localsec",
        description="An agentic system.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    commands = parser.add_subparsers(dest="command")
    patch = commands.add_parser(
        "patch",
        help="Have the agents find and fix a vulnerability in a tree.",
        description=(
            "Triage a report, plan a fix, apply it, and check it. Every change "
            "is put to the approval gate first."
        ),
    )
    patch.add_argument("description", help="what the suspected vulnerability is")
    patch.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="the tree the agents work in",
    )
    patch.add_argument(
        "--image",
        default="python:3.12-slim",
        help="the container the tests run in",
    )
    patch.add_argument(
        "--test-command",
        default="",
        help="how to run the tests; the agents decide when this is not set",
    )
    patch.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="how many times to try the fix before giving up",
    )
    patch.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="where to write the audit log",
    )
    patch.add_argument(
        "--yes",
        action="store_true",
        help="approve every change without asking",
    )
    patch.add_argument(
        "--no-tests",
        action="store_true",
        help="do not run the tests, and do not use the sandbox",
    )
    return parser


def _report(outcome: RunOutcome) -> None:
    """Print what the run did."""
    print("\nFindings:")
    if outcome.findings.findings:
        for finding in outcome.findings.findings:
            where = finding.file + (f":{finding.line}" if finding.line else "")
            print(f"  [{finding.severity}] {finding.title} ({where})")
            if finding.description:
                print(f"    {finding.description}")
    else:
        print("  none")

    print("\nPlan:")
    if outcome.plan.summary:
        print(f"  {outcome.plan.summary}")
    for edit in outcome.plan.edits:
        print(f"  {edit.path}: {edit.rationale or 'change'}")
    if not outcome.plan.edits:
        print("  no changes proposed")

    print("\nWhat happened:")
    print(f"  applied:  {', '.join(outcome.applied) or 'nothing'}")
    print(f"  declined: {', '.join(outcome.declined) or 'nothing'}")
    print(f"  failed:   {', '.join(outcome.failed) or 'nothing'}")
    print(f"  rounds:   {outcome.rounds}")

    if outcome.verification is not None:
        print(f"\nVerification: {'passed' if outcome.verification.passed else 'failed'}")
        print(f"  {outcome.verification.summary}")
        for issue in outcome.verification.remaining_issues:
            print(f"  - {issue}")


async def _patch(args: argparse.Namespace) -> int:
    """Run the patching agents."""
    workspace = Workspace(args.workspace)
    sandbox: DockerSandbox | None = None
    if not args.no_tests:
        sandbox = DockerSandbox(image=args.image)
        if not sandbox.available():
            print("docker is not available, so the tests cannot be run", file=sys.stderr)
            sandbox = None

    deps = RepoDeps(
        workspace=workspace,
        sandbox=sandbox,
        test_command=args.test_command,
    )
    audit = AuditLog(args.audit) if args.audit is not None else None
    approver = (lambda _request: Decision.ALLOW) if args.yes else console_approver
    gate = ApprovalGate(approver, audit=audit)

    async with ModelSession() as session:
        run = PatchingRun(
            session.model,
            deps,
            gate=gate,
            audit=audit,
            max_rounds=args.rounds,
        )
        outcome = await run.run(args.description)

    _report(outcome)
    return 0 if outcome.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``localsec`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "patch":
        return asyncio.run(_patch(args))
    parser.error(f"unknown command: {args.command}")
