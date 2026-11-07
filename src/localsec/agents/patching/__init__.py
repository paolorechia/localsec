"""The security patching agents.

Three agents with one job each -- triage the report, write the fix, check
the fix -- and a run that drives them. Everything that changes the workspace
goes through the approval gate first.
"""

from __future__ import annotations

from localsec.agents.patching.agents import PatchAgent, TriageAgent, VerifyAgent
from localsec.agents.patching.orchestrator import PatchingRun, RunOutcome
from localsec.agents.patching.types import Edit, Finding, Findings, PatchPlan, Verification
from localsec.agents.patching.workspace import (
    RepoDeps,
    Workspace,
    WorkspaceError,
    mutating_tools,
    read_only_tools,
)

__all__ = [
    "Edit",
    "Finding",
    "Findings",
    "PatchAgent",
    "PatchPlan",
    "PatchingRun",
    "RepoDeps",
    "RunOutcome",
    "TriageAgent",
    "Verification",
    "VerifyAgent",
    "Workspace",
    "WorkspaceError",
    "mutating_tools",
    "read_only_tools",
]
