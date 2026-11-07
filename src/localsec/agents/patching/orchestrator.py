"""A patching run: work out what is wrong, write the fix, check the fix.

The run drives the three agents and does the changing itself. Nothing is
written and no code is run until the approval gate has said yes, so a run
with nobody watching changes nothing at all: the write is declined, the
agent is told it was declined, and the run ends with a plan the operator can
read and apply themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.models import Model

from localsec.agents.patching.agents import PatchAgent, TriageAgent, VerifyAgent
from localsec.agents.patching.types import Findings, PatchPlan, Verification
from localsec.agents.patching.workspace import RepoDeps, mutating_tools
from localsec.runtime.approvals import ApprovalGate
from localsec.runtime.audit import AuditLog
from localsec.runtime.tools import ToolRegistry

__all__ = ["PatchingRun", "RunOutcome"]

_DECLINED = "DECLINED:"
_ERROR = "ERROR"


@dataclass
class RunOutcome:
    """What one patching run did."""

    findings: Findings
    plan: PatchPlan
    applied: list[str] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    verification: Verification | None = None
    rounds: int = 0

    @property
    def passed(self) -> bool:
        """Return whether the run finished the work, and the work was verified.

        A run with nothing to fix passes. A run whose changes were all
        declined does not: the verification may well say the tests are green,
        but the fix was never made.
        """
        if self.verification is None or not self.verification.passed:
            return False
        if self.findings.findings:
            return bool(self.applied)
        return True


class PatchingRun:
    """Runs the patching agents over one workspace."""

    def __init__(
        self,
        model: Model,
        deps: RepoDeps,
        *,
        gate: ApprovalGate | None = None,
        audit: AuditLog | None = None,
        max_rounds: int = 2,
    ) -> None:
        """Set up the agents, the tools, and the gate."""
        self._deps = deps
        self._triage = TriageAgent(model, deps)
        self._patch = PatchAgent(model, deps)
        self._verify = VerifyAgent(model, deps)
        self._registry = ToolRegistry(gate)
        self._registry.extend(mutating_tools(deps))
        self._audit = audit
        self._max_rounds = max_rounds

    def _record(self, event: str, **fields: object) -> None:
        """Write one entry to the audit log, when there is one."""
        if self._audit is not None:
            self._audit.record(event, **fields)

    async def run(self, description: str) -> RunOutcome:
        """Triage, plan, apply, and verify -- as far as the gate allows."""
        self._record("triage_started", description=description)
        findings = await self._triage.run(description)
        self._record("triage_completed", findings=len(findings.findings))

        plan = await self._patch.plan(findings, description)
        self._record("plan_created", edits=len(plan.edits))

        outcome = RunOutcome(findings=findings, plan=plan)
        feedback = ""
        for round_number in range(1, self._max_rounds + 1):
            outcome.rounds = round_number
            applied, declined, failed = self._apply(plan)
            outcome.applied.extend(applied)
            outcome.declined.extend(declined)
            outcome.failed.extend(failed)
            self._record(
                "round_applied",
                round=round_number,
                applied=applied,
                declined=declined,
                failed=failed,
            )

            test_output = self._run_tests(plan)
            self._record("tests_completed", round=round_number, output=test_output[:2000])
            verification = await self._verify.check(plan, test_output, round_number)
            outcome.verification = verification
            self._record(
                "verification_completed",
                round=round_number,
                passed=verification.passed,
                summary=verification.summary,
            )
            if verification.passed:
                return outcome

            feedback = verification.summary
            if round_number < self._max_rounds:
                plan = await self._patch.plan(findings, description, feedback)
                self._record("plan_revised", edits=len(plan.edits))

        return outcome

    def _apply(self, plan: PatchPlan) -> tuple[list[str], list[str], list[str]]:
        """Make the changes in a plan, asking the gate for each one."""
        applied: list[str] = []
        declined: list[str] = []
        failed: list[str] = []
        edit = self._registry.function("edit_file")
        write = self._registry.function("write_file")
        for change in plan.edits:
            if change.old:
                result = edit(path=change.path, old=change.old, new=change.new)
            else:
                result = write(path=change.path, content=change.new)
            if result.startswith(_DECLINED):
                declined.append(change.path)
            elif result.startswith(_ERROR):
                failed.append(change.path)
            else:
                applied.append(change.path)
        return applied, declined, failed

    def _run_tests(self, plan: PatchPlan) -> str:
        """Run the tests, asking the gate first.

        The command the run was configured with wins: the operator knows what
        the container holds, and the agent is guessing.
        """
        run_tests = self._registry.function("run_tests")
        command = self._deps.test_command.strip() or plan.test_command
        return str(run_tests(command=command))
