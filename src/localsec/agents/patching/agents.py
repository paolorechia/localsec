"""The three patching agents.

Each has one job: work out where the vulnerability is, write the fix, and
check that the fix holds. They share the model and the workspace, and each
returns a typed result rather than prose.
"""

from __future__ import annotations

from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models import Model

from localsec.agents.patching.types import Findings, PatchPlan, Verification
from localsec.agents.patching.workspace import RepoDeps, read_only_tools

__all__ = ["PatchAgent", "TriageAgent", "VerifyAgent"]

TRIAGE_INSTRUCTIONS = """\
You are a security analyst. You are given a report of a suspected vulnerability
and the source tree it lives in.

Use the tools to look at the code before you answer. Find the file and the
line that the vulnerability is in, and say what an attacker could do with it.

Report every vulnerability you find. If the tree does not hold the vulnerable
code, report no findings and explain why in the notes.\
"""

PATCH_INSTRUCTIONS = """\
You are a security engineer. You are given a set of confirmed vulnerabilities
and the source tree they live in.

Use the tools to read the code you are about to change. Then write the
smallest change that removes the vulnerability: fix the check that failed
rather than rewriting everything around it.

For each change give the exact text to look for and the exact text to put in
its place. The text you look for has to appear exactly once in the file. Do
not propose changes to code that is not part of the vulnerability.

Give the command that runs the project's tests. The tests run in a throwaway
container that holds almost nothing, so the command has to install whatever
the tests need before it runs them.\
"""

VERIFY_INSTRUCTIONS = """\
You are a verification engineer. You are given a patch that was applied and the
output of the test suite.

Decide whether the fix holds. The tests have to pass, and the change has to
address the vulnerability itself rather than only making the tests quiet.

If the tests fail, say what failed and what it means. If something is still
wrong, list it. Do not change any code -- report what you find.\
"""


def _render_findings(findings: Findings) -> str:
    """Render the findings for the prompt."""
    if not findings.findings:
        return f"No findings were reported. Notes: {findings.notes}"
    lines = []
    for finding in findings.findings:
        where = finding.file + (f":{finding.line}" if finding.line else "")
        lines.append(f"- [{finding.severity}] {finding.title} ({where})")
        if finding.cve:
            lines.append(f"  {finding.cve}")
        if finding.description:
            lines.append(f"  {finding.description}")
    return "\n".join(lines)


class TriageAgent:
    """Works out where the vulnerability is."""

    def __init__(self, model: Model, deps: RepoDeps) -> None:
        """Build the agent around a model and a workspace."""
        self._agent: Agent[RepoDeps, Findings] = Agent(
            model,
            deps_type=RepoDeps,
            tools=read_only_tools(deps),
            output_type=ToolOutput(Findings),
            instructions=TRIAGE_INSTRUCTIONS,
            retries=2,
        )
        self._deps = deps

    async def run(self, description: str) -> Findings:
        """Look at the code and report the vulnerabilities."""
        result = await self._agent.run(description, deps=self._deps)
        return result.output


class PatchAgent:
    """Writes the fix."""

    def __init__(self, model: Model, deps: RepoDeps) -> None:
        """Build the agent around a model and a workspace."""
        self._agent: Agent[RepoDeps, PatchPlan] = Agent(
            model,
            deps_type=RepoDeps,
            tools=read_only_tools(deps),
            output_type=ToolOutput(PatchPlan),
            instructions=PATCH_INSTRUCTIONS,
            retries=2,
        )
        self._deps = deps

    async def plan(self, findings: Findings, description: str, feedback: str = "") -> PatchPlan:
        """Write the plan that fixes the findings."""
        parts = [
            "The report of the suspected vulnerability:",
            description,
            "",
            "The findings:",
            _render_findings(findings),
        ]
        sandbox = self._deps.sandbox
        if sandbox is not None:
            parts += [
                "",
                f"The tests run in a throwaway {sandbox.image} container.",
            ]
        if feedback:
            parts += [
                "",
                "The previous attempt did not hold up. Here is what went wrong:",
                feedback,
            ]
        result = await self._agent.run("\n".join(parts), deps=self._deps)
        return result.output


class VerifyAgent:
    """Checks that the fix holds."""

    def __init__(self, model: Model, deps: RepoDeps) -> None:
        """Build the agent around a model and a workspace."""
        self._agent: Agent[RepoDeps, Verification] = Agent(
            model,
            deps_type=RepoDeps,
            tools=read_only_tools(deps),
            output_type=ToolOutput(Verification),
            instructions=VERIFY_INSTRUCTIONS,
            retries=2,
        )
        self._deps = deps

    async def check(self, plan: PatchPlan, test_output: str, round_number: int) -> Verification:
        """Judge the fix against the test output."""
        parts = [
            f"This is round {round_number} of the patching run.",
            "",
            "The patch that was applied:",
            plan.summary or "(no summary)",
            *[f"- {edit.path}: {edit.rationale or 'change'}" for edit in plan.edits],
            "",
            "The output of the test suite:",
            test_output,
        ]
        result = await self._agent.run("\n".join(parts), deps=self._deps)
        return result.output
