"""The tree the agents work in, and the tools that reach it.

Every path an agent names is resolved inside the workspace, and anything that
points outside is refused. The agents cannot read or write the rest of the
machine, whatever they are told to do.

The tools come in two sets. The read-only ones -- listing, reading, searching
-- are handed to the agents so they can look at the code. The mutating ones
-- writing, editing, running the tests -- are handed to the run, which asks
the approval gate before each one.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localsec.runtime.approvals import ApprovalPolicy
from localsec.runtime.sandbox import DockerSandbox
from localsec.runtime.tools import tool

__all__ = ["RepoDeps", "Workspace", "WorkspaceError", "mutating_tools", "read_only_tools"]

_HIDDEN = "."
_MAX_FILE_BYTES = 200_000
_MAX_RESULTS = 200


class WorkspaceError(Exception):
    """The agent asked for something outside the workspace."""


class Workspace:
    """A directory the agents may read, write, and search."""

    def __init__(self, root: Path, *, max_bytes: int = _MAX_FILE_BYTES) -> None:
        """Open a workspace. The root has to be a directory that exists."""
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"{self.root} is not a directory")
        self.max_bytes = max_bytes

    def resolve(self, path: str) -> Path:
        """Resolve a path the agent named, refusing to leave the workspace."""
        if not path.strip():
            raise WorkspaceError("the path is empty")
        candidate = Path(path)
        if candidate.is_absolute():
            raise WorkspaceError(f"{path} is absolute; paths are relative to the workspace")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise WorkspaceError(f"{path} points outside the workspace")
        return resolved

    def relative(self, path: Path) -> str:
        """Return a path as the agent should see it."""
        return str(path.relative_to(self.root))

    def files(self) -> list[Path]:
        """Return every file in the workspace that the agents may see."""
        found: list[Path] = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if any(part.startswith(_HIDDEN) for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            found.append(path)
        return found

    def listing(self, pattern: str = "*") -> list[str]:
        """Return the files whose path matches a pattern."""
        matches = [
            self.relative(path)
            for path in self.files()
            if fnmatch.fnmatch(self.relative(path), pattern) or fnmatch.fnmatch(path.name, pattern)
        ]
        return matches[:_MAX_RESULTS]

    def read(self, path: str) -> str:
        """Return the contents of a file."""
        target = self.resolve(path)
        if target.is_dir():
            raise WorkspaceError(f"{path} is a directory")
        if not target.is_file():
            raise WorkspaceError(f"{path} does not exist")
        data = target.read_text(encoding="utf-8", errors="replace")
        if len(data) > self.max_bytes:
            return data[: self.max_bytes] + f"\n... truncated at {self.max_bytes} bytes"
        return data

    def search(self, query: str) -> list[str]:
        """Return the lines that contain a piece of text, wherever it is."""
        if not query.strip():
            raise WorkspaceError("the search text is empty")
        needle = query.lower()
        hits: list[str] = []
        for path in self.files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    hits.append(f"{self.relative(path)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= _MAX_RESULTS:
                        return hits
        return hits

    def write(self, path: str, content: str) -> str:
        """Write a file, whole."""
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {path}"

    def edit(self, path: str, old: str, new: str) -> str:
        """Replace one piece of text in a file.

        The text to replace has to appear exactly once, so a change can never
        land in the wrong place by accident.
        """
        if not old:
            raise WorkspaceError("there is no text to replace")
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"{path} does not exist")
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise WorkspaceError(f"{path} does not contain the text to replace")
        if count > 1:
            raise WorkspaceError(f"{path} contains that text {count} times; it has to be unique")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"replaced one block in {path}"


@dataclass
class RepoDeps:
    """What the agents need to do their work."""

    workspace: Workspace
    sandbox: DockerSandbox | None = None
    test_command: str = ""
    """How to run the tests.

    The operator knows what the container holds, so a command set here wins
    over the one the patch agent proposes. Leave it empty to let the agent
    decide.
    """


def read_only_tools(deps: RepoDeps) -> list[Callable[..., Any]]:
    """Return the tools that let an agent look at the code."""

    @tool(policy=ApprovalPolicy.READ_ONLY, action="list the files in the workspace")
    def list_files(pattern: str = "*") -> str:
        """List the files in the workspace. Use '*' for all of them."""
        files = deps.workspace.listing(pattern)
        return "\n".join(files) if files else "(no files matched)"

    @tool(policy=ApprovalPolicy.READ_ONLY, action="read a file")
    def read_file(path: str) -> str:
        """Read a file from the workspace."""
        try:
            return deps.workspace.read(path)
        except WorkspaceError as error:
            return f"ERROR: {error}"

    @tool(policy=ApprovalPolicy.READ_ONLY, action="search the code")
    def search_code(query: str) -> str:
        """Search the workspace for a piece of text. It shows the file and line."""
        try:
            hits = deps.workspace.search(query)
        except WorkspaceError as error:
            return f"ERROR: {error}"
        return "\n".join(hits) if hits else "(no matches)"

    return [list_files, read_file, search_code]


def mutating_tools(deps: RepoDeps) -> list[Callable[..., Any]]:
    """Return the tools that change the workspace, or run code in it."""

    @tool(policy=ApprovalPolicy.MUTATING, action="write a file")
    def write_file(path: str, content: str) -> str:
        """Write a file in the workspace, replacing whatever was there."""
        try:
            return deps.workspace.write(path, content)
        except WorkspaceError as error:
            return f"ERROR: {error}"

    @tool(policy=ApprovalPolicy.MUTATING, action="edit a file")
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one piece of text in a file. The text has to appear once."""
        try:
            return deps.workspace.edit(path, old, new)
        except WorkspaceError as error:
            return f"ERROR: {error}"

    @tool(policy=ApprovalPolicy.MUTATING, action="run the tests in the sandbox")
    def run_tests(command: str = "") -> str:
        """Run the project's tests in a throwaway container and show the output.

        Leave the command empty to run the one the project is configured with.
        """
        sandbox = deps.sandbox
        if sandbox is None:
            return "ERROR: no sandbox is configured, so the tests cannot be run"
        script = command.strip() or deps.test_command
        result = sandbox.run(script, workspace=deps.workspace.root)
        parts = [
            f"command: {script}",
            f"exit code: {result.exit_code}",
            "--- stdout ---",
            result.stdout.strip() or "(nothing)",
            "--- stderr ---",
            result.stderr.strip() or "(nothing)",
        ]
        return "\n".join(parts)

    return [write_file, edit_file, run_tests]
