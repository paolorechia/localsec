"""Running code in a container.

The patching agent runs tests, builds, and proof-of-concept code, which is
code it wrote and cannot be trusted to behave. It runs in a throwaway
container instead: the workspace is copied in, the container is thrown away,
and the host is never touched.

The workspace is copied rather than bind-mounted because a bind mount on
macOS goes through a filesystem shim and is slow enough to make a test run
feel broken.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DockerSandbox", "SandboxResult"]

_IGNORED = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache")


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """What a command did inside the container."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float

    @property
    def ok(self) -> bool:
        """Return whether the command succeeded."""
        return self.exit_code == 0


class DockerSandbox:
    """Runs commands in a throwaway container."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        *,
        docker: str = "docker",
        timeout: float = 600.0,
        network: str | None = None,
    ) -> None:
        """Configure the container runtime."""
        self.image = image
        self.docker = docker
        self.timeout = timeout
        self.network = network

    def available(self) -> bool:
        """Return whether a container runtime is on the machine."""
        return shutil.which(self.docker) is not None

    def command(self, script: str, *, workspace: str) -> list[str]:
        """Return the command that runs `script` in the container."""
        args = [self.docker, "run", "--rm", "-v", f"{workspace}:/work", "-w", "/work"]
        if self.network is not None:
            args += ["--network", self.network]
        args += [self.image, "sh", "-c", script]
        return args

    def run(self, script: str, *, workspace: Path) -> SandboxResult:
        """Copy the workspace into a temporary directory and run `script` in it."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "work"
            shutil.copytree(workspace, root, symlinks=True, ignore=_IGNORED)
            args = self.command(script, workspace=str(root))
            command = " ".join(args)
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    command=command,
                    exit_code=124,
                    stdout="",
                    stderr=f"the command timed out after {self.timeout:g}s",
                    duration=time.monotonic() - started,
                )
            return SandboxResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration=time.monotonic() - started,
            )
