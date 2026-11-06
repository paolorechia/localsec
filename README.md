# localsec

An agentic system.

This repository is the base Python structure for the project: a uv-managed
package with a `src/` layout, a test suite, and the standard linting, type
checking, and CI tooling. There is no domain logic here yet — the agent
runtime, its dependencies, and its interfaces are still to be designed.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-groups
```

This creates a virtual environment in `.venv` and installs the project in
editable mode along with the development tools.

## Layout

```
.
├── src/localsec/          # package source
│   ├── __init__.py        # package metadata and public exports
│   ├── __main__.py        # `python -m localsec`
│   ├── cli.py             # `localsec` console entry point
│   └── py.typed           # marks the package as typed (PEP 561)
└── tests/                 # test suite
```

The package uses a `src/` layout, so tests import the installed package rather
than the working directory.

## Commands

```bash
uv run localsec --version   # run the CLI
uv run pytest                # run the tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src tests        # type check
```

Pre-commit hooks are configured in `.pre-commit-config.yaml` and cover the same
checks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

CI (`.github/workflows/ci.yml`) runs lint, format check, type check, and tests
on every push and pull request.

## Dependencies

The project currently has **no runtime dependencies** — only the development
group (ruff, pytest, pytest-cov, mypy, pre-commit). Add whatever the agent
runtime needs as it gets designed:

```bash
uv add <package>        # runtime dependency
uv add --dev <package>  # development dependency
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
