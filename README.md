# localsec

An agentic system.

This repository is a uv-managed package with a `src/` layout, a test suite, and
the standard linting, type checking, and CI tooling. It holds the agent
runtime the agents share, the client for the local inference server, and the
security patching agents.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Splash](https://formulae.brew/sh/splash) — the local inference server
- Docker — the tests the agents write run in a container

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
│   ├── splash.py          # client for the Splash inference server
│   ├── runtime/           # what the agents share
│   │   ├── provider.py    # the model, and the resident model lookup
│   │   ├── approvals.py   # the gate every change goes through
│   │   ├── audit.py       # the record of what was done
│   │   ├── tools.py       # tools, and the policy that says which change things
│   │   └── sandbox.py     # running code in a throwaway container
│   ├── agents/            # the agents themselves
│   │   └── patching/      # triage, plan, apply, verify
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

## Splash client

`localsec.splash` is a client for the [Splash](https://formulae.brew/sh/splash)
local inference server, which serves an OpenAI-compatible HTTP API on
`http://127.0.0.1:8000`. It is a relay: a prompt goes in, the model's text
comes out.

```python
from localsec.splash import SplashClient

client = SplashClient()  # http://127.0.0.1:8000

answer = client.ask("Say hello")
print(answer)
```

The answer is a `Reply`, which *is* the answer text — it compares and formats
as the string it is. Splash's extras hang off it:

```python
reply = client.ask("What is 2+2? Answer with just the number.")

reply.reasoning  # the model's thinking trace
reply.usage  # token counts
reply.metrics  # ttft, tokens/second, cache lane, speculative decoding
reply.timings  # the same, timed
reply.raw  # the payload exactly as it arrived
```

The client omits the `model` field unless one is given, which asks the server
for whichever model it currently holds resident — so it keeps working when the
resident model changes.

**Streaming.** `ask_stream` yields the answer's text as it arrives:

```python
for piece in client.ask_stream("Count to ten"):
    print(piece, end="", flush=True)
```

**Reasoning.** The resident model is a reasoning model, and Splash lets the
thinking be governed:

```python
client.ask("...", thinking=False)  # no thinking at all
client.ask("...", reasoning_effort="high")  # none, minimal, low, medium, high, xhigh, max
client.ask("...", thinking_budget=512)  # cap the thinking tokens
client.ask("...", preserve_thinking=True)  # keep the trace in the assistant message
```

`thinking` is Splash's `chat_template_kwargs.enable_thinking`; it outranks
`reasoning_effort`, which is read as "off" when it is `"none"`.

**Conversations.** `system` leads, `history` carries the earlier turns, and
the prompt is the last message:

```python
client.ask(
    "and now?",
    system="be brief",
    history=[{"role": "assistant", "content": "earlier"}],
)
```

**Tool calls.** Not modelled here. Splash implements the OpenAI tool schema —
including `strict` tools, whose arguments are generated from their schema by a
grammar — so a request may carry tools through `extra`, and the calls come
back on `reply.raw`:

```python
reply = client.ask(
    "Weather in Paris?",
    extra={"tools": [{"type": "function", "function": {"name": "get_weather", ...}}]},
)
```

Interpreting them belongs to whatever drives the agent.

**Async.** `AsyncSplashClient` mirrors the same methods for `async` code:

```python
from localsec.splash import AsyncSplashClient

async with AsyncSplashClient() as client:
    answer = await client.ask("Say hello")
```

**Splash specifics.** The server refuses connections rather than queueing them
once its connection slots are full, so the client retries a refused connection
with a short backoff. `client.wait_until_ready()` polls `/health` until the
server answers, which is useful at agent startup.

The server also serves `/v1/messages` (Anthropic's schema) and `/v1/responses`;
this module covers the OpenAI-compatible chat endpoints only.

## The agent runtime

`localsec.runtime` is what the agents share: the model, the tools, and the
gate that every change has to pass.

**The model.** `ModelSession` looks up whichever model the server holds
resident and hands back a Pydantic AI model bound to it:

```python
from localsec.runtime.provider import ModelSession

async with ModelSession() as session:
    agent = Agent(session.model, instructions="...")
```

The client is built with `trust_env=False`, so a proxy in the environment
never redirects a request meant for the local server.

**The gate.** Every tool is either read-only or mutating. A mutating tool
asks the gate first, and **a gate with no approver declines** — an unattended
agent cannot change anything by accident. A declined tool returns `DECLINED:`
to the model rather than failing the run, so the agent can try something
else:

```python
from localsec.runtime.approvals import ApprovalGate, console_approver

gate = ApprovalGate(console_approver)  # ask on the terminal
gate = ApprovalGate()  # decline everything
```

**The audit log.** One JSON object per line, each with a UTC timestamp:

```python
from localsec.runtime.audit import AuditLog

audit = AuditLog("audit.jsonl")
audit.record("triage_started", description="...")
```

**The sandbox.** `DockerSandbox` runs a command in a throwaway container with
a copy of the workspace. The copy rather than a bind mount is deliberate: a
bind mount on macOS goes through a filesystem shim and is slow enough to make
a test run feel broken.

## Security patching

`localsec.agents.patching` has three agents and a run that drives them:

| Agent | Job | Result |
| --- | --- | --- |
| `TriageAgent` | finds the vulnerable code | `Findings` |
| `PatchAgent` | writes the fix | `PatchPlan` |
| `VerifyAgent` | checks the fix holds | `Verification` |

The run goes: triage → plan → apply → verify, and when the verification
fails it asks the patch agent for a revision and goes round again, up to
`max_rounds`.

The agents are read-only. The run does the changing, so every write goes
through the approval gate with the exact text of the change in front of the
operator.

```bash
uv run localsec patch "the archive extraction helper writes outside the destination" \
  --workspace ./my-project \
  --test-command "pip install --quiet pytest && pytest -q" \
  --audit audit.jsonl
```

Without `--yes` the operator is asked before each change. With it, every
change is approved as it happens — which is the only way to run unattended,
and the reason the gate declines by default.

The agents can only touch the workspace: every path they name is resolved
inside it, and anything that points outside — `../`, an absolute path, a
symlink — is refused.

**Structured output.** The agents fill in typed results rather than prose.
The server compiles the schema into a grammar, so the fields come back as
they were asked for:

```python
@dataclass
class Finding:
    title: str
    severity: str
    file: str
    line: int | None = None
    cve: str | None = None
    description: str = ""
```

**The test command.** The operator's command wins over the one the patch
agent proposes, because the operator knows what the container holds and the
agent is guessing. The container is bare, so the command usually has to
install what the tests need:

```bash
--test-command "pip install --quiet pytest && pytest -q"
```

## Dependencies

The project depends on `httpx2`, used by the Splash client, and
`pydantic-ai-slim[openai]`, which is the agent framework. The `slim` extra
keeps the tree to the OpenAI-compatible providers — the full `pydantic-ai`
package pulls in around a hundred packages that this project does not use.

```bash
uv add <package>        # runtime dependency
uv add --dev <package>  # development dependency
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
