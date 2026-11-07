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
│   ├── splash.py          # client for the Splash inference server
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

## Dependencies

The project currently depends only on `httpx`, used by the Splash client.
Add whatever else the agent runtime needs as it gets designed:

```bash
uv add <package>        # runtime dependency
uv add --dev <package>  # development dependency
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
