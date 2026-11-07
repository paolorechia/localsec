"""Client for the Splash local inference server.

Splash serves an OpenAI-compatible HTTP API. This module is a thin relay: a
prompt goes in, the model's text comes out. It covers ``POST
/v1/chat/completions`` for chat, ``GET /v1/models`` for the resident model,
and ``GET /health`` for readiness.

What the module adds over a plain OpenAI client is the part of Splash that
is Splash's own: the reasoning trace the resident model writes and the
controls that govern it, the resident model itself (the ``model`` field is
left out of requests unless one is given, which asks the server for whichever
GGUF it currently holds), and the fact that the server refuses connections
rather than queueing them once its slots fill.

Tool calling is not modelled here. Splash implements the OpenAI tool schema
-- including ``strict`` tools, whose arguments are generated from the schema
by a grammar -- so a request may carry tools through ``extra`` and the
response's ``tool_calls`` are reachable on ``Reply.raw``, but interpreting
them belongs to whatever drives the agent.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, Final

import httpx2

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "REASONING_EFFORTS",
    "AsyncSplashClient",
    "Reply",
    "SplashAPIError",
    "SplashClient",
    "SplashConnectionError",
    "SplashError",
]

DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT: Final[float] = 600.0
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_BACKOFF: Final[float] = 0.25

#: The reasoning efforts Splash accepts, from no reasoning at all to the most.
REASONING_EFFORTS: Final[tuple[str, ...]] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

_CHAT_COMPLETIONS_PATH: Final[str] = "/v1/chat/completions"
_MODELS_PATH: Final[str] = "/v1/models"
_HEALTH_PATH: Final[str] = "/health"
_DONE_SENTINEL: Final[str] = "[DONE]"


class SplashError(Exception):
    """Base class for errors raised by this module."""


class SplashConnectionError(SplashError):
    """The server could not be reached."""


class SplashAPIError(SplashError):
    """The server answered with an error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type: str | None = None,
        code: str | None = None,
        body: str = "",
    ) -> None:
        """Record the server's error."""
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.type = type
        self.code = code
        self.body = body


class Reply(str):
    """The model's answer, with Splash's extras attached.

    A reply *is* the answer text: it compares, formats, and concatenates as
    the string it is. The fields Splash adds hang off it -- the reasoning
    trace, the token counts, the server's own ``metrics`` and ``timings``
    (time to first token, tokens per second, the speculative decoding
    counters, the cache lane), and the payload exactly as it arrived, which
    is where anything this module does not interpret (such as
    ``tool_calls``) stays reachable.
    """

    __slots__ = ("metrics", "raw", "reasoning", "timings", "usage")

    raw: Mapping[str, Any]
    reasoning: str
    usage: Mapping[str, Any] | None
    metrics: Mapping[str, Any]
    timings: Mapping[str, Any]

    def __new__(
        cls,
        text: str,
        *,
        raw: Mapping[str, Any] | None = None,
        reasoning: str = "",
        usage: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        timings: Mapping[str, Any] | None = None,
    ) -> Reply:
        """Build a reply from the answer text and the server's payload."""
        reply = super().__new__(cls, text)
        reply.raw = raw if raw is not None else {}
        reply.reasoning = reasoning
        reply.usage = usage
        reply.metrics = metrics if metrics is not None else {}
        reply.timings = timings if timings is not None else {}
        return reply


def _build_body(
    prompt: str,
    *,
    system: str | None = None,
    history: Sequence[Mapping[str, Any]] = (),
    model: str | None = None,
    stream: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    stop: str | Sequence[str] | None = None,
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    preserve_thinking: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body of a chat completion request."""
    messages: list[dict[str, Any]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.extend(dict(message) for message in history)
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {"messages": messages}
    if model:
        body["model"] = model
    if stream:
        body["stream"] = True
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if stop is not None:
        body["stop"] = stop
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if thinking_budget is not None:
        body["thinking_token_budget"] = thinking_budget
    if preserve_thinking is not None:
        body["preserve_thinking"] = preserve_thinking
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    if extra:
        body.update(extra)
    return body


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _decode_json(response: httpx2.Response) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _reply_from_payload(payload: Mapping[str, Any]) -> Reply:
    """Build a reply from a completion payload."""
    choices = payload.get("choices")
    choice = _as_mapping(choices[0]) if isinstance(choices, list) and choices else {}
    message = _as_mapping(choice.get("message"))

    content = message.get("content")
    reasoning = message.get("reasoning_content")
    usage = payload.get("usage")
    metrics = payload.get("metrics")
    timings = payload.get("timings")
    return Reply(
        content if isinstance(content, str) else "",
        raw=payload,
        reasoning=reasoning if isinstance(reasoning, str) else "",
        usage=usage if isinstance(usage, Mapping) else None,
        metrics=metrics if isinstance(metrics, Mapping) else {},
        timings=timings if isinstance(timings, Mapping) else {},
    )


def _delta_text(payload: Mapping[str, Any]) -> str:
    """Return the answer text a streamed chunk carries."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = _as_mapping(_as_mapping(choices[0]).get("delta"))
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _parse_sse_line(line: str) -> str | None:
    """Parse one line of an event stream, if it carries answer text."""
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or not stripped.startswith("data:"):
        return None
    data = stripped[len("data:") :].strip()
    if not data or data == _DONE_SENTINEL:
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return _delta_text(payload)


def _raise_for_status(response: httpx2.Response) -> None:
    if response.is_success:
        return
    payload = _decode_json(response)
    error = _as_mapping(payload.get("error"))
    message = error.get("message")
    raise SplashAPIError(
        response.status_code,
        message if isinstance(message, str) else response.text.strip(),
        type=error.get("type") if isinstance(error.get("type"), str) else None,
        code=error.get("code") if isinstance(error.get("code"), str) else None,
        body=response.text,
    )


class _SplashTransport:
    """The parts of the client that do not depend on sync or async."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ) -> None:
        """Record where the server is and how to authenticate to it."""
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def url(self, path: str) -> str:
        """Return the absolute URL of a server path."""
        return f"{self.base_url}{path}"

    def headers(self) -> dict[str, str]:
        """Return the headers every request carries."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _backoff(self, attempt: int) -> float:
        """Return how long to wait before retrying."""
        return self._retry_backoff * float(2**attempt)


class SplashClient(_SplashTransport):
    """A synchronous client for the Splash API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        http_client: httpx2.Client | None = None,
    ) -> None:
        """Create a client, reusing ``http_client`` when one is given."""
        super().__init__(
            base_url,
            api_key=api_key,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(timeout=timeout, trust_env=False)

    def close(self) -> None:
        """Close the underlying HTTP client, if this client owns it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SplashClient:
        """Enter a context that closes the client on exit."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close the client."""
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> httpx2.Response:
        """Send a request, retrying while the server refuses connections.

        Splash refuses a connection rather than queueing it once its
        connection slots are full, so a refused connection says nothing about
        the server being down and is worth retrying.
        """
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    self.url(path),
                    json=body,
                    headers=self.headers(),
                )
            except httpx2.ConnectError as exc:
                if attempt < self._max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
                raise SplashConnectionError(
                    f"could not reach the Splash server at {self.base_url}: {exc}"
                ) from exc
            except httpx2.HTTPError as exc:
                raise SplashConnectionError(
                    f"could not reach the Splash server at {self.base_url}: {exc}"
                ) from exc
            _raise_for_status(response)
            return response
        raise SplashConnectionError(  # pragma: no cover - the loop always returns
            f"could not reach the Splash server at {self.base_url}"
        )

    def health(self) -> bool:
        """Return whether the server is up and serving."""
        try:
            response = self._client.get(self.url(_HEALTH_PATH), headers=self.headers())
        except httpx2.HTTPError:
            return False
        return response.status_code == 200

    def wait_until_ready(self, timeout: float = 60.0, interval: float = 0.5) -> bool:
        """Wait until the server answers, or the timeout passes.

        Returns whether the server became ready.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.health():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    def models(self) -> list[str]:
        """Return the ids of the models the server can answer for."""
        payload = _decode_json(self._request("GET", _MODELS_PATH))
        data = payload.get("data")
        entries = data if isinstance(data, list) else []
        return [str(item["id"]) for item in entries if isinstance(item, Mapping) and "id" in item]

    def ask(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: Sequence[Mapping[str, Any]] = (),
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        preserve_thinking: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Reply:
        """Send a prompt and return the model's answer."""
        body = _build_body(
            prompt,
            system=system,
            history=history,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            preserve_thinking=preserve_thinking,
            extra=extra,
        )
        payload = _decode_json(self._request("POST", _CHAT_COMPLETIONS_PATH, body))
        return _reply_from_payload(payload)

    def ask_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: Sequence[Mapping[str, Any]] = (),
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        preserve_thinking: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream the model's answer, yielding its text as it arrives.

        The reasoning trace is not part of the stream: it is written before
        the answer, and the answer is what this yields.
        """
        body = _build_body(
            prompt,
            system=system,
            history=history,
            model=model,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            preserve_thinking=preserve_thinking,
            extra=extra,
        )
        for attempt in range(self._max_retries + 1):
            started = False
            try:
                with self._client.stream(
                    "POST",
                    self.url(_CHAT_COMPLETIONS_PATH),
                    json=body,
                    headers=self.headers(),
                ) as response:
                    if not response.is_success:
                        response.read()
                    _raise_for_status(response)
                    for line in response.iter_lines():
                        text = _parse_sse_line(line)
                        if text:
                            started = True
                            yield text
                return
            except httpx2.ConnectError as exc:
                if started or attempt >= self._max_retries:
                    raise SplashConnectionError(
                        f"could not reach the Splash server at {self.base_url}: {exc}"
                    ) from exc
                time.sleep(self._backoff(attempt))
            except httpx2.HTTPError as exc:
                raise SplashConnectionError(
                    f"could not reach the Splash server at {self.base_url}: {exc}"
                ) from exc


class AsyncSplashClient(_SplashTransport):
    """An asynchronous client for the Splash API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create a client, reusing ``http_client`` when one is given."""
        super().__init__(
            base_url,
            api_key=api_key,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self._owns_client = http_client is None
        self._client = http_client or httpx2.AsyncClient(timeout=timeout, trust_env=False)

    async def aclose(self) -> None:
        """Close the underlying HTTP client, if this client owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncSplashClient:
        """Enter a context that closes the client on exit."""
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        """Close the client."""
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> httpx2.Response:
        """Send a request, retrying while the server refuses connections."""
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    self.url(path),
                    json=body,
                    headers=self.headers(),
                )
            except httpx2.ConnectError as exc:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise SplashConnectionError(
                    f"could not reach the Splash server at {self.base_url}: {exc}"
                ) from exc
            except httpx2.HTTPError as exc:
                raise SplashConnectionError(
                    f"could not reach the Splash server at {self.base_url}: {exc}"
                ) from exc
            _raise_for_status(response)
            return response
        raise SplashConnectionError(  # pragma: no cover - the loop always returns
            f"could not reach the Splash server at {self.base_url}"
        )

    async def health(self) -> bool:
        """Return whether the server is up and serving."""
        try:
            response = await self._client.get(self.url(_HEALTH_PATH), headers=self.headers())
        except httpx2.HTTPError:
            return False
        return response.status_code == 200

    async def wait_until_ready(self, timeout: float = 60.0, interval: float = 0.5) -> bool:
        """Wait until the server answers, or the timeout passes.

        Returns whether the server became ready.
        """
        deadline = time.monotonic() + timeout
        while True:
            if await self.health():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(interval)

    async def models(self) -> list[str]:
        """Return the ids of the models the server can answer for."""
        payload = _decode_json(await self._request("GET", _MODELS_PATH))
        data = payload.get("data")
        entries = data if isinstance(data, list) else []
        return [str(item["id"]) for item in entries if isinstance(item, Mapping) and "id" in item]

    async def ask(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: Sequence[Mapping[str, Any]] = (),
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        preserve_thinking: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Reply:
        """Send a prompt and return the model's answer."""
        body = _build_body(
            prompt,
            system=system,
            history=history,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            preserve_thinking=preserve_thinking,
            extra=extra,
        )
        payload = _decode_json(await self._request("POST", _CHAT_COMPLETIONS_PATH, body))
        return _reply_from_payload(payload)

    async def ask_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: Sequence[Mapping[str, Any]] = (),
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        stop: str | Sequence[str] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        preserve_thinking: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the model's answer, yielding its text as it arrives.

        The reasoning trace is not part of the stream: it is written before
        the answer, and the answer is what this yields.
        """
        body = _build_body(
            prompt,
            system=system,
            history=history,
            model=model,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            preserve_thinking=preserve_thinking,
            extra=extra,
        )
        for attempt in range(self._max_retries + 1):
            started = False
            try:
                async with self._client.stream(
                    "POST",
                    self.url(_CHAT_COMPLETIONS_PATH),
                    json=body,
                    headers=self.headers(),
                ) as response:
                    if not response.is_success:
                        await response.aread()
                    _raise_for_status(response)
                    async for line in response.aiter_lines():
                        text = _parse_sse_line(line)
                        if text:
                            started = True
                            yield text
                return
            except httpx2.ConnectError as exc:
                if started or attempt >= self._max_retries:
                    raise SplashConnectionError(
                        f"could not reach the Splash server at {self.base_url}: {exc}"
                    ) from exc
                await asyncio.sleep(self._backoff(attempt))
            except httpx2.HTTPError as exc:
                raise SplashConnectionError(
                    f"could not reach the Splash server at {self.base_url}: {exc}"
                ) from exc
