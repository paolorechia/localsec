"""Tests for the Splash API client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from localsec.splash import (
    AsyncSplashClient,
    Reply,
    SplashAPIError,
    SplashClient,
    SplashConnectionError,
    _build_body,
)

BASE_URL = "http://127.0.0.1:8000"


def _completion_payload(**overrides: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 18, "completion_tokens": 3, "total_tokens": 21},
        "metrics": {"decode": {"tokens": 3}},
        "timings": {"predicted_per_second": 42.0},
    }
    payload.update(overrides)
    return payload


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = None,
) -> tuple[SplashClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return SplashClient(BASE_URL, api_key=api_key, http_client=http_client), http_client


def test_ask_sends_the_prompt_and_returns_the_answer() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        reply = client.ask("ping")
    finally:
        http_client.close()

    assert seen["url"] == f"{BASE_URL}/v1/chat/completions"
    assert seen["body"] == {"messages": [{"role": "user", "content": "ping"}]}
    assert reply == "pong"
    assert reply.usage is not None
    assert reply.usage["total_tokens"] == 21


def test_ask_sends_the_system_prompt_and_the_history() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        client.ask(
            "and now?",
            system="be brief",
            history=[{"role": "assistant", "content": "earlier"}],
        )
    finally:
        http_client.close()

    assert seen["body"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "and now?"},
    ]


def test_ask_sends_the_requested_options() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        client.ask(
            "ping",
            model="some-model",
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            max_tokens=64,
            stop=["\n\n"],
        )
    finally:
        http_client.close()

    body = seen["body"]
    assert body["model"] == "some-model"
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["top_k"] == 40
    assert body["max_tokens"] == 64
    assert body["stop"] == ["\n\n"]


def test_ask_sends_the_reasoning_controls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        client.ask(
            "ping",
            thinking=False,
            reasoning_effort="high",
            thinking_budget=512,
            preserve_thinking=True,
        )
    finally:
        http_client.close()

    body = seen["body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "high"
    assert body["thinking_token_budget"] == 512
    assert body["preserve_thinking"] is True


def test_ask_leaves_the_reasoning_controls_alone_by_default() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        client.ask("ping")
    finally:
        http_client.close()

    body = seen["body"]
    assert "chat_template_kwargs" not in body
    assert "reasoning_effort" not in body
    assert "thinking_token_budget" not in body
    assert "preserve_thinking" not in body


def test_reply_carries_the_reasoning_trace() -> None:
    payload = _completion_payload(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "4",
                    "reasoning_content": "2 + 2 is 4.",
                },
                "finish_reason": "stop",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        reply = client.ask("what is 2+2?")
    finally:
        http_client.close()

    assert reply == "4"
    assert reply.reasoning == "2 + 2 is 4."


def test_reply_carries_the_splash_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        reply = client.ask("ping")
    finally:
        http_client.close()

    assert reply.metrics == {"decode": {"tokens": 3}}
    assert reply.timings == {"predicted_per_second": 42.0}


def test_reply_keeps_the_payload_as_it_arrived() -> None:
    payload = _completion_payload(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client, http_client = _client(handler)
    try:
        reply = client.ask("weather?")
    finally:
        http_client.close()

    # Tool calls are not interpreted here, but they are not lost either.
    assert reply.raw == payload
    calls = reply.raw["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "get_weather"


def test_extra_reaches_the_request_body() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        client.ask("ping", extra={"tools": [{"type": "function"}], "cache_salt": "abc"})
    finally:
        http_client.close()

    assert seen["body"]["tools"] == [{"type": "function"}]
    assert seen["body"]["cache_salt"] == "abc"


def test_ask_raises_on_an_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": "model nope not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    client, http_client = _client(handler)
    try:
        with pytest.raises(SplashAPIError) as caught:
            client.ask("x", model="nope")
    finally:
        http_client.close()

    assert caught.value.status_code == 404
    assert caught.value.message == "model nope not found"
    assert caught.value.code == "model_not_found"
    assert caught.value.type == "invalid_request_error"


def test_connection_failure_raises_splash_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, http_client = _client(handler)
    try:
        with pytest.raises(SplashConnectionError):
            client.ask("x")
    finally:
        http_client.close()


def test_models_lists_the_resident_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL",
                        "object": "model",
                        "created": 0,
                        "owned_by": "splash",
                        "context_length": 229369,
                    }
                ],
            },
        )

    client, http_client = _client(handler)
    try:
        models = client.models()
    finally:
        http_client.close()

    assert models == ["unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL"]


def test_health_reports_the_server_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    client, http_client = _client(handler)
    try:
        assert client.health() is True
    finally:
        http_client.close()


def test_health_is_false_when_the_server_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, http_client = _client(handler)
    try:
        assert client.health() is False
    finally:
        http_client.close()


def test_ask_stream_yields_the_answer_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        events = [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "po"}}],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "ng"}}],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        payload = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        return httpx.Response(200, content=payload + b"data: [DONE]\n\n")

    client, http_client = _client(handler)
    try:
        chunks = list(client.ask_stream("ping"))
    finally:
        http_client.close()

    assert chunks == ["po", "ng"]


def test_ask_stream_leaves_the_reasoning_trace_out() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        events = [
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}],
            },
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "answer"}}],
            },
        ]
        payload = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        return httpx.Response(200, content=payload + b"data: [DONE]\n\n")

    client, http_client = _client(handler)
    try:
        chunks = list(client.ask_stream("ping"))
    finally:
        http_client.close()

    assert chunks == ["answer"]


def test_async_client_asks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_payload())

    async def run() -> Reply:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncSplashClient(BASE_URL, http_client=http_client)
        try:
            return await client.ask("ping")
        finally:
            await http_client.aclose()

    reply = asyncio.run(run())

    assert reply == "pong"
    assert reply.reasoning == ""


def test_async_client_streams() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
        }
        payload = b"data: " + json.dumps(event).encode() + b"\n\n" + b"data: [DONE]\n\n"
        return httpx.Response(200, content=payload)

    async def run() -> list[str]:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncSplashClient(BASE_URL, http_client=http_client)
        try:
            return [chunk async for chunk in client.ask_stream("ping")]
        finally:
            await http_client.aclose()

    assert asyncio.run(run()) == ["hi"]


def test_async_client_raises_on_an_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async def run() -> None:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncSplashClient(BASE_URL, http_client=http_client)
        try:
            await client.ask("x")
        finally:
            await http_client.aclose()

    with pytest.raises(SplashAPIError) as caught:
        asyncio.run(run())

    assert caught.value.status_code == 400
    assert caught.value.message == "bad request"


def test_async_client_lists_the_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    async def run() -> list[str]:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncSplashClient(BASE_URL, http_client=http_client)
        try:
            return await client.models()
        finally:
            await http_client.aclose()

    assert asyncio.run(run()) == ["m"]


def test_async_client_reports_a_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def run() -> None:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncSplashClient(BASE_URL, http_client=http_client)
        try:
            await client.ask("x")
        finally:
            await http_client.aclose()

    with pytest.raises(SplashConnectionError):
        asyncio.run(run())


def test_a_refused_connection_is_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler)
    try:
        reply = client.ask("ping")
    finally:
        http_client.close()

    # Splash refuses a connection rather than queueing it, so the first
    # refusal says nothing about the server being down.
    assert len(calls) == 2
    assert reply == "pong"


def test_a_refused_connection_is_retried_when_streaming() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        event = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
        }
        payload = b"data: " + json.dumps(event).encode() + b"\n\n" + b"data: [DONE]\n\n"
        return httpx.Response(200, content=payload)

    client, http_client = _client(handler)
    try:
        chunks = list(client.ask_stream("ping"))
    finally:
        http_client.close()

    assert len(calls) == 2
    assert chunks == ["hi"]


def test_a_read_timeout_is_a_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, http_client = _client(handler)
    try:
        with pytest.raises(SplashConnectionError):
            client.ask("x")
    finally:
        http_client.close()


def test_a_stream_that_fails_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, http_client = _client(handler)
    try:
        with pytest.raises(SplashConnectionError):
            list(client.ask_stream("x"))
    finally:
        http_client.close()


def test_the_async_client_closes_its_connection() -> None:
    async def run() -> None:
        client = AsyncSplashClient(BASE_URL)
        await client.aclose()

    asyncio.run(run())


def test_the_async_client_can_be_used_as_a_context_manager() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_payload())

    async def run() -> None:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with AsyncSplashClient(BASE_URL, http_client=http_client) as client:
            assert await client.ask("ping") == "pong"
        await http_client.aclose()

    asyncio.run(run())


def test_the_async_client_reports_a_server_that_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def run() -> bool:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncSplashClient(BASE_URL, http_client=http_client)
        try:
            assert await client.health() is False
            return await client.wait_until_ready(timeout=0.05, interval=0.01)
        finally:
            await http_client.aclose()

    assert asyncio.run(run()) is False


def test_an_async_stream_that_fails_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async def run() -> None:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncSplashClient(BASE_URL, http_client=http_client)
        try:
            [chunk async for chunk in client.ask_stream("x")]
        finally:
            await http_client.aclose()

    with pytest.raises(SplashConnectionError):
        asyncio.run(run())


def test_waiting_for_a_server_that_stays_down_gives_up() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, http_client = _client(handler)
    try:
        assert client.wait_until_ready(timeout=0.05, interval=0.01) is False
    finally:
        http_client.close()


def test_a_response_that_is_not_json_gives_an_empty_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client, http_client = _client(handler)
    try:
        assert client.ask("x") == ""
    finally:
        http_client.close()


def test_ask_stream_skips_the_lines_that_carry_nothing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = b'data: not json\n\n: keepalive\n\ndata: {"choices": []}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=payload)

    client, http_client = _client(handler)
    try:
        chunks = list(client.ask_stream("ping"))
    finally:
        http_client.close()

    assert chunks == []


def test_the_client_can_be_used_as_a_context_manager() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_payload())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    with SplashClient(BASE_URL, http_client=http_client) as client:
        assert client.ask("ping") == "pong"
    http_client.close()


def test_closing_the_client_closes_its_connection() -> None:
    client = SplashClient(BASE_URL)

    client.close()


def test_api_key_is_sent_as_a_bearer_token() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization", "")
        return httpx.Response(200, json=_completion_payload())

    client, http_client = _client(handler, api_key="s3cret")
    try:
        client.ask("ping")
    finally:
        http_client.close()

    assert seen["authorization"] == "Bearer s3cret"


def test_a_reply_is_the_answer_text() -> None:
    reply = Reply("pong")

    assert reply == "pong"
    assert reply.upper() == "PONG"
    assert f"{reply}!" == "pong!"


def test_body_building_leaves_the_model_out_by_default() -> None:
    body = _build_body("ping")

    assert "model" not in body
    assert body["messages"] == [{"role": "user", "content": "ping"}]


@pytest.fixture(scope="module")
def live_client() -> SplashClient:
    """Return a client for a running Splash server, or skip the test."""
    client = SplashClient(BASE_URL, timeout=300)
    if not client.wait_until_ready(timeout=30.0):
        client.close()
        pytest.skip("no Splash server is running on this host")
    return client


def test_live_server_answers_a_prompt(live_client: SplashClient) -> None:
    reply = live_client.ask("Reply with exactly: pong", thinking=False, max_tokens=64)

    assert reply.strip() == "pong"
    assert reply.usage is not None
    assert reply.usage["completion_tokens"] > 0


def test_live_server_reports_a_reasoning_trace(live_client: SplashClient) -> None:
    reply = live_client.ask("What is 2+2? Answer with just the number.", max_tokens=1024)

    assert reply.strip() == "4"
    assert reply.reasoning.strip()


def test_live_server_lists_a_model(live_client: SplashClient) -> None:
    models = live_client.models()

    assert models
    assert models[0]


def test_live_server_streams(live_client: SplashClient) -> None:
    chunks = list(live_client.ask_stream("Say hello", thinking=False, max_tokens=64))

    assert chunks
    assert "".join(chunks).strip()


def test_live_server_rejects_an_unknown_model(live_client: SplashClient) -> None:
    with pytest.raises(SplashAPIError) as caught:
        live_client.ask("x", model="no-such-model")

    assert caught.value.status_code == 404
