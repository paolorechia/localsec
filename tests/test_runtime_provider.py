"""Tests for the model session."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from localsec.runtime import provider
from localsec.runtime.provider import ModelSession, SplashModelConfig, resident_model_id
from localsec.splash import SplashError


class _FakeClient:
    """Stands in for the Splash client when looking the resident model up."""

    held: ClassVar[list[str]] = ["a/b"]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def models(self) -> list[str]:
        return type(self).held


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Replace the Splash client with one that holds a known model."""
    _FakeClient.held = ["a/b"]
    monkeypatch.setattr(provider, "SplashClient", _FakeClient)
    return _FakeClient


def test_the_resident_model_is_looked_up(fake_client: type[_FakeClient]) -> None:
    assert resident_model_id() == "a/b"


def test_a_server_that_holds_no_model_is_an_error(fake_client: type[_FakeClient]) -> None:
    fake_client.held = []

    with pytest.raises(SplashError):
        resident_model_id()


def test_the_session_binds_the_model_it_is_given() -> None:
    session = ModelSession(SplashModelConfig(model_id="some/model"))

    assert session.model_id == "some/model"


def test_the_session_looks_the_model_up_when_it_is_not_given(
    fake_client: type[_FakeClient],
) -> None:
    session = ModelSession(SplashModelConfig())

    assert session.model_id == "a/b"


def test_the_session_points_at_the_v1_api() -> None:
    session = ModelSession(SplashModelConfig(model_id="m", base_url="http://127.0.0.1:9000"))

    base_url: Any = session._client.base_url
    assert str(base_url).rstrip("/") == "http://127.0.0.1:9000/v1"


def test_proxy_variables_do_not_reach_the_server() -> None:
    session = ModelSession(SplashModelConfig(model_id="m"))

    transport: Any = session._client._client
    assert transport._trust_env is False
