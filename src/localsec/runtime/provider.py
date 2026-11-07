"""The model the agents run on: a Pydantic AI model bound to Splash.

Splash answers to the name of the model it currently holds, and to nothing
else, so the id is looked up when the session starts rather than configured:
the resident model changes with the GGUF the server was launched with.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from localsec.splash import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, SplashClient, SplashError

__all__ = ["ModelSession", "SplashModelConfig", "resident_model_id"]


@dataclass(frozen=True, slots=True)
class SplashModelConfig:
    """Where the model server is, and which model to ask it for."""

    base_url: str = DEFAULT_BASE_URL
    model_id: str | None = None
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT


def resident_model_id(base_url: str = DEFAULT_BASE_URL, *, timeout: float = 10.0) -> str:
    """Return the id of the model the server currently holds.

    Raises:
        SplashError: the server holds no model.
    """
    with SplashClient(base_url, timeout=timeout) as client:
        models = client.models()
    if not models:
        raise SplashError(f"the server at {base_url} holds no model")
    return models[0]


class ModelSession:
    """A model bound to Splash, and the client it owns.

    The client is created with `trust_env=False`: proxy environment
    variables would otherwise be applied to a request aimed at localhost.
    """

    def __init__(self, config: SplashModelConfig | None = None) -> None:
        """Bind a model to the server the configuration points at."""
        settings = config or SplashModelConfig()
        model_id = settings.model_id or resident_model_id(
            settings.base_url, timeout=settings.timeout
        )
        self._client = AsyncOpenAI(
            base_url=f"{settings.base_url.rstrip('/')}/v1",
            api_key=settings.api_key or "splash",
            http_client=httpx2.AsyncClient(timeout=settings.timeout, trust_env=False),
        )
        self._provider = OpenAIProvider(openai_client=self._client)
        self.model = OpenAIChatModel(model_id, provider=self._provider)

    @property
    def model_id(self) -> str:
        """Return the id of the model this session talks to."""
        return self.model.model_name

    async def aclose(self) -> None:
        """Close the connection to the server."""
        await self._client.close()

    async def __aenter__(self) -> ModelSession:
        """Enter a context that closes the client on exit."""
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        """Close the client."""
        await self.aclose()
