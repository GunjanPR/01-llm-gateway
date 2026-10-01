"""Clients for upstream model servers that speak the OpenAI-compatible HTTP API.

Ollama (`/v1`), vLLM (`vllm serve`) and most hosted providers all expose the same
`/chat/completions` and `/embeddings` routes, so a single adapter covers them.
"""

from collections.abc import AsyncGenerator
from typing import Any

import httpx

from llm_gateway.config import BackendConfig

# Upstream statuses that mean the request itself is bad. Retrying or falling back to
# another backend would fail the same way, so these go straight back to the client.
CLIENT_ERROR_STATUSES = {400, 413, 422}


class BackendError(Exception):
    def __init__(self, backend: str, message: str, status_code: int | None = None) -> None:
        super().__init__(f"{backend}: {message}")
        self.backend = backend
        self.message = message
        self.status_code = status_code

    @property
    def is_client_error(self) -> bool:
        return self.status_code in CLIENT_ERROR_STATUSES

    @property
    def kind(self) -> str:
        return "connection" if self.status_code is None else f"http_{self.status_code}"


class OpenAICompatibleBackend:
    def __init__(
        self, config: BackendConfig, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.config = config
        self.name = config.name
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds, connect=5.0),
            transport=transport,
        )

    def serves(self, model: str) -> bool:
        return self.config.serves(model)

    def _upstream(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "model": self.config.upstream_model(payload["model"])}

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=self._upstream(payload))
        except httpx.HTTPError as exc:
            raise BackendError(self.name, f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise BackendError(self.name, _error_text(response), response.status_code)
        return response.json()

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/chat/completions", {**payload, "stream": False})

    async def embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/embeddings", payload)

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Yield raw SSE `data:` lines from the upstream stream."""
        body = {**self._upstream(payload), "stream": True}
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise BackendError(self.name, _error_text(response), response.status_code)
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        yield line
        except httpx.HTTPError as exc:
            raise BackendError(self.name, f"{type(exc).__name__}: {exc}") from exc

    async def health(self) -> bool:
        try:
            response = await self._client.get("/models", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    async def aclose(self) -> None:
        await self._client.aclose()


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error or data)[:500]
