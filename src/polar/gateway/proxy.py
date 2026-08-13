"""HTTP client for forwarding requests to an OpenAI-compatible inference server.

Backend differences (request params, response shape) are isolated in the
``InferenceEngine`` strategy this client holds; the HTTP/streaming/pause logic
here is backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from polar.gateway.engine import InferenceEngine

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Base class for upstream gateway failures."""


class UpstreamHTTPError(UpstreamError):
    """Raised when the upstream returns a non-2xx status."""

    def __init__(self, status_code: int, body: dict[str, Any] | str | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(self._build_message(status_code, body))

    @staticmethod
    def _build_message(status_code: int, body: dict[str, Any] | str | None) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = body.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(body, str) and body:
            return body
        return f"Upstream request failed with status {status_code}"


class UpstreamTimeoutError(UpstreamError):
    """Raised when the upstream times out."""


class UpstreamTransportError(UpstreamError):
    """Raised for connection and transport failures."""


class InferenceClient:
    """Direct httpx client to an inference server's OpenAI-compatible API.

    Per-call bound comes from the session's remaining-timeout budget
    (`_await_with_budget` at the gateway node). The internal httpx timeout
    is a high liveness ceiling so that a stuck engine can't pin a request
    past the session deadline. The ``engine`` strategy injects backend-specific
    request params and canonicalizes responses.
    """

    _DEFAULT_LIVENESS_TIMEOUT_SECONDS = 900.0
    _LIVENESS_TIMEOUT_ENV = "POLAR_INFERENCE_REQUEST_TIMEOUT_SECONDS"

    def __init__(
        self,
        base_url: str,
        engine: InferenceEngine,
        *,
        liveness_timeout_seconds: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        self._liveness_timeout_seconds = (
            self._coerce_liveness_timeout_seconds(liveness_timeout_seconds)
            if liveness_timeout_seconds is not None
            else self._read_liveness_timeout_seconds()
        )
        self._client: httpx.AsyncClient | None = None
        self._generation_paused = False
        self._inflight_generations = 0
        self._generation_condition = asyncio.Condition()

    @classmethod
    def _read_liveness_timeout_seconds(cls) -> float:
        raw = os.environ.get(cls._LIVENESS_TIMEOUT_ENV)
        if raw is None or raw.strip() == "":
            return cls._DEFAULT_LIVENESS_TIMEOUT_SECONDS
        try:
            return cls._coerce_liveness_timeout_seconds(float(raw))
        except ValueError as exc:
            raise ValueError(
                f"{cls._LIVENESS_TIMEOUT_ENV} must be a positive number of seconds"
            ) from exc

    @staticmethod
    def _coerce_liveness_timeout_seconds(value: float) -> float:
        timeout = float(value)
        if timeout <= 0:
            raise ValueError("liveness timeout must be greater than 0")
        return timeout

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._liveness_timeout_seconds, connect=30),
            )
        return self._client

    async def _read_error_body(self, response: httpx.Response) -> dict[str, Any] | str | None:
        content = await response.aread()
        if not content:
            return None

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        body = await self._read_error_body(response)
        await response.aclose()
        raise UpstreamHTTPError(response.status_code, body)

    @staticmethod
    def _translate_transport_error(exc: httpx.RequestError) -> UpstreamError:
        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError("Upstream request timed out")
        return UpstreamTransportError(f"Upstream request failed: {exc}")

    async def completion(
        self, request: dict[str, Any], *, trace_headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full JSON response.

        ``trace_headers`` (e.g. ``x-polar-trace-id``) are forwarded to the engine so the
        engine-side per-request logger can join back to this gateway completion. Purely
        observability -- never affects generation.
        """
        await self._acquire_generation_slot()
        client = await self._get_client()
        from copy import deepcopy

        request_copy = deepcopy(request)
        request_copy.pop("stream", None)
        request_copy["stream"] = False
        request_copy = self.engine.prepare_request(request_copy)
        headers = {"Content-Type": "application/json", "x-polar-engine-url": self.base_url}
        if trace_headers:
            headers.update({str(k): str(v) for k, v in trace_headers.items()})
            # Session affinity. vime's LB proxy pins a session to one engine when
            # x-session-id is present (select_server_by_session); without it the
            # proxy falls back to active_tokens load balancing. We never sent this
            # header, so a session's turns drifted between engines: measured 16.5%
            # of turns switched, and those got 45% prefix cache hit vs 91% for
            # turns that stayed. The session id is already carried in
            # x-polar-trace-id as "{session_id}:{turn_seq}".
            _trace_id = headers.get("x-polar-trace-id", "")
            if _trace_id and "x-session-id" not in headers:
                _session_id = _trace_id.rsplit(":", 1)[0]
                if _session_id:
                    headers["x-session-id"] = _session_id
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json=request_copy,
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        finally:
            await self._release_generation_slot()

        await self._raise_for_status(resp)
        return self.engine.normalize_response(resp.json())

    async def _acquire_generation_slot(self) -> None:
        async with self._generation_condition:
            await self._generation_condition.wait_for(lambda: not self._generation_paused)
            self._inflight_generations += 1

    async def _release_generation_slot(self) -> None:
        async with self._generation_condition:
            self._inflight_generations -= 1
            self._generation_condition.notify_all()

    async def pause_generation(self, *, timeout_seconds: float = 300.0) -> dict[str, Any]:
        """Block new generation requests and wait for current inference calls to drain."""
        async with self._generation_condition:
            self._generation_paused = True
            self._generation_condition.notify_all()
            await asyncio.wait_for(
                self._generation_condition.wait_for(lambda: self._inflight_generations == 0),
                timeout=timeout_seconds,
            )
            return self.generation_status()

    async def resume_generation(self) -> dict[str, Any]:
        async with self._generation_condition:
            self._generation_paused = False
            self._generation_condition.notify_all()
            return self.generation_status()

    def generation_status(self) -> dict[str, Any]:
        return {
            "paused": self._generation_paused,
            "inflight": self._inflight_generations,
            "base_url": self.base_url,
            "engine": self.engine.name,
            "request_timeout_seconds": self._liveness_timeout_seconds,
        }

    async def list_models(self) -> dict[str, Any]:
        """Passthrough GET /v1/models."""
        client = await self._get_client()
        try:
            resp = await client.get("/v1/models")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        return resp.json()

    async def health(self) -> dict[str, Any]:
        """Passthrough GET /health."""
        client = await self._get_client()
        try:
            resp = await client.get("/health")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        content = await resp.aread()
        if not content:
            return {"status": "ok"}

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return {"status": "ok"}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "ok", "body": text}

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
