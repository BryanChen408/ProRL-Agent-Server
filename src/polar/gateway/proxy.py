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
from collections.abc import Callable
from typing import Any

import httpx

from polar.gateway.engine import InferenceEngine

logger = logging.getLogger(__name__)


def _nonnegative_token_count(value: Any) -> int:
    """Return a safe token count for optional OpenAI-compatible usage fields."""
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _extract_token_counts(result: Any) -> tuple[int, int]:
    """Extract prompt/output counts, preferring exact returned output token IDs."""
    if not isinstance(result, dict):
        return 0, 0
    usage = result.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens = _nonnegative_token_count(usage.get("prompt_tokens"))

    choices = result.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    first_choice = first_choice if isinstance(first_choice, dict) else {}
    token_ids = first_choice.get("token_ids")
    if isinstance(token_ids, list):
        response_tokens = len(token_ids)
    else:
        # Some OpenAI-compatible engines report standard usage but omit the
        # non-standard ``return_token_ids`` response extension.
        response_tokens = _nonnegative_token_count(usage.get("completion_tokens"))
    return prompt_tokens, response_tokens


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
        initially_paused: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        self._liveness_timeout_seconds = (
            self._coerce_liveness_timeout_seconds(liveness_timeout_seconds)
            if liveness_timeout_seconds is not None
            else self._read_liveness_timeout_seconds()
        )
        self._client: httpx.AsyncClient | None = None
        # A gateway restarted in the middle of a weight transition must not briefly
        # reopen inference before the rollout coordinator reconnects.
        self._generation_paused = bool(initially_paused)
        self._inflight_generations = 0
        self._generation_drained = asyncio.Event()
        self._generation_drained.set()
        self._generation_condition = asyncio.Condition()
        # Per-call timing (reset on each completion() call)
        self.last_acquire_wait_ms: float = 0.0     # B: semaphore wait
        self.last_prepare_ms: float = 0.0          # C: prepare_request
        self.last_sglang_wait_ms: float = 0.0      # D+E+F+G: httpx→SGLang→response
        self.last_normalize_ms: float = 0.0        # H: normalize_response
        self.last_roundtrip_ms: float = 0.0         # t0→t2 (不含 acquire, 不含 storage)
        self.last_prompt_tokens: int = 0            # prompt token count
        self.last_response_tokens: int = 0          # response token count
        self.last_post_ms: float = 0.0              # I: storage + response formatting
        # Keyed by returned response identity so concurrent sessions cannot overwrite
        # one another's trace boundaries while retaining the legacy ``last_*`` fields.
        self._response_trace_timings: dict[int, dict[str, int]] = {}
        # 12 & 17: Agent↔Gateway boundary timing
        self.arrival_time: float | None = None      # 12: request arrived at gateway
        self.gateway_total_ms: float = 0.0          # 12→17: total gateway processing

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
                # No connection reuse. Closing an idle keep-alive connection is a
                # UNILATERAL server decision with no protocol handshake, and a write
                # to a half-closed socket SUCCEEDS -- the peer's FIN only surfaces on
                # the following read, as an httpx.ReadError wrapping anyio.EndOfStream
                # (which stringifies to ""). The upstream LB proxy runs uvicorn with
                # its default timeout_keep_alive=5s while agent turns idle for minutes
                # between tool calls (compile / NPU verify), so a pooled connection is
                # routinely dead by the time the next turn reuses it. Measured on run
                # 092443: 252 such failures, ~52/h, each one flagging an otherwise
                # complete session non-trainable. Client-side expiry tuning only
                # narrows the window (both ends time independently); dropping reuse
                # removes the possibly-dead-pooled-socket state entirely. Cost is one
                # TCP handshake per request -- both ends are on the same host, no TLS,
                # so ~50-200us against a request that generates for seconds to minutes.
                # max_connections is restated because httpx.DEFAULT_LIMITS only applies
                # when `limits` is omitted entirely -- a bare Limits(...) leaves it None,
                # i.e. UNLIMITED concurrent sockets, which is not the change we want.
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=0),
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
        # Carry the exception CLASS: several httpx.RequestError subclasses stringify to
        # "" (ReadError / WriteError / CloseError wrapping a bare socket failure), which
        # collapsed this message to a bare "Upstream request failed: " and made the
        # dominant transport failure mode undiagnosable from logs alone.
        return UpstreamTransportError(f"Upstream request failed: {type(exc).__name__}: {exc}")

    async def completion(
        self,
        request: dict[str, Any],
        *,
        trace_headers: dict[str, str] | None = None,
        generation_guard: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion with per-phase timing.

        ``trace_headers`` (e.g. ``x-polar-trace-id``) are forwarded to the engine so the
        engine-side per-request logger can join back to this gateway completion. Purely
        observability -- never affects generation.

        Timing segments (all in milliseconds):
          B. acquire_wait_ms  — semaphore wait before request processing
          C. prepare_ms       — deepcopy + engine.prepare_request
          D-G. sglang_wait_ms — httpx POST → SGLang (prefill+decode) → response
          H. normalize_ms     — engine.normalize_response
          I. (post_ms)        — save_message + transform_response (in server.py)
        """
        import time as _time
        from copy import deepcopy

        # ── B: acquire slot ──
        t_acquire_start = _time.monotonic()
        t_acquire_start_ns = _time.time_ns()
        await self._acquire_generation_slot()
        t_acquire_end = _time.monotonic()
        t_acquire_end_ns = _time.time_ns()

        try:
            # A request may have entered the gateway before pause and waited behind
            # the condition while the serving epoch advanced. Re-check at the exact
            # engine-admission point so it cannot run with the next policy's weights.
            if generation_guard is not None and not generation_guard():
                raise UpstreamError("generation rejected by policy epoch fence")

            client = await self._get_client()

            # ── C: prepare ──
            t_prepare_start = _time.monotonic()
            t_prepare_start_ns = _time.time_ns()
            request_copy = deepcopy(request)
            request_copy.pop("stream", None)
            request_copy["stream"] = False
            request_copy = self.engine.prepare_request(request_copy)
            t_prepare_end = _time.monotonic()
            t_prepare_end_ns = _time.time_ns()

            headers = {
                "Content-Type": "application/json",
                "x-polar-engine-url": self.base_url,
            }
            if trace_headers:
                headers.update({str(k): str(v) for k, v in trace_headers.items()})
                # 引擎会话亲和:vime 的 PD/LB proxy 收到 x-session-id 时,把同一 session 的每轮
                # 稳定哈希到固定引擎(prefill/decode 各自 sticky),让前缀 KV 跨轮复用、只 prefill
                # 增量;收不到就退回 round-robin/active_tokens,前缀被打散 → 每轮重灌整段上下文。
                # gateway 自管 session id 但从不下发,这条亲和路径从上线起没被走到过。session id
                # 已在 x-polar-trace-id 里(格式 "{session_id}:{turn_seq}"),取冒号前段补发。
                # 无 trace-id → 不补,行为不变(PD 分离功能零影响)。
                trace_id = headers.get("x-polar-trace-id", "")
                if trace_id and "x-session-id" not in headers:
                    session_id = trace_id.rsplit(":", 1)[0]
                    if session_id:
                        headers["x-session-id"] = session_id

            # ── D-G: SGLang inference ──
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=request_copy,
                    headers=headers,
                )
                t_sglang_end = _time.monotonic()
                t_sglang_end_ns = _time.time_ns()
            except httpx.RequestError as exc:
                raise self._translate_transport_error(exc) from exc
        finally:
            # Session deletion and engine abort can cancel the same request more than
            # once.  Slot release must therefore contain no suspension point: a second
            # cancellation must not interrupt the decrement and leave a false non-zero
            # drain count that blocks the next training boundary forever.
            self._release_generation_slot()

        await self._raise_for_status(resp)

        # ── H: normalize ──
        t_normalize_start = _time.monotonic()
        t_normalize_start_ns = _time.time_ns()
        result = self.engine.normalize_response(resp.json())
        t_normalize_end = _time.monotonic()
        t_normalize_end_ns = _time.time_ns()

        # ── Store per-phase timing ──
        self.last_acquire_wait_ms = (t_acquire_end - t_acquire_start) * 1000.0
        self.last_prepare_ms      = (t_prepare_end - t_prepare_start) * 1000.0
        self.last_sglang_wait_ms  = (t_sglang_end - t_prepare_end) * 1000.0
        self.last_normalize_ms    = (t_normalize_end - t_normalize_start) * 1000.0
        self.last_roundtrip_ms    = (t_normalize_end - t_acquire_end) * 1000.0

        # ── Extract token counts ──
        self.last_prompt_tokens, self.last_response_tokens = _extract_token_counts(result)

        self._response_trace_timings[id(result)] = {
            "request_started_at_ns": t_acquire_start_ns,
            "acquire_started_at_ns": t_acquire_start_ns,
            "acquire_finished_at_ns": t_acquire_end_ns,
            "prepare_started_at_ns": t_prepare_start_ns,
            "prepare_finished_at_ns": t_prepare_end_ns,
            "sglang_started_at_ns": t_prepare_end_ns,
            "sglang_finished_at_ns": t_sglang_end_ns,
            "normalize_started_at_ns": t_normalize_start_ns,
            "normalize_finished_at_ns": t_normalize_end_ns,
        }

        return result

    def pop_trace_timing(self, response: dict[str, Any]) -> dict[str, int]:
        """Return per-response clock boundaries without cross-session races."""
        return self._response_trace_timings.pop(id(response), {})

    async def _acquire_generation_slot(self) -> None:
        async with self._generation_condition:
            await self._generation_condition.wait_for(lambda: not self._generation_paused)
            self._inflight_generations += 1
            self._generation_drained.clear()

    def _release_generation_slot(self) -> None:
        self._inflight_generations -= 1
        if self._inflight_generations == 0:
            self._generation_drained.set()

    async def pause_generation(
        self,
        *,
        timeout_seconds: float = 300.0,
        wait_for_drain: bool = True,
    ) -> dict[str, Any]:
        """Block new generations and optionally wait for existing calls to drain.

        Reaching the drain timeout does *not* reopen admission and is not a transport
        failure: callers may deliberately abort the remaining engine requests before a
        colocated weight update.  Keep ``paused=True`` and expose the two states
        independently so orchestration can make that decision without guessing from a
        504 response. ``wait_for_drain=False`` is the synchronous-training admission
        fence: it returns immediately so the trainer can abort engines itself.
        """
        async with self._generation_condition:
            self._generation_paused = True
            self._generation_condition.notify_all()

        timed_out = False
        if wait_for_drain and self._inflight_generations != 0:
            # The scalar is authoritative.  Reconcile the notification event here as
            # well so recovery from an older process state (or test instrumentation)
            # cannot turn a non-zero count into a false drained acknowledgement.
            self._generation_drained.clear()
            try:
                await asyncio.wait_for(
                    self._generation_drained.wait(),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                timed_out = True

        status = self.generation_status()
        status["timed_out"] = timed_out
        status["wait_for_drain"] = wait_for_drain
        return status

    async def resume_generation(self) -> dict[str, Any]:
        async with self._generation_condition:
            self._generation_paused = False
            self._generation_condition.notify_all()
            return self.generation_status()

    def generation_status(self) -> dict[str, Any]:
        return {
            "paused": self._generation_paused,
            "drained": self._inflight_generations == 0,
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
