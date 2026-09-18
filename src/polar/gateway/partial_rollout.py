"""Keep agent sessions alive and restart interrupted chat calls from their prompt.

Only a locally recorded policy-boundary cancellation is resumable. An unfinished
response is discarded completely; completed turns and their behaviour logprobs
stay in normal session storage. No vLLM source extensions are required.
Checkpoints are audit artifacts, not permission to replay tools after a crash.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from polar.gateway.proxy import UpstreamError, _extract_token_counts


class PartialRollout:
    def __init__(self, owner, checkpoint_dir: str | Path):
        self.owner = owner
        self.checkpoint_dir = Path(checkpoint_dir)
        self.policy_version = lambda: None
        self.waiting: dict[str, float] = {}
        self.paused_seconds: dict[str, float] = {}
        self._session_pause_started: dict[str, float] = {}
        self.priority: set[str] = set()
        self.active: set[str] = set()
        self._attempts: dict[str, asyncio.Task] = {}
        self._preempted: set[str] = set()

    def pause_seconds(self, session_id: str) -> float:
        started = self._session_pause_started.get(session_id)
        return self.paused_seconds.get(session_id, 0.0) + (
            time.monotonic() - started if started is not None else 0.0
        )

    def checkpoint(self, key: str, state: dict) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(state, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(self.checkpoint_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def interrupt_active(self) -> None:
        """Cancel upstream HTTP attempts, leaving logical calls and agents alive.

        Closing even requests awaiting headers avoids depending on a final abort
        response. The trainer must still obtain the native engine pause/abort
        acknowledgement before training. Already completed attempts are retained.
        """
        for key, task in self._attempts.items():
            if not task.done() and key not in self._preempted:
                self._preempted.add(key)
                self.priority.add(key)
                task.cancel()

    async def acquire(self, key: str):
        owner = self.owner
        session = key.rsplit(":", 1)[0]
        async with owner._generation_condition:
            if owner._generation_paused:
                self.waiting.setdefault(key, time.monotonic())
                self._session_pause_started.setdefault(session, time.monotonic())
            try:
                await owner._generation_condition.wait_for(
                    lambda: not owner._generation_paused
                    and (not self.priority or key in self.priority)
                )
            finally:
                was_waiting = self.waiting.pop(key, None) is not None
                if was_waiting and not any(k.rsplit(":", 1)[0] == session for k in self.waiting):
                    started = self._session_pause_started.pop(session)
                    self.paused_seconds[session] = (
                        self.paused_seconds.get(session, 0.0) + time.monotonic() - started
                    )
            self.priority.discard(key)
            owner._generation_condition.notify_all()
            owner._inflight_generations += 1
            owner._generation_drained.clear()
            self.active.add(key)

    def release(self, key):
        self.active.discard(key)
        self.owner._release_generation_slot()

    async def _complete(self, request, headers):
        client = await self.owner._get_client()
        response = await client.post(
            "/v1/chat/completions", json=deepcopy(request), headers=headers
        )
        await self.owner._raise_for_status(response)
        result = self.owner.engine.normalize_response(response.json())
        choices = result.get("choices") or []
        if len(choices) != 1 or choices[0].get("finish_reason") not in {
            "stop",
            "tool_calls",
            "length",
        }:
            raise UpstreamError("unplanned/incomplete chat generation")
        choice = choices[0]
        ids = choice.get("token_ids")
        probs = (choice.get("logprobs") or {}).get("content")
        try:
            valid = isinstance(ids, list) and isinstance(probs, list) and len(ids) == len(probs)
            valid = valid and all(
                math.isfinite(float(p["logprob"])) and -9999 < float(p["logprob"]) <= 0
                for p in probs
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise UpstreamError("partial rollout requires exact per-token behaviour logprobs")
        return result

    async def run(self, request, trace_headers, generation_guard):
        trace_id = (trace_headers or {}).get("x-polar-trace-id")
        if not trace_id:
            raise UpstreamError("partial rollout requires a stable session/turn identity")
        from polar.gateway.inflight import request_fingerprint

        session_id = trace_id.rsplit(":", 1)[0]
        key = f"{session_id}:{request_fingerprint(request)}"
        owner = self.owner
        headers = {
            "Content-Type": "application/json",
            "x-polar-engine-url": owner.base_url,
            **(trace_headers or {}),
            "x-session-id": session_id,
        }
        chat_request = owner.engine.prepare_request(deepcopy(request))
        chat_request["stream"] = False
        chat_request.pop("stream_options", None)
        if chat_request.get("n", 1) != 1:
            raise UpstreamError("partial rollout requires n=1")
        # Contains the original history plus this turn's prompt. Never append the
        # interrupted attempt's output or rerun earlier agent tools.
        state = {
            "key": key,
            "mode": "session_restart",
            "request": chat_request,
            "status": "running",
            "pause_count": 0,
        }
        started = time.monotonic()
        started_ns = time.time_ns()
        paused_before = self.pause_seconds(session_id)
        try:
            while True:
                await self.acquire(key)
                try:
                    if generation_guard is not None and not generation_guard():
                        raise UpstreamError("partial generation rejected by policy epoch fence")
                    version = self.policy_version()
                    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
                        raise UpstreamError(
                            "partial rollout requires a verified serving policy version"
                        )
                    first_version = state.setdefault("first_attempt_policy_version", version)
                    if not 0 <= version - first_version <= 1:
                        raise UpstreamError("partial generation exceeded one policy update")
                    attempt = asyncio.create_task(self._complete(chat_request, headers))
                    self._attempts[key] = attempt
                    try:
                        result = await attempt
                    except asyncio.CancelledError:
                        # An external session cancellation wins even if it races a
                        # planned boundary; never resurrect a cancelled agent.
                        if key not in self._preempted or asyncio.current_task().cancelling():
                            raise
                    else:
                        if key not in self._preempted:
                            break
                    state["status"] = "paused"
                    state["pause_count"] += 1
                    state["discarded_attempt_policy_version"] = version
                    # Save the request BEFORE acknowledging drain. The interrupted
                    # response and its logprobs are never stored or trained.
                    self.checkpoint(key, state)
                finally:
                    self._attempts.pop(key, None)
                    self._preempted.discard(key)
                    self.release(key)

            elapsed = time.monotonic() - started
            paused = self.pause_seconds(session_id) - paused_before
            result["_polar_partial"] = {
                "verified": True,
                "mode": "session_restart",
                "policy_version": version,
                "pause_count": state["pause_count"],
                "planned_pause_seconds": paused,
                "active_seconds": max(0.0, elapsed - paused),
            }
            state.update(status="completed", policy_version=version)
            if state["pause_count"]:
                self.checkpoint(key, state)
            owner.last_prompt_tokens, owner.last_response_tokens = _extract_token_counts(result)
            owner.last_roundtrip_ms = elapsed * 1000
            owner.last_acquire_wait_ms = paused * 1000
            owner.last_sglang_wait_ms = max(0.0, elapsed - paused) * 1000
            owner._response_trace_timings[id(result)] = {
                "request_started_at_ns": started_ns,
                "response_finished_at_ns": time.time_ns(),
            }
            return result
        finally:
            self.priority.discard(key)
            async with owner._generation_condition:
                owner._generation_condition.notify_all()
