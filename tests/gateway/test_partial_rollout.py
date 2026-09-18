from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from polar.gateway.engine import VLLMEngine
from polar.gateway.inflight import InflightGenerationTracker
from polar.gateway.partial_rollout import PartialRollout
from polar.gateway.proxy import InferenceClient, UpstreamError
from polar.gateway.node import GatewayNodeManager
from polar.gateway.storage import SessionStore


def completion(ids=(11, 12), finish="stop", probs=None):
    return {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "token_ids": list(ids),
                "finish_reason": finish,
                "message": {"role": "assistant", "reasoning": "checked", "content": "finished"},
                "logprobs": {
                    "content": probs
                    if probs is not None
                    else [{"token": str(t), "logprob": -t / 100} for t in ids]
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": len(ids)},
    }


def test_restart_turn_44_keeps_43_turns_and_discards_unfinished_response(tmp_path):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine())
        partial = owner.partial = PartialRollout(owner, tmp_path)
        storage = SessionStore()
        storage.set_policy_version(0)
        partial.policy_version = storage.get_policy_version
        entered = asyncio.Event()
        closed = asyncio.Event()
        requests = []

        class InterruptedBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                # No terminal JSON/abort will ever arrive. Even a partial tool
                # response must never reach the agent, storage, or training.
                try:
                    yield b'{"choices":[{"message":{"tool_calls":[{"id":"DISCARDED"'
                    entered.set()
                    await asyncio.Future()
                finally:
                    closed.set()

        async def transport(request):
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["x-session-id"] == "session"
            requests.append(json.loads(request.content))
            if len(requests) == 44:
                return httpx.Response(200, stream=InterruptedBody())
            result = completion()
            if len(requests) == 45:
                result = completion((21, 22), "tool_calls")
                result["choices"][0]["message"]["tool_calls"] = [
                    {
                        "id": "new-tool",
                        "type": "function",
                        "function": {"name": "run_verify", "arguments": "{}"},
                    }
                ]
            return httpx.Response(200, json=result)

        owner._client = httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(transport)
        )
        messages = []
        for turn in range(1, 44):
            messages.append({"role": "user", "content": f"turn {turn}"})
            request = {"model": "test", "messages": deepcopy(messages), "max_tokens": 32768}
            result = await owner.completion(
                request, trace_headers={"x-polar-trace-id": f"session:{turn}"}
            )
            storage.save_message("session", request, result)
            messages.append(deepcopy(result["choices"][0]["message"]))
        completed_before = deepcopy(storage.get_completions("session"))
        assert len(completed_before) == 43 and not list(tmp_path.glob("*.json"))
        messages.append({"role": "user", "content": "turn 44"})
        request = {
            "model": "test",
            "messages": deepcopy(messages),
            "max_tokens": 32768,
            "stop": ["STOP"],
            "frequency_penalty": 0.2,
        }
        task = asyncio.create_task(
            owner.completion(request, trace_headers={"x-polar-trace-id": "session:44"})
        )
        await asyncio.wait_for(entered.wait(), 1)
        await owner.pause_generation(wait_for_drain=False)
        await asyncio.wait_for(owner._generation_drained.wait(), 1)
        assert closed.is_set() and not task.done()
        saved = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert saved["request"] == requests[43]
        assert "token_ids" not in saved and "logprobs" not in saved
        assert saved["status"] == "paused"
        assert storage.get_completions("session") == completed_before
        assert owner.generation_status()["resume_pending"] == 1
        await asyncio.sleep(0.01)
        storage.set_policy_version(1)
        await owner.resume_generation()
        result = await asyncio.wait_for(task, 1)
        storage.save_message("session", request, result)
        assert len(requests) == 45 and requests[44] == requests[43]
        assert requests[44]["max_tokens"] == 32768
        assert len(requests[44]["messages"]) == 87
        assert request["messages"] == messages  # prepare_request did not mutate history
        choice = result["choices"][0]
        assert choice["token_ids"] == [21, 22]
        assert [p["logprob"] for p in choice["logprobs"]["content"]] == [-0.21, -0.22]
        assert choice["message"]["reasoning_content"] == "checked"
        assert choice["message"]["tool_calls"][0]["id"] == "new-tool"
        assert result["_polar_partial"]["policy_version"] == 1
        assert result["_polar_partial"]["pause_count"] == 1
        saved_turns = storage.get_completions("session")
        assert len(saved_turns) == 44 and saved_turns[:43] == completed_before
        assert [c["metadata"]["policy_version"] for c in saved_turns] == [0] * 43 + [1]
        assert "DISCARDED" not in json.dumps(saved_turns)
        assert partial.pause_seconds("session") >= 0.01
        await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "finish,probs,error",
    [
        ("abort", None, "unplanned"),
        (None, None, "incomplete"),
        ("stop", [], "exact per-token"),
        ("stop", [{"logprob": float("nan")}, {"logprob": -0.1}], "exact per-token"),
    ],
)
def test_unplanned_abort_and_missing_logprobs_are_not_restarted(tmp_path, finish, probs, error):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine())
        owner.partial = PartialRollout(owner, tmp_path)
        owner.partial.policy_version = lambda: 0
        calls = 0

        async def transport(request):
            nonlocal calls
            calls += 1
            # Use content rather than json= to permit an invalid NaN fixture.
            return httpx.Response(200, content=json.dumps(completion(finish=finish, probs=probs)))

        owner._client = httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(transport)
        )
        with pytest.raises(UpstreamError, match=error):
            await owner.completion({"messages": []}, trace_headers={"x-polar-trace-id": "s:1"})
        assert calls == 1 and owner.generation_status()["drained"]
        await owner.close()

    asyncio.run(run())


def test_native_length_finish_is_not_restarted(tmp_path):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine())
        owner.partial = PartialRollout(owner, tmp_path)
        owner.partial.policy_version = lambda: 0

        async def transport(request):
            return httpx.Response(200, json=completion(finish="length"))

        owner._client = httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(transport)
        )
        result = await owner.completion(
            {"messages": []}, trace_headers={"x-polar-trace-id": "s:1"}
        )
        assert result["choices"][0]["finish_reason"] == "length"
        assert result["_polar_partial"]["pause_count"] == 0
        assert not list(tmp_path.glob("*.json"))
        await owner.close()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["inflight", "paused", "race"])
def test_session_cancellation_is_never_restarted(tmp_path, phase):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine())
        owner.partial = PartialRollout(owner, tmp_path)
        owner.partial.policy_version = lambda: 0
        entered = asyncio.Event()
        calls = 0

        async def transport(request):
            nonlocal calls
            calls += 1
            entered.set()
            await asyncio.Future()

        owner._client = httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(transport)
        )
        task = asyncio.create_task(
            owner.completion({"messages": []}, trace_headers={"x-polar-trace-id": "s:1"})
        )
        await asyncio.wait_for(entered.wait(), 1)
        if phase != "inflight":
            await owner.pause_generation(wait_for_drain=False)
        if phase == "paused":
            await asyncio.wait_for(owner._generation_drained.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        await asyncio.wait_for(owner.resume_generation(), 1)
        assert calls == 1 and owner.generation_status()["drained"]
        assert not owner.partial.priority and not owner.partial.waiting
        await owner.close()

    asyncio.run(run())


def test_completed_attempt_at_pause_is_not_replayed_and_keeps_actual_policy(tmp_path):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine())
        owner.partial = PartialRollout(owner, tmp_path)
        owner.partial.policy_version = lambda: 0
        returned = asyncio.Event()
        calls = 0

        async def transport(request):
            nonlocal calls
            calls += 1
            returned.set()
            return httpx.Response(200, json=completion())

        owner._client = httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(transport)
        )
        task = asyncio.create_task(
            owner.completion({"messages": []}, trace_headers={"x-polar-trace-id": "s:1"})
        )
        await returned.wait()
        await owner.pause_generation(wait_for_drain=False)
        result = await asyncio.wait_for(task, 1)
        storage = SessionStore()
        storage.set_policy_version(1)
        storage.save_message("s", {"messages": []}, result)
        assert storage.get_completions("s")[0]["metadata"]["policy_version"] == 0
        assert storage.session_gen_version("s") == 0
        assert calls == 1 and not owner.partial.priority
        assert not list(tmp_path.glob("*.json"))
        await owner.close()

    asyncio.run(run())


def test_second_update_expires_waiting_call_before_request_is_sent(tmp_path):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine())
        owner.partial = PartialRollout(owner, tmp_path)
        version = 0
        owner.partial.policy_version = lambda: version
        entered = asyncio.Event()
        calls = 0

        async def transport(request):
            nonlocal calls
            calls += 1
            entered.set()
            await asyncio.Future()

        owner._client = httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(transport)
        )
        task = asyncio.create_task(
            owner.completion({"messages": []}, trace_headers={"x-polar-trace-id": "s:1"})
        )
        await entered.wait()
        for version in (1, 2):
            await owner.pause_generation(wait_for_drain=False)
            await asyncio.wait_for(owner._generation_drained.wait(), 1)
            entered.clear()
            await owner.resume_generation()
            if version == 1:
                await entered.wait()
        with pytest.raises(UpstreamError, match="exceeded one"):
            await asyncio.wait_for(task, 1)
        assert calls == 2 and owner.generation_status()["drained"]
        await owner.close()

    asyncio.run(run())


def test_resume_admission_precedes_fresh_calls(tmp_path):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine(), initially_paused=True)
        partial = owner.partial = PartialRollout(owner, tmp_path)
        partial.priority.add("old:1")
        order = []

        async def acquire(key):
            await partial.acquire(key)
            order.append(key)
            partial.release(key)

        fresh = asyncio.create_task(acquire("new:1"))
        await asyncio.sleep(0)
        old = asyncio.create_task(acquire("old:1"))
        await owner.resume_generation()
        await asyncio.gather(fresh, old)
        assert order == ["old:1", "new:1"]

    asyncio.run(run())


def test_parallel_subagent_waits_do_not_double_count_pause(tmp_path):
    async def run():
        owner = InferenceClient("http://engine", VLLMEngine(), initially_paused=True)
        partial = owner.partial = PartialRollout(owner, tmp_path)
        first = asyncio.create_task(partial.acquire("s:first"))
        second = asyncio.create_task(partial.acquire("s:second"))
        await asyncio.sleep(0)
        started = asyncio.get_running_loop().time()
        await asyncio.sleep(0.02)
        elapsed = asyncio.get_running_loop().time() - started
        assert 0.9 * elapsed < partial.pause_seconds("s") < 1.2 * elapsed
        await owner.resume_generation()
        await asyncio.gather(first, second)
        partial.release("s:first")
        partial.release("s:second")
        assert partial.pause_seconds("s") < 1.3 * elapsed

    asyncio.run(run())


def test_completed_retry_reuses_tool_ids_and_saves_once():
    async def run():
        tracker = InflightGenerationTracker(retain_completed=True)
        calls = 0

        async def factory():
            nonlocal calls
            calls += 1
            return {"tool_id": "stable"}

        first = await tracker.run("s", {"messages": []}, factory)
        retry = await tracker.run("s", {"messages": []}, factory)
        assert calls == 1 and first.should_save and not retry.should_save
        assert retry.response is first.response
        await tracker.close_session("s")
        assert tracker.status()["active"] == 0

    asyncio.run(run())


def test_planned_pause_extends_execution_budget_without_freezing_tools():
    async def run():
        manager = object.__new__(GatewayNodeManager)
        paused = 0.0
        manager.partial_rollout = SimpleNamespace(pause_seconds=lambda session: paused)
        managed = SimpleNamespace(
            execution_deadline=asyncio.get_running_loop().time() + 0.02,
            request=SimpleNamespace(session_id="s"),
        )

        async def operation():
            nonlocal paused
            paused = 0.1
            await asyncio.sleep(0.04)
            return "ok"

        assert await manager._await_with_budget(operation(), managed) == "ok"
        # Tool execution that is not waiting behind the generation gate spends budget.
        paused = 0
        with pytest.raises(TimeoutError):
            manager._remaining_budget(managed)

    asyncio.run(run())
