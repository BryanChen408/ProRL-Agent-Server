from __future__ import annotations

import pytest

from polar.rollout.agent_actions import (
    classify_shell_command,
    extract_completed_agent_actions,
    is_agent_runner_command,
)
from polar.rollout.models import SessionTiming
from polar.rollout.timer import StageTimer
from polar.rollout.trace_exporter import build_chrome_trace_document, export_chrome_trace


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rg -n TODO src | head", "grep"),
        ("grep -R error logs", "grep"),
        ("find . -name '*.py'", "find"),
        ("bash -lc 'python3 tools/check.py'", "python"),
        ("bash tools/ascendc_eval_pipeline.sh --op-name foo", "eval"),
        ("cmake -S . -B build && ninja -C build", "build"),
        ("git status --short", "git"),
        ("pip install pydantic", "package"),
        ("curl -fsSL https://example.test", "network"),
        ("mkdir -p output && cp a output/a", "fs"),
        ("sed -i 's/a/b/' file", "edit"),
        ("cat README.md", "read"),
        ("cat README.md 2>/dev/null | head", "read"),
        ("cat > app.py <<'PY'\nimport requests\nPY", "write"),
        ("# run inline code\npython3 - <<'PY'\nprint('x')\nPY", "python"),
        ("echo ready", "shell"),
    ],
)
def test_classify_shell_command(command: str, expected: str) -> None:
    assert classify_shell_command(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "claude --output-format=stream-json -p task",
        'export GEMINI_API_KEY="$GOOGLE_API_KEY" && gemini --prompt=task',
        "mkdir -p /tmp/config && codex exec -- task",
        "mkdir -p ~/.openclaw && openclaw agent --message task",
        (
            'PYTHON_BIN="/opt/openhands-sdk-venv/bin/python"; "$PYTHON_BIN" /app/run.py '
            "2>&1 | tee /polar/session/logs/agent/openhands-sdk.txt"
        ),
    ],
)
def test_detect_outer_agent_runner(command: str) -> None:
    assert is_agent_runner_command(command)


def test_extract_parallel_anthropic_actions_in_tool_result_order() -> None:
    request = {
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": "Read",
                        "input": {"file_path": "/workspace/src/app.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "bash-1",
                        "name": "Bash",
                        "input": {
                            "command": "rg -n TODO src",
                            "description": "Find TODOs",
                        },
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "bash-1", "content": "x"},
                    {"type": "tool_result", "tool_use_id": "read-1", "content": "y"},
                ],
            },
        ]
    }

    actions = extract_completed_agent_actions(request)

    assert [action["kind"] for action in actions] == ["grep", "read"]
    assert actions[0]["command"] == "rg -n TODO src"
    assert actions[1]["target"] == "app.py"
    assert all("content" not in action for action in actions)


def test_extract_openai_tool_action() -> None:
    request = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command":"python3 verify.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ]
    }

    assert extract_completed_agent_actions(request)[0]["kind"] == "eval"


def test_stage_timer_allocates_parallel_gap_and_marks_estimate() -> None:
    timer = StageTimer()
    timer.record_llm_call(sglang_wait_ms=10)
    timer.patch_last_llm_agent_side_gap(
        gap_ms=100,
        actions=[
            {"kind": "read", "summary": "a.py"},
            {"kind": "grep", "summary": "rg TODO"},
            {"kind": "write", "summary": "b.py"},
        ],
    )

    actions = timer._llm_calls[-1]["agent_actions"]
    assert sum(action["duration_ms"] for action in actions) == 100
    assert [action["duration_ms"] for action in actions] == [33.33, 33.33, 33.34]
    assert all(action["duration_estimated"] for action in actions)
    assert all(action["parallel"] for action in actions)


def test_stage_timer_uses_think_for_unattributed_gap() -> None:
    timer = StageTimer()
    timer.record_llm_call(sglang_wait_ms=10)
    timer.patch_last_llm_agent_side_gap(gap_ms=12.5)

    assert timer._llm_calls[-1]["agent_actions"] == [
        {
            "kind": "think",
            "tool_name": "",
            "tool_use_id": "",
            "summary": "agent think",
            "is_error": False,
            "duration_ms": 12.5,
            "duration_estimated": True,
            "gap_allocation": "whole_gap",
            "parallel": False,
        }
    ]


def test_trace_uses_classified_actions_and_hides_outer_agent_runner() -> None:
    timing = SessionTiming(
        run_agent_exec_ms=200,
        llm_calls=[
            {"round": 1, "sglang_wait_ms": 10, "agent_side_gap_ms": 0},
            {
                "round": 2,
                "sglang_wait_ms": 20,
                "agent_side_gap_ms": 90,
                "agent_actions": [
                    {
                        "kind": "grep",
                        "tool_name": "Bash",
                        "tool_use_id": "tool-1",
                        "summary": "rg -n TODO src",
                        "command": "rg -n TODO src",
                        "duration_ms": 90,
                        "duration_estimated": True,
                        "gap_allocation": "whole_gap",
                        "parallel": False,
                    }
                ],
            },
        ],
        tool_execs=[
            {
                "idx": 0,
                "command": "mkdir -p /polar/session/.claude",
                "duration_ms": 5,
                "exit_code": 0,
            },
            {
                "idx": 1,
                "command": "claude --output-format=stream-json -p task",
                "duration_ms": 200,
                "exit_code": 0,
            }
        ],
    )

    events = export_chrome_trace(timing, session_id="session-1")
    spans = [event for event in events if event.get("ph") == "X"]
    names = [event["name"] for event in spans]

    assert "llm_call_2/tool/grep/00: rg -n TODO src" in names
    assert not any("claude --output-format" in name for name in names)
    assert not any("mkdir -p /polar/session/.claude" in name for name in names)
    grep_span = next(event for event in spans if "/tool/grep/" in event["name"])
    assert grep_span["args"]["action_kind"] == "grep"
    assert grep_span["args"]["duration_estimated"] is True


def test_v2_trace_preserves_measured_clock_positions() -> None:
    timer = StageTimer()
    timer._clock_monotonic_origin = 10.0
    timer._clock_epoch_origin_ns = 1_700_000_000_000_000_000
    timer._marks.update({
        "dispatch_started": 10.0,
        "init_started": 10.125,
        "init_finished": 10.375,
        "run_started": 10.5,
        "run_finished": 11.0,
        "return_finished": 11.2,
    })
    timer.record_llm_call(
        sglang_wait_ms=100,
        trace_timing={
            "request_started_at_ns": 1_700_000_000_600_000_000,
            "sglang_started_at_ns": 1_700_000_000_650_000_000,
            "sglang_finished_at_ns": 1_700_000_000_750_000_000,
            "response_finished_at_ns": 1_700_000_000_800_000_000,
        },
        trace_id="session-1:1",
        engine_name="vllm",
        engine_url="http://engine:8000",
        engine_metrics={
            "queue_ms": 10,
            "prefill_ms": 40,
            "decode_ms": 50,
            "num_cached_tokens": 8,
            "prefix_cache_hit_pct": 80,
        },
    )

    timing = timer.to_session_timing()
    document = build_chrome_trace_document(timing, session_id="session-1")
    spans = [event for event in document["traceEvents"] if event.get("ph") == "X"]

    assert document["schemaVersion"] == 2
    assert document["metadata"]["traceStartTimeNs"] == 1_700_000_000_000_000_000
    queue = next(event for event in spans if event["name"] == "register_to_init_queue")
    assert queue["ts"] == 1_700_000_000_000_000
    assert queue["dur"] == 125_000
    inference = next(event for event in spans if event["name"] == "llm_call_1/inference")
    assert inference["ts"] == 1_700_000_000_650_000
    assert inference["dur"] == 100_000
    assert inference["args"]["measured"] is True
    assert inference["args"]["engine"] == "vllm"
    assert inference["args"]["engine_url"] == "http://engine:8000"
    prefill = next(event for event in spans if event["name"] == "llm_call_1/engine/prefill")
    decode = next(event for event in spans if event["name"] == "llm_call_1/engine/decode")
    assert prefill["dur"] == 40_000
    assert prefill["args"]["position_derived"] is True
    assert decode["args"]["num_cached_tokens"] == 8
