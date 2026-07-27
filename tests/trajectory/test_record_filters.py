"""Shape-only exclusion of non-agent-side completions.

Regression cover for the hardcoded-Triton-handbook whitelist that used to gate
``_is_non_agent_side_completion``: the shape check was right, but only fired when
the payload was one specific document, so the same bare-request shape carrying
cmake sources / judge output / skill frontmatter became a trainable trace. Cases
below mirror shapes observed in real rollout dumps.
"""

from __future__ import annotations

from typing import Any

from polar.trajectory.builder.record_filters import (
    exclude_completion_reason,
    filter_trainable_completions,
)
from polar.trajectory.models import CompletionRecord

_RESPONSE: dict[str, Any] = {
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
}

_CMAKE_BODY = (
    "# ------------------------------------------------------------------\n"
    "# Copyright (c) 2025 Huawei Technologies Co., Ltd.\n"
    "function(ascendc_compile_kernel target_name)\n"
    "    set(CMAKE_CXX_STANDARD 17)\n"
    "endfunction()\n"
)

_TRITON_HANDBOOK = "# Triton Ascend 基础知识参考手册\n\n本文档汇集 Triton Ascend 编程要点。\n"


def _completion(request: dict[str, Any], *, completion_id: str = "c1") -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        request=request,
        response=_RESPONSE,
    )


def _bare_request(content: Any) -> dict[str, Any]:
    """The offending shape: lone user message, no system / tools / SDK fields."""
    return {"messages": [{"role": "user", "content": content}]}


def _agent_request(**extra: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "system": "You are Claude Code, Anthropic's official CLI for Claude.",
        "messages": [{"role": "user", "content": "Implement the 3_Add operator."}],
        "tools": [{"type": "function", "function": {"name": "Read"}}],
        "stream": True,
    }
    request.update(extra)
    return request


# --- the shape is excluded regardless of payload -----------------------------


def test_excludes_bare_request_carrying_cmake_source() -> None:
    """The AscendC-era leak: a Read of function.cmake came back as a bare request."""
    completion = _completion(_bare_request(_CMAKE_BODY))

    assert exclude_completion_reason(completion) == "non_agent_side_completion"


def test_still_excludes_the_triton_handbook_that_motivated_the_old_whitelist() -> None:
    completion = _completion(_bare_request(_TRITON_HANDBOOK))

    assert exclude_completion_reason(completion) == "non_agent_side_completion"


def test_excludes_bare_request_carrying_judge_output() -> None:
    body = "数值验证失败: 测试第 1/3 组输入...\nDumping intermediate results to /root/.triton/dump/abc\n"
    completion = _completion(_bare_request(body))

    assert exclude_completion_reason(completion) == "non_agent_side_completion"


def test_excludes_bare_request_with_structured_content_blocks() -> None:
    completion = _completion(_bare_request([{"type": "text", "text": _CMAKE_BODY}]))

    assert exclude_completion_reason(completion) == "non_agent_side_completion"


def test_shape_check_reads_original_request_when_present() -> None:
    completion = CompletionRecord(
        completion_id="c1",
        original_request=_bare_request(_CMAKE_BODY),
        request=_agent_request(),
        response=_RESPONSE,
    )

    assert exclude_completion_reason(completion) == "non_agent_side_completion"


# --- agent-side traffic must survive -----------------------------------------


def test_keeps_normal_agent_request() -> None:
    assert exclude_completion_reason(_completion(_agent_request())) is None


def test_keeps_single_user_message_when_system_prompt_is_present() -> None:
    """A subagent's first turn: one user message, but a system prompt is set."""
    request = {
        "system": "You are a file search specialist. READ-ONLY MODE.",
        "messages": [{"role": "user", "content": "Locate kernel/op_host/*.cpp"}],
    }

    assert exclude_completion_reason(_completion(request)) is None


def test_keeps_single_user_message_when_system_is_inline() -> None:
    request = {
        "messages": [{"role": "system", "content": "You are Claude Code."}],
    }

    assert exclude_completion_reason(_completion(request)) is None


def test_keeps_request_whose_only_agent_signal_is_tools() -> None:
    request = {
        "messages": [{"role": "user", "content": _CMAKE_BODY}],
        "tools": [{"type": "function", "function": {"name": "Bash"}}],
    }

    assert exclude_completion_reason(_completion(request)) is None


def test_keeps_request_whose_only_agent_signal_is_an_sdk_field() -> None:
    for field, value in (
        ("thinking", {"type": "enabled"}),
        ("stream", True),
        ("context_management", {"edits": []}),
        ("output_config", {"format": "text"}),
    ):
        request = {"messages": [{"role": "user", "content": _CMAKE_BODY}], field: value}

        assert exclude_completion_reason(_completion(request)) is None, field


def test_keeps_resumed_agent_turn_with_no_tool_calls() -> None:
    """Zero tool calls is not a signal: real trailing turns wrap up in prose.

    Mirrors the legit short traces seen in dumps — a final segment woken by a
    task-notification or an output-token-limit resume, which answers in prose
    and calls nothing. Excluding on tool-call count would drop these.
    """
    request = _agent_request(
        messages=[
            {"role": "user", "content": "Implement the operator."},
            {"role": "assistant", "content": "Generation phase exhausted (5/5)."},
            {"role": "user", "content": "<task-notification>...</task-notification>"},
        ],
    )

    assert exclude_completion_reason(_completion(request)) is None


def test_keeps_multi_message_history_without_system_prompt() -> None:
    request = {
        "messages": [
            {"role": "user", "content": "Fix the CopyOut bug."},
            {"role": "assistant", "content": "Reading the kernel."},
        ],
    }

    assert exclude_completion_reason(_completion(request)) is None


# --- precedence and bookkeeping ----------------------------------------------


def test_empty_completion_takes_precedence_over_shape() -> None:
    completion = CompletionRecord(
        completion_id="c1",
        request=_bare_request(_CMAKE_BODY),
        response={"choices": []},
    )

    assert exclude_completion_reason(completion) == "empty_completion"


def test_missing_request_is_not_excluded() -> None:
    assert exclude_completion_reason(_completion({})) is None


def test_filter_records_excluded_ids_and_reasons() -> None:
    kept_one = _completion(_agent_request(), completion_id="keep-1")
    dropped = _completion(_bare_request(_CMAKE_BODY), completion_id="drop-1")
    kept_two = _completion(_agent_request(), completion_id="keep-2")

    result = filter_trainable_completions([kept_one, dropped, kept_two])

    assert [item.completion_id for item in result.kept] == ["keep-1", "keep-2"]
    assert result.metadata["input_completions"] == 3
    assert result.metadata["kept_completions"] == 2
    assert result.metadata["excluded_completions"] == 1
    assert result.metadata["excluded_reasons"] == {"non_agent_side_completion": 1}
    assert result.metadata["excluded_completion_ids"] == ["drop-1"]
