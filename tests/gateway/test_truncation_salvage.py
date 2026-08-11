"""截断续命注入(_salvage_message_for)的单测。

只在「空截断」(finish=length + content 空 + 无 tool_calls)时注入残稿;
其余一律不碰。
"""

from __future__ import annotations

from polar.gateway.server import _salvage_message_for


def _completion(finish_reason, content=None, tool_calls=None, reasoning=""):
    return {
        "response": {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                        "reasoning_content": reasoning,
                    },
                }
            ]
        }
    }


def test_no_completions_no_injection() -> None:
    assert _salvage_message_for([]) is None


def test_normal_turn_no_injection() -> None:
    comps = [_completion("tool_calls", content=None, tool_calls=[{"function": {"name": "Bash"}}])]
    assert _salvage_message_for(comps) is None


def test_stop_turn_no_injection() -> None:
    comps = [_completion("stop", content="做完了")]
    assert _salvage_message_for(comps) is None


def test_truncation_with_tool_call_no_injection() -> None:
    # 截断但已发出动作(半截 tool call 场景)——有产出,不干预
    comps = [_completion("length", content=None, tool_calls=[{"function": {"name": "Write"}}])]
    assert _salvage_message_for(comps) is None


def test_truncation_with_content_no_injection() -> None:
    comps = [_completion("length", content="半截正文")]
    assert _salvage_message_for(comps) is None


def test_empty_truncation_injects_full_draft() -> None:
    draft = "OK, let me finalize the code. I'll rewrite the op_kernel file..." * 10
    comps = [_completion("length", content=None, tool_calls=None, reasoning=draft)]
    msg = _salvage_message_for(comps)
    assert msg is not None
    assert msg["role"] == "user"
    assert "被截断" in msg["content"]
    assert draft in msg["content"]            # 全稿注入
    assert "小步 tool call" in msg["content"]


def test_empty_truncation_without_reasoning_uses_fallback_prompt() -> None:
    comps = [_completion("length", content=None, tool_calls=None, reasoning="")]
    msg = _salvage_message_for(comps)
    assert msg is not None
    assert "被截断" in msg["content"]
    assert "「" not in msg["content"]          # 无残稿引用块


def test_uses_last_completion_only() -> None:
    comps = [
        _completion("length", content=None, reasoning="old draft"),
        _completion("tool_calls", content=None, tool_calls=[{"function": {"name": "Bash"}}]),
    ]
    assert _salvage_message_for(comps) is None
