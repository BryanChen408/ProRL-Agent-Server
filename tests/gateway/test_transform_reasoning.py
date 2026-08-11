"""Round-trip tests for reasoning_content across all four transformers.

Each test:
1. Builds a SGLang chat completion with `reasoning_content`.
2. Calls transform_response → expects API-specific reasoning block emitted.
3. Feeds the harness-shaped output back as input via transform_request.
4. Verifies reasoning_content survives the round trip.
"""

from __future__ import annotations

from polar.gateway.transform.anthropic import (
    AnthropicStreamState,
    AnthropicTransformer,
)
from polar.gateway.transform.google import GoogleTransformer
from polar.gateway.transform.openai_chat import OpenAIChatTransformer
from polar.gateway.transform.openai_responses import (
    OpenAIResponsesTransformer,
    ResponsesStreamState,
)

REASONING_TEXT = "Let me think step by step. 1+1 must equal 2."
ANSWER_TEXT = "The answer is 2."


def _sglang_response(*, with_tool_call: bool = False) -> dict:
    message: dict = {
        "role": "assistant",
        "content": ANSWER_TEXT,
        "reasoning_content": REASONING_TEXT,
    }
    if with_tool_call:
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "answer", "arguments": '{"x": 2}'},
            }
        ]
    return {
        "id": "chatcmpl-1",
        "model": "MiniMax-M2.5",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


# ---------- OpenAI Chat: pure passthrough ----------


def test_openai_chat_preserves_reasoning_content_on_response() -> None:
    t = OpenAIChatTransformer()
    out = t.transform_response(_sglang_response(), {"model": "anything"})
    assert out["choices"][0]["message"]["reasoning_content"] == REASONING_TEXT


def test_openai_chat_preserves_reasoning_content_on_request() -> None:
    t = OpenAIChatTransformer()
    req = {
        "messages": [
            {"role": "user", "content": "compute"},
            {
                "role": "assistant",
                "content": ANSWER_TEXT,
                "reasoning_content": REASONING_TEXT,
            },
            {"role": "user", "content": "explain"},
        ],
    }
    out = t.transform_request(req)
    assert out["messages"][1]["reasoning_content"] == REASONING_TEXT


# ---------- Anthropic: emit thinking blocks, ingest them back ----------


def test_anthropic_response_emits_thinking_block_before_text() -> None:
    t = AnthropicTransformer()
    out = t.transform_response(_sglang_response(), {"model": "claude-3"})
    blocks = out["content"]
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == REASONING_TEXT
    assert blocks[0]["signature"]  # signature populated
    assert any(b.get("type") == "text" for b in blocks)


def test_anthropic_response_emits_thinking_with_tool_use() -> None:
    t = AnthropicTransformer()
    out = t.transform_response(
        _sglang_response(with_tool_call=True), {"model": "claude-3"}
    )
    types = [b["type"] for b in out["content"]]
    assert types[0] == "thinking"
    assert "tool_use" in types


def test_anthropic_request_recovers_reasoning_from_thinking_block() -> None:
    t = AnthropicTransformer()
    req = {
        "_polar_model_served": "MiniMax-M2.5",
        "messages": [
            {"role": "user", "content": "compute 1+1"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": REASONING_TEXT,
                        "signature": "sg_polar_xxx",
                    },
                    {"type": "text", "text": ANSWER_TEXT},
                ],
            },
            {"role": "user", "content": "ok next"},
        ],
        "max_tokens": 100,
    }
    out = t.transform_request(req)
    # Find the assistant message
    assistant_msg = next(m for m in out["messages"] if m.get("role") == "assistant")
    assert assistant_msg["reasoning_content"] == REASONING_TEXT


def test_anthropic_thinking_request_param_enables_thinking() -> None:
    t = AnthropicTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is True


def test_anthropic_streaming_emits_thinking_signature() -> None:
    state = AnthropicStreamState(
        model="claude-3", finish_to_stop_reason=AnthropicTransformer.FINISH_TO_STOP_REASON
    )
    chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": ANSWER_TEXT,
                    "reasoning_content": REASONING_TEXT,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 7},
    }
    events = state.process_chunk(chunk, is_first=True) + state.finalize()
    types = [e.get("type") for e in events]
    deltas = [e.get("delta", {}).get("type") for e in events if e.get("type") == "content_block_delta"]
    assert "content_block_start" in types
    assert "thinking_delta" in deltas
    assert "signature_delta" in deltas
    assert "text_delta" in deltas


# ---------- Gemini: emit thought parts, ingest them back ----------


def test_gemini_response_emits_thought_part_before_text() -> None:
    t = GoogleTransformer()
    out = t.transform_response(_sglang_response(), {})
    parts = out["candidates"][0]["content"]["parts"]
    assert parts[0].get("thought") is True
    assert parts[0]["text"] == REASONING_TEXT
    assert parts[0]["thoughtSignature"]
    # User-facing text must NOT include reasoning text.
    visible_text = next(
        p for p in parts[1:] if isinstance(p, dict) and "text" in p and not p.get("thought")
    )
    assert visible_text["text"] == ANSWER_TEXT


def test_gemini_request_recovers_reasoning_from_thought_part() -> None:
    t = GoogleTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "contents": [
                {"role": "user", "parts": [{"text": "compute"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "thought": True,
                            "text": REASONING_TEXT,
                            "thoughtSignature": "sg_polar_xxx",
                        },
                        {"text": ANSWER_TEXT},
                    ],
                },
                {"role": "user", "parts": [{"text": "ok"}]},
            ],
        }
    )
    assistant_msg = next(m for m in out["messages"] if m.get("role") == "assistant")
    assert assistant_msg["reasoning_content"] == REASONING_TEXT
    # Visible content must not leak the reasoning text.
    assert REASONING_TEXT not in (assistant_msg.get("content") or "")


def test_gemini_thinking_config_enables_thinking() -> None:
    t = GoogleTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"thinkingConfig": {"includeThoughts": True}},
        }
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is True


# ---------- Responses: emit reasoning output items, ingest them back ----------


def test_responses_response_emits_reasoning_item_first() -> None:
    t = OpenAIResponsesTransformer()
    out = t.transform_response(_sglang_response(), {"model": "gpt-5.4"})
    items = out["output"]
    assert items[0]["type"] == "reasoning"
    assert items[0]["summary"][0]["text"] == REASONING_TEXT
    assert items[0]["content"][0]["text"] == REASONING_TEXT
    assert items[0]["encrypted_content"]
    assert items[1]["type"] == "message"


def test_responses_request_recovers_reasoning_attached_to_assistant() -> None:
    t = OpenAIResponsesTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "input": [
                {"type": "message", "role": "user", "content": "compute"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": REASONING_TEXT}],
                    "content": [{"type": "reasoning_text", "text": REASONING_TEXT}],
                    "encrypted_content": "polar:LWE=",
                },
                {"type": "message", "role": "assistant", "content": ANSWER_TEXT},
                {"type": "message", "role": "user", "content": "explain"},
            ],
        }
    )
    assistant_msg = next(m for m in out["messages"] if m.get("role") == "assistant")
    assert assistant_msg["reasoning_content"] == REASONING_TEXT


def test_responses_request_attaches_reasoning_to_function_call() -> None:
    t = OpenAIResponsesTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "input": [
                {"type": "message", "role": "user", "content": "use tool"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": REASONING_TEXT}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "answer",
                    "arguments": '{"x": 2}',
                },
            ],
        }
    )
    assistant_msg = next(m for m in out["messages"] if m.get("role") == "assistant")
    assert assistant_msg.get("reasoning_content") == REASONING_TEXT
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "answer"


def test_responses_reasoning_request_param_enables_thinking() -> None:
    t = OpenAIResponsesTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "input": "compute",
            "reasoning": {"effort": "medium"},
        }
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is True


def test_responses_streaming_emits_reasoning_item_events() -> None:
    state = ResponsesStreamState(model="gpt-5.4")
    chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": ANSWER_TEXT,
                    "reasoning_content": REASONING_TEXT,
                },
                "finish_reason": "stop",
            }
        ]
    }
    events = state.process_chunk(chunk, is_first=True) + state.finalize()
    types = [e.get("type") for e in events]
    assert "response.reasoning_summary_text.delta" in types
    assert "response.reasoning_summary_text.done" in types
    # output_item.added/done for reasoning
    added_items = [
        e["item"]["type"]
        for e in events
        if e.get("type") == "response.output_item.added"
    ]
    assert "reasoning" in added_items
    assert "message" in added_items
    # Final response.completed includes both items.
    completed = next(e for e in events if e.get("type") == "response.completed")
    output_types = [it["type"] for it in completed["response"]["output"]]
    assert output_types[0] == "reasoning"
    assert "message" in output_types


def test_anthropic_thinking_also_sets_preserve_thinking(monkeypatch) -> None:
    # preserve_thinking 默认开:历史 assistant 轮的 <think> 块在 chat template 渲染时
    # 保留 → user 注入(截断 resume/skill)不再引起拆链,训推一致。
    monkeypatch.delenv("POLAR_PRESERVE_THINKING", raising=False)
    t = AnthropicTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "Qwen3.6",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert out["chat_template_kwargs"]["preserve_thinking"] is True


def test_anthropic_preserve_thinking_gate_off(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_PRESERVE_THINKING", "0")
    t = AnthropicTransformer()
    out = t.transform_request(
        {
            "_polar_model_served": "Qwen3.6",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert "preserve_thinking" not in out["chat_template_kwargs"]
