from __future__ import annotations

from polar.gateway.transform.anthropic import AnthropicTransformer

IMAGE_B64 = "abc123"
IMAGE_URL = f"data:image/png;base64,{IMAGE_B64}"


def test_anthropic_request_maps_all_fields_and_image_input_to_chat() -> None:
    transformer = AnthropicTransformer()

    transformed = transformer.transform_request(
        {
            "_polar_model_served": "Qwen/Qwen3.5-4B",
            "system": "x-anthropic-billing-header: cch=unstable;\nBe direct.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Count stars."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": IMAGE_B64,
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will call a tool."},
                        {
                            "type": "tool_use",
                            "id": "toolu-1",
                            "name": "write_answer",
                            "input": {"answer": 2},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu-1",
                            "content": [
                                {"type": "text", "text": "ok"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": IMAGE_B64,
                                    },
                                },
                            ],
                        },
                        {"type": "text", "text": "Thanks."},
                    ],
                },
            ],
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
            "stream": True,
            "tools": [
                {
                    "name": "write_answer",
                    "description": "Write the answer",
                    "input_schema": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "tool", "name": "write_answer"},
        }
    )

    assert transformed["messages"] == [
        {"role": "system", "content": "Be direct."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Count stars."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        },
        {
            "role": "assistant",
            "content": "I will call a tool.",
            "tool_calls": [
                {
                    "id": "toolu-1",
                    "type": "function",
                    "function": {
                        "name": "write_answer",
                        "arguments": '{"answer": 2}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu-1", "content": "ok"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE_URL}}]},
        {"role": "user", "content": "Thanks."},
    ]
    assert transformed["max_tokens"] == 128
    assert transformed["temperature"] == 0.2
    assert transformed["top_p"] == 0.9
    assert transformed["top_k"] == 40
    assert transformed["stop"] == ["END"]
    assert transformed["stream"] is True
    assert transformed["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "write_answer",
                "description": "Write the answer",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert transformed["tool_choice"] == {
        "type": "function",
        "function": {"name": "write_answer"},
    }
    assert transformed["chat_template_kwargs"]["enable_thinking"] is False


def test_anthropic_request_uses_configurable_default_max_tokens(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_ANTHROPIC_DEFAULT_MAX_TOKENS", "32768")
    transformer = AnthropicTransformer()

    transformed = transformer.transform_request(
        {"messages": [{"role": "user", "content": "hello"}]}
    )

    assert transformed["max_tokens"] == 32768


def test_anthropic_request_maps_multi_turn_reasoning_and_parallel_tools() -> None:
    transformer = AnthropicTransformer()

    transformed = transformer.transform_request(
        {
            "messages": [
                {"role": "user", "content": "Plan and call tools."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Need two independent lookups.",
                            "signature": "sig-1",
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu-a",
                            "name": "lookup",
                            "input": {"q": "a"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu-b",
                            "name": "lookup",
                            "input": {"q": "b"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu-a",
                            "content": "A",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu-b",
                            "content": [{"type": "text", "text": "B"}],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Combine both results.",
                            "signature": "sig-2",
                        },
                        {"type": "text", "text": "A and B"},
                    ],
                },
            ],
            "max_tokens": 256,
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "Plan and call tools."},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Need two independent lookups.",
            "tool_calls": [
                {
                    "id": "toolu-a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "a"}'},
                },
                {
                    "id": "toolu-b",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "b"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "toolu-a", "content": "A"},
        {"role": "tool", "tool_call_id": "toolu-b", "content": "B"},
        {
            "role": "assistant",
            "content": "A and B",
            "reasoning_content": "Combine both results.",
        },
    ]


def test_anthropic_tool_choice_variants_map_to_openai() -> None:
    transformer = AnthropicTransformer()
    base_body = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
    }

    assert (
        transformer.transform_request({**base_body, "tool_choice": {"type": "auto"}})[
            "tool_choice"
        ]
        == "auto"
    )
    assert (
        transformer.transform_request({**base_body, "tool_choice": {"type": "any"}})[
            "tool_choice"
        ]
        == "required"
    )
    assert (
        transformer.transform_request({**base_body, "tool_choice": {"type": "none"}})[
            "tool_choice"
        ]
        == "none"
    )
    assert transformer.transform_request(
        {**base_body, "tool_choice": {"type": "tool", "name": "lookup"}}
    )["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


def test_anthropic_adaptive_thinking_request_param_enables_thinking() -> None:
    transformer = AnthropicTransformer()

    transformed = transformer.transform_request(
        {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 2048,
        }
    )

    assert transformed["chat_template_kwargs"]["enable_thinking"] is True


def test_anthropic_response_maps_openai_content_and_usage_back() -> None:
    transformer = AnthropicTransformer()

    response = transformer.transform_response(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "There are two."},
                            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                        ],
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "write_answer", "arguments": '{"answer": 2}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
        {"model": "claude-test"},
    )

    assert response["id"] == "msg_chatcmpl-1"
    assert response["type"] == "message"
    assert response["role"] == "assistant"
    assert response["model"] == "claude-test"
    assert response["stop_reason"] == "tool_use"
    assert response["usage"] == {"input_tokens": 10, "output_tokens": 4}
    assert response["content"] == [
        {"type": "text", "text": "There are two."},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": IMAGE_B64},
        },
        {
            "type": "tool_use",
            "id": "call-1",
            "name": "write_answer",
            "input": {"answer": 2},
        },
    ]


def test_anthropic_response_preserves_cached_usage_tokens() -> None:
    transformer = AnthropicTransformer()

    response = transformer.transform_response(
        {
            "choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        },
        {"model": "claude-test"},
    )

    assert response["usage"] == {
        "input_tokens": 6,
        "output_tokens": 3,
        "cache_read_input_tokens": 4,
    }


def test_anthropic_response_skips_empty_openai_content_with_tool_call() -> None:
    transformer = AnthropicTransformer()

    response = transformer.transform_response(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "content": [],
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "view_image", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
        {"model": "claude-test"},
    )

    assert response["content"] == [
        {
            "type": "tool_use",
            "id": "call-1",
            "name": "view_image",
            "input": {},
        }
    ]


def test_anthropic_stream_state_emits_ordered_text_tool_and_usage_events() -> None:
    transformer = AnthropicTransformer()
    state = transformer.create_stream_state({"model": "claude-test"})

    events = state.process_chunk(
        {
            "choices": [{"delta": {"content": "Hi"}}],
            "usage": {"completion_tokens": 1},
        },
        is_first=True,
    )
    events.extend(
        state.process_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "write_answer", "arguments": '{"answer"'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    events.extend(
        state.process_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ": 2}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"completion_tokens": 3},
            }
        )
    )
    events.extend(state.finalize())

    event_types = [event["type"] for event in events]
    assert event_types[0] == "message_start"
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert events[-2] == {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 3},
    }
    assert events[-1] == {"type": "message_stop"}


def test_anthropic_request_handles_url_image_source() -> None:
    transformer = AnthropicTransformer()
    transformed = transformer.transform_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.test/cat.png"},
                        },
                    ],
                }
            ],
            "max_tokens": 16,
        }
    )

    assert transformed["messages"][0]["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.test/cat.png"}},
    ]


def test_anthropic_request_handles_document_text_source() -> None:
    transformer = AnthropicTransformer()
    transformed = transformer.transform_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Doc body.",
                            },
                        },
                        {"type": "text", "text": "Summarize."},
                    ],
                }
            ],
            "max_tokens": 16,
        }
    )

    # Document text flattens into the user message text content.
    assert transformed["messages"][0]["content"] == "Doc body.\nSummarize."


def test_anthropic_request_handles_document_content_source() -> None:
    transformer = AnthropicTransformer()
    transformed = transformer.transform_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "content",
                                "content": [
                                    {"type": "text", "text": "Page 1"},
                                    {"type": "text", "text": "Page 2"},
                                ],
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 16,
        }
    )

    assert transformed["messages"][0]["content"] == "Page 1\nPage 2"


def test_anthropic_request_drops_base64_pdf_documents() -> None:
    transformer = AnthropicTransformer()
    transformed = transformer.transform_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQK",
                            },
                        },
                        {"type": "text", "text": "What's in the PDF?"},
                    ],
                }
            ],
            "max_tokens": 16,
        }
    )

    # Binary PDFs can't be rendered to the chat template; drop the document
    # and forward the surrounding text so the model still sees the question.
    assert transformed["messages"][0]["content"] == "What's in the PDF?"


def test_anthropic_tool_result_is_error_marks_content() -> None:
    transformer = AnthropicTransformer()
    transformed = transformer.transform_request(
        {
            "messages": [
                {"role": "user", "content": "Run a command."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu-1",
                            "name": "shell",
                            "input": {"cmd": "fail"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu-1",
                            "content": "Permission denied",
                            "is_error": True,
                        },
                    ],
                },
            ],
            "max_tokens": 16,
        }
    )

    tool_msg = next(m for m in transformed["messages"] if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "toolu-1"
    assert tool_msg["content"].startswith("[Tool Error]")
    assert "Permission denied" in tool_msg["content"]


def test_anthropic_response_maps_extended_stop_reasons() -> None:
    transformer = AnthropicTransformer()

    refusal = transformer.transform_response(
        {
            "choices": [
                {"message": {"content": "blocked"}, "finish_reason": "content_filter"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {"model": "claude-3"},
    )
    assert refusal["stop_reason"] == "refusal"

    stop_seq = transformer.transform_response(
        {
            "choices": [
                {"message": {"content": "x"}, "finish_reason": "stop_sequence"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        {"model": "claude-3"},
    )
    assert stop_seq["stop_reason"] == "stop_sequence"


def test_anthropic_request_drops_server_side_tools() -> None:
    transformer = AnthropicTransformer()
    transformed = transformer.transform_request(
        {
            "messages": [{"role": "user", "content": "do it"}],
            "max_tokens": 16,
            "tools": [
                {"name": "lookup", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "code_execution_20250522", "name": "code_execution"},
            ],
        }
    )

    # Only the custom function tool reaches SGLang; server-side tools are
    # dropped because Polar can't execute them.
    assert transformed["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "",
                "parameters": {"type": "object"},
            },
        },
    ]


def test_anthropic_request_system_list_with_cache_control_annotations() -> None:
    transformer = AnthropicTransformer()
    transformed = transformer.transform_request(
        {
            "system": [
                {
                    "type": "text",
                    "text": "You are a careful assistant.",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Always cite sources."},
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        }
    )

    # cache_control is dropped silently; both text blocks are joined into one
    # system message (Anthropic supports per-block cache markers; SGLang does
    # not, so we forward only the prompt text).
    assert transformed["messages"][0] == {
        "role": "system",
        "content": "You are a careful assistant.\nAlways cite sources.",
    }


def test_anthropic_stream_state_emits_parallel_tool_use_blocks() -> None:
    transformer = AnthropicTransformer()
    state = transformer.create_stream_state({"model": "claude-test"})

    # Both tools opened in a single chunk; arguments split across two chunks.
    events = state.process_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu-a",
                                "function": {"name": "lookup_a", "arguments": '{"q":'},
                            },
                            {
                                "index": 1,
                                "id": "toolu-b",
                                "function": {"name": "lookup_b", "arguments": '{"q":'},
                            },
                        ]
                    }
                }
            ]
        },
        is_first=True,
    )
    events.extend(
        state.process_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"a"}'}},
                                {"index": 1, "function": {"arguments": '"b"}'}},
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"completion_tokens": 5},
            }
        )
    )
    events.extend(state.finalize())

    # Two distinct content blocks: index 0 carries toolu-a, index 1 carries toolu-b.
    starts = [
        event for event in events if event["type"] == "content_block_start"
    ]
    assert len(starts) == 2
    assert (starts[0]["index"], starts[0]["content_block"]["id"], starts[0]["content_block"]["name"]) == (
        0,
        "toolu-a",
        "lookup_a",
    )
    assert (starts[1]["index"], starts[1]["content_block"]["id"], starts[1]["content_block"]["name"]) == (
        1,
        "toolu-b",
        "lookup_b",
    )

    # Per-index argument deltas land on the right block.
    deltas_by_index: dict[int, list[str]] = {}
    for event in events:
        if event["type"] == "content_block_delta" and event["delta"].get(
            "type"
        ) == "input_json_delta":
            deltas_by_index.setdefault(event["index"], []).append(
                event["delta"]["partial_json"]
            )
    assert "".join(deltas_by_index[0]) == '{"q":"a"}'
    assert "".join(deltas_by_index[1]) == '{"q":"b"}'

    # Both blocks are explicitly closed before the final message_delta.
    stops = [
        event["index"]
        for event in events
        if event["type"] == "content_block_stop"
    ]
    assert stops == [0, 1]
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    assert events[-1] == {"type": "message_stop"}


def test_anthropic_stream_state_closes_thinking_before_tool_use() -> None:
    transformer = AnthropicTransformer()
    state = transformer.create_stream_state({"model": "claude-test"})

    events = state.process_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "I should call a tool.",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu-1",
                                "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"completion_tokens": 3},
        },
        is_first=True,
    )
    events.extend(state.finalize())

    indexed_events = [
        (event["type"], event.get("index"), event.get("delta", {}).get("type"))
        for event in events
        if event["type"].startswith("content_block")
    ]
    thinking_stop_pos = indexed_events.index(("content_block_stop", 0, None))
    tool_start_pos = next(
        i
        for i, event in enumerate(indexed_events)
        if event == ("content_block_start", 1, None)
    )

    assert indexed_events[:4] == [
        ("content_block_start", 0, None),
        ("content_block_delta", 0, "thinking_delta"),
        ("content_block_delta", 0, "signature_delta"),
        ("content_block_stop", 0, None),
    ]
    assert thinking_stop_pos < tool_start_pos
