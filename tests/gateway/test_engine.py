from __future__ import annotations

import pytest

from polar.gateway.engine import SGLangEngine, VLLMEngine, get_engine


def test_get_engine_returns_the_right_strategy() -> None:
    assert isinstance(get_engine("sglang"), SGLangEngine)
    assert isinstance(get_engine("vllm"), VLLMEngine)


def test_get_engine_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown inference engine"):
        get_engine("tgi")


def test_sglang_engine_requests_logprobs_and_passes_through() -> None:
    engine = SGLangEngine()
    request = {"messages": []}
    out = engine.prepare_request(request)
    assert out is request and out["logprobs"] is True
    response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    assert engine.normalize_response(response) is response


def test_vllm_prepare_request_requests_token_ids_and_logprobs() -> None:
    out = VLLMEngine().prepare_request({"messages": [], "logprobs": True})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_vllm_prepare_request_keeps_explicit_top_logprobs() -> None:
    out = VLLMEngine().prepare_request({"logprobs": True, "top_logprobs": 5})
    assert out["top_logprobs"] == 5


def test_thinking_budget_is_opt_in_and_does_not_change_output_or_tools(monkeypatch):
    request = {"max_tokens": 49152, "messages": [], "tools": [{"type": "function"}]}
    monkeypatch.delenv("POLAR_THINKING_TOKEN_BUDGET", raising=False)
    assert "thinking_token_budget" not in VLLMEngine().prepare_request(dict(request))
    monkeypatch.setenv("POLAR_THINKING_TOKEN_BUDGET", "40960")
    out = VLLMEngine().prepare_request(dict(request))
    assert out["thinking_token_budget"] == 40960
    assert out["max_tokens"] == 49152 and out["tools"] == request["tools"]
    assert out["return_tokens_as_token_ids"] is False
    assert "thinking_token_budget" not in SGLangEngine().prepare_request(dict(request))
    assert VLLMEngine().prepare_request({"thinking_token_budget": 32})["thinking_token_budget"] == 32
    monkeypatch.setenv("POLAR_THINKING_TOKEN_BUDGET", "0")
    assert VLLMEngine().prepare_request({})["thinking_token_budget"] == 0


@pytest.mark.parametrize("budget", ["-1", "abc", "1.5", ""])
def test_thinking_budget_rejects_invalid_env(monkeypatch, budget):
    monkeypatch.setenv("POLAR_THINKING_TOKEN_BUDGET", budget)
    with pytest.raises(ValueError):
        VLLMEngine().prepare_request({})


def test_vllm_prepare_request_forces_logprobs_when_absent() -> None:
    out = VLLMEngine().prepare_request({"messages": []})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_vllm_normalize_renames_reasoning_to_reasoning_content() -> None:
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "a", "reasoning": "because"}}
        ]
    }
    message = VLLMEngine().normalize_response(response)["choices"][0]["message"]
    assert message["reasoning_content"] == "because"
    assert "reasoning" not in message


def test_vllm_normalize_keeps_existing_reasoning_content() -> None:
    response = {"choices": [{"message": {"reasoning": "new", "reasoning_content": "kept"}}]}
    message = VLLMEngine().normalize_response(response)["choices"][0]["message"]
    assert message["reasoning_content"] == "kept"


def test_vllm_normalize_without_reasoning_is_noop() -> None:
    response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    out = VLLMEngine().normalize_response(response)
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hi"}


def test_vllm_normalize_stamps_token_ids_onto_logprobs() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "token_ids": [10, 11],
                "logprobs": {
                    "content": [
                        {"token": "h", "logprob": -0.1},
                        {"token": "i", "logprob": -0.2},
                    ]
                },
            }
        ]
    }
    content = VLLMEngine().normalize_response(response)["choices"][0]["logprobs"]["content"]
    assert [entry["token_id"] for entry in content] == [10, 11]


def test_vllm_normalize_skips_token_id_stamp_on_length_mismatch() -> None:
    response = {
        "choices": [
            {
                "token_ids": [10, 11, 12],
                "logprobs": {"content": [{"token": "h", "logprob": -0.1}]},
            }
        ]
    }
    content = VLLMEngine().normalize_response(response)["choices"][0]["logprobs"]["content"]
    assert "token_id" not in content[0]
