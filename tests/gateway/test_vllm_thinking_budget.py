"""Optional engine tests; the regular gateway environment need not install vLLM."""

import os
import json
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessors, MoveDirectionality  # noqa: E402

from polar.gateway.vllm_thinking_budget import ThinkingBudgetLogitsProcessor  # noqa: E402


@pytest.fixture(autouse=True)
def qwen_tokenizer(monkeypatch):
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {
            "<think>": 1,
            "</think>": 2,
            "<tool_call>": 6,
            "</tool_call>": 7,
        }
    )
    monkeypatch.setattr(
        "polar.gateway.vllm_thinking_budget.cached_tokenizer_from_config",
        lambda *a, **kw: tokenizer,
    )
    return tokenizer


def config(**overrides):
    values = dict(
        reasoning_config=SimpleNamespace(
            reasoning_start_token_ids=[1],
            reasoning_end_token_ids=[2],
            enabled=True,
        ),
        speculative_config=None,
        additional_config={},
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(),
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("budget,output,expected", [(0, [], 2), (3, [2], 4), (None, [4] * 4, 4)])
def test_zero_natural_end_and_no_budget(budget, output, expected):
    processor = ThinkingBudgetLogitsProcessor(config(), torch.device("cpu"), False)
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(thinking_token_budget=budget), [0, 1], output)],
        )
    )
    logits = torch.zeros(1, 8)
    logits[:, 4] = 5
    assert processor.apply(logits).argmax(-1).tolist() == [expected]


def test_swap_and_reused_batch_row_keep_request_budgets_separate():
    processor = ThinkingBudgetLogitsProcessor(config(), torch.device("cpu"), False)
    output = [4]
    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=(),
            moved=(),
            added=[
                (0, SamplingParams(thinking_token_budget=1), [0, 1], output),
                (1, SamplingParams(), [0, 1], []),
            ],
        )
    )
    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert set(processor.req_info) == {1}
    logits = torch.zeros(2, 8)
    logits[:, 4] = 5
    assert processor.apply(logits).argmax(-1).tolist() == [4, 2]
    from vllm.v1.outputs import ModelRunnerOutput

    snapshot = ModelRunnerOutput(
        req_ids=["plain", "budget"], req_id_to_index={"plain": 0, "budget": 1}
    )
    snapshot.collect_thinking_budget_forcing(LogitsProcessors([processor]))
    assert snapshot.thinking_budget_forced_tokens == {"budget": 2}
    output.append(2)
    logits = torch.zeros(2, 8)
    logits[:, 4] = 5
    assert processor.apply(logits).argmax(-1).tolist() == [4, 4]
    assert processor.get_thinking_budget_forced_tokens(["plain", "budget"]) == {"budget": None}
    assert snapshot.thinking_budget_forced_tokens == {
        "budget": 2
    }  # Async snapshot is immutable by later steps.
    processor.update_state(BatchUpdate(batch_size=1, removed=[1], added=(), moved=()))
    assert not processor.req_info

    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(thinking_token_budget=0), [0, 1], [])],
        )
    )
    assert processor.req_info
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(), [0, 1], [])],
        )
    )
    assert not processor.req_info


@pytest.mark.parametrize("output_kind", ["DELTA", "FINAL_ONLY"])
def test_forcing_survives_ipc_output_processing_and_choice_aggregation(output_kind):
    import msgspec
    import pickle
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.outputs import ModelRunnerOutput

    processor = ThinkingBudgetLogitsProcessor(config(), torch.device("cpu"), False)
    buffers = {"forced": [], "natural": []}
    params = SamplingParams(
        thinking_token_budget=1,
        detokenize=False,
        output_kind=getattr(RequestOutputKind, output_kind),
    )
    processor.update_state(
        BatchUpdate(
            batch_size=2,
            removed=(),
            moved=(),
            added=[(i, params, [0, 1], buffers[name]) for i, name in enumerate(buffers)],
        )
    )
    frontend = OutputProcessor(tokenizer=None, log_stats=False)
    for name in buffers:
        frontend.add_request(
            EngineCoreRequest(
                request_id=name,
                external_req_id=name,
                prompt_token_ids=[0, 1],
                mm_features=None,
                sampling_params=params,
                pooling_params=None,
                arrival_time=0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            ),
            prompt=None,
        )

    accumulated = {}
    for step in range(3):
        logits = torch.zeros(2, 8)
        logits[0, 4 if step == 0 else 6] = 5
        logits[1, 2 if step == 0 else 6] = 5
        sampled = processor.apply(logits).argmax(-1).tolist()
        runner = ModelRunnerOutput(
            req_ids=list(buffers), req_id_to_index={"forced": 0, "natural": 1}
        )
        runner.collect_thinking_budget_forcing(LogitsProcessors([processor]))
        runner = pickle.loads(pickle.dumps(runner))
        outputs = []
        for i, (name, tokens) in enumerate(buffers.items()):
            indices = runner.thinking_budget_indices(name, len(tokens), [sampled[i]])
            event = EngineCoreOutput(
                request_id=name,
                new_token_ids=[sampled[i]],
                thinking_budget_forced_token_indices=indices,
                finish_reason=FinishReason.STOP if step == 2 else None,
            )
            outputs.append(
                msgspec.msgpack.decode(msgspec.msgpack.encode(event), type=EngineCoreOutput)
            )
            tokens.append(sampled[i])
        for res in frontend.process_outputs(outputs).request_outputs:
            if res.request_id in accumulated:
                accumulated[res.request_id].add(res, aggregate=True)
            else:
                accumulated[res.request_id] = res
    assert buffers == {"forced": [4, 2, 6], "natural": [2, 6, 6]}
    for name, expected in [("forced", [1]), ("natural", [])]:
        out = accumulated[name].outputs[0]
        assert list(out.token_ids) == buffers[name]
        assert out.thinking_budget_forced_token_indices == expected
    # Discarded partial-prefill samples never acquire a forced position.
    runner.thinking_budget_forced_tokens = {"forced": 2}
    assert runner.thinking_budget_indices("forced", 0, []) is None
    assert runner.thinking_budget_indices("legacy", 0, [2]) is None
    with pytest.raises(ValueError, match="does not match"):
        runner.thinking_budget_indices("forced", 0, [4])


@pytest.mark.parametrize(
    "overrides",
    [
        {"speculative_config": object()},
        {"additional_config": {"enable_reduce_sample": True}},
        {"parallel_config": SimpleNamespace(pipeline_parallel_size=2)},
    ],
)
def test_unsupported_engine_modes_fail_explicitly(overrides):
    with pytest.raises(ValueError, match="Polar thinking budget"):
        ThinkingBudgetLogitsProcessor(config(**overrides), torch.device("cpu"), False)


def test_missing_reasoning_config_fails_only_for_budgeted_requests():
    processor = ThinkingBudgetLogitsProcessor(
        config(reasoning_config=None), torch.device("cpu"), False
    )
    assert processor.new_req_logits_processor(SamplingParams()) is None
    with pytest.raises(ValueError, match="reasoning-parser"):
        processor.new_req_logits_processor(SamplingParams(thinking_token_budget=1))


@pytest.mark.parametrize("forced", [None, [], [1]])
def test_actual_chat_serving_returns_attribution_in_full_and_streamed_responses(forced):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.outputs import CompletionOutput, RequestOutput

    serving = object.__new__(OpenAIServingChat)
    for name, value in dict(
        use_harmony=False,
        tool_call_id_type="random",
        response_role="assistant",
        enable_auto_tools=False,
        tool_parser=None,
        parser_cls=None,
        enable_force_include_usage=False,
        enable_prompt_tokens_details=False,
        enable_log_outputs=False,
        system_fingerprint=None,
    ).items():
        setattr(serving, name, value)
    output = RequestOutput(
        request_id="test",
        prompt=None,
        prompt_token_ids=[0, 1],
        prompt_logprobs=None,
        finished=True,
        outputs=[
            CompletionOutput(
                index=0,
                text="body",
                token_ids=[4, 2, 6],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
                thinking_budget_forced_token_indices=forced,
            )
        ],
    )

    async def generate():
        yield output

    async def run():
        request = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "q"}], return_token_ids=True
        )
        full = await serving.chat_completion_full_generator(
            request, generate(), "test", "test", [], None, SimpleNamespace()
        )
        assert full.choices[0].thinking_budget_forced_token_indices == forced
        request.stream = True
        chunks = [
            s
            async for s in serving.chat_completion_stream_generator(
                request, generate(), "test", "test", [], None, SimpleNamespace()
            )
        ]
        events = [json.loads(s[6:]) for s in chunks if s.startswith("data: {")]
        assert not any("error" in e for e in events), events
        final = [c for e in events for c in e.get("choices", []) if c.get("finish_reason")]
        assert len(final) == 1
        assert final[0]["thinking_budget_forced_token_indices"] == forced
        assert final[0]["token_ids"] == [4, 2, 6]

    asyncio.run(run())


@pytest.mark.parametrize(
    "prompt,output",
    [
        ([0, 1], [4, 6, 4, 4]),  # Tool call implicitly ends thinking.
        ([0, 1, 4, 6], [4, 4]),  # A tool call was already started in a prefill.
        ([0, 1], [2, 4, 4, 4]),  # Natural explicit thinking end.
    ],
)
def test_budget_leaves_body_and_tool_arguments_unchanged(prompt, output):
    processor = ThinkingBudgetLogitsProcessor(config(), torch.device("cpu"), False)
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(thinking_token_budget=2), prompt, output)],
        )
    )
    for _ in range(4):
        logits = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        before = logits.clone()
        assert torch.equal(processor.apply(logits), before)
        output.append(4)


@pytest.mark.parametrize("explicit_end", [True, False])
def test_real_qwen_parsers_preserve_body_and_tool_arguments(qwen_tokenizer, explicit_end):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                },
            },
        }
    ]
    request = ChatCompletionRequest(model="qwen36", messages=[], tools=tools)
    tail = "<tool_call>\n<function=Bash>\n<parameter=command>echo ok</parameter>\n</function>\n</tool_call>"
    output = "plan" + ("</think>正文\n" if explicit_end else "") + tail
    reasoning, content = Qwen3ReasoningParser(qwen_tokenizer).extract_reasoning(output, request)
    parsed = Qwen3CoderToolParser(qwen_tokenizer, request.tools).extract_tool_calls(
        content, request
    )
    assert reasoning == "plan"
    assert parsed.tools_called and len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].function.name == "Bash"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {"command": "echo ok"}
    assert parsed.content == ("正文\n" if explicit_end else None)


@pytest.mark.parametrize("force_end", [False, True])
def test_sampling_to_body_tool_parsing_and_precise_training_mask(
    qwen_tokenizer, monkeypatch, force_end
):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
    from vllm.v1.outputs import ModelRunnerOutput

    from polar.trajectory.builder.record_utils import build_trace_from_completion
    from polar.trajectory.models import CompletionRecord

    monkeypatch.setenv("POLAR_MASK_REASONING", "0")
    budget = 1 if force_end else 8
    output = []
    processor = ThinkingBudgetLogitsProcessor(config(), torch.device("cpu"), False)
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            moved=(),
            added=[(0, SamplingParams(thinking_token_budget=budget), [0, 1], output)],
        )
    )
    text = {
        3: "plan",
        2: "</think>",
        4: "正文\n",
        6: "<tool_call>\n",
        5: "<function=Bash>\n<parameter=command>echo ok</parameter>\n</function>\n",
        7: "</tool_call>",
    }
    forced_indices = []
    for index, desired in enumerate([3, 2, 4, 6, 5, 7]):
        logits = torch.zeros(1, 8)
        logits[0, 3 if force_end and index == 1 else desired] = 5
        before = logits.clone()
        sampled = processor.apply(logits).argmax(-1).item()
        if index > 1:
            assert torch.equal(logits, before)  # No intervention in body/tools.
        assert sampled == desired
        runner = ModelRunnerOutput(req_ids=["test"], req_id_to_index={"test": 0})
        runner.collect_thinking_budget_forcing(LogitsProcessors([processor]))
        forced_indices.extend(runner.thinking_budget_indices("test", index, [sampled]))
        output.append(sampled)
    assert forced_indices == ([1] if force_end else [])
    request = ChatCompletionRequest(
        model="qwen36",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    )
    reasoning, content = Qwen3ReasoningParser(qwen_tokenizer).extract_reasoning(
        "".join(text[t] for t in output), request
    )
    parsed = Qwen3CoderToolParser(qwen_tokenizer, request.tools).extract_tool_calls(
        content, request
    )
    assert reasoning == "plan" and parsed.content == "正文\n"
    assert len(parsed.tool_calls) == 1
    assert json.loads(parsed.tool_calls[0].function.arguments) == {"command": "echo ok"}
    logprobs = [-0.1 * (i + 1) for i in range(len(output))]
    record = CompletionRecord(
        completion_id="test",
        timestamp="2026-09-18T00:00:00Z",
        request={},
        response={
            "prompt_token_ids": [0, 1],
            "_polar_thinking_token_budget": budget,
            "choices": [
                {
                    "token_ids": output,
                    "finish_reason": "tool_calls",
                    "thinking_budget_forced_token_indices": forced_indices,
                    "message": {
                        "reasoning_content": reasoning,
                        "content": parsed.content,
                        "tool_calls": [t.model_dump() for t in parsed.tool_calls],
                    },
                    "logprobs": {
                        "content": [
                            {"token": text[t], "token_id": t, "logprob": lp}
                            for t, lp in zip(output, logprobs)
                        ]
                    },
                }
            ],
        },
    )
    trace = build_trace_from_completion(record)
    assert trace.prompt_ids == [0, 1] and trace.response_ids == output
    assert trace.response_logprobs == logprobs
    assert trace.loss_mask == ([1, 0, 1, 1, 1, 1] if force_end else [1] * 6)


@pytest.mark.skipif(os.environ.get("POLAR_TEST_NPU") != "1", reason="opt-in real NPU check")
def test_unmodified_ascend_async_batch_and_plugin_loader():
    from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns
    from vllm.v1.worker.gpu_input_batch import CachedRequestState
    from vllm_ascend.sample.sampler import AscendSampler
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch

    (cls,) = _load_logitsprocs_by_fqcns(
        ["polar.gateway.vllm_thinking_budget:ThinkingBudgetLogitsProcessor"]
    )
    original = NPUInputBatch._make_sampling_metadata
    # Restore the class after testing the plugin's process-local hook.
    with patch.object(NPUInputBatch, "_make_sampling_metadata", original):
        processor = cls(config(), torch.device("npu:0"), False)
        installed = NPUInputBatch._make_sampling_metadata
        cls(config(), torch.device("npu:0"), False)
        assert NPUInputBatch._make_sampling_metadata is installed
        with patch("vllm_ascend.worker.npu_input_batch.MultiGroupBlockTable"):
            batch = NPUInputBatch(
                max_num_reqs=2,
                max_model_len=32,
                max_num_batched_tokens=32,
                device=torch.device("npu:0"),
                pin_memory=False,
                vocab_size=8,
                block_sizes=[16],
                kernel_block_sizes=[[16]],
                logitsprocs=LogitsProcessors([processor]),
                # Simulate the hybrid KV reinitialization that loses this flag.
                logitsprocs_need_output_token_ids=False,
            )
        assert batch.thinking_budget_state_holder is None
        for name, budget in [("budget", 3), ("plain", None)]:
            batch.add_request(
                CachedRequestState(
                    req_id=name,
                    prompt_token_ids=[0, 1, 3],
                    mm_features=[],
                    sampling_params=SamplingParams(thinking_token_budget=budget, temperature=0),
                    generator=None,
                    block_ids=([],),
                    num_computed_tokens=0,
                    output_token_ids=[],
                )
            )
        batch.refresh_metadata()
        assert batch.sampling_metadata.output_token_ids is batch.req_output_token_ids
        sampler = AscendSampler()

        def sample():
            batch.update_async_output_token_ids()
            logits = torch.zeros((batch.num_reqs, 8), device="npu:0")
            logits[:, 4] = 5
            # Check NPU logits without depending on the tiny-vocabulary argmax kernel.
            return (
                sampler.apply_logits_processors(logits, batch.sampling_metadata, False)
                .cpu()
                .argmax(-1)
                .tolist()
            )

        assert sample() == [4, 4]
        batch.req_output_token_ids[0].append(-1)
        batch.prev_req_id_to_index = dict(batch.req_id_to_index)
        event = Mock()
        batch.set_async_sampled_token_ids(torch.tensor([[4], [4]]), event)
        assert sample() == [4, 4]
        event.synchronize.assert_called_once()
        batch.req_output_token_ids[0].append(4)
        batch.swap_states(0, 1)
        batch.refresh_metadata()
        assert sample() == [4, 2]
        from vllm.v1.outputs import ModelRunnerOutput

        reported = ModelRunnerOutput(
            req_ids=list(batch.req_ids), req_id_to_index=dict(batch.req_id_to_index)
        )
        reported.collect_thinking_budget_forcing(batch.logitsprocs)
        assert reported.thinking_budget_forced_tokens == {"budget": 2}
        assert reported.thinking_budget_indices("budget", 2, [2]) == [2]
        batch.req_output_token_ids[1].append(2)
        assert sample() == [4, 4]
        assert processor.get_thinking_budget_forced_tokens(batch.req_ids) == {"budget": None}
        batch.remove_request("budget")
        batch.condense()
        batch.refresh_metadata()
        assert batch.sampling_metadata.output_token_ids == []
