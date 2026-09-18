"""Opt-in vLLM Ascend V1 plugin; loaded only via --logits-processors.

Use a separate native thinking state holder per request. The upstream adapter
handles batch moves, so holders never enter the affected cross-request swap path.
"""

from functools import wraps

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


class ThinkingBudgetLogitsProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        self.reasoning_config = vllm_config.reasoning_config
        self.device = device
        self.is_pin_memory = is_pin_memory
        if vllm_config.speculative_config is not None:
            raise ValueError("Polar thinking budget does not support speculative decoding")
        if vllm_config.additional_config.get("enable_reduce_sample", False):
            raise ValueError("Polar thinking budget requires enable_reduce_sample=False")
        # ponytail: scope this plugin to the current PP=1 rollout. VPP shares
        # processor instances across yielded batches and needs separate validation.
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Polar thinking budget currently requires rollout PP=1")
        self.reasoning_parser = Qwen3ReasoningParser(
            cached_tokenizer_from_config(vllm_config.model_config)
        )
        if device.type == "npu":
            _enable_async_output_ids()

    def is_argmax_invariant(self):
        return False

    def get_thinking_budget_forced_tokens(self, req_ids):
        """Snapshot per-step forcing by request ID, before the batch can move.

        Missing means legacy engine; None means observed, but not forced.
        The scheduler verifies the sampled token before attributing a position.
        """
        return {
            req_ids[index]: processor.func.forced_token_id
            for index, processor in self.req_info.items()
            if hasattr(processor.func, "forced_token_id")
        }

    def new_req_logits_processor(self, params):
        if params.thinking_token_budget is None:
            return None
        if self.reasoning_config is None or not self.reasoning_config.enabled:
            raise ValueError("Polar thinking budget requires --reasoning-parser qwen3")
        if self.reasoning_config.reasoning_start_token_ids != [
            self.reasoning_parser.start_token_id
        ] or self.reasoning_config.reasoning_end_token_ids != [self.reasoning_parser.end_token_id]:
            raise ValueError("Polar thinking budget requires Qwen <think>/</think> delimiters")
        holder = ThinkingBudgetStateHolder(
            self.reasoning_config, 1, 0, self.device, self.is_pin_memory
        )
        initialized = False
        finished = False
        seen_tokens = 0

        def apply(prompt_ids, output_ids, logits):
            nonlocal initialized, finished, seen_tokens
            if hasattr(holder, "last_forced_token_ids"):
                apply.forced_token_id = None
            if finished:
                return logits
            # Match the serving parser: <tool_call> can implicitly end reasoning.
            # Once content/tool generation starts, never force a token inside it.
            if (
                not initialized and self.reasoning_parser.is_reasoning_end(prompt_ids)
            ) or self.reasoning_parser.is_reasoning_end_streaming(
                output_ids, output_ids[seen_tokens:]
            ):
                finished = True
                return logits
            seen_tokens = len(output_ids)
            if not initialized:
                holder.sync_batch(
                    BatchUpdate(
                        batch_size=1,
                        removed=(),
                        moved=(),
                        added=[(0, params, prompt_ids, output_ids)],
                    )
                )
                initialized = True
            holder.update_state([output_ids], None)
            holder.apply_to_logits(logits.unsqueeze(0), False, None)
            if hasattr(holder, "last_forced_token_ids"):
                apply.forced_token_id = holder.last_forced_token_ids.get(0)
            return logits

        return apply


def _enable_async_output_ids():
    """Preserve async token feedback after Ascend recreates a hybrid KV batch.

    Ascend 0.23 loses logitsprocs_need_output_token_ids during that recreation.
    This local, idempotent hook changes metadata only for our budgeted requests;
    no installed vLLM/Ascend source files are modified.
    """
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch

    original = NPUInputBatch._make_sampling_metadata
    if getattr(original, "_polar_thinking_budget", False):
        return

    @wraps(original)
    def make_metadata(batch):
        metadata = original(batch)
        if any(
            isinstance(processor, ThinkingBudgetLogitsProcessor) and processor.req_info
            for processor in batch.logitsprocs.non_argmax_invariant
        ):
            metadata.output_token_ids = batch.req_output_token_ids
        return metadata

    make_metadata._polar_thinking_budget = True
    NPUInputBatch._make_sampling_metadata = make_metadata
