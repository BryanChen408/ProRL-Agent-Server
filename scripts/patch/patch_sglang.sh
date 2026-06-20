#!/usr/bin/env bash
#
# Patch SGLang to expose the token IDs Polar needs for training:
#   - token_id on each response-token logprob entry (ChatCompletionTokenLogprob
#     and LogProbs) for response-side token extraction.
#   - input_token_ids on each choice (streaming and non-streaming) for
#     prompt-side token extraction, sourced from the already-tokenized
#     request.input_ids so no per-prompt-token logprob is computed.
#
# Supported SGLang version: 0.5.10.  Upstream refactors any of the targeted
# snippets often; an explicit version pin makes mismatch failures actionable.

set -euo pipefail

SERVER_URL="${1:-${SGLANG_SERVER_URL:-}}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN=("${VIRTUAL_ENV}/bin/python")
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=(".venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_BIN=(uv run python)
else
  PYTHON_BIN=(python3)
fi

"${PYTHON_BIN[@]}" - <<'PY'
from __future__ import annotations

from pathlib import Path
import importlib.util
import os


SUPPORTED_SGLANG_VERSION = "0.5.10"


def fail(message: str) -> None:
    raise SystemExit(message)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        fail(f"Failed to patch {label}: expected snippet not found")
    return text.replace(old, new, 1)


def revert_once(text: str, old: str, new: str) -> str:
    """Idempotent reverse replace — a no-op when *new* is already in place."""
    if new in text:
        return text
    if old not in text:
        return text
    return text.replace(old, new, 1)


root_override = os.environ.get("SGLANG_ROOT")
if root_override:
    root = Path(root_override).resolve()
    print(f"SGLANG_ROOT override -> patching {root}")
else:
    import sglang  # noqa: E402

    if sglang.__version__ != SUPPORTED_SGLANG_VERSION:
        if os.environ.get("SGLANG_ALLOW_VERSION_MISMATCH") != "1":
            fail(
                f"patch_sglang.sh is pinned to sglang=={SUPPORTED_SGLANG_VERSION} "
                f"but found {sglang.__version__}. Install the pinned version before patching, "
                "set SGLANG_ALLOW_VERSION_MISMATCH=1, or set SGLANG_ROOT=<dir>."
            )
        print(
            f"WARNING: sglang {sglang.__version__} != pinned {SUPPORTED_SGLANG_VERSION}; "
            "continuing because SGLANG_ALLOW_VERSION_MISMATCH=1"
        )

    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        fail("sglang is not installed in the active Python environment")

    root = Path(spec.origin).resolve().parent
protocol_path = root / "srt/entrypoints/openai/protocol.py"
utils_path = root / "srt/entrypoints/openai/utils.py"
serving_chat_path = root / "srt/entrypoints/openai/serving_chat.py"
tokenizer_manager_path = root / "srt/managers/tokenizer_manager.py"

for path in (protocol_path, utils_path, serving_chat_path, tokenizer_manager_path):
    if not path.exists():
        fail(f"Expected SGLang file is missing: {path}")

# ---------------------------------------------------------------------------
# protocol.py — extend response schemas with token_id / token_ids / input_token_ids.
# ---------------------------------------------------------------------------
protocol_text = protocol_path.read_text()
protocol_text = replace_once(
    protocol_text,
    "class LogProbs(BaseModel):\n"
    "    text_offset: List[int] = Field(default_factory=list)\n"
    "    token_logprobs: List[Optional[float]] = Field(default_factory=list)\n"
    "    tokens: List[str] = Field(default_factory=list)\n"
    "    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)\n",
    "class LogProbs(BaseModel):\n"
    "    text_offset: List[int] = Field(default_factory=list)\n"
    "    token_logprobs: List[Optional[float]] = Field(default_factory=list)\n"
    "    token_ids: List[int] = Field(default_factory=list)\n"
    "    tokens: List[str] = Field(default_factory=list)\n"
    "    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)\n",
    label=str(protocol_path),
)
# Upgrade older Polar patches that added the field with a silent default. Token
# id 0 is a valid vocabulary id, so missing data must fail rather than default.
protocol_text = protocol_text.replace("    token_id: int = 0\n", "    token_id: int\n")
protocol_text = replace_once(
    protocol_text,
    "class ChatCompletionTokenLogprob(BaseModel):\n"
    "    token: str\n"
    "    bytes: List[int]\n"
    "    logprob: float\n"
    "    top_logprobs: List[TopLogprob]\n",
    "class ChatCompletionTokenLogprob(BaseModel):\n"
    "    token: str\n"
    "    token_id: int\n"
    "    bytes: List[int]\n"
    "    logprob: float\n"
    "    top_logprobs: List[TopLogprob]\n",
    label=str(protocol_path),
)
protocol_text = replace_once(
    protocol_text,
    "class ChatCompletionResponseChoice(BaseModel):\n"
    "    index: int\n"
    "    message: ChatMessage\n"
    "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
    "    finish_reason: Optional[\n",
    "class ChatCompletionResponseChoice(BaseModel):\n"
    "    index: int\n"
    "    message: ChatMessage\n"
    "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
    "    input_token_ids: Optional[List[int]] = None\n"
    "    finish_reason: Optional[\n",
    label=str(protocol_path),
)
protocol_text = replace_once(
    protocol_text,
    "class ChatCompletionResponseStreamChoice(BaseModel):\n"
    "    index: int\n"
    "    delta: DeltaMessage\n"
    "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
    "    finish_reason: Optional[\n",
    "class ChatCompletionResponseStreamChoice(BaseModel):\n"
    "    index: int\n"
    "    delta: DeltaMessage\n"
    "    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None\n"
    "    input_token_ids: Optional[List[int]] = None\n"
    "    finish_reason: Optional[\n",
    label=str(protocol_path),
)
protocol_path.write_text(protocol_text)

# ---------------------------------------------------------------------------
# utils.py — append token_id alongside the existing (logprob, text) emissions.
# ---------------------------------------------------------------------------
utils_text = utils_path.read_text()
utils_text = replace_once(
    utils_text,
    "    def append_token_logprobs(token_logprobs):\n"
    "        for logprob, _, token_text in token_logprobs:\n"
    "            ret_logprobs.tokens.append(token_text)\n"
    "            ret_logprobs.token_logprobs.append(logprob)\n"
    "\n"
    "            # Not supported yet\n"
    "            ret_logprobs.text_offset.append(-1)\n",
    "    def append_token_logprobs(token_logprobs):\n"
    "        for logprob, token_id, token_text in token_logprobs:\n"
    "            ret_logprobs.tokens.append(token_text)\n"
    "            ret_logprobs.token_logprobs.append(logprob)\n"
    "            ret_logprobs.token_ids.append(token_id)\n"
    "\n"
    "            # Not supported yet\n"
    "            ret_logprobs.text_offset.append(-1)\n",
    label=str(utils_path),
)
utils_path.write_text(utils_text)

# ---------------------------------------------------------------------------
# tokenizer_manager.py — persist canonical prompt token IDs on ReqState right
# after tokenization, then expose them via meta_info so serving_chat can emit
# input_token_ids without paying per-prompt-token logprob compute.
# ---------------------------------------------------------------------------
tokenizer_manager_text = tokenizer_manager_path.read_text()
tokenizer_manager_text = replace_once(
    tokenizer_manager_text,
    "    output_ids: List[int] = dataclasses.field(default_factory=list)\n"
    "    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)\n",
    "    output_ids: List[int] = dataclasses.field(default_factory=list)\n"
    "    input_token_ids: List[int] = dataclasses.field(default_factory=list)\n"
    "    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)\n",
    label=str(tokenizer_manager_path),
)
tokenizer_manager_text = replace_once(
    tokenizer_manager_text,
    "        tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats\n"
    "        self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()\n"
    "\n"
    "        return tokenized_obj\n",
    "        state = self.rid_to_state[obj.rid]\n"
    "        tokenized_obj.time_stats = state.time_stats\n"
    "        state.time_stats.set_tokenize_finish_time()\n"
    "        if isinstance(input_ids, list) and input_ids:\n"
    "            if isinstance(input_ids[0], int):\n"
    "                state.input_token_ids = list(input_ids)\n"
    "            elif isinstance(input_ids[0], list) and input_ids[0]:\n"
    "                state.input_token_ids = list(input_ids[0])\n"
    "\n"
    "        return tokenized_obj\n",
    label=str(tokenizer_manager_path),
)
tokenizer_manager_text = replace_once(
    tokenizer_manager_text,
    "            # Build meta_info and return value\n"
    "            meta_info = {\n"
    "                \"id\": rid,\n"
    "                \"finish_reason\": recv_obj.finished_reasons[i],\n"
    "                \"prompt_tokens\": recv_obj.prompt_tokens[i],\n"
    "                \"weight_version\": self.server_args.weight_version,\n"
    "                \"total_retractions\": recv_obj.retraction_counts[i],\n"
    "            }\n",
    "            # Build meta_info and return value\n"
    "            meta_info = {\n"
    "                \"id\": rid,\n"
    "                \"finish_reason\": recv_obj.finished_reasons[i],\n"
    "                \"prompt_tokens\": recv_obj.prompt_tokens[i],\n"
    "                \"weight_version\": self.server_args.weight_version,\n"
    "                \"total_retractions\": recv_obj.retraction_counts[i],\n"
    "            }\n"
    "            _state_input_ids = getattr(state, \"input_token_ids\", None)\n"
    "            _obj_input_ids = _state_input_ids or getattr(state.obj, \"input_ids\", None)\n"
    "            if isinstance(_obj_input_ids, list) and _obj_input_ids:\n"
    "                if isinstance(_obj_input_ids[0], int):\n"
    "                    meta_info[\"input_token_ids\"] = list(_obj_input_ids)\n"
    "                elif isinstance(_obj_input_ids[0], list) and _obj_input_ids[0]:\n"
    "                    meta_info[\"input_token_ids\"] = list(_obj_input_ids[0])\n",
    label=str(tokenizer_manager_path),
)
tokenizer_manager_path.write_text(tokenizer_manager_text)

# ---------------------------------------------------------------------------
# serving_chat.py — read input_token_ids from meta_info, plumb through to the
# choice payloads, leave logprob_start_len at SGLang's default (-1).
# ---------------------------------------------------------------------------
serving_chat_text = serving_chat_path.read_text()

# Revert the historical logprob_start_len=0 override if a prior patch applied it.
serving_chat_text = revert_once(
    serving_chat_text,
    "            return_logprob=request.logprobs,\n"
    "            logprob_start_len=0,\n"
    "            top_logprobs_num=request.top_logprobs or 0,\n",
    "            return_logprob=request.logprobs,\n"
    "            logprob_start_len=-1,\n"
    "            top_logprobs_num=request.top_logprobs or 0,\n",
)
# Drop the historical input_token_logprobs-based derivation if present —
# meta_info["input_token_ids"] is now the source of truth.
serving_chat_text = revert_once(
    serving_chat_text,
    "                input_token_logprobs = content[\"meta_info\"].get(\"input_token_logprobs\")\n"
    "                input_token_ids = None\n"
    "                if isinstance(input_token_logprobs, list):\n"
    "                    input_token_ids = [\n"
    "                        int(token_id)\n"
    "                        for _, token_id, _ in input_token_logprobs\n"
    "                    ]\n"
    "\n"
    "                finish_reason = content[\"meta_info\"].get(\"finish_reason\", None)\n",
    "                finish_reason = content[\"meta_info\"].get(\"finish_reason\", None)\n",
)
serving_chat_text = revert_once(
    serving_chat_text,
    "            input_token_logprobs = ret_item[\"meta_info\"].get(\"input_token_logprobs\")\n"
    "            input_token_ids = None\n"
    "            if isinstance(input_token_logprobs, list):\n"
    "                input_token_ids = [\n"
    "                    int(token_id)\n"
    "                    for _, token_id, _ in input_token_logprobs\n"
    "                ]\n"
    "\n"
    "            choice_data = ChatCompletionResponseChoice(\n",
    "            choice_data = ChatCompletionResponseChoice(\n",
)

# Non-streaming: capture input_token_ids from meta_info, emit on the choice.
serving_chat_text = replace_once(
    serving_chat_text,
    "            choice_data = ChatCompletionResponseChoice(\n"
    "                index=idx,\n"
    "                message=ChatMessage(\n"
    "                    role=\"assistant\",\n"
    "                    content=text if text else None,\n"
    "                    tool_calls=tool_calls,\n"
    "                    reasoning_content=reasoning_text if reasoning_text else None,\n"
    "                ),\n"
    "                logprobs=choice_logprobs,\n"
    "                finish_reason=finish_reason[\"type\"] if finish_reason else None,\n"
    "                matched_stop=(\n"
    "                    finish_reason[\"matched\"]\n"
    "                    if finish_reason and \"matched\" in finish_reason\n"
    "                    else None\n"
    "                ),\n"
    "                hidden_states=hidden_states,\n"
    "            )\n",
    "            input_token_ids = ret_item[\"meta_info\"].get(\"input_token_ids\")\n"
    "\n"
    "            choice_data = ChatCompletionResponseChoice(\n"
    "                index=idx,\n"
    "                message=ChatMessage(\n"
    "                    role=\"assistant\",\n"
    "                    content=text if text else None,\n"
    "                    tool_calls=tool_calls,\n"
    "                    reasoning_content=reasoning_text if reasoning_text else None,\n"
    "                ),\n"
    "                logprobs=choice_logprobs,\n"
    "                input_token_ids=input_token_ids,\n"
    "                finish_reason=finish_reason[\"type\"] if finish_reason else None,\n"
    "                matched_stop=(\n"
    "                    finish_reason[\"matched\"]\n"
    "                    if finish_reason and \"matched\" in finish_reason\n"
    "                    else None\n"
    "                ),\n"
    "                hidden_states=hidden_states,\n"
    "            )\n",
    label=str(serving_chat_path),
)

# Streaming: capture input_token_ids from meta_info, pass through the tool-call
# helper, and emit on every StreamChoice that currently carries logprobs.
serving_chat_text = replace_once(
    serving_chat_text,
    "                # Handle logprobs\n"
    "                choice_logprobs = None\n"
    "                if request.logprobs:\n"
    "                    n_prev_token = n_prev_tokens.get(index, 0)\n"
    "                    total_output_logprobs = content[\"meta_info\"][\n"
    "                        \"output_token_logprobs_length\"\n"
    "                    ]\n"
    "                    if n_prev_token < total_output_logprobs:\n"
    "                        choice_logprobs = self._process_streaming_logprobs(\n"
    "                            content, n_prev_token, total_output_logprobs\n"
    "                        )\n"
    "                    n_prev_tokens[index] = total_output_logprobs\n"
    "\n"
    "                finish_reason = content[\"meta_info\"].get(\"finish_reason\", None)\n",
    "                # Handle logprobs\n"
    "                choice_logprobs = None\n"
    "                if request.logprobs:\n"
    "                    n_prev_token = n_prev_tokens.get(index, 0)\n"
    "                    total_output_logprobs = content[\"meta_info\"][\n"
    "                        \"output_token_logprobs_length\"\n"
    "                    ]\n"
    "                    if n_prev_token < total_output_logprobs:\n"
    "                        choice_logprobs = self._process_streaming_logprobs(\n"
    "                            content, n_prev_token, total_output_logprobs\n"
    "                        )\n"
    "                    n_prev_tokens[index] = total_output_logprobs\n"
    "\n"
    "                input_token_ids = content[\"meta_info\"].get(\"input_token_ids\")\n"
    "\n"
    "                finish_reason = content[\"meta_info\"].get(\"finish_reason\", None)\n",
    label=str(serving_chat_path),
)
serving_chat_text = serving_chat_text.replace(
    "            token_id = logprobs.token_ids[token_idx] if token_idx < len(logprobs.token_ids) else 0\n",
    "            if len(logprobs.token_ids) != len(logprobs.token_logprobs):\n"
    "                raise ValueError(\n"
    "                    \"SGLang logprob token_id contract violated: \"\n"
    "                    f\"len(token_ids)={len(logprobs.token_ids)} != \"\n"
    "                    f\"len(token_logprobs)={len(logprobs.token_logprobs)}\"\n"
    "                )\n"
    "            if len(logprobs.tokens) != len(logprobs.token_logprobs):\n"
    "                raise ValueError(\n"
    "                    \"SGLang logprob token text contract violated: \"\n"
    "                    f\"len(tokens)={len(logprobs.tokens)} != \"\n"
    "                    f\"len(token_logprobs)={len(logprobs.token_logprobs)}\"\n"
    "                )\n"
    "            token_id = logprobs.token_ids[token_idx]\n",
)
serving_chat_text = replace_once(
    serving_chat_text,
    "                    async for chunk in self._process_tool_call_stream(\n"
    "                        index,\n"
    "                        delta,\n"
    "                        parser_dict,\n"
    "                        content,\n"
    "                        request,\n"
    "                        has_tool_calls,\n"
    "                        continuous_usage_stats,\n"
    "                    ):\n",
    "                    async for chunk in self._process_tool_call_stream(\n"
    "                        index,\n"
    "                        delta,\n"
    "                        parser_dict,\n"
    "                        content,\n"
    "                        request,\n"
    "                        has_tool_calls,\n"
    "                        choice_logprobs,\n"
    "                        input_token_ids,\n"
    "                        continuous_usage_stats,\n"
    "                    ):\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "                        choice_data = ChatCompletionResponseStreamChoice(\n"
    "                            index=index,\n"
    "                            delta=DeltaMessage(content=delta),\n"
    "                            finish_reason=None,\n"
    "                            matched_stop=None,\n"
    "                            logprobs=choice_logprobs,\n"
    "                        )\n",
    "                        choice_data = ChatCompletionResponseStreamChoice(\n"
    "                            index=index,\n"
    "                            delta=DeltaMessage(content=delta),\n"
    "                            finish_reason=None,\n"
    "                            matched_stop=None,\n"
    "                            logprobs=choice_logprobs,\n"
    "                            input_token_ids=input_token_ids,\n"
    "                        )\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "    async def _process_tool_call_stream(\n"
    "        self,\n"
    "        index: int,\n"
    "        delta: str,\n"
    "        parser_dict: Dict[int, FunctionCallParser],\n"
    "        content: Dict[str, Any],\n"
    "        request: ChatCompletionRequest,\n"
    "        has_tool_calls: Dict[int, bool],\n"
    "        continuous_usage_stats: bool = False,\n"
    "    ):\n",
    "    async def _process_tool_call_stream(\n"
    "        self,\n"
    "        index: int,\n"
    "        delta: str,\n"
    "        parser_dict: Dict[int, FunctionCallParser],\n"
    "        content: Dict[str, Any],\n"
    "        request: ChatCompletionRequest,\n"
    "        has_tool_calls: Dict[int, bool],\n"
    "        choice_logprobs: Optional[ChoiceLogprobs],\n"
    "        input_token_ids: Optional[List[int]],\n"
    "        continuous_usage_stats: bool = False,\n"
    "    ):\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "        # Yield normal text\n"
    "        if normal_text:\n"
    "            choice_data = ChatCompletionResponseStreamChoice(\n"
    "                index=index,\n"
    "                delta=DeltaMessage(content=normal_text),\n"
    "                finish_reason=None,\n"
    "            )\n",
    "        metadata_emitted = False\n"
    "\n"
    "        # Yield normal text\n"
    "        if normal_text:\n"
    "            choice_data = ChatCompletionResponseStreamChoice(\n"
    "                index=index,\n"
    "                delta=DeltaMessage(content=normal_text),\n"
    "                logprobs=choice_logprobs,\n"
    "                input_token_ids=input_token_ids,\n"
    "                finish_reason=None,\n"
    "            )\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "            yield f\"data: {chunk.model_dump_json()}\\n\\n\"\n"
    "\n"
    "        # Yield tool calls\n",
    "            yield f\"data: {chunk.model_dump_json()}\\n\\n\"\n"
    "            metadata_emitted = True\n"
    "\n"
    "        # Yield tool calls\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "            choice_data = ChatCompletionResponseStreamChoice(\n"
    "                index=index,\n"
    "                delta=DeltaMessage(tool_calls=[tool_call]),\n"
    "                finish_reason=None,\n"
    "            )\n",
    "            choice_data = ChatCompletionResponseStreamChoice(\n"
    "                index=index,\n"
    "                delta=DeltaMessage(tool_calls=[tool_call]),\n"
    "                logprobs=choice_logprobs if not metadata_emitted else None,\n"
    "                input_token_ids=input_token_ids if not metadata_emitted else None,\n"
    "                finish_reason=None,\n"
    "            )\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "            yield f\"data: {chunk.model_dump_json()}\\n\\n\"\n",
    "            yield f\"data: {chunk.model_dump_json()}\\n\\n\"\n"
    "            metadata_emitted = True\n",
    label=str(serving_chat_path),
)
serving_chat_text = replace_once(
    serving_chat_text,
    "        for token_idx, (token, logprob) in enumerate(\n"
    "            zip(logprobs.tokens, logprobs.token_logprobs)\n"
    "        ):\n"
    "            token_bytes = list(token.encode(\"utf-8\"))\n"
    "            top_logprobs = []\n"
    "            if logprobs.top_logprobs:\n"
    "                # - Non-streaming (use_token_index=True): uses token_idx for full data\n"
    "                # - Streaming (use_token_index=False): uses index 0 for pre-sliced data\n"
    "                top_logprobs_idx = token_idx if use_token_index else 0\n"
    "                for top_token, top_logprob in logprobs.top_logprobs[\n"
    "                    top_logprobs_idx\n"
    "                ].items():\n"
    "                    top_token_bytes = list(top_token.encode(\"utf-8\"))\n"
    "                    top_logprobs.append(\n"
    "                        TopLogprob(\n"
    "                            token=top_token,\n"
    "                            bytes=top_token_bytes,\n"
    "                            logprob=top_logprob,\n"
    "                        )\n"
    "                    )\n"
    "            token_logprobs.append(\n"
    "                ChatCompletionTokenLogprob(\n"
    "                    token=token,\n"
    "                    bytes=token_bytes,\n"
    "                    logprob=logprob,\n"
    "                    top_logprobs=top_logprobs,\n"
    "                )\n"
    "            )\n",
    "        for token_idx, (token, logprob) in enumerate(\n"
    "            zip(logprobs.tokens, logprobs.token_logprobs)\n"
    "        ):\n"
    "            if len(logprobs.token_ids) != len(logprobs.token_logprobs):\n"
    "                raise ValueError(\n"
    "                    \"SGLang logprob token_id contract violated: \"\n"
    "                    f\"len(token_ids)={len(logprobs.token_ids)} != \"\n"
    "                    f\"len(token_logprobs)={len(logprobs.token_logprobs)}\"\n"
    "                )\n"
    "            if len(logprobs.tokens) != len(logprobs.token_logprobs):\n"
    "                raise ValueError(\n"
    "                    \"SGLang logprob token text contract violated: \"\n"
    "                    f\"len(tokens)={len(logprobs.tokens)} != \"\n"
    "                    f\"len(token_logprobs)={len(logprobs.token_logprobs)}\"\n"
    "                )\n"
    "            token_id = logprobs.token_ids[token_idx]\n"
    "            token_bytes = list(token.encode(\"utf-8\"))\n"
    "            top_logprobs = []\n"
    "            if logprobs.top_logprobs:\n"
    "                # - Non-streaming (use_token_index=True): uses token_idx for full data\n"
    "                # - Streaming (use_token_index=False): uses index 0 for pre-sliced data\n"
    "                top_logprobs_idx = token_idx if use_token_index else 0\n"
    "                for top_token, top_logprob in logprobs.top_logprobs[\n"
    "                    top_logprobs_idx\n"
    "                ].items():\n"
    "                    top_token_bytes = list(top_token.encode(\"utf-8\"))\n"
    "                    top_logprobs.append(\n"
    "                        TopLogprob(\n"
    "                            token=top_token,\n"
    "                            bytes=top_token_bytes,\n"
    "                            logprob=top_logprob,\n"
    "                        )\n"
    "                    )\n"
    "            token_logprobs.append(\n"
    "                ChatCompletionTokenLogprob(\n"
    "                    token=token,\n"
    "                    token_id=token_id,\n"
    "                    bytes=token_bytes,\n"
    "                    logprob=logprob,\n"
    "                    top_logprobs=top_logprobs,\n"
    "                )\n"
    "            )\n",
    label=str(serving_chat_path),
)
serving_chat_path.write_text(serving_chat_text)

protocol_check = protocol_path.read_text()
serving_chat_check = serving_chat_path.read_text()
if "token_id: int = 0" in protocol_check:
    fail("SGLang patch contract failed: unsafe token_id default remains in protocol.py")
if "token_id = logprobs.token_ids[token_idx] if token_idx < len(logprobs.token_ids) else 0" in serving_chat_check:
    fail("SGLang patch contract failed: unsafe token_id fallback remains in serving_chat.py")
for needle in (
    "SGLang logprob token_id contract violated",
    "SGLang logprob token text contract violated",
):
    if needle not in serving_chat_check:
        fail(f"SGLang patch contract failed: missing strict check {needle!r}")

print(f"Patched SGLang in {root}")
PY

if [[ -n "${SERVER_URL}" ]]; then
  "${PYTHON_BIN[@]}" - "${SERVER_URL}" <<'PY'
from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


server_url = sys.argv[1].rstrip("/")


def fetch_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


models = fetch_json(f"{server_url}/v1/models")
model_id = models["data"][0]["id"]
response = fetch_json(
    f"{server_url}/v1/chat/completions",
    {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word ok."}],
        "max_tokens": 8,
        "temperature": 0,
        "logprobs": True,
    },
)

choice = response["choices"][0]
input_token_ids = choice.get("input_token_ids")
if not isinstance(input_token_ids, list) or not input_token_ids:
    raise SystemExit("Patch smoke test failed: missing/nonempty choice.input_token_ids")

content = (choice.get("logprobs") or {}).get("content")
if not isinstance(content, list) or not content:
    raise SystemExit("Patch smoke test failed: missing/nonempty choice.logprobs.content")

for idx, item in enumerate(content):
    if not isinstance(item, dict):
        raise SystemExit(f"Patch smoke test failed: logprobs.content[{idx}] is not an object")
    if "token_id" not in item or item["token_id"] is None:
        raise SystemExit(f"Patch smoke test failed: missing logprobs.content[{idx}].token_id")
    if "logprob" not in item or item["logprob"] is None:
        raise SystemExit(f"Patch smoke test failed: missing logprobs.content[{idx}].logprob")

token_ids = [item["token_id"] for item in content]
logprobs = [item["logprob"] for item in content]
if len(token_ids) != len(logprobs):
    raise SystemExit(
        f"Patch smoke test failed: response token/logprob length mismatch "
        f"{len(token_ids)} != {len(logprobs)}"
    )

print("SGLang patch smoke test passed")
PY
fi
