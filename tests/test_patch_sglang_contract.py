from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts" / "patch" / "patch_sglang.sh"


STRICT_TOKEN_ID_BLOCK = (
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
)

UNSAFE_TOKEN_ID_FALLBACK = (
    "            token_id = logprobs.token_ids[token_idx] "
    "if token_idx < len(logprobs.token_ids) else 0\n"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def _write_unpatched_sglang_tree(root: Path) -> None:
    _write(
        root / "srt/entrypoints/openai/protocol.py",
        """
        from typing import Dict, List, Literal, Optional, Union
        from pydantic import BaseModel, Field


        class LogProbs(BaseModel):
            text_offset: List[int] = Field(default_factory=list)
            token_logprobs: List[Optional[float]] = Field(default_factory=list)
            tokens: List[str] = Field(default_factory=list)
            top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


        class TopLogprob(BaseModel):
            token: str
            bytes: List[int]
            logprob: float


        class ChatCompletionTokenLogprob(BaseModel):
            token: str
            bytes: List[int]
            logprob: float
            top_logprobs: List[TopLogprob]


        class ChoiceLogprobs(BaseModel):
            content: List[ChatCompletionTokenLogprob]


        class ChatMessage(BaseModel):
            role: Optional[str] = None


        class DeltaMessage(BaseModel):
            role: Optional[str] = None


        class ChatCompletionResponseChoice(BaseModel):
            index: int
            message: ChatMessage
            logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
            finish_reason: Optional[
                Literal[
                    "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
                ]
            ] = None


        class ChatCompletionResponseStreamChoice(BaseModel):
            index: int
            delta: DeltaMessage
            logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
            finish_reason: Optional[
                Literal[
                    "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
                ]
            ] = None
        """,
    )
    _write(
        root / "srt/entrypoints/openai/utils.py",
        """
        def to_openai_style_logprobs():
            ret_logprobs = object()

            def append_token_logprobs(token_logprobs):
                for logprob, _, token_text in token_logprobs:
                    ret_logprobs.tokens.append(token_text)
                    ret_logprobs.token_logprobs.append(logprob)

                    # Not supported yet
                    ret_logprobs.text_offset.append(-1)
        """,
    )
    _write(
        root / "srt/managers/tokenizer_manager.py",
        """
        class ReqState:
            output_ids: List[int] = dataclasses.field(default_factory=list)
            input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)


        class TokenizerManager:
            def _tokenize_one_request(self, obj):
                tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats
                self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()

                return tokenized_obj

            def _handle_outputs(self, recv_obj):
                for i, rid in enumerate(recv_obj.rids):
                    # Build meta_info and return value
                    meta_info = {
                        "id": rid,
                        "finish_reason": recv_obj.finished_reasons[i],
                        "prompt_tokens": recv_obj.prompt_tokens[i],
                        "weight_version": self.server_args.weight_version,
                        "total_retractions": recv_obj.retraction_counts[i],
                    }
        """,
    )
    serving_chat_path = root / "srt/entrypoints/openai/serving_chat.py"
    serving_chat_path.parent.mkdir(parents=True, exist_ok=True)
    serving_chat_path.write_text(
        "class OpenAIServingChat:\n"
        "    def handle_non_stream(self):\n"
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
        "            )\n"
        "\n"
        "    async def handle_stream(self):\n"
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
        "                finish_reason = content[\"meta_info\"].get(\"finish_reason\", None)\n"
        "\n"
        "                    async for chunk in self._process_tool_call_stream(\n"
        "                        index,\n"
        "                        delta,\n"
        "                        parser_dict,\n"
        "                        content,\n"
        "                        request,\n"
        "                        has_tool_calls,\n"
        "                        continuous_usage_stats,\n"
        "                    ):\n"
        "                        pass\n"
        "\n"
        "                        choice_data = ChatCompletionResponseStreamChoice(\n"
        "                            index=index,\n"
        "                            delta=DeltaMessage(content=delta),\n"
        "                            finish_reason=None,\n"
        "                            matched_stop=None,\n"
        "                            logprobs=choice_logprobs,\n"
        "                        )\n"
        "\n"
        "    async def _process_tool_call_stream(\n"
        "        self,\n"
        "        index: int,\n"
        "        delta: str,\n"
        "        parser_dict: Dict[int, FunctionCallParser],\n"
        "        content: Dict[str, Any],\n"
        "        request: ChatCompletionRequest,\n"
        "        has_tool_calls: Dict[int, bool],\n"
        "        continuous_usage_stats: bool = False,\n"
        "    ):\n"
        "        # Yield normal text\n"
        "        if normal_text:\n"
        "            choice_data = ChatCompletionResponseStreamChoice(\n"
        "                index=index,\n"
        "                delta=DeltaMessage(content=normal_text),\n"
        "                finish_reason=None,\n"
        "            )\n"
        "            yield f\"data: {chunk.model_dump_json()}\\n\\n\"\n"
        "\n"
        "        # Yield tool calls\n"
        "        for tool_call in tool_calls:\n"
        "            choice_data = ChatCompletionResponseStreamChoice(\n"
        "                index=index,\n"
        "                delta=DeltaMessage(tool_calls=[tool_call]),\n"
        "                finish_reason=None,\n"
        "            )\n"
        "            yield f\"data: {chunk.model_dump_json()}\\n\\n\"\n"
        "\n"
        "    def _process_logprobs_tokens(\n"
        "        self, logprobs: LogProbs, use_token_index: bool = False\n"
        "    ) -> List[ChatCompletionTokenLogprob]:\n"
        "        token_logprobs = []\n"
        "\n"
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
        encoding="utf-8",
    )


def _run_patch(root: Path) -> None:
    result = _run_patch_process(root)
    assert result.returncode == 0, result.stdout + result.stderr


def _run_patch_process(root: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SGLANG_ROOT": str(root)}
    return subprocess.run(
        ["bash", str(PATCH)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _snapshot_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
    }


def _assert_strict_contract(root: Path) -> None:
    protocol = (root / "srt/entrypoints/openai/protocol.py").read_text()
    serving_chat = (root / "srt/entrypoints/openai/serving_chat.py").read_text()

    assert "token_ids: List[int] = Field(default_factory=list)" in protocol
    assert "token_id: int\n" in protocol
    assert "token_id: int = 0" not in protocol
    assert "input_token_ids: Optional[List[int]] = None" in protocol
    assert UNSAFE_TOKEN_ID_FALLBACK not in serving_chat
    assert STRICT_TOKEN_ID_BLOCK in serving_chat


def test_patch_dry_run_patches_unpatched_fixture_tree(tmp_path: Path) -> None:
    _write_unpatched_sglang_tree(tmp_path)

    _run_patch(tmp_path)

    _assert_strict_contract(tmp_path)


def test_patch_dry_run_upgrades_old_unsafe_fixture_tree(tmp_path: Path) -> None:
    _write_unpatched_sglang_tree(tmp_path)
    _run_patch(tmp_path)

    protocol = tmp_path / "srt/entrypoints/openai/protocol.py"
    protocol.write_text(
        protocol.read_text().replace("    token_id: int\n", "    token_id: int = 0\n"),
        encoding="utf-8",
    )
    serving_chat = tmp_path / "srt/entrypoints/openai/serving_chat.py"
    serving_chat.write_text(
        serving_chat.read_text().replace(STRICT_TOKEN_ID_BLOCK, UNSAFE_TOKEN_ID_FALLBACK),
        encoding="utf-8",
    )

    _run_patch(tmp_path)

    _assert_strict_contract(tmp_path)


def test_patch_dry_run_is_idempotent(tmp_path: Path) -> None:
    _write_unpatched_sglang_tree(tmp_path)
    _run_patch(tmp_path)
    first = _snapshot_tree(tmp_path)

    _run_patch(tmp_path)

    assert _snapshot_tree(tmp_path) == first
    _assert_strict_contract(tmp_path)


def test_patch_dry_run_fails_fast_when_required_file_missing(tmp_path: Path) -> None:
    _write_unpatched_sglang_tree(tmp_path)
    (tmp_path / "srt/entrypoints/openai/utils.py").unlink()

    result = _run_patch_process(tmp_path)

    assert result.returncode != 0
    assert "Expected SGLang file is missing" in result.stdout + result.stderr


def test_patch_dry_run_fails_when_strict_contract_is_removed(tmp_path: Path) -> None:
    _write_unpatched_sglang_tree(tmp_path)
    _run_patch(tmp_path)

    serving_chat = tmp_path / "srt/entrypoints/openai/serving_chat.py"
    serving_chat.write_text(
        serving_chat.read_text(encoding="utf-8").replace(
            STRICT_TOKEN_ID_BLOCK,
            "            token_id = logprobs.token_ids[token_idx]\n",
        ),
        encoding="utf-8",
    )

    result = _run_patch_process(tmp_path)

    assert result.returncode != 0
    assert (
        "Failed to patch" in result.stdout + result.stderr
        or "missing strict check" in result.stdout + result.stderr
    )


def test_server_smoke_checks_full_training_contract() -> None:
    text = PATCH.read_text()

    assert "missing/nonempty choice.input_token_ids" in text
    assert "missing/nonempty choice.logprobs.content" in text
    assert "missing logprobs.content[{idx}].token_id" in text
    assert "missing logprobs.content[{idx}].logprob" in text
    assert "response token/logprob length mismatch" in text
