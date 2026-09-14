"""Gateway-history pipeline accounting shared by the host watcher and completion guard."""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Any

from polar.trajectory.builder.attempt_spans import claim_backgrounded_verdicts, is_pipeline_invocation

PIPELINE_MARKER = "tools/triton_eval_pipeline.sh"
ASCENDC_PIPELINE_MARKER = "tools/ascendc_eval_pipeline.sh"
SUCCESS_RE = re.compile(
    r"(\[triton-eval\]\s+done\s+.*success=true|verdict\s+.*(?:success|operator_valid)=True|cached (?:verdict|evaluation)\s+.*(?:success|operator_valid)=True)",
    re.IGNORECASE,
)


def _text_block(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                typ = item.get("type")
                if typ == "text":
                    parts.append(str(item.get("text", "")))
                elif typ == "tool_result":
                    parts.append(str(item.get("content", "")))
                elif typ == "tool_use":
                    parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _response_message(record: dict[str, Any]) -> dict[str, Any] | None:
    resp = record.get("response")
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    msg = (choices[0] or {}).get("message")
    return msg if isinstance(msg, dict) else None


def _tool_uses(content: Any) -> list[dict[str, Any]]:
    uses: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                uses.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input") or {},
                    }
                )
    return uses


def _tool_results(content: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(
                    {
                        "tool_use_id": block.get("tool_use_id"),
                        "content": str(block.get("content", "")),
                    }
                )
    return results


def _response_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in msg.get("tool_calls") or []:
        fn = item.get("function") or {}
        args: Any = fn.get("arguments") or item.get("input") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        out.append({"id": item.get("id"), "name": fn.get("name") or item.get("name"), "input": args})
    return out


def _tool_command(tool: dict[str, Any]) -> str:
    if str(tool.get("name") or "") != "Bash":
        return ""
    inp = tool.get("input") if isinstance(tool.get("input"), dict) else {}
    return str(inp.get("command") or "")


def _is_pipeline_command(command: str) -> bool:
    if is_pipeline_invocation(command):
        return True
    for segment in re.split(r"\s*(?:\||&&|\|\||;)\s*", command.strip()):
        try:
            parts = shlex.split(segment)
        except ValueError:
            continue
        if not parts:
            continue
        executable = parts[0].replace("\\", "/")
        if executable in {"bash", "sh", "/bin/bash", "/bin/sh"} and len(parts) > 1:
            script = parts[1].replace("\\", "/")
        elif executable in {"python", "python3", "/usr/bin/python", "/usr/bin/python3"} and len(parts) > 1:
            script = parts[1].replace("\\", "/")
        else:
            script = executable
        if script.endswith(PIPELINE_MARKER) or script.endswith("/triton_eval_pipeline.sh"):
            return True
        # AscendC(backend=ascendc)的 agent 侧固定入口叫 ascendc_selfcheck.sh —— 只认 triton 的
        # 名字会让预算对 ascendc 完全失效(数不到一次调用 → 永不 cancel → session 无限迭代)。
        # 纯增量:triton 侧命中的仍是上面两条,行为不变。
        if script.endswith(ASCENDC_PIPELINE_MARKER) or script.endswith("/ascendc_eval_pipeline.sh") \
           or script.endswith("/ascendc_selfcheck.sh"):  # 薄壳转发,一并计数
            return True
        if script.endswith("verify.py") or script.endswith("/verify.py"):
            return True
    return False


def _is_pipeline_feedback(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return (
        "[pipeline-budget]" in low
        or "polar pipeline budget exhausted" in low
        or "[triton-eval]" in low
        or "verify_result.json" in low
        or "验证结果已保存到" in text
        or "完整错误已写入" in text
        or "judge_out/metrics_error.log" in low
        or "success=true" in low
    )


def _is_pipeline_cache_hit(text: str) -> bool:
    """True only for the fixed pipeline's zero-budget unchanged-source path."""
    low = text.lower()
    return (
        "[pipeline-budget]" not in low
        and "[ascendc-eval] cached evaluation" in low
        and "不消耗预算" in text
    )


@dataclass
class PipelineCall:
    index: int
    turn: int
    command: str
    result: str
    success: bool
    completed: bool
    cached: bool


@dataclass
class BudgetState:
    session_id: str
    pipeline_calls: list[PipelineCall]
    generation_calls: int
    optimization_calls: int
    first_success_index: int | None


def _extract_pipeline_calls(record: dict[str, Any]) -> list[PipelineCall]:
    req = record.get("original_request") if isinstance(record.get("original_request"), dict) else {}
    messages = req.get("messages") if isinstance(req.get("messages"), list) else []
    current = _response_message(record)
    calls: list[PipelineCall] = []
    turn = 0

    # Count a call the moment the assistant issues a pipeline command (command
    # match) -- independent of whether its tool-result is recognised -- so
    # runaway / bypassed calls cannot escape the count. The result, when present,
    # only marks success (used for the gen/opt split).
    pending: dict[str, PipelineCall] = {}
    pipeline_ids = {}
    verdicts = {}
    all_calls = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            turn += 1
            for tool in _tool_uses(content):
                command = _tool_command(tool)
                tool_id = str(tool.get("id") or "")
                inp = tool.get("input") or {}
                if tool_id:
                    target = command or str(inp.get("file_path") or inp.get("task_id") or "")
                    all_calls.append((tool_id, target))
                if not _is_pipeline_command(command):
                    continue
                call = PipelineCall(
                    index=len(calls) + 1,
                    turn=turn,
                    command=command,
                    result="",
                    success=False,
                    completed=True,
                    cached=False,
                )
                calls.append(call)
                tool_id = tool.get("id")
                if tool_id:
                    pending[str(tool_id)] = call
                    pipeline_ids[str(tool_id)] = (call.index, str(tool_id))
        elif role == "user":
            for item in _tool_results(content):
                verdicts[str(item.get("tool_use_id") or "")] = item.get("content")
                call = pending.pop(str(item.get("tool_use_id") or ""), None)
                if call is not None:
                    result = str(item.get("content") or "")
                    call.result = result
                    call.success = bool(SUCCESS_RE.search(result))
                    call.cached = _is_pipeline_cache_hit(result)

    claim_backgrounded_verdicts(pipeline_ids, verdicts, all_calls)
    by_index = {call.index: call for call in calls}
    for tool_id, (index, _) in pipeline_ids.items():
        result = str(verdicts.get(tool_id) or "")
        by_index[index].result = result
        by_index[index].success = bool(SUCCESS_RE.search(result))
        by_index[index].cached = _is_pipeline_cache_hit(result)

    if current:
        turn += 1
        for tool in _response_tool_calls(current):
            command = _tool_command(tool)
            if _is_pipeline_command(command):
                calls.append(
                    PipelineCall(
                        index=len(calls) + 1,
                        turn=turn,
                        command=command,
                        result="",
                        success=False,
                        completed=False,
                        cached=False,
                    )
                )
    return calls


def analyze_budget(session_id: str, record: dict[str, Any]) -> BudgetState:
    calls = _extract_pipeline_calls(record)
    counted = [call for call in calls if not call.cached]
    first_success = next(
        (call.index for call in counted if call.completed and call.success), None
    )
    if first_success is None:
        gen = len(counted)
        opt = 0
    else:
        gen = sum(1 for call in counted if call.index <= first_success)
        opt = sum(1 for call in counted if call.index > first_success)
    return BudgetState(
        session_id=session_id,
        pipeline_calls=calls,
        generation_calls=gen,
        optimization_calls=opt,
        first_success_index=first_success,
    )
