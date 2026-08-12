"""P3 stage-2: per-attempt SEGMENT spans + scores for event-level credit.

The prefix-merging builder attaches, to each trainable trace's metadata, a list
of ``[start, end, attempt_idx, score|None]`` in RESPONSE-token coordinates.
The trainer side (vime_bridge/attempt_credit.py) turns these into a discounted
reward-to-go per-token advantage. The trace/chain splitting is NOT touched —
this is additive metadata only.

Segment model (plan §6.2): spans PARTITION the trace's response stream —
every response token belongs to exactly one segment.
  * An event's segment covers [its calling turn's start, the next event's
    start) — inter-event work turns are folded into the preceding event's
    credit ("事件 e 之间的对话轮归入事件段 e").
  * The opening segment (before the first event — e.g. the Skill-dispatch
    trace0) carries ``attempt_idx = -1`` and is credited with the reward-to-go
    at the first scored event downstream ("trace0 = 事件 1 之前的段" — it must
    carry signal, never masked, never boosted beyond what the event math gives).

Ordinals are SERVER-SIDE executed-attempt indices, assigned in session time
order AT DETECTION time by the builder's pre-pass — independent of verdict
availability and of any agent-visible counter (the printed
``[pipeline-budget] attempt=N`` comes from a file the agent can delete, so it
must never be used for group alignment).

Anti-forge: an attempt counts ONLY when the assistant's tool call is an exact
``bash tools/ascendc_eval_pipeline.sh ...`` invocation (no shell chaining — so
`rm ...budget && bash ...` budget-reset hacks get no credit) AND its tool
result carries the pipeline's canonical verdict line. The verdict is the
OUTPUT of the harness running the real (read-only) pipeline; the agent cannot
forge it under a whitelisted command. A detected call whose verdict is missing
keeps its ordinal with ``score=None`` (position held for group alignment, no
credit downstream — never fabricated).

本实现与 polar_zxp 原版逐段对齐;两处适配:pipeline 名 triton→ascendc、
verdict 行前缀 [triton-eval]→[ascendc-eval](参数形态 --task 改可选)。
score 走本仓 operator_reward 的六档 ladder(0.2-0.75 soft-saturating),
与终局 reward 同尺度(credit 数学全相对,跨 fork 无需再协调)。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Exact pipeline invocation; anchored so any shell chaining (rm, ;, &&, |) fails.
# 兼容两种 agent 实际会用的写法:
#   1) bash tools/ascendc_eval_pipeline.sh ...            (任务书固定入口的规范形)
#   2) OPERATOR_NAME=.. OPERATOR_ARCH=.. bash tools/...  (agent 从 CLAUDE.md 学来的
#      env 前缀习惯; pipeline 并不读这两个变量, 纯属装饰, 与裸 bash 等价不可伪造)
# 其它 env 前缀一律拒绝: VERIFY_IMPL_REPEAT(会削弱自己的确定性检查刷分)、
# LD_PRELOAD/BASH_ENV(可向评测进程注入代码) 等。
# 兼容 SKILL.md 教的多行续行格式(bash ... \ 换行续行):先折叠续行再匹配。
# --task 在本 pipeline 里可选(固定入口不带;部分 agent 会加)。
_PIPELINE_CMD_RE = re.compile(
    r"^(?:(?:OPERATOR_NAME|OPERATOR_ARCH)=\S+\s+)*"
    r"bash\s+(?:\S*/)?ascendc_eval_pipeline\.sh"
    r"\s+--op_name\s+\S+\s+--impl\s+\S+(?:\s+--task\s+\S+)?\s+--out_dir\s+\S+"
    r"(?:\s+--incremental)?"
    r"\s*(?:2>&1)?\s*$"
)


def _normalize_command(command: str) -> str:
    """折叠 shell 续行(反斜杠+换行)与多余空白,便于精确匹配。"""
    return re.sub(r"\s+", " ", command.replace("\\\n", " ").replace("\\\r\n", " ")).strip()


_VERDICT_RE = re.compile(
    # 兼容三种 pipeline 输出行:
    #   [ascendc-eval] verdict — success=False ... error_type=... speedup_vs_torch=...  (fail_hint)
    #   [ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch=2.08 (成功路径)
    #   [ascendc-eval] cached verdict — success=... (哈希短路)
    # 成功(done)路径无 error_type 字段、且无 ast_check_ok 字段 → 两组皆可选。
    r"\[ascendc-eval\]\s*(?:verdict|done|cached verdict)\b.*?success=(?P<success>\S+)"
    r"(?:\s+ast_check_ok=(?P<ast>\S+))?(?:\s+correctness_ok=(?P<corr>\S+))?"
    r"(?:\s+error_type=(?P<etype>\S+))?\s+speedup_vs_torch=(?P<speedup>\S+)"
)


def _as_bool(tok: str | None) -> bool:
    return str(tok).strip().lower() in ("true", "1", "yes")


def _as_float_or_none(tok: str | None) -> float | None:
    t = str(tok).strip() if tok is not None else ""
    if t.lower() in ("none", "null", "nan", ""):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def pipeline_tool_call_id(assistant_msg: dict[str, Any]) -> str | None:
    """Return the tool_call id iff this assistant message ran the exact pipeline
    command (whitelisted), else None."""
    if not isinstance(assistant_msg, dict) or assistant_msg.get("role") != "assistant":
        return None
    for tc in assistant_msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        if fn.get("name") != "Bash":
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                continue
        command = (args or {}).get("command", "") if isinstance(args, dict) else ""
        if _PIPELINE_CMD_RE.match(_normalize_command(str(command))):
            return tc.get("id")
    return None


def parse_verdict(tool_content: Any) -> dict[str, Any] | None:
    """Parse the canonical pipeline verdict line into a metrics dict, or None."""
    if not isinstance(tool_content, str):
        # some harnesses wrap content in a list of parts
        if isinstance(tool_content, list):
            tool_content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in tool_content
            )
        else:
            return None
    m = _VERDICT_RE.search(tool_content)
    if not m:
        return None
    success = _as_bool(m.group("success"))
    speedup = _as_float_or_none(m.group("speedup"))
    etype = m.group("etype")
    etype = None if etype is None or str(etype).lower() == "none" else etype
    # done(成功)行没有 ast/corr 字段,按 success 语义补全;verdict 行字段齐全直接用
    ast_ok = _as_bool(m.group("ast")) if m.group("ast") is not None else success
    corr_ok = _as_bool(m.group("corr")) if m.group("corr") is not None else success
    metrics: dict[str, Any] = {
        "success": success,
        "ast_check_ok": ast_ok,
        "correctness_ok": corr_ok,
        "error_type": etype,
    }
    if success:
        metrics["perf_data"] = {"speedup_vs_torch": speedup if speedup is not None else 1.0}
    return metrics


def verdict_score(metrics: dict[str, Any]) -> float:
    """Ladder value of an attempt's verdict — same ladder as the terminal reward."""
    try:
        from polar.trajectory.evaluator.operator_reward import reward_from_metrics

        score = float(reward_from_metrics(metrics))
        if score == score:  # NaN guard
            return score
    except Exception:
        pass
    # Conservative fallback mirroring this repo's ladder coarse shape
    # (0.2/0.25/0.3/0.35/0.4/0.5-0.75+): only used when the import/scoring breaks.
    if metrics.get("success"):
        return 0.5
    if metrics.get("correctness_ok"):
        return 0.4
    if metrics.get("ast_check_ok"):
        return 0.25
    return 0.2


def build_spans(
    event_records: list[tuple[int, int, str]],
    verdict_by_call_id: dict[str, Any],
    leading_idx: int | None,
    chain_resp_end: int,
) -> list[list[Any]]:
    """Segment spans for one finalized trace, in RESPONSE-token coordinates.

    event_records: ``[(resp_start, ordinal, call_id)]`` of this trace's kept
    completions that ran a whitelisted pipeline call, in order. ``ordinal`` is
    the server-side executed-attempt index (trajectory-level, assigned at
    detection time — see module docstring).

    leading_idx: ordinal for the opening span (everything before this trace's
    first event): ``-1`` when no earlier event exists in the trajectory
    (segment 0 — the Skill-dispatch trace0 case), else the previous event's
    ordinal (segment continuation across a chain/segment break). ``None`` when
    the trajectory has no events at all -> no spans (trajectory-level fallback).

    Each event's segment extends to the next event's start (or the trace end),
    so every response token is covered exactly once. Score is the verdict's
    reward-ladder value when the verdict was found (cross-chain included,
    thanks to the session-wide pre-pass), else ``None``.
    """
    spans: list[list[Any]] = []
    first_start = event_records[0][0] if event_records else chain_resp_end
    if leading_idx is not None and first_start > 0:
        # Opening segment: segment 0 (idx=-1) or continuation of the previous
        # event's segment after a break. Score stays None — its R is resolved
        # downstream from the referenced event position.
        spans.append([0, int(first_start), int(leading_idx), None])
    for pos, (start, ordinal, call_id) in enumerate(event_records):
        seg_end = event_records[pos + 1][0] if pos + 1 < len(event_records) else chain_resp_end
        if seg_end <= start:
            continue
        score = None
        content = verdict_by_call_id.get(call_id)
        if content is not None:
            metrics = parse_verdict(content)
            if metrics is not None:
                score = verdict_score(metrics)
        spans.append([int(start), int(seg_end), int(ordinal), score])
    return spans


def env_on() -> bool:
    """POLAR_ATTEMPT_CREDIT 默认开;=0/false/no/off 关闭(回退到无 spans 的旧行为)。"""
    return os.environ.get("POLAR_ATTEMPT_CREDIT", "1").lower() not in ("0", "false", "no", "off")
