"""P3 stage-2: per-attempt SEGMENT spans + scores for event-level credit.

The prefix-merging builder attaches, to each trainable trace's metadata, a list
of ``[start, end, attempt_idx, score|None]`` in RESPONSE-token coordinates.
The trainer side (vime_bridge/attempt_credit.py) turns these into a discounted
reward-to-go per-token advantage. The trace/chain splitting is NOT touched —
the spans themselves are additive metadata only.

This module also hosts the ONE consumer that does touch training: ``best_ordinal``
drives post-best masking (turns after the attempt that wrote ``.best`` get
``loss_mask=0``), reusing the same whitelisted-command + harness-verdict detection
instead of scraping pipeline stdout markers.

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

# Whitelisted pipeline invocation. The bar is "this shell line can only have run the
# real, read-only pipeline and shown its output verbatim" — NOT "it is spelled the
# canonical way". Measured on run 165820 (1408 invocations): the strict-spelling form
# recognised 70.9%; the wrappers below take it to 80.8% with zero regressions, and the
# 271 still rejected are all either not invocations at all (ps/ls/find/head over the
# script) or genuine anti-forge chains (rm/tar/cp before the run).
#
# BENIGN WRAPPERS (each provably cannot alter what the pipeline prints):
#   cd <abs path> &&    — 56x. Rejected if the path contains ".."; see the path note.
#   timeout [-k N] D    — 10x. coreutils wrapper, no text of its own.
#   env prefix          — inline `VAR=v` or `export VAR=v &&`, ALLOW-LIST only
#                         (OPERATOR_NAME/OPERATOR_ARCH are decorative; ASCEND_HOME(_PATH)
#                         is what the agent actually sets). A deny-list would be wrong
#                         here: PATH / LD_PRELOAD / BASH_ENV / VERIFY_IMPL_REPEAT all
#                         subvert the run, and the next such variable is unknown today.
#   | tail|head|grep|cat — 42x. These SELECT lines; they cannot write text that was not
#                         already there, so a verdict surviving them is still the
#                         harness's own. sed/awk/tee/tr and friends stay rejected —
#                         they can rewrite `success=false` into `success=true`.
#   --impl optional     — 19x. Naming no impl just makes the pipeline take its default.
#
# PATH: tightened from `(?:\S*/)?` to `(?:/\S*/)?tools/`. The old wildcard accepted
# `bash /tmp/mine/ascendc_eval_pipeline.sh`, i.e. the agent's own script — a forge hole
# that already existed and that allowing `cd` would otherwise widen (a relative
# `tools/...` would resolve wherever the cd landed). Requiring a `tools/` parent closes
# both. Anti-forge remains defence in depth, not the last line: the reward itself comes
# from the host-side judge in a fresh container, so a forged verdict can only misplace
# per-attempt credit.
#
# Multi-line continuations (bash ... \ newline) are folded by _normalize_command first.
_SAFE_ENV_NAME = r"(?:OPERATOR_NAME|OPERATOR_ARCH|ASCEND_HOME|ASCEND_HOME_PATH)"
_PIPELINE_CMD_RE = re.compile(
    r"^"
    rf"(?:(?:export\s+)?{_SAFE_ENV_NAME}=\S+\s*(?:&&\s*|\s+))*"
    r"(?:timeout\s+(?:-k\s+\S+\s+)?\d+[smhd]?\s+)?"
    r"bash\s+(?:/\S*/)?tools/ascendc_eval_pipeline\.sh"
    r"\s+--op_name\s+\S+(?:\s+--impl\s+\S+)?(?:\s+--task\s+\S+)?\s+--out_dir\s+\S+"
    r"(?:\s+--incremental)?"
    r"(?:\s*2>&1)?"
    r"(?:\s*\|\s*(?:tail|head|grep|cat)\b[^|;&<>]*)*"
    r"(?:\s*2>&1)?\s*$"
)


# A chained prefix (`rm ... && bash ...`) is judged by WHAT IT TOUCHES, not by the
# mere fact that it chains. The blanket "any chaining is a forgery attempt" rule this
# replaces was inherited verbatim from polar_zxp, where no landed session ever exercised
# it; measured here on run 165820 it rejected 134 genuine pipeline runs to catch ONE
# budget reset — 47x deleting a self-check hash to force a real re-run, 41x clearing the
# eval workdir, 34x `tar`-ing the submission before scoring (the flow the task prompt
# itself prescribes). Cost of that: 23/184 trainable sessions lost attempts, and in 6 of
# them the excluded run had scored HIGHER than anything counted, so `best_ordinal` — and
# with it the post-best mask boundary — landed on the wrong turn. One session peaked at
# 0.760 while the spans recorded a 0.300 ceiling, masking the entire climb.
#
# What actually forges a verdict is (a) running a different script or (b) rewriting the
# output text; those are closed by the `tools/` path anchor and the read-only pipe list
# above, not by banning `rm`. Budget circumvention — the one thing this rule did catch —
# is not this module's job either: the enforcing layer counts fixed-entry invocations off
# the gateway completion stream (see ascendc_eval_pipeline.sh's own note that the workdir
# counters are "第二信号 + 遥测" and the watcher path is the one the agent cannot reach).
#
# So the prefix must be a pure file operation (nothing that writes text to stdout, which
# is the only way to smuggle a fake verdict line into the tool result) AND must keep away
# from three things: the budget counter, the shared NPU lock dir, and tools/ itself.
_SEGMENT_SPLIT_RE = re.compile(r"\s*(?:&&|;)\s*")
_PREFIX_CMD_RE = re.compile(
    r"^(?:rm|tar|cp|mv|mkdir|touch|chmod|true)\b"
    r"|^cd\s+(?!\S*\.\.)/\S*\s*$"
    # `export ASCEND_HOME=... && bash ...` — same allow-list as the inline env prefix
    # inside _PIPELINE_CMD_RE; it just lands in its own segment once we split on `&&`.
    rf"|^(?:export\s+)?{_SAFE_ENV_NAME}=\S+\s*$"
)
_PREFIX_FORBIDDEN_RE = re.compile(
    r"\.budget|budget\.json"          # 预算计数器 -> 绕过 attempt 预算
    r"|/dev/shm/npu-locks"             # 别的会话的 NPU 租约 -> 抢卡
    r"|(?:^|[\s/])tools/"              # pipeline 脚本自身 -> 直接伪造 verdict
)


def is_pipeline_invocation(command: str) -> bool:
    """True iff this shell line can only have run the real pipeline and shown its
    output verbatim. The invocation must be the LAST segment (a trailing
    ``; echo '[ascendc-eval] ...'`` would otherwise inject a forged verdict), and every
    preceding segment must be a harmless file operation."""
    segments = _SEGMENT_SPLIT_RE.split(_normalize_command(command))
    if not _PIPELINE_CMD_RE.match(segments[-1]):
        return False
    return all(
        _PREFIX_CMD_RE.match(seg) and not _PREFIX_FORBIDDEN_RE.search(seg)
        for seg in segments[:-1]
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


def bash_calls(assistant_msg: dict[str, Any]) -> list[tuple[str, str]]:
    """``[(tool_call_id, command)]`` for every Bash call in an assistant message."""
    if not isinstance(assistant_msg, dict) or assistant_msg.get("role") != "assistant":
        return []
    out: list[tuple[str, str]] = []
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
        out.append((tc.get("id"), str(command)))
    return out


def pipeline_tool_call_id(assistant_msg: dict[str, Any]) -> str | None:
    """Return the tool_call id iff this assistant message ran the exact pipeline
    command (whitelisted), else None."""
    for call_id, command in bash_calls(assistant_msg):
        if is_pipeline_invocation(command):
            return call_id
    return None


def _as_text(tool_content: Any) -> str | None:
    """Tool-result content as text; some harnesses wrap it in a list of parts."""
    if isinstance(tool_content, str):
        return tool_content
    if isinstance(tool_content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in tool_content
        )
    return None


# Harness-generated acknowledgement when a long command is auto-backgrounded.
# The AscendC pipeline compiles + takes an NPU lease + runs msprof, so it routinely
# exceeds the Bash timeout and lands here — the launching turn then carries no
# verdict at all (see `claim_backgrounded_verdicts`).
_BG_ACK_RE = re.compile(
    r"Command running in background with ID:\s*(?P<id>[A-Za-z0-9_-]+)"
    r"(?:.{0,200}?Output is being written to:\s*(?P<path>\S+?)\.?(?:\s|$))?",
    re.S,
)


def background_handle(tool_content: Any) -> tuple[str, str | None] | None:
    """``(bg_id, out_path)`` iff this tool result is the harness's background-launch
    acknowledgement, else None. Both fields are emitted by the harness, not the
    agent, so their presence cannot be forged as the precondition for a claim."""
    text = _as_text(tool_content)
    if not text:
        return None
    m = _BG_ACK_RE.search(text)
    return (m.group("id"), m.group("path")) if m else None


def parse_verdict(tool_content: Any) -> dict[str, Any] | None:
    """Parse the canonical pipeline verdict line into a metrics dict, or None."""
    tool_content = _as_text(tool_content)
    if tool_content is None:
        return None
    # LAST verdict, not the first. One pipeline run can print several of these lines
    # (an early `cached verdict` / a per-stage `verdict` before the final `done`), and
    # the last one is the outcome. Measured on run 165820: 59 tool results carry two or
    # three verdicts whose first and last disagree by a full ladder span (0.200 vs
    # 0.750). None of them is scored today because they all sit on commands the old
    # whitelist rejected — which is exactly why this has to land in the same change as
    # the widened whitelist, not after it.
    matches = list(_VERDICT_RE.finditer(tool_content))
    if not matches:
        return None
    m = matches[-1]
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


def claim_backgrounded_verdicts(
    ordinal_by_completion_id: dict[str, tuple[int, str]],
    verdict_by_call_id: dict[str, Any],
    calls_in_order: list[tuple[str, str]],
) -> int:
    """Re-attach verdicts that came back through the background-output channel.

    The AscendC pipeline routinely gets auto-backgrounded, so the launching turn's
    tool result is only ``Command running in background with ID: X ... written to:
    P`` — the verdict arrives several turns later, when the agent reads ``P``. That
    read is not a whitelisted pipeline call, so its verdict is orphaned and the
    attempt scores None (measured: ~26% of judge-successful sessions had zero
    scored attempts).

    The agent learns the outcome by following the harness-issued handle; the
    attempt follows the same link. For each attempt whose own result is a
    background ack, the LAST later tool call whose command references that
    ``bg_id``/``path`` and whose output parses as a verdict is attributed to it
    (last, not first — the agent polls the file while it is still being written).

    Anti-forge: the handle is harness-generated, so an attempt must genuinely have
    launched the pipeline before anything can be claimed for it. A forged verdict
    can still misplace *credit*, but never the reward — that comes from the
    host-side judge in a fresh container.

    Mutates ``verdict_by_call_id`` in place; returns the number of claims made.
    """
    position = {call_id: i for i, (call_id, _) in enumerate(calls_in_order)}
    claimed = 0
    for _ordinal, call_id in ordinal_by_completion_id.values():
        if parse_verdict(verdict_by_call_id.get(call_id)) is not None:
            continue
        handle = background_handle(verdict_by_call_id.get(call_id))
        if handle is None:
            continue
        bg_id, path = handle
        needles = [n for n in (path, bg_id) if n]
        start = position.get(call_id, -1)
        for other_id, command in calls_in_order[start + 1:]:
            if not any(n in command for n in needles):
                continue
            if parse_verdict(verdict_by_call_id.get(other_id)) is not None:
                verdict_by_call_id[call_id] = verdict_by_call_id[other_id]
                claimed += 1
    return claimed


def best_ordinal(
    ordinal_by_completion_id: dict[str, tuple[int, str]],
    verdict_by_call_id: dict[str, Any],
) -> int | None:
    """Trajectory-level ordinal of the attempt that PEAKED — the FIRST one to reach
    the trajectory-wide max ladder score (``>``, so a tie keeps the earlier one).

    This is the moment improvement stopped, which is what post-best masking is
    about: everything after it is work that never beat what the agent already had.

    NOT the same as "who last wrote ``{op}_impl.best.tar.gz``", and deliberately so.
    ``pack_submission.sh`` compares tiers and falls through to ``update = True`` when
    the tier merely TIES below tier 3, so at a tied failure tier the tarball is
    rewritten every time and its last writer is the LAST tied attempt. Measured: the
    max is tied in 72/138 sessions on run 092443 and 9/36 on 165820, ALWAYS at a
    failure tier (0 success-tier ties), median 3 ordinals between first and last
    tied attempt. Taking the first is the more aggressive of the two — tied-max
    sessions mask a median 33.7% of their tokens vs 2.6% for a unique max, and
    account for 67% of all masking on 165820 — but it is the boundary the masking
    means, so it is the one to keep. Anyone "fixing" this to match ``.best`` would
    be trading the intended semantics for a filesystem detail.

    Attempts whose verdict never parsed (``score=None``) are not eligible. Returns
    ``None`` when no attempt in the trajectory carries a score, in which case the
    caller masks nothing.

    Derived from the same server-side records as the spans (whitelisted command
    + harness-produced verdict line), so it inherits their anti-forge property:
    the agent cannot move this boundary by printing anything.
    """
    best_ord: int | None = None
    best_score: float | None = None
    for ordinal, call_id in sorted(ordinal_by_completion_id.values()):
        content = verdict_by_call_id.get(call_id)
        if content is None:
            continue
        metrics = parse_verdict(content)
        if metrics is None:
            continue
        score = verdict_score(metrics)
        if best_score is None or score > best_score:
            best_score, best_ord = score, ordinal
    return best_ord


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
    """POLAR_ATTEMPT_CREDIT 默认开;=0/false/no/off 关闭。

    关闭时 spans 与 post-best 掩码一并停用 —— 两者共用同一份会话级 pre-pass,
    是同一个开关的两面。
    """
    return os.environ.get("POLAR_ATTEMPT_CREDIT", "1").lower() not in ("0", "false", "no", "off")
