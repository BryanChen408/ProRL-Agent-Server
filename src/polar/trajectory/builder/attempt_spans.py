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

t3a(cannbot 复刻)检测(G1,POLAR_T3A_ATTEMPT_SPANS=1 开启,默认关——
t2a 轨迹里下列标记天然不出现,开启与否行为逐字不变):
t3a 没有只读固定入口,评测在官方 claude-harness 复刻环境内由
skill_script_hook(PreToolUse)拦截代跑。transcript 里每次评测留下:
  - assistant Bash 调用:evaluate_ascendc.sh / verification_ascendc.py
    (合法形态白名单,见 profile.t3a.yaml;python -c 内联注入形态由 hook R6 同抓);
  - 该调用的 tool_result:"[hook-noop] ..."(hook 把原命令换成了 no-op);
  - 判决块:"[skill_script_hook intercepted execution]\ncommand: ...\nexit_code: N\n
    --- stdout ---\n..." —— harness 以 additionalContext 注入,或被 agent 从
    hook-*-additionalContext.txt sidecar 读回(带 cat-n 行号前缀)。
span 边界 = 发起调用的轮次(与 t2a 相同);分数只从判决块取(绝不从普通
tool result 文本刮——那是 agent 可伪造面)。判决块按 command: 文本与
attempt 归一化匹配,FIFO 配对(同名命令重跑各归各)。伪造一个判决块只能
错配 per-attempt credit:终局 reward 来自 host 侧 judge 对 sha256 快照的
复判,信任模型与 t2a ladder 相同(防御纵深,不是最后防线)。
分类规则是 hook 侧 _classify_result 的移植(改动需两边同步:
operator_runtime_t3a/hooks/skill_script_hook.py)。
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
    # 兼容新旧 pipeline 输出行:
    #   [ascendc-eval] verdict — success=False ... error_type=... speedup_vs_torch=...  (fail_hint)
    #   [ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch=2.08 (成功路径)
    #   [ascendc-eval] cached verdict — success=... (哈希短路)
    #   [ascendc-eval] verdict — operator_valid=True task_complete=False ... (新状态机)
    #   [ascendc-eval] cached evaluation — operator_valid=True task_complete=False ...
    # operator_valid 与历史 success 的判分语义相同；task_complete 只控制 agent 能否结束，
    # 不得改变 reward ladder。
    # 成功(done)路径无 error_type 字段、且无 ast_check_ok 字段 → 两组皆可选。
    r"\[ascendc-eval\]\s*(?:verdict|done|cached verdict|cached evaluation)\b.*?"
    r"(?:success|operator_valid)=(?P<success>\S+)"
    r"(?:\s+task_complete=(?P<complete>\S+))?"
    r"(?:\s+ast_check_ok=(?P<ast>\S+))?(?:\s+correctness_ok=(?P<corr>\S+))?"
    r"(?:\s+error_type=(?P<etype>\S+))?\s+speedup_vs_torch=(?P<speedup>[-+0-9.eE]+|None|null|nan|NaN)\b"
)
# 为什么 speedup 必须是「真数字或显式空值」而不是 \S+:pipeline 自己的源码里有一行
#   echo "[ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch=$SP"
# agent 常去读脚本源码搞清楚判分规则,那一行原样进 tool result,\S+ 会把 `$SP` 也收下 ->
# 解析成「成功但 speedup 未知」-> 下面旧代码用 1.0 顶上 -> 0.75 分,凭空多出一次成功档评测。
# 实测两个 run 里这样的误判各有 6 次和 3 次(目前都落在非白名单调用上,进不了记分路径,
# 是被外围条件挡住的,解析本身认不出真假)。收紧字符类后 `$SP` / `${sp}` 直接不匹配。


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
        if speedup is None:
            # 成功档的分完全由 speedup 决定(0.75 + 0.25*tanh(ln s)),取不到数就是取不到,
            # 不能拿 1.0 顶上 —— 那等于凭空判一个 0.75。真实成功路径必然带得出数字
            # (pipeline 在 $SP 为空时走的是 benchmark FAILED 分支,根本到不了 done 行),
            # 所以到这里只说明这行不是真的运行输出。返回 None,交给既有的 score=None 兜底:
            # 位置保留、不给分、绝不编造。
            return None
        metrics["perf_data"] = {"speedup_vs_torch": speedup}
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
    score_by_call_id: dict[str, float | None] | None = None,
) -> int | None:
    """Trajectory-level ordinal of the attempt that PEAKED — the FIRST one to reach
    the trajectory-wide max ladder score (``>``, so a tie keeps the earlier one).

    This is the moment improvement stopped, which is what post-best masking is
    about: everything after it is work that never beat what the agent already had.

    The T2A ``pack_submission.sh`` now uses the same strict ``>`` reward ordering,
    so a newly produced ``.best.tar.gz`` also keeps the first tied maximum. Older
    archived sessions can still contain the former last-writer behavior, but this
    function is trajectory-derived and intentionally remains independent of that
    filesystem artifact. Measured historical context: the max was tied in 72/138
    sessions on run 092443 and 9/36 on 165820, always at a failure tier, with a
    median 3 ordinals between first and last tied attempt. The first maximum is the
    boundary post-best masking means and must remain stable across old/new runs.

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
        if score_by_call_id is not None:
            # t3a:hook 判决块已分类给分,跳过 verdict 行解析
            score = score_by_call_id.get(call_id)
            if score is None:
                continue
        else:
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
    score_by_call_id: dict[str, float | None] | None = None,
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
        if score_by_call_id is not None:
            # t3a:hook 判决块给分;缺块 = None(位置保留,不给分)
            score = score_by_call_id.get(call_id)
        else:
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


def post_best_mask_on() -> bool:
    """POLAR_POST_BEST_MASK 默认开;=0/false/no/off 关闭。

    与 POLAR_ATTEMPT_CREDIT 解耦:关掉它只停 post-best 掩码(峰值后轮次不再
    loss_mask=0),attempt_spans 与 vime 侧 attempt credit 照常 —— 峰值后段改由
    credit 的 R_e=0 负项柔和接管。用于「掩码 vs credit」的 A/B 对照。
    """
    return os.environ.get("POLAR_POST_BEST_MASK", "1").lower() not in ("0", "false", "no", "off")


# ────────────────────────────────────────────────────────────────────────
# t3a(cannbot 复刻)attempt 检测 —— 全部只在 t3a_on() 时被调用。
# ────────────────────────────────────────────────────────────────────────


def t3a_on() -> bool:
    """POLAR_T3A_ATTEMPT_SPANS=1 开启 t3a 检测;默认关(t2a 行为逐字不变)。"""
    return os.environ.get("POLAR_T3A_ATTEMPT_SPANS", "0").lower() in ("1", "true", "yes", "on")


# 合法评测调用形态(与 hook should_intercept 主匹配 + R6 内联 -c 同形,改动需同步:
# operator_runtime_t3a/hooks/skill_script_hook.py)。只取四个评测脚本;
# validate_*/build_* 不是「评测轮」,不进 span。
_T3A_SCRIPT_ALT = (
    r"(?:evaluate_ascendc\.sh|verification_ascendc\.py"
    r"|evaluate_tilelang\.sh|verification_tilelang\.py)"
)
_T3A_EVAL_CMD_RE = re.compile(
    r"^\s*"
    r"(?:export\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
    r"(?:cd\s+\S+\s*&&\s*)?"
    r"(?:export\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
    r"(?:&&\s+)*"
    r"(?:bash|sh|python|python3|python3\.\d+)\s+"
    r"(?:(?:\S*/)?" + _T3A_SCRIPT_ALT + r"(?:\s|$)"
    r"|-c\s+[\"'][\s\S]*?" + _T3A_SCRIPT_ALT + r")"
)


def is_t3a_eval_invocation(command: str) -> bool:
    """True iff this Bash command is a legal t3a evaluation call (span boundary).

    宽松于 t2a 的 is_pipeline_invocation:这里只定边界,分数另有来源(hook 判决块),
    尾部 `| tail` 之类不影响归属。读源码形态(cat/sed/grep)不匹配「解释器+脚本」
    主调,天然排除。
    """
    return bool(_T3A_EVAL_CMD_RE.search(_normalize_command(command)))


def t3a_eval_call_id(assistant_msg: dict[str, Any]) -> str | None:
    """t3a 版 pipeline_tool_call_id:该 assistant 消息发起合法评测调用则返回其 id。"""
    for call_id, command in bash_calls(assistant_msg):
        if is_t3a_eval_invocation(command):
            return call_id
    return None


# hook 判决块。stdout/stderr 经 hook truncate()(头尾保留,中间 [... truncated ...]),
# 分类标记(Result:/error:/max_abs_diff)都在头尾,截断不影响判定。
_T3A_BLOCK_RE = re.compile(
    r"\[skill_script_hook intercepted execution\]\s*\n"
    r"command: (?P<command>[\s\S]*?)\n"
    r"cwd: [^\n]*\n"
    r"exit_code: (?P<exit>-?\d+)\s*\n"
    r"duration_ms: [0-9]+\s*\n"
    r"--- stdout ---\n(?P<stdout>[\s\S]*?)\n"
    r"--- stderr ---\n(?P<stderr>[\s\S]*?)"
    r"(?=\n\[skill_script_hook intercepted execution\]|\Z)"
)
_T3A_CATN_RE = re.compile(r"(?m)^\s*\d+\t")


def t3a_blocks_from_text(text: str) -> list[tuple[str, int, str, str]]:
    """从任意消息文本抽出判决块 ``[(norm_command, exit_code, stdout, stderr)]``。

    两遍:先按原样匹配;没匹配到但含标记时,按 cat-n 行号前缀(agent 从 sidecar
    读回的形态)剥一遍再匹配。
    """
    if "skill_script_hook intercepted execution" not in text:
        return []
    blocks = [
        (_normalize_command(m.group("command")), int(m.group("exit")),
         m.group("stdout"), m.group("stderr"))
        for m in _T3A_BLOCK_RE.finditer(text)
    ]
    if not blocks:
        stripped = _T3A_CATN_RE.sub("", text)
        blocks = [
            (_normalize_command(m.group("command")), int(m.group("exit")),
             m.group("stdout"), m.group("stderr"))
            for m in _T3A_BLOCK_RE.finditer(stripped)
        ]
    return blocks


# ── 分类规则:hook _classify_result / _extract_case_stats 的移植(同步见模块 docstring)。
_T3A_A_CLASS_RE = re.compile(
    r"\berror:\s|\bfatal error:\s|undefined reference|Segmentation fault|core dumped"
    r"|cannot find -l|No such file or directory|CMake Error|make\[\d+\]: \*\*\*",
    re.IGNORECASE,
)
_T3A_PASS_RE = re.compile(r"^\s*Result:\s*(pass|fail)\s*$", re.MULTILINE | re.IGNORECASE)
_T3A_NUMERIC_RE = re.compile(
    r"max_abs_diff\s*=\s*[\d.e+\-]+|MERE\s*=\s*[\d.e+\-]+|matched_ratio\s*=\s*[\d.]+"
)
_T3A_CASE_LINE_RE = re.compile(r"case\[(\d+)\]:([^\n]*)")
_T3A_CASE_FAIL_RE = re.compile(r"mismatch|differ|FAIL|error", re.IGNORECASE)
_T3A_RATIO_RE = re.compile(r"Result:\s*(\d+)\s*/\s*(\d+)\s+passed")


def t3a_classify(exit_code: int, stdout: str, stderr: str) -> str:
    """PASS / D / A / UNKNOWN —— 与 hook _classify_result 逐行同规则。"""
    combined = f"{stdout}\n{stderr}"
    pass_matches = _T3A_PASS_RE.findall(combined)
    if pass_matches and pass_matches[-1].lower() == "pass":
        return "PASS"
    if _T3A_A_CLASS_RE.search(combined):
        return "A"
    if _T3A_NUMERIC_RE.search(combined):
        return "D"
    if exit_code != 0:
        return "A"
    return "UNKNOWN"


def t3a_case_stats(stdout: str) -> tuple[int, int]:
    """(passed, total):case[N]: 行优先;没有则退 Result: P/T passed 行。"""
    cases: dict[str, bool] = {}
    for idx, rest in _T3A_CASE_LINE_RE.findall(stdout or ""):
        cases[idx] = cases.get(idx, True) and (not _T3A_CASE_FAIL_RE.search(rest))
    if cases:
        return sum(1 for ok in cases.values() if ok), len(cases)
    m = _T3A_RATIO_RE.search(stdout or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def t3a_verdict_score(classification: str | None, case_pass: int = 0, case_total: int = 0) -> float | None:
    """hook 分类 → operator_reward ladder 同尺度分(终局 reward 同一把尺):

      PASS(本地全过) -> 0.4   correctness_ok 档;benchmark 只有 judge 可判,不进 success 档
      D(对拍跑完没对)-> 0.3 + 0.1*通过率(缺统计回退 0.35)  correctness_failed 档同公式
      A(编译/崩溃)   -> 0.2   「编译过但没能有效跑完」档
      UNKNOWN/无判决 -> None  位置保留、不给分、绝不编造(与 t2a score=None 同语义)

    全档严格低于 judge success 下限 0.5:本地 PASS ≠ success,正确性门控语义不变。
    """
    if classification == "PASS":
        return 0.4
    if classification == "D":
        if case_total > 0 and 0 <= case_pass <= case_total:
            return 0.3 + 0.1 * min(case_pass / case_total, 0.999)
        return 0.35
    if classification == "A":
        return 0.2
    return None
