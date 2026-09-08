"""Prefix-merging trajectory builder.

Reconstructs a single token-level training trace out of the many independent
LLM completions an agent emits during one rollout.  A harness (claude_code,
codex, pi, ...) drives the agent and each turn hits the gateway as a separate
completion request; this builder stitches those completions back into the
``prompt + response_1 + interstitial + response_2 + ...`` stream that an RL
trainer needs, without introducing tokenization drift.

Design in two stages:

1. **Grouping** — route each completion to the chain it append-extends, tested
   purely on tokens: a completion joins the chain whose last prompt is a prefix
   of it (``C_k.prompt_ids`` is a prefix of ``C_{k+1}.prompt_ids``).  This routes
   correctly even when parallel agents / sub-agents interleave (each has a
   distinct prompt prefix), and is robust to BPE re-tokenization because it
   compares only server-tokenized prompts, whose shared prefix is stable across
   the special-token generation-prompt boundary.  We never compare the *sampled*
   ``response_ids`` (those can re-tokenize in the next prompt, e.g.
   ``[fish, ing]`` → ``[fishing]``); a completion that extends no open chain
   starts a fresh one.

2. **Finalization** — walk each chain and build a merged token stream:

   - Assistant bodies come from the **raw** ``response_ids`` actually sampled
     by the model.  Their logprobs are real and we never decode→re-encode,
     so BPE non-canonicality cannot bite.
   - Interstitials (tool results, chat-template glue, intermediate user
     turns) come from ``C_{i+1}.prompt_ids`` — the server's **canonical**
     tokenization.  The boundary between "canonical copy of the previous
     assistant body" and the actual interstitial is the first end-of-turn
     token (``<|im_end|>`` on Qwen / ChatML; auto-detected or configurable).
   - Interstitial slots get synthesized logprobs and a zero ``loss_mask``;
     sampled assistant slots keep their real logprobs and a one ``loss_mask``.

"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder import attempt_spans as _attempt_spans
from polar.trajectory.builder.record_filters import filter_trainable_completions
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionRecord, CompletionSession, Trace, Trajectory

logger = logging.getLogger(__name__)

# finish_reasons where the model emitted the natural end-of-turn token itself.
_NATURAL_STOP_REASONS = frozenset({"stop", "tool_calls", "stop_sequence"})


def _strip_generation_prompt(tip: list[int]) -> list[int]:
    """去掉 prompt 末尾的 generation-prompt 尾(``<|im_start|>assistant[\\n<think>\\n]``)。

    空截断(finish=length + content 空 + 无 tool_calls)被 CLI 整个丢弃后,下一轮
    请求在同一位置变成 user 消息 —— 与 tip 末尾的生成头错位、判链失败(实测分叉点
    精确在 tip 末尾那 4 个 token)。生成头是模板构件(每个请求各自重新生成),
    不应参与「两条 prompt 是否同源」的判断。

    结构依据:gen-prompt 恒为 prompt 最后一段,以最后一个 ``<|im_start|>`` 开头且
    很短(<=16 token);找不到短尾时原样返回 —— 宁可不剥,不可误剥真实内容。
    """
    if len(tip) < 3:
        return tip
    im_start = tip[0]  # ChatML prompt 恒以 <|im_start|> 开头
    for i in range(len(tip) - 1, max(len(tip) - 20, 1), -1):
        if tip[i] == im_start and 0 < len(tip) - i <= 16:
            return tip[:i]
    return tip


def _is_discarded_empty_truncation(trace: "Trace") -> bool:
    """该 completion 是「会被 CLI 整个丢弃」的空截断轮:
    finish=length + content 空 + 无 tool_calls。

    实测此类轮次在下一请求历史中不存在(被 CLI 删),gateway 的 salvage 注入条件
    与此完全同形(server.py ``_salvage_message_for``)。与之区分:
    半截截断(有 content 或有 tool_calls)会被 CLI 保留进历史,正常入流。
    """
    if trace.finish_reason != "length":
        return False
    for m in trace.response_messages or []:
        if m.get("role") != "assistant":
            continue
        if (m.get("content") or "").strip():
            return False
        if m.get("tool_calls"):
            return False
    return True


def _pipeline_call_id(messages: list[dict[str, Any]]) -> str | None:
    """First whitelisted-pipeline tool-call id among a completion's response messages."""
    for m in messages or []:
        cid = _attempt_spans.pipeline_tool_call_id(m)
        if cid:
            return cid
    return None


def _prepare_attempt_span_state(kept: list[CompletionRecord]) -> dict[str, Any]:
    """Session-wide pre-pass for P3 attempt spans (plan §6.2/§6.3).

    Both products are derived ONLY from server-side records:
      - verdict_by_call_id: every tool result in every kept completion's prompt.
        Session-wide (not per-chain), so a verdict is still found when a chain
        break walls it off from the chain that issued the call — notably a
        break right after the calling turn (its verdict lands in the next
        chain's first prompt).
      - ordinal_by_completion_id: executed-attempt ordinals assigned in session
        time order AT DETECTION time — independent of verdict parsing and of any
        agent-visible counter (the printed ``[pipeline-budget] attempt=N`` comes
        from an agent-writable file and must never be used for group alignment).
      - prev_ordinal_by_completion_id: last event ordinal strictly before each
        completion (-1 when none) — resolves a no-event trace/segment's leading
        span: -1 = segment 0 (trace0), k = continuation of event k's segment.
    """
    if _attempt_spans.t3a_on():
        return _prepare_t3a_attempt_span_state(kept)
    verdict_by_call_id: dict[str, Any] = {}
    ordinal_by_completion_id: dict[str, tuple[int, str]] = {}
    prev_ordinal_by_completion_id: dict[str, int] = {}
    # Every Bash call in session order — lets a backgrounded pipeline's verdict be
    # re-attached to the attempt that launched it (see claim_backgrounded_verdicts).
    calls_in_order: list[tuple[str, str]] = []
    next_ordinal = 0
    for completion in kept:
        trace = build_trace_from_completion(completion)
        for message in trace.prompt_messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "tool"
                and message.get("tool_call_id")
            ):
                verdict_by_call_id[message["tool_call_id"]] = message.get("content")
        prev_ordinal_by_completion_id[completion.completion_id] = next_ordinal - 1
        for message in trace.response_messages:
            calls_in_order.extend(
                (cid, cmd) for cid, cmd in _attempt_spans.bash_calls(message) if cid
            )
        call_id = _pipeline_call_id(trace.response_messages)
        if call_id:
            ordinal_by_completion_id[completion.completion_id] = (next_ordinal, call_id)
            next_ordinal += 1
    _attempt_spans.claim_backgrounded_verdicts(
        ordinal_by_completion_id, verdict_by_call_id, calls_in_order
    )
    return {
        "verdict_by_call_id": verdict_by_call_id,
        "ordinal_by_completion_id": ordinal_by_completion_id,
        "prev_ordinal_by_completion_id": prev_ordinal_by_completion_id,
        "total_events": next_ordinal,
        # Ordinal of the attempt that wrote `.best` — trajectory-level, so a chain
        # break cannot lose the boundary. None -> no scored attempt -> nothing to mask.
        "best_ordinal": _attempt_spans.best_ordinal(
            ordinal_by_completion_id, verdict_by_call_id
        ),
    }


def _t3a_message_text(message: dict[str, Any]) -> str:
    """消息文本化:str content 直取;list content 递归拼 text/tool_result 文本件。"""

    def _text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                _text_of(p.get("text", p.get("content")) if isinstance(p, dict) else p)
                for p in content
            )
        return ""

    return _text_of(message.get("content"))


_T3A_SIDECAR_RE = re.compile(r"hook-(tool_[A-Za-z0-9_-]+)-\d+-additionalContext\.txt")


def _t3a_sidecar_reads(assistant_msg: dict[str, Any]) -> list[tuple[str, str]]:
    """assistant 消息里读 hook sidecar 的调用:[(本次调用 id, sidecar 文件名里的原调用 id)]。

    sidecar 文件名 hook-<tool_call_id>-N-additionalContext.txt 自带调用身份——
    这是判决块与 attempt 的真实锚点:同名命令重跑(编译挂→修复→通过)按命令文本
    FIFO 会把后一次 attempt 配上前一次的旧判决(审查实证 0.2/0.4/0.2 错配),
    身份绑定后只有无 sidecar 途径的块(harness 内联注入)才退到 FIFO。
    """
    out: list[tuple[str, str]] = []
    if not isinstance(assistant_msg, dict) or assistant_msg.get("role") != "assistant":
        return out
    for tc in assistant_msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                continue
        if not isinstance(args, dict):
            continue
        text = str(args.get("file_path") or "") + str(args.get("command") or "")
        m = _T3A_SIDECAR_RE.search(text)
        if m and tc.get("id"):
            out.append((tc["id"], m.group(1)))
    return out


def _prepare_t3a_attempt_span_state(kept: list[CompletionRecord]) -> dict[str, Any]:
    """t3a(cannbot 复刻)版 session 级 pre-pass —— 与 t2a 版同构,只在
    POLAR_T3A_ATTEMPT_SPANS=1 时被 _prepare_attempt_span_state 调用。

    差异仅在判决来源:t2a 从固定入口的 verdict 行解析;t3a 从 hook 判决块
    (``[skill_script_hook intercepted execution]``)解析。块的归属按信任优先级:
    ① sidecar 文件名身份绑定(hook-<tool_call_id>-N-...txt);② 剩余块按命令文本
    FIFO(仅 harness 内联注入形态走这里)。全部块按 (命令,exit,stdout) 全局去重——
    嵌套历史会让同一块在每个后续 completion 重复目击(审查实证:连续折叠不够,
    同输出真重跑宁可少记一次,也绝不错配)。产物与 t2a 版同形,外加
    ``score_by_call_id`` 直给分(build_spans/best_ordinal 走 score 快路)。
    """
    verdict_by_call_id: dict[str, Any] = {}
    ordinal_by_completion_id: dict[str, tuple[int, str]] = {}
    prev_ordinal_by_completion_id: dict[str, int] = {}
    attempts: list[tuple[int, str, str]] = []  # (ordinal, call_id, norm_command)
    blocks: list[tuple[str, float | None]] = []  # 未绑定块 FIFO 池(全局去重后)
    bound_scores: dict[str, float | None] = {}  # sidecar 身份绑定:target_call_id → score
    sidecar_bound: dict[str, str] = {}  # 读 sidecar 的调用 id → 原调用 id
    seen_blocks: set[tuple[str, int, int]] = set()
    next_ordinal = 0
    for completion in kept:
        trace = build_trace_from_completion(completion)
        for message in trace.prompt_messages:
            if not isinstance(message, dict):
                continue
            bound_id = None
            if message.get("role") == "tool" and message.get("tool_call_id"):
                verdict_by_call_id[message["tool_call_id"]] = message.get("content")
                bound_id = sidecar_bound.get(message["tool_call_id"])
            text = _t3a_message_text(message)
            for norm_cmd, exit_code, stdout, stderr in _attempt_spans.t3a_blocks_from_text(text):
                cls = _attempt_spans.t3a_classify(exit_code, stdout, stderr)
                cp, ct = _attempt_spans.t3a_case_stats(stdout)
                score = _attempt_spans.t3a_verdict_score(cls, cp, ct)
                if bound_id:
                    # 身份绑定块不去重:sidecar 文件名即身份——内容相同但来自不同
                    # sidecar 的块是不同事件(同输出重跑各自归各);同一 read 结果在
                    # 嵌套历史里重复目击只是重复赋值同一目标,幂等无害。
                    bound_scores[bound_id] = score
                    continue
                key = (norm_cmd, exit_code, hash(stdout))
                if key in seen_blocks:
                    continue
                seen_blocks.add(key)
                blocks.append((norm_cmd, score))
        prev_ordinal_by_completion_id[completion.completion_id] = next_ordinal - 1
        call_id = None
        for message in trace.response_messages:
            for read_id, target_id in _t3a_sidecar_reads(message):
                sidecar_bound[read_id] = target_id
            if call_id is None:
                call_id = _attempt_spans.t3a_eval_call_id(message)
        if call_id:
            cmd = next(
                (c for cid, c in _attempt_spans.bash_calls(message) if cid == call_id), ""
            )
            ordinal_by_completion_id[completion.completion_id] = (next_ordinal, call_id)
            attempts.append((next_ordinal, call_id, _attempt_spans._normalize_command(cmd)))
            next_ordinal += 1
    # 配对:身份绑定优先;未绑定 attempt 按命令文本从 FIFO 池取(内联注入形态)
    score_by_call_id: dict[str, float | None] = dict(bound_scores)
    queues: dict[str, list[float | None]] = {}
    for norm_cmd, score in blocks:
        queues.setdefault(norm_cmd, []).append(score)
    for _ordinal, call_id, norm_cmd in attempts:
        if call_id in score_by_call_id:
            continue
        q = queues.get(norm_cmd)
        if q:
            score_by_call_id[call_id] = q.pop(0)
    return {
        "verdict_by_call_id": verdict_by_call_id,
        "ordinal_by_completion_id": ordinal_by_completion_id,
        "prev_ordinal_by_completion_id": prev_ordinal_by_completion_id,
        "total_events": next_ordinal,
        "score_by_call_id": score_by_call_id,
        "best_ordinal": _attempt_spans.best_ordinal(
            ordinal_by_completion_id, verdict_by_call_id, score_by_call_id
        ),
    }


def _upstream_failure_truncated_session(
    metadata: dict[str, Any], completion_count: int
) -> bool:
    """True iff the LAST upstream blip left the session with nothing after it.

    A blip mid-session is recoverable and routinely recovered: the CLI retries, the
    session runs on and finishes naturally, and the judge scores the real work. A blip
    that is the session's last event is the opposite -- generation stopped there, so the
    judge scored a truncated attempt and its reward reflects our plumbing, not the
    agent. That truncation is the false negative the counter was introduced for.

    ``upstream_failures_last_at`` is the saved-completion count at blip time, so
    ``completion_count > last_at`` means at least one turn was persisted afterwards.
    Absent (sessions recorded before the field existed) -> not treated as truncating:
    the measured base rate is 244/252 recovered, so keeping is the better default, and
    every session recorded from now on carries the field.

    A 4xx is exempt even when terminal. It is the engine REJECTING the request before
    generating anything (context exhausted, malformed request), so it never becomes a
    CompletionRecord and the trajectory's last turn is a complete one -- nothing was
    cut off. The session simply ran out of budget, which is a legitimate end to the
    episode: the judge scored the agent's real best submission, not a fragment. Only
    transport/timeout/5xx can leave a half-generated turn behind. Measured on run
    165820: all 3 discarded sessions were terminal `http_400` at 249856 prompt tokens,
    each carrying a real judge verdict (0.25/0.29/0.30) and 50k-86k trainable tokens --
    the LARGEST sample in its group every time, because context exhaustion selects for
    the longest episodes. Discarding them biased training toward short sessions.
    """
    raw = metadata.get("upstream_failures_last_at")
    if raw is None:
        return False
    try:
        last_at = int(raw)
    except (TypeError, ValueError):
        return False
    if completion_count > last_at:
        return False
    return not _blip_kind_is_client_rejection(metadata)


def _blip_kind_is_client_rejection(metadata: dict[str, Any]) -> bool:
    """True iff the session's LAST upstream blip was a 4xx (rejected pre-generation).

    Prefers ``upstream_failures_last_kind`` (exact). Sessions recorded before that
    field existed fall back to the kind->count map, which can only answer this when
    every kind in it is a 4xx -- a mixed session (16 of 150 in run 092443) keeps the
    conservative "may have truncated" reading.
    """
    kind = metadata.get("upstream_failures_last_kind")
    if isinstance(kind, str) and kind:
        return kind.startswith("http_4")
    kinds = metadata.get("upstream_failures")
    if not isinstance(kinds, dict) or not kinds:
        return False
    return all(str(k).startswith("http_4") for k in kinds)


def _completion_finish_reason(completion: CompletionRecord) -> str | None:
    """Raw per-completion finish_reason from a completion record (pre-merge).

    dev_09: any completion with finish_reason=="abort" (weight-update cutoff)
    marks the WHOLE session non-trainable via trajectory.status="ERROR".
    """
    resp = completion.response if isinstance(completion.response, dict) else {}
    choices = resp.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    return first.get("finish_reason")


def _completion_policy_version(completion: CompletionRecord) -> int | None:
    """Live weights version when this turn was generated, stamped by the gateway at
    completion-record time (``metadata['policy_version']``).  A session whose raw
    completions span >1 version crossed a weight update mid-interaction (mixed-weight).
    """
    meta = completion.metadata if isinstance(completion.metadata, dict) else {}
    v = meta.get("policy_version")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _session_upstream_failures(metadata: dict[str, Any]) -> dict[str, int]:
    """Upstream (inference engine) failures stamped on the session by the gateway.

    A 5xx from the engine raises before the completion is saved, so it leaves NO
    CompletionRecord — the session merely looks shorter and gets scored as if the agent
    had produced that truncated work.  Measured: 56/64 sessions across three ascendc runs
    ended on ``API Error: 502 Upstream request failed`` yet were stored COMPLETED with
    reward 0.2, i.e. an engine outage became a real training signal.

    ``SessionStore.record_upstream_failure`` counts them into session metadata. Whether
    a blip is fatal depends on WHERE it landed -- see
    ``_upstream_failure_truncated_session``.
    """
    raw = metadata.get("upstream_failures")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            out[str(key)] = count
    return out


def _session_policy_versions(completions: list[CompletionRecord]) -> set[int]:
    """Distinct policy_versions across a session's raw completions (>1 == mixed-weight)."""
    return {v for c in completions if (v := _completion_policy_version(c)) is not None}


@dataclass(frozen=True, slots=True)
class _FinalizedChain:
    trace: Trace
    kept_count: int
    break_reason: str | None = None


def _chain_system_key(chain: list[CompletionRecord]) -> str | None:
    """链的 system 指纹:取链首请求的第一条 system 消息内容做角色判定。

    主链与其 compaction 续段共享同一 system prompt(判同角色);Skill/Agent
    派发的子会话 system 不同(判 sub)。没有 system 消息时取第一条消息内容;
    消息为空返回 None(不参与角色判定,链按 sub 处理)。
    """
    if not chain:
        return None
    messages = build_trace_from_completion(chain[0]).prompt_messages or []
    if not messages:
        return None
    first = messages[0] if isinstance(messages[0], dict) else {}
    content = first.get("content")
    if isinstance(content, list):
        content = "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    if content:
        return f"{first.get('role')}::{len(str(content))}::{hash(str(content))}"
    if len(messages) > 1 and isinstance(messages[1], dict):
        c2 = messages[1].get("content")
        if isinstance(c2, list):
            c2 = "".join(str(b.get("text", "")) for b in c2 if isinstance(b, dict))
        if c2:
            return f"{messages[1].get('role')}::{len(str(c2))}::{hash(str(c2))}"
    return None


class PrefixMergingBuilder(BaseTrajectoryBuilder):
    """Rebuild a chain's merged token stream using raw + canonical-interstitial.

    Parameters
    ----------
    end_of_turn_token_id:
        Explicit end-of-turn (EOT) token id used to locate the
        canonical-tail split between the prior assistant body and the
        interstitial.  When None (default), the builder auto-detects it
        from the last token of the first completion with a natural stop
        reason.  For Qwen / ChatML templates this is the
        ``<|im_end|>`` token id.
    """

    def __init__(
        self,
        *,
        end_of_turn_token_id: int | None = None,
    ) -> None:
        self._configured_eot_id = end_of_turn_token_id

    async def build(self, session: CompletionSession) -> Trajectory:
        filter_result = filter_trainable_completions(session.completions)
        if not session.completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "prefix_merging",
                    "session_id": session.session_id,
                    "task_metadata": dict(session.metadata),
                    "record_count": 0,
                    **_top_level_scheduler_metadata(session.metadata),
                },
                traces=[],
                error="no completions",
            )
        if not filter_result.kept:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "prefix_merging",
                    "session_id": session.session_id,
                    "task_metadata": dict(session.metadata),
                    "record_count": len(session.completions),
                    "completion_filter": filter_result.metadata,
                    **_top_level_scheduler_metadata(session.metadata),
                },
                traces=[],
                error="no trainable completions after completion filter",
            )

        chains: list[list[CompletionRecord]] = []
        chain_tips: list[list[int]] = []  # last completion's prompt_ids, per chain
        # Per chain: does it CONTINUE an earlier conversation (history rewritten /
        # a pure-user injection moved the query point), or is it an INDEPENDENT
        # sub-conversation (Skill / Agent dispatch, ~0 shared prefix)? Only the
        # former is causally downstream of an earlier chain, so only the former may
        # be masked wholesale by post-best. Measured on 80 sessions: independent
        # sub-conversations are 31% of splits, and 17% of multi-chain sessions
        # interleave — masking those by session time order alone is wrong.
        chain_continues: list[bool] = []

        for completion in filter_result.kept:
            prompt_ids = build_trace_from_completion(completion).prompt_ids
            chain_idx = self._find_extendable_chain(prompt_ids, chain_tips)
            if chain_idx is None:
                chain_idx = len(chains)
                chains.append([])
                chain_tips.append([])
                chain_continues.append(self._continues_existing(prompt_ids, chain_tips))
            chains[chain_idx].append(completion)
            chain_tips[chain_idx] = prompt_ids

        stats: dict[str, Any] = {
            "chains_total": len(chains),
            "chains_reconstructed_full": 0,
            "chains_reconstructed_truncated": 0,
            "raw_completions_total": len(session.completions),
            "completions_total": len(filter_result.kept),
            "completions_merged": 0,
            "completions_preserved": 0,
            "completions_dropped": 0,
            "break_reasons": {},
        }
        # chain_role:主链角色判定。以会话首链的 system 消息为基准,system 相同
        # 的链视为同一「主」角色(compaction 续段与主链共享 system,自然归入);
        # 不同 system 的链 = 独立子会话(Skill/Agent 派发)。只写 metadata,
        # 是否按角色掩码由下游 adapter 决定(默认不动)。
        main_system_key = _chain_system_key(chains[0]) if chains else None
        final_traces: list[Trace] = []
        # P3 stage-2: session-wide attempt-span state (env-gated,默认开;
        # POLAR_ATTEMPT_CREDIT=0 关). Created ONCE per trajectory and shared by
        # every chain/segment finalization so event ordinals are trajectory-level
        # and verdicts pair across chain breaks.
        span_state = (
            _prepare_attempt_span_state(filter_result.kept)
            if _attempt_spans.env_on()
            else None
        )
        for chain_index, chain in enumerate(chains):
            start = 0
            segment_index = 0
            chain_had_break = False
            chain_role = (
                "main" if main_system_key is not None
                and _chain_system_key(chain) == main_system_key
                else "sub"
            )
            while start < len(chain):
                finalized = self._finalize_chain(
                    chain[start:],
                    chain_index=chain_index,
                    chain_length=len(chain),
                    segment_index=segment_index,
                    segment_start=start,
                    span_state=span_state,
                    chain_continues=chain_continues[chain_index],
                    chain_role=chain_role,
                )
                # A trace that post-best masking emptied is dropped so
                # trajectory_trace_counts stays honest. The masked_tokens guard is
                # load-bearing: a trace that was ALREADY all-zero (degenerate empty
                # responses) must still be emitted, or flipping the flag on would
                # silently drop traces this feature never touched.
                if finalized.trace.metadata.get("post_best_masked_tokens") and not any(
                    finalized.trace.loss_mask
                ):
                    stats["traces_dropped_post_best"] = (
                        stats.get("traces_dropped_post_best", 0) + 1
                    )
                else:
                    final_traces.append(finalized.trace)
                stats["completions_preserved"] += finalized.kept_count
                if finalized.kept_count > 1:
                    stats["completions_merged"] += finalized.kept_count
                if finalized.break_reason:
                    chain_had_break = True
                    self._increment_break_reason(stats, finalized.break_reason)
                start += finalized.kept_count
                segment_index += 1

            if chain_had_break:
                stats["chains_reconstructed_truncated"] += 1
            else:
                stats["chains_reconstructed_full"] += 1

        # dev_09: any raw completion aborted (weight-update cutoff) -> whole session
        # is non-trainable. status="ERROR" propagates via gateway (SessionResult.status
        # = trajectory.status) to slime, which only trains status=="COMPLETED" sessions
        # -> the aborted session is dropped and oversampling backfills a clean one.
        session_had_abort = any(
            _completion_finish_reason(c) == "abort" for c in session.completions
        )
        # version-span fallback (behind the gateway entry-interception): a session whose
        # raw completions were generated under >1 policy_version crossed a weight update
        # mid-interaction (mixed-weight).  Precise -- only true spans, no false kills.
        session_versions = _session_policy_versions(session.completions)
        session_spanned = len(session_versions) > 1
        upstream_failures = _session_upstream_failures(dict(session.metadata))
        # Only POLICY-IDENTITY failures are fatal here. An aborted or version-spanning
        # session has no single behaviour policy, so its importance ratio has no
        # well-defined denominator and no amount of reward validity rescues it.
        #
        # An upstream transport blip is a different question, and folding it in here
        # answered that question wrongly. Measured on run 092443 (198 sessions, 252
        # blips): 97% of blips are followed by a median of 30 MORE turns -- the CLI
        # retries, the session recovers and finishes naturally (94/119 end on
        # finish_reason=stop with a wrap-up summary and no tool call), and the judge
        # scores it in a fresh container (106/106 had verdicts, 56 success=True, 29 at
        # reward >= 0.7). A failed request leaves NO CompletionRecord at all (the
        # gateway records the counter and returns before save_message), so the trace is
        # a gap, never a half-written turn -- every recorded turn is complete.
        #
        # 118/198 sessions were being discarded for this, 57 of them on a single blip.
        # And discarding is not the neutral choice: blip rate rises monotonically with
        # context length (33% at 50-75k prompt tokens -> 84% at 200-225k), so it
        # selectively removed the long, deep-iteration, high-reward sessions and biased
        # the surviving group baseline upward. Keeping them is unbiased.
        #
        # Genuine infra failures still get caught downstream: the evaluator's
        # `judge_outcome` maps INFRA_ERROR_TYPES / missing metrics to status=ERROR +
        # retry, and a session that produced no trainable completions is already
        # rejected above. `upstream_failures` stays in metadata below, and the gateway
        # still logs every blip, so nothing becomes invisible.
        upstream_truncated = bool(upstream_failures) and _upstream_failure_truncated_session(
            dict(session.metadata), len(session.completions)
        )
        _non_trainable = session_had_abort or session_spanned or upstream_truncated
        if session_had_abort:
            _span_error: str | None = "aborted generation (weight-update cutoff)"
        elif session_spanned:
            _span_error = f"policy_version span (mixed-weight): {sorted(session_versions)}"
        elif upstream_truncated:
            _span_error = f"upstream engine failure truncated the session: {upstream_failures}"
        else:
            _span_error = None
        # 截断事件计数(completion 级,供截断惩罚):空截断轮修复后不再单独成
        # trace(response 不入流),按 trace 数会漏计 —— 改从原始 completion 统计,
        # operator_judge 优先读这个字段。
        truncation_events = sum(
            1
            for c in session.completions
            if _is_discarded_empty_truncation(build_trace_from_completion(c))
        )
        return Trajectory(
            status="ERROR" if _non_trainable else "COMPLETED",
            error=_span_error,
            metadata={
                "builder": "prefix_merging",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(session.completions),
                "task_metadata": dict(session.metadata),
                "trace_count": len(final_traces),
                "reconstruction_stats": stats,
                "completion_filter": filter_result.metadata,
                "upstream_failures": upstream_failures or None,
                "truncation_events": truncation_events,
                **_top_level_scheduler_metadata(session.metadata),
            },
            traces=final_traces,
        )

    # ------------------------------------------------------------------
    # Chain finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _continues_existing(prompt_ids: list[int], chain_tips: list[list[int]]) -> bool:
        """Does this new chain continue an existing conversation?

        A continuation shares most of some open chain's prompt (the split came from
        a history rewrite or a pure-user injection that moved the query point); an
        independent sub-conversation shares essentially nothing. Testing the first
        HALF of a tip separates the two cleanly — measured shared-prefix fractions
        cluster at >90% (continuation) vs <5% (independent) — and it is one
        C-level slice compare rather than a token-by-token walk over 100k+ prompts.
        """
        for tip in chain_tips:
            half = len(tip) // 2
            if half and len(prompt_ids) >= half and prompt_ids[:half] == tip[:half]:
                return True
        return False

    @staticmethod
    def _post_best_pos(
        chain: list[CompletionRecord], span_state: dict, continues: bool
    ) -> int | None:
        """Chain position AFTER WHICH every turn is post-best — i.e. the turn just
        before the first attempt LATER than the best one. ``-1`` = the whole chain is
        post-best; ``None`` = nothing here is (the best attempt's segment is still
        open at the chain's end, no scored attempt at all, or this chain is not
        causally downstream of the best one).

        The boundary is the NEXT attempt's calling turn, NOT the best attempt's own.
        The segment model folds the work turns between attempt e and e+1 into e's
        segment (``事件 e 之间的对话轮归入事件段 e``), and ``attempt_credit`` credits
        segment e with the reward-to-go those turns go on to produce — so they are the
        best attempt's own credit-bearing region, not its aftermath. Cutting at the
        best attempt's own turn split that region in half and put the two mechanisms
        in direct contradiction: measured on run 165820, 1.09M of the 2.74M masked
        tokens sat INSIDE the best attempt's own segment, and 0.51M of those carried a
        POSITIVE attempt-credit term that masking then deleted.

        Aligning on the segment boundary makes the two agree by construction: within
        the best segment reward-to-go is positive (credit rewards it, masking keeps
        it); from the next attempt on, Δ-best makes reward-to-go 0, so credit's term
        is ``-baseline`` <= 0 and masking removes those tokens outright. Masking is
        still doing real work there — 1.65M tokens on the same run, about half of them
        at TRLOO positions too thin for credit to score at all, which is precisely the
        blind spot it exists to cover.

        Resolved from the trajectory-level pre-pass, so a chain split between the best
        attempt and its aftermath cannot lose the boundary — but only for a chain that
        CONTINUES the earlier conversation. An independent sub-conversation merely
        started later in session time; masking it wholesale would punish work the best
        attempt never caused.
        """
        best = span_state.get("best_ordinal")
        if best is None or not chain:
            return None
        by_cid = span_state["ordinal_by_completion_id"]

        def _opens_later_attempt() -> int | None:
            """Chain position of the first turn opening an attempt after ``best``.

            Ordinals are assigned in session time order and a chain's completions keep
            that order, so such a turn can only sit after the best attempt's own.
            """
            for pos, completion in enumerate(chain):
                event = by_cid.get(completion.completion_id)
                if event is not None and event[0] > best:
                    return pos
            return None

        for completion in chain:
            event = by_cid.get(completion.completion_id)
            if event is not None and event[0] == best:
                nxt = _opens_later_attempt()
                return None if nxt is None else nxt - 1
        if not continues:
            return None
        prev = span_state["prev_ordinal_by_completion_id"].get(chain[0].completion_id, -1)
        if prev > best:
            # A later attempt already opened before this chain started, so every turn
            # here belongs to a post-best segment.
            return -1
        if prev < best:
            return None
        # prev == best: this chain starts INSIDE the best attempt's still-open segment.
        nxt = _opens_later_attempt()
        return None if nxt is None else nxt - 1

    def _finalize_chain(
        self,
        chain: list[CompletionRecord],
        *,
        chain_index: int,
        chain_length: int,
        segment_index: int,
        segment_start: int,
        span_state: dict | None = None,
        chain_continues: bool = False,
        chain_role: str = "sub",
    ) -> _FinalizedChain:
        # Everything in C_1.prompt_ids is the non-trainable
        # prompt; C_1.response_ids plus every subsequent raw response +
        # canonical interstitial becomes the trainable response.  No role-shape
        # constraint on the initial conversation — a harness preamble like
        # codex's [system, user, user, assistant, tool, ...] is treated as
        # static context.
        first_trace = build_trace_from_completion(chain[0])
        eot_id = self._resolve_eot_id(chain)

        prompt_ids = list(first_trace.prompt_ids)
        stream_ids: list[int] = list(prompt_ids)
        response_slots: list[float | None] = []
        loss_mask: list[int] = []
        response_messages: list[dict[str, Any]] = []

        # Track the canonical prompt_ids of the most recently merged
        # completion — used for the canonical-vs-canonical prefix check.
        prev_prompt_ids: list[int] = list(first_trace.prompt_ids)
        prev_raw_response: list[int] = list(first_trace.response_ids)

        # Running count of messages consumed = prompt_messages + all response_messages emitted.
        msg_acc = len(first_trace.prompt_messages)

        # P3 stage-2: per-attempt segment spans (env-gated; additive metadata only).
        # Detection + verdicts come from the session-wide pre-pass (span_state);
        # here we only map kept events to their response-token offsets — anchored
        # at the CALLING turn's response start (事件段从发起调用的那一轮开始).
        _want_spans = span_state is not None
        event_records: list[tuple[int, int, str]] = []
        # Post-best masking: chain position of the best attempt's calling turn;
        # every LATER turn is post-best. None -> nothing to mask here.
        # POLAR_POST_BEST_MASK 独立开关:关时峰值后段不再掩零,交由 attempt credit
        # 的 R_e=0 负项柔和接管(A/B 对照用);spans 与 credit 不受影响。
        _pb = (
            self._post_best_pos(chain, span_state, chain_continues)
            if _want_spans and _attempt_spans.post_best_mask_on()
            else None
        )
        _pb_masked = 0

        _rs = len(stream_ids) - len(prompt_ids)
        _prev_discarded = _is_discarded_empty_truncation(first_trace)
        if _prev_discarded:
            # 空截断轮(CLI 已整个丢弃):response 不入流,训练流与推理现场逐 token 一致。
            # 其思考内容仅经下一轮 prompt 的 user 消息(limit/salvage 引用)进入序列,
            # 作为 interstitial 恒掩零 —— 不产生全零 loss 孤儿 trace,前缀也不重复送。
            pass
        else:
            _pb_masked += self._append_response_tokens(
                first_trace, stream_ids, response_slots, loss_mask,
                force_zero_loss=_pb is not None and 0 > _pb,
            )
        if _want_spans:
            _ev = span_state["ordinal_by_completion_id"].get(chain[0].completion_id)
            if _ev is not None:
                event_records.append((_rs, _ev[0], _ev[1]))
        response_messages.extend(deepcopy(m) for m in first_trace.response_messages)
        msg_acc += len(first_trace.response_messages)
        kept = 1
        break_reason: str | None = None

        for i in range(1, len(chain)):
            Ci_trace = build_trace_from_completion(chain[i])
            Ci_prompt_ids = list(Ci_trace.prompt_ids)

            # Canonical-vs-canonical prefix check: both sides are server-side
            # tokenizations of the same message prefix — matches reliably
            # unless the harness rewrote prior messages. 与判链一致,比较前剥掉
            # prev prompt 末尾的生成头(空截断被丢弃后下一轮同位置是 user 消息)。
            prev_core = _strip_generation_prompt(prev_prompt_ids)
            if (
                len(Ci_prompt_ids) < len(prev_core)
                or Ci_prompt_ids[: len(prev_core)] != prev_core
            ):
                logger.debug(
                    "prefix_merging: canonical prefix break at step %d/%d",
                    i,
                    len(chain),
                )
                break_reason = "canonical_prefix_break"
                break

            # canonical_tail = canonical tokens for [prev assistant msg + new interstitials].
            canonical_tail = Ci_prompt_ids[len(prev_core):]
            if eot_id is None:
                logger.debug(
                    "prefix_merging: eot unavailable at step %d/%d",
                    i,
                    len(chain),
                )
                break_reason = "eot_unavailable"
                break
            if _prev_discarded:
                # prev 是空截断轮(CLI 已丢):canonical_tail 里没有 prev assistant 体,
                # 全部内容(limit/salvage user 消息 + gen glue)都是 interstitial。
                interstitial = list(canonical_tail)
            else:
                interstitial = self._slice_interstitial(
                    canonical_tail=canonical_tail,
                    prev_raw_response=prev_raw_response,
                    eot_id=eot_id,
                )
            if interstitial is None:
                logger.debug(
                    "prefix_merging: interstitial split failed at step %d/%d "
                    "(eot_id=%r, tail_len=%d)",
                    i,
                    len(chain),
                    eot_id,
                    len(canonical_tail),
                )
                break_reason = "interstitial_split_failed"
                break

            if interstitial:
                stream_ids.extend(interstitial)
                response_slots.extend([None] * len(interstitial))
                loss_mask.extend([0] * len(interstitial))

            # Message-level interstitial bookkeeping.
            if len(Ci_trace.prompt_messages) > msg_acc:
                interstitial_msgs = Ci_trace.prompt_messages[msg_acc:]
                response_messages.extend(deepcopy(m) for m in interstitial_msgs)
                msg_acc += len(interstitial_msgs)

            _rs = len(stream_ids) - len(prompt_ids)
            _prev_discarded = _is_discarded_empty_truncation(Ci_trace)
            if _prev_discarded:
                pass  # 空截断轮 response 不入流(见上)
            else:
                _pb_masked += self._append_response_tokens(
                    Ci_trace, stream_ids, response_slots, loss_mask,
                    force_zero_loss=_pb is not None and i > _pb,
                )
            if _want_spans:
                _ev = span_state["ordinal_by_completion_id"].get(chain[i].completion_id)
                if _ev is not None:
                    event_records.append((_rs, _ev[0], _ev[1]))
            response_messages.extend(deepcopy(m) for m in Ci_trace.response_messages)
            msg_acc += len(Ci_trace.response_messages)

            prev_prompt_ids = Ci_prompt_ids
            # 丢弃轮的 raw response 视为不存在,下一轮的 interstitial 取全段 canonical_tail
            prev_raw_response = [] if _prev_discarded else list(Ci_trace.response_ids)
            kept += 1

        response_ids = stream_ids[len(prompt_ids):]
        response_logprobs = self._finalize_logprobs(response_slots)
        last_kept_trace = build_trace_from_completion(chain[kept - 1])

        _metadata = self._chain_metadata(
            chain[:kept],
            chain_index=chain_index,
            chain_length=chain_length,
            segment_index=segment_index,
            segment_start=segment_start,
            kept_completion_count=kept,
            break_reason=break_reason,
        )
        _metadata["chain_role"] = chain_role
        if _want_spans:
            # Segment spans (plan §6.2): the opening span covers everything before
            # this trace's first event — idx -1 = segment 0 (e.g. the Skill-dispatch
            # trace0), idx k = continuation of event k's segment after a break.
            # Each event's span extends to the next event's start / trace end, so
            # every response token belongs to exactly one segment. Trajectories
            # with zero events keep no key -> trajectory-level fallback downstream.
            chain_resp_end = len(stream_ids) - len(prompt_ids)
            if event_records:
                _leading = -1 if event_records[0][1] == 0 else event_records[0][1] - 1
            elif span_state["total_events"] > 0:
                _leading = span_state["prev_ordinal_by_completion_id"].get(
                    chain[0].completion_id, -1
                )
            else:
                _leading = None
            _spans = _attempt_spans.build_spans(
                event_records,
                span_state["verdict_by_call_id"],
                _leading,
                chain_resp_end,
                score_by_call_id=span_state.get("score_by_call_id"),
            )
            if _spans:
                _metadata["attempt_spans"] = _spans

        if _pb_masked:
            _metadata["post_best_masked_tokens"] = _pb_masked

        trace = Trace(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            loss_mask=loss_mask,
            prompt_messages=[deepcopy(m) for m in first_trace.prompt_messages],
            response_messages=response_messages,
            tools=deepcopy(first_trace.tools),
            finish_reason=last_kept_trace.finish_reason,
            response_logprobs=response_logprobs,
            metadata=_metadata,
        )
        return _FinalizedChain(trace=trace, kept_count=kept, break_reason=break_reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_eot_id(self, chain: list[CompletionRecord]) -> int | None:
        """Return configured EOT id, else auto-detect from the chain.

        Auto-detection uses the last token of the first completion whose
        ``finish_reason`` indicates the model emitted the natural stop
        marker itself (stop / tool_calls / stop_sequence).
        """
        if self._configured_eot_id is not None:
            return self._configured_eot_id
        for completion in chain:
            trace = build_trace_from_completion(completion)
            if (
                trace.finish_reason in _NATURAL_STOP_REASONS
                and trace.response_ids
            ):
                return trace.response_ids[-1]
        return None

    @staticmethod
    def _slice_interstitial(
        *,
        canonical_tail: list[int],
        prev_raw_response: list[int],
        eot_id: int | None,
    ) -> list[int] | None:
        """Extract the canonical interstitial from C_{i+1}'s prompt tail.

        ``canonical_tail`` = canonical tokens for [prev assistant msg +
        harness-inserted messages + generation-prompt glue].  The first
        occurrence of ``eot_id`` marks the end of the prev assistant
        body; everything after is interstitial.

        If ``prev_raw_response`` already ends with ``eot_id`` (natural
        stop / tool_calls), skip it in the canonical tail to avoid
        duplication; otherwise (truncation) include it so the stream
        still closes the assistant turn.

        Returns None if ``eot_id`` is unknown or not present — caller
        should treat this as a break.
        """
        if eot_id is None:
            return None
        try:
            k = canonical_tail.index(eot_id)
        except ValueError:
            return None
        if prev_raw_response and prev_raw_response[-1] == eot_id:
            return canonical_tail[k + 1 :]
        return canonical_tail[k:]

    @staticmethod
    def _append_response_tokens(
        trace: Trace,
        stream_ids: list[int],
        response_slots: list[float | None],
        loss_mask: list[int],
        force_zero_loss: bool = False,
    ) -> int:
        """Append a completion's response_ids and parallel logprob slots.

        ``force_zero_loss`` (a post-best turn) appends loss_mask=0 for the whole
        turn — ``response_ids`` and the logprob slots still go in at full length,
        so array LENGTHS never change (CP layout and rollout-logprob alignment are
        preserved); only mask values flip. Returns the number of tokens thereby
        suppressed (0 otherwise).
        """
        response_ids = list(trace.response_ids)
        stream_ids.extend(response_ids)
        trace_loss_mask = list(trace.loss_mask) or [1] * len(response_ids)
        if len(trace_loss_mask) != len(response_ids):
            raise ValueError("trace loss_mask length must match response_ids length")
        suppressed = 0
        if force_zero_loss:
            suppressed = sum(trace_loss_mask)
            trace_loss_mask = [0] * len(response_ids)
        loss_mask.extend(trace_loss_mask)
        logprobs = trace.response_logprobs or []
        for pos in range(len(response_ids)):
            value = logprobs[pos] if pos < len(logprobs) else None
            response_slots.append(float(value) if isinstance(value, (int, float)) else None)
        return suppressed

    @staticmethod
    def _finalize_logprobs(
        slots: list[float | None],
    ) -> list[float] | None:
        # Interstitial slots (tool results, chat glue) get 0.0; loss_mask=0
        # makes the trainer ignore them.
        if not any(slot is not None for slot in slots):
            return None
        return [slot if slot is not None else 0.0 for slot in slots]

    @staticmethod
    def _chain_metadata(
        chain: list[CompletionRecord],
        *,
        chain_index: int,
        chain_length: int,
        segment_index: int,
        segment_start: int,
        kept_completion_count: int,
        break_reason: str | None,
    ) -> dict[str, Any]:
        completion_metadata = [dict(completion.metadata) for completion in chain]
        merged = dict(completion_metadata[0]) if completion_metadata else {}
        merged["completion_metadata"] = completion_metadata
        merged["chain_index"] = chain_index
        merged["chain_length"] = chain_length
        merged["chain_segment_index"] = segment_index
        merged["chain_segment_start"] = segment_start
        merged["source_completion_ids"] = [completion.completion_id for completion in chain]
        merged["kept_completion_count"] = kept_completion_count
        # dev_09: had_abort removed — abort is now handled session-level via
        # trajectory.status="ERROR" in build(), not via this per-trace metadata
        # flag (which was dropped in Polar->slime serialization, never took effect).
        if break_reason:
            merged["break_reason"] = break_reason
        return merged

    @staticmethod
    def _increment_break_reason(stats: dict[str, Any], reason: str) -> None:
        reasons = stats.setdefault("break_reasons", {})
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    @staticmethod
    def _find_extendable_chain(
        prompt_ids: list[int],
        chain_tips: list[list[int]],
    ) -> int | None:
        """Return the open chain this completion append-extends, else None.

        A completion continues a chain iff its prompt begins with that chain's
        last prompt (``tip`` is a token-prefix of ``prompt_ids``).  This routes
        completions to the right chain even when parallel agents / sub-agents
        interleave — each conversation has a distinct prompt prefix — and
        tolerates the just-finished turn being re-serialized in history (tool-call
        argument reformatting, whitespace), since that divergence falls *after*
        the prompt.  The compared prefix is two server-side tokenizations of the
        same text, so BPE re-tokenization of the sampled response never enters
        the decision.  On overlap the longest matching tip wins (most advanced
        chain).
        """
        best_idx: int | None = None
        best_len = -1
        for idx, tip in enumerate(chain_tips):
            # 剥掉 tip 末尾的生成头再比前缀:空截断轮被 CLI 丢弃后,下一轮同位置
            # 是 user 消息,带生成头的完整 tip 永远匹配失败 → 截断轮被误判成新链
            # (孤儿链 + prompt 重复训练)。剥头后「前缀全等、仅末尾新增消息」的
            # 请求被正确认回主链;子 agent/历史重写在前几个 token 就分叉,不受影响。
            core = _strip_generation_prompt(tip)
            n = len(core)
            if n > best_len and 0 < n <= len(prompt_ids) and prompt_ids[:n] == core:
                best_idx, best_len = idx, n
        return best_idx


def _top_level_scheduler_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    return {key: metadata[key] for key in keys if key in metadata}
