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

import logging
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
    verdict_by_call_id: dict[str, Any] = {}
    ordinal_by_completion_id: dict[str, tuple[int, str]] = {}
    prev_ordinal_by_completion_id: dict[str, int] = {}
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
        call_id = _pipeline_call_id(trace.response_messages)
        if call_id:
            ordinal_by_completion_id[completion.completion_id] = (next_ordinal, call_id)
            next_ordinal += 1
    return {
        "verdict_by_call_id": verdict_by_call_id,
        "ordinal_by_completion_id": ordinal_by_completion_id,
        "prev_ordinal_by_completion_id": prev_ordinal_by_completion_id,
        "total_events": next_ordinal,
    }


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

    ``SessionStore.record_upstream_failure`` counts them into session metadata; here they
    get the same treatment as a weight-update abort (dev_09): the whole session is
    non-trainable, status="ERROR", and oversampling backfills a clean one.
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

        for completion in filter_result.kept:
            prompt_ids = build_trace_from_completion(completion).prompt_ids
            chain_idx = self._find_extendable_chain(prompt_ids, chain_tips)
            if chain_idx is None:
                chain_idx = len(chains)
                chains.append([])
                chain_tips.append([])
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
            while start < len(chain):
                finalized = self._finalize_chain(
                    chain[start:],
                    chain_index=chain_index,
                    chain_length=len(chain),
                    segment_index=segment_index,
                    segment_start=start,
                    span_state=span_state,
                )
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
        _non_trainable = session_had_abort or session_spanned or bool(upstream_failures)
        if upstream_failures:
            _span_error: str | None = f"upstream engine failure: {upstream_failures}"
        elif session_had_abort:
            _span_error = "aborted generation (weight-update cutoff)"
        elif session_spanned:
            _span_error = f"policy_version span (mixed-weight): {sorted(session_versions)}"
        else:
            _span_error = None
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
                **_top_level_scheduler_metadata(session.metadata),
            },
            traces=final_traces,
        )

    # ------------------------------------------------------------------
    # Chain finalization
    # ------------------------------------------------------------------

    def _finalize_chain(
        self,
        chain: list[CompletionRecord],
        *,
        chain_index: int,
        chain_length: int,
        segment_index: int,
        segment_start: int,
        span_state: dict | None = None,
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

        _rs = len(stream_ids) - len(prompt_ids)
        self._append_response_tokens(first_trace, stream_ids, response_slots, loss_mask)
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
            # unless the harness rewrote prior messages.
            if (
                len(Ci_prompt_ids) < len(prev_prompt_ids)
                or Ci_prompt_ids[: len(prev_prompt_ids)] != prev_prompt_ids
            ):
                logger.debug(
                    "prefix_merging: canonical prefix break at step %d/%d",
                    i,
                    len(chain),
                )
                break_reason = "canonical_prefix_break"
                break

            # canonical_tail = canonical tokens for [prev assistant msg + new interstitials].
            canonical_tail = Ci_prompt_ids[len(prev_prompt_ids):]
            if eot_id is None:
                logger.debug(
                    "prefix_merging: eot unavailable at step %d/%d",
                    i,
                    len(chain),
                )
                break_reason = "eot_unavailable"
                break
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
            self._append_response_tokens(Ci_trace, stream_ids, response_slots, loss_mask)
            if _want_spans:
                _ev = span_state["ordinal_by_completion_id"].get(chain[i].completion_id)
                if _ev is not None:
                    event_records.append((_rs, _ev[0], _ev[1]))
            response_messages.extend(deepcopy(m) for m in Ci_trace.response_messages)
            msg_acc += len(Ci_trace.response_messages)

            prev_prompt_ids = Ci_prompt_ids
            prev_raw_response = list(Ci_trace.response_ids)
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
            )
            if _spans:
                _metadata["attempt_spans"] = _spans

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
    ) -> None:
        """Append a completion's response_ids and parallel logprob slots."""
        response_ids = list(trace.response_ids)
        stream_ids.extend(response_ids)
        trace_loss_mask = list(trace.loss_mask) or [1] * len(response_ids)
        if len(trace_loss_mask) != len(response_ids):
            raise ValueError("trace loss_mask length must match response_ids length")
        loss_mask.extend(trace_loss_mask)
        logprobs = trace.response_logprobs or []
        for pos in range(len(response_ids)):
            value = logprobs[pos] if pos < len(logprobs) else None
            response_slots.append(float(value) if isinstance(value, (int, float)) else None)

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
            n = len(tip)
            if n > best_len and 0 < n <= len(prompt_ids) and prompt_ids[:n] == tip:
                best_idx, best_len = idx, n
        return best_idx


def _top_level_scheduler_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    return {key: metadata[key] for key in keys if key in metadata}
