"""Export ``SessionTiming`` as Chrome Trace Event JSON for Perfetto.

Each session produces one self-contained ``trace.json`` that can be dragged
into https://ui.perfetto.dev for waterfall / flame-graph analysis.

Process layout (``pid``):
  1 — Gateway (init / run / postrun stages)
  2 — SGLang (inference)
  3 — Agent (tool execution + thinking)

All events share ``tid = session_id`` so they appear on the same horizontal
track in Perfetto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polar.rollout.agent_actions import classify_shell_command, is_agent_runner_command
from polar.rollout.models import SessionTiming

# Perfetto process IDs — stable across sessions
GW_PID = 1      # Gateway
SGL_PID = 2     # SGLang inference server
AGT_PID = 3     # Agent (inside runtime container)


def export_chrome_trace(
    timing: SessionTiming,
    *,
    session_id: str,
    node_id: str = "",
    task_id: str = "",
) -> list[dict[str, Any]]:
    """Build a Chrome Trace Event array from a session timing snapshot.

    Returns a list of trace events suitable for ``json.dump``.
    """
    events: list[dict[str, Any]] = []
    t = _Timeline()

    # ── Gateway coarse stages ──────────────────────────────────────────
    _add_event(events, t, timing.register_to_init_queue_ms,
               "register_to_init_queue", "gateway", GW_PID, session_id)

    # ── INIT ──
    init_anchor = t.cursor
    # runtime_create with docker sub-spans
    rc_anchor = t.cursor
    if timing.init_docker_create_ms > 0 or timing.init_docker_start_ms > 0:
        _add_event(events, t, timing.init_docker_create_ms,
                   "init/runtime_create/docker_create", "gateway,init,docker", GW_PID, session_id)
        _add_event(events, t, timing.init_docker_start_ms,
                   "init/runtime_create/docker_start", "gateway,init,docker", GW_PID, session_id)
        rc_dur = t.cursor - rc_anchor
        _add_group(events, rc_anchor, rc_dur,
                   "init/runtime_create", "gateway,init", GW_PID, session_id)
    else:
        _add_event(events, t, timing.init_runtime_create_ms,
                   "init/runtime_create", "gateway,init", GW_PID, session_id)
    _add_event(events, t, timing.init_prepare_ms,
               "init/prepare", "gateway,init", GW_PID, session_id)
    _add_group(events, init_anchor, t.cursor - init_anchor,
               "init", "gateway,init", GW_PID, session_id)

    # ── READY WAIT ──
    _add_event(events, t, timing.ready_wait_ms,
               "ready_wait", "gateway", GW_PID, session_id)

    # ── RUN ──
    run_anchor = t.cursor
    _add_event(events, t, timing.run_harness_setup_ms,
               "run/harness_setup", "gateway,run", GW_PID, session_id)

    agent_exec_anchor = t.cursor
    _emit_llm_call_events(events, t, timing.llm_calls, timing.tool_execs, session_id)
    # If no LLM calls filled the agent_exec span, use the reported value
    agent_exec_filled = t.cursor - agent_exec_anchor
    agent_exec_reported = timing.run_agent_exec_ms
    if agent_exec_reported > agent_exec_filled:
        t.cursor = agent_exec_anchor + agent_exec_reported
    else:
        # Extend cursor to cover the computed span
        t.cursor = agent_exec_anchor + max(agent_exec_filled, agent_exec_reported)

    _add_event(events, t, timing.run_harness_postprocess_ms,
               "run/harness_postprocess", "gateway,run", GW_PID, session_id)
    _add_group(events, run_anchor, t.cursor - run_anchor,
               "run", "gateway,run", GW_PID, session_id)

    # ── POSTRUN ──
    postrun_anchor = t.cursor
    _add_event(events, t, timing.postrun_build_ms,
               "postrun/build", "gateway,postrun", GW_PID, session_id)
    _add_event(events, t, timing.postrun_eval_ms,
               "postrun/eval", "gateway,postrun", GW_PID, session_id)
    # teardown with docker sub-spans
    td_anchor = t.cursor
    if timing.postrun_docker_kill_ms > 0 or timing.postrun_docker_rm_ms > 0:
        _add_event(events, t, timing.postrun_docker_kill_ms,
                   "postrun/teardown/docker_kill", "gateway,postrun,docker", GW_PID, session_id)
        _add_event(events, t, timing.postrun_docker_rm_ms,
                   "postrun/teardown/docker_rm", "gateway,postrun,docker", GW_PID, session_id)
        _add_group(events, td_anchor, t.cursor - td_anchor,
                   "postrun/teardown", "gateway,postrun", GW_PID, session_id)
    else:
        _add_event(events, t, timing.postrun_teardown_ms,
                   "postrun/teardown", "gateway,postrun", GW_PID, session_id)
    _add_event(events, t, timing.postrun_push_result_ms,
               "postrun/push_result", "gateway,postrun", GW_PID, session_id)
    _add_group(events, postrun_anchor, t.cursor - postrun_anchor,
               "postrun", "gateway,postrun", GW_PID, session_id)

    # ── Perfetto metadata ──
    events.append({
        "name": "process_name",
        "ph": "M",
        "pid": GW_PID,
        "args": {"name": "Gateway"},
    })
    events.append({
        "name": "process_name",
        "ph": "M",
        "pid": SGL_PID,
        "args": {"name": "SGLang"},
    })
    events.append({
        "name": "process_name",
        "ph": "M",
        "pid": AGT_PID,
        "args": {"name": "Agent"},
    })
    events.append({
        "name": "thread_name",
        "ph": "M",
        "pid": GW_PID,
        "tid": 0,  # default track
        "args": {"name": session_id[:16]},
    })
    if task_id:
        events.append({
            "name": "session_metadata",
            "ph": "M",
            "pid": GW_PID,
            "args": {"session_id": session_id, "task_id": task_id, "node_id": node_id},
        })

    return events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _Timeline:
    """Monotonic microsecond cursor for building sequential traces."""
    cursor: int = 0  # microseconds from session start


def _ms_to_us(ms: float) -> int:
    return max(0, int(ms * 1000.0))


def _add_event(
    events: list[dict], t: _Timeline,
    duration_ms: float, name: str, cat: str,
    pid: int, tid: str, *, args: dict[str, Any] | None = None,
) -> None:
    if duration_ms <= 0:
        return
    dur = _ms_to_us(duration_ms)
    event = {
        "name": name, "cat": cat, "ph": "X",
        "ts": t.cursor, "dur": dur,
        "pid": pid, "tid": tid,
    }
    if args:
        event["args"] = args
    events.append(event)
    t.cursor += dur


def _add_group(
    events: list[dict], start_us: int, dur_us: int,
    name: str, cat: str, pid: int, tid: str,
) -> None:
    """Emit a parent span that wraps child events (shown as a grouping row)."""
    if dur_us <= 0:
        return
    # Render with a distinct category so Perfetto can colour it differently.
    events.append({
        "name": name, "cat": cat, "ph": "X",
        "ts": start_us, "dur": dur_us,
        "pid": pid, "tid": tid,
        "args": {"_group": True},
    })


def _emit_llm_call_events(
    events: list[dict], t: _Timeline,
    llm_calls: list[dict], tool_execs: list[dict], session_id: str,
) -> None:
    """Place every LLM call and the classified actions before it."""
    if not llm_calls:
        # Still emit tool execs even without LLM calls (pure tool session)
        for te in tool_execs:
            _emit_one_tool_exec(events, t, te, session_id)
        return

    # ExecInput records outer harness setup + agent CLI commands, while
    # agent_actions records the inner Read/Bash/Edit calls.  Once detailed
    # actions exist, none of the outer steps should be rendered as tools.
    has_detailed_actions = any(call.get("agent_actions") for call in llm_calls)
    if has_detailed_actions:
        tool_execs = []
    else:
        tool_execs = [
            te
            for te in tool_execs
            if not is_agent_runner_command(str(te.get("command", "")))
        ]
    te_idx = 0

    for idx, call in enumerate(llm_calls):
        round_num = call.get("round", idx + 1)
        call_label = f"llm_call_{round_num}"

        # ── Agent side gap (before this call) ──
        agent_gap_ms = call.get("agent_side_gap_ms", 0.0)
        if agent_gap_ms > 0:
            gap_anchor = t.cursor
            actions = call.get("agent_actions") or []
            if actions:
                _emit_agent_actions(
                    events,
                    t,
                    actions,
                    call_label=call_label,
                    session_id=session_id,
                )
            else:
                # Backward compatibility for timings captured before agent action
                # extraction existed: fit any discrete ExecInput steps into the gap.
                remaining = agent_gap_ms
                while te_idx < len(tool_execs) and remaining > 0:
                    te = tool_execs[te_idx]
                    te_ms = te.get("duration_ms", 0.0)
                    if te_ms <= remaining + 1:  # tolerate 1ms rounding
                        _emit_one_tool_exec(events, t, te, session_id)
                        remaining -= te_ms
                        te_idx += 1
                    else:
                        break

            # Rounding, missing tool metadata, or old traces can leave part of
            # the measured gap unattributed.  Keep that remainder visible as an
            # explicitly estimated think span.
            total_placed_ms = (t.cursor - gap_anchor) / 1000.0
            if total_placed_ms < agent_gap_ms:
                leftover = agent_gap_ms - total_placed_ms
                if leftover > 0:
                    _add_event(events, t, leftover,
                               f"{call_label}/agent/think", "agent,think",
                               AGT_PID, session_id,
                               args={
                                   "duration_estimated": True,
                                   "gap_allocation": "unattributed_remainder",
                               })
        elif agent_gap_ms == 0 and idx == 0 and te_idx < len(tool_execs):
            # First call with gap=0: place any initial tool execs here
            gap_anchor = t.cursor
            while te_idx < len(tool_execs):
                te = tool_execs[te_idx]
                tool_before_first_llm = (
                    te.get("command", "").startswith("pip")
                    or te.get("command", "").startswith("apt")
                )
                if tool_before_first_llm:
                    _emit_one_tool_exec(events, t, te, session_id)
                    te_idx += 1
                else:
                    break

        gateway_start = t.cursor
        # ── Acquire wait ──
        acquire_ms = call.get("acquire_wait_ms", 0.0)
        _add_event(events, t, acquire_ms,
                   f"{call_label}/acquire_wait", "gateway,llm",
                   GW_PID, session_id)

        # ── Prepare ──
        prep_ms = call.get("prepare_ms", 0.0)
        _add_event(events, t, prep_ms,
                   f"{call_label}/prepare", "gateway,llm",
                   GW_PID, session_id)

        # ── SGLang inference ──
        sglang_ms = call.get("sglang_wait_ms", 0.0)
        _add_event(events, t, sglang_ms,
                   f"{call_label}/sglang", "sglang",
                   SGL_PID, session_id)

        # ── Normalize + post ──
        norm_ms = call.get("normalize_ms", 0.0)
        _add_event(events, t, norm_ms,
                   f"{call_label}/normalize", "gateway,llm",
                   GW_PID, session_id)
        post_ms = call.get("post_ms", 0.0)
        _add_event(events, t, post_ms,
                   f"{call_label}/post", "gateway,llm",
                   GW_PID, session_id)

        # ── Gateway total span (group) ──
        gw_dur = t.cursor - gateway_start
        if gw_dur > 0:
            _add_group(events, gateway_start, gw_dur,
                       call_label, "gateway,llm", GW_PID, session_id)

        # ── Per-call metadata ──
        events.append({
            "name": "llm_call_metadata",
            "ph": "M",
            "pid": GW_PID,
            "tid": session_id,
            "args": {
                "round": round_num,
                "prompt_tokens": call.get("prompt_tokens", 0),
                "response_tokens": call.get("response_tokens", 0),
            },
        })

    # ── Remaining tool execs after last LLM call ──
    while te_idx < len(tool_execs):
        _emit_one_tool_exec(events, t, tool_execs[te_idx], session_id)
        te_idx += 1


def _emit_agent_actions(
    events: list[dict],
    t: _Timeline,
    actions: list[dict],
    *,
    call_label: str,
    session_id: str,
) -> None:
    """Emit the classified tool/think actions assigned to one agent gap."""
    for index, action in enumerate(actions):
        duration_ms = float(action.get("duration_ms", 0.0) or 0.0)
        kind = str(action.get("kind") or "other")
        summary = str(action.get("summary") or action.get("tool_name") or kind)
        summary = _preview(summary, 100)
        if kind == "think":
            name = f"{call_label}/agent/think"
            category = "agent,think"
        else:
            name = f"{call_label}/tool/{kind}/{index:02d}: {summary}"
            category = f"agent,tool,{kind}"

        args = {
            "action_kind": kind,
            "tool_name": action.get("tool_name", ""),
            "tool_use_id": action.get("tool_use_id", ""),
            "is_error": bool(action.get("is_error", False)),
            "duration_ms": duration_ms,
            "duration_estimated": bool(action.get("duration_estimated", True)),
            "gap_allocation": action.get("gap_allocation", "unknown"),
            "parallel": bool(action.get("parallel", False)),
        }
        for key in ("command", "target", "description"):
            if action.get(key):
                args[key] = action[key]
        _add_event(
            events,
            t,
            duration_ms,
            name,
            category,
            AGT_PID,
            session_id,
            args=args,
        )


def _emit_one_tool_exec(
    events: list[dict], t: _Timeline,
    te: dict, session_id: str,
) -> None:
    """Emit a single tool execution as a Chrome Trace span."""
    dur_ms = te.get("duration_ms", 0.0)
    cmd = te.get("command", "")
    exit_code = te.get("exit_code", 0)
    idx = te.get("idx", 0)

    first_line = cmd.split("\n")[0].strip()
    tool_type = classify_shell_command(cmd)

    # Show the actual command in the span name so Perfetto waterfall is readable
    cmd_preview = _preview(first_line, 100)
    name = f"tool/{tool_type}/{idx:02d}: {cmd_preview}"

    _add_event(
        events,
        t,
        dur_ms,
        name,
        f"agent,tool,{tool_type}",
        AGT_PID,
        session_id,
        args={
            "action_kind": tool_type,
            "command": first_line[:256],
            "exit_code": exit_code,
            "duration_ms": dur_ms,
            "duration_estimated": False,
        },
    )
    events.append({
        "name": f"tool_metadata_{idx}",
        "ph": "M",
        "pid": AGT_PID,
        "tid": session_id,
        "args": {
            "tool_idx": idx,
            "command": first_line[:256],
            "exit_code": exit_code,
            "duration_ms": dur_ms,
        },
    })


def _preview(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
