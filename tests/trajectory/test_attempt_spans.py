"""P3 attempt-span segment model tests (plan §6.2/§6.3)— 与 polar_zxp 原版对齐。

差异适配:pipeline 名 ascendc_eval_pipeline.sh、verdict 前缀 [ascendc-eval]、
ladder 用本仓六档(0.2-0.75 soft-saturating):correctness_failed → 0.35;
success + speedup=2.0 → 0.75+0.25*(4-1)/(4+1) = 0.9。
原版 trapped-recovery 跨链用例改写为通用前缀断链(本仓无 trapped_recovery)。

运行:PYTHONPATH=src pytest tests/trajectory/test_attempt_spans.py
"""

from __future__ import annotations

import asyncio
import json
import os

from polar.trajectory.builder import attempt_spans
from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession

EOT = 99

_PIPELINE_CMD = (
    "bash tools/ascendc_eval_pipeline.sh --op_name OP "
    "--impl output/submission/OP_impl.tar.gz --out_dir judge_out"
)
_VERDICT_OK = (
    "[ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch=2.0"
)  # 本仓 ladder: 0.9
_VERDICT_FAIL = (
    "[ascendc-eval] verdict — success=False ast_check_ok=True correctness_ok=False "
    "error_type=correctness_failed speedup_vs_torch=None"
)  # 本仓 ladder: 0.35


def _tool_calls(name: str, call_id: str, command: str | None = None) -> list[dict]:
    args = {"command": command} if command is not None else {}
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
    ]


def _record(
    completion_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    finish_reason: str = "stop",
    prompt_messages: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
) -> CompletionRecord:
    message: dict = {"role": "assistant", "content": completion_id}
    if tool_calls:
        message["tool_calls"] = tool_calls
    choice: dict = {
        "input_token_ids": list(prompt_ids),
        "message": message,
        "finish_reason": finish_reason,
        "logprobs": {
            "content": [
                {"token": f"t{t}", "token_id": t, "logprob": -0.01, "bytes": []}
                for t in response_ids
            ]
        },
    }
    return CompletionRecord(
        completion_id=completion_id,
        request={"system": "harness", "messages": prompt_messages or [{"role": "user", "content": completion_id}]},
        response={"choices": [choice]},
        metadata={},
    )


def _verdict_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _build(records: list[CompletionRecord]):
    return asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=EOT).build(
            CompletionSession(session_id="session-1", completions=records)
        )
    )


class _EnvGuard:
    """默认开;测试里显式控制。"""

    def __init__(self, value: str | None):
        self.value = value

    def __enter__(self):
        self.saved = os.environ.pop("POLAR_ATTEMPT_CREDIT", None)
        if self.value is not None:
            os.environ["POLAR_ATTEMPT_CREDIT"] = self.value
        return self

    def __exit__(self, *_):
        os.environ.pop("POLAR_ATTEMPT_CREDIT", None)
        if self.saved is not None:
            os.environ["POLAR_ATTEMPT_CREDIT"] = self.saved


# ---------------------------------------------------------------------------
# 白名单命令匹配
# ---------------------------------------------------------------------------


class TestPipelineCmdRegex:
    @staticmethod
    def _msg(command: str) -> dict:
        return {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_x", "type": "function",
                 "function": {"name": "Bash", "arguments": json.dumps({"command": command})}}
            ],
        }

    def test_bare_form_matches(self):
        assert attempt_spans.pipeline_tool_call_id(self._msg(_PIPELINE_CMD)) == "call_x"

    def test_absolute_path_matches(self):
        cmd = _PIPELINE_CMD.replace("bash tools/", "bash /opt/workspace/agent_workdir/tools/")
        assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) == "call_x"

    def test_inert_env_prefix_matches(self):
        cmd = "OPERATOR_NAME=op_x OPERATOR_ARCH=ascend910_9382 " + _PIPELINE_CMD
        assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) == "call_x"

    def test_verify_impl_repeat_prefix_rejected(self):
        # agent must not weaken its own determinism check to farm credit
        assert attempt_spans.pipeline_tool_call_id(self._msg("VERIFY_IMPL_REPEAT=1 " + _PIPELINE_CMD)) is None

    def test_injection_env_prefix_rejected(self):
        for var in ("LD_PRELOAD=/tmp/x.so", "BASH_ENV=/tmp/x.sh", "ENV=/tmp/x.sh"):
            assert attempt_spans.pipeline_tool_call_id(self._msg(var + " " + _PIPELINE_CMD)) is None, var

    def test_chained_commands_rejected(self):
        for cmd in (
            "rm judge_out/.budget.json && " + _PIPELINE_CMD,
            _PIPELINE_CMD + " && echo done",
            _PIPELINE_CMD + " | tee log",
            "echo hi; " + _PIPELINE_CMD,
        ):
            assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) is None, cmd

    def test_continuation_lines_match(self):
        cmd = _PIPELINE_CMD.replace(" --", " \\\n  --")
        assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) == "call_x"


# ---------------------------------------------------------------------------
# build_spans 几何
# ---------------------------------------------------------------------------


class TestBuildSpans:
    def test_event_spans_extend_to_next_event_and_chain_end(self):
        verdicts = {"c1": _VERDICT_FAIL, "c2": _VERDICT_OK}
        spans = attempt_spans.build_spans(
            [(10, 0, "c1"), (40, 1, "c2")], verdicts, leading_idx=-1, chain_resp_end=80,
        )
        assert spans == [
            [0, 10, -1, None],
            [10, 40, 0, 0.35],
            [40, 80, 1, 0.9],
        ]

    def test_missing_verdict_keeps_position_with_none_score(self):
        spans = attempt_spans.build_spans(
            [(10, 0, "c1"), (40, 1, "c2")], {}, leading_idx=-1, chain_resp_end=80,
        )
        assert spans == [[0, 10, -1, None], [10, 40, 0, None], [40, 80, 1, None]]

    def test_no_events_whole_trace_is_one_leading_span(self):
        assert attempt_spans.build_spans([], {}, leading_idx=2, chain_resp_end=50) == [[0, 50, 2, None]]
        assert attempt_spans.build_spans([], {}, leading_idx=-1, chain_resp_end=50) == [[0, 50, -1, None]]

    def test_no_events_in_trajectory_no_spans(self):
        assert attempt_spans.build_spans([], {}, leading_idx=None, chain_resp_end=50) == []

    def test_event_at_offset_zero_skips_leading_span(self):
        spans = attempt_spans.build_spans(
            [(0, 0, "c1")], {"c1": _VERDICT_OK}, leading_idx=-1, chain_resp_end=30,
        )
        assert spans == [[0, 30, 0, 0.9]]

    def test_degenerate_zero_length_event_dropped(self):
        spans = attempt_spans.build_spans(
            [(50, 0, "c1"), (50, 1, "c2")], {}, leading_idx=-1, chain_resp_end=50,
        )
        assert spans == [[0, 50, -1, None]]


# ---------------------------------------------------------------------------
# Builder 集成
# ---------------------------------------------------------------------------


class TestBuilderIntegration:
    def test_segment_zero_on_dispatch_trace_and_event_on_work_trace(self):
        with _EnvGuard("1"):
            records = [
                _record("00-dispatch", [1, 2], [5, EOT], finish_reason="tool_calls",
                        tool_calls=_tool_calls("Skill", "call_skill")),
                _record("01-work1", [7, 7], [10, EOT], finish_reason="tool_calls",
                        tool_calls=_tool_calls("Bash", "call_1", _PIPELINE_CMD)),
                _record("02-work2", [7, 7, 10, EOT, 50], [20, EOT],
                        prompt_messages=[{"role": "user", "content": "task"},
                                         {"role": "assistant", "content": "01-work1"},
                                         _verdict_msg("call_1", _VERDICT_OK)]),
            ]
            traj = _build(records)
            assert len(traj.traces) == 2
            trace0, trace1 = traj.traces
            assert trace0.metadata["attempt_spans"] == [[0, 2, -1, None]]
            assert trace1.metadata["attempt_spans"] == [[0, 5, 0, 0.9]]
            assert trace1.response_ids == [10, EOT, 50, 20, EOT]
            assert trace1.loss_mask == [1, 1, 0, 1, 1]

    def test_verdict_pairs_across_chain_break(self):
        """断链后 verdict 落在下一条链的 prompt 里:session 级 pre-pass 仍配上。"""
        with _EnvGuard("1"):
            records = [
                _record("00-caller", [1, 2], [10, EOT], finish_reason="tool_calls",
                        tool_calls=_tool_calls("Bash", "call_1", _PIPELINE_CMD)),
                # 前缀不延续 → 新链;其 prompt 携带 call_1 的 verdict
                _record("01-after", [9, 9, 9], [20, EOT],
                        prompt_messages=[{"role": "user", "content": "task"},
                                         _verdict_msg("call_1", _VERDICT_OK)]),
            ]
            traj = _build(records)
            assert len(traj.traces) == 2
            chain0_trace, chain1_trace = traj.traces
            assert chain0_trace.metadata["attempt_spans"] == [[0, 2, 0, 0.9]]
            # chain1 无自己的事件 → 延续事件 0 的段
            assert chain1_trace.metadata["attempt_spans"] == [[0, 2, 0, None]]

    def test_ordinals_are_server_side_and_verdict_independent(self):
        """序号跟随白名单调用的 session 时序;verdict 缺失的调用保住序号(score None)。"""
        with _EnvGuard("1"):
            records = [
                _record("00-w1", [1, 2], [10, EOT], finish_reason="tool_calls",
                        tool_calls=_tool_calls("Bash", "call_1", _PIPELINE_CMD)),
                # call_1 的 verdict 永远不到(session 在工具结果前被截断)
                _record("01-w2", [1, 2, 10, EOT, 50], [20, EOT], finish_reason="tool_calls",
                        prompt_messages=[{"role": "user", "content": "task"},
                                         {"role": "assistant", "content": "00-w1"}],
                        tool_calls=_tool_calls("Bash", "call_2", _PIPELINE_CMD)),
                _record("02-w3", [1, 2, 10, EOT, 50, 20, EOT, 51], [30, EOT],
                        prompt_messages=[{"role": "user", "content": "task"},
                                         {"role": "assistant", "content": "00-w1"},
                                         {"role": "assistant", "content": "01-w2"},
                                         _verdict_msg("call_2", _VERDICT_OK)]),
            ]
            traj = _build(records)
            assert len(traj.traces) == 1
            spans = traj.traces[0].metadata["attempt_spans"]
            # response = [10,EOT, 50, 20,EOT, 51, 30,EOT] -> 8 tokens
            # 事件 0 = call_1(发起轮起点 0,verdict 缺失);事件 1 = call_2(起点 3,0.9)
            assert spans == [[0, 3, 0, None], [3, 8, 1, 0.9]]

    def test_budget_reset_hack_call_gets_no_ordinal(self):
        """`rm ...budget && bash pipeline` 非白名单:不给序号不给 credit。"""
        with _EnvGuard("1"):
            hack_cmd = "rm judge_out/.eval_budget.json && " + _PIPELINE_CMD
            records = [
                _record("00-hack", [1, 2], [10, EOT], finish_reason="tool_calls",
                        tool_calls=_tool_calls("Bash", "call_1", hack_cmd)),
                _record("01-clean", [1, 2, 10, EOT, 50], [20, EOT], finish_reason="tool_calls",
                        prompt_messages=[{"role": "user", "content": "task"},
                                         {"role": "assistant", "content": "00-hack"},
                                         _verdict_msg("call_1", _VERDICT_OK)],
                        tool_calls=_tool_calls("Bash", "call_2", _PIPELINE_CMD)),
            ]
            traj = _build(records)
            spans = traj.traces[0].metadata["attempt_spans"]
            # response = [10,EOT, 50, 20,EOT] -> 5 tokens;只有干净调用得序号(其 verdict 未到)
            assert spans == [[0, 3, -1, None], [3, 5, 0, None]]

    def test_default_on_env_off_zero_metadata_change(self):
        # 默认开:不设 env 也产 spans
        with _EnvGuard(None):
            records = [
                _record("00-w1", [1, 2], [10, EOT], finish_reason="tool_calls",
                        tool_calls=_tool_calls("Bash", "call_1", _PIPELINE_CMD)),
            ]
            traj = _build(records)
            assert "attempt_spans" in traj.traces[0].metadata
        # 显式关:键整个缺席
        with _EnvGuard("0"):
            traj = _build(records)
            assert "attempt_spans" not in traj.traces[0].metadata


if __name__ == "__main__":
    import sys
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("Test") and callable(v)]
    failed = 0
    total = 0
    for name, cls in tests:
        for mname in dir(cls):
            if mname.startswith("test_"):
                total += 1
                try:
                    getattr(cls(), mname)()
                    print(f"  [OK] {name}.{mname}")
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    print(f"  [XX] {name}.{mname}: {type(e).__name__}: {e}")
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
