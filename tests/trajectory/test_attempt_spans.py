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
import re
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

    def test_benign_wrappers_match(self):
        """实测占漏检大头的良性写法。每条都无法改变 pipeline 打印的内容。"""
        for cmd in (
            "cd /opt/workspace/agent_workdir && " + _PIPELINE_CMD,          # 56x
            "cd /opt/workspace/agent_workdir && " + _PIPELINE_CMD + " 2>&1",
            "timeout 600000 " + _PIPELINE_CMD,                              # 10x
            "timeout -k 5s 1800s " + _PIPELINE_CMD,
            "export ASCEND_HOME=$ASCEND_HOME_PATH && " + _PIPELINE_CMD,     # 9x
            _PIPELINE_CMD + " 2>&1 | tail -50",                             # 42x
            _PIPELINE_CMD + " 2>&1 | head -150",
            _PIPELINE_CMD + " 2>&1 | grep ascendc | tail -20",
        ):
            assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) == "call_x", cmd

    def test_impl_arg_is_optional(self):
        """19x:不给 --impl 只是让 pipeline 走默认实现,不构成伪造面。"""
        cmd = re.sub(r"\s--impl\s+\S+", "", _PIPELINE_CMD)
        assert "--impl" not in cmd
        assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) == "call_x"

    def test_rewriting_pipes_rejected(self):
        """tail/head/grep/cat 只能挑行,sed/awk/tee 能把 success=false 改写成 true。"""
        for filt in ("sed s/false/true/", "awk '{print}'", "tee /tmp/f", "tr a b", "python3 -c pass"):
            cmd = _PIPELINE_CMD + " 2>&1 | " + filt
            assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) is None, cmd

    def test_env_prefix_is_an_allow_list(self):
        """白名单而非黑名单:下一个能颠覆运行的变量名今天还不知道。"""
        for var in ("PATH=/tmp/fake:$PATH", "PYTHONPATH=/tmp", "IFS=x", "SHELLOPTS=xtrace"):
            assert attempt_spans.pipeline_tool_call_id(self._msg(var + " " + _PIPELINE_CMD)) is None, var

    def test_cd_cannot_escape_and_script_must_live_under_tools(self):
        """放开 cd 的同时把脚本路径收紧到 tools/ —— 否则相对路径会跟着 cd 漂走,
        而旧正则的 `(?:\S*/)?` 本来就允许 agent 指向自己写的脚本。"""
        assert attempt_spans.pipeline_tool_call_id(
            self._msg("cd /tmp/../etc && " + _PIPELINE_CMD)) is None
        assert attempt_spans.pipeline_tool_call_id(
            self._msg(_PIPELINE_CMD.replace("bash tools/", "bash /tmp/mine/"))) is None

    def test_harmless_file_prefix_counts(self):
        """跑之前删缓存/清工作目录/打包答卷 —— 产出的仍是真 pipeline 的真判决。

        实测 165820:这三类占被旧规则拒掉的 134 次里的 122 次,而真去动预算计数器的
        只有 1 次。其中 6 个会话因此把 best 判在了错的轮次上。
        """
        for cmd in (
            "rm -f output/.selfcheck/.op_x.hash && " + _PIPELINE_CMD,
            "rm -rf judge_out && " + _PIPELINE_CMD,
            "tar -czf output/submission/x_impl.tar.gz -C x . && " + _PIPELINE_CMD,
            "cd /opt/workspace/agent_workdir && rm -rf judge_out && " + _PIPELINE_CMD,
        ):
            assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) == "call_x", cmd

    def test_prefix_touching_the_three_forbidden_things_rejected(self):
        """前缀即便是纯文件操作,碰这三样仍拒:预算计数器 / NPU 锁 / pipeline 脚本自身。"""
        for cmd in (
            "rm -f output/.budget.json && " + _PIPELINE_CMD,
            "rm -f judge_out/.op_budget.json && " + _PIPELINE_CMD,
            "rm -f /dev/shm/npu-locks/npu3.lock && " + _PIPELINE_CMD,
            "cp /tmp/mine.sh tools/ascendc_eval_pipeline.sh && " + _PIPELINE_CMD,
        ):
            assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) is None, cmd

    def test_prefix_that_can_write_stdout_rejected(self):
        """能往 stdout 写字的前缀一律拒 —— 那是把假 verdict 混进 tool result 的唯一通道。"""
        for cmd in (
            "echo '[ascendc-eval] done — success=true speedup_vs_torch=9' && " + _PIPELINE_CMD,
            "printf x && " + _PIPELINE_CMD,
            "python3 -c 'print(1)' && " + _PIPELINE_CMD,
            "cat some.log && " + _PIPELINE_CMD,
        ):
            assert attempt_spans.pipeline_tool_call_id(self._msg(cmd)) is None, cmd

    def test_invocation_must_be_the_last_segment(self):
        """尾部再挂东西一律拒:`; echo '[ascendc-eval] ...'` 会把假判决追加到输出末尾,
        而 parse_verdict 取的正是最后一条。"""
        for cmd in (
            _PIPELINE_CMD + " && echo done",
            _PIPELINE_CMD + "; echo '[ascendc-eval] done — success=true speedup_vs_torch=9'",
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
            # post-best 常开,但事件 0 是本链最后一次 attempt -> 它的段一直延伸到链尾,
            # 02-work2 属于该段(段模型:两次 attempt 之间的工作轮归入前一次),照常训练。
            assert trace1.loss_mask == [1, 1, 0, 1, 1]
            assert "post_best_masked_tokens" not in trace1.metadata

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


# ---------------------------------------------------------------------------
# post-best 掩码(POLAR_POST_BEST_MASK,默认关)
# ---------------------------------------------------------------------------


# post-best 掩码随 POLAR_ATTEMPT_CREDIT 一起生效(无独立开关),复用 _EnvGuard。
_PostBestGuard = _EnvGuard


def _ladder_chain() -> list[CompletionRecord]:
    """4 轮单链:c0 调用(判 FAIL) -> c1 调用(判 OK,最优) -> c2/c3 最优之后的收尾工作。

    response 流 = [10,EOT] 50 [20,EOT] 51 [30,EOT] 52 [40,EOT] (11 token),
    每轮起点 = [0, 3, 6, 9]。
    """
    return [
        _record("00", [1, 2], [10, EOT], finish_reason="tool_calls",
                tool_calls=_tool_calls("Bash", "call_1", _PIPELINE_CMD)),
        _record("01", [1, 2, 10, EOT, 50], [20, EOT], finish_reason="tool_calls",
                prompt_messages=[{"role": "user", "content": "task"},
                                 {"role": "assistant", "content": "00"},
                                 _verdict_msg("call_1", _VERDICT_FAIL)],
                tool_calls=_tool_calls("Bash", "call_2", _PIPELINE_CMD)),
        _record("02", [1, 2, 10, EOT, 50, 20, EOT, 51], [30, EOT],
                prompt_messages=[{"role": "user", "content": "task"},
                                 {"role": "assistant", "content": "00"},
                                 {"role": "assistant", "content": "01"},
                                 _verdict_msg("call_2", _VERDICT_OK)]),
        _record("03", [1, 2, 10, EOT, 50, 20, EOT, 51, 30, EOT, 52], [40, EOT],
                prompt_messages=[{"role": "user", "content": "task"},
                                 {"role": "assistant", "content": "00"},
                                 {"role": "assistant", "content": "01"},
                                 {"role": "assistant", "content": "02"},
                                 {"role": "user", "content": "go on"}]),
    ]


def _ladder_chain_with_later_attempt() -> list[CompletionRecord]:
    """5 轮单链:c0 调用(FAIL) -> c1 调用(OK,最优) -> c2 收尾工作 -> c3 再调用(FAIL) -> c4 收尾。

    c2 落在最优段内(事件 1 的段一直延伸到事件 2 的发起轮),c3/c4 才是真正的最优之后。
    response 流 = [10,EOT] 50 [20,EOT] 51 [30,EOT] 52 [40,EOT] 53 [50,EOT](14 token),
    每轮起点 = [0, 3, 6, 9, 12]。
    """
    return [
        _record("00", [1, 2], [10, EOT], finish_reason="tool_calls",
                tool_calls=_tool_calls("Bash", "call_1", _PIPELINE_CMD)),
        _record("01", [1, 2, 10, EOT, 50], [20, EOT], finish_reason="tool_calls",
                prompt_messages=[{"role": "user", "content": "task"},
                                 {"role": "assistant", "content": "00"},
                                 _verdict_msg("call_1", _VERDICT_FAIL)],
                tool_calls=_tool_calls("Bash", "call_2", _PIPELINE_CMD)),
        _record("02", [1, 2, 10, EOT, 50, 20, EOT, 51], [30, EOT],
                prompt_messages=[{"role": "user", "content": "task"},
                                 {"role": "assistant", "content": "00"},
                                 {"role": "assistant", "content": "01"},
                                 _verdict_msg("call_2", _VERDICT_OK)]),
        _record("03", [1, 2, 10, EOT, 50, 20, EOT, 51, 30, EOT, 52], [40, EOT],
                finish_reason="tool_calls",
                prompt_messages=[{"role": "user", "content": "task"},
                                 {"role": "assistant", "content": "00"},
                                 {"role": "assistant", "content": "01"},
                                 {"role": "assistant", "content": "02"}],
                tool_calls=_tool_calls("Bash", "call_3", _PIPELINE_CMD)),
        _record("04", [1, 2, 10, EOT, 50, 20, EOT, 51, 30, EOT, 52, 40, EOT, 53], [50, EOT],
                prompt_messages=[{"role": "user", "content": "task"},
                                 {"role": "assistant", "content": "00"},
                                 {"role": "assistant", "content": "01"},
                                 {"role": "assistant", "content": "02"},
                                 {"role": "assistant", "content": "03"},
                                 _verdict_msg("call_3", _VERDICT_FAIL)]),
    ]


class TestParseVerdictPicksLast:
    def test_last_verdict_wins(self):
        """一次运行可能先打 cached/分段 verdict 再打最终 done,末条才是结论。
        实测 165820 有 59 个 tool result 首末相差整整一个量程(0.200 vs 0.750)。"""
        text = (_VERDICT_FAIL + "\n...compiling...\n" + _VERDICT_OK)
        assert attempt_spans.verdict_score(attempt_spans.parse_verdict(text)) == \
               attempt_spans.verdict_score(attempt_spans.parse_verdict(_VERDICT_OK))

    def test_single_verdict_unchanged(self):
        assert attempt_spans.verdict_score(attempt_spans.parse_verdict(_VERDICT_FAIL)) == \
               attempt_spans.verdict_score(attempt_spans.parse_verdict(_VERDICT_FAIL))

    def test_no_verdict_still_none(self):
        assert attempt_spans.parse_verdict("nothing here") is None


class TestBestOrdinal:
    def test_first_attempt_reaching_max_wins(self):
        # ordinal 1 得 0.9 > ordinal 0 的 0.35
        assert attempt_spans.best_ordinal(
            {"a": (0, "c1"), "b": (1, "c2")},
            {"c1": _VERDICT_FAIL, "c2": _VERDICT_OK},
        ) == 1

    def test_tie_keeps_the_earlier_attempt(self):
        # 打平取「首次达到最高分」= 改进停止的时刻,这才是掩码要的边界。
        # 注意它与 .best 的最后写入者不同:pack_submission.sh 在 tier<3 打平时
        # 每次都覆写,所以那个文件是最后一个打平者写的。别按 .best 语义改这里。
        assert attempt_spans.best_ordinal(
            {"a": (0, "c1"), "b": (1, "c2")},
            {"c1": _VERDICT_OK, "c2": _VERDICT_OK},
        ) == 0

    def test_tie_spanning_a_dip_still_keeps_the_first(self):
        """实测最常见的形态:打平之间夹着更低分,首末相隔中位 3 个序号。

        092443 有 72/138 条 session 最高分打平(且全在失败档),此处锁住边界仍取首个,
        避免有人为了对齐 .best 把它改成末个而悄悄放宽掩码。
        """
        assert attempt_spans.best_ordinal(
            {"a": (0, "c1"), "b": (1, "c2"), "c": (2, "c3"), "d": (3, "c4")},
            {"c1": _VERDICT_OK, "c2": _VERDICT_FAIL, "c3": _VERDICT_FAIL, "c4": _VERDICT_OK},
        ) == 0

    def test_unscored_attempt_not_eligible(self):
        # call_2 的 verdict 从未到达 -> 无从判断它是否改进过,不参与取最大
        assert attempt_spans.best_ordinal(
            {"a": (0, "c1"), "b": (1, "c2")}, {"c1": _VERDICT_FAIL},
        ) == 0

    def test_no_scored_attempt_returns_none(self):
        assert attempt_spans.best_ordinal({"a": (0, "c1")}, {}) is None
        assert attempt_spans.best_ordinal({}, {"c1": _VERDICT_OK}) is None


class TestPostBestMasking:
    def test_attempt_credit_off_disables_masking(self):
        """POLAR_ATTEMPT_CREDIT=0 时没有 span state,post-best 一并停用。"""
        with _PostBestGuard("0"):
            traj = _build(_ladder_chain())
        trace = traj.traces[0]
        assert trace.loss_mask == [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
        assert "post_best_masked_tokens" not in trace.metadata

    def test_masking_starts_at_the_next_attempt_not_at_best(self):
        """边界 = 下一次 attempt 的发起轮。最优段(c1 的发起轮 + c2 的工作轮)完整存活。

        这正是 attempt_credit 给正分的区间;按 c1 自己那一轮切会把它劈成两半 —— 实测
        165820 上 274 万被掩 token 里有 109 万落在最优段内,其中 51 万附加项为正。
        """
        with _PostBestGuard("1"):
            traj = _build(_ladder_chain_with_later_attempt())
        trace = traj.traces[0]
        assert trace.metadata["attempt_spans"] == [[0, 3, 0, 0.35], [3, 9, 1, 0.9], [9, 14, 2, 0.35]]
        #            c0 保留 | c1 保留 | c2 保留(仍在最优段内)| c3/c4 清零
        assert trace.loss_mask == [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        assert trace.metadata["post_best_masked_tokens"] == 4

    def test_best_segment_runs_to_chain_end_when_no_later_attempt(self):
        """最优是本链最后一次 attempt -> 它的段吃到链尾,没有 post-best 区,不掩任何东西。"""
        with _PostBestGuard("1"):
            traj = _build(_ladder_chain())
        trace = traj.traces[0]
        assert trace.metadata["attempt_spans"] == [[0, 3, 0, 0.35], [3, 11, 1, 0.9]]
        assert trace.loss_mask == [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
        assert "post_best_masked_tokens" not in trace.metadata

    def test_lengths_never_change(self):
        # 用真会掩码的夹具,否则 on/off 无差别,这条不变量等于没测
        with _PostBestGuard("0"):
            off = _build(_ladder_chain_with_later_attempt()).traces[0]
        with _PostBestGuard("1"):
            on = _build(_ladder_chain_with_later_attempt()).traces[0]
        assert on.metadata["post_best_masked_tokens"] == 4
        assert on.response_ids == off.response_ids
        assert len(on.loss_mask) == len(off.loss_mask)
        assert len(on.response_logprobs or []) == len(off.response_logprobs or [])

    def test_no_scored_attempt_masks_nothing(self):
        records = _ladder_chain()
        # 抽掉两条 verdict -> 没有任何带判决的 attempt
        for rec in records[1:]:
            rec.request["messages"] = [
                m for m in rec.request["messages"] if m.get("role") != "tool"
            ]
        with _PostBestGuard("1"):
            traj = _build(records)
        assert traj.traces[0].loss_mask == [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

    def test_preexisting_empty_trace_is_still_emitted(self):
        """回归:开关打开时,**本来就**没有可训练 token 的 trace 必须照常发出。

        丢弃只针对「被 post-best 清空的」trace。早先的实现用 `any(loss_mask)`
        做判据,会把基线保留的退化 trace 一起丢掉 —— 一个与本特性无关的行为变化
        (真实数据回放里实测到 3 条)。
        """
        records = [
            _record("00", [1, 2], [], finish_reason="stop"),
        ]
        with _PostBestGuard("0"):
            base = _build(records)
        with _PostBestGuard("1"):
            on = _build(records)
        assert not any(base.traces[0].loss_mask)          # 基线:全零但保留
        assert len(on.traces) == len(base.traces)         # 开启后不得少
        assert on.metadata["reconstruction_stats"].get("traces_dropped_post_best", 0) == 0

    @staticmethod
    def _split_after_best(
        second_prompt: list[int], *, later_attempt: bool = False
    ) -> list[CompletionRecord]:
        """最优在链 0;链 1 由 ``second_prompt`` 起头(是否算续接由它决定)。

        ``later_attempt`` = 链 1 自己再发起一次(更差的)attempt,于是链 1 从第一轮起
        就属于最优之后的段;否则链 1 整条还在最优段内。
        """
        records = [
            _record("00", [1, 2, 3, 4], [10, EOT], finish_reason="tool_calls",
                    tool_calls=_tool_calls("Bash", "call_1", _PIPELINE_CMD)),
            _record("01", second_prompt, [20, EOT],
                    finish_reason="tool_calls" if later_attempt else "stop",
                    prompt_messages=[{"role": "user", "content": "task"},
                                     _verdict_msg("call_1", _VERDICT_OK)],
                    tool_calls=(_tool_calls("Bash", "call_2", _PIPELINE_CMD)
                                if later_attempt else None)),
        ]
        if later_attempt:
            records.append(
                _record("02", list(second_prompt) + [20, EOT, 60], [30, EOT],
                        prompt_messages=[{"role": "user", "content": "task"},
                                         {"role": "assistant", "content": "01"},
                                         _verdict_msg("call_2", _VERDICT_FAIL)]),
            )
        return records

    def test_continuation_chain_that_opens_a_later_attempt_is_masked_and_dropped(self):
        """链 1 续接链 0(共享过半前缀)且自己就发起了更晚的 attempt -> 整条清零并丢弃。"""
        records = self._split_after_best([1, 2, 9, 9], later_attempt=True)  # 共享 [1,2] = tip 的一半
        with _PostBestGuard("0"):
            assert len(_build(records).traces) == 2
        with _PostBestGuard("1"):
            traj = _build(records)
        assert len(traj.traces) == 1                      # 后一条被丢弃
        assert traj.traces[0].loss_mask == [1, 1]         # 最优的发起轮保住
        assert traj.metadata["reconstruction_stats"]["traces_dropped_post_best"] == 1

    def test_continuation_chain_still_inside_best_segment_is_kept(self):
        """同样是续接链,但链 1 没有再发起 attempt -> 它整条仍在最优段内,不该被掩。

        断链发生在最优段中间,补出来的首段带的正是最优的 ordinal;掩掉它等于删掉
        attempt_credit 正在给正分的那段。
        """
        records = self._split_after_best([1, 2, 9, 9])
        with _PostBestGuard("1"):
            traj = _build(records)
        assert len(traj.traces) == 2
        assert all(any(t.loss_mask) for t in traj.traces)
        assert "traces_dropped_post_best" not in traj.metadata["reconstruction_stats"]

    def test_independent_subconversation_after_best_is_untouched(self):
        """链 1 几乎不共享前缀(Skill/Agent 子对话)-> 只是时间上更晚,不该被掩码。

        实测:独立子对话占拆链 31%,且 17% 的多 chain session 会交错 —— 只按会话
        时间序判定会把与最优无因果关系的工作一起清零。
        """
        records = self._split_after_best([9, 9, 9])
        with _PostBestGuard("1"):
            traj = _build(records)
        assert len(traj.traces) == 2
        assert all(any(t.loss_mask) for t in traj.traces)
        assert "traces_dropped_post_best" not in traj.metadata["reconstruction_stats"]


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
