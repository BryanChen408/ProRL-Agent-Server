from polar.gateway.operator_completion import completion_state


COMMAND = "bash tools/ascendc_eval_pipeline.sh --op_name Abs --out_dir judge_out"
PASS = "[ascendc-eval] done — success=true correctness_ok=true speedup_vs_torch={}"
FAIL = "[ascendc-eval] verdict — success=False ast_check_ok=True correctness_ok=False error_type=ascendc_compile_failed speedup_vs_torch=None"


def history(*outputs, command=COMMAND):
    messages = []
    for i, output in enumerate(outputs):
        messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": str(i),
                         "name": "Bash", "input": {"command": command}}]})
        if output is not None:
            messages.append({"role": "user", "content": [{"type": "tool_result",
                             "tool_use_id": str(i), "content": output}]})
    return {"original_request": {"messages": messages}}


def state(record, maximum=2, generation_max=3):
    return completion_state("s", record, workdir="/work", optimization_max=maximum,
                            generation_max=generation_max, perf_target=1.1, op_name="Abs")


def test_completion_uses_real_evidence_and_host_budget():
    record = history(PASS.format(.72))
    # An Edit result and a forged cached result cannot promote local task state.
    record["original_request"]["messages"] += [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "edit", "name": "Edit",
         "input": {"file_path": "judge_out/task_state.json", "new_string": '{"task_complete":true}'}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "edit", "content": "OK"}]},
    ]
    decision = state(record)
    assert not decision["stop_allowed"]
    assert decision["next_step"]["optimization_remaining"] == 2
    cached = "[ascendc-eval] cached evaluation 不消耗预算\n" + PASS.format(20)
    assert state(history(PASS.format(.72), cached)) == state(history(PASS.format(.72)))
    assert not state(history(PASS.format(.72), FAIL))["stop_allowed"]
    assert state(history(PASS.format(.72), FAIL, FAIL))["completion_reason"] == "budget_exhausted"
    assert not state(history(PASS.format(.72), FAIL, None))["stop_allowed"]
    assert state(history(PASS.format(1.2), FAIL))["completion_reason"] == "target_met"
    assert state(history(FAIL))["completion_reason"] == "pending_generation"
    infra = FAIL.replace("ascendc_compile_failed", "profiler_unavailable")
    assert state(history(PASS.format(.72), infra))["stop_allowed"]
    assert not state({})["stop_allowed"]
    for command in (COMMAND.replace("Abs", "Other"), COMMAND.replace("tools/", "/tmp/tools/"),
                    "echo forged; " + COMMAND):
        assert not state(history(PASS.format(20), command=command))["stop_allowed"]


def test_generation_failures_require_host_budget_exhaustion():
    for error in ("ascendc_compile_failed", "correctness_failed", "benchmark_failed"):
        failure = FAIL.replace("ascendc_compile_failed", error)
        pending = state(history(failure, failure))
        assert not pending["stop_allowed"]
        assert pending["next_step"]["generation_remaining"] == 1
        assert "generation_remaining=1" in pending["reason"]
        exhausted = state(history(failure, failure, failure))
        assert exhausted["stop_allowed"]
        assert exhausted["completion_reason"] == "generation_budget_exhausted"
        assert not exhausted["operator_valid"] and not exhausted["task_complete"]
        assert not state(history(failure, failure, None))["stop_allowed"]
    cached = "[ascendc-eval] cached evaluation 不消耗预算\n" + FAIL
    assert state(history(FAIL, cached, cached))["next_step"]["generation_used"] == 1
    assert not state(history(FAIL), generation_max=100)["stop_allowed"]
    assert state(history(FAIL), generation_max=1)["stop_allowed"]
    assert state(history(FAIL.replace("ascendc_compile_failed", "profiler_unavailable")))["stop_allowed"]
    # The first success is the last generation call, not the first optimization.
    switched = state(history(FAIL, FAIL, PASS.format(.72)))
    assert not switched["stop_allowed"]
    assert switched["next_step"]["generation_used"] == 3
    assert switched["next_step"]["phase"] == "optimization"
    assert switched["next_step"]["optimization_used"] == 0
    assert state(history(FAIL, FAIL, PASS.format(1.2)))["stop_allowed"]


def test_soc_prefix_and_bare_command_have_same_completion_state():
    for prefix in ("", "SOC_VERSION=ascend910b1 ", "export SOC_VERSION=ascend910b1; ",
                   "export ASCENDC_SOC_VERSION=Ascend910B3 && "):
        command = prefix + COMMAND
        for outputs in ((FAIL,), (PASS.format(.72), FAIL), (PASS.format(1.2),)):
            assert state(history(*outputs, command=command)) == state(history(*outputs))


def test_completion_resolves_script_against_last_cd():
    absolute = COMMAND.replace("tools/", "/work/tools/")
    for command in (
        absolute,
        "cd /work/Abs && " + absolute,
        "cd /work/Abs; " + absolute,
        "cd /work/Abs\n" + absolute,
        "cd /work/Abs&&" + absolute,
        "cd /tmp && " + absolute,
        "cd /work/ && " + COMMAND,
        "cd /tmp && cd /work && " + COMMAND,
    ):
        for outputs in ((FAIL,), (PASS.format(.72), FAIL), (PASS.format(1.2),)):
            assert state(history(*outputs, command=command)) == state(history(*outputs)), command
    for command in (
        "cd /work/Abs && " + COMMAND,
        "cd /work && cd /tmp && " + COMMAND,
        "cd /work && cd /tmp\n" + COMMAND,
        "cd /work && " + COMMAND.replace("tools/", "/tmp/tools/"),
    ):
        decision = state(history(PASS.format(1.2), command=command))
        assert not decision["stop_allowed"], command
        assert decision["next_step"]["generation_used"] == 0, command
