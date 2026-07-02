from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "ascend_operator" / "tools" / "polar_rollout_observer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("polar_rollout_observer", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyze_session_messages_extracts_pipeline_stage_summary() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "bash tools/triton_eval_pipeline.sh --op_name op"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": (
                                "[triton-eval] done — success=true ast_check_ok=true "
                                "correctness_ok=true speedup_vs_torch=2.5\n"
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "Bash",
                            "input": {"command": "bash tools/triton_eval_pipeline.sh --op_name op"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_2",
                            "content": (
                                "[triton-eval] Step2 verify\n"
                                "[triton-eval] verify FAILED\n"
                            ),
                        }
                    ],
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["pipeline_runs"] == 2
    assert summary["validation"]["success"] == 1
    assert summary["validation"]["verify_fail"] == 1
    assert summary["pipeline_stage_counts"]["precision"]["attempts"] == 2
    assert summary["pipeline_stage_counts"]["precision"]["pass"] == 1
    assert summary["pipeline_stage_counts"]["precision"]["fail"] == 1
    assert summary["pipeline_stage_counts"]["profiling"]["attempts"] == 1
    assert summary["pipeline_stage_counts"]["profiling"]["pass"] == 1


def test_analyze_session_messages_keeps_ast_fail_out_of_precision_and_profile() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_ast",
                            "name": "Bash",
                            "input": {"command": "bash tools/triton_eval_pipeline.sh --op_name op"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_ast",
                            "content": (
                                "[triton-eval] Step1 AST check\n"
                                "[triton-eval] AST FAILED\n"
                                "[triton-eval] verdict — success=False ast_check_ok=False "
                                "correctness_ok=False error_type=ast_check_failed "
                                "speedup_vs_torch=None\n"
                            ),
                        }
                    ],
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["pipeline_runs"] == 1
    assert summary["validation"]["ast_fail"] == 1
    detail = summary["pipeline_runs_detail"][0]
    assert detail["status"] == "ast_fail"
    assert detail["precision_status"] == "not_reached"
    assert detail["profiling_status"] == "not_reached"
    assert summary["pipeline_stage_counts"]["precision"]["attempts"] == 0
    assert summary["pipeline_stage_counts"]["profiling"]["attempts"] == 0


def test_analyze_session_messages_recognizes_cached_success_case_insensitively() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_cached",
                            "name": "Bash",
                            "input": {"command": "bash tools/triton_eval_pipeline.sh --op_name op"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_cached",
                            "content": (
                                "[triton-eval] cached verdict — success=True "
                                "ast_check_ok=True correctness_ok=True speedup_vs_torch=7.1364\n"
                            ),
                        }
                    ],
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["validation"]["success"] == 1
    detail = summary["pipeline_runs_detail"][0]
    assert detail["status"] == "success"
    assert detail["precision_status"] == "pass"
    assert detail["profiling_status"] == "pass"
    assert summary["pipeline_stage_counts"]["precision"]["attempts"] == 1
    assert summary["pipeline_stage_counts"]["profiling"]["attempts"] == 1


def test_analyze_session_messages_recognizes_cannbot_verify_and_benchmark() -> None:
    module = _load_module()
    verify_json = json.dumps(
        {"total_cases": 2, "passed_cases": 2, "failed_cases": 0},
        ensure_ascii=False,
    )
    perf_json = json.dumps(
        {"total_cases": 2, "passed_cases": 2, "failed_cases": 0, "speedup_vs_torch": 3.5},
        ensure_ascii=False,
    )
    payload = {
        "original_request": {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Implement the Ascend Triton operator `safe_op`.\n"
                        "The prepared reference task is at `input/safe_op.py`.\n"
                        "Follow `./CLAUDE.md` for the workflow."
                    ),
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "verify",
                            "name": "Bash",
                            "input": {
                                "command": (
                                    "python3 /opt/canonical/skills/triton-op-verifier/scripts/verify.py "
                                    "--op_name safe_op --verify_dir output/iter_0/verify"
                                )
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "verify",
                            "content": f"验证结果已保存到: output/iter_0/verify/verify_result.json\n{verify_json}",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "bench",
                            "name": "Bash",
                            "input": {
                                "command": (
                                    "python3 /opt/canonical/skills/triton-op-verifier/scripts/benchmark.py "
                                    "--op_name safe_op --verify_dir output/iter_0/verify"
                                )
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "bench",
                            "content": f"性能测试结果:\n结果已保存到: output/iter_0/perf_result.json\n{perf_json}",
                        }
                    ],
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["request"]["task_prompt_is_operator"] is True
    assert summary["pipeline_runs"] == 1
    assert summary["validation"]["success"] == 2
    assert summary["pipeline_stage_counts"]["precision"]["attempts"] == 1
    assert summary["pipeline_stage_counts"]["precision"]["pass"] == 1
    assert summary["pipeline_stage_counts"]["profiling"]["attempts"] == 1
    assert summary["pipeline_stage_counts"]["profiling"]["pass"] == 1
    verify_detail, benchmark_detail = summary["pipeline_runs_detail"]
    assert verify_detail["kind"] == "cannbot_verify"
    assert verify_detail["budget_counted"] is True
    assert verify_detail["status"] == "verify_success"
    assert benchmark_detail["kind"] == "cannbot_benchmark"
    assert benchmark_detail["budget_counted"] is False
    assert benchmark_detail["status"] == "success"


def test_analyze_session_messages_audits_forbidden_write_tools() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "write_tools",
                            "name": "Write",
                            "input": {"file_path": "tools/triton_eval_pipeline.sh", "content": "bad"},
                        },
                        {
                            "type": "tool_use",
                            "id": "edit_skill",
                            "name": "Edit",
                            "input": {"file_path": ".agents/skills/triton-op-verifier/scripts/verify.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "multi_claude",
                            "name": "MultiEdit",
                            "input": {"file_path": "CLAUDE.md", "edits": []},
                        },
                        {
                            "type": "tool_use",
                            "id": "write_submission",
                            "name": "Write",
                            "input": {"file_path": "output/submission/demo_impl.py", "content": "ok"},
                        },
                    ],
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["forbidden_writes"] == [
        {"turn": 1, "tool": "Write", "path": "tools/triton_eval_pipeline.sh"},
        {"turn": 1, "tool": "Edit", "path": ".agents/skills/triton-op-verifier/scripts/verify.py"},
        {"turn": 1, "tool": "MultiEdit", "path": "CLAUDE.md"},
    ]
    assert summary["readonly_mutation_attempts"] == []


def test_analyze_session_messages_audits_bash_readonly_mutations_without_pipeline_false_positive() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "pipeline",
                            "name": "Bash",
                            "input": {"command": "bash tools/triton_eval_pipeline.sh --op_name op"},
                        },
                        {
                            "type": "tool_use",
                            "id": "redirect",
                            "name": "Bash",
                            "input": {"command": "echo x >> tools/triton_eval_pipeline.sh"},
                        },
                        {
                            "type": "tool_use",
                            "id": "tee",
                            "name": "Bash",
                            "input": {"command": "printf x | tee .agents/skills/foo/SKILL.md"},
                        },
                        {
                            "type": "tool_use",
                            "id": "sed",
                            "name": "Bash",
                            "input": {"command": "sed -i 's/a/b/' CLAUDE.md"},
                        },
                        {
                            "type": "tool_use",
                            "id": "chmod",
                            "name": "Bash",
                            "input": {"command": "chmod +w tools/env.sh"},
                        },
                        {
                            "type": "tool_use",
                            "id": "safe",
                            "name": "Bash",
                            "input": {"command": "python3 output/submission/demo_impl.py"},
                        },
                    ],
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["pipeline_runs"] == 1
    assert summary["readonly_mutation_attempts"] == [
        {
            "turn": 1,
            "tool": "Bash",
            "kind": "redirect",
            "path": "tools/triton_eval_pipeline.sh",
            "command": "echo x >> tools/triton_eval_pipeline.sh",
        },
        {
            "turn": 1,
            "tool": "Bash",
            "kind": "tee",
            "path": ".agents/skills/foo/SKILL.md",
            "command": "printf x | tee .agents/skills/foo/SKILL.md",
        },
        {
            "turn": 1,
            "tool": "Bash",
            "kind": "sed_i",
            "path": "CLAUDE.md",
            "command": "sed -i 's/a/b/' CLAUDE.md",
        },
        {
            "turn": 1,
            "tool": "Bash",
            "kind": "chmod",
            "path": "tools/env.sh",
            "command": "chmod +w tools/env.sh",
        },
    ]


def test_analyze_session_messages_tracks_submission_workflow_and_doc_drift() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "这是一份非常完整的 Triton Ascend 参考文档，我先点评其结构。",
                        },
                        {
                            "type": "tool_use",
                            "id": "write_1",
                            "name": "Write",
                            "input": {"file_path": "output/submission/demo_impl.py", "content": "v1"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "write_2",
                            "name": "Edit",
                            "input": {"file_path": "/opt/workspace/agent_workdir/output/submission/demo_impl.py"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "pipeline",
                            "name": "Bash",
                            "input": {"command": "bash tools/triton_eval_pipeline.sh --op_name demo"},
                        },
                    ],
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["submission_writes"] == 2
    assert summary["submission_write_turns"] == [1, 2]
    assert summary["first_submission_write_turn"] == 1
    assert summary["first_pipeline_turn"] == 3
    assert summary["writes_before_first_pipeline"] == 2
    assert summary["workflow_violation"] is True
    assert summary["doc_drift_count"] == 1
    assert summary["doc_drift_turns"] == [1]


def test_analyze_session_messages_marks_raw_tool_call_text_abnormal() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": (
                        'I will call the tool now.\n'
                        '{"type": "tool_use", "id": "call_1", "name": "Write", '
                        '"input": {"file_path": "output/submission/op_impl.py"}}'
                    ),
                },
            ]
        },
        "response": {"choices": [{"message": {"content": "done", "tool_calls": []}, "finish_reason": "stop"}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["abnormal_termination"] is True
    assert summary["abnormal_reasons"] == ["raw_tool_call_text"]
    assert summary["abnormal_events"][0]["turn"] == 1
    assert "tool_use" in summary["abnormal_events"][0]["snippet"]
    assert (
        module.ObserverStore.classify_session_status(
            None,
            has_files=True,
            result={"status": "COMPLETED"},
            timeout_info=None,
            summary=summary,
        )
        == "ABNORMAL_ON_DISK"
    )


def test_analyze_session_messages_allows_structured_tool_use_blocks() -> None:
    module = _load_module()
    payload = {
        "original_request": {
            "messages": [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I will inspect the task."},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"file_path": "/opt/workspace/agent_workdir/src/op.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "1\timport torch\n",
                        }
                    ],
                },
            ]
        },
        "response": {
            "choices": [
                {
                    "message": {"content": "I read the task.", "tool_calls": []},
                    "finish_reason": "stop",
                }
            ]
        },
    }

    summary = module.analyze_session_messages(payload)

    assert summary["tool_counts"] == {"Read": 1}
    assert summary["abnormal_termination"] is False
    assert summary["abnormal_reasons"] == []


def test_analyze_session_messages_marks_empty_stop_response_abnormal() -> None:
    module = _load_module()
    payload = {
        "original_request": {"messages": [{"role": "user", "content": "task"}]},
        "response": {"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "stop"}]},
    }

    summary = module.analyze_session_messages(payload)

    assert summary["abnormal_termination"] is True
    assert summary["abnormal_reasons"] == ["empty_stop_response"]
    assert summary["abnormal_events"][0]["finish_reason"] == "stop"
    assert (
        module.ObserverStore.classify_session_status(
            "COMPLETED",
            has_files=False,
            result=None,
            timeout_info=None,
            summary=summary,
        )
        == "ABNORMAL"
    )


def test_observer_marks_completed_on_disk_session_abnormal_for_review(tmp_path: Path) -> None:
    module = _load_module()
    task = tmp_path / "rollout_results" / "task_smoke_20260629-polar-op-0-0"
    completions = task / "sessions" / "sk-abnormal" / "completions"
    completions.mkdir(parents=True)
    (completions / "0001.json").write_text(
        json.dumps(
            {
                "original_request": {"messages": [{"role": "user", "content": "task"}]},
                "response": {"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "stop"}]},
            }
        ),
        encoding="utf-8",
    )
    (task / "ses_sk-abnormal.json").write_text(
        json.dumps({"session_id": "sk-abnormal", "task_id": "smoke_20260629-polar-op-0-0", "status": "COMPLETED"}),
        encoding="utf-8",
    )

    store = module.ObserverStore(tmp_path, "http://127.0.0.1:1")
    store.gateway_health = lambda: {"status": "ok"}
    store.gateway_sessions = lambda: []
    store.device_map = lambda: {}

    summary = store.session_summary()
    session = summary["sessions"][0]

    assert session["status"] == "ABNORMAL_ON_DISK"
    assert session["result_status"] == "COMPLETED"
    assert session["quality"]["abnormal_termination"] is True
    assert session["quality"]["abnormal_reasons"] == ["empty_stop_response"]


def test_observer_groups_legacy_and_run_scoped_sessions(tmp_path: Path) -> None:
    module = _load_module()
    results = tmp_path / "rollout_results"
    legacy = results / "task_polar-op-0-0" / "sessions" / "sk-legacy" / "completions"
    current = results / "task_regular_20260623-120000-polar-op-0-0" / "sessions" / "sk-current" / "completions"
    nested = (
        results
        / "run_train-a"
        / "task_train-a-polar-op-1-0"
        / "sessions"
        / "sk-nested"
        / "completions"
    )
    legacy.mkdir(parents=True)
    current.mkdir(parents=True)
    nested.mkdir(parents=True)
    legacy_file = legacy / "0001.json"
    current_file = current / "0001.json"
    nested_file = nested / "0001.json"
    legacy_file.write_text("{}", encoding="utf-8")
    current_file.write_text("{}", encoding="utf-8")
    nested_file.write_text("{}", encoding="utf-8")
    os.utime(legacy_file, (1000, 1000))
    os.utime(current_file, (2000, 2000))
    os.utime(nested_file, (3000, 3000))

    store = module.ObserverStore(tmp_path, "http://127.0.0.1:1")
    store.gateway_health = lambda: {"status": "ok"}
    store.gateway_sessions = lambda: []
    store.device_map = lambda: {}

    summary = store.session_summary()
    by_session = {item["session_id"]: item for item in summary["sessions"]}
    by_group = {item["run_id"]: item for item in summary["session_groups"]}

    assert by_session["sk-legacy"]["run_id"] == "legacy"
    assert by_session["sk-current"]["run_id"] == "regular_20260623-120000"
    assert by_session["sk-nested"]["run_id"] == "train-a"
    assert summary["latest_run_id"] == "train-a"
    assert by_group["legacy"]["sessions"] == 1
    assert by_group["regular_20260623-120000"]["sessions"] == 1
    assert by_group["train-a"]["sessions"] == 1
    assert by_group["regular_20260623-120000"]["latest_mtime"] == 2000
    assert by_group["train-a"]["latest_mtime"] == 3000


def test_observer_can_use_explicit_results_dir(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "e2e"
    results = tmp_path / "external_rollout_results"
    completions = (
        results
        / "task_regular_20260627-010101-polar-op-0-0"
        / "sessions"
        / "sk-explicit"
        / "completions"
    )
    completions.mkdir(parents=True)
    completion = completions / "0001.json"
    completion.write_text("{}", encoding="utf-8")
    os.utime(completion, (2000, 2000))

    store = module.ObserverStore(root, "http://127.0.0.1:1", results)
    store.gateway_health = lambda: {"status": "ok"}
    store.gateway_sessions = lambda: []
    store.device_map = lambda: {}

    summary = store.session_summary()

    assert summary["root"] == str(root)
    assert summary["results_dir"] == str(results)
    assert summary["sessions"][0]["session_id"] == "sk-explicit"
    assert summary["sessions"][0]["run_id"] == "regular_20260627-010101"


def test_observer_aggregates_historical_quality_when_latest_completion_lacks_history(tmp_path: Path) -> None:
    module = _load_module()
    completions = (
        tmp_path
        / "rollout_results"
        / "task_smoke_20260624-010101-polar-op-0-0"
        / "sessions"
        / "sk-stable"
        / "completions"
    )
    completions.mkdir(parents=True)
    first = completions / "0001.json"
    second = completions / "0002.json"
    first.write_text(
        json.dumps(
            {
                "original_request": {
                    "messages": [
                        {"role": "user", "content": "task"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_pipeline",
                                    "name": "Bash",
                                    "input": {"command": "bash tools/triton_eval_pipeline.sh --op_name op"},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_pipeline",
                                    "content": (
                                        "[triton-eval] done success=true ast_check_ok=true "
                                        "correctness_ok=true speedup_vs_torch=2.0\n"
                                    ),
                                }
                            ],
                        },
                    ]
                },
                "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "original_request": {"messages": [{"role": "user", "content": "next turn only"}]},
                "response": {"choices": [{"message": {"content": "", "tool_calls": []}}]},
            }
        ),
        encoding="utf-8",
    )
    os.utime(first, (1000, 1000))
    os.utime(second, (2000, 2000))

    store = module.ObserverStore(tmp_path, "http://127.0.0.1:1")
    store.gateway_health = lambda: {"status": "ok"}
    store.gateway_sessions = lambda: []
    store.device_map = lambda: {}
    store.gateway_completions = lambda _session_id: []

    summary = store.session_summary()
    session = summary["sessions"][0]
    quality = session["quality"]

    assert session["latest_completion"] == "0002.json"
    assert quality["pipeline_runs"] == 1
    assert quality["pipeline_stage_counts"]["precision"]["attempts"] == 1
    assert quality["pipeline_stage_counts"]["precision"]["pass"] == 1
    assert quality["pipeline_stage_counts"]["profiling"]["attempts"] == 1
    assert quality["validation"]["success"] == 1

    detail = store.get_session("sk-stable")
    assert detail is not None
    assert detail["summary_source"] == "disk_aggregate"
    assert detail["summary"]["pipeline_runs"] == 1
    assert detail["summary"]["pipeline_runs_detail"][0]["status"] == "success"


def test_observer_lists_cannbot_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    session = (
        tmp_path
        / "rollout_results"
        / "task_run-a-polar-op-0-0"
        / "sessions"
        / "sk-cannbot"
    )
    (session / "completions").mkdir(parents=True)
    (session / "completions" / "0001.json").write_text(
        json.dumps({"original_request": {"messages": [{"role": "user", "content": "task"}]}}),
        encoding="utf-8",
    )
    artifacts = session / "artifacts"
    artifacts.mkdir()
    (artifacts / "verify_result.json").write_text(
        json.dumps({"total_cases": 2, "passed_cases": 2, "failed_cases": 0}),
        encoding="utf-8",
    )
    (artifacts / "perf_result.json").write_text(
        json.dumps({"total_cases": 2, "passed_cases": 2, "failed_cases": 0, "speedup_vs_torch": 4.25}),
        encoding="utf-8",
    )
    output = session / "output"
    output.mkdir()
    (output / "generated_code.py").write_text("class ModelNew: pass\n", encoding="utf-8")

    store = module.ObserverStore(tmp_path, "http://127.0.0.1:1")
    store.gateway_health = lambda: {"status": "ok"}
    store.gateway_sessions = lambda: []
    store.device_map = lambda: {}

    summary = store.session_summary()
    quality = summary["sessions"][0]["quality"]

    assert quality["cannbot_artifacts"] == {
        "verify_result": 1,
        "perf_result": 1,
        "generated_code": 1,
        "optimized_code": 0,
    }

    detail = store.get_session("sk-cannbot")
    assert detail is not None
    by_kind = {item["kind"]: item for item in detail["cannbot_artifacts"]}
    assert by_kind["verify_result"]["success"] is True
    assert by_kind["verify_result"]["passed_cases"] == 2
    assert by_kind["perf_result"]["speedup_vs_torch"] == 4.25
    assert by_kind["generated_code"]["path"] == "output/generated_code.py"
