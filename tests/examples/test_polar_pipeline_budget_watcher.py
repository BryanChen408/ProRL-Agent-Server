from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WATCHER = ROOT / "deploy" / "ascend_operator" / "tools" / "polar_pipeline_budget_watcher.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("polar_pipeline_budget_watcher", WATCHER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_status(
    root: Path,
    session_id: str,
    *,
    attempt: int,
    limit: int,
    phase: str = "generation",
    exhausted: bool,
) -> Path:
    path = root / "rollout_results" / "task_t" / "sessions" / session_id / "artifacts" / "pipeline_budget_status.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        (
            "{\n"
            '  "schema_version": 1,\n'
            f'  "session_id": "{session_id}",\n'
            f'  "task_id": "task_t",\n'
            f'  "phase": "{phase}",\n'
            f'  "attempt": {attempt},\n'
            f'  "limit": {limit},\n'
            f'  "limit_exhausted": {str(exhausted).lower()}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return path


def _write_host_status(
    session_base_dir: Path,
    session_id: str,
    *,
    attempt: int,
    limit: int,
    phase: str = "generation",
    exhausted: bool,
) -> Path:
    path = (
        session_base_dir
        / "session-sk-polar-shortid"
        / "artifacts"
        / "pipeline_budget_status.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        (
            "{\n"
            '  "schema_version": 1,\n'
            f'  "session_id": "{session_id}",\n'
            f'  "task_id": "task_t",\n'
            f'  "phase": "{phase}",\n'
            f'  "attempt": {attempt},\n'
            f'  "limit": {limit},\n'
            f'  "limit_exhausted": {str(exhausted).lower()}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return path


def test_status_at_limit_does_not_cancel(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_status(tmp_path, "s1", attempt=6, limit=6, exhausted=False)

    status = module._load_pipeline_status(tmp_path, "s1")
    cancel, reason = module.should_cancel_from_status(status)

    assert status["_status_path"] == str(path)
    assert not cancel
    assert "within budget" in reason


def test_status_over_limit_cancels(tmp_path: Path) -> None:
    module = _load_module()
    path = _write_status(tmp_path, "s1", attempt=7, limit=6, exhausted=True)

    status = module._load_pipeline_status(tmp_path, "s1")
    cancel, reason = module.should_cancel_from_status(status)

    assert status["_status_path"] == str(path)
    assert cancel
    assert "attempt=7>6" in reason


def test_status_loads_from_host_session_base_by_json_session_id(tmp_path: Path) -> None:
    module = _load_module()
    session_id = "sk-polar-long-session-id"
    session_base_dir = tmp_path / "polar_sessions"
    path = _write_host_status(
        session_base_dir,
        session_id,
        attempt=4,
        limit=3,
        exhausted=True,
    )

    status = module._load_pipeline_status(tmp_path, session_id, session_base_dir)
    cancel, reason = module.should_cancel_from_status(status)

    assert status["_status_path"] == str(path)
    assert cancel
    assert "attempt=4>3" in reason


def test_missing_status_does_not_cancel(tmp_path: Path) -> None:
    module = _load_module()

    cancel, reason = module.should_cancel_from_status(
        module._load_pipeline_status(tmp_path, "missing")
    )

    assert not cancel
    assert reason == "pipeline status missing"


def test_completion_parser_ignores_reading_pipeline_script() -> None:
    module = _load_module()
    record = {
        "original_request": {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {
                                "command": "cat tools/triton_eval_pipeline.sh | head -60"
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "script header",
                        }
                    ],
                },
            ]
        }
    }

    state = module.analyze_budget("s1", record)

    assert state.pipeline_calls == []
    assert module.should_cancel_from_status(None) == (False, "pipeline status missing")


def test_completion_parser_still_recognizes_real_pipeline_execution() -> None:
    module = _load_module()
    record = {
        "original_request": {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {
                                "command": "bash tools/triton_eval_pipeline.sh --op_name op 2>&1 | tail -80"
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "[triton-eval] done — success=true",
                        }
                    ],
                },
            ]
        }
    }

    state = module.analyze_budget("s1", record)

    assert len(state.pipeline_calls) == 1
    assert state.pipeline_calls[0].success is True
