from __future__ import annotations

from polar.rollout.balancer import NodeScheduler
from polar.rollout.pipeline import Pipeline


def test_pipeline_result_paths_are_scoped_by_training_run(tmp_path) -> None:
    pipeline = Pipeline(
        callback_url="http://127.0.0.1:8080/callbacks/session_result",
        save_dir=str(tmp_path),
        scheduler=NodeScheduler(),
    )

    path = pipeline.result_path_for(
        "train-a-polar-op-0-0",
        "sk-session",
        {"run_id": "train-a"},
    )

    assert path == str(
        tmp_path
        / "run_train-a"
        / "task_train-a-polar-op-0-0"
        / "ses_sk-session.json"
    )


def test_pipeline_legacy_result_paths_stay_flat(tmp_path) -> None:
    pipeline = Pipeline(
        callback_url="http://127.0.0.1:8080/callbacks/session_result",
        save_dir=str(tmp_path),
        scheduler=NodeScheduler(),
    )

    path = pipeline.result_path_for("polar-op-0-0", "sk-session")

    assert path == str(tmp_path / "task_polar-op-0-0" / "ses_sk-session.json")
