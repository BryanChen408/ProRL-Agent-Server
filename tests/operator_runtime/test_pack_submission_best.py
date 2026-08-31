"""T2A submission 打包与评测后 best 提升契约。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "operator_runtime_t2a" / "tools" / "pack_submission.sh"
REWARD = ROOT / "src" / "polar" / "trajectory" / "evaluator" / "operator_reward.py"
OP = "op_test"

_reward_spec = importlib.util.spec_from_file_location("t2a_operator_reward", REWARD)
assert _reward_spec and _reward_spec.loader
_reward_module = importlib.util.module_from_spec(_reward_spec)
sys.modules[_reward_spec.name] = _reward_module
_reward_spec.loader.exec_module(_reward_module)
is_infra_failure = _reward_module.is_infra_failure
reward_from_metrics = _reward_module.reward_from_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_workspace(root: Path, marker: str = "initial") -> Path:
    task = root / OP
    (task / "kernel" / "op_host").mkdir(parents=True)
    (task / "kernel" / "op_kernel").mkdir(parents=True)
    (task / "kernel" / "CMakeLists.txt").write_text("# test\n", encoding="utf-8")
    (task / "kernel" / "op_host" / "host.cpp").write_text("// host\n", encoding="utf-8")
    (task / "kernel" / "op_kernel" / "kernel.cpp").write_text("// kernel\n", encoding="utf-8")
    _set_marker(root, marker)
    return task


def _set_marker(root: Path, marker: str) -> None:
    (root / OP / "model_new_ascendc.py").write_text(
        f"# marker:{marker}\n# torch.ops.npu\n", encoding="utf-8"
    )


def _run_pack(
    root: Path,
    *args: str,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WORKDIR"] = str(root)
    env["POLAR_RUNTIME_SESSION_DIR"] = str(root / "no-session-mirror")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", str(PACK), OP, *args],
        cwd=cwd or root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _candidate(root: Path, name: str, marker: str) -> Path:
    _set_marker(root, marker)
    path = root / "output" / ".selfcheck" / "candidates" / f"{name}.tar.gz"
    proc = _run_pack(
        root,
        "--candidate",
        str(path),
        "--public",
        str(root / "output" / "submission" / f"{OP}_impl.tar.gz"),
    )
    assert proc.returncode == 0, proc.stderr
    return path


def _metrics(root: Path, name: str, candidate: Path, **values: object) -> Path:
    data: dict[str, object] = {
        "op_name": OP,
        "success": False,
        "ast_check_ok": True,
        "correctness_ok": False,
        "error_type": "ascendc_compile_failed",
        "perf_data": None,
        "cases_passed": None,
        "cases_total": None,
    }
    data.update(values)
    data["evaluated_candidate_sha256"] = _sha256(candidate)
    path = root / "metrics" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _promote(
    root: Path,
    candidate: Path,
    metrics: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run_pack(
        root,
        "--promote",
        "--candidate",
        str(candidate),
        "--metrics",
        str(metrics),
        env_overrides=env_overrides,
    )


def _best(root: Path) -> Path:
    return root / "output" / "submission" / f"{OP}_impl.best.tar.gz"


def _meta(root: Path) -> Path:
    return root / "output" / "submission" / f".{OP}_impl.best.meta.json"


def _marker(tar_path: Path) -> str:
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.extractfile(f"{OP}/model_new_ascendc.py")
        assert member is not None
        return member.read().decode("utf-8")


def test_pack_only_keeps_public_path_discoverable_without_pre_evaluation_best(tmp_path: Path):
    _make_workspace(tmp_path)
    nested = tmp_path / OP / "kernel" / "op_kernel"
    candidate = Path("output/.selfcheck/candidates/relative.tar.gz")
    proc = _run_pack(tmp_path, "--candidate", str(candidate), cwd=nested)

    public = tmp_path / "output" / "submission" / f"{OP}_impl.tar.gz"
    resolved_candidate = tmp_path / candidate
    assert proc.returncode == 0, proc.stderr
    assert resolved_candidate.exists() and public.exists()
    assert resolved_candidate.read_bytes() == public.read_bytes()
    assert not _best(tmp_path).exists() and not _meta(tmp_path).exists()
    assert f"output/submission/{OP}_impl.tar.gz" in proc.stdout
    assert str(resolved_candidate) not in proc.stdout


def test_later_bad_candidate_cannot_overwrite_higher_reward_best(tmp_path: Path):
    _make_workspace(tmp_path)
    almost = _candidate(tmp_path, "almost", "almost")
    compile_bad = _candidate(tmp_path, "compile", "compile-bad")
    worse_cases = _candidate(tmp_path, "worse", "worse-cases")

    p1 = _promote(
        tmp_path,
        almost,
        _metrics(
            tmp_path,
            "almost",
            almost,
            error_type="correctness_failed",
            cases_passed=4,
            cases_total=5,
        ),
    )
    p2 = _promote(tmp_path, compile_bad, _metrics(tmp_path, "compile", compile_bad))
    p3 = _promote(
        tmp_path,
        worse_cases,
        _metrics(
            tmp_path,
            "worse",
            worse_cases,
            error_type="correctness_failed",
            cases_passed=2,
            cases_total=5,
        ),
    )

    assert p1.returncode == p2.returncode == p3.returncode == 0
    assert "best 已更新" in p1.stdout
    assert "best 保持不变" in p2.stdout and "best 保持不变" in p3.stdout
    assert _best(tmp_path).read_bytes() == almost.read_bytes()
    assert "marker:almost" in _marker(_best(tmp_path))


def test_equal_reward_keeps_earlier_candidate(tmp_path: Path):
    _make_workspace(tmp_path)
    first = _candidate(tmp_path, "first", "first")
    later = _candidate(tmp_path, "later", "later")
    first_metrics = _metrics(
        tmp_path,
        "first",
        first,
        error_type="correctness_failed",
        cases_passed=5,
        cases_total=10,
    )
    later_metrics = _metrics(
        tmp_path,
        "later",
        later,
        error_type="correctness_failed",
        cases_passed=None,
        cases_total=None,
    )

    assert _promote(tmp_path, first, first_metrics).returncode == 0
    proc = _promote(tmp_path, later, later_metrics)
    assert proc.returncode == 0 and "best 保持不变" in proc.stdout
    assert _best(tmp_path).read_bytes() == first.read_bytes()


def test_promote_uses_evaluated_tar_not_source_tree_or_latest_public_tar(tmp_path: Path):
    _make_workspace(tmp_path)
    evaluated = _candidate(tmp_path, "evaluated", "evaluated-A")
    metrics = _metrics(
        tmp_path,
        "evaluated",
        evaluated,
        correctness_ok=True,
        error_type="benchmark_failed",
    )
    latest = _candidate(tmp_path, "latest", "unverified-B")
    public = tmp_path / "output" / "submission" / f"{OP}_impl.tar.gz"
    assert public.read_bytes() == latest.read_bytes()

    proc = _promote(tmp_path, evaluated, metrics)
    assert proc.returncode == 0, proc.stderr
    assert _best(tmp_path).read_bytes() == evaluated.read_bytes()
    assert "marker:evaluated-A" in _marker(_best(tmp_path))
    assert "marker:unverified-B" in _marker(public)


@pytest.mark.parametrize(
    "values",
    [
        {"success": False, "ast_check_ok": False, "error_type": "ast_check_failed"},
        {"success": False, "ast_check_ok": True, "error_type": "ascendc_compile_failed"},
        {"success": False, "ast_check_ok": True, "error_type": "ascendc_run_timeout"},
        {"success": False, "ast_check_ok": True, "error_type": "unknown_failure"},
        {
            "success": False,
            "ast_check_ok": True,
            "error_type": "correctness_failed",
            "cases_passed": 8,
            "cases_total": 9,
        },
        {"success": False, "ast_check_ok": True, "correctness_ok": True, "error_type": "benchmark_failed"},
        {"success": True, "ast_check_ok": True, "correctness_ok": True, "perf_data": None, "error_type": None},
        {
            "success": True,
            "ast_check_ok": True,
            "correctness_ok": True,
            "perf_data": {"speedup_vs_torch": 1.7},
            "error_type": None,
        },
    ],
)
def test_best_score_matches_authoritative_training_reward(tmp_path: Path, values: dict[str, object]):
    case_root = tmp_path / hashlib.sha256(repr(values).encode()).hexdigest()[:10]
    _make_workspace(case_root)
    candidate = _candidate(case_root, "candidate", "score-contract")
    metrics_path = _metrics(case_root, "metrics", candidate, **values)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    proc = _promote(case_root, candidate, metrics_path)
    assert proc.returncode == 0, proc.stderr
    assert not is_infra_failure(metrics)
    stored = json.loads(_meta(case_root).read_text(encoding="utf-8"))["reward_score"]
    assert stored == pytest.approx(reward_from_metrics(metrics))


def test_case_weight_override_matches_training_reward(tmp_path: Path):
    _make_workspace(tmp_path)
    candidate = _candidate(tmp_path, "weighted", "weighted")
    metrics_path = _metrics(
        tmp_path,
        "weighted",
        candidate,
        error_type="correctness_failed",
        cases_passed=3,
        cases_total=4,
    )
    env = {"POLAR_CASE_PASS_WEIGHT": "0.05"}
    proc = _promote(tmp_path, candidate, metrics_path, env)
    assert proc.returncode == 0, proc.stderr
    stored = json.loads(_meta(tmp_path).read_text(encoding="utf-8"))["reward_score"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert stored == pytest.approx(reward_from_metrics(metrics, env))


def test_infra_and_mismatched_metrics_never_promote(tmp_path: Path):
    _make_workspace(tmp_path)
    candidate = _candidate(tmp_path, "candidate", "candidate")
    infra = _metrics(
        tmp_path,
        "infra",
        candidate,
        error_type="npu_runtime_unavailable",
    )
    assert "best 未更新" in _promote(tmp_path, candidate, infra).stdout
    assert not _best(tmp_path).exists()

    mismatch = _metrics(tmp_path, "mismatch", candidate, correctness_ok=True)
    data = json.loads(mismatch.read_text(encoding="utf-8"))
    data["evaluated_candidate_sha256"] = "0" * 64
    mismatch.write_text(json.dumps(data), encoding="utf-8")
    proc = _promote(tmp_path, candidate, mismatch)
    assert proc.returncode == 0 and "哈希不匹配" in proc.stdout
    assert not _best(tmp_path).exists()


def test_concurrent_promotions_keep_global_maximum(tmp_path: Path):
    _make_workspace(tmp_path)
    low = _candidate(tmp_path, "low", "low")
    high = _candidate(tmp_path, "high", "high")
    low_metrics = _metrics(tmp_path, "low", low, error_type="ascendc_run_crashed")
    high_metrics = _metrics(
        tmp_path,
        "high",
        high,
        correctness_ok=True,
        error_type="benchmark_failed",
    )
    env = os.environ.copy()
    env["WORKDIR"] = str(tmp_path)
    env["POLAR_RUNTIME_SESSION_DIR"] = str(tmp_path / "no-session-mirror")
    calls = [
        ["bash", str(PACK), OP, "--promote", "--candidate", str(low), "--metrics", str(low_metrics)],
        ["bash", str(PACK), OP, "--promote", "--candidate", str(high), "--metrics", str(high_metrics)],
    ]
    procs = [subprocess.Popen(c, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for c in calls]
    results = [p.communicate(timeout=20) + (p.returncode,) for p in procs]

    assert all(rc == 0 for _out, _err, rc in results), results
    assert _best(tmp_path).read_bytes() == high.read_bytes()
    assert json.loads(_meta(tmp_path).read_text(encoding="utf-8"))["reward_score"] == pytest.approx(0.4)


def test_incomplete_meta_first_transaction_recovers_from_immutable_candidate(tmp_path: Path):
    _make_workspace(tmp_path)
    old = _candidate(tmp_path, "old", "old")
    pending = _candidate(tmp_path, "pending", "pending")
    current = _candidate(tmp_path, "current", "current-low")
    best = _best(tmp_path)
    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_bytes(old.read_bytes())
    meta = {
        "schema_version": 2,
        "tier": 2,
        "reward_score": 0.4,
        "candidate_sha256": _sha256(pending),
        "candidate_path": str(pending),
    }
    _meta(tmp_path).write_text(json.dumps(meta), encoding="utf-8")

    proc = _promote(tmp_path, current, _metrics(tmp_path, "current", current))
    assert proc.returncode == 0, proc.stderr
    assert "best 保持不变" in proc.stdout
    assert best.read_bytes() == pending.read_bytes()
    assert "marker:pending" in _marker(best)
