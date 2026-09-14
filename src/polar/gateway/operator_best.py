"""Retain T2A evaluated candidates outside the agent's session bind mount.

Reuse the existing packer's hash checks, reward ordering and atomic promotion.
This protects artifact retention, not the truth of agent-side metrics: the
selected source is still independently compiled and judged in POSTRUN.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from tempfile import TemporaryDirectory

from polar.gateway.dispatcher import ManagedSession


PACK = Path(__file__).resolve().parents[3] / "operator_runtime_t2a/tools/pack_submission.sh"


def best_root(session_dir: Path) -> Path:
    # A sibling of the mounted session directory, never a child of it.
    return session_dir.parent / "_operator_best" / session_dir.name


def best_submission(session_dir: Path, op_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", op_name):
        raise ValueError("Invalid operator name")
    return best_root(session_dir) / "output/submission" / f"{op_name}_impl.best.tar.gz"


def selected_candidate(session_dir: Path, op_name: str) -> Path | None:
    best = best_submission(session_dir, op_name)
    meta = best.with_name(f".{op_name}_impl.best.meta.json")
    if not meta.exists():
        return None
    data = json.loads(meta.read_text())
    digest = data.get("candidate_sha256", "")
    if data.get("op_name") != op_name or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("Invalid retained best metadata")
    candidate = best_root(session_dir) / "candidates" / f"{digest}.tar.gz"
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Retained best candidate hash mismatch")
    # The existing packer commits metadata first. This frozen input remains
    # valid even if the process dies before replacing its convenience best tar.
    return candidate


def evaluation_records(session_dir: Path, op_name: str) -> dict:
    best = best_submission(session_dir, op_name)
    meta = best.with_name(f".{op_name}_impl.best.meta.json")
    records = {}
    for path in (best_root(session_dir) / "evaluations").glob("*.json"):
        record = json.loads(path.read_text())
        if record["record_id"] != path.stem or record["op_name"] != op_name:
            raise RuntimeError("Invalid retained evaluation record")
        records[path.stem] = record
    return {
        "records": records,
        "best_record_id": json.loads(meta.read_text()).get("evaluation_record_id") if meta.exists() else None,
    }


async def retain_best(managed: ManagedSession, candidate: str, metrics: str) -> dict:
    runtime = managed.runtime
    spec = managed.request.evaluator
    if runtime is None or spec is None or spec.strategy != "operator_judge":
        raise ValueError("No active operator runtime")
    if "ascendc_eval_pipeline.sh" not in str(spec.config.get("judge_command", "")):
        raise ValueError("Best retention is only available for the T2A pipeline")
    op_name = str(spec.config.get("op_name") or "")
    best = best_submission(managed.session_dir, op_name)
    workdir = PurePosixPath(runtime.spec.workdir or str(spec.config.get("workdir") or "/polar/session"))
    for name, raw in (("candidate", candidate), ("metrics", metrics)):
        path = PurePosixPath(raw)
        if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(workdir):
            raise ValueError(f"{name} must be an absolute path under the operator workdir")
    if not PurePosixPath(candidate).is_relative_to(workdir / "output/.selfcheck/candidates"):
        raise ValueError("Expected a packed Pipeline candidate")
    if PurePosixPath(metrics).name != "metrics.json":
        raise ValueError("Expected Pipeline metrics.json")

    async with managed.operator_best_lock:
        root = best_root(managed.session_dir)
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="incoming-", dir=root) as incoming:
            staged = Path(incoming) / "candidate.tar.gz"
            result = Path(incoming) / "metrics.json"
            await runtime.download_file(candidate, str(staged))
            await runtime.download_file(metrics, str(result))
            digest = hashlib.sha256(staged.read_bytes()).hexdigest()
            data = json.loads(result.read_text())
            if (not isinstance(data, dict) or data.get("op_name") != op_name
                    or data.get("evaluated_candidate_sha256") != digest):
                raise ValueError("Candidate does not match evaluated metrics")
            # Keep recovery inputs outside the mount too; local cache deletion
            # must not break the packer's interrupted-promotion recovery.
            frozen = root / "candidates" / f"{digest}.tar.gz"
            frozen.parent.mkdir(exist_ok=True)
            os.replace(staged, frozen)
            metrics_digest = hashlib.sha256(result.read_bytes()).hexdigest()
            frozen_metrics = root / "candidates" / f"{metrics_digest}.metrics.json"
            result.rename(frozen_metrics)
            # The unique packed candidate path identifies one actual invocation;
            # content hashes alone would merge separate runs of identical code.
            record_id = hashlib.sha256(json.dumps([candidate, digest, metrics_digest]).encode()).hexdigest()
            record_path = root / "evaluations" / f"{record_id}.json"
            env = {**os.environ, "WORKDIR": str(root), "PY_BIN": sys.executable,
                   "POLAR_OPERATOR_BEST_URL": "", "POLAR_RUNTIME_SESSION_DIR": str(root / "no-mirror"),
                   "POLAR_EVALUATION_RECORD_PATH": str(record_path)}
            proc = await asyncio.create_subprocess_exec(
                "bash", str(PACK), op_name, "--promote", "--candidate", str(frozen),
                "--metrics", str(frozen_metrics), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await proc.communicate()
            except BaseException:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
                raise
            if proc.returncode:
                raise RuntimeError(f"Host best promotion failed: {stderr.decode(errors='replace')[-1000:]}")
            message = next((line for line in stdout.decode().splitlines()
                            if line.startswith(("[pack] best 已更新", "[pack] best 保持不变", "[pack] best 未更新"))), None)
            if message is None:
                raise RuntimeError("Host best promotion returned no decision")
            if not record_path.is_file():
                raise RuntimeError("Host promotion did not retain an evaluation record")
            message += f"\n[pack] evaluation_record={record_id}"
            # These are convenience copies for agent rollback and existing
            # observers. They never participate in the host's next comparison.
            if best.is_file():
                meta = best.with_name(f".{op_name}_impl.best.meta.json")
                for local, remote in (
                    (best, workdir / "output/submission" / best.name),
                    (meta, workdir / "output/submission" / meta.name),
                    (best, PurePosixPath(runtime.runtime_session_dir) / "submission" / best.name),
                ):
                    try:
                        await runtime.upload_file(str(local), str(remote))
                    except Exception:
                        # The authoritative copy is already safe, including if
                        # the runtime dies while the convenience copy is written.
                        message += "\n[pack] WARN: 本地 best 副本同步失败；Gateway 留存仍有效"
                        break
            return {"message": message, "source": "gateway_best"}
