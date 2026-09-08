#!/usr/bin/env python3
"""[R5] t3a attempt stream 采集 + running-best promote(仅 RL 接线,agent 无感)。

每次 skill_script_hook 代跑 evaluate_ascendc.sh 后调用一次:
  1. 判决记录追加进 attempt stream(jsonl,过程分数据源);
  2. 按判据与 running best 比较,更好则把 {op}/ 快照进 ranked list(top-5)。

判据(当场免费算的近似 reward;最终名次由 judge 侧我方判分链说了算,这里只影响判几次):
  PASS(全 case)> 部分通过 > A类/D类/其他 → case 通过数 → speedup → 时间戳(后者优先)

存储位置(防篡改排序):
  $POLAR_T3A_CANDIDATES_DIR(显式指定)> $ARTIFACTS_DIR/t3a_candidates(gateway 侧,agent 改不到)
  > workdir/output/.t3a/t3a_candidates(降级:agent 可写,judge 侧用 sha256 校验兜底)
快照与 index 都带 sha256;judge 验收时逐字核对,篡改即作废该候选。
全部 best-effort:任何失败只告警,绝不阻塞 hook 主流程。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

_TAR_EXCLUDES = [
    "--exclude=build", "--exclude=dist", "--exclude=*.so", "--exclude=*.o", "--exclude=*.a",
    "--exclude=*.whl", "--exclude=__pycache__", "--exclude=*.egg-info",
    "--exclude=judge_out", "--exclude=output", "--exclude=.git",
]
_MAX_CANDIDATES = 5
_CASE_RE = re.compile(r"case\[(\d+)\]:([^\n]*)")
_FAIL_RE = re.compile(r"mismatch|differ|FAIL|error", re.IGNORECASE)
_RATIO_RE = re.compile(r"Result:\s*(\d+)\s*/\s*(\d+)\s+passed")
_SPEEDUP_STDOUT_RE = re.compile(r"geomean_speedup[\"'\s:=]+([0-9.eE+\-]+)")
_SRC_EXTS = (".cpp", ".cc", ".h", ".hpp", ".py", ".cmake", ".txt")
_SRC_SKIP_DIRS = ("build", "dist", "__pycache__", ".git", "judge_out", "output")


def _log(msg: str) -> None:
    print(f"[t3a-running-best] {msg}", file=sys.stderr)


def _extract_speedup(op_dir: str, stdout: str) -> float | None:
    """[F6] 真实 speedup:{op}/performance.json 优先,stdout 兜底;取不到 None。
    rank key 用它区分同为 PASS 的候选(原恒 None → 退化成「最近通过」)。"""
    try:
        with open(os.path.join(op_dir, "performance.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        for k in ("geomean_speedup", "mean_speedup", "median_speedup"):
            v = d.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return float(v)
    except Exception:
        pass
    m = _SPEEDUP_STDOUT_RE.search(stdout or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _src_hash(op_dir: str) -> str:
    """[F6] 工程源码内容 hash(kernel/host/register/model_new_*):源码没变不 promote,
    重复执行同一版代码不再白拿 +0.06。"""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(op_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SRC_SKIP_DIRS)
        for f in sorted(files):
            if not f.endswith(_SRC_EXTS):
                continue
            p = os.path.join(root, f)
            try:
                h.update(os.path.relpath(p, op_dir).encode())
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 16), b""):
                        h.update(chunk)
            except OSError:
                continue
    return h.hexdigest()


def _candidates_dir(cwd: str) -> str:
    explicit = os.environ.get("POLAR_T3A_CANDIDATES_DIR")
    if explicit:
        return explicit
    artifacts = os.environ.get("ARTIFACTS_DIR")
    if artifacts:
        return os.path.join(artifacts, "t3a_candidates")
    return os.path.join(cwd, "output", ".t3a", "t3a_candidates")


def _stream_path(cand_dir: str) -> str:
    return os.path.join(os.path.dirname(cand_dir), "t3a_attempt_stream.jsonl")


def _extract_case_stats(stdout: str) -> tuple[int, int]:
    """(passed, total):case[N]: 行优先;没有则退 verification 的 Result: P/T passed 行。
    (与 builder 侧 attempt_spans.t3a_case_stats 同规则,改动两边同步)"""
    cases: dict[str, bool] = {}
    for idx, rest in _CASE_RE.findall(stdout or ""):
        cases[idx] = cases.get(idx, True) and (not _FAIL_RE.search(rest))
    if cases:
        return sum(1 for ok in cases.values() if ok), len(cases)
    m = _RATIO_RE.search(stdout or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank_key(entry: dict) -> tuple:
    return (
        1 if entry.get("classification") == "PASS" else 0,
        int(entry.get("case_pass") or 0),
        float(entry.get("speedup") or 0.0),
        float(entry.get("ts") or 0.0),
    )


def _load_index(cand_dir: str) -> list[dict]:
    try:
        with open(os.path.join(cand_dir, "index.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_index(cand_dir: str, entries: list[dict]) -> None:
    tmp = os.path.join(cand_dir, ".index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(cand_dir, "index.json"))


def _find_op_dir(op: str, cwd: str, project_root: str | None) -> str | None:
    for base in (cwd, project_root or ""):
        if not base:
            continue
        cand = os.path.join(base, op)
        if os.path.isdir(cand):
            return cand
    return None


def _guess_op_name(cwd: str, project_root: str | None) -> str | None:
    """POLAR_OP_NAME 未注入时,从 input/*.py 的文件名推(判分拓扑里 input/{op}.py 恒在)。"""
    for base in (cwd, project_root or ""):
        if not base:
            continue
        in_dir = os.path.join(base, "input")
        if not os.path.isdir(in_dir):
            continue
        pys = sorted(f for f in os.listdir(in_dir) if f.endswith(".py"))
        if pys:
            return os.path.splitext(pys[0])[0]
    return None


def record_attempt(
    *,
    command: str,
    classification: str | None,
    exit_code: int,
    stdout: str,
    duration_ms: int,
    cwd: str,
    project_root: str | None,
    script: str = "evaluate_ascendc.sh",
) -> None:
    """hook 每次代跑 evaluate_ascendc.sh 后调用。任何异常只告警。"""
    try:
        op = os.environ.get("POLAR_OP_NAME") or _guess_op_name(cwd, project_root)
        cand_dir = _candidates_dir(cwd)
        os.makedirs(cand_dir, exist_ok=True)
        case_pass, case_total = _extract_case_stats(stdout)
        rec = {
            "ts": time.time(),
            "script": script,
            "classification": classification,
            "exit_code": exit_code,
            "case_pass": case_pass,
            "case_total": case_total,
            "duration_ms": duration_ms,
        }
        with open(_stream_path(cand_dir), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if not op:
            return
        op_dir = _find_op_dir(op, cwd, project_root)
        if op_dir is None:
            return
        promoted_rank = _maybe_promote(op, op_dir, classification, case_pass, case_total, cand_dir, stdout)
        if promoted_rank is not None:
            with open(_stream_path(cand_dir), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "event": "promote", "rank": promoted_rank},
                                    ensure_ascii=False) + "\n")
    except Exception as exc:  # best-effort,never block
        _log(f"record_attempt failed: {type(exc).__name__}: {exc}")


def _maybe_promote(
    op: str,
    op_dir: str,
    classification: str | None,
    case_pass: int,
    case_total: int,
    cand_dir: str,
    stdout: str = "",
) -> int | None:
    entries = _load_index(cand_dir)
    src_hash = _src_hash(op_dir)
    best = max(entries, key=_rank_key, default=None)
    if best is not None and best.get("src_hash") == src_hash:
        return None  # [F6] 源码未变:重复执行不 promote、不记分
    candidate = {
        "classification": classification,
        "case_pass": case_pass,
        "case_total": case_total,
        "speedup": _extract_speedup(op_dir, stdout),
        "src_hash": src_hash,
        "ts": time.time(),
    }
    if best is not None and _rank_key(best) >= _rank_key(candidate):
        return None  # 没有更好,不 promote

    tar_path = os.path.join(cand_dir, f"cand_{int(candidate['ts'] * 1000)}.tar.gz")
    subprocess.run(
        ["tar", "czf", tar_path, *_TAR_EXCLUDES, "-C", os.path.dirname(op_dir), os.path.basename(op_dir)],
        check=True,
    )
    candidate.update({
        "op": op,
        "file": tar_path,
        "sha256": _sha256(tar_path),
        "tamper_evident": not cand_dir.startswith(
            os.environ.get("ARTIFACTS_DIR") or "\0"),
    })
    entries.append(candidate)
    entries.sort(key=_rank_key, reverse=True)
    for path in [e["file"] for e in entries[_MAX_CANDIDATES:] if e.get("file")]:
        try:
            os.remove(path)
        except OSError:
            pass
    entries = entries[:_MAX_CANDIDATES]
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    _save_index(cand_dir, entries)
    return 1
