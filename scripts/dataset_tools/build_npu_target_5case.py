#!/usr/bin/env python3
"""Build a fresh NPUKernelBench-targeted five-case training dataset.

The NPUKernelBench source cases are reduced by deterministic diversity sampling;
the complete source JSON is retained under full_cases/ for audit. CANN and CUDA
tasks are selected by operator-family overlap and rewritten to one shared
AscendC task prompt. Existing datasets are never modified.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil
from pathlib import Path

FAMILIES = (
    "permute", "cat", "split", "pad", "repeat", "gather", "scatter", "nonzero",
    "pool", "interpol", "quant", "swiglu", "nms", "iou", "add", "abs", "cumsum",
    "histc", "sum", "sort", "rmsnorm", "rope", "attention", "softmax", "embedding",
    "rotary", "fft", "topk", "groupnorm", "moe", "layernorm", "norm",
)
CUDA_PER_FAMILY = 5


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(meta: dict) -> str | None:
    text = (str(meta.get("op_name", "")) + " " + " ".join(map(str, meta.get("ops", [])))).lower()
    return next((x for x in FAMILIES if x in text), None)


def json_sig(case: dict) -> str:
    def shape(x):
        if isinstance(x, dict):
            if "shape" in x: return (tuple(x.get("shape") or ()), x.get("dtype"), x.get("type"), x.get("name"))
            return tuple(sorted((k, shape(v)) for k, v in x.items() if k in {"value", "dtype", "shape", "type", "name"}))
        if isinstance(x, list): return tuple(shape(v) for v in x)
        return type(x).__name__
    return repr(shape(case))


def select_five(cases: list[dict]) -> list[dict]:
    if len(cases) <= 5: return cases
    # Greedy farthest-point selection over shape/dtype/attribute signatures,
    # seeded with endpoints so small and large regimes are retained.
    sigs = [json_sig(x) for x in cases]
    chosen = [0, len(cases) - 1]
    while len(chosen) < 5:
        best = max(
            (i for i in range(len(cases)) if i not in chosen),
            key=lambda i: min(sum(a != b for a, b in zip(sigs[i], sigs[j])) + abs(len(sigs[i]) - len(sigs[j])) for j in chosen),
        )
        chosen.append(best)
    chosen.sort()
    out = []
    for i in chosen:
        c = dict(cases[i]); c["source_case_index"] = i; out.append(c)
    return out


def prompt(op: str) -> str:
    return f"""Implement an AscendC operator for Ascend NPU. The reference task is at input/{op}.py (class Model with get_input_groups() and get_init_inputs() when present). Complete the pre-generated project directory `{op}/` in place. Preserve the reference input/output contract and all five JSON cases in input/{op}.json. The candidate must use torch.ops.npu.{op} in ModelNew.forward, with no plain-torch fallback. Use the fixed AscendC validation entry and submit a single tarball under output/submission/{op}_impl.tar.gz."""


def load_rows(path: Path) -> dict[str, dict]:
    return {json.loads(x)["label"]: json.loads(x) for x in path.read_text().splitlines() if x.strip()}


def copy_task(src: Path, dst: Path, op: str, cases: list[dict]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "op_tasks" / (op + ".py"), dst / (op + ".py"))
    (dst / (op + ".json")).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in cases) + "\n")


def copy_npu_task(py: Path, dst: Path, op: str, cases: list[dict]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    source = py.read_text()
    # NPUKernelBench references its sidecar by the original basename.
    source = source.replace(f"{py.stem}.json", f"{op}.json")
    (dst / (op + ".py")).write_text(source)
    (dst / (op + ".json")).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in cases) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu-root", type=Path, default=Path("/home/docker/NPUKernelBench"))
    ap.add_argument("--merged", type=Path, default=Path("/home/docker/datasets/operator_tasks_merged_5case"))
    ap.add_argument("--cann-fixed", type=Path, default=Path("/home/docker/datasets/operator_tasks_merged_5case_cannops_fixed"))
    ap.add_argument("--out", type=Path, default=Path("/home/docker/datasets/operator_tasks_npu_target_5case_v1"))
    a = ap.parse_args(); out = a.out.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite existing output: {out}")
    (out / "data/op_tasks").mkdir(parents=True); (out / "full_cases").mkdir()
    rows = []; selected = {"npu": [], "cann": [], "cuda": []}

    # All level 1/2 NPUKernelBench tasks, reduced to five diverse native cases.
    npu_count = 0
    for level in (1, 2):
        for py in sorted((a.npu_root / f"level{level}").glob("*.py")):
            op = f"npukernelbench_level{level}_{py.stem}"; js = py.with_suffix(".json")
            cases = [json.loads(x) for x in js.read_text().splitlines() if x.strip()]
            chosen = select_five(cases)
            copy_npu_task(py, out / "data/op_tasks", op, chosen)
            shutil.copy2(js, out / "full_cases" / f"{op}.jsonl")
            meta = {"op_name": op, "entry_point": "Model", "operator_backend": "ascendc", "arch": "ascend910b1", "ops": [op.split("_", 1)[1]], "data_source": "npu-kernel-bench", "ability": "code", "level": level, "case_count": 5, "task_mode": "npu_target", "source_case_count": len(cases), "source_case_indices": [x.get("source_case_index") for x in chosen], "task_source_sha256": digest(out / "data/op_tasks" / (op + ".py"))}
            rows.append({"label": op, "metadata": meta, "prompt": [{"role": "user", "content": prompt(op)}]}); selected["npu"].append(op); npu_count += 1

    source_rows = load_rows(a.merged / "operator_tasks.jsonl")
    # CANN: retain every fixed task whose declared family overlaps the target family set.
    for op, row in sorted(source_rows.items()):
        m = row["metadata"]
        if m.get("data_source") != "cann-ops" or not family(m): continue
        cases = [json.loads(x) for x in (a.cann_fixed / "op_tasks" / (op + ".json")).read_text().splitlines() if x.strip()]
        copy_task(a.cann_fixed, out / "data/op_tasks", op, cases[:5]); meta = dict(m, task_mode="npu_target", target_family=family(m), case_count=5, task_source_sha256=digest(out / "data/op_tasks" / (op + ".py")))
        rows.append({"label": op, "metadata": meta, "prompt": [{"role": "user", "content": prompt(op)}]}); selected["cann"].append(op)
    # CUDA: cap each matched family at five tasks, sorted by difficulty then name.
    by = {}
    for op, row in source_rows.items():
        m = row["metadata"]
        if m.get("data_source") == "cuda_llm" and family(m): by.setdefault(family(m), []).append((float(m.get("difficulty_score") or 0), op, row))
    for fam, vals in sorted(by.items()):
        for _, op, row in sorted(vals, key=lambda x: (x[0], x[1]))[:CUDA_PER_FAMILY]:
            cases = [json.loads(x) for x in (a.merged / "op_tasks" / (op + ".json")).read_text().splitlines() if x.strip()]
            copy_task(a.merged, out / "data/op_tasks", op, cases[:5]); m = dict(row["metadata"], task_mode="npu_target", target_family=fam, case_count=5, task_source_sha256=digest(out / "data/op_tasks" / (op + ".py")))
            rows.append({"label": op, "metadata": m, "prompt": [{"role": "user", "content": prompt(op)}]}); selected["cuda"].append(op)
    (out / "operator_tasks.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    (out / "selection.json").write_text(json.dumps({"npu_level1_2": npu_count, "selected": selected, "case_policy": "five diverse native cases; full NPU JSON retained under full_cases", "cann_source": str(a.cann_fixed), "cuda_source": str(a.merged)}, ensure_ascii=False, indent=2))
    (out / "README.md").write_text(f"NPU target dataset v1\n\nNPU level1/2: {npu_count} tasks, five representative cases selected from native cases. CANN fixed matched tasks: {len(selected['cann'])}. CUDA matched tasks: {len(selected['cuda'])}. Full native NPU case files are under full_cases/.\n")
    print(json.dumps({"out": str(out), "tasks": len(rows), "npu": npu_count, "cann": len(selected['cann']), "cuda": len(selected['cuda'])}, indent=2))

if __name__ == "__main__": main()
