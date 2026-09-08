"""Derive T3A prompts without changing tasks, cases, labels or sampling metadata."""
import argparse
import json
import re
from pathlib import Path


def convert(row: dict) -> dict:
    op = row["metadata"]["op_name"]
    if not isinstance(op, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", op):
        raise ValueError(f"Invalid op_name: {op!r}")
    prompt = (
        f"Implement the AscendC operator {op}. The reference is input/{op}.py "
        f"and its full test cases are input/{op}.json. Output project: {op}/.\n"
        "Follow ./CLAUDE.md and the original cannbot skill workflow. No project skeleton "
        "has been generated: the project-init skill must initialize it.\n"
        "As the main orchestrator, dispatch the registered subagent "
        "tilelang2ascendc-kernel-generator for end-to-end development, passing the reference, "
        "case and output paths. Use the same subagent for precision-repair reentry. "
        "Monitor its progress; do not replace it with general-purpose or implement the kernel yourself.\n"
        "The developer must follow all applicable phases: classification, initialization, case "
        "simplification, design/development, AscendC verification, profiling, full-case validation "
        "and trace recording. Use the original skill scripts, with NPU work under the provided "
        "lease mechanism. Read the relevant skill instructions. The input/ reference files are immutable. "
        "Copy the reference and case JSON into the output project; Phase 2 may simplify only that "
        "working copy after backing it up, and Phase 6 must restore the full cases before validation. "
        "Do not modify verification scripts or tools; do not set device visibility or run npu-smi. "
        "Device selection belongs to the lease executor, not an npu argument chosen by the agent.\n"
        "Evaluation hooks collect candidate snapshots automatically; the separate judge rebuilds "
        "and scores them. Do not search for a T2A fixed pipeline or manually package a submission. "
        "This is non-interactive: proceed without asking the user questions.\n"
    )
    return {**row, "prompt": [{"role": "user", "content": prompt}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must differ")
    rows = [convert(json.loads(line)) for line in args.source.read_text().splitlines() if line.strip()]
    # Refuse overwrite; a new derived dataset must not replace a user's existing one.
    with args.destination.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} T3A tasks to {args.destination}")


if __name__ == "__main__":
    main()
