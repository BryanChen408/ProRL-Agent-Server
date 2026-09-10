"""Derive T3A prompts without changing tasks, cases, labels or sampling metadata."""
import argparse
import json
import re
from pathlib import Path


def convert(row: dict, *, case_mode: str = "full", developer: bool = False) -> dict:
    op = row["metadata"]["op_name"]
    if not isinstance(op, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", op):
        raise ValueError(f"Invalid op_name: {op!r}")
    if case_mode not in ("full", "simple"):
        raise ValueError(f"Invalid case_mode: {case_mode!r}")
    case_instructions = (
        "This task uses the NPUKernelBench simple case set: exactly 5 provided cases. "
        "Phase 2 must check and back up the working JSON, retaining all 5 cases unchanged; "
        "do not simplify it again. "
        "Do not search for or substitute the original larger benchmark case set. "
        "Here simple describes the case set, not the operator's development route; "
        "classify the operator using the existing skill rules. "
        if case_mode == "simple" else
        "Phase 2 may simplify only that working copy after backing it up. "
    )
    prompt = (
        f"Implement the AscendC operator {op}. The reference is input/{op}.py "
        f"and its provided test cases are input/{op}.json. Output project: {op}/.\n"
        "Follow ./CLAUDE.md and the original cannbot developer workflow. No project skeleton "
        "has been generated: the developer subagent must initialize it using its Phase 1.2 "
        "kernel/ layout and the installed project-init templates. Do not run the standalone "
        "project-init workflow or create ascend-kernel/csrc/ops/.\n"
        + ("You are the developer: execute the native developer workflow yourself, "
        "using Skill for design, translation and diagnosis. Do not dispatch nested agents.\n"
        if developer else
        "As the main orchestrator, dispatch the registered subagent "
        "tilelang2ascendc-kernel-generator for end-to-end development, passing the reference, "
        "case and output paths, this task's case policy, lease policy and Phase 6 policy verbatim. "
        "Keep the output root unchanged on reentry. Use the same subagent for precision-repair reentry. "
        "Monitor its progress; do not replace it with general-purpose or implement the kernel yourself.\n") +
        "The developer must follow all applicable phases: classification, initialization, case "
        "preparation, design/development, AscendC verification, profiling and trace recording. "
        "Phase 6 is disabled in this RL workflow: go directly from Phase 5 to Phase 7; "
        "do not restore cases or run an extra final validation in the agent. "
        "The separate judge validates candidates against all provided input cases. "
        "Use the original skill scripts, with NPU work under the provided "
        "lease mechanism. Read the relevant skill instructions. The input/ reference files are immutable. "
        "Copy the reference and case JSON into the output project. "
        + case_instructions +
        "Do not modify verification scripts or tools; do not set device visibility or run npu-smi. "
        "Device selection belongs to the lease executor, not an npu argument chosen by the agent.\n"
        "Evaluation hooks collect candidate snapshots automatically; the separate judge rebuilds "
        "and scores them. TileLang validation is intermediate; it does not replace Phase 4 AscendC "
        "development and evaluate_ascendc.sh. Before reporting completion, inspect the real "
        "AscendC evaluation result, kernel/ sources, model_new_ascendc.py and trace.md. "
        "If the subagent returns early, resume that developer from the unfinished phase. "
        "Do not claim success from stubs or a TileLang-only project. "
        "Pipeline/verify calls have no count limit; the external pipeline time budget controls execution. "
        "Do not search for a T2A fixed pipeline or manually package a submission. "
        "This is non-interactive: proceed without asking the user questions.\n"
    )
    return {**row, "prompt": [{"role": "user", "content": prompt}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--case-mode", choices=("full", "simple"), default="full",
                        help="simple: keep all 5 curated NPUKernelBench cases")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must differ")
    rows = [convert(json.loads(line), case_mode=args.case_mode)
            for line in args.source.read_text().splitlines() if line.strip()]
    # Refuse overwrite; a new derived dataset must not replace a user's existing one.
    with args.destination.open("x", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} T3A tasks to {args.destination}")


if __name__ == "__main__":
    main()
