# Proposal: Migrate CANNBot Debug Strengths Back To Legacy Runtime

Date: 2026-07-02

Status: draft for discussion, not an approved implementation plan.

## Goal Under Discussion

Keep the legacy Polar operator runtime as the main RL path, but migrate the
CANNBot pieces that can plausibly improve operator correctness:

- richer verifier feedback;
- explicit AST fallback checks;
- multi-shape correctness signals;
- precision-debug guidance;
- better failure classification and repair reasoning.

The working direction is to avoid migrating the full CANNBot human-engineering
workflow. The legacy runtime should stay short, budgeted, and aligned with
Polar reward collection.

## Keep Legacy Contract

The legacy path should continue to use:

```text
src/{op_name}.py
output/submission/{op_name}_impl.py
tools/triton_eval_pipeline.sh
judge_out/metrics_error.log
judge_out/metrics.json
```

The agent-facing contract remains:

- implement `class ModelNew`;
- write the implementation under `output/submission/`;
- use the fixed Polar pipeline result as the pass/fail decision source;
- respect generation and optimization budgets;
- preserve `best.py` or the best known passing implementation.

This avoids CANNBot-specific artifact friction such as:

```text
input/{op_name}.py
output/iter_*/generated_code.py
output/generated_code.py
report.md
summary.json
session export
```

Those are useful for human review but are not required for Polar reward.

## Candidate CANNBot Pieces To Port

### 1. Structured Verifier Feedback

CANNBot-positive difference:

```text
triton-op-verifier has structured verify_result.json with
passed_cases / total_cases / failures[], multi-shape execution, precision
classification, and AccuracyError-style metrics.
```

Migration target:

Extend the legacy fixed pipeline to keep the existing `metrics_error.log` and
`metrics.json` contract, while also emitting a compact structured failure
artifact, for example:

```text
judge_out/debug_context.json
```

Suggested fields:

```text
ast_check:
  passed: bool
  errors: [...]

correctness:
  passed_cases: int
  total_cases: int
  failures:
    - case_id
      input_shapes
      input_dtypes
      expected_shape
      actual_shape
      max_abs_error
      max_rel_error
      error_kind
      short_message

benchmark:
  ran: bool
  speedup
  warnings
```

The agent may read this artifact together with `metrics_error.log`. It should
not need to inspect verifier or benchmark source code.

Detailed change shape:

- Add a verifier result adapter inside the legacy pipeline.
  - Input: the current legacy correctness execution result.
  - Output: `debug_context.json` with CANNBot-like fields.
  - Compatibility: do not remove or rename existing `metrics_error.log` /
    `metrics.json`.
- If legacy verifier already runs multiple cases, preserve all case failures
  instead of truncating to the first one.
- If a task exposes `get_input_groups()`, run all groups and include each
  failed group's shape and dtype summary.
- Add precision category fields only when available. Do not block the pipeline
  on perfect CANNBot parity in the first migration step.
- Keep agent instructions simple:

```text
After a failed fixed-pipeline run, read judge_out/metrics_error.log. If
judge_out/debug_context.json exists, also read it and use its failures[] to
localize the issue.
```

What not to do:

- Do not require the agent to call CANNBot `verify.py` directly.
- Do not expose verifier source files as normal reading targets.
- Do not change the output implementation path.

### 2. AST Fallback Precheck

CANNBot-positive difference:

```text
validate_triton_impl.py is an explicit precheck that catches PyTorch fallback,
forward not calling a kernel, and partial torch computation before expensive
correctness execution.
```

Migration target:

Port the useful part of CANNBot's AST fallback detection into the legacy
pipeline before correctness execution.

The precheck should catch:

- PyTorch fallback in `ModelNew.forward`;
- `forward` not calling the custom Triton kernel;
- wrapping the original reference op;
- partial torch computation used as the real implementation;
- missing or malformed `ModelNew`.

This should be a pipeline check, not a separate agent workflow phase.

Detailed change shape:

- Add a precheck stage inside `tools/triton_eval_pipeline.sh`, before normal
  correctness execution.
- Reuse CANNBot `validate_triton_impl.py` logic if possible, but wrap it to
  understand the legacy implementation contract:

```text
implementation file: output/submission/{op_name}_impl.py
expected class: ModelNew
task file: src/{op_name}.py
```

- On failure:
  - write a concise AST failure section into `metrics_error.log`;
  - include `ast_check.passed=false` and `ast_check.errors[]` in
    `debug_context.json`;
  - count it as the current fixed-pipeline attempt, so budgets remain coherent.
- On success:
  - continue to existing correctness evaluation.

Prompt impact:

```text
If the fixed pipeline reports an AST fallback/precheck failure, treat it as an
implementation violation and fix output/submission/{op}_impl.py. Do not bypass
the check with PyTorch logic.
```

### 3. Reference Behavior Probes

Open question: should the legacy prompt loosen the current narrow "read error,
edit, rerun" behavior?

The argument for loosening it: forbidding all exploratory code blocks useful
diagnosis for cases such as default stride, broadcasting, dtype promotion, or
output shape inference.

Candidate rule:

- The fixed pipeline result remains the pass/fail decision source.
- The agent may run small, local, read-only Python probes against `src/{op}.py`
  to understand reference semantics, shapes, dtypes, broadcasting, strides, and
  boundary behavior.
- Probe output may guide debugging, but it must not replace fixed-pipeline
  pass/fail.

CANNBot wording note:

Original CANNBot does not use the phrase "the fixed pipeline is the only
judge". Its equivalent boundary is more concrete:

- verify pass/fail must be read from `verify_result.json`
  `passed_cases == total_cases`;
- benchmark has an L1 verify gate;
- `triton-op-verifier` explicitly forbids using `torch.allclose` or custom
  methods to replace `scripts/verify.py`.

Legacy migration should follow that style instead of introducing a new slogan.

The concern with phrasing this as "after a small patch, immediately rerun the
pipeline" is that it may be too narrow. It can recreate the old failure mode
where the agent reads one error, edits locally, and reruns without doing
necessary code reading, reference inspection, or semantic localization.

Candidate wording:

```text
After a failed fixed-pipeline run, perform the necessary prompt, reference,
code, and error-log reading to localize the cause. You may run small read-only
reference probes when they clarify semantics. Then update only the submission
implementation and use the fixed pipeline to validate the result.

Probe output is for diagnosis only. Do not use probe scripts, torch.allclose,
manual inspection, or custom tests as a substitute for fixed-pipeline pass/fail.
```

### 4. Conductor-Style Failure Reasoning

Open question: should the legacy prompt explicitly ask for conductor-style
failure classification, or should this be left implicit?

Candidate direction: port the useful reasoning discipline, not the full CANNBot
conductor workflow.

CANNBot-positive difference:

```text
verify failures are summarized into verifier_error and conductor_suggestion,
which gives the next generation attempt structured context instead of only a raw
metrics_error.log.
```

Migration target:

Keep the legacy runtime single-agent and short-loop, but make the failure
feedback more actionable.

After a failed fixed-pipeline run, the agent should be encouraged to derive:

```text
failure category
most likely root cause
next implementation change
```

If adopted, this should be a debugging aid, not a new required artifact and not
a hard instruction to make only a tiny edit before rerunning.

Do not require `conductor_suggestion` files, reports, summaries, or session
exports in the legacy path.

Detailed change shape:

- Do not add a separate Conductor phase or second agent.
- Do not require the agent to write `conductor_suggestion` to disk.
- Add a short prompt instruction that the next edit should be based on:
  - the fixed-pipeline failure;
  - `debug_context.json.failures[]` when available;
  - relevant reference-code or probe findings.
- Optionally add a compact section to `metrics_error.log`, generated by the
  pipeline, such as:

```text
[debug-context]
failure_category=shape_mismatch|dtype_mismatch|precision_error|ast_fallback|runtime_error|compile_error
primary_failure=...
failed_cases=...
```

This gives the model structured repair context without importing the full
CANNBot report/export workflow.

### 5. Multi-Shape Correctness Signals

If the source task provides `get_input_groups()`, the legacy pipeline should run
all provided groups and expose per-case failures in `debug_context.json`.

This is correctness-positive, but it can lower short-term pass rate because more
cases are checked. The feedback must make clear which shape failed so the agent
can repair the implementation instead of guessing.

### 6. Precision Debug Guidance

CANNBot-positive difference:

```text
triton-precision-debug gives targeted guidance for floating-point error,
especially loss, softmax, normalization, and reductions.
```

Migration target:

Bring over the useful precision-debug guidance as documentation or a small
skill reference. Use it for:

- reductions;
- softmax or normalization;
- loss functions;
- fp16/bf16 accumulation;
- tolerance-sensitive comparisons.

This should be guidance only. Do not add a mandatory precision-debug phase.

Detailed change shape:

- Copy or reference the precision-debug skill content in the legacy runtime.
- Add a short routing rule in legacy `CLAUDE.md`:

```text
If fixed-pipeline feedback indicates precision/tolerance mismatch, reduction
error, softmax/norm instability, or fp16/bf16 accumulation error, consult the
precision-debug guidance before editing the submission.
```

- Do not force precision-debug on compile errors, shape errors, missing
  `ModelNew`, or obvious AST fallback failures.
- Do not require extra output files from the precision-debug process.

### 7. Selected Optimization References

Some CANNBot references can help correctness or clear implementation structure,
especially for pooling, interpolation, reduction, and split-kernel patterns.

Port only references that are directly useful to operator generation. Avoid
turning the legacy prompt into a long human-engineering workflow.

## Pieces Not To Port

Do not port these into the legacy RL path:

- CANNBot `output/iter_*` artifact layout;
- mandatory `output/generated_code.py` copy step;
- report and summary generation as required steps;
- session export as a required step;
- long Phase 5/6 style documentation work;
- mandatory optimizer workflow before a correct implementation exists;
- reading verifier or benchmark implementation source as normal debugging;
- task extractor or shape JSONL dependency.

These increase friction or token cost without directly improving the trusted
reward signal.

## Prompt Direction Under Discussion

One proposed prompt shift is from:

```text
read error -> edit submission -> immediately rerun pipeline
```

to:

```text
fixed pipeline is the judge;
failed pipeline output guides debugging;
necessary prompt/reference/code/error reading is allowed;
small read-only reference probes are allowed for semantic localization;
only submission implementation should be changed;
rerun the fixed pipeline when the next candidate implementation is ready.
```

The intended benefit is to give the agent enough room to debug real semantic
mistakes while still preventing drift into tool-source archaeology. The risk is
that looser wording can increase unproductive exploration unless the boundaries
are precise.

## Possible Implementation Plan

1. Update legacy `CLAUDE.md` wording.
   - Keep the fixed pipeline and budget rules.
   - Relax the over-narrow "read only error then edit" implication.
   - Add explicit reference-probe permission with boundaries.

2. Extend legacy `tools/triton_eval_pipeline.sh`.
   - Add AST precheck.
   - Emit `judge_out/debug_context.json`.
   - Preserve existing `metrics_error.log` and `metrics.json` compatibility.

3. Port verifier result summarization.
   - Reuse CANNBot verifier logic where possible.
   - Do not expose verifier internals as normal agent reading targets.

4. Add precision and selected operator references.
   - Keep them optional and relevance-gated.
   - Avoid adding mandatory phases.

5. Test on known trajectories.
   - MaxPool default stride case should become easier to fix.
   - Tool/verifier-source drift should remain blocked.
   - Budget exhaustion should still stop cleanly.

## Expected Outcome If Adopted

The legacy runtime should gain most of CANNBot's useful debugging ability while
keeping the RL-friendly short loop:

```text
implement -> fixed pipeline -> structured failure feedback -> focused repair
```

The intended improvement is better semantic localization and correctness repair,
not a larger human-facing workflow.
