# CANNBot Polar Runtime Design

## Goal

Reuse the CANNBot Triton operator workflow with minimal changes, while adding the Polar runtime controls required for RL:

- NPU lease for verifier and benchmark execution.
- Generation and optimization attempt budgets.
- Fresh judge based on trusted verify and benchmark results.
- Direct consumption of Polar operator tasks without task extraction.

The design should keep CANNBot's core workflow and output format intact. Polar adapts to CANNBot artifacts; CANNBot should not be forced into the old Polar `output/submission` pipeline shape.

## Decisions

### Configure Runtime from Profile

The runtime selection and Polar-specific controls should come from the Polar profile YAML instead of scattered shell flags or hard-coded startup parameters.

Expected profile shape:

```yaml
operator_runtime:
  workflow: cannbot

  budget:
    generation_max: 5
    optimization_max: 3

  npu_lease:
    enabled: true
    pool: [8, 9, 10, 11, 12, 13, 14, 15]
    lock_dir: output/npu_locks
```

Legacy runtime can use the same shape:

```yaml
operator_runtime:
  workflow: legacy

  budget:
    generation_max: 5
    optimization_max: 3

  npu_lease:
    enabled: true
    pool: [8, 9, 10, 11, 12, 13, 14, 15]
    lock_dir: output/npu_locks
```

Runtime semantics:

- `workflow`: selects the operator runtime implementation. For Polar CANNBot, this should be `cannbot`.
- `budget`: generation and optimization verify/benchmark attempt limits.
- `npu_lease`: card pool and lock directory for verifier/benchmark execution.

The concrete runtime asset path is derived by Polar from the selected `workflow`
and the committed repository layout. It should not be repeated in the profile.

The runtime layer may translate this profile into environment variables for tool execution:

```text
POLAR_GEN_PIPELINE_MAX=5
POLAR_OPT_PIPELINE_MAX=3
POLAR_NPU_LEASE_POOL=8,9,10,11,12,13,14,15
POLAR_NPU_LOCK_DIR=...
```

The profile is the source of truth. Shell scripts should not duplicate these defaults.

### Remove Extractor Phase

For Polar RL, tasks are already prepared as verifier-ready Python files. The CANNBot task extractor phase should be removed from the Polar workflow instead of being kept with a prompt-level "do not call extractor" instruction.

Expected task input:

```text
input/{op_name}.py
```

The task file must provide:

```python
class Model(...)
def get_init_inputs()
def get_inputs()
```

or, for multi-shape evaluation:

```python
class Model(...)
def get_init_inputs()
def get_input_groups()
```

The separate shape JSONL used by the CANNBot extractor is not required in the Polar path. Multi-shape cases can be embedded directly in `get_input_groups()`.

### Keep CANNBot Artifact Layout

CANNBot may continue to produce its native artifacts, including:

```text
output/generated_code.py
output/optimized_code.py
{workdir}/{op_name}_generated.py
verify_result.json
perf_result.json
report.md
summary.json
session.jsonl
session.md
```

Polar reward must not trust agent-written reports. It should read only trusted verifier and benchmark outputs:

```text
verify_result.json
perf_result.json
```

Reports and session exports can remain for debugging and review.

### Stage Files for Native Verifier Contract

CANNBot `verify.py` and `benchmark.py` expect:

```text
{verify_dir}/{op_name}_torch.py
{verify_dir}/{op_name}_triton_ascend_impl.py
```

Polar should stage files into that contract:

```text
input/{op_name}.py
  -> {verify_dir}/{op_name}_torch.py

CANNBot generated implementation
  -> {verify_dir}/{op_name}_triton_ascend_impl.py
```

If a future task uses a sidecar `{op_name}.json`, copy it into the same `verify_dir`. The preferred Polar dataset format is still a self-contained task `.py`.

### Single-Process Verify

The CANNBot verifier should be made Polar-friendly in one step by removing the parent-child subprocess execution model.

New `verify.py` flow:

```text
parse args
check Polar budget if enabled
acquire NPU lease if enabled
set ASCEND_RT_VISIBLE_DEVICES
run verify_implementations()
write verify_result.json
release lease
```

The core verification logic should remain unchanged. The removed subprocess wrapper means script-level `--timeout` can no longer rely on `Popen.communicate(timeout=...)`; Polar should rely on the outer tool/session timeout unless a simple local timeout is added later.

### Benchmark Lease

`benchmark.py` is already single-process. Add the same NPU lease hook before any NPU execution:

```text
parse args
acquire NPU lease if enabled
set ASCEND_RT_VISIBLE_DEVICES
run benchmark_implementations()
write perf_result.json
release lease
```

The benchmark scoring and aggregation logic should remain unchanged.

### Budget Semantics

Attempt budgets apply to agent-driven verify/benchmark attempts:

- `verify.py` consumes one attempt.
- `benchmark.py` attaches to the current attempt and does not consume another.
- Generation and optimization have separate limits.
- Fresh judge does not pass budget environment variables, so it runs clean verify and benchmark without budget counting.
- When the budget is exhausted, keep the native CANNBot-style single-line error for now, e.g. `Polar pipeline budget exhausted: phase=generation attempt=4>3`. Do not reintroduce the old `triton_eval_pipeline.sh` multi-line hard-stop feedback in the initial CANNBot runtime.

Suggested environment variables:

```text
POLAR_PIPELINE_PHASE=generation|optimization
POLAR_GEN_PIPELINE_MAX=5
POLAR_OPT_PIPELINE_MAX=3
POLAR_NPU_LEASE_POOL=8,9,10,11
POLAR_NPU_LOCK_DIR=/tmp/polar_npu_locks
```

Budget state and `pipeline_budget_status.json` use the session `ARTIFACTS_DIR`
that Polar already injects. No environment variables means native CANNBot
behavior.

`POLAR_PIPELINE_PHASE` only selects which budget counter receives the current
verify attempt. It is not a runtime selector and does not encode any path.

Deferred follow-up:

- Decide whether CANNBot runtime should add stronger agent-facing budget feedback later. Current implementation writes `pipeline_budget_status.json` for watcher/observer and emits only the single-line budget-exhausted error to the agent.

## Polar Workflow

Agent-side Polar CANNBot flow:

```text
read input/{op_name}.py
design
generate output/generated_code.py
stage and run verify.py
if verify passes, run benchmark.py
optimize within budget
write CANNBot native report/session artifacts
```

Fresh judge flow:

```text
select final implementation artifact
stage input/{op_name}.py as {op_name}_torch.py
stage implementation as {op_name}_triton_ascend_impl.py
run verify.py without budget env
run benchmark.py without budget env
read verify_result.json and perf_result.json
compute reward
```

## Non-Goals

Do not do these in the first implementation:

- Do not keep the extractor phase in the Polar workflow.
- Do not rely on prompt wording to tell the agent to skip extractor.
- Do not keep the verifier subprocess wrapper.
- Do not rename `verify.py` or `benchmark.py`.
- Do not force CANNBot to emit `output/submission/{op}_impl.py`.
- Do not require shape JSONL for Polar tasks.
- Do not make agent reports or summaries the reward source.
- Do not replace CANNBot's workflow with the old Polar `triton_eval_pipeline.sh` flow.

## Remaining Step Redundancy Review

This review covers the remaining implementation steps after the workflow asset
import and verifier budget/lease hooks. The goal is to keep the Polar path as a
thin adaptation around the CANNBot workflow, not a second operator pipeline.

### Step 3: Stage Native Verifier Inputs

Planned action:

- Feed Polar tasks into CANNBot's native verifier contract:
  `{verify_dir}/{op_name}_torch.py` and
  `{verify_dir}/{op_name}_triton_ascend_impl.py`.

Decision:

- **Must do, but keep it mechanical.**

Required shape:

- Copy or hardlink `input/{op_name}.py` to `{verify_dir}/{op_name}_torch.py`.
- Copy the candidate implementation selected by the workflow to
  `{verify_dir}/{op_name}_triton_ascend_impl.py`.
- Copy `{op_name}.json` only when the task actually has a sidecar.

Avoid:

- Do not introduce `output/submission/{op}_impl.py` as another required
  submission shape.
- Do not rename `verify.py` or `benchmark.py`.
- Do not create a new verifier wrapper if existing CANNBot scripts can be
  called directly with staged files.
- Do not require shape JSONL for Polar tasks; prefer self-contained
  `get_input_groups()` in the task `.py`.

Rationale:

- CANNBot's verifier scripts already own the trusted validation contract. Polar
  only needs to adapt file names, not invent another contract.

### Step 4: Fresh Judge

Planned action:

- Reuse CANNBot `verify.py` and `benchmark.py` to compute reward from trusted
  result files.

Decision:

- **Must do, but do not fork the validation logic.**

Required shape:

- Select the final implementation artifact from the CANNBot workflow output.
- Stage files using the same Step 3 native verifier contract.
- Run `verify.py` and `benchmark.py` directly.
- Read `verify_result.json` and `perf_result.json` only.

Budget and lease:

- Fresh judge should not pass `POLAR_GEN_PIPELINE_MAX`,
  `POLAR_OPT_PIPELINE_MAX`, or `POLAR_PIPELINE_PHASE`; it must not consume agent
  attempt budget.
- Fresh judge may still use NPU lease env if it shares the same host/card pool,
  because lease is a resource safety mechanism rather than an attempt budget.

Avoid:

- Do not trust `report.md`, `summary.json`, or agent-written analysis as reward
  source.
- Do not add a separate "fresh judge verifier" implementation.
- Do not disable or bypass CANNBot's actual verify/benchmark checks.

Rationale:

- Polar's reward should be derived from scripts we trust, while preserving the
  same validation path the agent already sees.

### Step 5: Profile, Env, and Startup Integration

Planned action:

- Connect the CANNBot runtime selection, budget, and lease settings to the
  Polar profile and runtime environment.

Decision:

- **Simplify before implementing.** The profile should own user-tuned values;
  runtime paths should be derived by Polar.

Required profile fields:

- `operator_runtime.workflow`
- `operator_runtime.budget.generation_max`
- `operator_runtime.budget.optimization_max`
- `operator_runtime.npu_lease.enabled`
- `operator_runtime.npu_lease.pool`
- `operator_runtime.npu_lease.lock_dir`

Derived values:

- Runtime asset path, derived from `workflow` and repository layout.
- Docker mount source paths, materialized by the launcher from repo-relative
  paths.
- Budget state path, provided by the session `ARTIFACTS_DIR`.

Runtime env to inject:

- `POLAR_GEN_PIPELINE_MAX`
- `POLAR_OPT_PIPELINE_MAX`
- `POLAR_PIPELINE_PHASE` only for agent verify calls that need generation vs
  optimization attribution.
- `POLAR_NPU_LEASE_POOL`
- `POLAR_NPU_LOCK_DIR`
- existing session env such as `ARTIFACTS_DIR`, `SESSION_ID`, and `TASK_ID`.

Avoid:

- Do not reintroduce `operator_runtime.root` in the user profile.
- Do not reintroduce `POLAR_OPERATOR_RUNTIME`.
- Do not reintroduce `POLAR_BUDGET_DIR` or `POLAR_ARTIFACTS_DIR`; budget state
  uses `ARTIFACTS_DIR`.
- Do not expose operator runtime paths to Slime. Slime should only know the
  Polar rollout URL plus Slime-owned dataset and RL scheduling inputs.
- Do not keep shell defaults as a second source of truth after the profile owns
  the values.

Rationale:

- The profile should configure behavior; Polar should derive implementation
  details. This prevents the same setting from drifting across YAML, shell env,
  rendered topology, and runtime env.

Implemented notes:

- Default Ascend operator profiles now select `operator_runtime.workflow:
  cannbot`; runtime assets are derived as `operator_runtime/cannbot`.
- Agent sessions receive CANNBot tasks at `input/{op_name}.py`; user task
  prompts only name the operator, the input file, and `./CLAUDE.md`.
- Agent runtime env receives budget and NPU lease env. Fresh judge keeps NPU
  lease env but strips budget env, so it does not consume agent attempts.
- `POLAR_PIPELINE_PHASE` is not fixed in the profile. CANNBot verifier scripts
  infer generation vs optimization from `triton_impl_name`, with the env still
  available as an explicit override if needed.
- Legacy topology/render/preflight paths are intentionally left in place until
  the Step 7 live-run gates pass.

### Step 6: Observer, Watcher, and Artifacts

Planned action:

- Keep observer/watcher useful for Polar sessions under the CANNBot workflow.

Decision:

- **Keep only runtime-agnostic observability.**

Required shape:

- Watcher reads `pipeline_budget_status.json` from session artifacts, regardless
  of legacy or CANNBot workflow.
- Observer displays CANNBot artifacts and tool calls without assuming
  `output/submission`.
- Run artifacts stay under the run-scoped output directory.

Avoid:

- Do not make observer or watcher parse agent-written reports for reward state.
- Do not add CANNBot-specific duplicate budget state.
- Do not mix historical run artifacts into the active run view.

Rationale:

- Watcher and observer should describe actual session state, not become another
  source of pipeline policy.

Implemented notes:

- Watcher cancellation remains driven by `pipeline_budget_status.json`; legacy
  completion parsing is only a fallback diagnostic and now recognizes CANNBot
  `verify.py` attempts.
- Observer keeps the legacy `pipeline_runs` API field for compatibility, but
  the UI labels it as validation attempts. CANNBot `verify.py` is budget-counted;
  CANNBot `benchmark.py` contributes profiling status without consuming another
  attempt.
- Observer lists CANNBot trusted artifacts from the session directory, including
  `verify_result*.json`, `perf_result*.json`, `generated_code.py`, and
  `optimized_code.py`. This is display-only; reward still comes from fresh judge
  artifacts.

### Step 7: Cleanup Gates

Planned action:

- Remove legacy render/config paths only after the new flow is stable.

Decision:

- **Defer deletion. Add gates first.**

Deletion gates:

- One live session reaches verify and benchmark through CANNBot scripts.
- Fresh judge computes reward from CANNBot verify/benchmark outputs.
- Budget limits are visible in watcher status and inside runtime env.
- NPU lease status is written and cards are released after verify/benchmark.
- Slime launch uses only Polar rollout URL plus Slime-owned dataset/schedule
  config.

Avoid:

- Do not delete legacy runtime or old launcher paths in the same commit as
  CANNBot integration.
- Do not remove audit artifacts until live-run debugging no longer needs them.

Rationale:

- The current system is still being used for training runs. Cleanup should only
  follow after behavior is proven, not before.
