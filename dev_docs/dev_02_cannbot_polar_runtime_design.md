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
