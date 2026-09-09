# T3A rollout wiring

## Migration gate (2026-09-08)

The committed runtime is a rollback snapshot, **not** a verified reproduction
of the historical successful workflow. Keep the T2A service, scheduling, leases,
judge criteria and training boundary unchanged during migration. Do not restore
T2A solving rules inside the cannbot agent or enable training on empty main chains.

`baseline_audit.json` records the current evidence: 748 saved developer bodies,
129 body variants, zero exact matches against the pinned current agent. These
are file counts (exports can overlap), not independent success-rate measurements.
An exact text mismatch alone does not prove a semantic difference; representative
outlines additionally show older design-document/template-generation skills in
the successful corpus versus the current design/development/review route.

The older complete skill/script source is still needed to pin the workflow.
Do not assemble missing historical skills from similarly named modern ones.
The existing git history's initial `ascendc-ops-lab-developer` already documents
the newer route; the historical benchmark runner was not located there.

Reproduce the read-only audit (exit 2 means unverified, not a crash):

```bash
python3 deploy/ascend_operator/t3a/audit_workflow_baseline.py \
  --samples /home/docker/01_passed --repo /home/docker/cannbot-skills \
  --ref 13b2ae5652c75fe83a3e4114552a6477d3a01f3d \
  --agent-path plugins-community/tilelang2ascendc-ops-generator/agents/tilelang2ascendc-kernel-generator.md
```

Before the next stage, provide the old source/installation archive or explicitly
select a different documented baseline. Then validate fixed-checkpoint generation
and one full operator session before changing deployment defaults. Port 8011 was
serving a newer training run during this work; no replay requests or restarts
were sent to that live model.

Independent adapter fix: direct verification compound commands now execute as
`lease -- bash -c <quoted original command>`; a configured but missing executor
fails instead of silently running without a lease. CPU build/validation commands
remain unwrapped, and the self-wrapped AscendC pipeline is not double wrapped.

## Existing snapshot usage

Keep the original T2A dataset unchanged. Derive T3A instructions once:

```bash
python3 deploy/ascend_operator/t3a/convert_task_prompts.py \
  /home/docker/datasets/ascendc-rl-datasets/NPUKernelBench/operator_tasks.npukernelbench.jsonl \
  /home/docker/datasets/ascendc-rl-datasets/NPUKernelBench/operator_tasks.npukernelbench.t3a.jsonl
```

The converter refuses to overwrite an existing dataset. All non-prompt fields remain unchanged.
Use `profile.t3a.yaml` for Polar, and the derived JSONL as VIME's `OPERATOR_TASK_JSONL`.
`/workspace/vime/scripts/start_sync_hybrid_t3a.sh` wraps the existing local hybrid launcher
with those dataset settings and main-chain filtering; it does not change the resource layout.
For another launcher, set the dataset path there explicitly. Starting services is a separate action.

The snapshot's installed developer agent and all dispatch/reentry references use
`tilelang2ascendc-kernel-generator`. The replica build applies this installation adaptation;
the phases are preserved relative to that vendored source, not yet proven equal
to the historical success corpus.

Hook snapshots remain in agent session artifacts after the agent stops. The gateway recognizes
the candidate index without requiring a legacy submission tarball. The evaluator verifies each
snapshot hash, uploads only candidate files and the attempt stream, and uploads a relocated index
whose paths refer to the fresh judge container. `process_reward.json` is downloaded alongside
metrics; a failed download must not reuse a previous judge attempt's reward.

Keep `POLAR_T3A_ATTEMPT_SPANS=0` in the gateway launch environment for initial acceptance.
This disables the separate attempt-token parser, not the judge's final scalar reward.
Main-chain filtering remains enabled in the T3A VIME wrapper.

CPU regression:

```bash
PYTHONPATH=src python3 -m pytest tests/examples/test_t3a_task_prompts.py \
  tests/examples/test_t3a_prepare_tools.py tests/trajectory/test_operator_judge.py \
  tests/gateway/test_lazy_eval_runtime.py
```

Before long training, verify one real session: registered developer dispatch succeeds, original
skills initialize and evaluate the project, judge consumes the same candidate hash, and VIME has
nonzero subagent loss tokens. CPU tests do not establish NPU correctness or resolve inference NaNs.
