# Polar DockerRuntime Ascend Mainline

This example is the mainline preflight path for connecting slime training to
Polar's native DockerRuntime for operator-generation rollouts.

It intentionally does not use rllm, LocalRuntime, replay/seam demos, W8
faithfulness gates, hard-coded node scripts, or `polar-op-image`. The runtime
image contract is `sandbox:v1`.

## Files

- `polar_config.yaml`: slime `--custom-config-path` template for Polar
  DockerRuntime tasks.
- `topology.yaml`: Polar topology template rendered to the active slime SGLang
  router.
- `preflight.sh`: no-training/no-NPU preflight by default.
- `run_no_npu_checks.sh`: stable no-NPU CI/local review wrapper.
- `check_render_contract.py`: static TaskRequest/topology contract check.
- `gen_op_assets.py`: converts KernelBench-style parquet rows into
  `operator_tasks.jsonl` plus `op_tasks/<op>.py`.
- `tools/polar_rollout_observer.py`: local rollout observer web UI for prompt,
  response, tool, skill, and pipeline inspection.
- `tools/polar_pipeline_budget_watcher.py`: local watcher that cancels sessions
  exceeding the operator eval pipeline budget.
- `tools/probe_gateway_runtime.py`: host-side Polar gateway runtime probe; it
  submits a minimal DockerRuntime session and fails before training if the
  gateway cannot start containers.
- `fixtures/operator_assets/`: tiny offline fixture used by the default
  preflight.

## No-NPU Preflight

Run from the slime repo:

```bash
bash examples/ascend/polar_dockerruntime_e2e/preflight.sh
```

For CI/local review, run the wrapper that also executes the slime-side unit
tests:

```bash
bash examples/ascend/polar_dockerruntime_e2e/run_no_npu_checks.sh
```

By default this checks:

- `bash -n` for the touched shell scripts.
- skills/tool assets are present.
- `prepare_operator_workdir.py` and `gen_op_assets.py` compile.
- SGLang patch dry-run on a temp copy.
- rendered Polar `TaskRequest` validates against Polar schemas.
- topology renders to the slime SGLang router without `/v1`.
- focused Polar tests.

It does not start training, rollout servers, gateway nodes, Docker containers,
cleanup scripts, or NPU image gates.

## Real Operator Assets

The default preflight uses the bundled tiny fixture so the example is
self-contained. To point at real operator assets:

```bash
POLAR_TASKS_DIR=/path/to/op_assets/op_tasks \
POLAR_TASK_JSONL=/path/to/op_assets/operator_tasks.jsonl \
bash examples/ascend/polar_dockerruntime_e2e/preflight.sh
```

To generate those assets from a KernelBench-style parquet:

```bash
python3 examples/ascend/polar_dockerruntime_e2e/gen_op_assets.py \
  --parquet /home/docker/kernelbench_openhands.parquet \
  --out-dir /path/to/op_assets
```

`gen_op_assets.py` rejects unsafe `op_name` values. Names must be basename-like
and match `[A-Za-z0-9][A-Za-z0-9_.-]*`.

## Required Paths

Common overrides:

```bash
POLAR_ROOT=/home/docker/polar_debug/ProRL-Agent-Server
POLAR_SKILLS_DIR=/home/docker/polar_e2e/operator_runtime
SGLANG_ROOT=/workspace/sglang/python/sglang
POLAR_OP_IMAGE=sandbox:v1
POLAR_DEVICE_POOL=0
POLAR_LOCK_DIR=/dev/shm/npu-locks
POLAR_MODEL_SERVED=model-served-placeholder
SGLANG_ROUTER_IP=127.0.0.1
SGLANG_ROUTER_PORT=4077
```

`POLAR_OP_IMAGE` is expected to remain `sandbox:v1` for the mainline contract.

## NPU Gate

The preflight skips the runtime image gate unless explicitly requested:

```bash
POLAR_RUN_IMAGE_GATE=1 \
POLAR_IMAGE_GATE_DEVICE=11 \
bash examples/ascend/polar_dockerruntime_e2e/preflight.sh
```

This invokes the local `check_polar_runtime_image.sh` with `--with-npu`. Do not
run it on a shared node unless the target card is known to be available.

For the staged NPU smoke sequence, use:

```text
dev_docs/dev_15_polar_npu_smoke_runbook.md
```

Do not jump directly from no-NPU preflight to training.

## Legacy Lazy Judge NPU Probe

The mainline resource model is now pipeline-scoped NPU lease, not runtime
lifetime card lease. The old lazy shared-pool probe is still useful as an
isolated fresh-judge DockerRuntime smoke, but it is no longer the primary
resource scheduling check. Run it only from a host shell with Docker access:

```bash
python3 /home/docker/verify_polar_lazy_shared_judge_npu.py \
  --polar-root /home/docker/polar_debug/ProRL-Agent-Server \
  --image sandbox:v1 \
  --pool 8,9 \
  --lock-dir /dev/shm/npu-locks \
  --keep-session
```

This does not start training, the rollout server, or the gateway server. It
directly exercises the Polar lazy path with real DockerRuntime containers:
agent writes a submission, Polar extracts it to a host artifact, the agent
runtime stops, the fresh judge runtime starts from the same NPU pool, the
submission is uploaded into judge, and judge runs a minimal `torch_npu + Triton`
operator smoke.

## Mainline Runtime Contract

The rendered task must keep these properties:

- `runtime.backend: docker`
- `runtime.image: sandbox:v1`
- `runtime.network: host`
- `runtime.kwargs.ascend.pool`, `lock_dir`, and `lease_at_start: false` set
  from arguments.
- `runtime.env.POLAR_NPU_LEASE_POOL` and `POLAR_NPU_LOCK_DIR` set for
  pipeline-scoped verify/benchmark lease.
- readonly operator tools mount:
  `<readonly_tools_dir>:/opt/workspace/agent_workdir/tools:ro`
- `evaluator.runtime.kwargs.ascend.pool`, `lock_dir`, and
  `lease_at_start: false` set from `polar_eval_device_pool`.
- `evaluator.runtime.env.POLAR_NPU_LEASE_POOL` and `POLAR_NPU_LOCK_DIR` set
  for fresh judge pipeline lease.
- skills source mount: `<skills_dir>:/opt/canonical:ro`
- agent harness: `claude_code`
- evaluator: `operator_judge`
- evaluator `refresh_runtime: true`
- evaluator `config.lazy_refresh_runtime: true`
- builder: `prefix_merging`
- builder config renders as a mapping
- sub-agent smoke config does not ban `Agent` or `Workflow`, while plan/task
  bookkeeping tools such as `TaskCreate`/`TaskUpdate`/`TodoWrite` stay banned

The agent prepare step uses:

```bash
python3 /opt/canonical/runtime/prepare_operator_workdir.py --op-name <op> --workdir /opt/workspace/agent_workdir --require-claude --no-stub
```

The rendered command adds `--readonly-tools`; `prepare_operator_workdir.py`
then leaves `workdir/tools` to the Docker read-only bind mount instead of
copying a writable `tools/` directory into the session. It pre-creates
`output/submission/`, but the first implementation file is created by the agent.

The fresh judge runtime uses its own runtime spec and the same helper with
`--no-stub`. The example still enables lazy fresh judging to avoid prewarming
idle judge containers during RUN, but NPU scheduling is handled by
`tools/npu_lease_exec.py` inside `triton_eval_pipeline.sh`. Agent and fresh
judge runtimes may share the same pool, for example `polar_device_pool:
"8,9,10,11"` and `polar_eval_device_pool: "8,9,10,11"`, because neither
runtime holds a card at startup.

`max_run_workers` is logical agent session concurrency, not eval card count.
The default topology sets `max_run_workers: 32` and `max_postrun_workers: 32`;
`POLAR_NPU_LEASE_POOL` controls actual verify/benchmark/judge NPU concurrency.
Pipeline queue time counts against the Polar session timeout and
`operator_judge` timeout. The example keeps `timeout_seconds: 3600` and the
operator judge default cap of 1800 seconds; if `max_run_workers` is increased
substantially, revisit those budgets before real training.

`prefix_merging` is the default for black-box multi-turn/sub-agent agenticRL
smoke. `per_request` remains a debug fallback by overriding
`polar_builder_strategy`, but it should not be used for the sub-agent smoke
path.

Training commands must include the trajectory-aware reward hook:

```bash
--custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards
```

Without that hook, sessions with more traces can affect the GRPO group baseline
more than sessions with fewer traces.

## Before Training

Stay in this order:

1. Run the default no-NPU preflight.
2. Run the preflight against real operator assets.
3. Publish readonly tools into `/home/docker/polar_e2e/readonly_tools` when
   preparing host assets:

```bash
python3 examples/ascend/polar_dockerruntime_e2e/prepare_readonly_tools.py \
  --source /home/docker/polar_e2e/operator_runtime/tools \
  --dest /home/docker/polar_e2e/readonly_tools
```

4. Run the NPU image gate only when a specific card is available.
5. Start Polar rollout/gateway from a host shell that has Docker CLI and daemon
   access, then probe the gateway before occupying training NPUs:

```bash
python3 examples/ascend/polar_dockerruntime_e2e/tools/probe_gateway_runtime.py \
  --gateway-url http://127.0.0.1:28100 \
  --image sandbox:v1 \
  --pool 8,9,10,11 \
  --lock-dir /dev/shm/polar-npu-locks \
  --skills-dir /home/docker/polar_e2e/readonly_tools
```

The slime training container does not need Docker access, but the Polar gateway
process does. A healthy `/health` response is not enough for DockerRuntime E2E:
the probe must reach `status=COMPLETED`.

6. Start slime training with this custom config and rendered topology. For the
   isolated 28080/28100 + SGLang 24077 path, use one of the host wrappers:

```bash
bash /home/docker/polar_e2e/run_smoke_isolated.sh
bash /home/docker/polar_e2e/run_regular_isolated.sh
```

Do not run broad cleanup scripts on shared training hosts. Avoid any script that
can kill unrelated 35B or training processes.

## Rollout Observer

Start the observer on the host or training container that can reach the Polar
gateway:

```bash
python3 examples/ascend/polar_dockerruntime_e2e/tools/polar_rollout_observer.py \
  --root /home/docker/polar_e2e \
  --results-dir /home/docker/polar_e2e/rollout_results \
  --gateway http://127.0.0.1:8100 \
  --host 0.0.0.0 \
  --port 8899
```

Open it through an SSH tunnel if the node is behind a jump host. The detail pane
prefers gateway-memory completions for active sessions, so it can show full
latest responses even when on-disk completion JSON is field-truncated.

The observer focuses on runtime state, prompt/response inspection, skill usage,
actual `tools/triton_eval_pipeline.sh` calls, and precision/profiling feedback.
It does not delete or rewrite historical session data. Use the Run filter,
status tabs, search box, and time-window filter to focus on a specific rollout
attempt while keeping old `rollout_results` available for audit.

The regular launcher renders a run-scoped Polar config from `RUN_ID` before
training starts. For example, `RUN_ID=regular_20260623-120000` produces task ids
like `regular_20260623-120000-polar-op-0-0`, which the observer groups as one
run. Legacy task ids such as `polar-op-0-0` remain visible under `legacy`.

## Pipeline Budget Watcher

The skills workflow tells Claude Code to stop after 6 generation pipeline calls
or 3 optimization pipeline calls. Start the watcher as a hard fallback when
running real rollouts:

```bash
python3 examples/ascend/polar_dockerruntime_e2e/tools/polar_pipeline_budget_watcher.py \
  --root /home/docker/polar_e2e \
  --gateway http://127.0.0.1:8100 \
  --gen-max 6 \
  --opt-max 3 \
  --interval 2 \
  --log-file /home/docker/polar_e2e/logs/pipeline_budget_watcher.log
```

The watcher reads active gateway completions, counts actual
`tools/triton_eval_pipeline.sh` Bash calls, and cancels a session only after it
exceeds the budget. Use `--dry-run --once` to inspect decisions without
cancelling anything.

## Review Boundaries

Recommended review/commit split:

1. Polar repo (`/home/docker/polar_debug/ProRL-Agent-Server`)
   - SGLang patch contract: no silent `token_id=0`, strict token/logprob length
     checks, dry-run/idempotency/fail-fast tests.
   - DockerRuntime/Ascend static contract: host lock, leased
     `ASCEND_RT_VISIBLE_DEVICES`, validated driver mounts, no per-card device
     remap.
   - slime bridge render behavior: topology router override and
     `polar_model_served_name`.

2. slime repo (`/workspace/slime-ascend`)
   - This example directory, bundled offline fixture, preflight, render
     contract checker, and operator asset generator tests.

3. Skills repo (`/home/docker/polar_debug/Workspace`)
   - `skills-rl-deploy/runtime/prepare_operator_workdir.py` and focused tests.

Useful no-NPU verification commands:

```bash
# slime default offline fixture
bash examples/ascend/polar_dockerruntime_e2e/preflight.sh

# slime with real operator assets
POLAR_TASKS_DIR=/home/docker/blackbox/Workspace/polar_slime_ascend/op_assets/op_tasks \
POLAR_TASK_JSONL=/home/docker/blackbox/Workspace/polar_slime_ascend/op_assets/operator_tasks.jsonl \
bash examples/ascend/polar_dockerruntime_e2e/preflight.sh

# slime-only tests
pytest -q -p no:cacheprovider tests/examples/test_polar_render_contract.py tests/examples/test_polar_gen_op_assets.py

# slime no-NPU wrapper
bash examples/ascend/polar_dockerruntime_e2e/run_no_npu_checks.sh

# Polar focused tests
PYTHONPATH=/home/docker/polar_debug/ProRL-Agent-Server/src pytest -q -p no:cacheprovider \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/test_patch_sglang_contract.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/runtime/test_ascend.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/runtime/test_docker_runtime_contract.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/runtime/test_factory.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/trajectory/test_operator_judge.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/trajectory/test_prefix_merging_builder.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/trajectory/test_engine_trajectory_equivalence.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/slime_bridge/test_reward_post_process.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/slime_bridge/test_dockerruntime_mainline_contract.py \
  /home/docker/polar_debug/ProRL-Agent-Server/tests/slime_bridge/test_config.py

# Skills helper tests
pytest -q -p no:cacheprovider /home/docker/polar_debug/Workspace/tests/test_prepare_operator_workdir.py
```
