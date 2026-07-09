# Ascend Operator Deploy

This directory is the Polar-owned deploy entrypoint for Ascend operator rollouts.
It is part of the Polar repo and is safe to move with the repo.

## Layout

- `../../operator_runtime/`: committed Claude Code runtime assets mounted read-only into agent containers.
- `PHASE3_RUNTIME_SOURCE.md`: current runtime source-of-truth and legacy boundary notes.
- `PHASE4_LEGACY_PATHS.md`: non-destructive migration away from old host deploy paths.
- `profile.yaml`: Polar host run profile. This is the user-edited config source for rollout/gateway/runtime/observer/watcher.
- `polar_config.yaml`: legacy Slime bridge config template.
- `topology.yaml`, `topology.dual64polar.yaml`: legacy topology templates kept during migration.
- `start_polar_nohup.sh`, `restart_polar_host.sh`, `stop_polar.sh`: host service controls.
- `tools/polar_rollout_observer.py`: observer UI.
- `tools/polar_pipeline_budget_watcher.py`: pipeline budget cancellation watcher.

## Runtime Output

Runtime files are written under the repo-local output root and are ignored by git:

```text
output/ascend_operator/
  op_assets/
  runs/<polar-run-id>/
    logs/
    run_artifacts/
    rollout_results/
    polar_sessions/
    hostctl/
```

Set `POLAR_OUTPUT_DIR=/path/to/output` only when you intentionally want outputs outside the repo.

## Preflight

From the Polar repo root:

```bash
bash deploy/ascend_operator/preflight.sh
bash deploy/ascend_operator/run_no_npu_checks.sh
```

The default preflight uses the bundled tiny fixture. For real operator assets:

```bash
python3 deploy/ascend_operator/gen_op_assets.py   --parquet /path/to/kernelbench.parquet

OPERATOR_TASKS_DIR=output/ascend_operator/op_assets/op_tasks OPERATOR_TASK_JSONL=output/ascend_operator/op_assets/operator_tasks.jsonl bash deploy/ascend_operator/preflight.sh
```

For the legacy skills profile, emit or refresh the original strict prompt contract:

```bash
python3 deploy/ascend_operator/gen_op_assets.py --workflow legacy --parquet /path/to/kernelbench.parquet
python3 deploy/ascend_operator/tools/refresh_operator_task_prompts.py --workflow legacy <dataset>/operator_tasks.jsonl
```

## Start Polar

Single-host default:

```bash
bash deploy/ascend_operator/restart_polar_host.sh
```

Use an explicit run id when you want stable, user-named artifacts:

```bash
POLAR_RUN_ID=debug-$(date +%Y%m%d-%H%M%S) bash deploy/ascend_operator/restart_polar_host.sh
```

If `POLAR_RUN_ID` is not set, the launcher creates a timestamp run id automatically.
Each run writes isolated artifacts under `output/ascend_operator/runs/<run-id>/`,
so observer and watcher only see the current run.

Use another profile when needed:

```bash
POLAR_PROFILE=/path/to/profile.yaml bash deploy/ascend_operator/restart_polar_host.sh
```

Use `deploy/ascend_operator/profile.legacy.yaml` to run the old skills runtime.

The launcher materializes the profile to:

```text
output/ascend_operator/runs/<run-id>/run_artifacts/effective_topology.yaml
```

That file is a run artifact for the existing `serve_rollout -c` and
`serve_gateway -c` entrypoints. Do not edit it by hand; edit `profile.yaml`.

## Slime Contract

Slime should point at the Polar rollout URL and its prompt jsonl. It should not own Polar runtime paths, skills/tools paths, gateway topology, or NPU pools.

Typical Slime-side values:

```bash
OPERATOR_TASK_JSONL=<dataset>/operator_tasks.jsonl
POLAR_ROLLOUT_URL=http://<polar-host>:8080
```

## Multi-Host Sync

When Polar runs on another host, sync the Polar repo to that host at any path. Do not sync a separate `polar_e2e` tree.

```bash
deploy/ascend_operator/sync_polar_repo.sh root@80.48.5.64:/path/to/ProRL-Agent-Server/
```

The script excludes `output/`; runtime output is generated on the target host.
Operator datasets are Slime-owned and should be provided through
`OPERATOR_TASK_JSONL` and `OPERATOR_TASKS_DIR`.
