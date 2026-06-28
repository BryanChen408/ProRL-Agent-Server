# Ascend Operator Deploy

This directory is the Polar-owned deploy entrypoint for Ascend operator rollouts.
It is part of the Polar repo and is safe to move with the repo.

## Layout

- `../../operator_runtime/`: committed Claude Code runtime assets mounted read-only into agent containers.
- `PHASE3_RUNTIME_SOURCE.md`: current runtime source-of-truth and legacy boundary notes.
- `PHASE4_LEGACY_PATHS.md`: non-destructive migration away from old host deploy paths.
- `polar_config.yaml`: Slime bridge config template. Slime should only need Polar rollout URL and rollout scheduling knobs.
- `topology.yaml`: single-host Polar topology template.
- `topology.dual64polar.yaml`: dual-host template where one host runs Polar and another host runs Slime/SGLang.
- `tools/render_run_topology.py`: renders repo-relative topology paths into Docker host absolute paths at launch time.
- `start_polar_nohup.sh`, `restart_polar_host.sh`, `stop_polar.sh`: host service controls.
- `tools/polar_rollout_observer.py`: observer UI.
- `tools/polar_pipeline_budget_watcher.py`: pipeline budget cancellation watcher.

## Runtime Output

Runtime files are written under the repo-local output root and are ignored by git:

```text
output/ascend_operator/
  logs/
  op_assets/
  run_configs/
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

## Start Polar

Single-host default:

```bash
POLAR_GEN_PIPELINE_MAX=3 POLAR_OPT_PIPELINE_MAX=1 bash deploy/ascend_operator/restart_polar_host.sh
```

Dual-host example where this host runs Polar and another host runs SGLang:

```bash
POLAR_TOPOLOGY_TEMPLATE=deploy/ascend_operator/topology.dual64polar.yaml POLAR_ROLLOUT_URL=http://80.48.5.64:8080 POLAR_GATEWAY_URL=http://80.48.5.64:8100 SGLANG_ROUTER_URL=http://80.48.5.88:4077 POLAR_GEN_PIPELINE_MAX=3 POLAR_OPT_PIPELINE_MAX=1 bash deploy/ascend_operator/restart_polar_host.sh
```

The start script renders the active topology to:

```text
output/ascend_operator/run_configs/topology.rendered.yaml
```

That rendered file contains absolute host paths for Docker volume mounts. The committed topology templates do not.

## Slime Contract

Slime should point at the Polar rollout URL and its prompt jsonl. It should not own Polar runtime paths, skills/tools paths, gateway topology, or NPU pools.

Typical Slime-side values:

```bash
POLAR_CONFIG=<polar-repo>/deploy/ascend_operator/polar_config.yaml
OPERATOR_TASK_JSONL=<dataset>/operator_tasks.jsonl
OPERATOR_TASKS_DIR=<dataset>/op_tasks
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
