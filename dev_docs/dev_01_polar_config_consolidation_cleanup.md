# Polar Config Consolidation Cleanup

Date: 2026-06-30

## Decision

The desired deploy contract is one Polar-owned run profile as the only user-edited
source of truth for a Polar host run.

Slime should not know Polar runtime paths, skills/tools paths, gateway topology,
agent settings, evaluator settings, NPU pools, observer settings, or watcher
settings. Slime should only know the Polar rollout URL plus its own dataset and
RL scheduling inputs.

## Render Reality Check

Path materialization is needed somewhere because Docker bind mount sources are
host paths. A repo-portable config can say `operator_runtime`, but Docker must
receive a host path such as:

```text
/path/to/ProRL-Agent-Server/operator_runtime:/opt/canonical:ro
```

That does not mean we need a user-visible rendered YAML as another config source.
`output/ascend_operator/run_configs/topology.rendered.yaml` can be replaced by
in-process materialization or kept only as an audit artifact.

Current render responsibilities in
`deploy/ascend_operator/tools/render_run_topology.py`:

- convert repo-relative runtime/tool/op-asset/result paths to host paths;
- apply rollout/gateway public URLs and bind host;
- apply the SGLang router URL.

Only the first item is a hard runtime requirement for Docker volumes. The URL
and router overrides should come directly from the single run profile.

## Current Split-Brain Sources

These are the active config sources today:

- `deploy/ascend_operator/topology.yaml`
  - main Polar service topology template;
  - owns rollout, gateway, operator profile, runtime, agent, evaluator, builder.
- `deploy/ascend_operator/topology.dual64polar.yaml`
  - duplicate topology template for a specific historical dual-host layout.
- `deploy/ascend_operator/_paths.sh`
  - shell defaults for output, logs, sessions, op assets, and run config dirs.
- `deploy/ascend_operator/start_polar_nohup.sh`
  - reads shell env, renders topology, starts rollout/gateway, then starts
    watcher and observer.
- `deploy/ascend_operator/start_observer.sh`
  - reads `_paths.sh` and shell env, not topology/profile.
- `deploy/ascend_operator/start_pipeline_budget_watcher.sh`
  - reads `_paths.sh` and shell env, not topology/profile.
- `deploy/ascend_operator/polar_config.yaml`
  - Slime bridge config template, not Polar service config.
- `deploy/ascend_operator/tools/render_run_config.py`
  - legacy Polar-side bridge-config renderer.
- `/workspace/slime-ascend/tools/polar/render_slime_polar_config.py`
  - current Slime-side bridge-config renderer.

This is why budget limits, URLs, paths, and generated bridge configs can drift.

## Cleanup Inventory

### Keep, But Fold Into One Profile

- `deploy/ascend_operator/topology.yaml`
  - Keep the schema/content, but make it part of the single run profile or make
    it the profile itself.
  - The profile should also include sidecar settings currently outside topology:
    observer host/port, watcher gen/opt limits, watcher interval, output/log
    roots, and router URL.

- `deploy/ascend_operator/_paths.sh`
  - Keep only as a small repo-root discovery helper if needed.
  - Delete path policy from it after the profile owns output/log/session dirs.

- `deploy/ascend_operator/start_polar_nohup.sh`
  - Keep as the process launcher.
  - Change it to read one profile and derive rollout, gateway, observer, watcher,
    and runtime materialization from that profile.

### Generated Or Audit-Only

- `output/ascend_operator/run_configs/topology.rendered.yaml`
  - Not a user config.
  - Keep temporarily as an audit artifact while testing.
  - Stable deletion condition: services can start from a loaded in-memory
    materialized topology, and logs expose the materialized runtime paths/URLs.

- `output/polar_bridge/run_configs/polar_config.<run>.yaml` in Slime
  - Not a Polar config.
  - Keep temporarily as a Slime run artifact.
  - Stable deletion condition: Slime can pass the small bridge values directly
    or load one Slime-owned bridge YAML that does not contain Polar runtime
    details.

### Rename Or Move

- `deploy/ascend_operator/polar_config.yaml`
  - Misnamed: this is Slime bridge config, not Polar service config.
  - Rename target: `deploy/ascend_operator/slime_bridge.yaml`, or move the
    template to Slime as `configs/polar_bridge/operator_session_pool.yaml`.
  - It must not grow Polar runtime fields.

### Remove After Stability

- `deploy/ascend_operator/topology.dual64polar.yaml`
  - Remove as a separate committed topology once the single profile supports
    host-specific URL/router overrides cleanly.
  - Stable deletion condition: the same profile schema can express both local
    and dual-host runs without duplicating operator runtime blocks.

- `deploy/ascend_operator/tools/render_run_config.py`
  - Remove from Polar once Slime owns bridge rendering and Polar no longer
    produces Slime bridge configs.
  - Stable deletion condition: no active launcher, preflight, test, or doc uses
    the Polar-side bridge renderer.

- `deploy/ascend_operator/check_render_contract.py`
  - Replace with a profile contract check.
  - Stable deletion condition: new preflight validates the single profile,
    materialized Docker volumes, runtime env, watcher/observer config, and
    operator task injection.

- `deploy/ascend_operator/preflight.sh` references to `polar_config.yaml`
  - Update to validate the Polar profile and, separately, the Slime bridge
    contract.

### Sidecar Config Debt

- `deploy/ascend_operator/start_observer.sh`
  - Current problem: root/results/gateway/host/port come from shell env.
  - Target: read the same run profile, or receive generated arguments from
    `restart_polar_host.sh` after it loads the profile.

- `deploy/ascend_operator/start_pipeline_budget_watcher.sh`
  - Current problem: gen/opt limits are host env only, while the agent/evaluator
    pipeline sees Docker runtime env from topology.
  - Target: single profile owns `pipeline_budget.generation_max` and
    `pipeline_budget.optimization_max`; launcher injects the same values into:
    watcher args, runtime env, and evaluator runtime env.

## Proposed Single Profile Shape

Example shape only; field names can be adjusted before implementation.

```yaml
service:
  bind_host: 0.0.0.0
  rollout_url: http://80.48.5.88:8080
  gateway_url: http://80.48.5.88:8100
  sglang_router_url: http://80.48.5.64:4077

paths:
  output_dir: output/ascend_operator
  operator_runtime_dir: operator_runtime
  rollout_results_dir: output/ascend_operator/rollout_results
  session_base_dir: output/ascend_operator/polar_sessions
  log_dir: output/ascend_operator/logs

pipeline_budget:
  generation_max: 3
  optimization_max: 1
  interval_seconds: 2

observer:
  host: 0.0.0.0
  port: 18088

topology:
  rollout: ...
  gateway: ...
```

The launcher should resolve relative paths against the Polar repo root at
startup. The committed profile remains movable with the repo.

## Delete Gate

Do not delete old config/render paths until these checks pass on a live run:

- rollout/gateway start from one profile;
- observer and watcher show settings derived from the same profile;
- watcher gen/opt limits match the values visible inside agent/evaluator
  runtime commands;
- one session reaches pipeline and judge using the profile-derived runtime;
- Slime submit uses only Polar rollout URL plus Slime-owned dataset paths;
- generated files under `output/` are treated as artifacts, not edited inputs.
