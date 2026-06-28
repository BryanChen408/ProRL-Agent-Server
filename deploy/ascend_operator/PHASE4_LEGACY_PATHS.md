# Phase 4 Legacy Path Migration

Phase 4 removes active dependence on old host-side deploy directories without
deleting them immediately.

## New Path Contract

Use a repo-shaped Polar deploy root on every host:

```text
<any-parent>/ProRL-Agent-Server/
  operator_runtime/
  deploy/ascend_operator/
  src/polar/
  output/ascend_operator/       # generated at runtime, ignored by git
```

Do not maintain these as source paths for new runs:

- `/home/docker/polar_e2e`
- `/home/docker/op_skills_canonical`
- `/home/docker/polar_debug/Workspace/skills-rl-deploy`

Those paths may remain on disk as historical rollback artifacts until the new
repo-shaped deploy has passed live runs.

## Sync

Preferred sync:

```bash
deploy/ascend_operator/sync_polar_repo.sh root@<polar-host>:/home/docker/ProRL-Agent-Server/
```

The sync script excludes `.git/` and `output/`. It refuses destinations
containing `/polar_e2e` unless `POLAR_SYNC_ALLOW_LEGACY_DEST=1` is set.

Datasets are not Polar deploy assets. Slime owns:

```text
OPERATOR_TASK_JSONL
OPERATOR_TASKS_DIR
```

## Start

On the Polar host, run from the synced repo:

```bash
cd /home/docker/ProRL-Agent-Server
POLAR_TOPOLOGY_TEMPLATE=deploy/ascend_operator/topology.dual64polar.yaml \
POLAR_ROLLOUT_URL=http://<polar-host>:8080 \
POLAR_GATEWAY_URL=http://<polar-host>:8100 \
SGLANG_ROUTER_URL=http://<slime-host>:4077 \
POLAR_GEN_PIPELINE_MAX=3 \
POLAR_OPT_PIPELINE_MAX=1 \
bash deploy/ascend_operator/restart_polar_host.sh
```

## Delete Gate

Do not delete old directories until all are true:

- Polar services start from the repo-shaped deploy root.
- Observer and watcher use `deploy/ascend_operator/` scripts from the repo.
- Runtime topology render shows `operator_runtime` mounted from the repo.
- Slime submits with `OPERATOR_TASK_JSONL` and `OPERATOR_TASKS_DIR`.
- At least one live rollout reaches pipeline and judge using the new deploy.

After that, archive old directories before deletion:

```bash
mv /home/docker/polar_e2e /home/docker/polar_e2e.legacy.$(date +%Y%m%d-%H%M%S)
mv /home/docker/op_skills_canonical /home/docker/op_skills_canonical.legacy.$(date +%Y%m%d-%H%M%S)
mv /home/docker/polar_debug/Workspace/skills-rl-deploy \
  /home/docker/polar_debug/Workspace/skills-rl-deploy.legacy.$(date +%Y%m%d-%H%M%S)
```

Physical deletion is a later manual cleanup step, not part of the safe migration.
