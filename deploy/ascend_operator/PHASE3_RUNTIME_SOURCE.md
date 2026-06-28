# Phase 3 Runtime Source Status

This deploy stack no longer treats `/home/docker/polar_debug/Workspace/skills-rl-deploy`
as an active runtime source.

Current source-of-truth:

- `operator_runtime/CLAUDE.md`
- `operator_runtime/skills/`
- `operator_runtime/tools/`
- `operator_runtime/runtime/prepare_operator_workdir.py`
- `operator_runtime/.agents/skills/triton-op-verifier/scripts/`

Current runtime mount:

- `operator_runtime:/opt/canonical:ro`
- `operator_runtime/tools:/opt/workspace/agent_workdir/tools:ro`

Generated and runtime-only files live under:

- `output/ascend_operator/`

The legacy `.agents/skills/triton-op-verifier/scripts` layout is still used by
`operator_runtime/tools/triton_eval_pipeline.sh` and is intentionally kept for
now. Removing that layout is a separate verifier-layout cleanup, not part of
this phase.

Compatibility notes:

- `scripts/publish_operator_runtime.py` still exists as a manifest/publish
  helper. Its default source is `operator_runtime/`.
- Historical dev docs may mention `Workspace/skills-rl-deploy` or
  `/home/docker/polar_e2e`; those are historical records unless an active
  script/config references them.
