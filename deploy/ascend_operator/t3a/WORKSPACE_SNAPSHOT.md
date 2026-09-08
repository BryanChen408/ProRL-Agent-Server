# T3A workspace snapshot (2026-09-08)

This snapshot preserves the current implementation before redesign. It is not
an end-to-end acceptance or a claim that inference failures have been fixed.

## Preserved scope

- Runtime replica and build script, profile, task-prompt converter and prepare.
- Main/sub trace metadata and optional T3A attempt-span extraction.
- Hook candidate handoff to the independent judge and process-reward retrieval.
- Existing T2A deployment tuning is committed separately for independent rollback.

## Verification

CPU regressions cover prepare, task prompts, attempt spans, lazy judge creation
and candidate transfer. They do not run NPU inference or verify kernel accuracy.
The runtime replication ledger is a historical build report, not a fresh audit
of every file in this snapshot. The replica build script was not rerun.

## Open issues / boundaries

- Latest sampled first-turn token output includes incoherent text before any
  tool execution. Input adaptation versus inference execution is not isolated.
- The latest run's terminal disconnects followed the user's Ctrl+C; do not
  diagnose those disconnects as the original generation failure.
- `_lease_wrap_command` concatenates compound shell commands after `--`;
  `cd ... && python3 verification_ascendc.py ...` is not wrapped correctly.
- `judge_best.sh` accepts the first qualifying ranked candidate; it does not
  evaluate every candidate and maximize the final judge reward.
- The selected current cannbot replica is not yet version-aligned with all
  historical successful trajectories in `/home/docker/01_passed`.
- No service restart, deployment to node 56, or live training acceptance was
  performed as part of workspace organization.

The intended next direction is to preserve stable RL infrastructure while
version-aligning the cannbot solving workflow, not to restore T2A solving rules
inside the cannbot agent.
