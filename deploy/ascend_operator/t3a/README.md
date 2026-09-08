# T3A rollout wiring

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

The installed developer agent and all dispatch/reentry references use
`tilelang2ascendc-kernel-generator`. The replica build applies this installation adaptation;
the seven workflow phases are unchanged.

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
