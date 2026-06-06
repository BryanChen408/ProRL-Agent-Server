#!/usr/bin/env python3
"""Submit ONE operator-gen rollout to Polar — the real-machine eval+reward smoke.

Wires everything from this turn together against the conventions locked in
CANNBOT_SKILLS_MIGRATION_GUIDE.md (§B):

  workdir   /opt/workspace/agent_workdir          (fixed; no timestamped dirs)
  input     src/{op}.py (+ .json)                  (framework-placed via prepare)
  submit    output/submission/{op}_impl.py         (class ModelNew; the ONLY artifact)
  eval      tools/triton_eval_pipeline.sh          (fixed entry -> judge_out/metrics.json)
  reward    operator_judge evaluator re-runs it in a CLEAN judge runtime (anti-cheat)

Placement (agent works on WRITABLE copies — no read-only write errors / wasted RL steps):
  - The host skills dir is bind-mounted ``:ro`` ONLY as an immutable SOURCE at /opt/canonical
    (never the agent's working tree).
  - Each container cp's it into its OWN writable {workdir}/tools, so the agent runs + iterates with
    zero permission friction. Anti-cheat lives at the JUDGE: it runs in a SEPARATE clean container
    with a FRESH copy from the untouched source (= your _snapshot_canonical), so tampering an agent
    copy can't move the reward.
  - skills_path=/opt/canonical/skills; the claude_code preset copies it into CLAUDE_CONFIG_DIR/skills
    (also a writable copy). Nothing is baked into the image.

Ascend cards: runtime.kwargs.ascend -> the proven passthrough recipe (polar.runtime.ascend) is applied
to BOTH the agent (in-loop op runs) and the fresh judge runtime (authoritative reward); per-op card
pick is the in-container flock.

    # validate the request shape locally (needs polar+pydantic, no server/NPU):
    python examples/ascend/submit_operator.py --op-name add --dry-run
    # real rollout on the node:
    python examples/ascend/submit_operator.py --op-name add --image polar-op-agent:latest \
      --device-ids 8,9,10,11 --lock-dir /shared/npu-locks \
      --skills-dir /data/cannbot-skills --tasks-dir /data/op-tasks
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid

WORKDIR = "/opt/workspace/agent_workdir"


def _instruction(op: str) -> str:
    return (
        f"Implement a Triton operator for Ascend NPU. The reference task is at src/{op}.py. "
        f"Write your implementation as class ModelNew to output/submission/{op}_impl.py. "
        f"Use the triton-op-verifier skill (it runs tools/triton_eval_pipeline.sh) to test and iterate "
        f"until correctness passes, then optimize for speed. Do NOT edit anything under tools/. "
        f"Your kernel is scored by re-running the canonical pipeline in a clean environment."
    )


def build_operator_request(
    *, op_name: str, image: str, backend: str, device_ids: str, lock_dir: str,
    skills_dir: str, tasks_dir: str, model_name: str, task_json: bool,
) -> dict:
    sub = f"output/submission/{op_name}_impl.py"
    task_src = f"src/{op_name}.py"
    judge_command = (
        f"bash tools/triton_eval_pipeline.sh --op_name {op_name} "
        f"--impl {sub} --task {task_src} --out_dir judge_out"
    )
    # task input placed in BOTH runtimes (agent works on it; judge re-evals against it).
    place_task: list[dict] = [
        {"type": "upload_file", "source": f"{tasks_dir}/{op_name}.py", "target": f"{WORKDIR}/{task_src}"},
    ]
    if task_json:
        place_task.append(
            {"type": "upload_file", "source": f"{tasks_dir}/{op_name}.json", "target": f"{WORKDIR}/src/{op_name}.json"}
        )
    mk = f"mkdir -p {WORKDIR}/output/submission {WORKDIR}/judge_out"
    cp_tools = f"cp -r /opt/canonical/tools {WORKDIR}/tools"  # writable copy per container
    # orchestrator into the agent cwd so Claude Code reads it (skills_dir root has AGENTS.md)
    cp_agents = f"cp /opt/canonical/AGENTS.md {WORKDIR}/AGENTS.md"

    return {
        "task_id": f"op-{op_name}-{uuid.uuid4().hex[:8]}",
        "instruction": _instruction(op_name),
        "num_samples": 1,
        "timeout_seconds": 3600.0,
        "runtime": {
            "backend": backend,
            "image": (f"docker-daemon:{image}" if backend == "apptainer" else image),
            "network": "host",
            "workdir": WORKDIR,
            "kwargs": {
                # Ascend passthrough recipe (polar.runtime.ascend) — applied to agent AND fresh judge.
                "ascend": {"device_ids": device_ids, "lock_dir": lock_dir},
                # Immutable SOURCE only (read-only); never the agent's working tree. Each container cp's
                # tools into its OWN writable {workdir}/tools, so the agent never hits a read-only write
                # error (= no wasted RL steps). Anti-cheat is enforced by the JUDGE running in a SEPARATE
                # clean container with a FRESH copy from this untouched source (= your _snapshot_canonical).
                "volumes": [f"{skills_dir}:/opt/canonical:ro"],
            },
            # agent: writable tools copy (run + iterate freely, zero permission friction).
            "prepare": [*place_task, {"type": "exec", "command": f"{mk} && {cp_tools} && {cp_agents} && command -v claude"}],
            # judge (clean container): FRESH canonical tools from the untouched source -> authoritative.
            "eval_prepare": [*place_task, {"type": "exec", "command": f"{mk} && {cp_tools}"}],
        },
        # skills_path is a read-only SOURCE; the claude_code preset cp's it into CLAUDE_CONFIG_DIR/skills
        # (already a writable copy), so skills are no read-only hazard either.
        "agent": {"harness": "claude_code", "model_name": model_name, "skills_path": "/opt/canonical/skills"},
        "evaluator": {
            "strategy": "operator_judge",
            "refresh_runtime": True,  # fresh judge runtime (anti-cheat); inherits kwargs.ascend
            "config": {
                "op_name": op_name,
                "judge_command": judge_command,
                "submission_path": sub,
                "metrics_path": "judge_out/metrics.json",
                "workdir": WORKDIR,
            },
        },
        "builder": {"strategy": "per_request"},
    }


def _post(url: str, body: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _get(url: str, timeout: float = 30.0) -> dict:
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op-name", required=True)
    ap.add_argument("--image", default="polar-op-agent:latest")
    ap.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    ap.add_argument("--device-ids", default="8,9,10,11", help="NPU verification pool")
    ap.add_argument("--lock-dir", default="/shared/npu-locks")
    ap.add_argument("--skills-dir", default="/data/cannbot-skills", help="host dir with skills/ + tools/")
    ap.add_argument("--tasks-dir", default="/data/op-tasks", help="host dir with {op}.py (+ {op}.json)")
    ap.add_argument("--task-json", action="store_true", help="also upload {op}.json")
    ap.add_argument("--model-name", default="claude-opus-4-5", help="gateway remaps -> model_served")
    ap.add_argument("--rollout-url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="operator_session.json")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--timeout", type=float, default=4000.0)
    ap.add_argument("--dry-run", action="store_true", help="build + validate request shape locally, print, exit")
    args = ap.parse_args()

    req = build_operator_request(
        op_name=args.op_name, image=args.image, backend=args.backend,
        device_ids=args.device_ids, lock_dir=args.lock_dir,
        skills_dir=args.skills_dir, tasks_dir=args.tasks_dir,
        model_name=args.model_name, task_json=args.task_json,
    )

    if args.dry_run:
        print(json.dumps(req, indent=2, ensure_ascii=False))
        try:
            sys.path.insert(0, "src")
            from polar.rollout.models import TaskRequest
            TaskRequest(**req)
            print("\n[OK] request validates against polar.rollout.models.TaskRequest")
        except ModuleNotFoundError:
            print("\n[skip] polar not importable here — shape printed above (run on the node to validate)")
        except Exception as e:  # noqa: BLE001
            print(f"\n[FAIL] request does NOT validate: {type(e).__name__}: {e}")
            return 1
        return 0

    base = args.rollout_url.rstrip("/")
    print(f"[submit] {base}/rollout/task/submit op={args.op_name} evaluator=operator_judge image={args.image}")
    tid = _post(f"{base}/rollout/task/submit", req)["task_id"]
    print(f"[submit] task_id={tid}; polling every {args.poll:.0f}s (Claude Code writes + iterates the kernel) ...")

    deadline = time.monotonic() + args.timeout
    status = None
    while time.monotonic() < deadline:
        time.sleep(args.poll)
        st = _get(f"{base}/rollout/task/{tid}")
        s = str(st.get("status"))
        print(f"  [{time.monotonic() - (deadline - args.timeout):6.0f}s] status={s} "
              f"completed={st.get('completed_sessions')}/{st.get('total_sessions')}")
        if s.lower() in ("completed", "failed"):
            status = st
            break
    if status is None:
        print("[FAIL] polling timed out")
        return 1

    with open(args.out, "w") as f:
        json.dump(status, f, indent=2)
    print(f"[dump] full TaskStatus -> {args.out}")

    results = status.get("results") or []
    if not results:
        print(f"[FAIL] task {status.get('status')} with no SessionResult — inspect {args.out}")
        return 1
    sr = results[0]
    traces = (sr.get("trajectory") or {}).get("traces") or []
    rewards = [t.get("reward") for t in traces]
    meta = (sr.get("trajectory") or {}).get("metadata") or {}
    print(f"[result] session={sr.get('session_id')} status={sr.get('status')} traces={len(traces)} "
          f"reward={rewards[:1]} error={sr.get('error')}")
    print(f"[eval]   {json.dumps({k: meta.get(k) for k in ('mode','op_name','reward','error_type','success','speedup_vs_torch') if k in meta}, ensure_ascii=False)}")
    # reward present + a real trace = the full agent->judge->reward chain fired on real hardware.
    ok = bool(traces) and rewards and rewards[0] is not None and str(sr.get("status")).upper() == "COMPLETED"
    print(f"\n[{'PASS' if ok else 'CHECK'}] {'operator-gen reward chain fired (real inference + real judge on NPU)' if ok else 'no reward / not COMPLETED — inspect ' + args.out + ' (ERROR status => infra retry path)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
