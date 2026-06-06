#!/usr/bin/env python3
"""向一个已运行的 Polar rollout server 提交 ONE 真实 claude_code 任务,轮询,dump SessionResult。

这是「真实推理」smoke:Claude Code 在 Polar 的 runtime 容器里跑,它的 Anthropic 调用经 Polar gateway
翻译后打到 vllm-ascend,Polar 捕获一条 token-faithful 轨迹。本脚本驱动它,**不改 Polar 核心**(不碰 run.py)。
它把 TaskStatus(含 SessionResult)dump 成 JSON,之后可:
    python verify_trajectory.py <out>.json --expect-per-request

前置:Polar rollout(:8080)+ gateway 已起(用 topology.qwen35.yaml);vllm-ascend 在服务 qwen35;
agent 镜像已 build(见 Dockerfile / README)。零三方依赖(stdlib urllib)。

    python submit_claude_code.py --rollout-url http://127.0.0.1:8080 --image polar-ascend-agent:latest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid

INSTRUCTION = (
    "Create a file solution.py containing a function add(a, b) that returns a + b. "
    'Then run python3 -c "import solution; print(solution.add(2, 3))" and confirm it prints 5.'
)


def _post(url: str, body: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _get(url: str, timeout: float = 30.0) -> dict:
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def build_request(image: str, backend: str) -> dict:
    # CAPTURE smoke: no evaluator (reward not needed); pre-baked image so prepare just asserts claude.
    return {
        "task_id": f"ascend-smoke-{uuid.uuid4().hex[:8]}",
        "instruction": INSTRUCTION,
        "num_samples": 1,
        "timeout_seconds": 1200.0,
        "runtime": {
            "backend": backend,
            "image": (f"docker-daemon:{image}" if backend == "apptainer" else image),
            "prepare": [
                {"type": "exec", "command": "command -v claude && mkdir -p /polar/session/workspace"},
            ],
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": {"harness": "claude_code", "model_name": "claude-opus-4-5"},  # gateway remaps -> model_served (qwen35)
        "builder": {"strategy": "per_request"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-url", default="http://127.0.0.1:8080")
    ap.add_argument("--image", default="polar-ascend-agent:latest")
    ap.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    ap.add_argument("--out", default="ascend_session.json")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()
    base = args.rollout_url.rstrip("/")

    req = build_request(args.image, args.backend)
    print(f"[submit] {base}/rollout/task/submit  harness=claude_code builder=per_request image={args.image} backend={args.backend}")
    tid = _post(f"{base}/rollout/task/submit", req)["task_id"]
    print(f"[submit] task_id={tid}; polling every {args.poll:.0f}s (Claude Code is now running in a Polar container) ...")

    deadline = time.monotonic() + args.timeout
    status = None
    while time.monotonic() < deadline:
        time.sleep(args.poll)
        st = _get(f"{base}/rollout/task/{tid}")
        s = st.get("status")
        print(f"  [{time.monotonic() - (deadline - args.timeout):6.0f}s] status={s} "
              f"completed={st.get('completed_sessions')}/{st.get('total_sessions')}")
        if str(s).lower() in ("completed", "failed"):
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
    print(f"[result] session={sr.get('session_id')} status={sr.get('status')} traces={len(traces)} error={sr.get('error')}")
    for i, tr in enumerate(traces[:3]):
        lp = tr.get("response_logprobs")
        print(f"  trace[{i}] prompt_ids={len(tr.get('prompt_ids') or [])} response_ids={len(tr.get('response_ids') or [])} "
              f"logprobs={len(lp) if lp is not None else None} loss_mask={len(tr.get('loss_mask') or [])} finish={tr.get('finish_reason')}")

    ok = bool(traces) and all((tr.get("response_ids") and tr.get("prompt_ids")) for tr in traces)
    print(f"\n[{'PASS' if ok else 'CHECK'}] basic capture {'looks good — real inference hit + token-faithful traces captured' if ok else 'empty/missing token ids — inspect ' + args.out}")
    print(f"Next (full contract): python verify_trajectory.py {args.out} --expect-per-request")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
