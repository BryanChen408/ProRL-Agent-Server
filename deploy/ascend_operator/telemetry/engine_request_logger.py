"""vllm engine 侧 per-request 落盘(采集块②/③的 engine 真值)。

gateway 走非流式,拿不到 TTFT / 纯 prefill / 纯 decode / 排队 / prefix-cache / 抢占 —— 这些只在
engine 内部。本模块从 vllm 的 RequestOutput.metrics(RequestMetrics)提取,写一行 JSONL,用
gateway 注入的 `x-polar-trace-id`(= "{session_id}:{turn_seq}")与 completion_metrics.jsonl join。

=== 接入点(二选一,取决于你的 vllm 版本/fork)===
A) OpenAI serving 层:在 OpenAIServingChat/Completion 拿到最终 RequestOutput 后调用:
       from engine_request_logger import PolarEngineLogger
       LOGGER = PolarEngineLogger(engine_id="infer-0")            # 每个 engine 进程一个,填自己的 id
       LOGGER.log(final_output, trace_id=headers.get("x-polar-trace-id"),
                  policy_version=<当前权重版本>)
B) 输出处理循环:在 _process_model_outputs / output_processor 每完成一个 request 时调用 LOGGER.log(...)。

=== trace_id 怎么到 engine ===
gateway 在 httpx 请求头注入 `x-polar-trace-id`;vllm 的 OpenAI server 能读到 request.headers。
若你的路径拿不到 header,退而用请求体透传字段(见 gateway 补丁里的 extra_body 方案),
或用 vllm 的 request_id 再和 gateway 日志二次 join。

字段与 schema 见 SCHEMA。落盘路径默认 $POLAR_ENGINE_METRICS_DIR 或 ./engine_metrics/<engine_id>.jsonl。
写入走后台线程 + 有界队列,满则丢(计数),绝不阻塞推理热路径。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# 落盘字段(engine 真值);与 completion_metrics.jsonl 在 trace_id 上 join。
SCHEMA = [
    "schema_version", "engine_id", "trace_id", "session_id", "turn_seq",
    "request_id", "policy_version", "recorded_at",
    # 长度
    "num_prompt_tokens", "num_cached_tokens", "num_prefill_tokens", "num_generation_tokens",
    "prefix_cache_hit_pct",
    # 时间戳(engine 单调时钟纪元, 秒)
    "arrival_ts", "first_scheduled_ts", "first_token_ts", "finished_ts",
    # 分解耗时(ms)
    "queue_ms", "ttft_ms", "prefill_ms", "decode_ms",
    "decode_tps_engine",
    # 调度/状态
    "num_preemptions", "finish_reason", "aborted",
]


def _getattr_any(obj: Any, *names, default=None):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def extract(request_output: Any, *, engine_id: str, trace_id: str | None,
            policy_version: Any = None) -> dict[str, Any]:
    """从 vllm RequestOutput 抽 T4 engine 侧字段。跨版本防御式取值。"""
    m = getattr(request_output, "metrics", None)  # RequestMetrics
    req_id = getattr(request_output, "request_id", None)

    # 时间戳(RequestMetrics 用 time.monotonic 纪元)
    arrival = _getattr_any(m, "arrival_time")
    first_sched = _getattr_any(m, "first_scheduled_time")
    first_token = _getattr_any(m, "first_token_time")
    finished = _getattr_any(m, "finished_time")
    time_in_queue = _getattr_any(m, "time_in_queue")

    # 长度:prompt / prefix-cache / 生成
    prompt_ids = getattr(request_output, "prompt_token_ids", None)
    n_prompt = len(prompt_ids) if prompt_ids is not None else _getattr_any(m, "num_prompt_tokens")
    n_cached = _getattr_any(request_output, "num_cached_tokens", default=None)
    outs = getattr(request_output, "outputs", None) or []
    n_gen = sum(len(getattr(o, "token_ids", []) or []) for o in outs) if outs else None
    finish_reason = getattr(outs[0], "finish_reason", None) if outs else None
    aborted = bool(getattr(request_output, "finished", True) is False) or (finish_reason == "abort")

    n_prefill = (n_prompt - n_cached) if (n_prompt is not None and n_cached is not None) else n_prompt
    prefix_hit = round(100.0 * n_cached / n_prompt, 2) if (n_prompt and n_cached is not None) else None

    def ms(a, b):
        return round((b - a) * 1000.0, 3) if (a is not None and b is not None) else None

    queue_ms = round(time_in_queue * 1000.0, 3) if time_in_queue is not None else ms(arrival, first_sched)
    ttft_ms = ms(arrival, first_token)
    prefill_ms = ms(first_sched, first_token)
    decode_ms = ms(first_token, finished)
    decode_tps = round(n_gen / (decode_ms / 1000.0), 3) if (n_gen and decode_ms and decode_ms > 0) else None

    sid, turn = (None, None)
    if trace_id and ":" in trace_id:
        sid, _, t = trace_id.partition(":")
        turn = int(t) if t.isdigit() else None

    return {
        "schema_version": SCHEMA_VERSION,
        "engine_id": engine_id,
        "trace_id": trace_id,
        "session_id": sid,
        "turn_seq": turn,
        "request_id": req_id,
        "policy_version": policy_version,
        "recorded_at": _now_iso(),
        "num_prompt_tokens": n_prompt,
        "num_cached_tokens": n_cached,
        "num_prefill_tokens": n_prefill,
        "num_generation_tokens": n_gen,
        "prefix_cache_hit_pct": prefix_hit,
        "arrival_ts": arrival,
        "first_scheduled_ts": first_sched,
        "first_token_ts": first_token,
        "finished_ts": finished,
        "queue_ms": queue_ms,
        "ttft_ms": ttft_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tps_engine": decode_tps,
        "num_preemptions": _getattr_any(m, "num_preemptions", default=None),
        "finish_reason": finish_reason,
        "aborted": aborted,
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class PolarEngineLogger:
    """每 engine 进程一个;后台线程写 JSONL,满队列丢弃(计数),不堵热路径。"""

    def __init__(self, engine_id: str, out_dir: str | None = None, max_queue: int = 8192):
        self.engine_id = engine_id
        base = Path(out_dir or os.environ.get("POLAR_ENGINE_METRICS_DIR", "./engine_metrics"))
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"{engine_id}.jsonl"
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        threading.Thread(target=self._drain, daemon=True).start()

    def log(self, request_output: Any, *, trace_id: str | None = None, policy_version: Any = None):
        try:
            row = extract(request_output, engine_id=self.engine_id,
                          trace_id=trace_id, policy_version=policy_version)
        except Exception:  # noqa: BLE001 — 观测绝不能打断推理
            return
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _drain(self):
        with open(self.path, "a", buffering=1) as f:
            while True:
                row = self._q.get()
                try:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                except Exception:  # noqa: BLE001
                    pass
