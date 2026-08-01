#!/usr/bin/env python3
"""NPU-SMI → Prometheus exporter(采集块①:显存/内存 + NPU 利用率,按 12 推理 / 4 验证两池分标签)。

设计:
  - 后台线程每 --interval 秒对 card_topology.yaml 里的每张卡跑 `npu-smi info -t common/-t usages`,
    解析 key:value(比 dashboard 表稳),缓存快照;/metrics 处理器只吐缓存,scrape 不阻塞在 npu-smi 上。
  - 标签:card_id / pool{inference|verify} / engine_id / tp_rank。→ 池间、engine 间、engine 内 4 卡不均都能看。
  - 纯标准库,无三方依赖。

用法:
  python3 npu_smi_exporter.py --topology card_topology.yaml --port 9800 --interval 5
  # Prometheus scrape_config: targets: ['<host>:9800']
  python3 npu_smi_exporter.py --once            # 打印一次采集结果(校准/调试)
  python3 npu_smi_exporter.py --raw --card 0    # 打印某卡 npu-smi 原始输出(对字段名)
"""
from __future__ import annotations

import argparse
import http.server
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# --- key 别名:不同 CANN 版本字段名有差异,尽量都覆盖 ---
_ALIASES = {
    "aicore_util_pct": ["Aicore Usage Rate(%)", "AI Core Usage Rate(%)", "AICore Usage Rate(%)"],
    "npu_util_pct":    ["NPU Real-time Utilization(%)", "NPU Utilization(%)", "Chip Usage Rate(%)",
                        "NPU Real-time Power Utilization(%)"],
    "hbm_util_pct":    ["HBM Usage Rate(%)"],
    "mem_util_pct":    ["Memory Usage Rate(%)", "DDR Usage Rate(%)"],
    "hbm_total_mb":    ["HBM Capacity(MB)"],
    "hbm_used_mb":     ["HBM Usage(MB)", "HBM Used(MB)"],
    "power_w":         ["Power(W)", "NPU Real-time Power(W)", "Chip Power(W)", "Power Dissipation(W)"],
    "temp_c":          ["Temperature(C)", "NPU Temperature(C)", "Chip Temperature(C)",
                        "NPU Real-time Temperature(C)"],
}

_KV = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ()/%.-]*?)\s*[:=]\s*(.+?)\s*$")


def _load_topology(path: Path):
    """极简 YAML 读取(避免依赖 pyyaml):只解析我们这份固定结构。"""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text())
    except Exception:
        pass
    # 退化解析器:够读本仓库这份格式(- {k: v, ...} 行)。
    doc = {"pool_inference": [], "pool_verify": [], "engine_endpoints": {}}
    section = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith(":") and not s.startswith("-"):
            section = s[:-1].strip()
            continue
        if s.startswith("- {") and section in ("pool_inference", "pool_verify"):
            body = s[3:].rstrip("}")
            row = {}
            for pair in body.split(","):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    row[k.strip()] = v.strip()
            doc[section].append(row)
        elif section == "engine_endpoints" and ":" in s:
            k, v = s.split(":", 1)
            doc["engine_endpoints"][k.strip()] = v.strip()
    return doc


def _cards(topo):
    out = []
    for pool, key in (("inference", "pool_inference"), ("verify", "pool_verify")):
        for row in topo.get(key) or []:
            out.append({
                "card_id": int(row["card_id"]),
                "chip_id": int(row.get("chip_id", 0)),
                "engine_id": str(row.get("engine_id", "")),
                "tp_rank": int(row.get("tp_rank", 0)),
                "pool": pool,
            })
    return out


def _run_smi(args: list[str]) -> str:
    exe = shutil.which("npu-smi") or "npu-smi"
    try:
        return subprocess.run([exe, "info", *args], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__ {e}"


def _parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _KV.match(line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def _pick(kv: dict[str, str], field: str):
    for name in _ALIASES[field]:
        if name in kv:
            raw = kv[name].split()[0].split("/")[0]
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def sample_card(card) -> dict[str, float]:
    i, c = card["card_id"], card["chip_id"]
    kv = _parse_kv(_run_smi(["-t", "common", "-i", str(i), "-c", str(c)]))
    kv.update(_parse_kv(_run_smi(["-t", "usages", "-i", str(i), "-c", str(c)])))
    # power/temp/mem 在部分 CANN 版本单独子表里(key:value,比默认表格稳)。取不到就跳过,不报错。
    for sub in ("power", "temp", "mem"):
        extra = _run_smi(["-t", sub, "-i", str(i), "-c", str(c)])
        if not extra.startswith("__ERROR__"):
            kv.update(_parse_kv(extra))
    vals: dict[str, float] = {}
    for field in _ALIASES:
        v = _pick(kv, field)
        if v is not None:
            vals[field] = v
    # 若只有 rate 没有 used MB,用 rate×capacity 估
    if "hbm_used_mb" not in vals and {"hbm_util_pct", "hbm_total_mb"} <= vals.keys():
        vals["hbm_used_mb"] = round(vals["hbm_total_mb"] * vals["hbm_util_pct"] / 100.0, 1)
    return vals


class Sampler(threading.Thread):
    def __init__(self, cards, interval: float, out_dir=None):
        super().__init__(daemon=True)
        self.cards, self.interval = cards, interval
        self.snapshot: str = "# no sample yet\n"
        self._stop = threading.Event()
        # T1 落文件:不依赖 Prometheus,直接把每卡采样写 jsonl,供 analyze/DuckDB。
        self.out_path = None
        if out_dir:
            import os as _os
            _os.makedirs(out_dir, exist_ok=True)
            self.out_path = _os.path.join(out_dir, "npu_card.jsonl")

    def run(self):
        while not self._stop.is_set():
            samples = [(card, sample_card(card)) for card in self.cards]
            self.snapshot = self._render(samples)
            if self.out_path:
                self._write_jsonl(samples)
            self._stop.wait(self.interval)

    def _write_jsonl(self, samples):
        import json as _json
        ts = round(_now(), 3)
        try:
            with open(self.out_path, "a", buffering=1) as f:
                for card, vals in samples:
                    row = {"recorded_at_unix": ts, "card_id": card["card_id"],
                           "pool": card["pool"], "engine_id": card["engine_id"],
                           "tp_rank": card["tp_rank"], **vals}
                    f.write(_json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _render(self, samples=None) -> str:
        if samples is None:
            samples = [(card, sample_card(card)) for card in self.cards]
        lines = [
            "# HELP npu_aicore_util_pct AICore utilization percent",
            "# TYPE npu_aicore_util_pct gauge",
        ]
        metric_help = {
            "aicore_util_pct": "npu_aicore_util_pct",
            "npu_util_pct": "npu_util_pct",
            "hbm_util_pct": "npu_hbm_util_pct",
            "hbm_used_mb": "npu_hbm_used_mb",
            "hbm_total_mb": "npu_hbm_total_mb",
            "mem_util_pct": "npu_mem_util_pct",
            "power_w": "npu_power_watts",
            "temp_c": "npu_temp_celsius",
        }
        for card, vals in samples:
            lbl = (f'card_id="{card["card_id"]}",pool="{card["pool"]}",'
                   f'engine_id="{card["engine_id"]}",tp_rank="{card["tp_rank"]}"')
            for field, metric in metric_help.items():
                if field in vals:
                    lines.append(f"{metric}{{{lbl}}} {vals[field]}")
        lines.append(f"npu_exporter_last_scrape_unixtime {int(_now())}")
        return "\n".join(lines) + "\n"

    def stop(self):
        self._stop.set()


# time.time() 在本仓某些沙箱被 ban;真实部署无此限制。用 monotonic 兜底纪元。
def _now() -> float:
    try:
        return time.time()
    except Exception:
        return time.monotonic()


def _serve(sampler: Sampler, port: int):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("", "/metrics"):
                body = sampler.snapshot.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):  # 静默
            pass

    http.server.HTTPServer(("0.0.0.0", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--topology", type=Path, default=here / "card_topology.yaml")
    ap.add_argument("--port", type=int, default=9810)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="采集一次打印后退出(校准用)")
    ap.add_argument("--raw", action="store_true", help="打印某卡 npu-smi 原始输出")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--out", default=None, help="T1 落文件目录:每 interval 每卡写 npu_card.jsonl(不依赖 Prometheus)")
    args = ap.parse_args()

    if args.raw:
        print("### -t common ###\n", _run_smi(["-t", "common", "-i", str(args.card), "-c", "0"]))
        print("### -t usages ###\n", _run_smi(["-t", "usages", "-i", str(args.card), "-c", "0"]))
        return

    topo = _load_topology(args.topology)
    cards = _cards(topo)
    if not cards:
        sys.exit(f"no cards parsed from {args.topology}")

    if args.once:
        for card in cards:
            print(card["pool"], card["engine_id"], f"card{card['card_id']}", sample_card(card))
        return

    sampler = Sampler(cards, args.interval, out_dir=args.out)
    sampler.start()
    print(f"npu-smi exporter on :{args.port}/metrics — {len(cards)} cards, interval {args.interval}s"
          + (f"; T1 落文件 → {args.out}/npu_card.jsonl" if args.out else ""))
    try:
        _serve(sampler, args.port)
    except OSError as e:
        # 端口被占等 → HTTP 端点起不来,但落文件采集必须继续(T1 与 /metrics 端点解耦)。
        print(f"[warn] :{args.port} 起不来({e}); /metrics 端点跳过,继续落文件采集。", file=sys.stderr)
        if not args.out:
            sys.exit(f"既无 --out 落文件,:{args.port} 又被占,退出。换端口或加 --out。")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
