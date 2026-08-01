#!/usr/bin/env python3
"""T3 host_proc:主机 + 关键进程系统负载采集(常驻 daemon,落文件)。

纯标准库 + /proc 解析,零三方依赖(psutil 不装也能跑)。每 --interval 秒写:
  - 一行 kind="host":内存(MemAvailable/Cached/Dirty/Writeback)、swap、loadavg、cpu%、
    每网卡 rx/tx 字节(累计+区间速率)、NFS(/mnt/share)读写字节(mountstats)。
  - 每个匹配到的关键进程一行 kind="proc":role/pid/rss_mb/cpu_pct/num_threads/num_fds。
    role 按 cmdline 归类:gateway/rollout/proxy/vllm_engine/ray/agent/lease。

用法:
  python3 host_proc_exporter.py --out /mnt/share/polar_engine_metrics/host_state --interval 5
落 <out>/host_proc.jsonl。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict

_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_NCPU = os.cpu_count() or 1

# cmdline 独特 token → role(先到先得)。用独特 token 避免把路径里的普通词也算进来。
# exporter/poller 自身排除,免自采。
_ROLES = [
    ("dp_load_balance_proxy", "proxy"),
    ("npu_lease_exec", "lease"),
    ("gateway_server", "gateway"), ("polar.gateway", "gateway"), ("uvicorn", "gateway"),
    ("train_async", "rollout"), ("rollout_worker", "rollout"),
    ("vllm.entrypoints", "vllm_engine"), ("EngineCore", "vllm_engine"), ("VllmWorker", "vllm_engine"),
    ("raylet", "ray"), ("gcs_server", "ray"), ("ray::", "ray"),
    (".claude", "agent"), ("claude-code", "agent"),
]
_EXCLUDE = ("host_proc_exporter", "npu_smi_exporter", "vllm_metrics_poller", "telemetry_aggregator")
# 这些 role 进程数动辄成百(ray actor),滚成一条汇总行(带 n_procs + 求和),不逐进程灌爆文件。
_ROLLUP = {"ray"}


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return time.monotonic()


def _meminfo() -> dict:
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, rest = line.partition(":")
            out[k.strip()] = int(rest.strip().split()[0])  # kB
    except Exception:
        pass
    return out


def _loadavg():
    try:
        p = open("/proc/loadavg").read().split()
        return float(p[0]), float(p[1]), float(p[2])
    except Exception:
        return None, None, None


def _cpu_jiffies():
    try:
        p = open("/proc/stat").readline().split()[1:]
        vals = [int(x) for x in p]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle
    except Exception:
        return None, None


def _net_dev():
    out = {}
    try:
        for line in open("/proc/net/dev").readlines()[2:]:
            iface, _, rest = line.partition(":")
            f = rest.split()
            if len(f) >= 9:
                out[iface.strip()] = (int(f[0]), int(f[8]))  # rx_bytes, tx_bytes
    except Exception:
        pass
    return out


def _nfs_bytes(mount_substr="share"):
    """从 /proc/self/mountstats 读 NFS 挂载的读写字节(agentic RL 落 artifact 的 I/O 压力)。"""
    try:
        cur, hit = None, {}
        for line in open("/proc/self/mountstats"):
            if line.startswith("device ") and " mounted on " in line:
                cur = line.split(" mounted on ")[1].split(" with ")[0].strip()
            elif cur and mount_substr in cur and line.strip().startswith("bytes:"):
                b = line.split()[1:]
                # bytes: normal_read normal_write direct_read direct_write server_read server_write ...
                if len(b) >= 6:
                    hit[cur] = {"read_bytes": int(b[4]), "write_bytes": int(b[5])}
                cur = None
        return hit or None
    except Exception:
        return None


def _proc_stat_cpu(pid):
    try:
        data = open(f"/proc/{pid}/stat").read()
        after = data[data.rfind(")") + 2:].split()
        utime, stime = int(after[11]), int(after[12])  # 14th,15th field overall
        return utime + stime
    except Exception:
        return None


def _proc_status(pid):
    rss_kb = threads = None
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("Threads:"):
                threads = int(line.split()[1])
            if rss_kb is not None and threads is not None:
                break
    except Exception:
        pass
    return rss_kb, threads


def _cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _role(cmd):
    if any(x in cmd for x in _EXCLUDE):
        return None
    for kw, role in _ROLES:
        if kw in cmd:
            return role
    return None


def _num_fds(pid):
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except Exception:
        return None


def sample(prev):
    """采一轮。prev = {'cpu':(total,idle),'ts':t,'procs':{pid:jiffies},'net':{iface:(rx,tx)}}。返回 (rows, newprev)。"""
    ts = round(_now(), 3)
    rows = []
    mi = _meminfo()
    la1, la5, la15 = _loadavg()
    tot, idle = _cpu_jiffies()
    cpu_pct = None
    if prev.get("cpu") and tot is not None:
        dt, di = tot - prev["cpu"][0], idle - prev["cpu"][1]
        cpu_pct = round(100.0 * (dt - di) / dt, 2) if dt > 0 else None
    net = _net_dev()
    net_row = {}
    for iface, (rx, tx) in net.items():
        if iface == "lo":
            continue
        e = {"rx_bytes": rx, "tx_bytes": tx}
        if prev.get("net", {}).get(iface) and prev.get("ts"):
            dtw = ts - prev["ts"]
            if dtw > 0:
                e["rx_bps"] = round((rx - prev["net"][iface][0]) / dtw, 1)
                e["tx_bps"] = round((tx - prev["net"][iface][1]) / dtw, 1)
        net_row[iface] = e
    host = {
        "recorded_at_unix": ts, "kind": "host",
        "mem_total_mb": round(mi.get("MemTotal", 0) / 1024, 1),
        "mem_avail_mb": round(mi.get("MemAvailable", 0) / 1024, 1),
        "mem_used_mb": round((mi.get("MemTotal", 0) - mi.get("MemAvailable", 0)) / 1024, 1),
        "page_cache_mb": round(mi.get("Cached", 0) / 1024, 1),
        "dirty_mb": round(mi.get("Dirty", 0) / 1024, 1),
        "writeback_mb": round(mi.get("Writeback", 0) / 1024, 1),
        "swap_used_mb": round((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1024, 1),
        "cpu_pct": cpu_pct, "ncpu": _NCPU,
        "load1": la1, "load5": la5, "load15": la15,
        "net": net_row, "nfs": _nfs_bytes(),
    }
    rows.append(host)

    # 关键进程。海量同类(ray actor)滚成一条汇总,服务类逐进程。
    new_procs = {}
    dtw = ts - prev.get("ts", ts)
    per_proc = []          # 服务类:逐进程
    roll = defaultdict(lambda: {"n": 0, "rss_mb": 0.0, "cpu_pct": 0.0, "num_threads": 0, "num_fds": 0})
    for pid_dir in glob.glob("/proc/[0-9]*"):
        pid = pid_dir.rsplit("/", 1)[-1]
        cmd = _cmdline(pid)
        role = _role(cmd) if cmd else None
        if not role:
            continue
        jif = _proc_stat_cpu(pid)
        new_procs[pid] = jif
        p_cpu = None
        if jif is not None and pid in prev.get("procs", {}) and dtw > 0 and prev["procs"][pid] is not None:
            p_cpu = round(100.0 * (jif - prev["procs"][pid]) / _CLK / dtw, 2)
        rss_kb, threads = _proc_status(pid)
        rss_mb = round(rss_kb / 1024, 1) if rss_kb else None
        fds = _num_fds(pid)
        if role in _ROLLUP:
            a = roll[role]
            a["n"] += 1
            a["rss_mb"] += rss_mb or 0.0
            a["cpu_pct"] += p_cpu or 0.0
            a["num_threads"] += threads or 0
            a["num_fds"] += fds or 0
        else:
            per_proc.append({
                "recorded_at_unix": ts, "kind": "proc", "role": role, "pid": int(pid),
                "rss_mb": rss_mb, "cpu_pct": p_cpu, "num_threads": threads,
                "num_fds": fds, "cmd": cmd[:120],
            })
    for role, a in roll.items():
        rows.append({
            "recorded_at_unix": ts, "kind": "proc", "role": role, "rollup": True,
            "n_procs": a["n"], "rss_mb": round(a["rss_mb"], 1), "cpu_pct": round(a["cpu_pct"], 2),
            "num_threads": a["num_threads"], "num_fds": a["num_fds"],
        })
    rows.extend(per_proc)
    return rows, {"cpu": (tot, idle), "ts": ts, "procs": new_procs,
                  "net": {k: v for k, v in net.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("POLAR_ENGINE_METRICS_DIR",
                    "/mnt/share/polar_engine_metrics") + "/host_state")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "host_proc.jsonl")
    print(f"host_proc_exporter → {path} (interval {args.interval}s, {_NCPU} cpu)")

    prev = {}
    # 预热一拍(拿基线,cpu%/速率需两点差分)
    _, prev = sample(prev)
    time.sleep(min(args.interval, 2.0))
    while True:
        rows, prev = sample(prev)
        with open(path, "a", buffering=1) as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if args.once:
            n_proc = sum(1 for r in rows if r["kind"] == "proc")
            print(f"wrote 1 host + {n_proc} proc rows"); return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
