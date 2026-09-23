#!/usr/bin/env python3
"""Host RAM incident recorder. Run as root on the host, not in a PID-isolated container."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from host_proc_exporter import _meminfo


def read(path, limit=131072):
    try:
        with open(path) as f:
            return f.read(limit)
    except OSError as exc:
        return f"ERROR: {exc}"


def processes(previous, root=Path('/proc')):
    rows = []
    for p in root.iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            start = stat[19]
            fields = dict(line.split(':', 1) for line in (p / 'status').read_text().splitlines())
            rss = int(fields.get('VmRSS', '0 kB').split()[0])
            key = (int(p.name), start)
            rows.append(dict(pid=int(p.name), start=start, name=fields['Name'].strip(),
                             ppid=int(fields['PPid']), rss_kb=rss,
                             swap_kb=int(fields.get('VmSwap', '0 kB').split()[0]),
                             growth_kb=max(0, rss-previous.get(key, 0)),
                             cgroup=read(p / 'cgroup').strip()))
        except (OSError, ValueError, KeyError, IndexError):
            continue  # A process may disappear during the scan.
    return sorted(rows, key=lambda r: r['rss_kb'], reverse=True)


def level(used_percent, thresholds):
    return sum(used_percent >= t for t in thresholds)


def command(argv, path, timeout=3):
    # Write directly to disk, never accumulate potentially large tool output in RAM.
    with path.open('w') as f:
        try:
            result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            f.write(f'\nexit_code={result.returncode}\n')
        except (OSError, subprocess.TimeoutExpired) as exc:
            f.write(f'\nERROR: {exc}\n')


def snapshot(out, rows, host, spy, top):
    dest = out / ('incident-' + str(time.time_ns()))
    dest.mkdir()
    (dest / 'host.json').write_text(json.dumps(host))
    (dest / 'processes.json').write_text(json.dumps(rows))
    for name in ('meminfo', 'vmstat', 'slabinfo', 'pressure/memory', 'pressure/io'):
        (dest / name.replace('/', '-')).write_text(read('/proc/' + name, 1048576))
    # Keep the first incident plus the latest 31, retaining the onset of the failure.
    old = sorted(out.glob('incident-*'))
    for path in old[1:-31]:
        shutil.rmtree(path)
    for cg in {r['cgroup'] for r in rows}:
        for line in cg.splitlines():
            if not line.startswith('0::'):
                continue
            path = (Path('/sys/fs/cgroup') / line[3:].lstrip('/')).resolve()
            if not path.is_relative_to('/sys/fs/cgroup'):
                continue
            record = {'path': str(path)}
            for name in ('memory.current', 'memory.stat', 'memory.events', 'memory.max'):
                record[name] = read(path / name)
            with (dest / 'cgroups.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
    command(['dmesg', '--ctime', '--since', '10 minutes ago'], dest / 'dmesg.txt')
    selected = {r['pid']: r for r in rows[:top]}
    selected.update({r['pid']: r for r in sorted(rows, key=lambda r: r['growth_kb'], reverse=True)[:top]
                     if r['growth_kb'] > 0})
    for pid, row in selected.items():
        if pid == os.getpid():
            continue
        p = Path('/proc') / str(pid)
        try:
            if (p / 'stat').read_text().rsplit(')', 1)[1].split()[19] != row['start']:
                continue  # PID reused since sampling.
        except (OSError, IndexError):
            continue
        folder = dest / str(pid)
        folder.mkdir()
        for name in ('status', 'smaps_rollup', 'cgroup', 'wchan'):
            (folder / name).write_text(read(p / name))
        # No environ or local variables: these can contain credentials and huge tensors.
        tasks = sorted((p / 'task').glob('[0-9]*'))
        with (folder / 'kernel_stacks.txt').open('w') as f:
            f.write(f'threads={len(tasks)}, captured_at_most=64\n')
            for task in tasks[:64]:
                f.write(f'\nTID {task.name}\n{read(task / "stack")}\n')
        # Attempt even with renamed Python workers (Ray/vLLM change process titles).
        command([spy, 'dump', '--pid', str(pid), '--nonblocking', '--full-filenames'],
                folder / 'python_stacks.txt')
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--py-spy', default=shutil.which('py-spy'))
    ap.add_argument('--interval', type=float, default=2)
    ap.add_argument('--thresholds', type=float, nargs='+', default=[85, 90, 95, 98])
    ap.add_argument('--cooldown', type=float, default=60)
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--allow-container', action='store_true', help='Partial visibility only; for testing')
    args = ap.parse_args()
    if (args.interval <= 0 or args.cooldown <= 0 or not 1 <= args.top <= 20
            or not all(0 < t < 100 for t in args.thresholds)
            or args.thresholds != sorted(set(args.thresholds))):
        ap.error('require positive intervals, top 1..20, increasing unique thresholds in (0,100)')
    if not args.allow_container and read('/proc/1/comm').strip() not in ('systemd', 'init'):
        ap.error('Run on the host: PID 1 is not systemd/init. Container process visibility is incomplete.')
    if not args.py_spy or not os.access(args.py_spy, os.X_OK):
        ap.error('provide --py-spy /absolute/path/to/py-spy')
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'probe.pid').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('a probe is already running in this output directory')
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    previous, last_level, last_capture = {}, 0, float('-inf')
    print(f'memory probe pid={os.getpid()} out={args.out} thresholds={args.thresholds}', flush=True)
    while True:
        mi = _meminfo()
        if not mi.get('MemTotal') or 'MemAvailable' not in mi:
            raise RuntimeError('/proc/meminfo missing MemTotal/MemAvailable')
        used = 100 * (1 - mi['MemAvailable'] / mi['MemTotal'])
        rows = processes(previous)
        current = level(used, args.thresholds)
        host = dict(time=time.time(), used_percent=used, meminfo_kb=mi,
                    pid1=read('/proc/1/comm').strip(), boot_id=read('/proc/sys/kernel/random/boot_id').strip(),
                    visible_processes=len(rows), rss_sum_kb=sum(r['rss_kb'] for r in rows))
        log = args.out / 'memory.jsonl'
        if log.exists() and log.stat().st_size > 32 * 1024**2:
            log.replace(args.out / 'memory.previous.jsonl')
        with log.open('a') as f:
            f.write(json.dumps({**host, 'top_rss': rows[:30], 'top_growth':
                               sorted(rows, key=lambda r: r['growth_kb'], reverse=True)[:30]}) + '\n')
        if current and (current > last_level or time.monotonic()-last_capture >= args.cooldown):
            print(f'capture used={used:.2f}% {snapshot(args.out, rows, host, args.py_spy, args.top)}', flush=True)
            last_capture = time.monotonic()
        previous = {(r['pid'], r['start']): r['rss_kb'] for r in rows}
        last_level = current
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
