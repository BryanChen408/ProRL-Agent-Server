import importlib.util
from pathlib import Path
import sys


def test_memory_probe_thresholds_growth_and_snapshot(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'deploy/ascend_operator/telemetry'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('memory_probe', scripts / 'memory_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    proc = tmp_path / 'proc' / '42'
    proc.mkdir(parents=True)
    (proc / 'stat').write_text('42 (worker (renamed)) ' + ' '.join(['S'] + ['0']*18 + ['123']))
    (proc / 'status').write_text('Name:\tworker\nPPid:\t1\nVmRSS:\t4096 kB\nVmSwap:\t512 kB\n')
    (proc / 'cgroup').write_text('0::/test-container\n')
    rows = probe.processes({(42, '123'): 1024}, proc.parent)
    assert rows[0]['growth_kb'] == 3072
    assert rows[0]['cgroup'] == '0::/test-container'
    assert probe.processes({(42, 'old-pid'): 9999}, proc.parent)[0]['growth_kb'] == 4096
    assert [probe.level(n, [85, 90, 95, 98]) for n in [84, 85, 94, 99]] == [0, 1, 2, 4]
    # Exercise real file output without probing any running workload.
    monkeypatch.setattr(probe, 'command', lambda argv, path, **kw: path.write_text('test'))
    out = tmp_path / 'out'
    out.mkdir()
    dest = probe.snapshot(out, [], {'used_percent': 99}, sys.executable, 5)
    assert (dest / 'processes.json').read_text() == '[]'
    assert 'MemTotal:' in (dest / 'meminfo').read_text()
    assert (dest / 'dmesg.txt').read_text() == 'test'
