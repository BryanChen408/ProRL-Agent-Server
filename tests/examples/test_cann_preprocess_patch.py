"""Exercise the shipped vendor patch without needing an Ascend compiler or device."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_nondestructive_merge_repeats_and_propagates_failure(tmp_path):
    sdk = tmp_path / 'sdk'
    shutil.copytree(ROOT / 'tests/fixtures/cann_preprocess_9_2', sdk)
    patch = ROOT / 'operator_runtime_t2a/skills/tilelang2ascend-translator/scripts/cann_preprocess.patch'
    with patch.open('rb') as f:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1'], cwd=sdk, stdin=f, check=True)
    linker = tmp_path / 'fake linker'
    linker.write_text('''#!/usr/bin/env python3
import sys
from pathlib import Path
src = Path(sys.argv[sys.argv.index('-Ttext=0') + 1])
dst = Path(sys.argv[sys.argv.index('-o') + 1])
data = src.read_bytes()
dst.write_bytes(b'LINKED:' + data)
if data == b'FAIL':
    sys.exit(7)
''')
    linker.chmod(0o755)

    def merge(*inputs):
        return subprocess.run(['bash', str(sdk / 'util/merge_obj_text.sh'), '-l',
                               str(linker), *map(str, inputs)], capture_output=True)

    a, b = tmp_path / 'first file.o', tmp_path / 'second.o'
    a.write_bytes(b'RAW1')
    b.write_bytes(b'RAW2')
    for _ in range(3):
        assert merge(a, b).returncode == 0
        assert a.read_bytes() == b'RAW1' and b.read_bytes() == b'RAW2'
        assert Path(str(a) + '.merged').read_bytes() == b'LINKED:RAW1'
        assert Path(str(b) + '.merged').read_bytes() == b'LINKED:RAW2'
    old_output = Path(str(a) + '.merged').read_bytes()
    a.write_bytes(b'FAIL')
    assert merge(a).returncode == 7
    assert a.read_bytes() == b'FAIL'
    assert Path(str(a) + '.merged').read_bytes() == old_output
    assert not list(tmp_path.glob('*.tmp.*'))

    def independent(i):
        path = tmp_path / f'parallel {i}.o'
        data = f'RAW{i}'.encode()
        path.write_bytes(data)
        assert merge(path).returncode == 0
        assert path.read_bytes() == data
        assert Path(str(path) + '.merged').read_bytes() == b'LINKED:' + data

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(independent, range(16)))
    assert not list(tmp_path.glob('*.tmp.*'))
