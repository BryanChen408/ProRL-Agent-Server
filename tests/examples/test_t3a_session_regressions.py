"""CPU regressions from the 2026-09-09 native sessions; no live services."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'operator_runtime_t3a'
ASSETS = ROOT / 'deploy/ascend_operator/t3a'


def test_fresh_replica_reproduces_installed_runtime(tmp_path):
    upstream = Path(os.environ.get('CANNBOT_SRC', '/home/docker/cannbot-skills'))
    if not upstream.is_dir():
        pytest.skip('local cannbot source is required for rebuild verification')
    destination = tmp_path / 'runtime'
    result = subprocess.run(['bash', str(ROOT / 'deploy/ascend_operator/build_t3a_replica.sh')],
                            env={**os.environ, 'T3A_DST': str(destination)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    def contents(root):
        return {p.relative_to(root): p.read_bytes() for p in root.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.name != 'REPLICATION_LEDGER.md'}
    assert contents(destination) == contents(RUNTIME)


def test_rpaths_repairs_partial_install_and_is_idempotent(tmp_path):
    shutil.copytree(RUNTIME / 'skills', tmp_path / 'skills')
    validator = tmp_path / 'skills/tilelang2ascend-tilelang-designer/scripts/validate_tilelang_impl.py'
    text = validator.read_text()
    start = text.index('def _skill_scripts(name):')
    end = text.index('_ASCENDC_SCRIPTS = (', start)
    validator.write_text(text[:start] + text[end:])
    patch = runpy.run_path(str(ASSETS / 'patch_rpaths.py'))
    patch['main'](tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.py')}
    patch['main'](tmp_path)
    assert all(p.read_bytes() == b for p, b in before.items())
    result = subprocess.run([sys.executable, str(validator), '--help'],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'NameError' not in result.stderr
    with pytest.raises(ValueError, match='anchor missing'):
        patch['patch'](validator, 'not an upstream anchor', 'nor an installed patch')


def test_tilelang_verifier_imports_task_design_and_restores_path(tmp_path, monkeypatch):
    # Run the actual verification function and native loader; only NPU/model
    # execution is stubbed. This used to return ModuleNotFoundError: design.
    source = RUNTIME / 'skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py'
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_verification')
    load_source = RUNTIME / 'skills/ops-profiling/scripts/msprof_perf_summary.py'
    load_fn = next(n for n in ast.parse(load_source.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == '_load_module')
    (tmp_path / 'design').mkdir()
    (tmp_path / 'design/__init__.py').write_text('VALUE = 123\n')
    (tmp_path / 'model.py').write_text('get_init_inputs = lambda: []\n')
    (tmp_path / 'model_new_tilelang.py').write_text('from design import VALUE\nassert VALUE == 123\n')
    class Model:
        def to(self, device): return self
        def eval(self): return self
    namespace = dict(sys=sys, os=os, importlib=__import__('importlib'),
                     WORKDIR=tmp_path / 'unrelated_skill_dir',
                     torch=SimpleNamespace(manual_seed=lambda x: None),
                     _make_report=lambda op: {}, _resolve_task_dir=lambda *a, **k: tmp_path,
                     _find_model_class=lambda *a: Model, _clone_value=lambda x: x,
                     _get_input_groups=lambda m: [1], _get_device=lambda: 'cpu',
                     _run_comparisons=lambda *a: (True, ['matched'], []))
    import traceback
    namespace['traceback'] = traceback
    exec(compile(ast.Module(body=[load_fn, fn], type_ignores=[]), str(source), 'exec'), namespace)
    monkeypatch.delitem(sys.modules, 'design', raising=False)
    old_path = sys.path[:]
    try:
        assert namespace['_run_verification']('OP')['ok'] is True
        assert sys.path == old_path
        (tmp_path / 'model_new_tilelang.py').write_text('raise RuntimeError("candidate bug")\n')
        assert 'candidate bug' in namespace['_run_verification']('OP')['error']
        assert sys.path == old_path
    finally:
        for name in ('design', 'OP_ref_model', 'OP_tilelang_model'):
            sys.modules.pop(name, None)


def test_attempt_candidates_and_completion_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('POLAR_OP_NAME', 'OP')
    monkeypatch.setenv('POLAR_T3A_CANDIDATES_DIR', str(tmp_path / 'artifacts/t3a_candidates'))
    module = runpy.run_path(str(ASSETS / 'hooks/t3a_running_best.py'))
    assert (ASSETS / 'hooks/t3a_running_best.py').read_bytes() == (RUNTIME / 'hooks/t3a_running_best.py').read_bytes()
    op = tmp_path / 'OP'
    op.mkdir()
    (op / 'model_new_tilelang.py').write_text('# intermediate\n')
    def attempt(script, classification='PASS', passed=5):
        module['record_attempt'](command=f'bash {script} OP', classification=classification,
            exit_code=0 if classification == 'PASS' else 1,
            stdout=f'Result: {passed}/5 passed', duration_ms=1,
            cwd=str(tmp_path), project_root=str(tmp_path), script=script)
    index = tmp_path / 'artifacts/t3a_candidates/index.json'
    attempt('evaluate_tilelang.sh')
    attempt('evaluate_ascendc.sh')  # Malformed project is never a submission.
    assert not index.exists()
    (op / 'trace.md').write_text('Phase 3 only\n')
    assert module['stop_reason'](str(tmp_path))
    (op / 'kernel').mkdir()
    (op / 'kernel/op.cpp').write_text('// candidate one\n')
    (op / 'model_new_ascendc.py').write_text('# wrapper\n')
    attempt('evaluate_tilelang.sh')  # Still must not rank using TileLang scores.
    assert not index.exists()
    attempt('evaluate_ascendc.sh', 'A', 0)
    assert module['stop_reason'](str(tmp_path)) is None  # Real failed eval can be reported honestly.
    stream = tmp_path / 'artifacts/t3a_attempt_stream.jsonl'
    events = [json.loads(s) for s in stream.read_text().splitlines()]
    assert not any(e.get('event') == 'promote' for e in events)
    # A transient failure may recover without a source edit; keep the PASS.
    attempt('evaluate_ascendc.sh')
    entries = json.loads(index.read_text())
    assert entries[0]['classification'] == 'PASS'
    assert all(e['backend'] == 'ascendc' for e in entries)
    with tarfile.open(entries[0]['file']) as archive:
        assert 'OP/model_new_ascendc.py' in archive.getnames()
        assert 'OP/kernel/op.cpp' in archive.getnames()
    before = index.read_bytes()
    (op / 'design.md').write_text('newer timestamp is not better correctness\n')
    attempt('evaluate_ascendc.sh')
    assert index.read_bytes() == before
    events = [json.loads(s) for s in stream.read_text().splitlines()]
    assert sum(e.get('event') == 'promote' for e in events) == 1
    (op / 'trace.md').unlink()
    hook = RUNTIME / 'hooks/t3a_running_best.py'
    result = subprocess.run([sys.executable, str(hook), '--stop'],
                            input=json.dumps({'cwd': str(tmp_path), 'stop_hook_active': True}),
                            env={k: v for k, v in os.environ.items() if k != 'CLAUDE_PROJECT_DIR'},
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'tilelang2ascendc-kernel-generator' in result.stderr


def test_t3a_precision_gate_has_no_terminal_retry_quota(tmp_path):
    module = runpy.run_path(str(RUNTIME / 'hooks/doc_gate.py'))
    gate = module['check_precision_gate']
    gate.__globals__['read_precision_state'] = lambda: dict(
        d_class_active=True, edits_allowed=0, stage='D2', debug_call_count=7, tune_call_count=100)
    denied, reason = gate()
    assert denied  # Must still consult the diagnostic skill before editing.
    assert 'ascendc-precision-tuning' in reason
    assert 'TERMINAL' not in reason


def test_device_override_denied_before_lease_execution(tmp_path, monkeypatch):
    monkeypatch.setenv('POLAR_NPU_LEASE_POOL', '0')
    script = RUNTIME / 'hooks/skill_script_hook.py'
    payload = {'tool_name': 'Bash', 'tool_input': {'command':
        'ASCEND_RT_VISIBLE_DEVICES=7 python3 verification_tilelang.py OP'}}
    result = subprocess.run([sys.executable, str(script)], input=json.dumps(payload),
                            capture_output=True, text=True, cwd=tmp_path, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['hookSpecificOutput']['permissionDecision'] == 'deny'
