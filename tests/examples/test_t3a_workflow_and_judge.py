"""Actual hook/driver regressions for the latest T3A sessions; CPU only."""
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import re
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'operator_runtime_t3a'
SOURCE = ROOT / 'deploy/ascend_operator/t3a'


def test_native_agent_declared_skills_are_installed():
    for agent in (RUNTIME / 'agents').glob('*.md'):
        frontmatter = agent.read_text().split('---', 2)[1]
        skills = re.search(r'^skills:\n((?:\s+-\s+.+\n?)*)', frontmatter, re.MULTILINE).group(1)
        for name in re.findall(r'^\s+-\s+(.+)$', skills, re.MULTILINE):
            assert (RUNTIME / 'skills' / name / 'SKILL.md').is_file(), name


def test_intercepted_pipeline_does_not_hide_failure_behind_tail(tmp_path, monkeypatch):
    module = runpy.run_path(str(RUNTIME / 'hooks/skill_script_hook.py'))
    code, stdout, _, _ = module['_run_command']("printf 'Result: pass\\n'; false | tail -1", str(tmp_path))
    assert code != 0
    assert module['_classify_result'](code, stdout, '') != 'PASS'


def prepare(tmp_path, monkeypatch):
    op = 'npukernelbench_level1_20_Gather'
    (tmp_path / 'input').mkdir()
    (tmp_path / f'input/{op}.py').write_text('# reference\n')
    (tmp_path / f'input/{op}.json').write_text('{}\n' * 5)
    argv = ['prepare', '--op-name', op, '--workdir', str(tmp_path), '--canonical-root', str(RUNTIME)]
    monkeypatch.setattr(sys, 'argv', argv)
    runpy.run_path(str(SOURCE / 'runtime/prepare_operator_workdir.py'))['main']()
    return op


def test_prepared_hook_enforces_roles_and_updates_old_dispatch(tmp_path, monkeypatch):
    op = prepare(tmp_path, monkeypatch)
    contract = json.loads((tmp_path / '.claude/task_contract.json').read_text())
    assert contract['case_mode'] == 'simple'
    assert 'As the main orchestrator' not in contract['developer_prompt']
    assert f'{tmp_path}/input/{op}.py' in contract['developer_prompt']
    assert 'retaining all 5 cases unchanged' in contract['developer_prompt']
    hook = tmp_path / '.claude/hooks/workflow_hook.py'
    # Any execution by the skill hook is observable without touching an NPU.
    delegated = tmp_path / 'executed'
    (hook.parent / 'skill_script_hook.py').write_text(
        f'from pathlib import Path\nPath({str(delegated)!r}).write_text("ran")\n'
        'print("{}")\n')
    env = {**os.environ, 'CLAUDE_PROJECT_DIR': str(tmp_path)}
    def call(name, args=None, **extra):
        p = subprocess.run([sys.executable, str(hook)], input=json.dumps({
            'tool_name': name, 'tool_input': args or {}, 'cwd': str(tmp_path), **extra}),
            env=env, capture_output=True, text=True, timeout=10)
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout).get('hookSpecificOutput', {})
    for name in ('Write', 'Edit', 'Bash', 'Skill'):
        assert call(name)['permissionDecision'] == 'deny'
    assert not delegated.exists()  # Denied main-chain Bash never reaches execution hook.
    assert call('Bash', {'command': 'python3 -c "import torch; print(torch.npu.is_available())"'},
                agent_id='sub')['permissionDecision'] == 'deny'
    assert not delegated.exists()
    assert call('Agent', {'subagent_type': 'general-purpose'})['permissionDecision'] == 'deny'
    assert call('Agent', {'subagent_type': 'tilelang2ascend-tilelang-designer'}, agent_id='sub')['permissionDecision'] == 'deny'
    args = {'subagent_type': 'tilelang2ascendc-kernel-generator', 'prompt': 'OLD: use wrong/output', 'resume': 'agent-123'}
    updated = call('Agent', args)['updatedInput']
    assert updated['resume'] == 'agent-123'
    assert updated['prompt'].endswith(contract['developer_prompt'])
    assert call('Agent', updated)['updatedInput'] == updated
    assert call('Skill', {'skill': 'tilelang2ascend-case-simplifier'}, agent_id='sub')['permissionDecision'] == 'deny'
    assert call('Skill', {'skill': 'tilelang2ascend-translator'}, agent_id='sub')['permissionDecision'] == 'allow'
    assert call('Write', {'file_path': str(tmp_path / 'input/bad.py')}, agent_id='sub')['permissionDecision'] == 'deny'
    assert call('Write', {'file_path': str(tmp_path / op / 'kernel/op.cpp')}, agent_id='sub')['permissionDecision'] == 'allow'
    assert call('Write', {'file_path': str(tmp_path / op / 'kernel/op.cpp')},
                transcript_path='/project/subagents/agent-123.jsonl')['permissionDecision'] == 'allow'
    call('Bash', {'command': 'bash evaluate_ascendc.sh OP'}, agent_id='sub')
    assert delegated.exists()
    context = call(None, hook_event_name='UserPromptSubmit')['additionalContext']
    assert 'Do not run the standalone' in context
    settings = json.loads((tmp_path / '.claude/settings.json').read_text())
    bash_groups = [g for g in settings['hooks']['PreToolUse'] if 'Bash' in g['matcher']]
    assert len(bash_groups) == 1  # No parallel execution hook bypasses the deny.


def test_judge_preserves_ranked_failure_and_hash_then_accepts_later_candidate(tmp_path):
    judge = tmp_path / 'judge'
    judge.mkdir()
    shutil.copy2(SOURCE / 'judge/judge_best.sh', judge)
    # Real driver, deterministic replacement only for the NPU evaluator.
    (judge / 'ascendc_eval_pipeline.sh').write_text('''#!/usr/bin/env bash
python3 - "$@" <<'FAKE'
import json, sys, tarfile
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = Path(args['--out_dir']); out.mkdir(parents=True, exist_ok=True)
with tarfile.open(args['--impl']) as t: kind = t.extractfile('kind').read().decode()
if kind == 'no_metrics': sys.exit(2)
ok = kind == 'pass'
d = dict(success=ok, correctness_ok=ok, ast_check_ok=kind != 'submission_missing',
         error_type=None if ok else kind, cases_passed=5 if ok else 0, cases_total=5)
(out / 'metrics.json').write_text(json.dumps(d))
(out / 'metrics_error.log').write_text('original detail: ' + kind)
print('verdict ' + kind)
sys.exit(0 if ok else 1)
FAKE
''')
    candidates = tmp_path / 'candidates'
    candidates.mkdir()
    entries = []
    def candidate(kind):
        path = candidates / f'{len(entries)}.tar.gz'
        data = kind.encode()
        with tarfile.open(path, 'w:gz') as t:
            info = tarfile.TarInfo('kind'); info.size = len(data); t.addfile(info, io.BytesIO(data))
        e = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rank=len(entries)+1)
        entries.append(e)
        return e
    first = candidate('ascendc_run_crashed')
    candidate('submission_missing')
    out = tmp_path / "judge output's"
    def run():
        (candidates / 'index.json').write_text(json.dumps(entries))
        p = subprocess.run(['bash', str(judge / 'judge_best.sh'), '--op_name', 'OP', '--out_dir', str(out)],
                           env={**os.environ, 'POLAR_T3A_CANDIDATES_DIR': str(candidates),
                                'ARTIFACTS_DIR': str(tmp_path / 'artifacts')},
                           cwd=tmp_path, capture_output=True, text=True, timeout=20)
        return p, json.loads((out / 'metrics.json').read_text())
    p, metrics = run()
    assert p.returncode == 1, p.stdout + p.stderr
    assert metrics['error_type'] == 'ascendc_run_crashed'
    assert metrics['evaluated_candidate_sha256'] == first['sha256']
    assert (out / 'metrics_error.log').read_text() == 'original detail: ascendc_run_crashed'
    assert len(metrics['judge_candidates']) == 2
    accepted = candidate('pass')
    p, metrics = run()
    assert p.returncode == 0, p.stdout + p.stderr
    assert metrics['success'] and metrics['cases_passed'] == 5
    assert metrics['evaluated_candidate_sha256'] == accepted['sha256']
    # A subsequent invocation with no metrics cannot reuse that old success.
    entries.clear(); candidate('no_metrics')
    p, metrics = run()
    assert p.returncode == 1 and metrics['success'] is False
    assert metrics['error_type'] == 'judge_no_metrics'
    # Unreadable/tampered candidates must still leave a failure result.
    entries[0]['sha256'] = 'bad-hash'
    p, metrics = run()
    assert p.returncode == 1 and metrics['error_type'] == 'submission_missing'


def test_old_failed_promotes_cannot_generate_positive_process_reward(tmp_path):
    synthesize = runpy.run_path(str(SOURCE / 'judge/t3a_process_reward.py'))['synthesize']
    stream = tmp_path / 'attempts.jsonl'
    records = [dict(script='evaluate_tilelang.sh', exit_code=0, classification='PASS', case_pass=5, case_total=5),
               dict(event='promote'), dict(event='promote')]
    stream.write_text(''.join(json.dumps(r)+'\n' for r in records))
    rejected = dict(success=False, correctness_ok=False, error_type='ascendc_compile_failed')
    result = synthesize(str(stream), rejected)
    assert result['total'] <= 0
    assert all(c['credit'] == 0 for c in result['per_attempt'] if c['event'] == 'promote')
    records[0]['script'] = 'evaluate_ascendc.sh'
    stream.write_text(''.join(json.dumps(r)+'\n' for r in records))
    accepted = synthesize(str(stream), dict(success=True, correctness_ok=True, error_type=None))
    assert accepted['total'] > 0
    assert [c['credit'] for c in accepted['per_attempt'] if c['event'] == 'promote'] == [0.06, 0.0]
    assert synthesize(str(stream), rejected)['total'] <= 0
