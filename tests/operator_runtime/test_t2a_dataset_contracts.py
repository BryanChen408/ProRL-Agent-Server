"""CPU regressions for the shared cudaLLM / NPUKernelBench evaluation contract."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'operator_runtime_t2a'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('provider', [
    'def get_inputs(): return [torch.ones(1, 3)]',
    'def get_input_groups(): return [{"x": torch.ones(1, 3)}, {"x": torch.zeros(1, 3)}]',
])
def test_random_parameters_and_reference_owned_init(tmp_path, monkeypatch, provider):
    verifier = load('t2a_contract_verifier', RUNTIME / 'skills/tilelang2ascend-translator/scripts/verification_ascendc.py')
    monkeypatch.setattr(verifier, '_get_device', lambda: torch.device('cpu'))
    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
    def forward(self, x): return self.linear(x)
def get_init_inputs(): return [3]
''' + provider + '\n'
    (tmp_path / 'model.py').write_text(source)
    candidate = source.replace('class Model(', 'class ModelNew(').replace('return [3]', 'return [999]')
    (tmp_path / 'model_new_ascendc.py').write_text(candidate)
    report = verifier._run_verification(str(tmp_path))
    assert report['ok'], report
    # A changed computation must still fail under identical initialization.
    (tmp_path / 'model_new_ascendc.py').write_text(candidate.replace('return self.linear(x)', 'return self.linear(x) + 10'))
    assert not verifier._run_verification(str(tmp_path))['ok']


@pytest.mark.parametrize('behavior,code', [('x + 1', 0), ('torch.zeros_like(x)', 1), ('1 / 0', 3)])
@pytest.mark.parametrize('runtime', ['operator_runtime_t2a/tools', 'operator_runtime_t3a/judge'])
def test_detector_uses_mapping_contract_and_distinguishes_errors(tmp_path, behavior, code, runtime):
    source = '''import torch
class Model(torch.nn.Module):
    def forward(self, *, x): return x + 1
def get_input_groups(): return [{'x': torch.ones(2)}, {'x': torch.zeros(2)}]
def get_init_inputs(): return []
'''
    (tmp_path / 'model.py').write_text(source)
    (tmp_path / 'model_new_ascendc.py').write_text(source.replace('class Model(', 'class ModelNew(').replace('return x + 1', 'return ' + behavior))
    detector = ROOT / runtime / 'detect_stateful_impl.py'
    # Run the real CLI; disable accelerator discovery for this CPU contract test.
    harness = "import runpy,sys,types,torch; sys.modules['torch_npu']=types.ModuleType('torch_npu'); torch.npu=types.SimpleNamespace(is_available=lambda:False); torch.cuda.is_available=lambda:False; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable, '-c', harness, str(detector), str(tmp_path)], text=True, capture_output=True)
    assert result.returncode == code, result.stdout + result.stderr
    assert ('PASS:', 'FAIL:', '', 'ERROR:')[code] in result.stdout + result.stderr


@pytest.mark.parametrize('code,text,expected', [
    (0, '[stateful-detect] PASS: varying outputs', ''),
    (1, '[stateful-detect] FAIL: constant output', 'stateful_impl_detected'),
    (1, 'RuntimeError: 507034', 'stateful_detector_error'),
    (3, '[stateful-detect] ERROR: invalid inputs', 'stateful_detector_error'),
    (2, '[stateful-detect] SKIP: one case', ''),
    (2, 'Python invocation failed', 'stateful_detector_error'),
])
def test_pipeline_detector_failure_classification(tmp_path, code, text, expected):
    source = (RUNTIME / 'tools/ascendc_eval_pipeline.sh').read_text()
    stage = source[source.index('DETECT="${_SCRIPT_DIR}/detect_stateful_impl.py"'):source.index('\n# Step3')]
    harness = 'run_npu_phase() { printf "%s\\n" "$DETECT_TEXT"; return "$DETECT_CODE"; }\nwrite_metrics() { printf "error_type=%s\\n" "$9"; }\nfail_hint() { :; }\n'
    env = {**os.environ, '_SCRIPT_DIR': str(RUNTIME / 'tools'), 'WORK': str(tmp_path), 'TASK_DIR': str(tmp_path), 'OUT_DIR': str(tmp_path), 'PY_BIN': sys.executable, 'DETECT_CODE': str(code), 'DETECT_TEXT': text}
    result = subprocess.run(['bash', '-c', harness + stage + '\nexit 0\n'], env=env, text=True, capture_output=True)
    assert result.returncode == bool(expected), result.stdout + result.stderr
    if expected:
        assert f'error_type={expected}' in result.stdout


def test_pool_mixed_parameters_and_ub_allocation(tmp_path):
    prep = load('t2a_contract_prepare', RUNTIME / 'runtime/prepare_operator_workdir.py')
    task = tmp_path / 'model.py'
    task.write_text('''import torch
class Model(torch.nn.Module):
    def forward(self, x, y, kernel_size=2, stride=None): return x
''')
    cases = [{'inputs': [{'name': 'kernel_size', 'type': 'attr', 'value': k}, {'name': 'stride', 'type': 'attr', 'value': s}]} for k,s in [(2, None), ([2,2,2], [1,1,1]), (3, 2)]]
    task.with_suffix('.json').write_text(''.join(json.dumps(c)+'\n' for c in cases))
    sig = prep._extract_op_signature(task, task.with_suffix('.json'))
    ordered, schema_args, cpp_args = prep._ordered_params(sig)
    assert 'int[] kernel_size=[2]' in schema_args
    assert 'int[]? stride=None' in schema_args
    op = 't2a_mixed_pool_contract'
    lib = torch.library.Library('npu', 'FRAGMENT')
    lib.define(f'{op}({schema_args}) -> Tensor')
    seen = []
    def impl(x, y, kernel_size=[2], stride=None):
        seen.append((kernel_size,stride))
        return x
    lib.impl(op, impl, 'CompositeExplicitAutograd')
    namespace = {'__file__': str(tmp_path / 'model_new_ascendc.py')}
    exec(prep._render_model_new(op,sig),namespace)
    model = namespace['ModelNew']()
    for k,s in [(2,None),([2,2,2],[1,1,1]),(3,2)]:
        model(torch.ones(1),torch.ones(1),k,s)
    assert seen == [([2],None),([2,2,2],[1,1,1]),([3],[2])]
    host = prep._render_op_host_cpp(op,sig,ordered,cpp_args)
    kernel = prep._render_op_kernel_cpp(op,'Mixed',sig,ordered)
    assert '_dtypeSize * 6;' in host
    assert kernel.count('pipe.InitBuffer(') == 3


def test_benchmark_wrappers_use_same_reference_initialization(tmp_path):
    perf = load('t2a_contract_perf', RUNTIME / 'skills/ops-profiling/scripts/msprof_perf_summary.py')
    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width,width)
    def forward(self, x): return self.linear(x)
def get_init_inputs(): return [3]
def get_inputs(): return [torch.ones(1,3)]
'''
    (tmp_path/'model.py').write_text(source)
    (tmp_path/'model_new_ascendc.py').write_text(source.replace('class Model(', 'class ModelNew(').replace('return [3]', 'return [999]'))
    states=[]
    for impl in ('reference','ascendc'):
        cfg = perf._WrapperConfig(out_dir=tmp_path,case_idx=0,impl=impl,seed=17,device_id=0,warmup=0)
        wrapper=perf._generate_wrapper_script(cfg)
        # Exercise generated Python and real random weights; only device operations are substituted.
        wrapper=wrapper.replace('device = torch.device("npu")','device = torch.device("cpu")').replace('torch.npu.synchronize()','pass')
        namespace={}
        exec(compile(wrapper,'benchmark_wrapper.py','exec'),namespace)
        states.append(namespace['model'].state_dict())
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])


@pytest.mark.parametrize('agent_side', [False, True])
@pytest.mark.parametrize('flags', [[], ['--incremental']])
def test_incremental_defaults_to_agent_only(tmp_path, agent_side, flags):
    source = (RUNTIME / 'tools/ascendc_eval_pipeline.sh').read_text()
    prologue = source[source.index('OP_NAME=""'):source.index('\nSTATE_DIR=')]
    if agent_side:
        (tmp_path / 'OP').mkdir()
    result = subprocess.run(
        ['bash', '-uc', prologue + '\nprintf "%s" "$INCREMENTAL"',
         'pipeline', '--op_name', 'OP', *flags],
        env={**os.environ, 'WORK_ROOT': str(tmp_path)},
        text=True, capture_output=True, check=True,
    )
    assert result.stdout == ('1' if agent_side else '0')


def test_incremental_uses_native_builder_and_invalidates_deleted_inputs(tmp_path):
    import shutil
    import tarfile
    if not all(shutil.which(tool) for tool in ('cmake','g++')):
        pytest.skip('native CMake/C++ required')
    source=(RUNTIME/'tools/ascendc_eval_pipeline.sh').read_text()
    snippet=source.split("<<'PYINCREMENTAL'\n",1)[1].split('\nPYINCREMENTAL',1)[0]
    prepare=source[source.index('WORK="$OUT_DIR/work"'):source.index('\nTASK_SRC="${TASK_FILE:-input/${OP_NAME}.py}"')]
    builder_path=RUNTIME/'skills/tilelang2ascend-translator/scripts/build_ascendc.py'
    builder=load('t2a_native_builder',builder_path)
    work,previous,payload=tmp_path/'work',tmp_path/'work.previous',tmp_path/'payload'
    kernel=payload/'kernel'
    kernel.mkdir(parents=True)
    (kernel/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\nproject(probe LANGUAGES CXX)\nadd_executable(probe main.cpp helper.cpp)\n')
    (kernel/'main.cpp').write_text('#include <iostream>\n#include "value.h"\nint main() { std::cout << VALUE; }\n')
    (kernel/'helper.cpp').write_text('int helper() { return 1; }\n')
    (kernel/'value.h').write_text('#define VALUE 1\n')
    (payload/'model_new_ascendc.py').write_text('# wrapper v1\n')
    (kernel/'build').mkdir()
    (kernel/'build/untrusted.o').write_text('must never enter the compiler cache')
    def cycle():
        task=work/'OP'
        # Restoring a candidate can bring older mtimes than the cached objects.
        for p in payload.rglob('*'):
            if p.is_file():os.utime(p,(1,1))
        archive=tmp_path/'candidate.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:tar.add(payload,arcname='OP')
        env={**os.environ,'OUT_DIR':str(tmp_path),'IMPL_FILE':str(archive),'SRC_DIR':str(payload),'OP_NAME':'OP','INCREMENTAL':'1'}
        prepared=subprocess.run(['bash','-c',prepare],env=env,text=True,capture_output=True)
        assert prepared.returncode==0,prepared.stdout+prepared.stderr
        assert not (task/'kernel/build').exists()
        result=subprocess.run([sys.executable,'-',str(task),str(work),str(previous),str(builder_path)],input=snippet,text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        build=builder.build(str(task),'test','Release',clean=False)
        value=subprocess.check_output([str(build/'probe')],text=True)
        objects={p.name:p.stat().st_mtime_ns for p in build.rglob('*.cpp.o')}
        return value,objects,result.stdout
    value,initial,_=cycle()
    assert value=='1'
    (payload/'model_new_ascendc.py').write_text('# wrapper v2\n')
    value,unchanged,log=cycle()
    assert value=='1' and unchanged==initial and 'reuse local build' in log
    (kernel/'value.h').write_text('#define VALUE 2\n')
    value,changed,_=cycle()
    assert value=='2' and changed['main.cpp.o']!=initial['main.cpp.o']
    assert changed['helper.cpp.o']==initial['helper.cpp.o']
    (kernel/'main.cpp').write_text('#include <iostream>\n#include "value.h"\nint main() { std::cout << VALUE + 1; }\n')
    assert cycle()[0]=='3'
    (kernel/'value.h').unlink()
    (kernel/'main.cpp').write_text('#include <iostream>\nint main() { std::cout << 4; }\n')
    value,_,log=cycle()
    assert value=='4' and 'clean build' in log
    assert not (work/'OP/kernel/value.h').exists()
    # A new build configuration also drops the local cache.
    (kernel/'CMakeLists.txt').write_text((kernel/'CMakeLists.txt').read_text()+'\n# configuration changed\n')
    assert 'clean build' in cycle()[2]


def test_detector_reference_fallback_never_moves_candidate_to_cpu(tmp_path, monkeypatch):
    from types import SimpleNamespace
    detector=load('t2a_detector_fallback',RUNTIME/'tools/detect_stateful_impl.py')
    source='''import torch
class Model(torch.nn.Module):
    def to(self, device):
        self.where=str(device)
        return self
    def forward(self, x):
        if self.where != 'cpu': raise RuntimeError('reference unsupported on NPU')
        return x+1
def get_input_groups(): return [{'x':torch.ones(2)},{'x':torch.zeros(2)}]
'''
    candidate=source.replace('class Model(', 'class ModelNew(').replace("if self.where != 'cpu': raise RuntimeError('reference unsupported on NPU')", "assert self.where == 'npu', 'candidate must remain on NPU'")
    (tmp_path/'model.py').write_text(source)
    (tmp_path/'model_new_ascendc.py').write_text(candidate)
    original_load=detector._load
    devices=[]
    def load_cpu_utilities(path,name):
        module=original_load(path,name)
        if name=='_sd_input_utils':
            module._seed_model=lambda *args:torch.manual_seed(0)
            def move(value,device):
                devices.append(device)
                return value
            module._move=move
        return module
    monkeypatch.setattr(detector,'_load',load_cpu_utilities)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace())
    monkeypatch.setattr(torch,'npu',SimpleNamespace(is_available=lambda:True),raising=False)
    assert detector.main([str(tmp_path)])==0
    assert devices[-2:]==['npu','npu'] and 'cpu' in devices
