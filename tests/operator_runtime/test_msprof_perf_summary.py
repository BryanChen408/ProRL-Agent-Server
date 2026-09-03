from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MSPROF_SUMMARY_PATH = (
    ROOT
    / "operator_runtime_t2a"
    / "skills"
    / "ops-profiling"
    / "scripts"
    / "msprof_perf_summary.py"
)


def load_msprof_summary():
    spec = importlib.util.spec_from_file_location("t2a_msprof_perf_summary", MSPROF_SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quick_parser_counts_non_meta_device_tasks(tmp_path):
    module = load_msprof_summary()
    csv_dir = tmp_path / "mindstudio_profiler_output"
    csv_dir.mkdir()
    (csv_dir / "task_time_1.csv").write_text(
        "kernel_type,task_time(us),kernel_name\n"
        "PROFILING_ENABLE,2.0,enable\n"
        "AI_VECTOR_CORE,18.4,log\n"
        "MEMCPY,11.0,d2h\n"
        "AICPU,7.0,other_device_task\n"
        "PROFILING_DISABLE,3.0,disable\n"
        "TASK_TIMEOUT_SET,4.0,timeout\n",
        encoding="utf-8",
    )

    duration, kernel_name, error = module._parse_msprof_duration_quick(str(tmp_path))

    assert duration == 36.4
    assert kernel_name == "multiple_kernels"
    assert error is None


def test_quick_parser_ignores_meta_events(tmp_path):
    module = load_msprof_summary()
    csv_dir = tmp_path / "mindstudio_profiler_output"
    csv_dir.mkdir()
    (csv_dir / "task_time_1.csv").write_text(
        "kernel_type,task_time(us),kernel_name\n"
        "PROFILING_ENABLE,2.0,enable\n"
        "PROFILING_DISABLE,3.0,disable\n"
        "TASK_TIMEOUT_SET,4.0,timeout\n",
        encoding="utf-8",
    )

    duration, kernel_name, error = module._parse_msprof_duration_quick(str(tmp_path))

    assert duration is None
    assert kernel_name is None
    assert error == "no task_time or api_statistic csv found"


def test_resolve_input_groups_keeps_get_inputs_as_one_case():
    module = load_msprof_summary()
    provider = SimpleNamespace(get_inputs=lambda: ["tensor", 3, (2, 4)])

    assert module._resolve_input_groups(provider) == [["tensor", 3, (2, 4)]]


def test_resolve_input_groups_preserves_multiple_cases():
    module = load_msprof_summary()
    provider = SimpleNamespace(get_input_groups=lambda: [["a"], ["b"]])

    assert module._resolve_input_groups(provider) == [["a"], ["b"]]


def test_bind_case_splits_flat_keyword_only_values_by_reference_signature():
    module = load_msprof_summary()

    class Model:
        def forward(self, x, scale=1.0, *, mode, keepdim=False):
            return x, scale, mode, keepdim

    args, kwargs = module._bind_case(Model, ["x", 0.5, "mean", True])

    assert args == ("x", 0.5)
    assert kwargs == {"mode": "mean", "keepdim": True}


def test_bind_case_accepts_named_mapping():
    module = load_msprof_summary()

    class Model:
        def forward(self, x, *, mode):
            return x, mode

    args, kwargs = module._bind_case(Model(), {"mode": "sum", "x": "tensor"})

    assert args == ()
    assert kwargs == {"mode": "sum", "x": "tensor"}


def test_invoke_model_does_not_swallow_internal_type_error():
    module = load_msprof_summary()

    class Model:
        def forward(self, value):
            raise TypeError("failure inside model body")

        __call__ = forward

    with pytest.raises(TypeError, match="inside model body"):
        module._invoke_model(Model(), [1])


def test_json_scalar_types_are_not_coerced_to_float():
    module = load_msprof_summary()

    assert module._jsonl_scalar_value({"dtype": "tuple", "value": [1, 2]}) == (1, 2)
    assert module._jsonl_scalar_value({"dtype": "str", "value": "mean"}) == "mean"
    assert module._jsonl_scalar_value({"dtype": "list", "value": [3, 4]}) == [3, 4]
    assert module._jsonl_scalar_value({"dtype": "int64", "value": None}) is None


def test_only_explicit_zero_dimension_is_an_empty_tensor_case():
    module = load_msprof_summary()

    assert not module._case_has_empty_tensor({
        "inputs": [{"type": "tensor", "shape": []}]
    })
    assert not module._case_has_empty_tensor({
        "inputs": [{"type": "tensor", "required": False, "shape": None}]
    })
    assert module._case_has_empty_tensor({
        "inputs": [{"type": "tensor", "shape": [4, 0, 8]}]
    })


def test_optional_json_tensor_with_null_shape_falls_back_to_none():
    module = load_msprof_summary()
    code = module._serialize_jsonl_inputs({
        "inputs": [{
            "name": "bias", "type": "tensor", "required": False,
            "dtype": "float16", "shape": None,
        }]
    })

    assert "fallback_case['bias'] = None" in code


def test_generated_wrapper_prefers_provider_and_binds_against_reference(tmp_path):
    module = load_msprof_summary()
    case = {
        "inputs": [
            {"name": "x", "type": "tensor", "dtype": "float16", "shape": [2, 3]},
            {"name": "repeats", "type": "attr", "dtype": "tuple", "value": [2, 1]},
            {"name": "mode", "type": "attr", "dtype": "str", "value": "mean"},
        ]
    }
    cfg = module._WrapperConfig(tmp_path, 0, "ascendc", 17, 0, 2, case)

    wrapper = module._generate_wrapper_script(cfg)

    compile(wrapper, "generated_wrapper.py", "exec")
    assert "torch.manual_seed(17)" in wrapper
    assert "_ref_mod.get_input_groups()" in wrapper
    assert "input_case = _ref_mod.get_inputs()" in wrapper
    assert "fallback_case['repeats'] = (2, 1)" in wrapper
    assert "fallback_case['mode'] = 'mean'" in wrapper
    assert "_bind_case(_contract_cls, input_case)" in wrapper
    assert "model(*_call_args, **_call_kwargs)" in wrapper
    assert "offsets" not in wrapper.lower()


def test_model_provider_owns_case_count_over_agent_json(tmp_path):
    module = load_msprof_summary()
    (tmp_path / "model.py").write_text(
        "def get_inputs():\n    return [1, 2]\n",
        encoding="utf-8",
    )
    (tmp_path / "agent_created.json").write_text(
        '{"inputs": [{"name": "x", "type": "tensor", "shape": [0]}]}\n'
        '{"inputs": [{"name": "x", "type": "tensor", "shape": [0]}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "agent_created.json").write_text(
        '{"inputs": []}\n{"inputs": []}\n',
        encoding="utf-8",
    )

    cases, source = module._load_compare_cases(tmp_path)

    assert cases == [None]
    assert source == "model:get_inputs/get_input_groups"


def test_matching_json_is_metadata_for_provider_cases(tmp_path):
    module = load_msprof_summary()
    (tmp_path / "model.py").write_text(
        "def get_input_groups():\n    return [[1], [2]]\n",
        encoding="utf-8",
    )
    (tmp_path / "cases.json").write_text(
        '{"inputs": [{"name": "x", "type": "scalar", "value": 1}]}\n'
        '{"inputs": [{"name": "x", "type": "scalar", "value": 2}]}\n',
        encoding="utf-8",
    )

    cases, source = module._load_compare_cases(tmp_path)

    assert len(cases) == 2
    assert cases[1]["inputs"][0]["value"] == 2
    assert source.endswith(":metadata")


def test_pipeline_case_cache_calls_provider_once_and_reuses_exact_values(tmp_path, monkeypatch):
    module = load_msprof_summary()
    (tmp_path / "model.py").write_text(
        "CALLS = 0\n"
        "class Model:\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "def get_input_groups():\n"
        "    global CALLS\n"
        "    CALLS += 1\n"
        "    return [[[CALLS, 2, 3]], [[4, 5]]]\n",
        encoding="utf-8",
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = ()
    fake_torch.device = lambda name: name
    fake_torch.manual_seed = lambda seed: None
    fake_torch.save = lambda value, path: Path(path).write_bytes(pickle.dumps(value))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    cache_dir = tmp_path / "case_cache"

    cases, cache_paths, source = module._materialize_compare_cases(
        tmp_path, cache_dir, seed=17
    )

    assert len(cases) == len(cache_paths) == 2
    assert "pipeline-cache" in source
    assert not module._case_has_empty_tensor(cases[0])
    assert sys.modules["ref_for_case_materialization"].CALLS == 1
    first_ref = pickle.loads(cache_paths[0].read_bytes())
    first_asc = pickle.loads(cache_paths[0].read_bytes())
    assert first_ref == first_asc
    first_ref[0][0] = 0
    assert first_ref != first_asc


def test_cached_wrapper_never_reinvokes_provider(tmp_path):
    module = load_msprof_summary()
    cache_path = tmp_path / "case_000.pt"
    cache_path.write_bytes(b"cached-case")
    cfg = module._WrapperConfig(
        tmp_path, 0, "ascendc", 17, 0, 2,
        jsonl_case={"inputs": []}, case_cache_path=cache_path,
    )

    wrapper = module._generate_wrapper_script(cfg)

    compile(wrapper, "generated_cached_wrapper.py", "exec")
    assert f"torch.load({str(cache_path)!r}" in wrapper
    assert "_ref_mod.get_input_groups()" not in wrapper
    assert "_ref_mod.get_inputs()" not in wrapper


def test_extract_app_crash_handles_msprof_zero_exit_pattern():
    module = load_msprof_summary()
    output = "Traceback (most recent call last):\nValueError: invalid generated case\n"

    assert module._extract_app_crash("", output) == "ValueError: invalid generated case"


def test_quick_warmup_uses_one_python_process_for_all_rounds(tmp_path, monkeypatch):
    module = load_msprof_summary()
    calls = []

    def fake_run(cmd, **_kwargs):
        script = Path(cmd[-1])
        calls.append((cmd[0], script.name, script.read_text(encoding="utf-8")))
        if cmd[0] == "msprof":
            output_dir = Path(next(x.split("=", 1)[1] for x in cmd if x.startswith("--output=")))
            (output_dir / "PROF_GROUP_1").mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module, "_parse_msprof_duration_quick", lambda _path: (8.0, "kernel", None)
    )
    out_dir = tmp_path / f"op_{tmp_path.name}"
    args = SimpleNamespace(retry=0, repeats=1, warmup=3, seed=17)

    duration, error, _ = module._measure_one_impl_quick(
        module._MeasureInput(out_dir, 0, "reference", args, 0)
    )

    assert duration == 8.0 and error is None
    assert len(calls) == 2
    assert calls[0][1] == "_warmup.py" and "range(2)" in calls[0][2]
    assert calls[1][0] == "msprof" and calls[1][1] == "_wrapper.py"
    assert "range(0)" in calls[1][2]


def test_performance_target_is_uniformly_1_1x():
    module = load_msprof_summary()
    assert module.PERF_TARGET_SPEEDUP == 1.1

    rows = module._build_batch_md_summary_table([
        {"name": "at_target", "data": {
            "n_cases_total": 1, "n_cases_valid": 1,
            "geomean_speedup": 1.1, "mean_speedup": 1.1,
        }},
        {"name": "below_target", "data": {
            "n_cases_total": 1, "n_cases_valid": 1,
            "geomean_speedup": 1.099, "mean_speedup": 1.099,
        }},
    ])

    assert any("| at_target |" in row and "| ✅ |" in row for row in rows)
    assert any("| below_target |" in row and "| ⚠️ |" in row for row in rows)


def test_historical_trace_row_is_normalized_to_1_1x_target():
    module = load_msprof_summary()
    old_row = (
        "| 2 | 1 | op | vector | ✅ | ✅ | 1.0 | 0.9 | 1.11 | "
        "成功 | 是 | 是 | 是 |"
    )

    normalized = module._normalize_trace_table_row(old_row)

    assert normalized.count("|") == 13
    assert normalized.endswith("| 是 |")
    assert "1.11" in normalized
