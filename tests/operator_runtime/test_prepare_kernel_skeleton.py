"""prepare_operator_workdir 骨架签名生成的单元测试。

覆盖签名提取/生成的关键特征(数据来自 cudallm189 / KernelBench / NPUKernelBench 三个
数据集的实测分布):单 tensor、init 标量、双 tensor、tuple 返回、Optional、tensor_list、
kwonly、*args 降级、不可解析 init 值降级、已存在不覆盖、解析失败回退模板。
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
CANONICAL = REPO / "operator_runtime_t2a"

spec = importlib.util.spec_from_file_location(
    "prepare_operator_workdir", CANONICAL / "runtime" / "prepare_operator_workdir.py"
)
prep = importlib.util.module_from_spec(spec)
sys.modules["prepare_operator_workdir"] = prep  # dataclass 解析注解时要查 sys.modules
spec.loader.exec_module(prep)


def _write_model(tmp_path: Path, src: str, case: dict | None = None) -> tuple[Path, Path]:
    import json as _json
    task = tmp_path / "input" / "my_op.py"
    task.parent.mkdir(parents=True, exist_ok=True)
    task.write_text(src, encoding="utf-8")
    json_path = None
    if case is not None:
        json_path = tmp_path / "input" / "my_op.json"
        json_path.write_text(_json.dumps(case) + "\n", encoding="utf-8")
    return task, json_path


def _instantiate(tmp_path: Path, task: Path, json_path, op: str = "my_op"):
    workdir = tmp_path / "wd"
    workdir.mkdir(exist_ok=True)
    prep._instantiate_kernel_skeleton(CANONICAL, workdir, op, task, json_path)
    return workdir / op


def _construct(model_new_src: str, *init_args, **init_kwargs):
    """执行生成的 model_new 源码并构造 ModelNew(校验 harness 契约的构造侧)。"""
    import torch  # noqa: F401
    ns: dict = {"__file__": "/tmp/fake_op/model_new_ascendc.py"}
    exec(compile(model_new_src, "<model_new>", "exec"), ns)
    return ns["ModelNew"](*init_args, **init_kwargs)


# ---------------------------------------------------------------- fixtures

SINGLE = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.abs(x)
def get_inputs():
    return [torch.randn(4)]
def get_init_inputs():
    return []
"""

INIT_INT = """
import torch
class Model(torch.nn.Module):
    def __init__(self, subtract_value):
        super().__init__()
        self.subtract_value = subtract_value
    def forward(self, x):
        return torch.subtract(x, self.subtract_value)
def get_inputs():
    return [torch.tensor([1, 2])]
def get_init_inputs():
    return [5]
"""

INIT_FLOAT = """
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope)
    def forward(self, x):
        return self.act(torch.sqrt(x))
negative_slope = 0.2
def get_init_inputs():
    return [negative_slope]
"""

TWO_TENSOR = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.mul(x, y)
def get_inputs():
    return [torch.randn(4), torch.randn(4)]
"""

TUPLE_RET = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.ne(x, 0), torch.isinf(x)
def get_inputs():
    return [torch.randn(4)]
"""

LAYERNORM_LIKE = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, normalized_shape: list,
                weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias)
"""

LAYERNORM_CASE = {"inputs": [
    {"name": "x", "type": "tensor", "required": True, "dtype": "float32"},
    {"name": "normalized_shape", "type": "attr", "required": True, "dtype": "int", "value": [4]},
    {"name": "weight", "type": "tensor", "required": False, "dtype": "float32"},
    {"name": "bias", "type": "tensor", "required": False, "dtype": "float32"},
]}

CAT_LIKE = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, tensors: list, dim: int = 0):
        return torch.cat(tensors, dim=dim)
"""

CAT_CASE = {"inputs": [
    {"name": "tensors", "type": "tensor_list", "required": True, "dtype": "float32"},
    {"name": "dim", "type": "attr", "required": False, "dtype": "int", "value": 0},
]}

SCALAR_ONLY = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, rows: int, cols: int = 4):
        return torch.ones((rows, cols))
"""

KWONLY = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *, keep_prob=1.0, sparse_mode=0):
        return torch.dropout(x, 1 - keep_prob, True)
"""

VARARG_INIT = """
import torch
class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return torch.abs(x)
def get_inputs():
    return [torch.randn(4)]
"""

UNPARSABLE_INIT = """
import torch
class Model(torch.nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        self.dim_size = dim_size
    def forward(self, x):
        return torch.sub(x, self.dim_size)
def get_inputs():
    return [torch.randn(4, 8)]
def get_init_inputs():
    return [get_inputs()[0].shape[1]]
"""


# ---------------------------------------------------------------- tests

def test_single_tensor_no_init(tmp_path):
    task, js = _write_model(tmp_path, SINGLE)
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert 'm.def("my_op(Tensor x) -> Tensor")' in reg
    mn = (out / "model_new_ascendc.py").read_text()
    model = _construct(mn)                      # ModelNew() 可构造
    assert list(inspect.signature(model.forward).parameters) == ["x"]
    assert "torch.ops.npu.my_op(x)" in mn
    host = (out / "kernel" / "op_host" / "my_op.cpp").read_text()
    assert "at::Tensor my_op(const at::Tensor &x)" in host


def test_init_scalar_int(tmp_path):
    task, js = _write_model(tmp_path, INIT_INT)
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert 'm.def("my_op(Tensor x, int subtract_value) -> Tensor")' in reg
    mn = (out / "model_new_ascendc.py").read_text()
    model = _construct(mn, 5)                   # cls(*get_init_inputs()) 契约
    assert model.subtract_value == 5
    assert "torch.ops.npu.my_op(x, self.subtract_value)" in mn
    with pytest.raises(TypeError):              # 与 Model 一致:缺参要报错
        _construct(mn)


def test_init_float_from_module_const(tmp_path):
    task, js = _write_model(tmp_path, INIT_FLOAT)
    sig = prep._extract_op_signature(task, js)
    assert sig.init[0].kind == "float"          # 模块常量 negative_slope=0.2 被解析
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert "float negative_slope" in reg
    kern = (out / "kernel" / "op_kernel" / "my_op_kernel.cpp").read_text()
    assert "float negative_slope" in kern       # 标量接进 kernel 入口


def test_two_tensor_forward(tmp_path):
    task, js = _write_model(tmp_path, TWO_TENSOR)
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert 'm.def("my_op(Tensor x, Tensor y) -> Tensor")' in reg
    kern = (out / "kernel" / "op_kernel" / "my_op_kernel.cpp").read_text()
    # kernel 内部按位置合成命名(in0/in1):输入叫 y 也不能撞输出 y/成员 yGm
    gsig = next(l for l in kern.splitlines() if "GM_ADDR in0" in l)   # 入口签名行
    assert gsig.count("GM_ADDR y") == 1 and "GM_ADDR in1" in gsig     # 输出唯一、双输入俱在
    assert kern.count("in0Gm;") == 1 and kern.count("in1Gm;") == 1 and kern.count("yGm;") == 1
    assert "inQueue0" in kern and "inQueue1" in kern     # 双输入各建 queue
    host = (out / "kernel" / "op_host" / "my_op.cpp").read_text()
    assert "x, y, _output" in host                       # EXEC_KERNEL_CMD 双输入都接


def test_tuple_return(tmp_path):
    task, js = _write_model(tmp_path, TUPLE_RET)
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert "-> (Tensor, Tensor)" in reg
    ops_h = (out / "kernel" / "ops.h").read_text()
    assert "std::tuple<at::Tensor, at::Tensor>" in ops_h


def test_optional_tensor_from_json(tmp_path):
    task, js = _write_model(tmp_path, LAYERNORM_LIKE, LAYERNORM_CASE)
    sig = prep._extract_op_signature(task, js)
    kinds = {p.name: p.kind for p in sig.fwd}
    assert kinds["x"] == "tensor"
    assert kinds["normalized_shape"] == "int[]"
    assert kinds["weight"] == "tensor?" and kinds["bias"] == "tensor?"
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert "Tensor? weight" in reg
    mn = (out / "model_new_ascendc.py").read_text()
    model = _construct(mn)
    sig_fwd = inspect.signature(model.forward)
    assert list(sig_fwd.parameters)[:2] == ["x", "normalized_shape"]


def test_tensor_list_from_json(tmp_path):
    task, js = _write_model(tmp_path, CAT_LIKE, CAT_CASE)
    sig = prep._extract_op_signature(task, js)
    kinds = {p.name: p.kind for p in sig.fwd}
    assert kinds["tensors"] == "tensor[]"
    assert kinds["dim"] == "int"
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert "Tensor[] tensors" in reg
    assert "TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)" in reg


def test_scalar_only_signature_uses_catchall_dispatch(tmp_path):
    """无 Tensor 实参时 dispatcher 无法推导 PrivateUse1，注册必须按签名属性切换。"""
    task, js = _write_model(tmp_path, SCALAR_ONLY)
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert 'm.def("my_op(int rows, int cols=4) -> Tensor")' in reg
    assert "TORCH_LIBRARY_IMPL(npu, CatchAll, m)" in reg
    assert "TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)" not in reg
    host = (out / "kernel" / "op_host" / "my_op.cpp").read_text()
    assert "该算子没有 tensor 输入" in host


def test_kwonly_mirrored_with_defaults(tmp_path):
    task, js = _write_model(tmp_path, KWONLY)
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert "float keep_prob=1.0" in reg and "int sparse_mode=0" in reg
    mn = (out / "model_new_ascendc.py").read_text()
    model = _construct(mn)
    sig_fwd = inspect.signature(model.forward)
    assert sig_fwd.parameters["keep_prob"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig_fwd.parameters["keep_prob"].default == 1.0
    assert "keep_prob=keep_prob" in mn


def test_vararg_init_loose_fallback(tmp_path):
    task, js = _write_model(tmp_path, VARARG_INIT)
    sig = prep._extract_op_signature(task, js)
    assert sig.init_storage == "args_kwargs"
    assert sig.notes                                 # 有降级说明
    out = _instantiate(tmp_path, task, js)
    mn = (out / "model_new_ascendc.py").read_text()
    model = _construct(mn, 1, 2, k=3)                # 任意参数都能构造(放宽兜底)
    assert model._init_args == (1, 2) and model._init_kwargs == {"k": 3}
    assert "TODO" in mn


def test_unparseable_init_value_excluded_with_todo(tmp_path):
    task, js = _write_model(tmp_path, UNPARSABLE_INIT)
    sig = prep._extract_op_signature(task, js)
    assert sig.init[0].in_op is False                # 不进 op 签名
    assert any("dim_size" in n for n in sig.notes)
    out = _instantiate(tmp_path, task, js)
    reg = (out / "kernel" / "register.cpp").read_text()
    assert "dim_size" not in reg                     # 签名里没有它
    mn = (out / "model_new_ascendc.py").read_text()
    model = _construct(mn, 8)                        # 但构造仍满足 harness 契约
    assert model.dim_size == 8
    assert "TODO" in mn


def test_existing_files_not_overwritten(tmp_path):
    task, js = _write_model(tmp_path, INIT_INT)
    out = _instantiate(tmp_path, task, js)
    marker = "# agent 手改的内容"
    (out / "model_new_ascendc.py").write_text(marker, encoding="utf-8")
    prep._instantiate_kernel_skeleton(CANONICAL, tmp_path / "wd", "my_op", task, js)
    assert (out / "model_new_ascendc.py").read_text() == marker


def test_extraction_failure_falls_back_to_template(tmp_path):
    task, js = _write_model(tmp_path, "import torch\n# 没有 Model 类\n")
    out = _instantiate(tmp_path, task, js)
    mn = (out / "model_new_ascendc.py").read_text()
    assert "class ModelNew" in mn                    # 模板兜底(单 tensor 形态)
    assert "def forward(self, x: torch.Tensor)" in mn


# --------------------------------------------------------------------------
# schema 字面量渲染回归(2026-08-10 实测:三类非法字面量 → dlopen 即崩,op 永不注册)
#   bug#1 bool 默认值被小写成 false → torch 只认 True/False("invalid numeric default value")
#   bug#2 __init__ 分支 None 默认值没包 optional → impl 非可选 int64_t 接 None 调用崩
#   bug#3 list 默认值渲成 Python 元组 (1, 1)/(1,) → torch 只认方括号 [1, 1]/[1]
#   bug#4 int[] 的 None 默认值没包 optional → `int[] size=None` 无法 parse_schema
# --------------------------------------------------------------------------

def _mdef_schema(out: Path) -> str:
    reg = (out / "kernel" / "register.cpp").read_text()
    line = next(l for l in reg.splitlines() if "m.def(" in l)
    return line.split('m.def("', 1)[1].split('")', 1)[0]


def _assert_schema_parses(schema: str) -> None:
    torch._C.parse_schema(schema)  # 抛异常即测试失败


CONV_LIKE = """
import torch
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    def forward(self, x):
        return self.conv(x)
def get_inputs():
    return [torch.randn(1, 2, 4, 4)]
def get_init_inputs():
    return [2, 4, 3, 1, 0, True]
"""

POOL_LIKE = """
import torch
class Model(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.maxpool = torch.nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    def forward(self, x):
        return self.maxpool(x)
def get_inputs():
    return [torch.randn(1, 2, 4, 4, 4)]
def get_init_inputs():
    return [3, 2, 1, 1]
"""

INTERPOLATE_OPTIONAL_SIZE_LIKE = """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, size=None, antialias=False):
        return torch.nn.functional.interpolate(
            x, size=size, mode="bilinear", antialias=antialias
        )
"""

INTERPOLATE_OPTIONAL_SIZE_CASE = {"inputs": [
    {"name": "x", "type": "tensor", "required": True, "dtype": "float32"},
    {"name": "size", "type": "attr", "required": False, "value": [8, 8]},
    {"name": "antialias", "type": "attr", "required": False, "value": False},
]}

CONV3D_TUPLE_LIKE = """
import torch
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), groups=1, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding,
                                    groups=groups, bias=bias)
    def forward(self, x):
        return self.conv(x)
def get_inputs():
    return [torch.randn(1, 2, 4, 4, 4)]
def get_init_inputs():
    return [2, 4, 3]
"""

SINGLE_ELEM_TUPLE_LIKE = """
import torch
class Model(torch.nn.Module):
    def __init__(self, num_parameters=1, init=0.25, alpha_shape=(1,)):
        super().__init__()
        self.act = torch.nn.PReLU(num_parameters, init)
        self.alpha_shape = alpha_shape
    def forward(self, x):
        return self.act(x)
def get_inputs():
    return [torch.randn(1, 4)]
def get_init_inputs():
    return [1]
"""


def test_bool_default_renders_capitalized(tmp_path):
    task, js = _write_model(tmp_path, CONV_LIKE)
    out = _instantiate(tmp_path, task, js)
    schema = _mdef_schema(out)
    assert "bool bias=False" in schema
    assert "=false" not in schema and "=true" not in schema
    _assert_schema_parses(schema)


def test_none_default_wrapped_optional(tmp_path):
    task, js = _write_model(tmp_path, POOL_LIKE)
    out = _instantiate(tmp_path, task, js)
    schema = _mdef_schema(out)
    assert schema == ("my_op(Tensor x, int kernel_size, int? stride=None, int padding=0, "
                      "int dilation=1, bool return_indices=False, bool ceil_mode=False) -> Tensor")
    _assert_schema_parses(schema)
    ops_h = (out / "kernel" / "ops.h").read_text()
    assert "const c10::optional<int64_t> &stride" in ops_h   # impl 侧同步 optional,传 None 不崩
    mn = (out / "model_new_ascendc.py").read_text()
    model = _construct(mn, 3, 2, 1, 1)                       # cls(*get_init_inputs()) 契约


def test_none_default_int_list_wrapped_optional(tmp_path):
    task, js = _write_model(
        tmp_path, INTERPOLATE_OPTIONAL_SIZE_LIKE, INTERPOLATE_OPTIONAL_SIZE_CASE
    )
    sig = prep._extract_op_signature(task, js)
    assert {p.name: p.kind for p in sig.fwd}["size"] == "int[]?"

    out = _instantiate(tmp_path, task, js)
    schema = _mdef_schema(out)
    assert schema == "my_op(Tensor x, int[]? size=None, bool antialias=False) -> Tensor"
    _assert_schema_parses(schema)

    ops_h = (out / "kernel" / "ops.h").read_text()
    assert "at::OptionalIntArrayRef size" in ops_h
    host = (out / "kernel" / "op_host" / "my_op.cpp").read_text()
    assert "TODO: size(int[]?)" in host


def test_list_default_renders_brackets(tmp_path):
    task, js = _write_model(tmp_path, CONV3D_TUPLE_LIKE)
    out = _instantiate(tmp_path, task, js)
    schema = _mdef_schema(out)
    assert "int[] stride=[1, 1, 1]" in schema and "int[] padding=[0, 0, 0]" in schema
    assert "=false" not in schema
    _assert_schema_parses(schema)


def test_single_element_tuple_default_no_trailing_comma(tmp_path):
    task, js = _write_model(tmp_path, SINGLE_ELEM_TUPLE_LIKE)
    out = _instantiate(tmp_path, task, js)
    schema = _mdef_schema(out)
    assert "int[] alpha_shape=[1]" in schema                 # (1,) → [1],不是 [1,]
    _assert_schema_parses(schema)


_REAL_DATASETS = [
    Path("/home/docker/datasets/op_tasks/op_assets_kernelbench_level1/op_tasks"),
    Path("/home/docker/datasets/op_assets_cudallm_filtered189/op_tasks"),
    Path("/home/docker/KernelBench/KernelBench"),
]


@pytest.mark.skipif(
    not all(d.is_dir() for d in _REAL_DATASETS),
    reason="三个真实数据集不在本机,跳过全量回归",
)
def test_real_datasets_schemas_all_parse():
    """三数据集全量渲染 + torch parse_schema:0 失败(2026-08-10 修复前的实测是
    36+2+51 个非法 schema,全是 dlopen 即崩、op 永不注册的必死题)。"""
    bad = []
    for root in _REAL_DATASETS:
        for p in sorted(root.rglob("*.py")):
            jp = p.with_suffix(".json")
            sig = prep._extract_op_signature(p, jp if jp.exists() else None)
            if sig is None:
                continue
            _, schema_args, _ = prep._ordered_params(sig)
            ret = "Tensor" if sig.ret_arity <= 1 else "(" + ", ".join(["Tensor"] * sig.ret_arity) + ")"
            schema = f"op_x({schema_args}) -> {ret}"
            try:
                torch._C.parse_schema(schema)
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{p.name}: {schema}  <- {exc}")
    assert not bad, "\n".join(bad[:20])


@pytest.mark.parametrize("source,dtype", [
    (SINGLE, "float16"),
    (SINGLE, "bfloat16"),
    (INIT_INT, "int64"),
    (TUPLE_RET, "float32"),
    (CONV_LIKE, "float32"),  # 输出 shape 不同于输入，不能把 empty_like 当契约
    (SCALAR_ONLY, "float32"),
    ("# no Model: use fallback skeleton\n", "float32"),
])
def test_skeleton_labels_semantic_placeholders_without_rewriting_reference(tmp_path, source, dtype):
    task, js = _write_model(tmp_path, source, {"inputs": [
        {"name": "x", "type": "tensor", "dtype": dtype, "shape": [4]},
    ]})
    originals = task.read_bytes(), js.read_bytes()
    out = _instantiate(tmp_path, task, js)
    assert (task.read_bytes(), js.read_bytes()) == originals
    host = (out / "kernel/op_host/my_op.cpp").read_text()
    kernel = (out / "kernel/op_kernel/my_op_kernel.cpp").read_text()
    assert "elementwise 占位不是本题语义契约" in host
    assert "empty_like、fp16/fp32/连续性限制、单输出接线与 tiling" in host
    assert "不能只替换 Compute" in kernel
    assert "不代表支持 BF16" in kernel
    assert "其余(tiling/双 buffer/dtype 分发)别动" not in kernel


@pytest.mark.parametrize("agent_side", [True, False])
def test_prepare_carries_existing_cannbot_design_resources_without_new_wiring(
    tmp_path, monkeypatch, agent_side,
):
    import json

    task, js = _write_model(tmp_path, LAYERNORM_LIKE, LAYERNORM_CASE)
    originals = task.read_bytes(), js.read_bytes()
    workdir = tmp_path / "session"
    monkeypatch.setattr(prep.shutil, "which", lambda name: "/unused/claude")
    args = ["--backend", "ascendc", "--op-name", "my_op",
            "--canonical-root", str(CANONICAL), "--workdir", str(workdir),
            "--task-path", str(task), "--only-project-skills"]
    if agent_side:
        args.append("--require-claude")
    assert prep.main(args) == 0
    assert (task.read_bytes(), js.read_bytes()) == originals
    assert (workdir / "input/my_op.py").read_bytes() == originals[0]
    assert (workdir / "input/my_op.json").read_bytes() == originals[1]
    for relative in (
        "workflows/templates/design-template.md",
        "skills/ascendc-tiling-design/SKILL.md",
        "skills/ascendc-tiling-design/references/reduction/patterns.md",
        "skills/ascendc-api-best-practices/SKILL.md",
        "skills/ascendc-api-best-practices/references/api-precision.md",
        "skills/tilelang2ascend-translator/SKILL.md",
    ):
        assert (workdir / ".claude" / relative).read_bytes() == (CANONICAL / relative).read_bytes()
    assert (workdir / "CLAUDE.md").read_bytes() == (CANONICAL / "CLAUDE.md").read_bytes()
    workflow = (workdir / "CLAUDE.md").read_text()
    translator = (workdir / ".claude/skills/tilelang2ascend-translator/SKILL.md").read_text()
    assert "简单算子只跳过 TileLang，不跳过设计" in workflow
    assert "简单算子跳过 Phase 3，" not in workflow
    for relative in ("workflows/templates/design-template.md",
                     "skills/ascendc-tiling-design/SKILL.md",
                     "skills/ascendc-api-best-practices/SKILL.md"):
        assert f".claude/{relative}" in translator
    assert not (workdir / ".claude/agents").exists()
    assert (workdir / "my_op/kernel").exists() == agent_side
    assert not (workdir / "output/submission").exists()
    assert not (workdir / "judge_out/metrics.json").exists()
    settings = json.loads((workdir / ".claude/settings.json").read_text())
    assert set(settings.get("hooks", {})) == ({"Stop"} if agent_side else set())
