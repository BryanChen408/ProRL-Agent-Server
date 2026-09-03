#!/usr/bin/env python3
"""Prepare a per-session operator workdir.

This script prepares one session workdir from canonical assets mounted at
/opt/canonical and creates a minimal editable ModelNew starter file.
"""

from __future__ import annotations

import argparse
import ast
import json
import keyword
import operator
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"required directory missing: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def _assert_upstream_paths(workdir: Path) -> None:
    """Fail fast when the doc-referenced paths don't resolve in the prepared workdir.

    Layout mirrors what upstream `init.sh` produces — everything flat under `.claude/`:
    `.claude/{skills,workflows}`.  Skill references use `../../../workflows/...`, which from
    `.claude/skills/<skill>/references/` lands on `.claude/` — same as post-install.

    Nothing in the repo cross-checks doc paths against the real tree, so a layout change
    upstream would otherwise only surface when a rollout trips over it.
    """
    probes = [
        # Phase 1.2 的固定动作要 cp 的文件(源已从旧 project-init
        # skill 改为 kernel_skeleton 模板)
        workdir / ".claude/workflows/templates/kernel_skeleton/kernel/utils/torch_kernel_helper.h",
    ]
    if (workdir / ".claude" / "workflows").is_dir():
        probes.append(workdir / ".claude/workflows/templates/archive_tasks")
    for probe in probes:
        if not probe.exists():
            raise FileNotFoundError(
                f"upstream path contract broken: {probe} 不可读 —— "
                f"canonical 布局变了,或 CLAUDE.md 引用的路径与实际不符。"
            )


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"required file missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # prepare 的 upload_file 可能已经把源文件直接放到目标位置(ascendc: input/{op}.py),
    # 此时 src 就是 dst,shutil.copy2 会抛 SameFileError → 视作"已就位"直接返回。
    # triton 路径 src(canonical/…)恒 ≠ dst(workdir/…),行为一字不变。
    try:
        if dst.exists() and src.samefile(dst):
            return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _camel(name: str) -> str:
    """op_name → 合法 C++ 类名后缀(Kernel{OpName})。下划线/非字母数字分段,逐段首字母大写。"""
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[^0-9A-Za-z]+", name) if p)


# ============================================================================
# 骨架签名生成
#
# 机制件(CMakeLists/setup.py/utils)从模板目录静态铺;签名件(model_new/register.cpp/
# ops.h/op_host/op_kernel)按 model.py 的 __init__/forward 静态解析生成 —— 签名本来就是
# 任务书的一部分,生成只是机械转写,agent 要写的仍只有 kernel 数学与 tiling。
#
# 原则:解析不了的特征一律放宽降级(*args 收纳 / 参数不进 op 签名 + 显式 TODO),
# 不报错、不排除算子,给 agent 留发挥空间。唯一硬约束是 harness 契约:
# ModelNew(*get_init_inputs()) 可构造、(*inputs) 可调用 —— 由"镜像 Model 签名"结构性保证。
# ============================================================================

_UNKNOWN = object()      # 值无法静态解析
_TENSOR_VAL = object()   # 值是 torch.tensor(...) 之类的张量构造

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv}


@dataclass
class _Param:
    name: str
    kind: str                 # tensor|tensor?|int|int?|float|float?|bool|str|int[]|int[]?|tensor[]|unknown
    default: str | None = None  # Python 源码形式的默认值(仅 forward/可选位)
    in_op: bool = True         # 不进 op 签名的参数在调用处留 TODO


@dataclass
class _OpSig:
    fwd: list[_Param] = field(default_factory=list)      # forward 位置参数
    kwonly: list[_Param] = field(default_factory=list)   # forward keyword-only 参数(带默认)
    init: list[_Param] = field(default_factory=list)     # __init__ 参数(get_init_inputs 顺序)
    ret_arity: int = 1
    init_storage: str = "named"   # named | args_kwargs(__init__ 不可静态镜像时的兜底)
    fwd_vararg: bool = False
    notes: list[str] = field(default_factory=list)


_SCHEMA_TYPE = {
    "tensor": "Tensor", "tensor?": "Tensor?",
    "int": "int", "int?": "int?", "float": "float", "float?": "float?",
    "bool": "bool", "str": "str", "int[]": "int[]", "int[]?": "int[]?",
    "tensor[]": "Tensor[]",
}
_CPP_TYPE = {
    "tensor": "const at::Tensor &", "tensor?": "const c10::optional<at::Tensor> &",
    "int": "int64_t ", "int?": "const c10::optional<int64_t> &",
    "float": "double ", "float?": "const c10::optional<double> &",
    "bool": "bool ", "str": "const std::string &",
    "int[]": "at::IntArrayRef ", "int[]?": "at::OptionalIntArrayRef ",
    "tensor[]": "at::TensorList ",
}
# 能接进 kernel 入口的标量 kind → kernel 形参类型;不在表里的 kind 只到 host(留 TODO)
_KERNEL_SCALAR = {"int": "int64_t", "float": "float", "bool": "int64_t"}


def _module_consts(tree) -> dict:
    """模块级常量。先做一遍字面量直收,再按引用展开 tuple/list/算术表达式
    (input_shape = (batch, features) 这类;解析不出就保持不可解析,放宽降级)。"""
    raw = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            raw[node.targets[0].id] = node.value
    consts: dict = {}

    def _resolve(node, depth=0):
        if depth > 8:
            return _UNKNOWN
        try:
            return ast.literal_eval(node)
        except Exception:
            pass
        if isinstance(node, ast.Name) and node.id in raw:
            return _resolve(raw[node.id], depth + 1)
        if isinstance(node, (ast.Tuple, ast.List)):
            vals = [_resolve(e, depth + 1) for e in node.elts]
            if any(v is _UNKNOWN for v in vals):
                return _UNKNOWN
            return type(node).__name__ == "Tuple" and tuple(vals) or vals
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            l, r = _resolve(node.left, depth + 1), _resolve(node.right, depth + 1)
            if l is not _UNKNOWN and r is not _UNKNOWN:
                try:
                    return _BINOPS[type(node.op)](l, r)
                except Exception:
                    return _UNKNOWN
        return _UNKNOWN

    for name, node in raw.items():
        v = _resolve(node)
        if v is not _UNKNOWN:
            consts[name] = v
    return consts


def _eval_const(node, consts):
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = [_eval_const(e, consts) for e in node.elts]
        if any(v is _UNKNOWN for v in vals):
            return _UNKNOWN
        return tuple(vals) if isinstance(node, ast.Tuple) else vals
    return _UNKNOWN


def _value_kind(v) -> str:
    if v is _TENSOR_VAL:
        return "tensor"
    if isinstance(v, bool):
        return "bool"          # 必须先于 int(bool 是 int 子类)
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        return "int[]"
    return "unknown"


def _ann_kind(node) -> str | None:
    if node is None:
        return None
    try:
        t = ast.unparse(node).replace(" ", "")
    except Exception:
        return None
    if t in ("torch.Tensor", "Tensor"):
        return "tensor"
    if t in ("Optional[torch.Tensor]", "Optional[Tensor]", "torch.Tensor|None", "Tensor|None"):
        return "tensor?"
    if t in ("int", "float", "bool", "str"):
        return t
    if t.startswith(("list", "tuple", "List", "Tuple")):
        return "int[]"          # shape 类;tensor_list 由 json 覆盖
    return None


def _json_param_kinds(json_path: Path | None) -> dict:
    """读 case json 首行,按输入项 name 给 forward 参数定型(NPUKernelBench 形态)。

    type 字段权威:tensor/scalar/attr/tensor_list。scalar 一律放宽为 float(schema float
    兼容 int 传入);required:false 的 tensor 按 Tensor? 处理。
    """
    kinds: dict = {}
    if not json_path or not Path(json_path).is_file():
        return kinds
    try:
        with open(json_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    break
            else:
                return kinds
    except Exception:
        return kinds
    for ent in case.get("inputs", []):
        name, typ = ent.get("name"), ent.get("type")
        if not name:
            continue
        if typ == "tensor":
            kinds[name] = "tensor?" if ent.get("required") is False else "tensor"
        elif typ == "tensor_list":
            kinds[name] = "tensor[]"
        elif typ == "scalar":
            kinds[name] = "float"   # 放宽:schema float 兼容 int 传入
        elif typ == "attr":
            v = ent.get("value")
            kind = _value_kind(v) if v is not None else "unknown"
            kinds[name] = kind if kind != "unknown" else "int"
    return kinds


def _get_init_values(tree, consts) -> list:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, (ast.List, ast.Tuple)):
                    vals = []
                    for elt in sub.value.elts:
                        v = _eval_const(elt, consts)
                        if v is _UNKNOWN and isinstance(elt, ast.Call) and "tensor" in ast.unparse(elt.func):
                            v = _TENSOR_VAL      # torch.tensor(...) 构造 → 按 tensor 处理
                        vals.append(v)
                    return vals
    return []


def _opt_wrap(kind: str) -> str:
    """None 默认值 → optional 形式(标量/tensor/int-list 都放宽兼容)。"""
    return {
        "tensor": "tensor?",
        "int": "int?",
        "float": "float?",
        "int[]": "int[]?",
    }.get(kind, kind)


def _extract_op_signature(task_path: Path, json_path: Path | None) -> _OpSig | None:
    """从 model.py 静态解析算子签名。整体解析失败返回 None(调用方回退模板)。"""
    try:
        tree = ast.parse(task_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    model = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            model = node
            break
    if model is None:
        return None
    init_fn = fwd_fn = None
    for item in model.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_fn = item
        elif isinstance(item, ast.FunctionDef) and item.name == "forward":
            fwd_fn = item
    if fwd_fn is None:
        return None

    consts = _module_consts(tree)
    json_kinds = _json_param_kinds(json_path)
    notes: list[str] = []

    def _fwd_kind(arg, dflt) -> str:
        kind = json_kinds.get(arg.arg) or _ann_kind(arg.annotation)
        if kind is None and dflt is not None:
            dv = _eval_const(dflt, consts)   # 默认值本身带类型信息(keep_prob=1.0 → float)
            if dv is not _UNKNOWN:
                k2 = _value_kind(dv)
                if k2 != "unknown":
                    kind = k2
        if kind is None:
            kind = "tensor"
        # 无注解时放宽默认 tensor —— cudallm/KernelBench 多参 forward 已验证全是 tensor;
        # NPUKernelBench 由 json type 字段覆盖。
        if isinstance(dflt, ast.Constant) and dflt.value is None:
            kind = _opt_wrap(kind)
        return kind

    # ---- forward ----
    fwd: list[_Param] = []
    kwonly: list[_Param] = []
    fwd_vararg = False
    a = fwd_fn.args
    pos = a.args[1:]
    if a.vararg or a.kwarg:
        fwd_vararg = True
        notes.append("forward 含 *args/**kwargs,op 调用按单输入兜底,请按算子语义自行接线")
    bad_names = any(not x.arg.isidentifier() or keyword.iskeyword(x.arg) for x in pos)
    if bad_names:
        fwd_vararg = True
        notes.append("forward 参数名不可静态镜像,按单输入兜底")
    if not fwd_vararg:
        defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        for arg, dflt in zip(pos, defaults):
            fwd.append(_Param(arg.arg, _fwd_kind(arg, dflt),
                              ast.unparse(dflt) if dflt is not None else None))
        for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
            kind = _fwd_kind(arg, dflt)
            kwonly.append(_Param(arg.arg, kind, ast.unparse(dflt) if dflt is not None else "None"))
    else:
        fwd = [_Param("x", "tensor")]

    # ---- 返回 arity(多返回值算子要生成多 Tensor 签名) ----
    ret_arity = 1
    for sub in ast.walk(fwd_fn):
        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
            ret_arity = max(ret_arity, len(sub.value.elts))

    # ---- __init__ ----
    init: list[_Param] = []
    init_storage = "named"
    if init_fn is not None and (init_fn.args.vararg or init_fn.args.kwarg):
        init_storage = "args_kwargs"
        notes.append("Model.__init__ 含 *args/**kwargs,ModelNew 用 *args/**kwargs 收纳;"
                     "op 签名只含 forward 参数,标量接线请自行补")
    init_values = _get_init_values(tree, consts)
    if init_fn is not None and init_storage == "named":
        ipos = init_fn.args.args[1:]
        if any(not x.arg.isidentifier() or keyword.iskeyword(x.arg) for x in ipos):
            init_storage = "args_kwargs"
            notes.append("Model.__init__ 参数名不可静态镜像,用 *args/**kwargs 收纳")
        else:
            idefaults = [None] * (len(ipos) - len(init_fn.args.defaults)) + list(init_fn.args.defaults)
            for i, (arg, dflt) in enumerate(zip(ipos, idefaults)):
                v = init_values[i] if i < len(init_values) else _UNKNOWN
                kind = _value_kind(v) if v is not _UNKNOWN else "unknown"
                if kind == "unknown" and dflt is not None:
                    dv = _eval_const(dflt, consts)
                    if dv is not _UNKNOWN:
                        kind = _value_kind(dv)
                in_op = kind != "unknown"
                if in_op and isinstance(dflt, ast.Constant) and dflt.value is None:
                    # None 默认值 → optional 形式(对齐 _fwd_kind)。不包的话 schema 写成
                    # `int stride=None`(类型仍是非 optional int,schema 能解析),
                    # 但生成的 C++ impl 是 int64_t 非可选,模型真传 None 时调用即崩
                    # (实测 0046 Average_Pooling_3D,init 值就是 None)。
                    kind = _opt_wrap(kind)
                if not in_op:
                    notes.append(f"init 参数 {arg.arg} 的值无法静态解析,未纳入 op 签名;"
                                 f"若 kernel 需要请自行接线并同步 register.cpp/ops.h/op_host")
                init.append(_Param(arg.arg, kind,
                                   ast.unparse(dflt) if dflt is not None else None,
                                   in_op=in_op))
    return _OpSig(fwd=fwd, kwonly=kwonly, init=init, ret_arity=ret_arity,
                  init_storage=init_storage, fwd_vararg=fwd_vararg, notes=notes)


def _schema_default(p: _Param) -> str | None:
    if p.default is None:
        return None
    d = p.default
    if p.kind == "bool":
        # torch schema 的 bool 默认值只认 Python 形式 True/False(及 0/1),
        # 小写 true/false 会被 parse_schema 拒("invalid numeric default value")。
        # d 来自 ast.unparse,本身就是 "True"/"False",原样透传。
        return d
    if p.kind == "str":
        return json.dumps(d.strip("'\""))
    if d == "None":
        return "None"
    if p.kind.endswith("[]"):
        # list 默认值必须是 torch 的方括号语法 [1, 1];ast.unparse 给出的 Python
        # 元组 (1, 1) / (1,) 直接被拒 —— 且单元素元组去尾逗号,(1,) → [1] 而非 [1,]。
        try:
            vals = ast.literal_eval(d)
        except (ValueError, SyntaxError):
            vals = None
        if isinstance(vals, (tuple, list)):
            return "[" + ", ".join(str(v) for v in vals) + "]"
    return d


def _ordered_params(sig: _OpSig) -> tuple[list[_Param], str, str]:
    """op 参数总序:forward 位置 → init → kwonly。schema 默认值只在尾段连续时保留
    (torch schema 要求带默认值的参数必须连续殿后;不满足就全部不写默认,调用方显式传全)。"""
    ordered = ([p for p in sig.fwd if p.in_op]
               + [p for p in sig.init if p.in_op]
               + [p for p in sig.kwonly if p.in_op])
    seen_default = False
    keep_defaults = True
    for p in ordered:
        if seen_default and p.default is None:
            keep_defaults = False
            break
        if p.default is not None:
            seen_default = True
    schema_args, cpp_args = [], []
    for p in ordered:
        d = _schema_default(p) if keep_defaults else None
        schema_args.append(f"{_SCHEMA_TYPE[p.kind]} {p.name}" + (f"={d}" if d else ""))
        cpp_args.append(f"{_CPP_TYPE[p.kind]}{p.name}")
    return ordered, ", ".join(schema_args), ", ".join(cpp_args)


def _ret_cpp(sig: _OpSig) -> str:
    if sig.ret_arity <= 1:
        return "at::Tensor"
    return "std::tuple<" + ", ".join(["at::Tensor"] * sig.ret_arity) + ">"


def _render_model_new(op: str, sig: _OpSig) -> str:
    lines = [
        "import glob as _glob",
        "import sys",
        "from pathlib import Path",
        "",
        "import torch",
        "import torch.nn as nn",
        "",
        f'_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"',
        f'_LIB_PATTERN = str(_KERNEL_BUILD / "{op}_ext*")',
        "",
        "# register.cpp 用 TORCH_LIBRARY(无 PYBIND11_MODULE),裸 import 会因缺 PyInit 失败;",
        "# 真正注册靠 fallback 的 torch.ops.load_library(...)。这段加载逻辑原样保留,别改。",
        "try:",
        f"    import {op}_ext  # noqa: F401",
        "except ImportError:",
        '    if _LIB_PATTERN not in "".join(sys.path):',
        "        _libs = _glob.glob(_LIB_PATTERN)",
        "        if _libs:",
        "            torch.ops.load_library(_libs[0])",
        "",
        "",
        "class ModelNew(nn.Module):",
        "    # 签名由 prepare 按 model.py 生成(与 register.cpp/ops.h/op_host 自洽),一般不用动;",
        "    # kernel 接口确需变化时,上述三处 + 本文件调用行要同步改。",
    ]
    if sig.init_storage == "args_kwargs":
        lines += [
            "    def __init__(self, *args, **kwargs):",
            "        super().__init__()",
            "        self._init_args = args",
            "        self._init_kwargs = kwargs",
            "        # TODO: Model.__init__ 无法静态镜像(*args/**kwargs 或参数名不可解析),",
            "        # op 签名只含 forward 参数;算子需要的标量请自行接线并同步 register.cpp/ops.h/op_host。",
        ]
    elif sig.init:
        parts = []
        for p in sig.init:
            parts.append(f"{p.name}={p.default}" if p.default is not None else p.name)
        lines.append(f"    def __init__(self, {', '.join(parts)}):")
        lines.append("        super().__init__()")
        for p in sig.init:
            lines.append(f"        self.{p.name} = {p.name}")
            if not p.in_op:
                lines.append(f"        # TODO: {p.name} 的值无法静态解析,未纳入 op 签名;"
                             f"若 kernel 需要请在下方调用处加入并同步 register.cpp/ops.h/op_host")
    else:
        lines += ["    def __init__(self):", "        super().__init__()"]
    lines.append("")
    if sig.fwd_vararg:
        lines += [
            "    def forward(self, *args):",
            "        x = args[0]",
            "        # TODO: forward 含 *args/**kwargs,按单输入兜底,请按算子语义接线。",
            f"        # 必须真调 torch.ops.npu.{op}(AST 退化检测会查),不要加 plain-torch 兜底。",
            f"        return torch.ops.npu.{op}(x)",
        ]
        return "\n".join(lines) + "\n"
    parts = []
    for p in sig.fwd:
        parts.append(f"{p.name}={p.default}" if p.default is not None else p.name)
    if sig.kwonly:
        parts.append("*")
        for p in sig.kwonly:
            parts.append(f"{p.name}={p.default if p.default is not None else 'None'}")
    lines.append(f"    def forward(self, {', '.join(parts)}):")
    call_args = [p.name for p in sig.fwd if p.in_op]
    call_args += [f"self.{p.name}" for p in sig.init if p.in_op]
    call_args += [f"{p.name}={p.name}" for p in sig.kwonly if p.in_op]
    lines.append(f"        # 必须真调 torch.ops.npu.{op}(AST 退化检测会查),不要加 plain-torch 兜底。")
    lines.append(f"        return torch.ops.npu.{op}({', '.join(call_args)})")
    return "\n".join(lines) + "\n"


def _render_register_cpp(op: str, sig: _OpSig, ordered: list[_Param], schema_args: str) -> str:
    ret = "Tensor" if sig.ret_arity <= 1 else "(" + ", ".join(["Tensor"] * sig.ret_arity) + ")"
    # Dispatcher 只有从必需 Tensor/Tensor[] 实参中才能稳定推导 PrivateUse1。
    # 纯标量或仅 Optional[Tensor] 的签名在实参无 Tensor 时没有 dispatch key，必须使用
    # CatchAll 才能进入 host 实现。这里按签名属性选择，不依赖算子名或数据集白名单。
    dispatch_key = (
        "PrivateUse1"
        if any(p.kind in ("tensor", "tensor[]") for p in ordered)
        else "CatchAll"
    )
    return f'''#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {{

// 用 TORCH_LIBRARY 注册(不加 PYBIND11_MODULE)。注册在 .so 被 dlopen 时触发,
// 由 model_new_ascendc.py 的 torch.ops.load_library(...) 完成加载。
// m.def 签名由 prepare 按 model.py 生成,与 ops.h、op_host/{op}.cpp、model_new 的调用自洽;
// 要改请四处同步。
TORCH_LIBRARY_FRAGMENT(npu, m)
{{
    m.def("{op}({schema_args}) -> {ret}");
}}

TORCH_LIBRARY_IMPL(npu, {dispatch_key}, m)
{{
    m.impl("{op}", TORCH_FN(ascend_kernel::{op}));
}}

}}  // namespace
'''


def _render_ops_h(op: str, sig: _OpSig, cpp_args: str) -> str:
    return f'''#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>
#include <string>
#include <tuple>

namespace ascend_kernel {{

// host 函数声明,由 prepare 按 model.py 的签名生成;
// register.cpp 的 m.def、op_host/{op}.cpp 的实现、model_new 的调用已与它一致。
{_ret_cpp(sig)} {op}({cpp_args});

}} // namespace ascend_kernel

#endif // OPS_H
'''


def _render_op_host_cpp(op: str, sig: _OpSig, ordered: list[_Param], cpp_args: str) -> str:
    tensors = [p for p in ordered if p.kind == "tensor"]
    tensor_lists = [p for p in ordered if p.kind == "tensor[]"]
    unwired = [
        p for p in ordered
        if p.kind in ("tensor?", "int?", "float?", "str", "int[]", "int[]?", "tensor[]")
    ]
    scalars = [p for p in ordered if p.kind in _KERNEL_SCALAR]
    lines: list[str] = []
    a = lines.append
    a(f"// {op} op_host — 校验 + tiling + EXEC_KERNEL_CMD 启动")
    a("// 签名由 prepare 按 model.py 生成(与 register.cpp/ops.h/model_new 自洽);")
    a("// tiling 壳取自已验证的 elementwise 模板:核间划分 + UB 感知 tileLength + 32B 对齐。")
    a("// 要改签名请四处同步;tiling/启动参数按你的算子调。")
    a("#include <algorithm>")
    a("#include <cstdint>")
    a("#include <tuple>")
    a("")
    a("#include <torch/extension.h>")
    a("#include <torch/library.h>")
    a("")
    a('#include "torch_kernel_helper.h"')
    a('#include "tiling/platform/platform_ascendc.h"')
    a("")
    a(f"// 由 build 从 op_kernel/{op}_kernel.cpp 的 __global__ 入口自动生成。")
    a(f'#include "aclrtlaunch_{op}_kernel.h"')
    a("")
    a("namespace ascend_kernel {")
    a("")
    a("constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;")
    a("")
    a(f"{_ret_cpp(sig)} {op}({cpp_args})")
    a("{")
    # host 侧局部变量一律下划线开头 —— 与数据集参数名(不会下划线开头)结构性错开,
    # 防止参数名撞上 tiling 局部变量(比如某输入就叫 y/output/tileLength)。
    first = None
    if tensors:
        first = tensors[0].name
    elif tensor_lists:
        p = tensor_lists[0]
        a(f"    // {p.name} 是 Tensor[];骨架按第 1 个输入接线,其余元素按算子语义自行扩展。")
        a(f'    TORCH_CHECK({p.name}.size() > 0, "{op}: {p.name} must be non-empty");')
        a(f"    at::Tensor _x0 = {p.name}[0];")
        first = "_x0"
    if first is None:
        # 纯标量驱动(无 tensor 输入)的算子(如 mask 生成类):elementwise 壳不适用,
        # 放宽为可编译占位 + TODO,host 逻辑由 agent 按算子语义写。
        for p in unwired:
            a(f"    // TODO: {p.name}({_SCHEMA_TYPE[p.kind]})未接入 kernel 入口,按算子语义自行接线")
        a("    // TODO: 该算子没有 tensor 输入(纯标量驱动),骨架给不出 tiling 壳;")
        a("    // 输出形状/启动逻辑请按算子语义自行实现(下方仅为可编译占位)。")
        out_names = []
        for i in range(max(1, sig.ret_arity)):
            a(f'    at::Tensor _out{i} = at::empty({{0}}, at::TensorOptions().dtype(at::kFloat));')
            out_names.append(f"_out{i}")
        ret_expr = out_names[0] if sig.ret_arity <= 1 else f"std::make_tuple({', '.join(out_names)})"
        a("")
        a(f"    return {ret_expr};")
        a("}")
        a("")
        a("}  // namespace ascend_kernel")
        return "\n".join(lines) + "\n"
    for t in tensors:
        a(f'    TORCH_CHECK({t.name}.scalar_type() == at::kHalf || {t.name}.scalar_type() == at::kFloat,')
        a(f'                "{op}: only float16 and float32 are supported, got ", {t.name}.scalar_type());')
        a(f'    TORCH_CHECK({t.name}.is_contiguous(), "{op}: {t.name} must be contiguous");')
    for p in unwired:
        a(f"    // TODO: {p.name}({_SCHEMA_TYPE[p.kind]})未接入 kernel 入口,按算子语义自行接线")
    if sig.ret_arity > 1:
        a(f"    // 多返回值算子:骨架只把第 1 个输出接进 kernel,其余输出请按算子语义接线。")
    a("")
    out_names = []
    if sig.ret_arity <= 1:
        a(f"    at::Tensor _output = at::empty_like({first});")
        out_names = ["_output"]
    else:
        for i in range(sig.ret_arity):
            a(f"    at::Tensor _out{i} = at::empty_like({first});")
        out_names = [f"_out{i}" for i in range(sig.ret_arity)]
    ret_expr = out_names[0] if sig.ret_arity <= 1 else f"std::make_tuple({', '.join(out_names)})"
    a("")
    a(f"    int64_t _totalLength = {first}.numel();")
    a("    if (_totalLength == 0) {")
    a(f"        return {ret_expr};")
    a("    }")
    a(f"    int64_t _dtypeSize = {first}.element_size();")
    a("")
    a("    auto _ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();")
    a("    int64_t _coreNum = static_cast<int64_t>(_ascendc_platform->GetCoreNumAiv());")
    a("    if (_coreNum <= 0) { _coreNum = 1; }")
    a("    uint64_t _ubSize = 0;")
    a("    _ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, _ubSize);")
    a("")
    a("    int64_t _totalLengthCore = (_totalLength + _coreNum - 1) / _coreNum;")
    a("    int64_t _totalLengthCoreAlign = (_totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1) /")
    a("                                   CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH;")
    a("")
    a("    int64_t _usedCoreNum = (_totalLength + _totalLengthCoreAlign - 1) / _totalLengthCoreAlign;")
    a("    int64_t _formerNum = _usedCoreNum - 1;")
    a("    int64_t _formerLength = _totalLengthCoreAlign;")
    a("    int64_t _tailLength = _totalLength - _formerNum * _formerLength;")
    a("")
    a("    int64_t _bufferCoefficient = _dtypeSize * 4;  // 按 UB 分配表调整(in/out 双 buffer)")
    a("    if (_bufferCoefficient <= 0) { _bufferCoefficient = 1; }")
    a("    int64_t _maxTileElements = static_cast<int64_t>(_ubSize) / _bufferCoefficient;")
    a("    int64_t _alignElements = 32 / (_dtypeSize > 0 ? _dtypeSize : 1);")
    a("    if (_alignElements <= 0) { _alignElements = 1; }")
    a("    int64_t _tileLength = (_maxTileElements / _alignElements) * _alignElements;")
    a("    if (_tileLength <= 0) { _tileLength = _alignElements; }")
    a("")
    a("    uint32_t _blockDim = static_cast<uint32_t>(_usedCoreNum);")
    a("")
    # EXEC_KERNEL_CMD 的位置实参与 kernel 入口按位置对应(输入 in0..inN → 输出 → tiling → 标量)
    exec_args = [t.name for t in tensors] or [first]
    exec_args += [out_names[0], "_formerNum", "_formerLength", "_tailLength", "_tileLength", "_dtypeSize"]
    for p in scalars:
        if p.kind == "float":
            a(f"    float _{p.name}_l = static_cast<float>({p.name});  // EXEC_KERNEL_CMD 需要左值")
            exec_args.append(f"_{p.name}_l")
        elif p.kind == "bool":
            a(f"    int64_t _{p.name}_l = {p.name} ? 1 : 0;  // bool 用 int64_t 左值替代")
            exec_args.append(f"_{p.name}_l")
        else:
            exec_args.append(p.name)   # int64_t 形参本身是左值,直接传
    if scalars:
        a("")
    a(f"    EXEC_KERNEL_CMD({op}_kernel, _blockDim,")
    a(f"                    {', '.join(exec_args)});")
    a("")
    a(f"    return {ret_expr};")
    a("}")
    a("")
    a("}  // namespace ascend_kernel")
    return "\n".join(lines) + "\n"


def _render_op_kernel_cpp(op: str, camel: str, sig: _OpSig, ordered: list[_Param]) -> str:
    n_inputs = sum(1 for p in ordered if p.kind == "tensor")
    if n_inputs == 0:
        n_inputs = 1   # tensor[] 接管/纯标量兜底:单输入占位,由 agent 按算子语义重写
    scalars = [p for p in ordered if p.kind in _KERNEL_SCALAR]
    # kernel 内部名字一律按位置合成(in0/in1...、inQueue0、in0Gm),不取数据集参数名 ——
    # 否则某输入叫 y 时会撞上输出 y/成员 yGm(C 函数按位置传参,名字本来就无需对应)。
    inputs = [(f"in{i}", f"inQueue{i}", f"in{i}Gm") for i in range(n_inputs)]
    scalar_decls = [(p, _KERNEL_SCALAR[p.kind]) for p in scalars]

    lines: list[str] = []
    a = lines.append
    a(f"// {op} device kernel — elementwise 骨架({n_inputs} 输入;dtype 分发 + 尾块 + 32B 对齐 + 双 buffer)")
    a("// 数学在 Compute() 里,默认把第 1 个输入恒等拷贝到输出(能编过、能注册、跑通打包链路,")
    a("// 对拍必然不过 —— 把 Compute() 换成你的算子数学)。多输入已各自建好 queue/GM。")
    a('#include "kernel_operator.h"')
    a("")
    a("constexpr int32_t BUFFER_NUM = 2;")
    a("")
    a("template <typename T>")
    a(f"class Kernel{camel} {{")
    a("public:")
    a(f"    __aicore__ inline Kernel{camel}() {{}}")
    a("")
    init_params = [f"GM_ADDR {n}" for n, _, _ in inputs] + ["GM_ADDR y"]
    init_params += ["int64_t formerNum", "int64_t formerLength", "int64_t tailLength", "int64_t tileLength"]
    init_params += [f"{kt} {p.name}" for p, kt in scalar_decls]
    a(f"    __aicore__ inline void Init({', '.join(init_params[:4])},")
    a(f"                                {', '.join(init_params[4:])})")
    a("    {")
    a("        int64_t blockIdx = AscendC::GetBlockIdx();")
    a("")
    a("        if (blockIdx < formerNum) {")
    a("            this->blockLength = formerLength;")
    a("            int64_t offset = formerLength * blockIdx;")
    for n, _, gm in inputs:
        a(f"            {gm}.SetGlobalBuffer((__gm__ T *){n} + offset, formerLength);")
    a("            yGm.SetGlobalBuffer((__gm__ T *)y + offset, formerLength);")
    a("        } else {")
    a("            this->blockLength = tailLength;")
    a("            int64_t tailIdx = blockIdx - formerNum;")
    a("            int64_t offset = formerLength * formerNum + tailLength * tailIdx;")
    for n, _, gm in inputs:
        a(f"            {gm}.SetGlobalBuffer((__gm__ T *){n} + offset, tailLength);")
    a("            yGm.SetGlobalBuffer((__gm__ T *)y + offset, tailLength);")
    a("        }")
    a("        this->tileLength = tileLength;")
    for p, _ in scalar_decls:
        a(f"        this->{p.name} = {p.name};")
    a("")
    for _, q, _ in inputs:
        a(f"        pipe.InitBuffer({q}, BUFFER_NUM, tileLength * sizeof(T));")
    a("        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));")
    a("    }")
    a("")
    a("    __aicore__ inline void Process()")
    a("    {")
    a("        int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;")
    a("        int64_t tailTileLength = this->blockLength - (tileNum - 1) * this->tileLength;")
    a("")
    a("        int64_t alignNum = 32 / static_cast<int64_t>(sizeof(T));")
    a("        int64_t alignedTailLen = ((tailTileLength + alignNum - 1) / alignNum) * alignNum;")
    a("")
    a("        for (int64_t i = 0; i < tileNum - 1; ++i) {")
    a("            CopyIn(i, this->tileLength);")
    a("            Compute(i, this->tileLength);")
    a("            CopyOut(i, this->tileLength);")
    a("        }")
    a("        if (tileNum > 0) {")
    a("            CopyIn(tileNum - 1, alignedTailLen);")
    a("            Compute(tileNum - 1, alignedTailLen);")
    a("            CopyOut(tileNum - 1, alignedTailLen);")
    a("        }")
    a("    }")
    a("")
    a("private:")
    a("    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)")
    a("    {")
    for n, q, gm in inputs:
        a(f"        AscendC::LocalTensor<T> {n}Local = {q}.AllocTensor<T>();")
        a(f"        AscendC::DataCopy({n}Local, {gm}[progress * this->tileLength], curTileLength);")
        a(f"        {q}.EnQue({n}Local);")
    a("    }")
    a("")
    a("    __aicore__ inline void Compute(int64_t progress, int64_t curTileLength)")
    a("    {")
    for n, q, _ in inputs:
        a(f"        AscendC::LocalTensor<T> {n}Local = {q}.DeQue<T>();")
    a("        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();")
    a("")
    a(f"        // TODO: 换成你的算子数学。默认恒等拷贝 y = in0(只为打通打包/注册链路)。")
    a(f"        // 例: AscendC::Abs(yLocal, in0Local, curTileLength);       // |x|")
    if n_inputs >= 2:
        a(f"        //     AscendC::Add(yLocal, in0Local, in1Local, curTileLength); // x + y")
    else:
        a(f"        //     AscendC::Adds(yLocal, in0Local, (T)1, curTileLength); // x + 1")
    a(f"        AscendC::DataCopy(yLocal, in0Local, curTileLength);")
    a("")
    a("        outQueueY.EnQue<T>(yLocal);")
    for n, q, _ in inputs:
        a(f"        {q}.FreeTensor({n}Local);")
    a("    }")
    a("")
    a("    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)")
    a("    {")
    a("        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();")
    a("        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, curTileLength);")
    a("        outQueueY.FreeTensor(yLocal);")
    a("    }")
    a("")
    a("private:")
    a("    AscendC::TPipe pipe;")
    for _, q, _ in inputs:
        a(f"    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> {q};")
    a("    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;")
    for _, _, gm in inputs:
        a(f"    AscendC::GlobalTensor<T> {gm};")
    a("    AscendC::GlobalTensor<T> yGm;")
    a("    int64_t blockLength;")
    a("    int64_t tileLength;")
    for p, kt in scalar_decls:
        a(f"    {kt} {p.name};")
    a("};")
    a("")
    gparams = [f"GM_ADDR {n}" for n, _, _ in inputs] + ["GM_ADDR y"]
    gparams += ["int64_t formerNum", "int64_t formerLength",
              "int64_t tailLength", "int64_t tileLength", "int64_t dtypeSize"]
    gparams += [f"{kt} {p.name}" for p, kt in scalar_decls]
    a(f'extern "C" __global__ __aicore__ void {op}_kernel(')
    a(f"    {', '.join(gparams[:2])},")
    a(f"    {', '.join(gparams[2:])})")
    a("{")
    init_call = [n for n, _, _ in inputs] + ["y", "formerNum", "formerLength",
                                             "tailLength", "tileLength"] + [p.name for p, _ in scalar_decls]
    a("    if (dtypeSize == 2) {")
    a(f"        Kernel{camel}<half> op;")
    a(f"        op.Init({', '.join(init_call)});")
    a("        op.Process();")
    a("    } else {")
    a(f"        Kernel{camel}<float> op;")
    a(f"        op.Init({', '.join(init_call)});")
    a("        op.Process();")
    a("    }")
    a("}")
    return "\n".join(lines) + "\n"


# 模板目录里会被生成器取代的签名件(相对模板根的路径,占位符未替换前)
_SKELETON_GENERATED = {
    "model_new_ascendc.py",
    "kernel/register.cpp",
    "kernel/ops.h",
    "kernel/op_host/{op_name}.cpp",
    "kernel/op_kernel/{op_name}_kernel.cpp",
}


def _instantiate_kernel_skeleton(canonical: Path, workdir: Path, op: str,
                                 task_path: Path | None = None,
                                 json_path: Path | None = None) -> None:
    """把 kernel_skeleton 铺进 workdir/{op}/(只 agent 侧)。

    机制件(CMakeLists/register.cpp 之外的打包件)从模板静态拷贝;签名件(model_new/
    register.cpp/ops.h/op_host/op_kernel)按 task_path(model.py)解析出的签名生成。
    解析失败回退模板的单 tensor 版签名件。已存在的文件不覆盖(agent 可能已写)。
    """
    skel = canonical / "workflows" / "templates" / "kernel_skeleton"
    if not skel.is_dir():
        return
    sig = _extract_op_signature(task_path, json_path) if task_path else None
    camel = _camel(op)
    for src in sorted(skel.rglob("*")):
        if not src.is_file():
            continue
        rel = str(src.relative_to(skel))
        if rel == "README.md":
            continue  # README 是模板说明,不进工程(CLAUDE.md 已有规则)
        if sig is not None and rel in _SKELETON_GENERATED:
            continue  # 签名件由生成器产出
        dst = workdir / op / rel.replace("{op_name}", op).replace("{OpName}", camel)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        dst.write_text(text.replace("{op_name}", op).replace("{OpName}", camel), encoding="utf-8")
    if sig is None:
        return
    ordered, schema_args, cpp_args = _ordered_params(sig)
    outputs = {
        "model_new_ascendc.py": _render_model_new(op, sig),
        "kernel/register.cpp": _render_register_cpp(op, sig, ordered, schema_args),
        "kernel/ops.h": _render_ops_h(op, sig, cpp_args),
        f"kernel/op_host/{op}.cpp": _render_op_host_cpp(op, sig, ordered, cpp_args),
        f"kernel/op_kernel/{op}_kernel.cpp": _render_op_kernel_cpp(op, camel, sig, ordered),
    }
    for rel, text in outputs.items():
        dst = workdir / op / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
    note = f"; 放宽降级: {'; '.join(sig.notes)}" if sig.notes else ""
    print(f"[prepare] 骨架签名已生成: forward {len(sig.fwd)} 参/init {len(sig.init)} 参/"
          f"kwonly {len(sig.kwonly)} 参/返回 {sig.ret_arity} 元{note}")


def _record_skeleton_hash(workdir: Path, op: str) -> None:
    """记录预生成骨架的内容哈希,供评测入口检测「骨架原封未动、但 output/submission/
    下有源码文件」的改错目录场景(与 ascendc_eval_pipeline.sh 的 CUR_HASH 同口径)。
    agent 删了它最多让提醒失效,不影响评测与预算。"""
    src = workdir / op
    if not src.is_dir():
        return
    find_expr = (
        r"\( -name build -o -name dist -o -name '*.egg-info' -o -name '__pycache__' \) -prune -o"
        r" -type f ! -name '*.so' ! -name '*.a' ! -name '*.o' ! -name '*.whl'"
        r" ! -name '.eval_last.log' ! -name 'performance.json' ! -name 'preformance.json'"
        " -print0"
    )
    cmd = (
        f"cd {shlex.quote(str(src))} && find . {find_expr} 2>/dev/null | sort -z"
        " | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1"
    )
    try:
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60)
        digest = out.stdout.strip()
    except Exception:
        return
    if not digest:
        return
    state = workdir / "output" / ".selfcheck"
    state.mkdir(parents=True, exist_ok=True)
    (state / f".{op}_skeleton.hash").write_text(digest, encoding="utf-8")


def _forward_arg_names(task_path: Path) -> list[str]:
    try:
        tree = ast.parse(task_path.read_text())
    except Exception:
        return ["x"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Model":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                names = [arg.arg for arg in item.args.args if arg.arg != "self"]
                return names or ["x"]
    return ["x"]


def _write_stub(task_path: Path, op_name: str, submission_path: Path) -> None:
    args = _forward_arg_names(task_path)
    if not all(arg.isidentifier() and not keyword.iskeyword(arg) for arg in args):
        args = ["x"]
    first = args[0]
    signature = ", ".join(args)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(
        f"""import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _{op_name.replace("-", "_").replace(".", "_")}_starter_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, {signature}):
        out = torch.empty_like({first})
        n_elements = {first}.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _{op_name.replace("-", "_").replace(".", "_")}_starter_kernel[grid]({first}, out, n_elements, BLOCK_SIZE=1024)
        return out
""",
        encoding="utf-8",
    )


def _prepare_tools(canonical: Path, workdir: Path, *, readonly_tools: bool) -> None:
    src = canonical / "tools"
    if not src.is_dir():
        raise FileNotFoundError(f"required directory missing: {src}")
    dst = workdir / "tools"
    if readonly_tools:
        # Docker binds canonical tools to this path as :ro. Do not remove or overwrite it.
        dst.mkdir(parents=True, exist_ok=True)
        return
    _copy_tree(src, dst)


# Claude Code CLI 自带(bundled)的 skill —— 不来自 canonical/skills,与写算子无关:
# 每个 session 白占 ~4.3k 字符 prompt,还可能被模型误调(WebSearch/WebFetch 已 disallow,调了必失败)。
# 清单实测自镜像里的 claude 2.1.168(session 首条 init 消息 + prompt 里的 skill listing)。
# ⚠️ 这是"名单"不是"规则":CLI 只支持 skillOverrides={精确名: off},无通配符、无 allowlist
#    (sessionSkillAllowlist 只对 Agent SDK 开放,`claude -p` 够不到)。CLI 升级新增 bundled skill
#    时这里要跟着补 —— 巡检办法:读 session 的 logs/agent/claude-code.txt 首行 init 的 slash_commands,
#    出现不在本表也不在 canonical/skills 里的名字即为新增。
# ⭐ 根治:镜像里的 claude 升到 2.1.216+,那版支持 CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1
#    (官方描述:bundled skills 整体移除,.claude/skills/ 不受影响)——一条规则、零名单。
#    profile.ascendc.yaml 已经把该 env 配好,升级镜像即自动接管。
_CLI_BUNDLED_SKILLS = (
    "deep-research",
    "update-config",
    "keybindings-help",
    "verify",
    "code-review",
    "simplify",
    "fewer-permission-prompts",
    "loop",
    "claude-api",
    "run",
    "init",
    "review",
    "security-review",
)

_STOP_GUARD_COMMAND = 'python3 "$CLAUDE_PROJECT_DIR/tools/ascendc_stop_guard.py"'


def _non_project_skills(canonical: Path) -> list[str]:
    """规则:凡不是本项目(canonical/skills)提供的 CLI 自带 skill,一律关掉。

    以 canonical/skills 的实际内容为准 —— 往 canonical 里加/删 skill 自动跟随,
    同名时(比如哪天我们自己也叫 review)以本项目的为准,不会被误关。
    """
    ours = {p.name for p in (canonical / "skills").iterdir() if p.is_dir()}
    return [name for name in _CLI_BUNDLED_SKILLS if name not in ours]


def _write_claude_settings(
    workdir: Path,
    names: list[str],
    *,
    enable_stop_guard: bool = False,
) -> None:
    """合并 project settings：关闭无关 skill，并按需安装只读完成门禁。

    `{name: "off"}` = 既不列进 prompt 也不允许模型调用(claude 2.1.168 起支持,
    实测 schema: skillOverrides: Record<str, on|name-only|user-invocable-only|off>)。
    更高版本另有 CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1 可一把关(2.1.168 不认)。
    Stop hook 不执行评测、不改状态，只在 metrics 明确 task_complete=false 时拒绝结束。
    两部分都没启用时一个字节都不写；已有 permissions/hooks 原样合并保留。
    """
    if not names and not enable_stop_guard:
        return
    settings_path = workdir / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if settings_path.is_file():  # 别覆盖别人写的 hooks/permissions
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            loaded = None
        if isinstance(loaded, dict):
            data = loaded
    if names:
        overrides = data.get("skillOverrides")
        if not isinstance(overrides, dict):
            overrides = {}
        overrides.update({name: "off" for name in names})
        data["skillOverrides"] = overrides

    if enable_stop_guard:
        hooks = data.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}
        stop_groups = hooks.get("Stop")
        if not isinstance(stop_groups, list):
            stop_groups = []
        already_present = any(
            isinstance(group, dict)
            and isinstance(group.get("hooks"), list)
            and any(
                isinstance(hook, dict)
                and hook.get("type") == "command"
                and hook.get("command") == _STOP_GUARD_COMMAND
                for hook in group["hooks"]
            )
            for group in stop_groups
        )
        if not already_present:
            stop_groups.append(
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": _STOP_GUARD_COMMAND,
                            "timeout": 10,
                        }
                    ]
                }
            )
        hooks["Stop"] = stop_groups
        data["hooks"] = hooks
    settings_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_skill_overrides(workdir: Path, names: list[str]) -> None:
    """Backward-compatible helper retained for callers/tests that only manage skills."""
    _write_claude_settings(workdir, names)


def _prepare_ascendc_workdir(args) -> int:
    """AscendC 分支(与 triton 路径完全并行,triton 逻辑不受影响)。

    - input/{op}.py(+ 同名 .json;NPUKernelBench 的 get_input_groups 需读同名 .json)
    - canonical 的 tools/ + skills/(→ .claude/skills,对齐 init.sh 约定)+ CLAUDE.md
    - 不写 triton stub(agent 自己按 skill 建 {op}/kernel/)
    canonical(--canonical-root)= operator_runtime_ascendc(含 skills/ CLAUDE.md tools/)。
    """
    workdir = Path(args.workdir)
    canonical = Path(args.canonical_root)
    op = args.op_name
    # 不预创建 output/submission/:空目录会让 agent 误以为「骨架不存在、该在这里从零建工程」,
    # 而评测只打包顶层 {op}/(写 tarball 时 pack_submission.sh 会自行 mkdir)。
    for rel in ("input", "judge_out"):
        (workdir / rel).mkdir(parents=True, exist_ok=True)

    task_path = Path(args.task_path or (workdir / "input" / f"{op}.py"))
    if not task_path.is_file():
        raise FileNotFoundError(f"task file missing: {task_path}")
    _copy_file(task_path, workdir / "input" / f"{op}.py")
    json_src = task_path.with_suffix(".json")
    json_dst = workdir / "input" / json_src.name
    if json_src.is_file():
        _copy_file(json_src, json_dst)
    # NPUKernelBench 的 model.py 用 get_input_groups() 读同名 .json(用例规格)。
    # 缺了这个文件 agent 能写完整个 kernel、judge 侧 verification 才炸(烧掉整轮预算才暴露)
    # → 在 prepare 就 fail fast,错误明确指向缺件而不是"对拍失败"。
    if not json_dst.is_file():
        task_text = task_path.read_text(encoding="utf-8", errors="replace")
        if "get_input_groups" in task_text:
            raise FileNotFoundError(
                f"required case file missing: {json_dst} "
                f"({task_path.name} 用 get_input_groups() 读同名 .json)"
            )

    _prepare_tools(canonical, workdir, readonly_tools=args.readonly_tools)  # tools/ 可能是只读 bind mount
    if not (canonical / "skills").is_dir():
        raise FileNotFoundError(f"required directory missing: {canonical / 'skills'}")
    _copy_tree(canonical / "skills", workdir / ".claude" / "skills")
    # 与 install 产物同构:.claude/{skills,workflows}。
    # 不放 agents/ —— 注册进去就是可派发的 subagent,而 subagent 拿不到 CLAUDE.md 的覆盖区。
    _copy_tree(canonical / "workflows", workdir / ".claude" / "workflows")
    _assert_upstream_paths(workdir)
    _copy_file(canonical / "CLAUDE.md", workdir / "CLAUDE.md")
    # 预生成打包骨架(只 agent 侧): require_claude 是 agent prepare 的标记(judge 的
    # eval_prepare 不带)。judge 侧绝不能铺 —— 否则 AGENT_SIDE 检测会把空骨架当提交物打包,
    # 覆盖 agent 的真实提交。只在 agent 侧铺,agent 只写 kernel 数学。
    if args.require_claude:
        _instantiate_kernel_skeleton(
            canonical, workdir, op, task_path,
            json_dst if json_dst.is_file() else None)
        _record_skeleton_hash(workdir, op)
    off_names = _non_project_skills(canonical) if args.only_project_skills else []
    off_names += [
        n.strip()
        for n in (args.skill_overrides_off or "").split(",")
        if n.strip() and n.strip() not in off_names
    ]
    if off_names:
        print(f"[prepare] skillOverrides off ({len(off_names)}): {','.join(off_names)}")
    stop_guard_enabled = bool(
        args.require_claude and (workdir / "tools" / "ascendc_stop_guard.py").is_file()
    )
    _write_claude_settings(
        workdir,
        off_names,
        enable_stop_guard=stop_guard_enabled,
    )
    if stop_guard_enabled:
        print("[prepare] Stop hook enabled: task_complete guard")

    if args.require_claude and shutil.which("claude") is None:
        raise FileNotFoundError("required executable missing on PATH: claude")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--workdir", default="/opt/workspace/agent_workdir")
    parser.add_argument("--canonical-root", default="/opt/canonical")
    parser.add_argument("--task-path")
    parser.add_argument("--submission-path")
    parser.add_argument("--no-stub", action="store_true")
    parser.add_argument("--require-claude", action="store_true")
    parser.add_argument(
        "--backend", default="triton", choices=["triton", "ascendc"],
        help="triton(默认,原样不变)| ascendc(NPUKernelBench input/{op}.py+.json + canonical skills/CLAUDE.md,不写 triton stub)",
    )
    parser.add_argument(
        "--only-project-skills",
        action="store_true",
        help="规则:只保留 canonical/skills 提供的 skill,CLI 自带的一律 skillOverrides=off"
             "(既不列进 prompt 也不许模型调用)。不传=不写 settings.json,行为不变。仅 ascendc 分支。",
    )
    parser.add_argument(
        "--skill-overrides-off",
        default="",
        help="额外要关的 skill 名(逗号分隔),追加在 --only-project-skills 之上;"
             "CLI 升级新增 bundled skill 时可先用它兜住,不必改代码。",
    )
    parser.add_argument(
        "--readonly-tools",
        action="store_true",
        help="Do not copy canonical tools into workdir; expect workdir/tools to be a read-only bind mount.",
    )
    args = parser.parse_args(argv)

    if not SAFE_NAME.fullmatch(args.op_name):
        raise SystemExit(f"unsafe op_name: {args.op_name!r}")

    if args.backend == "ascendc":
        return _prepare_ascendc_workdir(args)

    workdir = Path(args.workdir)
    canonical = Path(args.canonical_root)
    task_path = Path(args.task_path or workdir / "src" / f"{args.op_name}.py")
    submission_path = Path(
        args.submission_path
        or workdir / "output" / "submission" / f"{args.op_name}_impl.py"
    )

    for rel in ("src", "output/submission", "judge_out"):
        (workdir / rel).mkdir(parents=True, exist_ok=True)

    _prepare_tools(canonical, workdir, readonly_tools=args.readonly_tools)
    _copy_tree(canonical / ".agents", workdir / ".agents")
    _copy_file(canonical / "CLAUDE.md", workdir / "CLAUDE.md")
    if not (canonical / "skills").is_dir():
        raise FileNotFoundError(f"required directory missing: {canonical / 'skills'}")
    if not task_path.is_file():
        raise FileNotFoundError(f"task file missing: {task_path}")

    if not args.no_stub:
        _write_stub(task_path, args.op_name, submission_path)

    if args.require_claude and shutil.which("claude") is None:
        raise FileNotFoundError("required executable missing on PATH: claude")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
