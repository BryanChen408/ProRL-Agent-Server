# AscendC 算子打包骨架（自包含，可复制即用)

> **注意（签名生成化之后）**：骨架目录里的 `model_new_ascendc.py` / `register.cpp` /
> `ops.h` / `op_host/{op_name}.cpp` / `op_kernel/{op_name}_kernel.cpp` 这 5 个**签名件**
> 是「单 tensor 算子」的示例。prepare 铺骨架时**不会直接拷贝它们**，而是按本题
> model.py 的 `__init__`/`forward` 签名（参数个数/类型/默认值/可选位/返回 arity）
> **逐题生成**；只有解析整体失败时才回退拷贝这里的单 tensor 版本。
> 所以这里的内容只作参考样例，实际工程以 workdir 里生成的为准。
> 机制件(CMakeLists.txt / setup.py / utils/)仍是原样静态拷贝。

这份骨架是**和判分链路（`ascendc_eval_pipeline.sh` + `build_ascendc.py` +
`verification_ascendc.py`）实测兼容的打包契约**，结构取自已跑通的
`archive_tasks/rms_norm` 与 output1 的 67 个算子。把它整份复制成你的算子目录，
只改标了 `{op_name}` / `{{ ... }}` 的地方和 kernel 数学,**打包相关的四件套
（CMakeLists.txt / register.cpp / setup.py / model_new_ascendc.py 顶部）原样保留**。

## 用法

```bash
cp -r .claude/workflows/templates/kernel_skeleton {op_name}
# 然后把 {op_name} 换成你的算子名(和数据集 op_name 一字不差),填 kernel 数学
```

## 三条硬规则（违反必判 op_not_registered / 编译失败）

1. **扩展名用扁平名 `{op_name}_ext`，不要用嵌套名 `{op_name}._C`。**
   嵌套名会把 `.so` 编进包内子目录，loader 找不到 → `ModuleNotFoundError`。
2. **`model_new_ascendc.py` 顶部必须保留「先 `import` 失败、再
   `torch.ops.load_library`」的兜底。** `register.cpp` 用 `TORCH_LIBRARY`（没有
   `PYBIND11_MODULE`），裸 `import {op_name}_ext` 会因缺 `PyInit_` 报 ImportError；
   真正注册靠 fallback 的 `load_library(kernel/build/{op_name}_ext*.so)`。
3. **注册走 `TORCH_LIBRARY` + `load_library`，不要走 `import` 路线。**
   不要给 register.cpp 加 `PYBIND11_MODULE`，也不要指望 wheel/`pip install`——
   judge 对 wheel 安装失败是放行的,load_library 直接读 `kernel/build/` 才是硬路径。

## 判分链路怎么用它

1. `pack_submission.sh` 把 `{op_name}/` 打成**纯源码** tarball（排除 build/、*.so、*.whl)。
2. judge 解包到全新 `judge_out/work/`，再清一遍残留 build/.so。
3. `build_ascendc.py` 用你 `kernel/CMakeLists.txt` 编出
   `kernel/build/{op_name}_ext.cpython-*.so`。
4. `verification_ascendc.py` 把 `kernel/build` 加进 `sys.path`,import 你的
   `model_new_ascendc.py` → 触发 fallback `load_library` → `torch.ops.npu.{op_name}` 可解析。
5. 对拍:golden `model.py`（judge 注入）vs 你的 `ModelNew`。

## 文件清单

```
{op_name}/
├── model_new_ascendc.py        # 入口;顶部加载逻辑别动,只写 ModelNew.forward
├── model.py                    # 可不交(judge 会注入 golden 覆盖)
└── kernel/
    ├── CMakeLists.txt          # 编译配方;只改 {op_name}
    ├── register.cpp            # TORCH_LIBRARY 注册;只改 op 名和签名
    ├── ops.h                   # host 函数声明
    ├── setup.py                # 自包含(已内联 kernel_setup);只改 {op_name}
    ├── op_host/{op_name}.cpp   # host:校验 + tiling + EXEC_KERNEL_CMD 启动
    ├── op_kernel/{op_name}_kernel.cpp  # device kernel;数学写在这里
    └── utils/torch_kernel_helper.h     # EXEC_KERNEL_CMD 宏(原样,别改)
```

kernel 的 `Compute()` 里默认是 `DataCopy`（恒等拷贝，能编过、能注册）。
**把它换成你算子的真实数学** —— 那是你唯一该动的计算逻辑。
