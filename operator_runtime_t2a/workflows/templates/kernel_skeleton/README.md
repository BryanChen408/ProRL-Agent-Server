# AscendC 算子打包骨架（由 Polar prepare 实例化）

> **注意（签名生成化之后）**：骨架目录里的 `model_new_ascendc.py` / `register.cpp` /
> `ops.h` / `op_host/{op_name}.cpp` / `op_kernel/{op_name}_kernel.cpp` 这 5 个**签名件**
> 是「单 tensor 算子」的示例。prepare 铺骨架时**不会直接拷贝它们**，而是按本题
> model.py 的 `__init__`/`forward` 签名（参数个数/类型/默认值/可选位/返回 arity）
> **逐题生成**；只有解析整体失败时才回退拷贝这里的单 tensor 版本。
> 所以这里的内容只作参考样例，实际工程以 workdir 里生成的为准。
> 机制件(CMakeLists.txt / setup.py / utils/)仍是原样静态拷贝。

这份骨架是**和判分链路（`ascendc_eval_pipeline.sh` + `build_ascendc.py` +
`verification_ascendc.py`）实测兼容的打包契约**，结构取自已跑通的
`archive_tasks/rms_norm` 与 output1 的 67 个算子。它是工程起点，**不是本题的正确实现**。
保留 CMakeLists.txt / setup.py / utils / wrapper loader；按原始 reference 实现数学、
输出与布局。接口变化时同步 wrapper 调用、register.cpp、ops.h 和 op_host。

初始化职责参考本机 cannbot 原仓 `97ee3d81` 的
`plugins-community/tilelang2ascendc-ops-generator/agents/tilelang2ascendc-kernel-generator.md`
Phase 1/4：保留工程机制，host/kernel 的计算由算子设计填充。Polar 仅额外预接 reference 签名。
当前 host 默认明确报未实现，device 只留入口；不预生成全类型分派、输出分配、tiling 或搬运。

## 用法

Polar prepare 已生成工作目录顶层的 `{op_name}/`，直接原位开发，不要再次复制模板或
另建工程。设计复用 `.claude/workflows/templates/design-template.md` 和本地 tiling/API
知识文件，执行方式服从 `CLAUDE.md` 与 translator 的内嵌设计指引。

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
├── model_new_ascendc.py        # 入口;保留 loader，按 reference 补全 init/forward 与接线
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

host 默认拒绝运行；device 空入口没有计算语义。实现真实计算后才能评测通过。
dtype 分派、输出分配、混合类型、tiling 与尾块由本题设计决定，不从工程骨架推断。
原仓的 rms_norm 分派仅作具体算子的参考，不自动套用到所有任务。
