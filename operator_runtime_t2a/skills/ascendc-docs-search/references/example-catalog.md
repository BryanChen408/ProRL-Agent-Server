# 示例代码目录

以下相对路径基于 T2A 当前挂载的 asc-devkit 9.2.0，使用时加 `$ASC_DEVKIT_DIR/`。
跨版本先 Glob `examples/`，再读对应目录的 README.md；不假定旧版目录名仍然存在。
示例支持平台以其 README 与代码为准，当前 SOC 不支持的示例只可阅读，不能直接采用。

| 用途 | 实际入口 |
| --- | --- |
| SIMD C++ 总索引 | `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/README.md` |
| 加法、内存分配与搬运 | `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/00_introduction/01_add/add_tpipe_tque/add_tpipe_tque.asc` |
| 多输入/动态 buffer 示例 | 9.2.0 未保留旧 `add_dynamic` 示例；从 `examples/` 搜索实际资源管理用例，不猜同名路径 |
| 矩阵计算 | `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/00_introduction/02_matrix/README.md` |
| 调试打印 | `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/01_utilities/00_printf/simple_printf/printf.asc` |
| 断言 | `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/01_utilities/01_assert/assert.asc` |
| Addcdiv | 9.2.0 未保留旧示例；在 `docs/zh/api/` 按 `*Addcdiv*.md` 搜索接口说明 |
| 性能优化索引 | `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/05_best_practices/README.md` |
| 兼容性索引 | `$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/06_compatibility_guide/README.md` |

未列出的示例直接从实际文件树检索，无需新建示例或猜旧版 vectoradd/sub 路径：

```bash
find "$ASC_DEVKIT_DIR/examples" -type f -iname '*sub*'
rg -l 'AscendC::Sub|Sub\(' "$ASC_DEVKIT_DIR/examples" -g '*.asc' -g '*.cpp' -g '*.h'
```

只读参考源码。构建、运行、对拍与测速仍统一通过 CLAUDE.md 中的固定 pipeline。

- [API 文档索引](api-index.md)
- [环境兼容性表](compatibility.md)
