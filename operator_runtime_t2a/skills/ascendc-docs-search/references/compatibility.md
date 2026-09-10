# 环境兼容性

T2A 当前 profile 提供 9.0.0 配套 asc-devkit，只读挂载在 `$ASC_DEVKIT_DIR`。
目标芯片以会话注入的 `SOC_VERSION` / `ASCENDC_SOC_VERSION` 为准；工具链位置以
`ASCEND_HOME_PATH` 为准。不使用旧资料中写死的 CANN 8.5.0、A3 或核数来覆盖环境变量。

1. 查 API 的全部同名变体，按文档中的产品支持、函数签名和 dtype 选择。
2. 对齐、stride、repeat 和临时 buffer 限制取决于具体接口与调用形式，读取相应约束章节；
   不把某条 DataCopy 重载的限制推广到所有搬运接口。
3. 当前资料仓把部分仅支持其他芯片的页面标成不可用；遇到此标记继续搜索同名变体，
   不从文件名后缀推断平台或版本，也不照搬不适用的示例。
4. 示例跨版本目录会变化，从 [示例代码目录](example-catalog.md) 或实际 examples 树检索。

检索只用文件操作，不运行 npu-smi 或占卡探测。需要验证实现时统一运行固定 pipeline，
不更改 SOC、精度阈值、warmup 或 repeats。

- [API 文档索引](api-index.md)
- [示例代码目录](example-catalog.md)
