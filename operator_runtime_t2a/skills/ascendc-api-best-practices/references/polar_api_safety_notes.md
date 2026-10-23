# AscendC API 与同步安全要点

这些规则用于避免常见的编译、越界、死锁和精度错误。具体函数签名、单位和 dtype 支持以
当前 `ASC_DEVKIT_DIR` 和编译工具链为准。

## DataCopy、DataCopyPad 和 LoadData

- 明确每个字段的单位：元素、字节、32B block、分形或行间 stride 不能混用。
- 多行搬运优先用批量 2D 搬运，但必须核对当前 SoC 的 blockCount、stride 和尾部限制。
- 完整对齐块可使用无 padding 的快路径，最后一个不完整块使用 DataCopyPad；两条路径的有效
  长度和输出写回必须一致。
- padding 只填充临时区域，不能把填充值写回正式输出。
- `LoadData_2D`、`LoadData_2D_V2` 和 MX 变体的参数结构、单位和 layout 不可互换；看到
  API 名相近时先查当前版本头文件和示例。

## UB、TPipe 和手动 buffer

- TPipe/TQue 与手动 LocalTensor 偏移都是合法方案，但不能让两套分配覆盖同一物理区域。
- 手动布局要统一计算偏移、保留 32B 或硬件要求的对齐，并把 padding 和双缓冲一起计入 UB 预算。
- 队列 buffer 在使用前后保持明确的所有权：CopyIn 完成后才能消费，消费完成后才能复用。
- 不为解决容量问题把 buffer 改成静态全局或隐式 alias；先重新计算 tile 和生命周期。

## 同步

- EnQue/DeQue、SetFlag/WaitFlag 和 PipeBarrier 的作用范围不同，先画清楚生产 pipe 和消费
  pipe，再选同步方式。
- 跨 pipe 的数据依赖需要对应的 HardEvent；单 pipe barrier 不保证另一条 pipe 已经看到数据。
- Set 和 Wait 必须配对，event id 不能在消费者尚未完成时复用，循环结束要处理残留事件。
- `PIPE_ALL` 会阻塞整个流水线，优先使用方向明确的事件或单 pipe barrier；只有确有依赖时才用
  全管屏障。
- 多核写同一输出时，区分 atomic 顺序误差、写冲突和未初始化内存；不要用同步删减掩盖真实
  数据依赖。

## Cast、Compare 和 dtype

- Cast 的源和目标不能在可能重叠的 buffer 上无序读写；升精度通常需要独立目标空间。
- BF16/FP16 累加优先规划 FP32 workspace，最后一次按 reference 语义降精度。
- Compare、mask 和 Select 的对齐要求要按当前 API 文档核对；不能只凭编译通过判断安全。
- host 的 dtype、layout、stride 和 kernel 模板必须来自同一份 reference 契约，不能用 reinterpret
  让不匹配的输入“看起来”可用。

## Host tiling 与硬件差异

- 核数、UB、对齐、可用 API 和同步资源从当前 SoC 查询；不要照搬历史模板常量。
- 动态 shape 的中间尺寸在 host 侧用足够宽的整数计算，再转换到 API 要求的类型。
- 每个 host 分支都要覆盖原始 case 的 dtype、属性和尾块；不支持某条优化路径时换实现路径，
  不能把合法输入直接排除。
- API 资料解决的是“如何正确实现”，不改变 reference、case、容差、评测入口或提交规则。
