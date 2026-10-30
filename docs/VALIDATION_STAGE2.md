# 第二阶段验证记录

本文件保留第二阶段的验收范围和证据；当前第五阶段及 179 项完整回归结果见 [VALIDATION_STAGE5.md](VALIDATION_STAGE5.md)。下文限制描述第二阶段验收时的范围。

CPython 3.12.14、NumPy 2.3.5、Pandas 2.2.3、Matplotlib 3.10.8，Linux x86_64。
没有新增依赖，仍使用固定的 `requirements.lock`。

## 自动化检查

- 第一阶段原有 42 项回归测试通过。
- 第二阶段组件增加 17 项，集成/导出增加 22 项，总计 **81 项通过，0 失败，0 跳过**。
- 先通过组件检查，再执行真实多 token 引擎和集成测试。
- 组件：[test_stage2_components.py](../tests/test_stage2_components.py)。
- 集成：[test_stage2_pipeline.py](../tests/test_stage2_pipeline.py)。
- 重现命令：`bash scripts/reproduce_stage2.sh`，运行完整测试，再对 tiny/batching/trace/Zipf 四个配置各运行两次，比较所有 CSV、JSON 和 PNG 的字节。
- 日志：脚本新建的 `results/stage2-*/tests.log`；重复性证据：同目录 `reproducibility.json`。每次结果有完整配置、trace、源码及输入摘要。

## 独立手算与观察结果

| 用例 | 手算/应有行为 | 验证结果 |
|---|---|---|
| 两 token、每个命中一个本地和一个远端 expert | RF 0–5、5–10 μs；Die 0 compute 0–2、2–4；Die 1 compute 5–7、10–12；token 完成 7、12 μs | makespan **12 μs**；p50=9.5、p95=11.75、p99=11.95 μs；4 assignments、2 broadcasts |
| 多目的广播：一个本地，另三个 expert 位于两个远端 Die | 只传一份 8-byte activation，目的集合 {1,2}；同 Die 两 expert 串行 | **9 μs**；一个无线包；3 个远端输入各计算一次 |
| 不同 Die 并行 | 本地 compute 0–10、RF 0–5、远端 compute 5–15 μs | **15 μs**，验证 compute 与 RF 重叠及跨 Die 并行 |
| 五 token 全本地，arrival 间隔 0.5 μs，窗口 2 μs、cap=3、R=2 | 第一 batch m=3 在 1 μs 关窗，读两次权重，compute 1–5；第二 batch m=2 在 3.5 关窗，排队后 compute 5–7 | **7 μs**，0 RF bytes，2 batches、3 weight reads；Die 忙碌时间只计 **6 μs** |
| 同上，每 batch 权重仅读一次 | 第一 batch compute 1–4，第二 4–6 | **6 μs** |
| 同上，窗口 0 | 5 个 singleton，每个 2 μs，共读 5 次权重 | **10 μs** |
| cap 提前关窗、过期 timer 在 100 μs | 两 token 在 0 μs 成 batch，2 μs 完成；过期 timer 无作用 | makespan **2 μs**，不误报为 100 μs |
| 窗口截止时刻到达 | 当前窗口右开，恰好到达截止时刻的请求进入下一 batch | batch membership 和结果不依赖 deadline/arrival 的同刻处理顺序 |

论文规模参数的阶段 2 示例：`stage2_zipf.json` 使用 128 token、256 experts、Top-8、16 Dies、Zipf 1.5；观察到 **1024 assignments、128 broadcasts、645 batches、651 weight reads**，部分流水线 makespan 为 **16.608152384 ms**。这是未标定模型的运行证据，不是完整 MoE 层结果，也不用于宣称 OFDMA 收益。

## 因果和统计审计

- 每个 `(token, expert)` 对恰好进入一个 batch，执行一次；每个 token 的阶段完成时间是其最慢 expert 的完成时间。
- 本地输入在 Gate 完成即可使用；远端输入不得早于对应广播结束；不在广播中重复计入 Source。
- 同 Die compute 不重叠；已关窗 batch FIFO，无无故空闲；不同 Die 可并行。
- batch 关闭、大小、成员、FIFO、reuse tile、FLOPs、权重字节数均核对；区间和事件日志核对。
- RF 序列、频谱和 payload bit 守恒核对。增加 Nsc 不改变 FIFO 服务的总吞吐预算。
- 主动损坏 assignment、广播目的、请求 ready 时间、计算开始、完成时刻、事件和区间，确认审计拒绝错误轨迹。
- 合成 trace 导出后重放，workload、transmissions、batch executions、completions 完全一致；真实 trace 的非归一化 gate weights 原样保留。
- all-local 的 `flows.csv` 保留列定义，可被 Pandas 正常读取；0 传输不生成伪无线流。

## 未验证范围

硬件参数和无限并发 Base/Gate 前端未标定。batch 窗口规则是用户接受的参考约定；没有真实调度 trace 或硬件结果作校准。
stage 2 截止于 Expert Compute，不含多 token Local Combine、Return、Source Combine、shared expert；不输出完整层延迟。
无线服务仍为 FIFO 单流全带宽；尚未实现 OFDMA 调度、五组对照和参数扫描。
浮点秒采用双精度，同刻 FIFO 和窗口边界依据实际事件时间；无法表示的时间推进会明确报错，未实现 cycle-accurate 时钟模型。
