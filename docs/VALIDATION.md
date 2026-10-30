# 阶段 1 验证记录

本文件保留第一阶段的验收范围和证据。当前第五阶段和 179 项完整回归结果见 [VALIDATION_STAGE5.md](VALIDATION_STAGE5.md)。下文“尚未验证”描述第一阶段验收时的范围。

## 执行范围

在当前 Linux x86_64 云环境、CPython 3.12.14 下执行。NumPy 2.3.5、Pandas 2.2.3、Matplotlib 3.10.8，传递依赖见 `requirements.lock`。

- 首批 17 项公式、事件队列、分配和路由测试全部通过。
- 集成后共 42 项 `unittest` 测试通过；无失败、无 skip、无 expected failure。
- 真实 CLI 运行、Pandas CSV 回读、Matplotlib Agg PNG 输出已执行。
- 每个示例运行两次并比较所有 CSV/JSON/PNG 的脚本：`bash scripts/reproduce_stage1.sh`。
- 时间线已作视觉检查，参考轨迹的 setup、传输、compute 和最终 Base 完成时刻与手算一致。

测试源码：[基础测试](../tests/test_primitives.py)、[集成和审计测试](../tests/test_reference.py)。
运行日志由重现脚本放在每次新建的 `results/stage1-*/tests.log`，比较清单在 `reproducibility.json`。

## 独立手算 oracle

`stage1_tiny.json` 是测试 fixture，不是硬件参数：

- activation：4 × 2 = 8 bytes = 64 bits。
- 净无线速率：16 MHz × 2 × 0.5 = 16 Mbit/s。
- 每个包：1 μs setup + 64/16e6 s payload = 5 μs。
- Expert FLOPs：6 × 4 × 2 = 48；48 / 48e6 = 1 μs。
- 权重：3 × 4 × 2 × 1 = 24 bytes；24/12e6 = 2 μs，故 roofline = 2 μs。
- 总时间：Base 2 + Gate 3 + Dispatch 5 + Compute 2 + singleton bypass 0 + Return 5 + Source Combine 1 + Base 4 = **22 μs**。
- 事件时间（μs）：`2, 5, 5, 10, 12, 12, 12, 17, 18, 22`。
- UTHz = PEPD = 10/22；payload PEPD = 8/22；Die 1 compute 利用率 = 2/22。
- 无线 payload = 16 bytes，已分配净容量 = 160 bits，BWwaste = 1−128/160 = 0.2。

`stage1_reference.json` 的独立手算：每个包 0.1+2.24=2.34 μs，expert compute 受内存限制为 44.040192 μs，总计 **48.720192 μs**。这是 Top-1 且没有 shared expert 的未标定参考值，不是 DeepSeek-V3 完整层性能。

## 主动构造的错误

自动测试验证以下情况被拒绝：过去/NaN/Inf 事件、零时间无限循环、重复/越界子载波、同时占用同一载波、速率超出分配容量、payload bits 不守恒、流超出 token 生命周期、共享 Die compute 重叠、compute 完成早于 Dispatch、遗漏最终回到 Base、重复/缺失事件和区间、错误精度/字段/参数，以及阶段 1 不支持的配置。

非退化统计样本 `[1,2,3,4]` 独立校验 linear p50=2.5、p95=3.85、p99=3.97。合成的两条正交流验证指标积分为 PEPD=2、UTHz=1；这仅验证离线统计与理想端点模型，**不代表已实现 OFDMA 调度器**。

## 发现和处理

首次绘图在只读 home 下触发 Matplotlib/fontconfig 缓存提示；重现脚本为两个缓存目录设置可写路径。后续测试和重现使用该环境设置。

尚未验证：真实硬件参数、shared expert、真实 routing trace、expert batch 窗口/共享资源调度、加权 Local Combine、多 token 通信/计算 overlap、CA 权重和 epoch、Baseline-A relay 语义、五组性能实验及扫描。详见 [假设表](SPECIFICATION.md)。阶段 1 测试通过不构成这些能力的正确性证明。
