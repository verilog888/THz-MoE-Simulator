# 第三阶段验证记录

本文件保留第三阶段的验收范围和证据；当前第五阶段及 179 项完整回归结果见 [VALIDATION_STAGE5.md](VALIDATION_STAGE5.md)。下文限制描述第三阶段验收时的范围。

环境：Linux x86_64、CPython 3.12.14；NumPy 2.3.5、Pandas 2.2.3、Matplotlib 3.10.8。未引入新依赖。

## 实际执行的检查

- 前两阶段 81 项测试全部保留通过。
- 新增 8 项加权向量组件测试，先执行后接入引擎；随后新增 23 项端到端/导出检查。
- **共 112 项测试通过，0 失败，0 跳过。** 命令：`python -m unittest discover -s tests -v`。
- 共享引擎重构后先运行回归；第三阶段 Return 通过同一个 RF 队列影响后续 Dispatch，没有事后叠加常数。
- 测试源码：[数值组件](../tests/test_stage3_combine.py)、[端到端与审计](../tests/test_stage3_layer.py)。
- `bash scripts/reproduce_stage3.sh` 运行完整测试，再将 tiny / no_local_combine / return_feedback / Zipf 四个配置各运行两次，比较 CSV、JSON、NPY、PNG，并检查 Combine on/off 使用同一 workload 和 PHY 配置、输出数值等价。
- 新运行目录 `results/stage3-*/` 保存 `tests.log`、`reproducibility.json` 和全部导出。详细模型选择见 [STAGE3.md](STAGE3.md)。

本次实际运行：`results/stage3-AMauNr/`，完整测试 112/112 通过；四个配置共 8 次 CLI 运行，每组 16 个导出文件全部逐字节一致（共 64 对文件比较），包括完整输出数组和时间线图片。Local Combine on/off 的 workload SHA-256、完整 PHY 配置相同，输出数值等价。环境安装脚本和 `pip check` 通过，包版本为 0.3.0。

上述 Zipf 参考配置的实际输出：完整层 makespan 为 **16.610862384 ms**，128 个 Dispatch、351 个 Return、返回 payload 为 **5,031,936 bytes**，最终输出最大绝对校验误差为 **2.220446049250313e-16**。这是未标定参数下的复现检查记录，不是硬件性能结论；不能与硬件或后续五方案实验直接比较。

## 独立手算 oracle

| 用例 | 预期 | 实际验证 |
|---|---|---|
| 原第一阶段 singleton | 完整回到 Base 为 22 μs | 新端到端引擎与原路径的 transmissions/completions 相同 |
| Local Combine on | Dispatch 0–5；远端专家 5–7、7–9；Local Reduce 9–10；Return 10–15；Source Reduce 15–16；Base 16–18 μs | **18 μs**，1 个 Return，8 bytes 返回 |
| Local Combine off | 两个 expert 分别 Return 7–12、12–17；Source Reduce 17–18；Base 18–20 μs | **20 μs**，2 个 Return，16 bytes 返回；最终向量与 on 一致 |
| Combine 成本很高 | Local Reduce 从 1 μs 改为 100 μs | **117 μs**，比 off 慢；不预设优化必然提高性能 |
| Return 反馈 | t0 Dispatch 0–5、Return 7–12；t1 在 8 μs 到达，其 Dispatch 被推迟至 12–17，Return 19–24 | **24 μs**；t1 expert 完成由无回传模型的 15 μs 推迟到 19 μs |
| 多目的回传 | 快 Die Return 7–12，慢 Die Return 12–17 | Source ready 恰为 **17 μs**，不在最早返回或全部 expert 刚完成时提前启动 |
| 全本地三个 expert | compute 0–6；Local Reduce 6–7；Source Reduce 7–8；Base 8–10 | **10 μs**，RF bytes=0，无自发自收流 |
| FIFO Combine 单元 | 两个本地 expert compute 0–2、2–4；weight 2–6、6–10；Source Reduce 10–11、11–12 | token 完成 **11、12 μs**，expert compute 未被 Combine 占用 |
| FP8 字节数 | fixture activation 从 8 bytes 变为 4，数值仍是 float64 语义检查 | 完整延迟 **14 μs**；总无线 8 bytes；输出与 BF16 字节配置一致 |

## 数值和因果检查

- 手工向量/非归一化权重的 Local→Source 求和与独立结果 `[0.25, 5.5, 3.75]` 一致。
- 完整路径使用确定性的 assignment-local 合成 expert 向量，与独立集中矩阵加权和比较；不是用同一层次归并函数计算期望值。
- 故意将 gate weight 乘两次，独立 oracle 会拒绝运行；零权重和真实 trace 的非归一化权重不被改写。
- 检查每个 `(token, expert)` 恰好一个 weight job、每个期望部分结果恰好一次交付、Return 指向原 Source、Source 等待全部相关部分结果。
- 主动破坏 partial 内容、delivery、output、ready 时间、最终 Base 完成、事件和区间，确认审计拒绝错误轨迹。
- 每 Die 的 weight/Local/Source Reduce FIFO 与专家 compute 独立；逐资源 busy interval 检查，Dispatch/Return 统一频谱预算。
- 没有跨 token barrier；过期 batch timer 不延长完整层统计窗口；arrival offset 不被算入 token latency。
- 128 token、Top-8、256 experts、16 Dies 的配置完成 1024 assignments 和 128 个 d=7168 输出；每个输出经过数值检查，Return 个数核对为逐 token 的远端命中 Die 数之和。
- 导出的合成 trace 重放后，调度和数值均一致，即使改变不再使用的 routing seed。数值 verification seed 与路由 seed 独立。

## 局限

参数和 Base/Gate/Combine 资源假设尚未经过硬件校准。数值为 float64 合成数据，未运行真实 FFN，BF16/FP8 只影响字节数；未验证量化误差、信道误码或重传。shared expert 仍未计入。
当前无线服务为共享 FIFO、单流全带宽；尚未实现静态 OFDMA、CA-OFDMA、五组完整对照或参数扫描。当前结果不能证明 OFDMA 收益。
