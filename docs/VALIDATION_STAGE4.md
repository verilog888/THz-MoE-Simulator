# 第四阶段验证记录

本文件保留第四阶段的验收范围和证据；当前第五阶段扫描和完整回归记录见 [VALIDATION_STAGE5.md](VALIDATION_STAGE5.md)。
2026-11-02 使用当前 0.5.0 代码重建的第四阶段完整数据和全部图表见[结果索引](evidence/stage4/README.md)，与下文历史 0.4.0 验收记录分别标识。

环境：Linux x86_64、CPython 3.12.14；NumPy 2.3.5、Pandas 2.2.3、Matplotlib 3.10.8。包版本 0.4.0；依赖未增加，editable 安装和 `pip check` 通过。硬件参数和未给定规则的具体含义见 [STAGE4.md](STAGE4.md)。

## 实际实施和测试

1. 开始本阶段前运行前三阶段 112 项回归，全部通过。
2. 先完成 13 项无线组件测试，实际运行通过；加入同刻仲裁优先级和共享计算审计重构后，125 项回归通过。
3. 接入完整五组数据流，增加 21 项集成/审计/CLI 测试。一项冲突注入 fixture 最初选到了同一流的相邻片段，没有真正制造冲突；改为另一条同时占用频谱的流后，审计正确拒绝。
4. **完整 146 项测试通过，0 失败，0 跳过。** 复现脚本再次执行全套测试，失败立即停止。

测试源码：[无线组件](../tests/test_stage4_radio.py)、[完整五组路径](../tests/test_stage4_layer.py)。前三阶段测试全部保留。

```bash
# 在仓库根目录执行
bash scripts/setup_env.sh
bash scripts/reproduce_stage4.sh
```

仅测试：

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" \
  .venv/bin/python -m unittest discover -s tests -v
```

## 独立手算与反例

| 用例 | 独立预期 | 实际检查 |
|---|---|---|
| `stage4_tiny` A | 两个远端 expert，各两跳 Dispatch 和两跳 Return，共八个 5 μs hop；compute 在传输间重叠，最后 Source/Base 3 μs | **43 μs，64 bytes** |
| 同例 B | 按 expert 发两份输入和两份输出，共四个 5 μs hop，最后 3 μs | **23 μs，32 bytes** |
| 同例 C/D/Proposed | 广播 0–5、远端专家 5–9、Local Reduce 9–10、Return 10–15、Source/Base 15–18 μs | **18 μs，16 bytes**；单流时 OFDMA 无额外收益 |
| `stage4_concurrent` C | 同带宽顺序发送，最后 Return 在 20 μs 完成 | **20 μs** |
| 同例 D | 两个 Dispatch 同时 0–9；远端专家 9–11、11–13；Return 分别 11–19、13–20.5 | **20.5 μs**，可比 TDMA 慢 |
| 同例 D 统计积分 | active-flow-time=18+8+7.5=33.5 μs；占满频谱等价时长=18.5 μs；净容量=296 bits，payload=256 bits | PEPD=33.5/20.5；U=18.5/20.5；BWwaste=1−256/296 |
| 同例大幅降低算力 | 同时发送确实出现，但计算空闲期使整个观察窗口的平均并发下降 | peak=2 且平均 **PEPD<1**，不把瞬时并发混同平均值 |
| 无线组件延后到达 | 总速率 8 bits/s，2-byte 流 t=0 发，1-byte 流 t=1 加入；保留前 8 bits 后均分速率 | 两流都在 t=3 完成；旧 t=2 完成事件失效 |
| setup 跨重分配 | setup=1 s，两流分别 t=0/0.5 到达，各 1 byte、总速率 8 bits/s | t=3/3.25 完成，每流 setup 恰好一次 |
| F>Nsc | 2 载波、5 个同时到达的 1-byte 流 | FIFO 准入时刻 0/0/2/2/4 s，完成 2/2/4/4/5 s，无丢失 |
| CA 边界 | Nsc=4，priority=[100,0,0,0] 原式超预算 | 修复为 [1,1,1,1]；全零时等权、余数平局确定性 |

## 数据依赖、预算和数值审计

- 每个 logical hop 恰好一次交付，payload bits 等于各 segment 的发送量之和。setup 占频谱，重分配保留 setup/已传 bits；小数 bits 跨多 epoch 测试通过。
- 载波 sweep 检查任意重叠时间的全局冲突，半开区间在同刻先释放后占用；允许同源/同目的 Die 在不同载波上并发。
- CA 分配记录的剩余 bits、归一化 Q/W、目的 Die idle、Return 标记和实际载波与离线状态相符；超预算修复有记录。
- A 的每条第二跳必须在第一跳完整接收后 ready，两跳使用相同 THz 预算。B 按 expert 单播；C/D/Proposed 广播目的去重。
- 所有方案保留本地旁路；Local Combine 的权重只乘一次。慢 expert/Return 的 fan-in、每 Die FIFO Combine、最终 Base 完成均独立检查。
- 故意破坏 segment bits、子载波、priority、admission、relay parent、Source Combine ready、输出向量和因果事件，审计会拒绝。
- 共同预算检查允许不同的字段仅为 `scheduling.scheme` 和 `combine.local_combine`（描述性 name 不参与摘要）；PHY、模型、计算、batch、arrival、routing、Combine 延迟和验证 seed 均相同。
- 额外带宽或变化的 workload 会被公平性审计拒绝。真实 trace 的非归一化/零权重保留，trace 重放和最终向量一致。
- 全本地、单载波、超量流、混合本地/远端和多个广播目的均测试。C 与第三阶段参考路径在手算和反馈用例上相符。

## 复现产物

脚本对 tiny/concurrent/limited/Zipf 四个配置分别运行 A/B/C/D/Proposed 两遍，共 40 次完整仿真。每个配置比较 93 个文件，共 **372 对 CSV/JSON/NPY/PNG**，包括 manifest、完整向量、分段无线明细、比较图和时间线。每组 `fairness.json` 验证共同预算、工作负载、资源配置与最终向量等价。首轮实际通过目录：`results/stage4-yHVSyJ/`；最终图表版本再次全部通过，目录为 `results/stage4-BnKaEt/`，内含 `tests.log` 和 `reproducibility.json`。最终 manifest 的 source SHA-256 与第四阶段验收时的代码一致。

时间线已视觉检查，并为小例增加实际流 ID 和边界，以便辨认同类并发传输。大例采用集合绘图，避免每个 epoch 独立绘制高开销图元。

128-token Zipf 配置完成每方案 1024 个 expert assignments、128 个 d=7168 输出。参考运行如下；这些值仅用于模型复现检查，未证明硬件或论文性能：

| 方案 | 完整层 makespan (ms) | 无线 payload，含所有 hop (bytes) | 最大向量误差 |
|---|---:|---:|---:|
| A | 32.447331520 | 55,451,648 | 0 |
| B | 30.879834592 | 27,725,824 | 0 |
| C | 16.610862384 | 6,866,944 | 2.22e-16 |
| D | 8.787671303 | 6,866,944 | 2.22e-16 |
| Proposed | 8.787653989 | 6,866,944 | 2.22e-16 |

D 和 Proposed 的该次 makespan 几乎相同，不能据此声称 CA 普遍显著优于静态 OFDMA。需要后续多 seeds/扫描以及模型校准。

## 未验证的假设与限制

- 用户选定的两跳无线 Base relay 是参考定义，不是方案已给出的硬件链路参数；Base 不聚合、不引入额外排队计算资源。
- 同 Die 理想多流收发、固定频谱效率、无功率分配/SNR 耦合、免费子载波切换、一次 setup 都未硬件标定。未模拟 OFDM symbol、真实波束切换、FEC 重传或 channel errors。
- CA 的归一化、W=ready 年龄、广播 idle 比例和 Return indicator 是公开的参数化选择；默认系数不代表经过优化的调度策略。准入为非抢占 FIFO，频谱速率可以动态改变。
- float64 合成输出仅验证加权/归并语义；BF16/FP8 只影响字节数。真实 FFN、量化精度、shared expert 和真实 Base/Gate 吞吐限制尚未验证。
- 仅当前锁定依赖、Python 和平台验证逐字节复现；极小 epoch 会增加事件和日志规模，`scheduling.max_events` 提供明确上限。参数扫描、多 seeds 置信区间和 demand-only 独立 ablation 尚未实现。
