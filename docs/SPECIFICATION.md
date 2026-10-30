# 规格审查与待定模型

依据：用户上传的《V-Die PIM 上基于 THz-OFDMA 的 MoE 专家并行：可复现单层仿真方案》，16 页。
SHA-256：`072f0287b6b79494564d0190f3143b348642f3328aaf9d5168eb385b29f40fdf`。
此文档记录模型依据，不将 PDF 中的预期性能结论视为已证事实。

## 明确的规格

| 内容 | 规格 | PDF 页码 |
|---|---|---|
| 层级 | 单 MoE 层；Base → Source → Gate → Dispatch → Compute → Combine → Base | 1、4–6 |
| 专家 | routed expert 完整驻留一个 Die；同一层位于同一 Block；权重不跨 Die 传输 | 1–2 |
| 模型默认 | d=7168、h=2048、E=256、Top-k=8；背景模型还包含 1 个 shared expert | 1–2 |
| 数据精度 | activation BF16/FP8：2/1 字节；权重 FP8：1 字节 | 2 |
| Dispatch | 同 token 的不同目的 Die 共用一次广播；目的集合去重 | 2、5 |
| Local Combine | 每个 Die 将本 token 的本地 expert 输出先乘 gate weight，再求和；每 Die 返回一个 d 维向量 | 5–6 |
| PHY | Δf=B/Nsc；Rsc=Δf·ηSE·ηPHY；Rf=nf·Rsc；T=Tsetup+Sbits/Rf | 3 |
| Compute | F=6dhm；SW=3dh·bw；Nread=ceil(m/R)（或每 batch 一次）；Te=max(F/P,Nread·SW/BWlocal) | 10 |
| 路由 | uniform、Zipf p(i)∝i^(-s)、trace | 10–11 |
| CA 权重 | αQ+βW+γI+δC，整数频谱分配和余数分配 | 6–7 |
| 指标 | makespan、token p50/p95/p99、吞吐、THz/compute 利用率、PEPD、BWwaste | 12–13 |

所有内部时间以秒、带宽以 Hz、吞吐以 bit/s、数据大小以 byte 表示。PDF 同时使用 packet bytes 和 S/R，代码必须显式将 bytes 乘 8。算力采用 FLOP/s，MAC 计为 2 FLOPs，不能将 TOPS 名称直接代入而不解释。

## 问题和建议假设（未获硬件验证）

以下是实现建议，**不是规格事实**。结果必须携带适用假设，不能只在 README 中声明。
表中记录各阶段的适用范围；批处理细节见 [STAGE2.md](STAGE2.md)，第三阶段 Combine/Return 细节见 [STAGE3.md](STAGE3.md)，并随结果导出。

| ID | 未充分定义的问题 | 建议及状态 |
|---|---|---|
| A01 | shared expert 的放置、计算和返回路径？ | 第一阶段仅 routed experts；shared expert 不计入。尚非完整 DeepSeek-V3 层。 |
| A02 | Base↔Source 使用何种链路，是否占用 THz，如何竞争？ | 共同前端/末端采用每 token 独立固定延迟，不计 THz 或 expert compute 占用。第四阶段 A 的中间 relay 另建无线两跳并共享总频谱；用户已选该参考规则，尚未标定。 |
| A03 | Gate、权重乘法和 Local/Source Combine 的算力/带宽开销？ | 用户已选第三阶段参考模型：每 Die 一个 FIFO Combine 单元，weight/Local/Source Reduce 共用它，与 expert compute 和 RF 独立；Gate 沿用独立前端。延迟可配置，新增 weight/local_reduce 默认 0，均未标定；提供非零延迟配置。singleton 免 Local Reduce，Source Reduce 保留固定开销。 |
| A04 | Pcompute、BWlocal、ηPHY、Tsetup 实测值？ | 均为配置参数。示例值仅验证模型，不作为硬件或论文结果。 |
| A05 | 如何聚合 expert batch，何时关窗，每 Die 多 expert 是否共享算力？ | 用户已选参考规则：每 expert 固定 half-open 窗口或达到 cap 提前关窗；每 Die 一个共享计算资源，已关窗 batch 按关闭时间/序号 FIFO。第二阶段已实现窗口、争用和 reuse tile R；硬件标定未完成。 |
| A06 | 一个 Die 能否同时发/收、多子载波并发接收？广播如何占接收资源？ | 用户选择理想模型：不同子载波允许同 Die 多流收发。所有流必须全局频谱正交，不假设空间复用。第四阶段已验证同端点 OFDMA 并发及频谱约束；硬件能力尚未验证。 |
| A07 | Top-k 是否允许重复、gate weights 的来源/归一化、Source 上 expert 如何处理？ | 无放回选 k 个唯一 expert；合成路由等权 1/k，trace 保留原始权重；本地命中和本地部分结果不发无线包。阶段 2/3 已支持多 token/Top-k。阶段 3 验证每 expert 仅加权一次，Source 汇聚不再次加权。 |
| A08 | setup 是否占用子载波？header 是否额外加字节？ | 第一阶段 setup 占用全部已分配子载波；ηPHY 已涵盖 header/FEC/guard，不重复加 header；有效净速率用于容量统计。 |
| A09 | 返回优先级 C 如何定义？广播 I 是任意/全部目的 expert 空闲？Q/W 是否归一化？ | 第四阶段使用 Q=remaining bytes/Qref、W=ready 后年龄/Wref、I=无运行/已关闭排队 batch 的目的 Die 比例、C=Return 标记；配置公开归一化和四个系数。这是参考参数化解释，不是 PDF 给定常数；详见 STAGE4。 |
| A10 | CA 的 max(1,floor(...)) 超预算，或 F>Nsc 时怎么办？ | 用户已选 FIFO 准入 F≤Nsc，活跃流保留至少一个载波直到完成。CA 原式可行时照用；超预算时先保底 1 再按剩余预算最大余数分配，记录修复标记。静态 floor 余数空闲。 |
| A11 | epoch 长度、重分配、抢占后的 setup、剩余 bytes 的精度？ | 用户已选首次一遍 setup、免费重分配、不重发 payload；CA 在事件触发之外以配置 epoch 限制最大刷新间隔。浮点剩余 bits 和 generation 事件失效已测试；无线切换与 symbol 粒度尚未标定。 |
| A12 | Baseline-A relay 每一跳的介质和预算、B 是否按 expert 或 Die 去重？ | 用户已选 A 每远端 expert 经 Source→Base→Expert / Expert→Base→Source，无线每跳共享同一预算，Base 仅转发。B 每 expert 单播并独立返回；所有方案本地命中旁路。第四阶段已实现并审计。 |
| A13 | 吞吐窗口、PEPD/利用率是否包括 setup、分位数算法？ | 阶段 1/3 窗口从最早 arrival 到最晚 Base 完成；阶段 2 截止于 expert 完成，明确使用部分流水线指标名。token latency=完成−自己的 arrival；PEPD/U 含占用 setup，并额外输出 payload-only PEPD；分位数采用 NumPy linear。 |
| A14 | 是否模拟真实 FFN、数值误差、SNR、重传或失败？ | 阶段 3 使用每个 assignment 独立生成的确定性 float64 合成向量，已验证层次归并与集中加权和一致。BF16/FP8 只影响字节数，未模拟数值量化/舍入；真实神经推理、信道校准、重传和电磁仿真尚未验证。 |
| A15 | broadcast 和 Local Combine 独立关闭时如何标记基线？ | 第五阶段新增明确标识的 Ablation 模式，独立改变两个开关，在 TDMA/static/CA 上比较；正式五组定义不变。 |
| A16 | 随机重复次数、误差条及加速比如何统计？ | PDF 未规定。参考计划使用 5 个路由种子、种子级均值/样本 std、同种子配对比值和 2000 次确定性 percentile bootstrap 95% 区间，明确标为探索性；单种子不生成 std/区间。 |

初始均衡放置使用 contiguous；后续将 mapping 策略作为独立变量。
第二阶段实现 fixed / round-robin Source 和 trace 来源，不声称这些策略最优。
用户已同意先使用明确标注的参考配置并暂不计 shared expert；这确认了仿真范围，不等于完成硬件标定。

## 五组对照及公平性门槛（第四阶段已实现）

| 组别 | Dispatch | PHY | Combine |
|---|---|---|---|
| A | Base relay / serialized | TDMA | Base relay |
| B | Direct P2P | TDMA | P2P |
| C | Broadcast-aware | TDMA | Local Combine |
| D | Broadcast-aware | Static OFDMA | Local Combine |
| Proposed | Broadcast-aware | CA-OFDMA | Local Combine |

五组必须共享 total_bandwidth_hz、Nsc、ηSE/ηPHY、setup 规则、placement、routing trace、计算资源和 arrival。
TDMA 独占全部 B；OFDMA 只能切分 B，不能复制带宽。输出配置/工作负载摘要用于自动审计。
Baseline-A 和调度补充假设已记录于 [STAGE4.md](STAGE4.md)。当前五组参考结果通过共同配置/工作负载摘要和数值等价检查，不等于硬件验证。

## 计划扫描（PDF 第 14 页）

- tokens：1、8、32、128、512、2048（规格范围也包含 1–4096）
- dies：8、16、32、64；E=256；Top-k：1、2、4、8
- Nsc：16、32、64、128、256、512（概述还允许到 1024）
- B：16、32、64、128、256 GHz；ηSE：1、2、4、6 bit/s/Hz
- Zipf s：0、0.5、1、1.5、2；activation：FP8/BF16
- broadcast / Local Combine：off/on；scheduler：TDMA/static/CA

不默认执行全笛卡尔积；先一因素扫描及小规模组合，保存 seeds 和配置。
第五阶段已提供完整 PDF 取值的 29 个唯一单因素点（5 种子×5 方案，共 725 次仿真）的可运行计划；本阶段仅预览该完整计划，实际执行代表性子集及独立消融，详见 [STAGE5.md](STAGE5.md) 和验证报告。共同带宽约束在每个参数点内部检查，带宽轴上的不同点有意改变总 B。
不存在“CA 必须更快”测试：等速、无 setup 的同带宽流体模型中，并行本身不增加总容量。PEPD>1 只在实际有并发时成立。
