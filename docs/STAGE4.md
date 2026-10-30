# 第四阶段：五组同带宽对照与动态频谱

沿用前三阶段的 batch、Die 共享计算和独立 FIFO Combine。第四阶段加入独立无线控制器，记录每个传输时间片的真实子载波、净速率和已发送 bits；频谱重分配后保留已传数据。完整层结束点仍是所有 token 回到 Base。

## 用户已选择的参考规则

下列规则补充 PDF 未明确的部分，属于模型假设，未经硬件验证；每次导出的 `manifest.json` 都包含它们。

| 方案 | Dispatch | Return / Combine | 无线调度 |
|---|---|---|---|
| A | 每个远端 expert 分别 Source→Base→Expert | 每 expert 先乘一次权重，再 Expert→Base→Source；Source 求和 | 单流全带宽 FIFO TDMA |
| B | 每个远端 expert 单独直连单播，即使多个 expert 同 Die | 每 expert 独立返回；Source 求和 | 同上 |
| C | 同 token 所有远端 Die 去重后一次广播 | 每 Die Local Combine 后返回 | 同上 |
| D | 同 C | 同 C | 活跃流均分：每流 floor(Nsc/F)，余数空闲 |
| Proposed | 同 C | 同 C | CA-OFDMA |

- A 中 Base 是 ID=`num_dies` 的独立转发端点，没有 expert。每跳完整接收后才转发，转发处理延迟为 0；两跳分别收取 setup 和 payload，均占同一个总 THz 频谱。Base 不合并。所有方案的初始 Base→Source 和最终 Source→Base 仍使用共同固定延迟配置；中间 relay 的无线 hop 单独记账。
- 所有方案本地命中直接计算，本地部分结果直接交给 Source；没有自发自收流。A/B 关闭 Local Combine；C/D/Proposed 开启。该开关由方案控制并写入有效配置。
- TDMA 单流占全部 Nsc。OFDMA 最多准入 Nsc 条流，活跃流保留准入直到完成，每流至少一个子载波；超出的流按 `(ready time, flow ID)` FIFO 等待。活跃流不会被新流抢出，有限正 payload 在有限队列内可完成。
- 同 Die 可在互斥子载波上多流收发；整个 Block 无空间复用。Base 在 A 中也使用同一全局预算。
- 新流到达/现有流完成时重新分配；CA 在活跃期间还按 `epoch_s` 限制两次更新的最大间隔。事件触发更新会重置这个间隔，未使用独立固定 frame 时钟。静态方案没有周期刷新。
- 每跳只在首次获得频谱时支付一次 setup，setup 占用获分配的子载波。重分配不重启 setup、不重发 payload、不额外收取切换开销。连续 bits 进度允许浮点小数；不是 packet fragmentation 或实际 OFDM symbol 模拟。
- 同一时刻的输入、compute/Combine 完成和零延迟因果链先处理，随后无线仲裁，避免由第一个入队 token 独占起始频谱。旧预测完成事件有 generation 标记，失效后不能触发交付。

## CA 权重的显式参数化

PDF 给出 `alpha*Q + beta*W + gamma*I + delta*C`，未规定单位标度、广播 I 或 C 的具体取值。本实现采用以下可复现参考解释，而不称其为论文给定常数：

- `Q = remaining_bytes / q_ref_bytes`。
- `W = (now - ready_time) / w_ref_s`，是从 ready 开始的年龄，包含已服务时间；FIFO 准入和至少一个子载波提供基本进展保证。
- Dispatch 的 `I` 为目的 Die 中既无运行 batch、也无已关闭排队 batch 的比例；Return 的 I=0。它是共享 Die 算力下的空闲代理，不预知未来请求；尚未关闭的 batch 窗口不算 busy。
- Return 的 C=1，Dispatch 的 C=0；没有假定最后一份返回的额外权重。
- 默认 alpha=beta=gamma=delta=1、q_ref_bytes=14336、w_ref_s=10 μs、epoch_s=10 μs，均为参考参数。所有权重可非负调整；全 0 时按等权分配。

优先使用 PDF 的 `max(1, floor(Nsc*priority/sum(priority)))`，可行时按最大 fractional remainder 补齐剩余载波，平局按 flow ID。若保底 1 导致超预算，则先为所有已准入流保留 1，再把剩余预算按权重及最大余数分配；`allocations.csv` 的 `repaired_overflow` 标出使用修正规则的时刻。F>Nsc 由准入队列解决，不能负分配或静默丢流。

## 接口与检查顺序

1. `radio.allocate_counts`：纯分配函数；先测 TDMA、静态 floor、CA 舍入/零权重/超预算。
2. `radio.SharedRadio`：独立事件控制器；测试 setup、同刻到达、速率变化、剩余 bits、F>Nsc、旧 completion。
3. `scheduled_layer.run_scheduled`：复用已有计算/Combine 引擎，接入五种数据流。
4. `radio_audit.audit_radio`：离线积分、全局载波冲突、一次 setup、每跳 bit 守恒、FIFO 准入、CA 特征/分配记录。
5. `scheduled_metrics.ScheduledResult.audit`：广播/relay/Return 逐路径依赖、batch/FIFO Compute、加权/局部/Source Combine、集中向量 oracle、完整层事件与指标。
6. `experiments.run_comparison`：五组共同配置摘要、workload 摘要、最终向量等价；只有方案及其 Local Combine 开关允许不同。

## 导出与解释

`flows.csv` 是逻辑单跳，记录 ready/admitted/completed 和 parent hop；`radio_segments.csv` 是实际占频谱的时间片；`allocations.csv` 记录每次分配的 Q/W/I/C、priority、真实载波和超预算修复。重分配可能使一条 flow 产生多行 segment，不能按 segment 重复累加完整 payload。

无线利用率、PEPD、容量和 BWwaste 从时间片积分，setup 计入占用，payload bytes 按 hop 只统计一次。静态方案未分配的余数影响利用率，但不进入 PDF 的“allocated capacity”分母。`wireless_hops` 包含 A 的两跳 relay；`dispatch_flows`/`return_flows` 只统计最终交付 hop，额外 hop 另报。

五组比较导出 `comparison.csv`、`fairness.json`、由 CSV 绘制的 `comparison.png`，以及每方案完整明细和可选 `timeline.png`。时间线上半部分绘制实际子载波/时间占用，下半部分绘制 expert/Combine 资源。

全局总带宽相同并不保证 OFDMA 更快；并发峰值大于 1 也不保证完整观察窗口的平均 PEPD>1。固定速率/零切换开销、理想多流端点、有限 payload 和未校准 compute 都限制结论。第五阶段再实现批量参数扫描和多 seeds 统计。
