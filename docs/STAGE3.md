# 第三阶段：多 token 端到端 Combine / Return

范围：Base → Source → Gate → Dispatch → Expert Compute → 加权 Local Combine → Return → Source Combine → Base。
沿用第二阶段的专家 batch 和每 Die 共享 compute 模型；Return 在同一事件循环中进入与 Dispatch 共用的 RF FIFO，会推迟后续 Dispatch 和输入到达，不能用阶段 2 结果事后加一个回传常数代替。

## 已选择的参考约定（非硬件事实）

用户选择每 Die 一个独立 FIFO Combine 单元。`combine.resource_model="per_die_fifo"` 是默认模式；weight、Local Reduce、Source Reduce 共用该 Die 单元，与 expert compute 和 RF 独立。`independent` 仅提供理想无限并发敏感性选项。

- 每个 expert 输出先且仅先乘一次原始 gate weight；trace 权重不归一化。
- `combine.local_combine=true` 时，同 token、同 Die 上全部命中 expert 的加权输出就绪后求和；每 Die 产生一份 d 维部分结果。
- singleton 组直接使用加权输出，不执行 Local Reduce。`false` 时每个 expert 独立产生加权部分结果，作为后续 ablation 的基础。
- Source 本地部分结果直接交付，不产生自发自收的无线流。远端部分结果每份发送一个 activation 大小的向量；没有跨 Die 权重搬运。
- Source 仅在本 token 全部预期部分结果到齐后启动 Source Combine，执行求和，不再乘 gate weights，也不等待无关 token。
- 每次 weight 延迟为 `combine.weight_s`；每个多 expert 本地组 Local Reduce 为 `combine.local_reduce_s`；每个 token Source Reduce 使用已有 `timing.source_combine_s`（含 singleton）。默认新增延迟为 0，表示未标定，不表示真实硬件无开销。
- 最终 Source→Base 使用已有固定延迟，不占 THz；Base/Gate 的无限并发固定延迟约定继续保留。因此报告的是此参考模型下的完整 routed-expert 层路径，未计 shared expert 和真实前端吞吐限制。
- FIFO RF 非抢占、每次独占全部总 THz 带宽；Dispatch 和 Return 没有独立频谱预算，也没有 Return 优先级。OFDMA/CA 和五组比较仍在第四阶段。

## 数值验证

每个 `(verification_seed, token_id, expert_id)` 独立生成固定的 d 维合成 expert 输出，取值在 [-1,1)，使用 NumPy float64。生成与事件顺序、路由 RNG、batch 组成无关。
数值路径执行逐 expert 加权、Die 内归并、回传、Source 汇聚，并与独立集中矩阵加权和比较。

BF16/FP8 在本阶段只决定通信字节数；未仿真 FP8/BF16 量化、累加舍入、真实 FFN 或信道误码。数值检查证明的是归并语义和数据依赖，不是神经网络或物理精度模型。

## 验收要点

1. 正确、唯一地覆盖每个 expert 权重和每个预期部分结果；非归一化和零权重不被改写。
2. Local Reduce 等待该 Die 最慢相关 expert，Return 不早于部分结果就绪；Source 不早于全部相关回传/本地部分结果到齐。
3. Combine on/off 数值等价，远端返回数量分别为命中 Die 数/命中 expert 数；本地结果始终无线 0 bytes。
4. 同 Die FIFO Combine 不重叠，不占 expert compute；无跨 token 全局 barrier。
5. 全部无线流共用一个频谱预算；回传确实影响后续 Dispatch 排队和计算时序。
6. 完整层延迟包括最终 Source→Base；专家完成时间另列；过期 batch timer 不延长窗口。
7. 保存数值校验误差、部分结果/输出摘要、完整输出数组及事件/资源时间线，重复运行可重现。
