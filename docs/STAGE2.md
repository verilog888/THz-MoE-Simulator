# 第二阶段：多 token Dispatch 与 Expert Compute

本阶段的边界是 `Base → Source → Gate → Dispatch → Expert Compute`。每个 token 的所有 routed experts 完成后记录 `experts_done_s`；Local Combine、Return 和最终 Base 完成属于第三阶段。
结果使用 `dispatch_compute_makespan_s` / `experts_done_latency_pXX_s`，不输出伪完整层延迟。

## 模型选择及依据

以下选择补足 PDF 未定义的细节，属于参考模型假设，不是硬件事实。用户已同意采用以下批处理规则并明确标注。

- **A02/A03（阶段 2 扩展）**：各 token 的 Base→Source 和 Gate 都使用配置固定延迟，视为独立、无限并发的前端服务；不占 THz 和专家计算资源。尚未标定 Base/Gate 吞吐，不能用当前结果推断完整系统吞吐。
- **A05**：每 Die 一个共享计算资源；同 Die 所有 experts 竞争该资源，不同 Die 可并行。每个已关闭的 batch 使用该 Die 全部配置算力/内存带宽，按 `max(6dhm/P, ceil(m/R)·3dh·bw/BW)` 执行。未配置 R 时每 batch 读取一次权重。
- 每个 expert 独立开窗 `[首请求 ready_s, ready_s+window_s)`，达到 `max_tokens` 提前关闭。截止时刻到达的请求属于下一个窗口，不依赖 heapq 同时刻事件顺序。
- `window_s=0` 立即形成 singleton batch，同刻请求也不自动合并。默认 0；非零窗口在示例中显式给出。
- 已关闭 batch 按 `(关闭时间, 关闭序号)` 在其 Die 上 FIFO 执行；开窗等待不占计算资源，也不阻塞其他已关闭 batch。关闭后不吸收后到请求；执行不抢占。提前关闭留下的过期定时事件无副作用，不延长统计窗口。
- **A06/A08**：保留用户选择的理想多子载波收发模型；当前 Dispatch 使用 FIFO、单流独占全部频谱的参考队列，setup 占用分配频谱。此服务尚未构成五组端到端基线，也未实现 OFDMA 并发。
- **A07**：Top-k 无放回；uniform / Zipf 合成路由使用等权 1/k。Zipf 的 p(i) 是首次抽样概率，后续在未选专家中重归一化，不能把多选 inclusion probability 等同于 k·p(i)。本地 expert 在 Gate 完成时立即获得输入，不等待远端广播。
- 同 token 多个远端 expert/Die 只发送一个 activation，目的 Die 去重；每个目的 Die 在广播完成后获得本地所有命中 assignment。全部命中 Source 时不生成无线包。
- shared expert 仍不计入；不存在跨 Die 权重搬运。通信和 compute 是独立资源，可发生 Dispatch/Compute overlap；完整 Return overlap 留在第三阶段。

## 路由 trace

配置 `workload.routing="trace"`、`source_policy="trace"`、`trace_file`；路径相对配置文件目录解析。`num_tokens` 必须与文件相符。每条 trace 必须显式携带唯一非负 token ID、arrival/source、Top-k 唯一 experts 和非负有限 weights。
真实 weights 原样保留，不强制和为 1；不自动补全、归一化或重新采样。

```json
{
  "schema_version": 1,
  "tokens": [
    {"token_id": 7, "arrival_s": 0, "source_die": 0,
     "expert_ids": [0, 2], "gate_weights": [0.3, 0.7]}
  ]
}
```

synthetic 模式支持固定 Source 或 round-robin，以及共同 arrival 和固定 interarrival。先完整生成 workload，再运行调度，因此队列变化不会改变随机路由。
每次导出 canonical `routing_trace.json`、语义 workload SHA-256；输入 trace 另记录文件字节 SHA-256。相同 trace 的字段/行顺序或空白变化不改变语义摘要。

## 第二阶段验收目标

1. 每个 `(token, expert)` 输入与计算恰好一次；Source 本地输入在 Gate 完成时可用，远端在广播结束后可用。
2. 一个 token 至多一条 Dispatch，remote destinations 完整且去重；all-local 的无线字节数为 0。
3. 同 Die compute 不重叠、不同 Die 可并行；batch membership、大小、FIFO 和 weight reuse 有独立手算验证。
4. token 的 `experts_done_s` 等于其最慢 expert 的完成时间；队列等待、batch 等待与执行时间分别记录。
5. 统计以最后一次真正的 expert 完成为终点；过期 timer 不改变 makespan；每个 batch 的忙碌区间仅计一次。
6. 第一阶段解析结果继续回归；合成/trace 运行、CSV 回读、时间线、逐字节复现全部实际执行。
