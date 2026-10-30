# 分阶段实施和接口

每个阶段按“实现 → 运行测试 → 诊断/修复 → 重跑 → 记录证据”推进。
当前已实现阶段 1–5；阶段 3/4/5 统计完整回到 Base 的层路径，阶段 2 入口仍截止于 Expert Compute 完成。实际测试与扫描执行证据见 [VALIDATION_STAGE5.md](VALIDATION_STAGE5.md)。
原定开发范围至第五阶段结束，没有尚未实施的必需第六阶段。硬件校准、真实 FFN/量化/shared expert 和完整大规模扫描是已列明的后续研究或验证工作，不计为本次已完成能力。
批处理策略和 trace 格式见 [STAGE2.md](STAGE2.md)；加权归并、Combine 资源和 Return 规则见 [STAGE3.md](STAGE3.md)。

| 阶段 | 范围 | 进入下一阶段前的验收 |
|---|---|---|
| 1 | heapq 引擎、配置、基础拓扑/放置、单 token 单远端 expert 的完整时序参考路径、纯计算/PHY 公式、CSV/时间线 | 手算 oracle、因果/单位/预算检查、错误输入、零延迟事件、固定种子与导出复现 |
| 2 | 多 token、多 expert；uniform/Zipf/trace；去重广播与本地命中；expert 队列和共享 Die 算力、batch/reuse | token/expert 不丢不重、广播一副本、本地无线 0 bytes、争用/排队、独立/共享计算资源 |
| 3 | 加权 Local Combine、远端 Return、Source Combine、fan-in 与 compute/communication overlap | 合成向量与集中加权和一致、同 Die 多 expert 只返回一次、关闭 combine 的流量差异、慢 expert 阻塞正确 |
| 4 | TDMA、静态 OFDMA、CA-OFDMA 和五组；已采用明确标注的 A/B 参考语义 | 频谱冲突、理想端点并发、F>Nsc、有限队列进展、分配舍入、epoch/速率变更、过期事件、共同预算、TDMA PEPD≤1；验证峰值并发与平均 PEPD 的区别 |
| 5 | 参数扫描、重复 seeds、CSV 汇总、延迟/吞吐/利用率/PEPD 图、通信计算时间线 | 共用 trace/config 摘要、重复性、字节守恒、积分指标与独立离线审计、图表从 CSV 生成 |

## 模块接口（阶段 1 参考路径保留）

| 模块 | 核心接口 | 阶段 1 状态 |
|---|---|---|
| config | `load_config(path) -> SimulationConfig` | 严格 JSON、单位和范围校验 |
| topology | `Topology.from_config(config)` | 单 Block，Die ID 校验；不虚构距离 |
| placement | `ExpertPlacement.contiguous(E, ND)`、`die_for(e)`、`group(experts)` | 均衡连续放置和目的去重 |
| moe | `Token`、`Flow`、`Interval` | 不保存真实大矩阵，记录语义和依赖 |
| routing | `uniform_route(E,k,rng)` | 无放回/等权；Zipf 和真实 trace 后续实现 |
| workload | `make_reference_token(config, placement)` | 固定 assignment 或有 seed 的 uniform |
| compute | `estimate_expert(model, compute, m)` | FLOP、权重读取和 roofline 纯函数 |
| thz_phy | `THzLink.transfer(bytes,nsc)` | 速率/时间纯函数，显式 byte→bit |
| ofdma | `Allocation`、`validate_allocations(...)` | 子载波预算/互斥基础；动态算法尚未实现 |
| scheduler | `IsolatedFullBandAllocator.allocate(flows)` | 只接受一条流；不是五组基线之一 |
| engine | `EventLoop.schedule(...)`、`run(handler)` | 确定性同刻 FIFO；拒绝过去/非有限时刻 |
| simulator | `run_reference(config) -> SimulationResult` | 仅 1 token、Top-1、远端专家，明确拒绝超范围配置 |
| metrics | `summarize(...)`、`SimulationResult.export(path)` | 端到端统计、逐事件/逐流/逐区间 CSV、manifest |
| plots | `plot_timeline(result,path)` | 当前参考路径的通信/计算时间线 |
| main | `python -m vdie_moe_sim ...` | 配置驱动的参考运行入口 |

阶段 2 增加/扩展：

| 模块 | 接口和职责 |
|---|---|
| routing / workload | `zipf_route(...)`、`build_workload(config, base_dir=...) -> Workload`；预生成 fixed/uniform/Zipf/trace 路由，支持 round-robin Source 和 arrival；导出可重放 trace |
| dispatch | `plan_dispatch(token, placement) -> DispatchPlan`；区分本地 expert 与去重远端 Die |
| batching | `ExpertBatcher.admit(request)` / `expire(id,time)`；half-open 时间窗口、提前关窗和不可变 ClosedBatch |
| pipeline | `run_dispatch_compute(config, trace_base_dir=...) -> PipelineResult`；多 token 事件、FIFO RF、每 Die 共享 compute、跨 Die 并行 |
| pipeline_metrics | `PipelineResult.audit()` / `summary` / `export(...)`；逐 assignment/batch 因果审计、部分流水线指标，忙碌区间不重复计数 |
| plots / main | `plot_pipeline_timeline(...)`；CLI `--stage stage2`，阶段 1 默认入口保持兼容 |

阶段 3 增加/扩展：

| 模块 | 接口和职责 |
|---|---|
| combine | `weighted_vector(...)`、`sum_vectors(...)`、`synthetic_expert_output(...)`；独立可测的数值原语和不可变部分结果/输出记录 |
| pipeline | `PipelineSimulation`；复用前端、batch 和 compute，实现 Dispatch/Return 共用的 FIFO RF，保留阶段 2 入口 |
| layer | `run_layer(config, trace_base_dir=...) -> LayerResult`；每 Die FIFO Combine、partial fan-in、Return、Source Combine 和最终 Base 完成 |
| layer_metrics | `LayerResult.audit()` / `summary` / `export(...)`；完整依赖/资源审计、独立集中数值 oracle、完整层统计及 NPY/CSV 导出 |
| config | `CombineConfig`；Local Combine 开关、固定 weight/reduce 延迟、资源模式和独立验证 seed |
| plots / main | `plot_layer_timeline(...)`；CLI `--stage stage3`，完整共享无线/计算/Combine 时间线 |

阶段 4 增加/扩展：

| 模块 | 接口和职责 |
|---|---|
| radio | `allocate_counts(...)`、`SharedRadio`；FIFO 准入、三种频谱分配、setup/小数 bits 进度、generation 事件失效 |
| radio_audit | `audit_radio(...)`；离线载波/bit/准入/分配特征审计与频谱时间积分 |
| execution_audit | `audit_batches(...)`；第二至四阶段共用的独立 batch/共享 Die FIFO 审计 |
| scheduled_layer | `run_scheduled(...)`、`effective_config(...)`；五组逐跳数据流，复用计算/Combine 引擎 |
| scheduled_metrics | `ScheduledResult.audit()` / `summary` / `export(...)`；完整路径、数值及事件检查，分段无线 CSV |
| experiments | `run_comparison(...)`、`audit_comparison(...)`；五方案共同配置、工作负载和向量等价检查 |
| plots / main | 子载波时间线、由 CSV 生成的比较图；`--stage stage4 --scheme ...` / `--compare` |

详细调度和 relay 假设见 [STAGE4.md](STAGE4.md)。每次重分配推进剩余 bits 并更新事件版本，不重发已传数据。

阶段 5 增加/扩展：

| 模块 | 接口和职责 |
|---|---|
| sweep_plan | `load_plan(...)` / `compile_plan(...)`；单因素/grid、坐标去重、规模与配置预检 |
| statistics | `describe(...)` / `paired_observations(...)`；跨种子均值/std、配对 speedup、探索性 bootstrap |
| offline | `audit_run(directory)`；从 CSV/NPY 重算无线积分、bits、计算、分位数和输出数值 |
| sweep | `run_sweep(...)` / `audit_sweep(...)`；逐次检查点、严格续跑、共同预算与配对统计、离线汇总复核 |
| sweep_plots | `plot_sweep(...)`；从 CSV 生成确定性 PNG/SVG 扫描图 |
| config / scheduled_layer | `AblationConfig`；broadcast/Local Combine 独立开关，结果名称与正式基线区分 |
| main | `--stage stage5 --plan ...`、`--dry-run`、`--resume`、`--audit` |

本阶段按纯函数 → 独立消融 → CSV 重载 → 完整扫描/续跑 → 多种子重复导出的顺序实施。详细统计规则和计划规模见 [STAGE5.md](STAGE5.md)。

## 测试策略

1. 单位/公式：BF16/FP8 字节数，6dhm，3dh·bw，ceil(m/R)，compute/memory-bound 两侧及 m=0；Sbits/R 的手算。
2. 引擎：过去/NaN/Inf 时间拒绝、同刻 FIFO（包括 callback 新增事件）、有界事件执行防止零时间死循环。
3. 参考路径：独立手算 22 μs 测例、偏移 arrival 后 latency 不变、完整事件类型序列、return 发生在 compute 后。
4. 统计：逐区间独立积分、THz/compute 利用率、PEPD、BWwaste、p50/p95/p99，不能拿接口自己计算的期望值测试自己。
5. 资源：载波 ID 唯一/范围/总预算；时间重叠时不得占同子载波；不允许静默多流退化为单流。
6. 输入和复现：非法 JSON 字段/精度/范围拒绝，固定 seed 路由一致，配置/工作负载 hash 稳定，CSV round-trip 和两次运行一致。
7. 已添加广播去重、局部/最终合并数值 oracle、错误二次加权注入、回传反馈、慢 Die fan-in、共享 Combine 资源、五方案共同预算及 OFDMA 并发审计。

8. 第五阶段检查参数点/种子覆盖、共同预算、同 seed 配对、均值与样本 std 的手算、单种子缺失区间、重复 seed 拒绝、artifact 篡改、断点续跑和图表重复性；统计所依据的明细必须可离线重算。

程序退出 0 不等于仿真正确；真实硬件验证另列。各阶段证据保留在 `VALIDATION*.md`，当前完整记录为 [VALIDATION_STAGE5.md](VALIDATION_STAGE5.md)。
