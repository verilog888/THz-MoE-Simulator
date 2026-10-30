# 第五阶段：参数扫描、种子统计和离线复核

本阶段复用已验证的完整离散事件引擎，不改变五组基线。新增扫描计划预检、逐参数点/种子运行、独立开关消融、配对统计、CSV 派生图表和断点续跑。参数值依据 PDF 第 14 页；随机重复和区间估计方法是显式参考选择，PDF 未指定这些统计规则。

## 计划和接口

| 模块 | 接口与职责 |
|---|---|
| `sweep_plan` | `load_plan(path)`、`compile_plan(...)`；严格字段校验、one-factor/grid 展开、等价参数点去重、运行数量预览 |
| `statistics` | `describe(...)`、`paired_observations(...)`；种子级均值/样本标准差、配对加速比、确定性 percentile bootstrap |
| `sweep` | `run_sweep(...)`、`audit_sweep(...)`；逐次保存、共同预算/路由审计、检查点与汇总重算 |
| `offline` | `audit_run(path)`；直接读取 CSV/NPY 复核频谱、bits、计算、时延与数值，不运行事件引擎 |
| `sweep_plots` | `plot_sweep(...)`；仅从 `aggregate.csv` / `views.csv` 生成 PNG/SVG |
| `config / scheduled_layer` | 显式 `AblationConfig`，允许 broadcast/Local Combine 独立变化，使用不同于正式基线的名称 |

先验证纯计划和统计函数，再验证独立消融、CSV 重载、完整扫描、续跑和重复导出。历史入口及测试全部保留。

## JSON 示例与展开规则

路径相对于扫描计划文件解析；`base_config` 内的 trace 路径相对于基础配置目录解析。

```json
{
  "schema_version": 1,
  "name": "Uncalibrated bandwidth sensitivity",
  "base_config": "../stage5_reference.json",
  "mode": "one_factor",
  "study": "five_scheme",
  "axes": {"phy.total_bandwidth_hz": [16000000000, 32000000000, 64000000000]},
  "seeds": [11, 23, 37, 51, 71],
  "statistics": {"bootstrap_resamples": 2000, "confidence": 0.95, "seed": 1729},
  "timelines": "representative",
  "max_runs": 10000
}
```

- `one_factor`：每次只改变一个坐标，其他参数保持 base 值。扫描轴按名称规范排序，轴内值按给定顺序保留。多个扫描轴上的同一个基准配置只运行一次，在多个图表视图中复用，不增加统计样本数。
- `grid`：只对计划明确列出的轴求笛卡尔积。图表逐项列出组合坐标，不将多个同时改变的参数冒充单因素曲线。
- `32000000000` 与 `32000000000.0` 视作同一参数点；字段精度不造成伪重复。配置、种子、原始 JSON 及哈希仍完整保存。
- `study=five_scheme` 始终运行 A/B/C/D/Proposed。`study=ablation` 运行 TDMA/static/CA 三种 PHY，C/D/Proposed 此时只是内部 PHY 选择键，结果名称明确为 `Ablation-...`。
- `max_runs` 在执行前限制展开规模，可显式调整。无效配置、不可整除的放置、无效 Top-k、重复种子/坐标、未知字段均拒绝；不会跳过错误点后悄悄计算残缺均值。
- `timelines=representative` 只给规范顺序下第一个参数点、第一个种子的各方案画时间线；所有点均保存完整事件/segment 明细。`all` 和 `none` 分别全部绘制或不绘制。

支持 PDF 的 token 数、Die 数、Top-k、Nsc、总 B、频谱效率、Zipf skew、activation 精度，以及独立 broadcast/Local Combine 开关；还可显式扫描 batch、算力/内存带宽、weight/reduce 延迟和 CA 参数。`workload.seed` 不作为普通坐标，统一由 `seeds` 控制；scheduler 作为方案序列比较。

## 统计假设与公平性

默认示例使用路由种子 `[11,23,37,51,71]`，共 5 次。合成向量的 `combine.verification_seed` 固定，不把数值测试噪声混入路由重复。相同参数点和种子下，各方案的实际 trace、完整 PHY 预算、计算/Combine 时延与 batch 参数必须相同；变化项只有明确的方案和该方案规定的 Local Combine 开关。

- 在带宽扫描中，不同横坐标的总 B 有意变化；相同横坐标的五组必须使用同一个 B。Nsc/precision 等扫描遵循相同原则。
- 只改变 PHY/compute 参数时，路由 trace 必须保持一致。Top-k/Zipf/token 数改变时工作负载可变化；Die 数改变且 Source 为 round-robin 时，Source 位置按模型变化。
- 均值/样本标准差的独立重复单位是一个路由种子，不是 token 或 expert 请求。p95/p99 列先在每次运行内算分位数，再跨种子汇总；不是把不同运行的 token 混在一起求分位数。
- 配对加速比先算同参数点、同种子的 `latency_C / latency_scheme`，再汇总这些比值。它不等于两个独立均值的比值。消融中的参考为相同 broadcast/combine 设置下的 TDMA。
- 默认 2000 次 bootstrap，对种子级观测有放回抽样，计算均值的 percentile 95% 区间；独立固定统计 seed，且每组/指标有确定性 RNG 标识。图带明确标注 **exploratory**。五个样本只能描述有限的路由变化，不足以证明总体优势、硬件不确定性或多重比较显著性。
- 单种子不伪造标准差或置信区间：对应 CSV 字段为空。`bootstrap_resamples=0` 可关闭区间，仅保留描述性统计。
- fixed/trace 工作负载不使用随机路由 seed，因此拒绝用多个 seed 重复它们；使用一个 seed。uniform/Zipf 中不同 seed 偶然产生相同 trace 是合法重复抽样，单独报告 `n_unique_workloads`，不将这些结果删除或夸大差异。

## 独立开关消融

`study=ablation` 扫描 `ablation.broadcast` 和 `ablation.local_combine`，可用 grid 覆盖四种组合。broadcast off 按 expert 单播；Local Combine off 按 expert 返回加权结果；开关相互独立。本地命中仍旁路无线，最终数值应保持一致。

没有修改正式 A/B/C/D/Proposed 的定义。带 ablation 配置的结果不能进入第四阶段正式五组 `--compare`。所有消融结果以 `Ablation-tdma/static/ca-b0/1-l0/1` 标识，并携带配置和假设。

## 保存、续跑与审计

每个 `(point, seed, series)` 在独立目录保留完整第四阶段明细。成功导出后写入 `completed.json`，包含配置/源码摘要和每个产物的 SHA-256，再以目录 rename 完成保存。中断留下的 `.pending-*` 目录不当成有效结果，也不自动删除。

`--resume` 只接受完全相同的扫描计划、基础配置、输入 trace、源码、版本和依赖指纹；每个已有结果仍要经过文件校验与 CSV/NPY 审计。完成的仿真不重跑。被修改的结果会报错，不能与新模型结果混合。已存在的普通非空目录不会被覆盖。

每个运行在线执行完整数据依赖/数值审计，保存后再从 CSV/NPY 复核：逐 hop bits 守恒、setup、载波冲突、准入/CA 分配、batch/compute、统计积分、输出数组。离线工具不重放事件引擎。`audit_sweep` 还重新汇总所有点/种子/方案，逐字节比较 raw、配对和 aggregate 表；hash 检查不能代替这些数值检查。

## 文件结构

| 文件 | 内容 |
|---|---|
| `sweep_manifest.json` | 完整计划、基础配置、源码/输入/依赖指纹、预期/已完成数、状态、汇总/图表 hash |
| `points.csv` / `views.csv` | 唯一参数点与绘图坐标的映射；共享基准只运行一次 |
| `raw_results.csv` | 每个点、种子、方案的原始完整层指标、平均 Die 利用率、trace/公平性摘要和明细路径 |
| `paired_results.csv` | 同种子配对加速比及延迟差，参考 C/匹配 TDMA |
| `aggregate.csv` | 每点/方案/指标的 n、独特 workload 数、均值、样本 std、min/max、探索性区间和方法 |
| `fairness.csv` | 每点/种子的共同预算、计算、trace 和跨方案输出等价检查 |
| `runs/<point>/seed_<seed>/<series>/` | 完整单次 CSV/JSON/NPY、检查点和可选时间线 |
| `plots/*.png` / `plots/*.svg` | 每个扫描轴的延迟、吞吐、p95/p99、无线/计算利用率、PEPD、BWwaste、配对加速比 |

所有图只依赖已导出的 aggregate/views CSV。SVG 固定 hash salt 并去除时间元数据，使当前锁定环境内的图表也可逐字节复现。

## 提供的计划与执行范围

| 计划 | 参数点 | 种子数 | 方案数 | 单遍仿真数 | 用途 |
|---|---:|---:|---:|---:|---|
| `stage5_smoke.json` | 5 | 5 | 5 | 125 | 小模型验证 token/Nsc 扫描与重复性 |
| `stage5_ablation.json` | 4 | 5 | 3 | 60 | 论文维度 d/h/E 下的独立 broadcast/Local Combine 消融 |
| `stage5_paper_sample.json` | 18 | 5 | 5 | 450 | 论文 d=7168、h=2048、E=256，覆盖八类扫描参数的代表性取值 |
| `stage5_paper_full.json` | 29 | 5 | 5 | 725 | PDF 第 14 页所有列出的数值，单因素计划；本阶段仅预览验证 |

代表性计划基准为 8 tokens，token 轴为 1/8/32；完整计划基准为 128 tokens，包含 512/2048 等大点。正式已执行范围和测试结果另见 [VALIDATION_STAGE5.md](VALIDATION_STAGE5.md)。完整计划并非全参数笛卡尔积，也不能把预览计数视作完成了仿真。

全部硬件、理想收发/切换、前端和算力假设继承第四阶段。未校准物理链路、未实现真实 FFN/数值量化/shared expert。大 token 点会保留完整 d 维合成输出和明细，可能消耗较多内存/磁盘；极小 epoch 增加事件数，仍受 `scheduling.max_events` 限制。
