# V-Die THz MoE 离散事件仿真器

依据用户提供的 16 页单层 MoE 仿真方案，分阶段实现与验证。**当前已实现阶段 1–5：完整 routed-expert 层路径、TDMA/静态 OFDMA/CA-OFDMA、五组同带宽对照，以及参数扫描、多种子统计、离线审计和图表。** 已支持广播/本地命中、batch/weight reuse、uniform/Zipf/trace、加权 Local/Source Combine、Return、完整 CSV/NPY 和实际子载波时间线。
历史阶段入口保留作回归基准。实现和公式经过自动检查；模型仍是未标定的架构参考，不代表真实硬件或完整 DeepSeek-V3 推理。

完整的[规格审查和假设](docs/SPECIFICATION.md)、[开发计划和模块接口](docs/DEVELOPMENT_PLAN.md)、[阶段 5 扫描/统计规则](docs/STAGE5.md)、[阶段 5 测试证据](docs/VALIDATION_STAGE5.md)在 `docs/` 中。批处理、Combine、调度约定见 [STAGE2.md](docs/STAGE2.md)、[STAGE3.md](docs/STAGE3.md)、[STAGE4.md](docs/STAGE4.md)。所有运行保存参数、假设、输入和源码摘要。

用户选择：使用未标定参考参数，暂不计 shared expert；同 Die 可在不同子载波上多流收发。采用固定窗口/cap batch、每 Die 一个共享 expert 计算资源，以及独立 FIFO Combine 单元。Dispatch 和 Return 共享全局频谱，回传会影响后续输入/计算时序。A 的 Base relay 两跳分别计入同一无线预算；每跳仅一次 setup，重分配额外开销设为 0。这些均是明确标注的参考假设。

## 安装

实测环境：Linux x86_64、CPython 3.12.14。使用 NumPy、Pandas、Matplotlib 和标准库 `heapq` / `unittest`，无需 GPU、神经网络模型或外部服务。

```bash
git clone https://github.com/verilog888/THz-MoE-Simulator.git
cd THz-MoE-Simulator
bash scripts/setup_env.sh
source .venv/bin/activate
```

`requirements.lock` 固定了实测直接/传递依赖；`pyproject.toml` 提供包和 CLI 入口。首次安装需要访问 PyPI；已安装后的运行和测试可离线执行。其他 Python 版本/平台的数值和图片逐字节一致性尚未验证。

`.python-version` 记录实测 Python 版本；安装脚本使用当前 `python3`，也可通过 `BOOTSTRAP_PYTHON` 指定已有解释器。后续命令均在仓库根目录执行。

## 文件组织

| 目录/文件 | 用途 |
|---|---|
| `vdie_moe_sim/` | 模型、事件引擎、调度、审计、统计和绘图模块 |
| `tests/` | 179 项公式、因果、资源、数值、统计和复现测试 |
| `configs/` | 各阶段手算/参考配置、trace 示例和扫描计划 |
| `scripts/` | 环境安装及五阶段复现入口 |
| `docs/` | 规格、假设、接口、阶段规则与验证记录 |
| `docs/evidence/stage5/` | 已执行实验的测试日志、审计摘要、汇总 CSV 和示例图 |
| `.github/workflows/ci.yml` | 安装锁定依赖，运行完整测试、小模型双遍扫描和 wheel 构建 |

`results/`、`.venv/`、缓存和构建产物均忽略；完整逐事件明细通过复现脚本生成。仓库内的[验证证据摘录](docs/evidence/stage5/README.md)说明了保留范围和校验方式。

## 运行与测试

完整重现第五阶段（全部测试、三套扫描各运行两遍、CSV/NPY 离线审计、CSV/JSON/NPY/PNG/SVG 逐字节比较）：

```bash
bash scripts/reproduce_stage5.sh
```

脚本创建新的 `results/stage5-XXXXXX/`，保留 `tests.log`、离线审计报告和 `reproducibility.json`。每遍包含小模型验证 125 次、独立开关消融 60 次、论文维度代表性扫描 450 次，共 635 次仿真；两遍共 1270 次。完整 PDF 取值计划另做 dry-run，未默认执行其中的大 token 点。快速复现可选 `bash scripts/reproduce_stage5.sh smoke`。

本次已实际通过 **179 项测试**，完成上述 **1270 次仿真**和六次离线审计，**11,486 对导出文件逐字节一致**。具体结果、修复记录和执行范围见[验证报告](docs/VALIDATION_STAGE5.md)。

每个参数点的五组方案共用 workload、完整 PHY 预算、计算和时延参数，并核对最终向量。启用 `pipefail`，测试/审计失败立即停止。历史阶段仍可用 `scripts/reproduce_stage1.sh` 至 `scripts/reproduce_stage4.sh` 重现。

预览、运行、续跑和独立审计：

```bash
export MPLCONFIGDIR="$PWD/.mplconfig"
export XDG_CACHE_HOME="$PWD/.cache"
.venv/bin/python -m vdie_moe_sim --stage stage5 \
  --plan configs/sweeps/stage5_paper_full.json --dry-run
.venv/bin/python -m vdie_moe_sim --stage stage5 \
  --plan configs/sweeps/stage5_paper_sample.json --out results/my-sweep
.venv/bin/python -m vdie_moe_sim --stage stage5 \
  --plan configs/sweeps/stage5_paper_sample.json --out results/my-sweep --resume
.venv/bin/python -m vdie_moe_sim --stage stage5 --audit --out results/my-sweep
```

`--dry-run` 不执行仿真、不写结果；`--resume` 只接受相同计划、输入、源码和依赖，并校验已有文件，完成的仿真不重跑。普通非空输出目录不会覆盖。第五阶段的时间线选项放在计划的 `timelines` 字段中。

单次五组对照：

```bash
export MPLCONFIGDIR="$PWD/.mplconfig"
export XDG_CACHE_HOME="$PWD/.cache"
.venv/bin/python -m vdie_moe_sim --stage stage4 --compare \
  --config configs/stage4_zipf.json \
  --out results/my-stage4-comparison --timeline
```

把 `--compare` 替换为 `--scheme D` 可单独运行静态 OFDMA；支持 A/B/C/D/Proposed。省略二者时使用 JSON 中的 `scheduling.scheme`。也可使用 `.venv/bin/vdie-moe-sim`。`--stage stage1`（兼容默认值）为单 token 参考路径；`stage2` 截止于 Expert Compute；`stage3` 为完整层共享 FIFO；`stage4` 支持五组方案。输出目录须不存在或为空，配置错误返回非零状态。

仅运行测试：

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" \
  .venv/bin/python -m unittest discover -s tests -v
```

云环境的 home 可能只读，上述缓存设置用于 Matplotlib/fontconfig。每个云任务已经隔离，使用当前 checkout；除非用户明确要求，不另建 Git worktree。

## 配置与输出

第五阶段计划位于 `configs/sweeps/`，基础配置路径相对于计划解析：

| 计划 | 内容 | 单遍仿真数 |
|---|---|---:|
| `stage5_smoke.json` | 小模型 token/Nsc 扫描，5 个唯一点×5 种子×5 方案 | 125 |
| `stage5_ablation.json` | 论文 d/h/E 下 broadcast/Local Combine 四组合×5 种子×3 PHY | 60 |
| `stage5_paper_sample.json` | 八类参数代表值，18 个唯一点×5 种子×5 方案；基准 8 tokens，token 轴含 1/8/32 | 450 |
| `stage5_paper_full.json` | PDF 第 14 页全部数值的单因素计划，29 点×5 种子×5 方案；含 512/2048 token，先预览 | 725 |

支持 `one_factor` 和显式 `grid`；共享基准点去重。独立 broadcast/Local Combine 消融以 `Ablation-...` 标记，不改写正式五组定义。每点默认 5 个路由种子，报告均值、样本 std、同种子配对加速比及探索性 bootstrap 95% 区间；统计规则并非 PDF 给定常数。fixed/trace 只允许一个 seed，单种子不生成虚假的标准差或区间。

| 第五阶段输出 | 内容 |
|---|---|
| `raw_results.csv` | 每点/种子/方案的原始指标、摘要和明细路径 |
| `paired_results.csv` / `aggregate.csv` | 同种子比值/差值，以及种子级均值、std、n、独特 workload 数和探索性区间 |
| `points.csv` / `views.csv` / `fairness.csv` | 参数点与图坐标、共享基准映射和共同预算/数值检查 |
| `sweep_manifest.json` | 计划、基础参数、状态、预期/完成数、源码/输入/依赖和图表摘要 |
| `runs/<point>/seed_<seed>/<series>/` | 每次仿真的完整明细、NPY、JSON trace、检查点及代表性时间线 |
| `plots/*.png` / `plots/*.svg` | 仅从 CSV 生成的延迟、吞吐、p95/p99、利用率、PEPD、BWwaste、配对加速比图 |

以下是历史阶段与手算配置，仍可单独运行：

- `configs/stage4_tiny.json`：单 token 手算五组，A/B/C/D/Proposed 完整延迟分别 **43/23/18/18/18 μs**。
- `configs/stage4_concurrent.json`：双 token 实际并发；C 为 **20 μs**，D 为 **20.5 μs**，不预设 OFDMA 必然更快。
- `configs/stage4_limited.json`：12 token、4 Dies、仅 2 个子载波，检查 F>Nsc 准入、排队和完整交付。
- `configs/stage4_zipf.json`：128 token、Top-8、256 experts、16 Dies、32 GHz、64 子载波的五组对照；硬件参数未标定。
- `configs/stage3_tiny.json`：一个 token 命中 3 个 expert、2 个 Die；开启 Local Combine，返回 8 bytes，完整层延迟手算 **18 μs**。
- `configs/stage3_no_local_combine.json`：同一工作负载和 PHY 预算，关闭 Local Combine，返回 16 bytes，手算 **20 μs**；最终数值结果一致。
- `configs/stage3_return_feedback.json`：两个 token 相隔 8 μs；第一个的 Return 阻塞第二个 Dispatch，完整层 makespan 手算 **24 μs**。
- `configs/stage3_zipf.json`：128 token、256 experts、Top-8、16 Dies、d=7168、Zipf 1.5；含非零 batch/weight/reduce 参考延迟，用于检查完整 fan-in 和负载不均衡。
- `configs/stage2_tiny.json`：两个 token、每个一个本地和一个远端 expert；阶段完成时间手算 12 μs。
- `configs/stage2_batching.json`：五个全本地 token，batch 窗口 2 μs、cap=3、reuse tile=2；手算 7 μs、无线 0 bytes。
- `configs/stage2_zipf.json`：128 token、256 experts、Top-8、16 Dies、Zipf 1.5；适合检查不均衡负载。硬件参数是未标定示例。
- `configs/stage2_trace.json`：示范 JSON trace；路径相对配置文件目录解析，保留 token ID、到达时刻、Source、expert IDs 和原始 gate weights。每次运行导出的 `routing_trace.json` 可作为后续对照的同一工作负载输入。
- `configs/stage1_tiny.json`：可手算 fixture，d=4、h=2、2 experts、2 Dies；预期 22 μs。
- `configs/stage1_reference.json`：采用 PDF 的 d=7168、h=2048、E=256；选择扫描允许的 Top-1、8 Dies、32 GHz。算力 10 TFLOP/s、内存带宽 1 TB/s、ηPHY=0.8、setup=100 ns 都是**未标定示例**；Base/Gate/Source Combine 延迟暂设 0。
- PDF 默认 Top-8 保留在 `ModelConfig` 和规格表中；阶段 2 已支持多 token / Top-k / 本地命中。阶段 1 入口仍严格限制为 Top-1 单远端 expert。
- 单位：秒、Hz、FLOP/s、byte/s；PHY 速率为 bit/s。精度字段使用 `bf16` 或 `fp8`。

Combine 配置字段（完整 JSON 示例见 `configs/stage3_*.json` 和 `configs/stage4_*.json`）：

```json
"combine": {
  "local_combine": true,
  "weight_s": 0.0,
  "local_reduce_s": 0.0,
  "resource_model": "per_die_fifo",
  "verification_seed": 7
}
```

`weight_s` 为每 expert 权重乘法延迟，`local_reduce_s` 为每个多 expert 本地组的归并延迟；Source Combine 沿用 `timing.source_combine_s`。三类操作共用本 Die FIFO Combine 单元，与 expert compute 独立。0 表示未标定参考值。`independent` 仅用于理想无限并发的敏感性分析。

第四阶段新增 `scheduling`：`scheme`、`epoch_s`、`alpha/beta/gamma/delta`、`q_ref_bytes`、`w_ref_s`、`max_events`。CA 的显式归一化、静态空闲余数、FIFO 准入和超预算修正规则见 [STAGE4.md](docs/STAGE4.md)。A/B 自动关闭 Local Combine，其他三组开启；有效配置完整保存在 manifest 中。

第四阶段单次运行保留下表中的计算、归并、数值和 provenance 导出，并扩展无线记录：

| 文件 | 第四阶段内容 |
|---|---|
| `flows.csv` | 每个逻辑单跳的 ready/admitted/completed、专家/partial 关联和 relay parent ID |
| `radio_segments.csv` | 实际 setup/payload 时间片、子载波、净速率、分段发送 bits；同一 flow 可多行 |
| `allocations.csv` | 每次频谱分配的 Q/W/I/C、priority、载波及超预算修复标记 |
| `summary.csv` | 完整层延迟/吞吐、利用率、PEPD、无线逐跳字节数、relay 额外 hop、独立数值误差 |
| `timeline.png` | 子载波×时间图，以及每 Die compute/Combine 资源图 |

`--compare` 另生成 `comparison.csv`、`fairness.json` 和从 CSV 绘制的 `comparison.png`；A/B/C/D/Proposed 各有独立明细子目录。字节按 hop 统计一次，不能将完整 payload 按 segment 重复累加。

第三阶段每次导出：

| 文件 | 内容 |
|---|---|
| `summary.csv` | `layer_makespan_s`、完整 token 延迟/吞吐、利用率、PEPD、BWwaste、返回节省字节数、数值误差和配置摘要 |
| `tokens.csv` | source、arrival、assignment、原始 gate weights、`experts_done_s`、`source_combined_s`、`completed_s`、`latency_s` |
| `flows.csv` | Dispatch/Return ready/排队/起止、setup/payload 分界、实际载波 ID、字节数和速率 |
| `requests.csv` | 每个 `(token, expert)` 的输入 ready、所属 batch、关窗等待、队列等待、compute 起止 |
| `batches.csv` | batch 成员、关窗原因、FLOPs、权重读取次数、计算起止 |
| `combine_jobs.csv` | weight/Local/Source Reduce 的 Die、依赖 ready、队列等待和执行区间 |
| `partials.csv` | 每个部分结果的 expert 集合、ready、到达 Source 时间、Return ID、向量 SHA-256 |
| `events.csv` | 事件时间、确定性处理顺序、关联 ID 和过期 batch timer 标记 |
| `intervals.csv` | 通信、计算和配置延迟区间；每个 batch 只有一个 busy interval，避免按成员数重复计数 |
| `compute_utilization.csv` | 每 Die 忙碌时间和利用率，含空闲 Die |
| `combine_utilization.csv` | 每 Die Combine 占用时间、利用率和平均活跃任务数 |
| `output_validation.csv`、`outputs.npy` | 独立集中加权和的校验误差、token→数组行映射，以及形状 `(tokens, d)` 的 float64 输出 |
| `manifest.json` | 全配置、seed、配置/工作负载/源码/PDF SHA-256、依赖版本、适用假设和未实现项 |
| `routing_trace.json` | 可重放的完整路由输入 |
| `timeline.png` | `--timeline` 时生成，区分 Dispatch/Return/setup、各 Die compute/Combine 和最终 Base 完成 |

阶段 1、3、4、5 的完整层窗口从最早 arrival 到最后一次回到 Base。**阶段 2 窗口截止于最后一个 expert 完成，不能当作完整层延迟**；过期 timer 不延长窗口。每 token 的 latency 减去自己的 arrival。setup 保留已分配频谱，故 THz 利用率和 PEPD 计入 setup；另报 payload-only PEPD。BWwaste 使用已乘 ηPHY 的净容量，不重复统计 FEC/header 开销。无无线传输时 BWwaste 约定为 0。

## 当前正确性范围与后续工作

已验证时序、公式、字节守恒、统一频谱预算、多 token 广播/本地输入、batch 窗口/权重复用、每 Die 排队和跨 Die 并行，以及逐 expert 加权、Local/Source Combine、Return 对后续计算的反馈、完整层指标和导出/trace 重放。独立集中加权和校验能检测故意重复乘 gate weight 的错误。

这些检查使用确定性的合成 float64 expert 输出；BF16/FP8 只决定通信字节数，未模拟量化/舍入或真实 FFN。硬件参数、前端和 Combine 资源尚未标定；shared expert 未计入。

第四阶段还验证了超载准入、CA 舍入和 epoch、setup/小数 bits 跨重分配保留、全局载波冲突、relay 两跳依赖和五方案共同预算。理想多流端点、零频谱切换开销和固定净速率均未硬件标定；并发峰值 >1 不保证平均 PEPD>1，也不保证更低延迟。

第五阶段验证了扫描去重、种子配对、保存/续跑、CSV/NPY 重载和汇总重算。置信区间只描述有限路由种子的变化，不包含硬件模型不确定性，也不是多重比较显著性证明。完整 PDF 大规模计划的实际执行范围见验证报告；不把 dry-run 当作实验完成。
