# THz-MoE Simulator

按照 [VDIE_MOE_SIM_SPEC.md](VDIE_MOE_SIM_SPEC.md) 执行三模型、单 MoE 层的离散事件时序与能耗仿真。仓库仅保留本次完整仿真，参数版本为 `VDIE-FULL-v1.0-review-1`，已获用户确认。

主要交付：

- [完整 Word 报告](results/simulation_report.docx)：目标、简述、全部参数含义、流程、结果、对照、结论。
- [Markdown 报告](results/report.md)、[完成状态](results/manifest.json)、[验收结果](results/validation.json)。
- [全部运行表](results/raw_runs.csv)、[五种子汇总](results/aggregate.csv)、[配对效果与 95% 区间](results/paired_effects.csv)。
- [参数审批文件](docs/approval/parameter_review.docx)、[参数表](results/parameters.csv)、[实验清单](docs/approval/experiment_plan.csv)。

**验收状态必须以 manifest 和 validation 为准。** 原 FP8 小算例 NRMSE 门槛仍为 5%；当前实现的端到端误差约为 6.4%–7.6%，这项验收失败。独立 E4M3FN 编解码对照与 FP32 Combine 等价性通过。完整计划运行和全部验收通过分别标记为 `experiments_complete` 与 `complete`，不得混淆。

## 安装与复现

需要 Linux、Python 3.12 和支持 C++20 的 `g++`，不需要 GPU。参考环境为 Python 3.12.14。首次获取公开数据集与官方 tokenizer 需要访问 Hugging Face；冻结 revision 和校验和见 [source_lock.json](docs/approval/source_lock.json)。不下载模型权重。Word 转 PDF 预览另需 LibreOffice，生成 DOCX 本身不需要。

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m vdie_moe_sim.run_all --spec VDIE_MOE_SIM_SPEC.md --output results --resume
```

新目录第一次运行去掉 `--resume`；该参数在重用已有结果时强制检查配置、事件内核、trace 和结果文件哈希。最多使用 4 个进程，每条记录保存后可续跑。也可使用 `bash scripts/reproduce.sh --resume`。

```bash
# 结构、资源、通信、容量与边界测试
.venv/bin/python -m pytest -q -m 'not numeric_gate'
# 原数值验收；真实失败会返回非零退出码
.venv/bin/python -m pytest -q -m numeric_gate
# 要求数值验收通过后才允许开始时序实验
.venv/bin/python -m vdie_moe_sim.run_all --output results --resume --require-numeric-gate
```

默认入口保存数值验收失败并继续已经批准的时序与能耗实验；不会改阈值，也不会把失败标成通过。输出的退出状态和验收状态用途不同，自动化验收应读取 `manifest.json` 的 `complete`。

## 实验范围

主实验 840 条：M1 GPT-oss-20B、M2 Qwen3-30B-A3B、M3 ERNIE-4.5-21B-A3B；8 种 T/O × P/B × N/L 组合；Prefill B=1/8/32、Decode B=1/8/32/128；5 个种子。每个种子无放回抽样 128 条原生请求，不做 padding。

另有强 TDMA 对照 60 条、原规范单因素敏感性 2,880 条、批准补充的向量能耗敏感性 240 条，共 4,020 条计划记录。其中 2,280 次运行时序模型，1,680 条从同一事件账本重算能量，60 条 Prefill 复用概率记录为明确的无效应等价控制。全部非默认敏感性点都保持其它参数不变。

T 是逐帧单 flow TDMA，O 是动态 OFDMA；P/B 分别为按目的 Die 去重的单播/广播 Dispatch；N/L 分别为关闭/开启 Local Combine。TW 是计入真实控制成本的包粒度强 TDMA。O 使用相同满带宽速率锚点并支付 14/17 效率损失。

真实数据仅为文本及分词长度；隐藏状态和 router 来自受控模型，标记 `dataset_tokens_synthetic_router`。Decode 是教师强制流量重放；单层时间不等于整模型 TBT。能耗分为 `frontend_proxy` 与 `baseband_stress` 两套账本；这些是架构估计，不是目标芯片的实测 PPA。

## 文件与复核

```text
configs/                 已批准配置与执行约定
docs/approval/           参数含义、来源、审批版本与完整计划
vdie_moe_sim/            Python 配置/数据/统计/报告与 C++ 事件内核
tests/                   独立核算及边界、数值、统计测试
scripts/                 安装、构建、完整复现入口
results/                 本次完整仿真及可审计原始记录
```

`results/token_metrics.parquet/` 是分片 Parquet 数据集；`raw_runs.csv` 的 `temporal_run_id` 连接时序原件，能量重算不复制 token 时间表。`runs/<plan_id>/` 保留事件计数、cohort/request/Decode 轮次指标和校验和。每张图提供 PNG、SVG 及源 CSV。

时序核算使用真实矩阵尺寸、整数皮秒、5ns 数字时钟对齐、双 tile 缓冲、共享 DMA、专家聚批、接收信用和半双工冲突。每个配置独立核对通信字节与矩阵 OP，依赖路径分解闭合到 cohort 时间；可重叠的资源 busy-time 另列。

本次整理前的历史代码和旧结果已校验备份到仓库外 `/workspace/THz-MoE-Simulator-history/`，不再混入当前结果。
