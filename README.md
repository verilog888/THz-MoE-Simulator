# ToV-MoE 单层与通信回放仿真

本仓库按 [ToV_MoE_MINIMAL_SIM_SPEC.md](ToV_MoE_MINIMAL_SIM_SPEC.md) 实际完成本次完整实验。当前版本仅保留本次代码、冻结输入和结果；所有仿真产物集中在 **[Results/](Results/)**，没有沿用旧阶段结果。

默认配置为 **16 个 V-Die、1 GHz、FP8 E4M3FN / FP32 累加**。每节点峰值为 8.192 TOPS，HB 峰值为 819.2 GB/s；kernel / DMA 分区有效带宽为 501.76 / 102.4 GB/s。模型结构和文本长度来自官方 config/AutoTokenizer，专家路由为规范规定的受控合成 Top-k。程序不下载或加载模型权重。

## 本次结果

| 对照 | 有效 D2D 吞吐倍数 SG | 单层延迟加速比 ST | ToV 延迟相对对照 |
|---|---:|---:|---:|
| TSV_NORM_MC_LC，264 Gb/s 聚合 raw budget | 3.1008× | 1.1368× | 降低 12.03% |
| TSV_HIGH_MC_LC，896 GB/s 聚合强有线情景 | 0.1343× | 0.9248× | 增加 8.13% |

NORM 的整体 95% 配对区间：SG `[2.9918, 3.2029]`，ST `[1.1306, 1.1419]`。Prefill / Decode 的延迟分别降低 14.96% / 9.01%。OFDMA 相对强 TDMA 的负收益也保留在机制对照中。

这些是**两套独立测量**：吞吐来自零计算 D2D 回放，延迟来自完整单层依赖仿真。NORM 是归一化设计预算，不是 TSV 技术上限或等面积实现。理想无线可达、无重传、roofline 和功率代理均为架构假设；结果不能解释为完整模型推理或实物芯片测量。

- [完整报告（Markdown）](Results/report.md)
- [完整报告（Word）](Results/simulation_report.docx)
- [全部参数及含义](Results/simulation_parameters.csv)
- [配对结果与置信区间](Results/summary.csv)
- [验证记录](Results/validation.json) / [独立全量审计](Results/independent_audit.json)

## 实验覆盖与原始数据

全部 **1,524 条计划记录**完成：576 个完整层主配置、576 个通信回放主配置、336 条单轴敏感性记录、36 条能耗账本重计价记录。其中 **1,296 个配置实际独立计时**，192 条敏感性记录复用相同或不受该轴影响的时序配置。没有减少请求、cohort、种子或模型。

三个模型为 GPT-oss-20B、Qwen3-30B-A3B、ERNIE-4.5-21B-A3B-PT。每种子 32 请求，种子 11/23/37，B=1/8，uniform/hot_shift，Prefill/Decode，八种方法。每个 cohort 的状态和时钟独立重置；Decode 所有有效请求按轮同步，无 padding。

| 文件 | 含义 |
|---|---|
| `config_resolved.json` / `simulation_parameters.csv` | 原始机器配置、派生参数、容量、软件版本；参数含义 |
| `inputs/` / `input_manifest.json` | 小型官方输入、revision、SHA-256、原始行号、token 长度 |
| `routes/*.jsonl.gz` / `route_manifest.json` | 36 个冻结合成路由及校验值 |
| `run_manifest.csv` | 所有主实验、回放、敏感性、能耗记录的状态和复用引用 |
| `layer_results.csv` / `replay_results.csv` | 主实验逐 cohort/Decode 轮增量与通信回放观测 |
| `sensitivity_layer_results.csv` / `sensitivity_replay_results.csv` | 全部敏感性逐项观测，含默认点 |
| `cohort_summary.csv` / `tail_latency.csv` | cohort 总量；单层观测/token/请求的尾延迟 |
| `method_summary.csv` / `summary.csv` | 每个 case 的绝对量；等权几何均值、配对 95% CI |
| `mechanisms.csv` / `sensitivity_summary.csv` | 机制消融和四个敏感性轴 |
| `energy_breakdown.csv` / `energy_sensitivity.csv` | energy proxy 分项和 0/0.1/0.4 W 基带功率重计价 |
| `resource_audit.csv` / `runs/*.json.gz` | 每 Die 忙时、有限缓冲、微批、原始轮增量和 token 时间 |
| `validation.json` / `independent_audit.json` | 单位/数学/守恒/负面对照/全量字节与 MAC 审计 |
| `figures/` / `report.md` / `simulation_report.docx` | 图表及说明文档 |
| `artifact_checksums.json` | 全部本次结果文件的 SHA-256，排除该校验文件自身 |

`runs/*.json.gz` 是本次原始实验记录，不是历史阶段快照。每个 token 的 `token_timings` 列依次为 `request_rank, token_index, admission_ps, completion_ps, latency_ps`。Prefill latency 包含 admission 等待。Decode 每轮能耗和活动量仅计增量，cohort 总量不重复摊入每轮。

回放 `goodput_Gbps` 的分子为 `Σ 2*d*远端目的 Die 数`；广播逻辑交付量与实际物理发送 bit 分开。`domain_*_data_busy_fraction` 是该频域数据空口的区间并集占比，不含控制/保护；`domain_*_RU_utilization` 是实际数据服务 RU×时间占全域总 RU×时间的比例。没有远端通信时吞吐留空，汇总报告排除数。

## 重跑

使用 Python 3.12（本次 3.12.14）。锁定依赖不包括 torch、整模型权重或 Ramulator。

```bash
bash scripts/setup_env.sh
bash scripts/reproduce.sh
```

`reproduce.sh` 默认离线读取已提交的 tokenizer/config/CSV，并执行验证、完整运行、全量审计、2000 次分层配对 bootstrap、图表和报告生成。已存在且配置/输入/路由/引擎 SHA-256 完全一致的原始运行会复用。新目录重跑可使用：

```bash
bash scripts/reproduce.sh /tmp/tov-reproduction
```

新目录会复制并校验仓库中冻结的小型输入。默认 4 个工作进程，可用 `TOV_WORKERS=2` 调整主机进程数；不改变芯片参数。若从零下载输入，程序只允许 config/tokenizer 与指定 CSV 文件，并首次冻结仓库 commit。

```bash
.venv/bin/python -m tov_sim.run_all --spec ToV_MoE_MINIMAL_SIM_SPEC.md --output Results --workers 4
.venv/bin/python -m tov_sim.run_all --output Results --summarize-only
.venv/bin/python -m pytest -q
.venv/bin/python -m tov_sim.decode_routes Results/routes/qwen3_30b_a3b_prefill_hot_shift_11.jsonl.gz
```

仿真器使用 Python `heapq` 跳跃到事件时间，整数 ps 记录依赖，不把独立通信和计算总量相加代替调度。NMP 与网络 DMA 使用固定 HB 分区；TSV 端口、无线方向/RU、staging、完整包流量及 contributor 集合均检查守恒。

参数和所有来源链接以所附规范为准。FP8 随机小矩阵校验只验证量化与累加路径，不是预训练模型准确率测试；energy proxy 未校准为真实芯片功耗。
