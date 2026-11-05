# 完整单MoE层仿真参数审批表

## 审批范围与当前状态

审批版本：VDIE-FULL-v1.0-review-1；日期：2026年11月5日（Asia/Shanghai）；状态：等待用户审批，尚未运行任何本次仿真。

本稿完整保留附件211项机器参数及含义，补充派生值、实验矩阵、来源版本和实现约定。原规范及其YAML不改写；新增约定在configs/review_options.yaml单独列出。

本次目标是按新规范重新实现并运行完整单MoE层仿真，再整理仓库和生成最终Word报告。现有stage1–stage5代码尚不满足新规范；历史测试与结果不算本次证据。

审批门槛来自用户本轮要求“我通过之后再进行仿真”。附件中“一次性运行”适用于审批之后，不覆盖该门槛。当前仅做文本解析、公开来源检查和静态公式验算。

建议一并审批两项明确化：①FP32向量能耗倍率单因素范围2/4/8，增加240条能耗重算；②all_non_source广播把所有非源Die加入实际RX和半双工约束，因此重跑时序。另保留60条仅对Decode有意义的Prefill复用概率记录，标为等价控制。

论文数值及证据标签按上传规范保留；本轮实际核对的是三模型官方配置、数据集和算术一致性，不宣称重新核验了所有论文原文。

## 1. 仿真目标与边界

测量单MoE层Base输入就绪→Base到owner→Gate/Top-k→Dispatch→专家计算→可选Local Combine→owner最终Combine→Base收到输出的端到端延迟和模型估计能耗。

仅计该层热态行为：权重预驻留；不计Attention、KV cache、其它层、启动装载和residual add。性能仿真使用真实模型尺寸及张量描述符；数值等价性另用指定小尺寸FP8矩阵验证。

数据为真实ChatGPT Prompts文本与官方分词长度；隐藏activation和router为受控合成，必须标记dataset_tokens_synthetic_router。Decode为教师强制回复token流量重放，不能当作实际模型生成或整模型TBT。

两种能耗账本同时报告：frontend_proxy（计算/向量/内存/刷新/泄漏/RF/待机）与baseband_stress（另加在线EQ及OFDMA基带proxy）。速率、OFDMA、16节点覆盖和PPA外推证据限制随报告保留。

## 2. 三模型与核心硬件

| 模型 | d / h | routed / Top-k / shared | 每Die占用 / 剩余（B） | 含义 |
|---|---|---|---|---|
| M1：openai/gpt-oss-20b | 2880 / 2880 | 32 / 4 / 0 | 54,083,520 / 9,916,480 | 第2个Transformer block；M3两shared分支在owner/buddy各执行一次，其余副本不激活 |
| M2：Qwen/Qwen3-30B-A3B | 2048 / 768 | 128 / 8 / 0 | 42,214,464 / 21,785,536 | 第2个Transformer block；M3两shared分支在owner/buddy各执行一次，其余副本不激活 |
| M3：baidu/ERNIE-4.5-21B-A3B-PT | 2560 / 1536 | 64 / 6 / 2 | 63,355,280 / 644,720 | 第2个Transformer block；M3两shared分支在owner/buddy各执行一次，其余副本不激活 |

| 参数 | 主值 | 含义 |
|---|---|---|
| 硬件 | 16 V-Die + 1 Base；200MHz；每Die 1TOPS | 计算、Gate、向量任务共用每Die资源；2OP/MAC |
| 内存 | 每Die 64,000,000B eDRAM；含4MiB工作区 | 64bank，读写/DMA共享默认204.8GB/s接口 |
| 精度 | FP8 E4M3FN输入/权重；FP32累加；128分块 | 不得用INT8替代；链路量化可能改变结合次序误差 |
| 无线 | base/peer两独立介质；132Gb/s单播、80Gb/s多播 | 每介质共享一份总带宽；同组半双工 |
| MAC | 100ns请求帧；16RU；OFDMA效率14/17 | Base/peer实际control为25/20ns，另5ns guard |
| 样本 | 5种子，每种子128请求；Prefill B=1/8/32；Decode B=1/8/32/128 | prompt≤1024 token；reply≤64步；原生长度不补齐 |
| 主结果路由 | uniform；Decode reuse=0 | Zipf α=1.2及reuse=0.8作为单因素敏感性 |
| 刷新 | 11µs；约3.02545W/Die，16Die约48.40727W | 此为给定参数的派生功耗，尚不是仿真能耗结果 |

## 3.1 完整参数表：规范版本

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| spec.version | 1.0 | — | 本次执行规范版本；2026-11-04 核对文献 | D |

## 3.2 完整参数表：仿真引擎、种子与数值容限

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| simulation.engine | discrete_event | — | Python 架构级离散事件仿真；不执行完整 LLM | D |
| simulation.time_unit_ps | 1 | ps | 事件时间使用整数皮秒，避免浮点累积误差 | D |
| simulation.seed_list | [20261104, 20261105, 20261106, 20261107, 20261108] | — | 五组配对随机种子；相同 trace 用于所有 baseline | D |
| simulation.fp8_numeric_test_tokens | 16 | token | 小尺寸数值校验的 token 数；性能仿真不实际做数十亿次乘加 | D |
| simulation.numeric_test_d | 64 | element | 小尺寸数值校验隐藏维度 | D |
| simulation.numeric_test_h | 32 | element | 小尺寸数值校验专家中间维度 | D |
| simulation.fp32_rtol | 1e-05 | — | 无链路量化时不同 Combine 顺序的相对误差容限 | D |
| simulation.fp32_atol | 1e-06 | — | 无链路量化时的绝对误差容限 | D |
| simulation.fp8_nrmse_limit | 0.05 | — | 受控随机小算例的 FP8 输出归一化误差阈值；不是模型精度保证 | D |
| simulation.fp8_cosine_min | 0.99 | — | 上述小算例的余弦相似度检查阈值 | D |

## 3.3 完整参数表：M1：GPT-oss-20B

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| models.M1.repo | openai/gpt-oss-20b | — | 模型及其 tokenizer 的官方仓库 | [P3] [C1] [C4] |
| models.M1.total_params_b | 20.91 | 10^9 parameters | 整模型参数量，仅说明规模；不乘入单层开销 | [P3] [C1] [C4] |
| models.M1.active_params_b | 3.61 | 10^9 parameters/token | 整模型激活参数量；各论文统计口径不完全相同 | [P3] [C1] [C4] |
| models.M1.d_model | 2880 | element | 单层输入、输出 activation 长度 | [P3] [C1] [C4] |
| models.M1.d_ff | 2880 | element | 每个 routed expert 的中间维度 | [P3] [C1] [C4] |
| models.M1.routed_experts | 32 | expert/layer | 本层独立 routed expert 数量 | [P3] [C1] [C4] |
| models.M1.top_k | 4 | expert/token | 每 token 选择的不重复 routed expert 数量 | [P3] [C1] [C4] |
| models.M1.shared_experts | 0 | expert/layer | 每 token 必须计算的 shared expert 数；不占 top-k | [P3] [C1] [C4] |
| models.M1.shared_d_ff | 0 | element | 每个逻辑 shared expert 的中间维度；M3 官方融合宽度为 3072 | [P3] [C1] [C4] |
| models.M1.transformer_layers | 24 | layer | 整模型层数，仅用于标识模型 | [P3] [C1] [C4] |
| models.M1.moe_layers | 24 | layer | 整模型 MoE 层数；本任务只计其中一层 | [P3] [C1] [C4] |
| models.M1.layer_index | 1 | zero-based | 统一选第 2 个 Transformer block 的 MoE；M3 第 1 个 block 是 dense | D；结构依据 [P3] [C1] [C4] |
| models.M1.expert_linear_bias | true | bool | M1 的 gate/up/down 含 bias；M2/M3 不含 | [P3] [C1] [C4] |
| models.M1.router_linear_bias | true | bool | M1 router 线性映射有 bias | [P3] [C1] [C4] |
| models.M1.router_correction_fp32 | false | bool | M3 的专家选择 correction bias；合成 trace 设零，真 trace 保留 | [C5]；默认值 D |
| models.M1.activation | gpt_oss_swiglu | — | 专家非线性；禁止把三种模型都改成同一激活 | [P3] [C1] [C4] |
| models.M1.swiglu_alpha | 1.702 | — | GPT-oss 自定义 SwiGLU sigmoid 系数 | [C4] |
| models.M1.swiglu_limit | 7.0 | — | GPT-oss gate 上限及 up 分支双侧限幅 | [C1] [C4] |
| models.M1.expert_matrix_params | 24883200 | parameter | 3*d_model*d_ff，三个矩阵的元素总数 | 派生：[P3] [C1] [C4] |
| models.M1.expert_params_including_bias | 24891840 | parameter | 单个专家参数量，M1 加上三个线性 bias | 派生：[P3] [C1] [C4] |
| models.M1.expert_storage_bytes | 24898464 | byte | FP8 矩阵/bias 加所有 FP32 block scales | 派生：模型尺寸 + [P6] 的分块规则 |
| models.M1.router_storage_bytes | 92288 | byte/die | 完整 router 副本及其 scale/bias/correction 的容量 | 派生：模型配置与本规范量化格式 |
| models.M1.resident_bytes_per_die | 54083520 | byte/die | 专家 + router + 固定 4MiB 工作区；必须不超过 64,000,000 | 派生：本规范映射与容量核算 |

## 3.4 完整参数表：M2：Qwen3-30B-A3B

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| models.M2.repo | Qwen/Qwen3-30B-A3B | — | 模型及其 tokenizer 的官方仓库 | [P4] [C2] |
| models.M2.total_params_b | 30.5 | 10^9 parameters | 整模型参数量，仅说明规模；不乘入单层开销 | [P4] [C2] |
| models.M2.active_params_b | 3.3 | 10^9 parameters/token | 整模型激活参数量；各论文统计口径不完全相同 | [P4] [C2] |
| models.M2.d_model | 2048 | element | 单层输入、输出 activation 长度 | [P4] [C2] |
| models.M2.d_ff | 768 | element | 每个 routed expert 的中间维度 | [P4] [C2] |
| models.M2.routed_experts | 128 | expert/layer | 本层独立 routed expert 数量 | [P4] [C2] |
| models.M2.top_k | 8 | expert/token | 每 token 选择的不重复 routed expert 数量 | [P4] [C2] |
| models.M2.shared_experts | 0 | expert/layer | 每 token 必须计算的 shared expert 数；不占 top-k | [P4] [C2] |
| models.M2.shared_d_ff | 0 | element | 每个逻辑 shared expert 的中间维度；M3 官方融合宽度为 3072 | [P4] [C2] |
| models.M2.transformer_layers | 48 | layer | 整模型层数，仅用于标识模型 | [P4] [C2] |
| models.M2.moe_layers | 48 | layer | 整模型 MoE 层数；本任务只计其中一层 | [P4] [C2] |
| models.M2.layer_index | 1 | zero-based | 统一选第 2 个 Transformer block 的 MoE；M3 第 1 个 block 是 dense | D；结构依据 [P4] [C2] |
| models.M2.expert_linear_bias | false | bool | M1 的 gate/up/down 含 bias；M2/M3 不含 | [P4] [C2] |
| models.M2.router_linear_bias | false | bool | M1 router 线性映射有 bias | [P4] [C2] |
| models.M2.router_correction_fp32 | false | bool | M3 的专家选择 correction bias；合成 trace 设零，真 trace 保留 | [C5]；默认值 D |
| models.M2.activation | silu_glu | — | 专家非线性；禁止把三种模型都改成同一激活 | [P4] [C2] |
| models.M2.expert_matrix_params | 4718592 | parameter | 3*d_model*d_ff，三个矩阵的元素总数 | 派生：[P4] [C2] |
| models.M2.expert_params_including_bias | 4718592 | parameter | 单个专家参数量，M1 加上三个线性 bias | 派生：[P4] [C2] |
| models.M2.expert_storage_bytes | 4719744 | byte | FP8 矩阵/bias 加所有 FP32 block scales | 派生：模型尺寸 + [P6] 的分块规则 |
| models.M2.router_storage_bytes | 262208 | byte/die | 完整 router 副本及其 scale/bias/correction 的容量 | 派生：模型配置与本规范量化格式 |
| models.M2.resident_bytes_per_die | 42214464 | byte/die | 专家 + router + 固定 4MiB 工作区；必须不超过 64,000,000 | 派生：本规范映射与容量核算 |

## 3.5 完整参数表：M3：ERNIE-4.5-21B-A3B

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| models.M3.repo | baidu/ERNIE-4.5-21B-A3B-PT | — | 模型及其 tokenizer 的官方仓库 | [P5] [C3] [C5] |
| models.M3.total_params_b | 21.0 | 10^9 parameters | 整模型参数量，仅说明规模；不乘入单层开销 | [P5] [C3] [C5] |
| models.M3.active_params_b | 3.0 | 10^9 parameters/token | 整模型激活参数量；各论文统计口径不完全相同 | [P5] [C3] [C5] |
| models.M3.d_model | 2560 | element | 单层输入、输出 activation 长度 | [P5] [C3] [C5] |
| models.M3.d_ff | 1536 | element | 每个 routed expert 的中间维度 | [P5] [C3] [C5] |
| models.M3.routed_experts | 64 | expert/layer | 本层独立 routed expert 数量 | [P5] [C3] [C5] |
| models.M3.top_k | 6 | expert/token | 每 token 选择的不重复 routed expert 数量 | [P5] [C3] [C5] |
| models.M3.shared_experts | 2 | expert/layer | 每 token 必须计算的 shared expert 数；不占 top-k | [P5] [C3] [C5] |
| models.M3.shared_d_ff | 1536 | element | 每个逻辑 shared expert 的中间维度；M3 官方融合宽度为 3072 | [P5] [C3] [C5] |
| models.M3.transformer_layers | 28 | layer | 整模型层数，仅用于标识模型 | [P5] [C3] [C5] |
| models.M3.moe_layers | 27 | layer | 整模型 MoE 层数；本任务只计其中一层 | [P5] [C3] [C5] |
| models.M3.layer_index | 1 | zero-based | 统一选第 2 个 Transformer block 的 MoE；M3 第 1 个 block 是 dense | D；结构依据 [P5] [C3] [C5] |
| models.M3.expert_linear_bias | false | bool | M1 的 gate/up/down 含 bias；M2/M3 不含 | [P5] [C3] [C5] |
| models.M3.router_linear_bias | false | bool | M1 router 线性映射有 bias | [P5] [C3] [C5] |
| models.M3.router_correction_fp32 | true | bool | M3 的专家选择 correction bias；合成 trace 设零，真 trace 保留 | [C5]；默认值 D |
| models.M3.activation | silu_glu | — | 专家非线性；禁止把三种模型都改成同一激活 | [P5] [C3] [C5] |
| models.M3.expert_matrix_params | 11796480 | parameter | 3*d_model*d_ff，三个矩阵的元素总数 | 派生：[P5] [C3] [C5] |
| models.M3.expert_params_including_bias | 11796480 | parameter | 单个专家参数量，M1 加上三个线性 bias | 派生：[P5] [C3] [C5] |
| models.M3.expert_storage_bytes | 11799360 | byte | FP8 矩阵/bias 加所有 FP32 block scales | 派生：模型尺寸 + [P6] 的分块规则 |
| models.M3.router_storage_bytes | 164176 | byte/die | 完整 router 副本及其 scale/bias/correction 的容量 | 派生：模型配置与本规范量化格式 |
| models.M3.resident_bytes_per_die | 63355280 | byte/die | 专家 + router + 固定 4MiB 工作区；必须不超过 64,000,000 | 派生：本规范映射与容量核算 |

## 3.6 完整参数表：精度与量化

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| precision.activation | FP8_E4M3FN | — | 所有矩阵输入、层输入/输出、跨 Die activation/partial 的载荷类型 | U；具体格式参考 [P6] |
| precision.weight | FP8_E4M3FN | — | 专家与 router 权重、线性 bias 按 FP8 存储；M1 不沿用原 MXFP4 | U；格式 [P6]；M1 原格式 [P3] |
| precision.accumulator | FP32 | — | MAC 累加、softmax、路由概率、Combine 累加的内部精度 | D，参考 [P6] 的高精度累加 |
| precision.weight_block | [128, 128] | element | FP8 权重每个二维 block 共用一个 FP32 scale | [P6] §3.3.2；对本架构的采用为 D |
| precision.activation_block | 128 | element | activation 每 token 每 128 元素一个 scale | [P6] §3.3.2 |
| precision.scale_bytes | 4 | byte | 每个 scale 的 FP32 字节数 | D |
| precision.fp8_max | 448 | — | E4M3FN 最大有限值；amax/448 决定 scale | E4M3FN 格式定义，采用 [P6] |
| precision.scale_zero_amax | 1.0 | — | 全零块的 scale，避免除零 | D |
| precision.rounding | nearest_even_saturate | — | 最近偶数舍入、有限范围饱和；不得用 INT8 代替 FP8 | D |

## 3.7 完整参数表：硬件与缓冲区

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| hardware.vdies | 16 | die | Expert V-Die 数量 | U |
| hardware.base_dies | 1 | die | Base Die 数量；只作为本层输入/输出端点 | U/按系统描述 |
| hardware.clock_hz | 200000000 | Hz | 主数字时钟；一个周期 5ns，不是 THz 载频 | U |
| hardware.peak_ops_s | 1000000000000 | OP/s/die | FP8 峰值 1TOPS；乘法和加法各计 1 OP | U；OP 口径 [P2] Table II |
| hardware.ops_per_mac | 2 | OP/MAC | 一次乘加计两个操作，防止 2 倍算力错误 | [P2] Table II 脚注 |
| hardware.edram_bytes | 64000000 | byte/die | 64MB 按十进制解释，比 64MiB 更保守；所有驻留内容合计受限 | U；单位约定 D |
| hardware.workspace_bytes | 4194304 | byte/die | 4MiB 工作区从上述 64MB 内扣除，不额外赠送容量 | D |
| hardware.buffer_partition_bytes | [524288, 1048576, 1048576, 1048576, 524288] | byte | 依次为权重 tile 双缓冲、RX、TX、activation/FP32 partial、控制描述符 | D |
| hardware.tx_return_reserve_bytes | 524288 | byte/die | TX缓冲中至少512KiB保留给Return/Output，避免Dispatch堵塞完成路径 | D |
| hardware.rx_return_reserve_bytes | 524288 | byte/die | RX缓冲中至少512KiB保留给Return，防止Dispatch占满接收空间 | D |
| hardware.partial_owner_reserve_bytes | 262144 | byte/die | activation/partial分区中256KiB保留给owner最终累加器，余下用于专家中间值 | D |
| hardware.owner_inflight_limit | 16 | token/die | 每 owner 最多保留 16 个未完成 token，超限在 Base 排队 | D |
| hardware.vector_lanes | 64 | element-OP/cycle | Top-k、激活函数、scale、Combine 的向量处理宽度 | D |
| hardware.compute_utilization | 1.0；单因素扫描：[0.5, 0.75, 1.0] | ratio | tile 内理想计算效率；边界用实际尺寸，不凭空填充全 tile | D；强制做敏感性检验 |
| hardware.compute_setup_cycles | 2 | cycle/task | 一项矩阵或向量任务的固定启动开销 | D |
| hardware.compute_voltage_v | 1.1 | V | 目标计算域工作电压，按 [P2] Fig.21 约 215MHz 点选择以覆盖 200MHz | [P2] Fig.21 图读近似 + D |
| hardware.edram_voltage_v | 0.9 | V | eDRAM 采用独立电压域；不是将两个论文工艺宣称为同一芯片 | [P7] Table 1；集成为 A |
| hardware.temperature_c | 27 | degC | eDRAM 参数标定温度；不模拟热反馈 | [P7] Table 1 |

## 3.8 完整参数表：计算任务与能耗修正

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| compute.tile_m | 16 | token | 专家一次最多合并 16 个已到达 token | D |
| compute.tile_n | 128 | element | 输出维度 tile 宽度 | D，匹配 [P6] 分块 |
| compute.tile_k | 128 | element | 归约维度 tile 宽度 | D，匹配 [P6] 分块 |
| compute.batch_wait_cycles | 20 | cycle | 最老 token 最多等 100ns 聚批；满 16 或上游窗口关闭可提前发射 | D |
| compute.activation_ops_per_element | 8 | vector-OP | 普通 SwiGLU 每中间元素的保守等效成本 | A：非线性电路未提供，固定于全部 baseline |
| compute.gpt_activation_ops_per_element | 12 | vector-OP | 含限幅与额外乘加的 GPT-oss SwiGLU 成本 | A；函数依据 [C4] |
| compute.softmax_ops_per_element | 8 | vector-OP | softmax/归一化的等效标量成本 | A |
| compute.quantize_ops_per_element | 3 | vector-OP | amax、缩放、转换的等效成本；scale 决策按 128 元素组 | A |
| compute.vector_energy_multiplier | 4.0；单因素扫描：[2.0, 4.0, 8.0]（新增范围，待审批） | ratio | FP32 向量操作相对 FP8 等效 OP 的能耗倍率 | A；必须随能耗系数一起检验 |
| compute.energy_scale | 1.0；单因素扫描：[0.5, 1.0, 2.0] | ratio | FP16 原型迁移到 FP8 的额外修正默认不赠送降耗 | A；1 表示不作位宽线性减半 |
| compute.extra_idle_power_w | 0.01；单因素扫描：[0.0, 0.01, 0.05] | W/die | 计算/控制域额外待机功耗，独立于 eDRAM 与 RF | A；[P2] 未给本目标 Die 待机功耗 |
| compute.base_io_idle_power_w | 0.01；单因素扫描：[0.0, 0.01, 0.05] | W | 只计 Base 在本层边界内的 I/O 控制待机 | A；不计 Base 的 Attention 等计算 |

## 3.9 完整参数表：eDRAM端口、刷新与能耗

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| memory.banks | 64 | bank/die | eDRAM 逻辑 bank 数 | D |
| memory.external_bytes_per_bank_cycle | 16 | byte/cycle | 每 bank 对计算/DMA 可提供 128bit，读写共享额度 | D |
| memory.bandwidth_scale | 1.0；单因素扫描：[0.5, 1.0, 2.0] | ratio | 有效用户端口吞吐相对64bank×16B/cycle的倍率；固定访问时延不缩放 | D；敏感性取0.5/1/2 |
| memory.access_latency_cycles | 1 | cycle | 最小访问时延 5ns；文献 2ns 访问能力向 200MHz 周期取整 | [P7] §3；向目标时钟换算 D |
| memory.read_pj_bit | 0.025 | pJ/bit | 小宏 read 能量，25fJ/bit；非 64MB 成品实测 | [P7] §3，电路仿真；容量扩展 A |
| memory.write_pj_bit | 0.04 | pJ/bit | 小宏 write 能量，40fJ/bit | [P7] §3，电路仿真；容量扩展 A |
| memory.refresh_pj_bit | 0.065 | pJ/bit/refresh | 一次完整 read+write refresh，65fJ/bit | [P7] §3 |
| memory.refresh_period_ps | 11000000；单因素扫描：[6000000, 11000000, 100000000] | ps | 11us 的标称保持/刷新周期；6us 作为保守敏感性点 | [P7] Table 1；最坏条件见 §3 |
| memory.leakage_pw_bit | 1.25 | pW/bit | 按 Table 1 取泄漏，正文另一工作条件为 0.93；不得混用 | [P7] Table 1 |
| memory.refresh_blocks_user_port | false | bool | 为 refresh 保留内部端口，用户端口不被阻塞；仍全额计刷新能量 | [P7] internal-refresh 模式；大容量集成 A |
| memory.refresh_scope | entire_capacity | — | 每个已启用 Die 的全部 64MB 保持数据；不靠 baseline 改刷新范围 | D，保守配置 |
| memory.energy_scale | 1.0；单因素扫描：[1.0, 2.0, 4.0] | ratio | 小宏扩展到64MB的read/write/refresh能量修正；不改变单列的bitcell泄漏，主值1为乐观proxy | A；强制检验 1/2/4 |

## 3.10 完整参数表：放置与路由器副本

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| mapping.routed | expert_id_mod_16 | — | routed expert e 放在 e mod 16；不复制、不切分 | D |
| mapping.owner | request_ordinal_mod_16 | — | 一个请求的所有 token 固定 owner=req_ordinal mod 16 | D |
| mapping.shared | parity_replica_pair | — | M3：S0 在偶数 Die、S1 在奇数 Die各8份；token 只用 owner 和 owner xor 1 两份 | D；原 shared 宽度依据 [C3] [C5] |
| mapping.router_replicas | 16 | copy | 每 Die 有完整且相同的 router；每 token 仅 owner 执行一次 Gate | D，符合用户数据流 |
| mapping.warm_weights | true | bool | 单层全部权重预驻留；不计启动装载，不发生层间换权重 | D，任务只比较一层热态输入到输出 |

## 3.11 完整参数表：无线介质

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| radio.groups_per_vdie | 2 | group/die | Base 通信组与 peer 通信组各一组 | U |
| radio.channels_per_group | 1 | channel/group | 每组按论文1的一条物理通道计，不把双通道 264Gb/s 给每组 | D；[P1] 通道定义 |
| radio.base_groups | 1 | group | Base 有一条聚合接入信道；不是16条独立满带宽链路 | D，避免扩张 Base 带宽 |
| radio.carrier_hz | 153800000000 | Hz | 实测锁定本振频率 153.8GHz，约 0.154THz | [P1] §V / Fig.21 |
| radio.reference_bandwidth_hz | 45000000000 | Hz | 5层/50um 模拟链路约45GHz带宽，仅作频谱约束背景，不拿载频算速率 | [P1] §II-B，EM 仿真 |
| radio.ptp_raw_bps | 132000000000 | bit/s/channel | 有离线 EQ 的单通道点对点峰值 | [P1] §V / Fig.25–26 / Table II |
| radio.broadcast_raw_bps | 80000000000；单因素扫描：[80000000000, 124000000000] | bit/s/channel | 较长路径广播实测80Gb/s；主实验不使用更乐观124Gb/s | [P1] §V / Fig.27(b)，after EQ |
| radio.rate_scale | 1.0；单因素扫描：[0.25, 0.5, 1.0] | ratio | 16节点覆盖、耦合/干扰与新PHY带来的未知降额；全部方法一起缩放 | A；[P1] 未验证16节点全互连 |
| radio.half_duplex_per_group | true | bool | 同组同一时刻 TX 或 RX；不同组可并发，不能将同组当全双工 | D，符合 [P1] TX/RX/OFF 切换 |
| radio.spatial_reuse | false | bool | 同一介质内不重用相同 RU；16节点共享总频谱预算 | D，保守互扰模型 |
| radio.independent_groups | true | bool | Base 介质与 peer 介质可并行，二者硬件参数相同 | U；两组足够隔离是 A |
| radio.rx_pipeline_cycles | 2 | cycle | 完整包接收后的数字处理延迟 10ns | A |
| radio.propagation_delay_ps | 20 | ps | 短距离传播延迟，仅架构占位，非16-Die实测 | A |
| radio.residual_packet_error | 0.0 | probability | 主实验假设可靠服务；不从两/四层 BER 外推16节点 BER | A；不做物理链路可靠性结论 |
| radio.multicast_receivers | destination_mask；单因素扫描：["destination_mask", "all_non_source"] | — | 控制时段所有节点监听；载荷仅目的 Die 启用接收前端 | D；需对 all_non_source 接收方式做敏感性检验 |

## 3.12 完整参数表：MAC、资源单元与包格式

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| mac.frame_cycles | 20；单因素扫描：[10, 20, 80] | cycle | TDMA/OFDMA 统一100ns调度帧，全部基线一致 | D |
| mac.control_cycles | 4 | cycle/frame | 20ns控制预算下限；降额时按报告/grant字节延长，并从数据窗口扣除 | A |
| mac.guard_cycles | 1 | cycle/frame | 5ns 保护/方向切换，从数据窗口扣除 | A |
| mac.queue_report_bytes | 8 | byte/node/frame | 每节点每组只报告一个最高优先级HOL候选的mask、bytes、age/type和credit | D |
| mac.grant_header_bytes | 8 | byte/frame | 调度grant固定头，包含帧号/状态及CRC | D |
| mac.grant_bytes_per_ru | 4 | byte/RU | 每RU发送者/目的mask/方向映射，T模式仍发同长度grant | D |
| mac.arbitration_cycles | 1 | cycle/frame | 控制报告结束到grant的数字仲裁预算；作为固定并行硬件假设 | A |
| mac.peer_arbiter_die | 0 | die_id | peer介质由Die0仲裁；base介质由Base仲裁，控制通信和固定仲裁延迟显式计费 | D |
| mac.fft_size | 256 | bin | OFDMA 服务模型的归一化子载波格点数；不是已有THz芯片参数 | D |
| mac.data_subcarriers | 224 | bin | 载荷格点，其余用于 pilot/guard | D |
| mac.pilot_subcarriers | 16 | bin | 归一化导频开销 | D |
| mac.guard_subcarriers | 16 | bin | 归一化频域保护开销 | D |
| mac.cp_samples | 16 | sample | 归一化循环前缀开销；额外效率=224/(256+16) | D |
| mac.resource_units | 16 | RU | 每RU含14个数据格点；按帧动态分配，绝不每个RU都给132Gb/s | D |
| mac.starvation_limit_frames | 16 | frame | 等待达到16帧的 flow 优先，保证有限队列无饥饿 | D |
| mac.packet_header_bytes | 32 | byte/packet | 固定网络头，含token UID/owner/消息类型/目的mask/分片信息 | D |
| mac.crc_bytes | 4 | byte/packet | CRC32开销；虽然主实验不注入误码仍计字节 | D |
| mac.dispatch_entry_bytes | 8 | byte/expert | expert_id 2B + gate weight FP32 4B + flags 2B | D |
| mac.mtu_payload_bytes | 4096 | byte/packet | 最大body，超长分片每片重计header/CRC；不计虚假padding token | D |

## 3.13 完整参数表：能耗锚点与账本

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| energy.lo_w_per_node | 0.032 | W/node | 一个共享VCO覆盖本Die两组；热态测试全程保持锁定 | [P1] §III-B为32mW电路仿真；两组共享采用 D |
| energy.tx_increment_w_per_group | 0.00635 | W/active group | (44.7mW−32mW)/2；从L4 TX双通道模块分解 | [P1] §V + 仿真LO功耗，分解值为 A/派生 |
| energy.rx_increment_w_per_group | 0.0056 | W/active group | (43.2mW−32mW)/2；从L3 RX双通道模块分解 | [P1] §V + 仿真LO功耗，分解值为 A/派生 |
| energy.equalizer_extra_w_per_rx | 0.02；单因素扫描：[0.0, 0.02, 0.1] | W/active RX group | 补偿论文1未集成的在线均衡功耗；不能标为实测 | A；另检验0/20/100mW |
| energy.ofdma_extra_w_per_active_group | 0.40896428571428567；单因素扫描：[0.0, 0.1, 0.40896428571428567] | W | 完整基带成本压力情景：347mW*(132/112)；不是已测THz OFDMA增量 | [P8] RX参照 + 速率线性外推 A；TX同值为 A |
| energy.report_modes | ["frontend_proxy", "baseband_stress"] | — | 必须同时输出：不含未标定EQ/OFDMA基带的能量；加入上述压力proxy的能量 | D，不能把任一值称为实测整机能耗 |
| energy.compute_pj_op | 0.06980500749971157 | pJ/OP | 21.4TFLOPS/W @0.9V 按V²外推1.1V；FP8修正默认1 | [P2] Table III + 动态能量V²近似 A |

## 3.14 完整参数表：数据、负载与路由

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| workload.dataset | MohamedRashad/ChatGPT-prompts | — | HybriMoE引用的ChatGPT Prompts，不是awesome-chatgpt-prompts | [P9] §VI-A5/参考文献30；[D1] |
| workload.split | train | — | 官方数据划分；本次页面列360行，执行时记录实际版本行数 | [D1] |
| workload.prompt_column | human_prompt | — | Prefill输入文本列 | [D1] |
| workload.response_column | chatgpt_response | — | Decode教师强制重放文本列，不宣称本模型生成 | [D1] |
| workload.requests_per_seed | 128 | request | 清洗后无放回采样128个请求；不足时失败，不偷偷复用 | D |
| workload.prefill_batches | [1, 8, 32] | request/batch | 原生prompt长度，每批独立释放并排空 | D；区分prefill/decode参考 [P9] |
| workload.prefill_length_cap | 1024 | token/request | 仅截断长prompt，不补齐短prompt | D；[P9]也报告至1024 token的prefill工作负载 |
| workload.decode_batches | [1, 8, 32, 128] | request/batch | 每活跃请求每轮只产生一个MoE输入token | D |
| workload.decode_steps_cap | 64 | step/request | 最多重放回复前64个token，短回复自然结束 | D |
| workload.decode_refill | false | bool | 同一cohort不回填请求，结束时报告真实有效B的变化 | D |
| workload.arrival | closed_cohort_barrier | — | Prefill批内t=0可用；Decode下一轮等本轮全部输出返回再释放 | D，阻止将完整response一次当decode输入 |
| workload.tokenization | official_plain_text_no_special_tokens | — | 每模型用官方tokenizer；编码原字段，不加入聊天模板 | D，固定长度口径 |
| workload.input_trace_mode | dataset_tokens_synthetic_router | — | 真实数据集token/长度 + 受控合成路由；不伪称真实隐藏状态/真实专家命中 | D，CPU默认可完整执行 |
| workload.routing_profile | uniform；单因素扫描：["uniform", "zipf"] | — | 主结果：所有专家可交换，无人为热点以保证无偏基线 | D |
| workload.zipf_alpha | 1.2 | — | 热点敏感性：先随机置换expert排名，再按r^(-1.2)生成偏置 | D；不是论文测得的特定模型分布 |
| workload.decode_reuse_probability | 0.0；单因素扫描：[0.0, 0.8] | probability | 主结果不强加跨token相关性；敏感性另取0.8 | D |
| workload.source_revision | resolve_and_freeze | — | 首次读取冻结dataset/tokenizer commit及SHA256；后续全部离线复用 | D，禁止随机跟随main变化 |

## 3.15 完整参数表：实验组合、扫描范围与统计

| 参数键 | 主值 / 扫描值 | 单位 | 含义与使用规则 | 依据 |
|---|---|---|---|---|
| experiment.baseline_ids | ["T_P_N", "T_B_N", "T_P_L", "T_B_L", "O_P_N", "O_B_N", "O_P_L", "O_B_L"] | — | T=单载波TDMA，O=动态OFDMA，P=逐目的Die点对点，B=广播，N/L=无/有Local Combine | U；采用完整2³消融 D |
| experiment.strong_tdma_ids | ["TW_P_N", "TW_B_L"] | — | 无固定帧空尾的包粒度work-conserving TDMA压力对照 | D，防止优势仅来自弱TDMA实现 |
| experiment.sensitivity_cases | ["prefill_B32", "decode_B32"] | — | 敏感性与强TDMA用两个代表负载；三模型、五种子全部运行 | D |
| experiment.sensitivity_baseline_ids | ["T_P_N", "O_P_N", "T_B_L", "O_B_L"] | — | 敏感性保留两组可比的T/O配对；完整三因素消融由主实验八组合承担 | D |
| experiment.rate_scale_values | [0.25, 0.5, 1.0] | ratio | 16节点覆盖和新PHY的速率降额 | A |
| experiment.broadcast_rate_values | [80000000000, 124000000000] | bit/s | 实测较长/较短广播路径的两个标定点 | [P1] Fig.27 |
| experiment.compute_utilization_values | [0.5, 0.75, 1.0] | ratio | 不改变1TOPS峰值的有效计算利用率敏感性 | D |
| experiment.memory_bandwidth_scale_values | [0.5, 1.0, 2.0] | ratio | 改变内部bank/DMA有效吞吐，暴露存储瓶颈 | D |
| experiment.memory_energy_scale_values | [1.0, 2.0, 4.0] | ratio | eDRAM小宏到大容量的能量外推敏感性 | A |
| experiment.refresh_period_values_ps | [6000000, 11000000, 100000000] | ps | 6/11us来自[P7]不同条件；100us来自[P10]另一2T设计，只是替代器件情景 | [P7] [P10]；跨设计采用 A |
| experiment.compute_energy_scale_values | [0.5, 1.0, 2.0] | ratio | 检验FP16到FP8、工作电压与电路映射的不确定性 | A |
| experiment.ofdma_extra_power_values_w | [0.0, 0.1, 0.40896428571428567] | W/group | 理想下界、中间情景、[P8]速率归一化压力proxy | D/A；高点依据 [P8] |
| experiment.equalizer_power_values_w | [0.0, 0.02, 0.1] | W/RX group | 在线EQ未实测，检验共同开销 | A |
| experiment.frame_cycles_values | [10, 20, 80] | cycle | 50/100/400ns请求调度粒度；实际帧长由控制报告/grant/仲裁预算和guard共同约束 | D |
| experiment.multicast_rx_values | ["destination_mask", "all_non_source"] | — | 只目标接收与全体监听载荷的能量差异 | A |
| experiment.routing_profiles | ["uniform", "zipf"] | — | 均匀与热点情景，必须分图报告 | D |
| experiment.decode_reuse_values | [0.0, 0.8] | probability | Decode跨步专家集合相关性的受控敏感性 | D |
| experiment.idle_power_values_w | [0.0, 0.01, 0.05] | W/die | 未测待机项敏感性，Base控制项同步采用同值 | A |
| experiment.bootstrap_resamples | 1000 | resample | 对配对请求/cohort统计作bootstrap，输出95%区间 | D |
| experiment.confidence | 0.95 | probability | 均值/节能率/加速比的置信区间水平 | D |
| experiment.no_cartesian_sensitivity | true | bool | 每次只改变一项；基线点去重，不跑所有敏感性参数笛卡尔积 | D |

## 4. 派生参数及公式复算

下表仅为静态公式计算，不是运行结果。三模型容量、工作区总和、能耗锚点均与规范一致。

| 派生项 | 值 | 单位 | 含义 | 公式 |
|---|---|---|---|---|
| clock_period | 5 | ns | 200MHz数字主时钟周期 | 1e9 / hardware.clock_hz |
| matrix_ops_per_cycle | 5000 | OP/cycle | 每Die矩阵总资源额度 | 1e12 / 2e8 |
| matrix_macs_per_cycle | 2500 | MAC/cycle | 按2OP/MAC换算 | 5000 / 2 |
| memory_user_bandwidth | 204.8 | GB/s/die | 默认共享读写/DMA额度（十进制GB） | 64 * 16 * 2e8 / 1e9 |
| buffer_partition_sum | 4194304 | byte | 工作区分区合计，不额外赠送 | 512KiB+1MiB+1MiB+1MiB+512KiB |
| M1.capacity_margin | 9916480 | byte/die | 扣除专家/router/4MiB工作区后余量 | 64,000,000 - resident_bytes_per_die |
| M1.vector_payload | 2972 | byte | 含FP32 scales的一份FP8 activation/partial | S(d)=d+4*ceil(d/128) |
| M1.contributors_bitmap | 4 | byte | 返回包贡献集合位图 | ceil((routed_experts+shared_experts)/8) |
| M1.expert_ops_per_token | 49766400 | OP | 单个routed expert三矩阵算量；不含向量操作 | 6*d_model*d_ff |
| M1.gate_ops_per_token | 184320 | OP | owner执行一次router矩阵映射 | 2*d_model*routed_experts |
| M1.owner_accumulator_max | 184320 | byte/die | 16个inflight时owner FP32最终累加器总量 | 4*d_model*16 |
| M2.capacity_margin | 21785536 | byte/die | 扣除专家/router/4MiB工作区后余量 | 64,000,000 - resident_bytes_per_die |
| M2.vector_payload | 2112 | byte | 含FP32 scales的一份FP8 activation/partial | S(d)=d+4*ceil(d/128) |
| M2.contributors_bitmap | 16 | byte | 返回包贡献集合位图 | ceil((routed_experts+shared_experts)/8) |
| M2.expert_ops_per_token | 9437184 | OP | 单个routed expert三矩阵算量；不含向量操作 | 6*d_model*d_ff |
| M2.gate_ops_per_token | 524288 | OP | owner执行一次router矩阵映射 | 2*d_model*routed_experts |
| M2.owner_accumulator_max | 131072 | byte/die | 16个inflight时owner FP32最终累加器总量 | 4*d_model*16 |
| M3.capacity_margin | 644720 | byte/die | 扣除专家/router/4MiB工作区后余量 | 64,000,000 - resident_bytes_per_die |
| M3.vector_payload | 2640 | byte | 含FP32 scales的一份FP8 activation/partial | S(d)=d+4*ceil(d/128) |
| M3.contributors_bitmap | 9 | byte | 返回包贡献集合位图 | ceil((routed_experts+shared_experts)/8) |
| M3.expert_ops_per_token | 23592960 | OP | 单个routed expert三矩阵算量；不含向量操作 | 6*d_model*d_ff |
| M3.gate_ops_per_token | 327680 | OP | owner执行一次router矩阵映射 | 2*d_model*routed_experts |
| M3.owner_accumulator_max | 163840 | byte/die | 16个inflight时owner FP32最终累加器总量 | 4*d_model*16 |
| ofdma_efficiency | 14/17 = 0.8235294117647058 | ratio | 数据子载波与CP只乘一次 | 224/(256+16) |
| fullband_unicast_ofdma | 108.70588235294117 | Gb/s | 仅扣OFDMA效率，尚未扣control/guard/包头 | 132 * 14/17 |
| fullband_broadcast_ofdma | 65.88235294117646 | Gb/s | 多目的广播满带宽数据窗口速率 | 80 * 14/17 |
| base.control | 25 | ns/frame | 报告、grant、仲裁取整后的控制窗口 | max(20ns,ceil_5ns(8*N*8/132e9 + 8*72/80e9 + 5ns)) |
| base.data_window | 70 | ns/frame | 默认100ns帧内可用数据窗口 | 100ns - control - 5ns_guard |
| peer.control | 20 | ns/frame | 报告、grant、仲裁取整后的控制窗口 | max(20ns,ceil_5ns(8*N*8/132e9 + 8*72/80e9 + 5ns)) |
| peer.data_window | 75 | ns/frame | 默认100ns帧内可用数据窗口 | 100ns - control - 5ns_guard |
| compute_energy_anchor | 0.06980500749971157 | pJ/OP | FP16原型能效及电压平方外推的FP8 proxy | (1/21.4)*(1.1/0.9)^2 |
| refresh_power_per_die | 3.0254545454545454 | W/die | 全64MB连续后台刷新；为估计值 | 64e6*8*0.065e-12/11e-6 |
| refresh_power_all_vdies | 48.407272727272726 | W | 16Die刷新功耗合计；不是仿真结果 | 16*refresh_power_per_die |
| memory_leak_per_die | 0.0006399999999999999 | W/die | 仅bitcell泄漏，外围另外计费 | 64e6*8*1.25e-12 |
| lo_power_system | 0.544 | W | 17节点热态LO全程功耗 | 17*0.032 |
| control_idle_system | 0.17 | W | 16Die加Base的默认额外待机 | 16*0.01 + 0.01 |
| ofdma_baseband_stress | 0.40896428571428567 | W/active group | DMT整条RX速率外推的压力proxy；TX采用对称假设 | 0.347*132/112 |

## 5. 论文原型标定值与证据边界

保留附件同属参数总表的原型记录。面积/工艺等参照值仅用于解释来源，不擅自加入目标Die的延迟/能量公式。

| 参数 | 值 | 含义及是否用于主仿真 | 来源 |
|---|---|---|---|
| THz原型工艺/模块面积 | 28nm / 0.35mm² | 一个双通道模块；不能再为两组各计0.35mm²并声称是论文值 | [P1] §IV |
| THz双通道吞吐 | 264Gb/s | 2×132，单组预算用132，不是264 | [P1] §V |
| THz无EQ单通道速率 | NRZ 58Gb/s，PAM4 86Gb/s | 仅作非均衡参考；不可配用有EQ广播值声称同一已测条件 | [P1] Fig.24 |
| THz短路径广播 | 124Gb/s/channel | 敏感性上端，主值80 | [P1] Fig.27(a) |
| THz同时双向测试 | 两通道各130Gb/s | 两个通道各用一个方向；不是同一通道全双工 | [P1] Fig.28 |
| TX/RX模块功耗 | L4 TX=44.7mW，L3 RX=43.2mW | 合计87.9mW；87.9mW/264Gb/s≈0.333pJ/bit | [P1] §V，方向按L4→L3确认 |
| SLAM原型总面积 | 3.675mm² | 1.75×2.10；不得直接当目标V-Die面积 | [P2] Fig.16 |
| SLAM原型存储 | CIM 576kbit + SRAM 120kB | 不是64MB eDRAM | [P2] §V-A |
| SLAM CIM宏总面积 | 1.53mm² | 12个48kbit宏，计算面积外推基准 | [P2] Table III |
| SLAM宏峰值吞吐 | 173GFLOPS @1.3V、300MHz | 按2OP/MAC；不是系统Table II中含LSSC的182GOPS | [P2] Fig.16 / Table III |
| 计算能效主锚点 | 21.4TFLOPS/W @0.9V、140MHz | FP16宏级值，换算成OP能耗后外推 | [P2] Table III |
| 不用于200MHz默认值的高能效 | 53.3TFLOPS/W @0.5V、12.5MHz | 还有50%输入元素稀疏条件；不可直接套用密集FP8 200MHz | [P2] Table II/III脚注 |
| SLAM系统31.53TFLOPS/W | @0.6V、40MHz，SLAM工作负载 | 不能替代目标200MHz计算与内存总功耗 | [P2] §V-A / Table I |
| 目标每周期算力 | 5,000OP/cycle = 2,500MAC/cycle | 1e12/2e8；所有Gate/专家计算共享这个Die计算资源 | U派生 |
| 每Die eDRAM接口带宽 | 204.8GB/s | 64bank×16B×200MHz，读写/DMA共享，刷新走独立内部端口 | D派生 |
| FP8等效计算能量 | 0.0698050075pJ/OP | (1/21.4)*(1.1/0.9)^2；不是实测FP8能效 | [P2]+A派生 |
| 计算宏面积proxy | 约13.266mm²/Die | 1.53*(1e12/(173e9*200/300))，只作FP16等效面积参考，不进入延迟/能量公式 | [P2]+A派生 |
| eDRAM 4T bitcell面积 | 0.23µm²/bit | 64MB纯单元几何面积约117.76mm²，尚未计外围和布线 | [P7] Table 1；大容量线性外推A |
| 目标整Die面积 | 不声称有已验证数值 | 本任务不输出未经布局的“实测总PPA”；计算宏proxy与存储单元面积只能分别报告 | 证据边界 |
| eDRAM保持时间保守点 | 6µs | 论文后仿真最坏条件，不与标称11µs混为同一条件 | [P7] §3 |
| eDRAM待机泄漏 | 0.00064W/Die | 64e6*8*1.25e-12，仅bitcell泄漏；外围另由待机/energy_scale覆盖 | [P7]+A派生 |
| eDRAM刷新功耗proxy | 3.02545W/Die | 64e6*8*65e-15/11e-6，16Die约48.407W；必须真实保留这个结果，不隐藏刷新以美化能耗 | [P7]+A派生 |
| OFDMA数据效率 | 14/17≈0.823529 | 224/(256+16)，已经同时包含空载波与CP，只乘一次 | D派生 |
| 每帧控制/数据窗口 | Base 25ns/70ns；peer 20ns/75ns | 按第6.2节控制字节与仲裁公式取整，另有5ns guard；control/guard不是重复收包头 | D派生 |
| 基带压力参考 | 112Gb/s、347mW、5nm | DMT有线接收器的完整接收数据通路，含ADC/DSP等；不能称为28nm THz FFT的实测增量 | [P8] |

## 6. 实验矩阵与运行规模

原规范：840主实验 + 60强TDMA + 2,880个去掉默认点的敏感性记录 = 3,780条。建议补齐向量能耗倍率的240条后，总计4,020条。每条都是128请求的一组模型/种子/负载/基线/参数情景，不是单token或单cohort。

拟执行2,280次时序配置，1,680条只改变能量系数的同轨迹重算，以及60条Prefill复用概率等价控制；两种能耗账本来自同一轨迹，不将它们计成两次时序实验。所有记录现均为pending_approval。

14个原规范敏感性轴加1个补充轴，均仅在Prefill B32、Decode B32上，三模型、五种子、T_P_N/O_P_N/T_B_L/O_B_L四基线执行。每次只改一个轴；idle轴同步改变Base I/O idle。默认点引用主实验，不重复计数。

三个机制在其它两项固定时分别作四对配对比较，并报告交互项。TW_P_N/TW_B_L分别对照O_P_N/O_B_L。不同模型只比较各自相对baseline变化，不用跨模型原始延迟证明机制收益。

| ID | 访问方式 | Dispatch | Local Combine |
|---|---|---|---|
| T_P_N | 单载波帧TDMA | 逐目的Die点对点 | 关 |
| T_B_N | 单载波帧TDMA | 广播 | 关 |
| T_P_L | 单载波帧TDMA | 逐目的Die点对点 | 开 |
| T_B_L | 单载波帧TDMA | 广播 | 开 |
| O_P_N | 动态OFDMA | 逐目的Die点对点 | 关 |
| O_B_N | 动态OFDMA | 广播 | 关 |
| O_P_L | 动态OFDMA | 逐目的Die点对点 | 开 |
| O_B_L | 动态OFDMA | 广播 | 开 |
| TW_P_N | 包粒度work-conserving TDMA | 逐目的Die点对点 | 关 |
| TW_B_L | 包粒度work-conserving TDMA | 广播 | 开 |

| 敏感性轴 | 全部水平 | 默认值 | 新增记录 | 处理方式 |
|---|---|---|---|---|
| radio.rate_scale | [0.25, 0.5, 1.0] | 1.0 | 240 | 重跑时序 |
| radio.broadcast_raw_bps | [80000000000, 124000000000] | 80000000000 | 120 | 重跑时序 |
| hardware.compute_utilization | [0.5, 0.75, 1.0] | 1.0 | 240 | 重跑时序 |
| memory.bandwidth_scale | [0.5, 1.0, 2.0] | 1.0 | 240 | 重跑时序 |
| memory.energy_scale | [1.0, 2.0, 4.0] | 1.0 | 240 | 能量重算 |
| memory.refresh_period_ps | [6000000, 11000000, 100000000] | 11000000 | 240 | 能量重算 |
| compute.energy_scale | [0.5, 1.0, 2.0] | 1.0 | 240 | 能量重算 |
| energy.ofdma_extra_w_per_active_group | [0.0, 0.1, 0.40896428571428567] | 0.40896428571428567 | 240 | 能量重算 |
| energy.equalizer_extra_w_per_rx | [0.0, 0.02, 0.1] | 0.02 | 240 | 能量重算 |
| mac.frame_cycles | [10, 20, 80] | 20 | 240 | 重跑时序 |
| radio.multicast_receivers | ["destination_mask", "all_non_source"] | destination_mask | 120 | 重跑时序（含实际RX半双工） |
| workload.routing_profile | ["uniform", "zipf"] | uniform | 120 | 重跑时序 |
| workload.decode_reuse_probability | [0.0, 0.8] | 0.0 | 120 | 60 Decode时序 + 60 Prefill等价控制 |
| compute.extra_idle_power_w | [0.0, 0.01, 0.05] | 0.01 | 240 | 能量重算 |
| compute.vector_energy_multiplier | [2.0, 4.0, 8.0] | 4.0 | 240 | 能量重算 |

## 7. 实现与统计约定（包含待审批补充）

以下选项把规范正文中未完全机器化的规则明确列出；不会静默覆盖原始YAML。新增向量倍率范围及全员接收语义已在首页单列。若审批后必须调整任何影响结果的参数或口径，将更新表格后再运行受影响实验。

| 选项键 | 值 | 单位 | 含义/执行规则 | 依据 |
|---|---|---|---|---|
| review.approval_id | VDIE-FULL-v1.0-review-1 | — | 本次参数审批版本；当前为待审批，不能据此自动运行。 | D：本次审阅标识 |
| review.simulation_allowed | false | bool | 仅用户明确通过本参数版本后才可运行数值仿真、性能仿真与正式实验。 | 用户本轮要求 |
| review.extra_vector_energy_values | [2.0, 4.0, 8.0] | ratio | 补齐附件要求的FP32向量能耗倍率敏感性；默认4，新增2/8，各自单因素扫描，额外240条。附件未给此轴范围，本稿提出供审批。 | D/A：新增待审批范围 |
| review.all_non_source_semantics | physical_receive_half_duplex | — | 广播时所有非源Die实际开启RX，并加入该组半双工冲突检查；逻辑贡献仍只交给目的mask。重跑时序。避免把同时TX的节点虚构成可RX；这是对附件能耗敏感性与半双工规则的明确化。 | D：待审批实现口径 |
| review.prefill_reuse_semantics | explicit_no_effect_control | — | 复用概率仅作用于Decode。保留60条Prefill p=0.8记录，明确引用相同默认时序/账本；标注不适用/等价复用，不算新仿真。 | D：附件适用范围 |
| runtime.python | 3.12.14 | version | 沿用当前已发布环境的CPython；额外依赖将在实现时固定版本并记入manifest。 | 当前环境观测 |
| runtime.max_processes | 4 | process | 当前4 CPU配额、32GiB内存；按独立运行配置并行，内存不足可减小进程数；不改变任何模型参数。 | 当前环境资源 |
| data.empty_rule | strip_only_for_empty_check | — | 空白字符串视为空；用于分词的原始文本不strip、不改写。 | D：清洗细化 |
| data.duplicate_rule | exact_original_prompt_first_row | — | human_prompt原字符串完全相同才去重，保留原行序首次出现的行。 | 附件第4节 |
| data.row_id | zero_based_original_csv_row | — | row_id在清洗前确定；跨模型保留同一抽样ID及顺序。 | D：可复现标识 |
| random.generator | numpy.PCG64 | — | 无放回抽样、路由和统计使用明确PRNG，不使用Python随机化hash。 | D：补全未指定实现 |
| random.request_sampling | PCG64(seed).choice(clean_rows,128,replace=False) | — | 抽样结果的顺序就是request_ordinal；同种子三个模型共用。 | D：附件采样规则 |
| random.seed_encoding | SHA256(UTF8(compact_JSON_tuple)) -> unsigned_big_endian_integer | — | 路由tuple为[seed,model_id,row_id,mode,token_position]；哈希整数输入PCG64。 | D：附件SHA256规则细化 |
| random.rank_domain | [seed,model_id,"expert_rank"] | — | Zipf排名随机置换使用独立子流；不按Die人为堆积热点。 | D：附件热点规则 |
| random.reuse_domain | [seed,model_id,row_id,"decode_reuse",token_position] | — | Decode复用决策独立于新抽样；第0步总是新采样，后续复用整个集合和权重。 | D：附件复用规则 |
| random.uniform_support | 0<U<1; redraw_exact_zero | — | Gumbel生成避免log(0)；相同logit时按expert_id确定顺序。 | D：数值规则 |
| numeric.test_seed | 20261104 | — | 小尺寸数值验证固定种子；activation标准正态、权重标准差1/sqrt(d)。 | D：附件分布与固定种子 |
| numeric.bias_rule | M1_zero_bias_fixture_plus_nonzero_bias_fixture | — | M1既验证零bias也验证非零bias（标准差1/sqrt(d)的正态），M2/M3不添加线性bias。 | D：检验bias实现 |
| numeric.nrmse_denominator_floor | 1e-12 | — | NRMSE=norm(y-y_ref)/max(norm(y_ref),1e-12)；余弦用展平输出。 | 附件第5.2节 |
| numeric.reference | same_small_FP32_weights_and_inputs | — | 逐模型激活函数与shared结构一致；FP8误差含已定义量化，不宣称真实模型精度。 | 附件第5.2/9节 |
| timing.digital_alignment_ps | 5000 | ps | 计算/DMA启动向上对齐5ns；无线完成向上取整到1ps，不提前交付。 | 派生：200MHz |
| timing.radio_service_remainder | exact_rational_bits_across_frames | — | 跨帧保留分数bit服务量；仅到完整包并完成RX pipeline后消费。 | 附件第6节 |
| timing.simultaneous_event_order | completion_release_then_ready_then_admission | — | 同一时刻先完成与释放credit，再产生依赖任务，最后调度；相同类用token_uid/expert_id等稳定键排序。 | D：确定性实现 |
| compute.task_priority | Combine > Gate > oldest_expert; tile_boundaries | — | 只在tile或向量任务边界切换；所有任务共用Die的1TOPS资源。 | 附件第5.3节 |
| compute.tile_loop | output_N_then_reduction_K | — | tile最多16×128×128；两个有界缓冲，每次调用复用该批token的权重读取。 | 附件第5.3节 |
| memory.dma_service_policy | ready_time_fifo_then_stable_id_nonpreemptive_burst | — | 每Die一个总额度共享接口；计算tile、网络RX/TX实际按burst竞争；刷新有独立内部端口。 | D：细化共享接口调度 |
| memory.tile_buffers_bytes | [262144, 262144] | byte | 512KiB权重tile区等分双缓冲，合计不超既定4MiB工作区。 | D：细化双缓冲分区 |
| memory.input_uses_dispatch_partition | true | bool | Base INPUT与DISPATCH使用非Return的RX额度；OUTPUT与RETURN使用完成路径额度。 | D：完成路径保护 |
| memory.owner_accumulator_bytes | 4*d_model*active_owner_tokens | byte | owner保留FP32最终累加器，限16个inflight；贡献到达累计后及时释放。 | 附件第5.3节 |
| energy.enabled_vdies | 16 | die | 整个cohort窗口全部16个Die的64MB刷新与泄漏均计入，不按路由命中关掉容量。 | 附件第7节 |
| energy.locked_lo_nodes | 17 | node | Base加16Die，全窗口预锁定；每Die两组并发仍只计一次LO。 | 附件第7节 |
| energy.refresh_integration | continuous_average_over_cohort_makespan | — | 按错相后台平均功耗积分；短于刷新周期也收费，不阻塞用户DMA。 | 附件第7节 |
| energy.control_frontends | base:1TX+16RX; peer:1TX+15RX | group | 整个控制窗口（含仲裁和取整余量）保守唤醒；控制不计OFDMA额外基带。 | 附件第7节 |
| statistics.bootstrap_unit | paired_seed_then_cohort_hierarchical | — | 先有放回抽5个种子，再在种子内成组抽cohort；所有baseline共用重抽索引；保留请求/token相关性。 | D：统计实现细化 |
| statistics.bootstrap_rng | SHA256([20261104,"bootstrap",model,mode,B,axis,value]) | — | 1000次配对分层bootstrap；结果取2.5%/97.5%分位，重复可复现。 | D：统计实现细化 |
| statistics.ratio_estimator | ratio_of_paired_aggregate_means | — | 加速比为成对均值延迟之比；节能率为1减成对均值能量比；不混用平均比值；同时保存逐cohort原始比值。 | D：比较口径 |
| statistics.percentiles | p50,p95; linear empirical interpolation | — | 延迟分别按token、请求、cohort/Decode轮报告；不将跨Die busy-time求和冒充延迟。 | 附件第7.2节及D |
| statistics.energy_per_token | sum_cohort_energy / actual_completed_tokens | — | 公共待机能量只累计一次；不按padding或配置最大token数稀释能耗。 | 附件第7.2节 |
| output.complete_rule | all_planned_records_resolved_and_all_acceptance_checks_passed | — | 失败、未运行、能耗重算及等价复用分别标记；不把生成实验清单视为仿真完成。 | 附件第9节 |
| output.final_report | results/simulation_report.docx | path | 审批后完成仿真再生成最终Word：目标、简述、参数表、流程、结果、整理与对照、结论。 | 用户本轮要求 |

## 8. 数据来源冻结与本轮核对

数据集train.csv：原始360行，空行剔除0，重复prompt剔除0，保留360行。三个模型共用每种子抽样的行ID。

已核对三模型官方config的d/h/E/Top-k/shared。Qwen/ERNIE使用moe_intermediate_size而不是dense intermediate_size。此轮仅下载config与数据CSV；tokenizer文件将在审批后从下表固定revision下载并记录SHA256。

不下载整模型权重。原文件、YAML、配置和数据来源哈希会纳入最终manifest；后续网络不可用则报缺失，不替换为随机文本。

| 对象 | 仓库 | 冻结revision | 已核对文件SHA256 |
|---|---|---|---|
| M1 | openai/gpt-oss-20b | 6cee5e81ee83917806bbde320786a8fb61efebee | 3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce |
| M2 | Qwen/Qwen3-30B-A3B | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 | 2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297 |
| M3 | baidu/ERNIE-4.5-21B-A3B-PT | 87db95487941cb39592ee0abca3b9155a6d19c5c | 7383c7387390746c1fa0327f8cab3d5c6b078a60bbf0841d4f42277eece5a5f6 |
| dataset | MohamedRashad/ChatGPT-prompts | 6c258c8c477a799c95246cca4cf8bf734d3c36ad | 4488c4093ee42953b556ffad71e5cb2cd03de85eca0739c2643618e4928bfae7 |

## 9. 审批后的仿真流程与验收

1）按本审批版本完成新实现和依赖锁定，冻结官方tokenizer，生成所有种子/模型/模式的描述符trace及哈希。

2）运行容量/算量、GPT-oss SwiGLU和bias、shared拆分、FP8编码与误差、守恒/背压/半双工/MTU、能量闭合等验收测试；失败须先定位修复，不跳过断言。

3）使用统一入口python -m vdie_moe_sim.run_all --spec VDIE_MOE_SIM_SPEC.md --output results执行完整已批准清单；同输入同trace公平对照，支持已校验进度恢复。该新入口尚待实现，当前不能用旧stage5入口替代。

4）输出每配置原始结果、token/request/cohort/轮统计、能量账本、事件与资源审计、配对效应和95%区间；用同一账本重算只改变能量的情景。

5）检查840主实验、60强TDMA以及所有敏感性记录完成性；验证资源与token守恒、统计可重算。只有全部必需检查通过且记录齐全才设complete=true。

6）从CSV绘图，保留PNG/SVG及对应源数据；编写最终Markdown与Word报告，展示负收益/无收益、证据限制及OFDMA功耗盈亏阈值。

## 10. 仓库整理与最终Word文档

现有仓库工作区开始时无修改。为保证替换过程可核对，在新实现验证后统一移除stage1–stage5专用入口、配置、旧报告、旧图表及旧归档；需要保全的历史文件先放在/workspace/THz-MoE-Simulator之外并校验。保留Git历史，不重写远端历史。

整理后的当前检出只保留本次规范、代码、相关测试、依赖/复现脚本、已批准参数与本次结果。当前审批准备尚未删除旧实现，避免留下不能运行的半成品仓库。

最终results/simulation_report.docx包含：仿真目标、仿真简述、完整参数表、仿真流程、仿真结果、结果整理与对照、结论。结果章节在实际完成后编写，不填入历史阶段数据或理论推导当作实测仿真。

结果目录包含manifest.json、config_resolved.yaml、parameters.csv、冻结traces、raw_runs.csv、token_metrics.parquet、paired_effects.csv、energy_breakdown.csv、validation.json、figures和report.md。

预期最终目录：VDIE_MOE_SIM_SPEC.md；vdie_moe_sim/；tests/；configs/；scripts/；docs/（参数与使用说明）；results/（仅本次完整运行）；pyproject.toml与requirements.lock。

本轮未发布环境、未提交或推送Git；环境安装/启动说明将在新代码可复现后按需更新，发布仍由用户控制。

## 11. 来源索引

U=用户固定；P=论文记录（实测与仿真须区分）；C=官方配置/实现；D=本次设计选择；A=未被目标硬件实测验证的外推或假设。表中“派生”按明确公式计算。

上传原规范SHA256：ba08e82351e2f1e886a7d2a952ebff529b4748d97cfb98281054e8a72377edbd。本文件与parameters.csv完整保留211个参数键；生成检查见static_checks.json。

| 编号 | URL |
|---|---|
| P1 | https://doi.org/10.1109/JSSC.2026.3720291 |
| P2 | https://doi.org/10.1109/JSSC.2024.3402808 |
| P3 | https://arxiv.org/abs/2508.10925 |
| P4 | https://arxiv.org/abs/2505.09388 |
| P5 | https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf |
| P6 | https://arxiv.org/abs/2412.19437 |
| P7 | https://doi.org/10.3390/jlpea14010002 |
| P8 | https://doi.org/10.1109/JSSC.2025.3618188 |
| P9 | https://arxiv.org/abs/2504.05897 |
| P10 | https://arxiv.org/abs/2502.09007 |
| C1 | https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json |
| C2 | https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json |
| C3 | https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/main/config.json |
| C4 | https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py |
| C5 | https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py |
| D1 | https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts |
