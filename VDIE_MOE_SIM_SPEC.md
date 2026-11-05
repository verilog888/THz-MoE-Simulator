# V-Die PIM 单 MoE 层仿真执行规范

版本：1.0；文献核对日期：2026-11-04。交付格式：UTF-8 Markdown，内嵌可直接提取的 YAML。**把本文件整体交给 Codex，即可要求其实现、测试并一次性运行全部规定实验；不拆成交付阶段。**

## 1. 给 Codex 的任务与结论边界

实现一个可在普通 CPU 环境运行的 Python 离散事件仿真器，仿真且只仿真：

`Base输入就绪 → Base到owner → owner的Gate/Top-k → Dispatch → 专家计算 → 可选Local Combine → 回到owner最终Combine → Base收到输出`。

目标是测量 **单 MoE 层延迟和能耗**，通过完整消融分离动态 OFDMA、Dispatch 广播、Local Combine 三项机制的作用。主时钟、计算、内存、路由、缓冲区背压、两组无线介质争用和真实依赖都必须进入时间模型。禁止仅用总字节/总带宽计算整层延迟，禁止预设“优化必然更快、更省电”。

计时边界是 Base 上已就绪的本层输入与 Base 收到本层输出。输入已完成本层前置 Norm；不计 Attention、KV cache、其余层、词嵌入、LM head、外部主存和启动时权重装载。本层输出不含 Transformer residual add。这里的“完整仿真”指完整的层内架构行为，不是完整模型生成回答，也不是晶体管/电磁波形仿真。

**默认数据路径能够在 CPU 完整执行，但不是从预训练整模型提取的真实路由。** ChatGPT Prompts 的文本和分词长度是真实数据；隐藏 activation 和 router logits 用受控随机模型生成，性能计算使用论文中的真实矩阵形状。结果必须标注 `dataset_tokens_synthetic_router`。若另有真实单层 trace，可以用本文规定的接口替换，但不得在缺少上游隐藏状态时把 token embedding 或随机向量称为真实层输入。默认执行不下载数十GB权重、不依赖GPU、不伪造真实模型精度。

### 必须保留的证据限制

- 论文1 [P1] 的 **264Gb/s 是两条通道合计**；单条最高132Gb/s。132Gb/s点对点及124/80Gb/s广播使用离线均衡，0.33pJ/bit不包含片上实时均衡器。论文没有实现本方案的动态OFDMA，也没有验证16个目标V-Die的全互连；文中还明确其既有耦合器不能直接稳健支持超过12层的堆叠。本文的16节点逻辑可达性、新PHY和两组介质隔离均为架构假设，必须进行速率降额检验。
- 论文2 [P2] 是28nm、FP16、SRAM-CIM的SLAM原型。它不是“1TOPS、200MHz、64MB eDRAM、FP8”的成品V-Die。本文保持用户硬件约束，用论文2标定计算能耗/面积基准，用2024年eDRAM论文[P7]补充内存读写、刷新和泄漏。跨精度、容量、工艺和电路组合都是显式外推。
- 2023–2026年公开资料不能直接给出本设计的OFDMA新增基带功耗、16节点链路矩阵、仲裁电路与64MB整体外围功耗。参数表已给出可执行默认值和敏感性范围；这些行必须保留“设计选择/假设”标签，不能伪装成论文实测。

参数来源标记：`U`=用户固定；`P`=论文记录，注意实测/仿真的区别；`C`=官方配置或实现补充；`D`=本仿真的确定设计选择；`A`=未被目标硬件实测验证的外推或假设。“派生”必须可从给出的公式重算。

## 2. 三个模型及容量可行性

按**整模型总参数同属约20–31B量级**选择，非强行使激活参数或单层计算量相等。跨模型只比较各自相对baseline的收益，不能把原始延迟差异归因为某种通信机制。

| 模型 | 结构类型 | 总参数/激活参数 | d / h | Routed / Top-k | Shared | 单个routed expert参数量 |
|---|---|---|---|---|---|---|
| M1：GPT-oss-20B | 相对较大、较少专家，无shared | 20.91B / 3.61B | 2880 / 2880 | 32 / 4 | 0 | 24,891,840，含8,640个bias |
| M2：Qwen3-30B-A3B | 细粒度、多专家，无shared | 30.5B / 约3.3B | 2048 / 768 | 128 / 8 | 0 | 4,718,592 |
| M3：ERNIE-4.5-21B-A3B | 较小、较多专家，有shared | 21B / 约3B | 2560 / 1536 | 64 / 6 | 2，每个h=1536 | 11,796,480 |

模型论文分别为[P3]、[P4]、[P5]，尺寸由对应官方配置[C1]–[C3]补全核验。M1单专家约为M2的5.28倍、M3的2.11倍；M2/M3的专家数分别为M1的4倍/2倍。“小且多”在这里是相对M1的明确定量分类，不暗示M2和M3专家一样大。

所有专家主矩阵均按 `P=3*d*h` 计数；每token三次投影的MAC算量为 `6*d*h OP`。激活函数、bias、Gate/Top-k、scale处理和Combine另外计费。不要用“整模型激活参数量/层数”替代这个精确单层公式。

M3官方实现将两个shared专家融合成一个`h=3072`的MLP [C5]。本设计沿中间维度拆成两个`h=1536`分支：分别切分gate/up的行和down的列，再将输出相加。因无bias，这在实数运算下等价。每个分支复制8份：偶数Die保存S0，奇数Die保存S1。token在owner和`owner xor 1`各计算一个shared分支，总共仍只算两次，不得把16份副本全部激活。

所有Die保留完整router；routed expert `e → e mod 16`。三个模型的权重均可热态驻留，禁止增加片外权重流量或改用更低位宽来“解决”容量问题。

| 模型 | 每Die routed数量 | 每Die shared副本数量 | 每Die总占用（B） | 距64,000,000B余量（B） |
|---|---:|---:|---:|---:|
| M1 | 2 | 0 | 54,083,520 | 9,916,480 |
| M2 | 8 | 0 | 42,214,464 | 21,785,536 |
| M3 | 4 | 1 | 63,355,280 | 644,720 |

占用包含：FP8矩阵与线性bias、128×128权重块的FP32 scales、bias的128元素scales、完整router及M3 correction向量、固定4MiB工作区。边界块只存有效元素，不把GPT-oss的2880维补齐后多算权重。M3仅余约0.645MB，因此实现不得另行添加未计入容量的大型持久缓存。专家列表、路由表、队列描述符纳入工作区，若实际实现超限应触发错误并报告。

## 3. 完整仿真参数表

以下是一张连续的总表。键与末尾YAML一一对应；数量、时间、能量均在代码中使用SI单位或明确的字节/皮秒单位。表中的论文编号均直接链接到论文，官方配置编号链接到原始配置。

| 参数键 | 参数值 | 单位 | 含义/使用规则 | 依据 |
|---|---|---|---|---|
| `spec.version` | `"1.0"` | — | 本次执行规范版本；2026-11-04 核对文献 | D |
| `simulation.engine` | `"discrete_event"` | — | Python 架构级离散事件仿真；不执行完整 LLM | D |
| `simulation.time_unit_ps` | `1` | ps | 事件时间使用整数皮秒，避免浮点累积误差 | D |
| `simulation.seed_list` | `[20261104,20261105,20261106,20261107,20261108]` | — | 五组配对随机种子；相同 trace 用于所有 baseline | D |
| `simulation.fp8_numeric_test_tokens` | `16` | token | 小尺寸数值校验的 token 数；性能仿真不实际做数十亿次乘加 | D |
| `simulation.numeric_test_d` | `64` | element | 小尺寸数值校验隐藏维度 | D |
| `simulation.numeric_test_h` | `32` | element | 小尺寸数值校验专家中间维度 | D |
| `simulation.fp32_rtol` | `1e-05` | — | 无链路量化时不同 Combine 顺序的相对误差容限 | D |
| `simulation.fp32_atol` | `1e-06` | — | 无链路量化时的绝对误差容限 | D |
| `simulation.fp8_nrmse_limit` | `0.05` | — | 受控随机小算例的 FP8 输出归一化误差阈值；不是模型精度保证 | D |
| `simulation.fp8_cosine_min` | `0.99` | — | 上述小算例的余弦相似度检查阈值 | D |
| `models.M1.repo` | `"openai/gpt-oss-20b"` | — | 模型及其 tokenizer 的官方仓库 | [P3] [C1] [C4] |
| `models.M1.total_params_b` | `20.91` | 10^9 parameters | 整模型参数量，仅说明规模；不乘入单层开销 | [P3] [C1] [C4] |
| `models.M1.active_params_b` | `3.61` | 10^9 parameters/token | 整模型激活参数量；各论文统计口径不完全相同 | [P3] [C1] [C4] |
| `models.M1.d_model` | `2880` | element | 单层输入、输出 activation 长度 | [P3] [C1] [C4] |
| `models.M1.d_ff` | `2880` | element | 每个 routed expert 的中间维度 | [P3] [C1] [C4] |
| `models.M1.routed_experts` | `32` | expert/layer | 本层独立 routed expert 数量 | [P3] [C1] [C4] |
| `models.M1.top_k` | `4` | expert/token | 每 token 选择的不重复 routed expert 数量 | [P3] [C1] [C4] |
| `models.M1.shared_experts` | `0` | expert/layer | 每 token 必须计算的 shared expert 数；不占 top-k | [P3] [C1] [C4] |
| `models.M1.shared_d_ff` | `0` | element | 每个逻辑 shared expert 的中间维度；M3 官方融合宽度为 3072 | [P3] [C1] [C4] |
| `models.M1.transformer_layers` | `24` | layer | 整模型层数，仅用于标识模型 | [P3] [C1] [C4] |
| `models.M1.moe_layers` | `24` | layer | 整模型 MoE 层数；本任务只计其中一层 | [P3] [C1] [C4] |
| `models.M1.layer_index` | `1` | zero-based | 统一选第 2 个 Transformer block 的 MoE；M3 第 1 个 block 是 dense | D；结构依据 [P3] [C1] [C4] |
| `models.M1.expert_linear_bias` | `true` | bool | M1 的 gate/up/down 含 bias；M2/M3 不含 | [P3] [C1] [C4] |
| `models.M1.router_linear_bias` | `true` | bool | M1 router 线性映射有 bias | [P3] [C1] [C4] |
| `models.M1.router_correction_fp32` | `false` | bool | M3 的专家选择 correction bias；合成 trace 设零，真 trace 保留 | [C5]；默认值 D |
| `models.M1.activation` | `"gpt_oss_swiglu"` | — | 专家非线性；禁止把三种模型都改成同一激活 | [P3] [C1] [C4] |
| `models.M1.swiglu_alpha` | `1.702` | — | GPT-oss 自定义 SwiGLU sigmoid 系数 | [C4] |
| `models.M1.swiglu_limit` | `7.0` | — | GPT-oss gate 上限及 up 分支双侧限幅 | [C1] [C4] |
| `models.M1.expert_matrix_params` | `24883200` | parameter | 3*d_model*d_ff，三个矩阵的元素总数 | 派生：[P3] [C1] [C4] |
| `models.M1.expert_params_including_bias` | `24891840` | parameter | 单个专家参数量，M1 加上三个线性 bias | 派生：[P3] [C1] [C4] |
| `models.M1.expert_storage_bytes` | `24898464` | byte | FP8 矩阵/bias 加所有 FP32 block scales | 派生：模型尺寸 + [P6] 的分块规则 |
| `models.M1.router_storage_bytes` | `92288` | byte/die | 完整 router 副本及其 scale/bias/correction 的容量 | 派生：模型配置与本规范量化格式 |
| `models.M1.resident_bytes_per_die` | `54083520` | byte/die | 专家 + router + 固定 4MiB 工作区；必须不超过 64,000,000 | 派生：本规范映射与容量核算 |
| `models.M2.repo` | `"Qwen/Qwen3-30B-A3B"` | — | 模型及其 tokenizer 的官方仓库 | [P4] [C2] |
| `models.M2.total_params_b` | `30.5` | 10^9 parameters | 整模型参数量，仅说明规模；不乘入单层开销 | [P4] [C2] |
| `models.M2.active_params_b` | `3.3` | 10^9 parameters/token | 整模型激活参数量；各论文统计口径不完全相同 | [P4] [C2] |
| `models.M2.d_model` | `2048` | element | 单层输入、输出 activation 长度 | [P4] [C2] |
| `models.M2.d_ff` | `768` | element | 每个 routed expert 的中间维度 | [P4] [C2] |
| `models.M2.routed_experts` | `128` | expert/layer | 本层独立 routed expert 数量 | [P4] [C2] |
| `models.M2.top_k` | `8` | expert/token | 每 token 选择的不重复 routed expert 数量 | [P4] [C2] |
| `models.M2.shared_experts` | `0` | expert/layer | 每 token 必须计算的 shared expert 数；不占 top-k | [P4] [C2] |
| `models.M2.shared_d_ff` | `0` | element | 每个逻辑 shared expert 的中间维度；M3 官方融合宽度为 3072 | [P4] [C2] |
| `models.M2.transformer_layers` | `48` | layer | 整模型层数，仅用于标识模型 | [P4] [C2] |
| `models.M2.moe_layers` | `48` | layer | 整模型 MoE 层数；本任务只计其中一层 | [P4] [C2] |
| `models.M2.layer_index` | `1` | zero-based | 统一选第 2 个 Transformer block 的 MoE；M3 第 1 个 block 是 dense | D；结构依据 [P4] [C2] |
| `models.M2.expert_linear_bias` | `false` | bool | M1 的 gate/up/down 含 bias；M2/M3 不含 | [P4] [C2] |
| `models.M2.router_linear_bias` | `false` | bool | M1 router 线性映射有 bias | [P4] [C2] |
| `models.M2.router_correction_fp32` | `false` | bool | M3 的专家选择 correction bias；合成 trace 设零，真 trace 保留 | [C5]；默认值 D |
| `models.M2.activation` | `"silu_glu"` | — | 专家非线性；禁止把三种模型都改成同一激活 | [P4] [C2] |
| `models.M2.expert_matrix_params` | `4718592` | parameter | 3*d_model*d_ff，三个矩阵的元素总数 | 派生：[P4] [C2] |
| `models.M2.expert_params_including_bias` | `4718592` | parameter | 单个专家参数量，M1 加上三个线性 bias | 派生：[P4] [C2] |
| `models.M2.expert_storage_bytes` | `4719744` | byte | FP8 矩阵/bias 加所有 FP32 block scales | 派生：模型尺寸 + [P6] 的分块规则 |
| `models.M2.router_storage_bytes` | `262208` | byte/die | 完整 router 副本及其 scale/bias/correction 的容量 | 派生：模型配置与本规范量化格式 |
| `models.M2.resident_bytes_per_die` | `42214464` | byte/die | 专家 + router + 固定 4MiB 工作区；必须不超过 64,000,000 | 派生：本规范映射与容量核算 |
| `models.M3.repo` | `"baidu/ERNIE-4.5-21B-A3B-PT"` | — | 模型及其 tokenizer 的官方仓库 | [P5] [C3] [C5] |
| `models.M3.total_params_b` | `21.0` | 10^9 parameters | 整模型参数量，仅说明规模；不乘入单层开销 | [P5] [C3] [C5] |
| `models.M3.active_params_b` | `3.0` | 10^9 parameters/token | 整模型激活参数量；各论文统计口径不完全相同 | [P5] [C3] [C5] |
| `models.M3.d_model` | `2560` | element | 单层输入、输出 activation 长度 | [P5] [C3] [C5] |
| `models.M3.d_ff` | `1536` | element | 每个 routed expert 的中间维度 | [P5] [C3] [C5] |
| `models.M3.routed_experts` | `64` | expert/layer | 本层独立 routed expert 数量 | [P5] [C3] [C5] |
| `models.M3.top_k` | `6` | expert/token | 每 token 选择的不重复 routed expert 数量 | [P5] [C3] [C5] |
| `models.M3.shared_experts` | `2` | expert/layer | 每 token 必须计算的 shared expert 数；不占 top-k | [P5] [C3] [C5] |
| `models.M3.shared_d_ff` | `1536` | element | 每个逻辑 shared expert 的中间维度；M3 官方融合宽度为 3072 | [P5] [C3] [C5] |
| `models.M3.transformer_layers` | `28` | layer | 整模型层数，仅用于标识模型 | [P5] [C3] [C5] |
| `models.M3.moe_layers` | `27` | layer | 整模型 MoE 层数；本任务只计其中一层 | [P5] [C3] [C5] |
| `models.M3.layer_index` | `1` | zero-based | 统一选第 2 个 Transformer block 的 MoE；M3 第 1 个 block 是 dense | D；结构依据 [P5] [C3] [C5] |
| `models.M3.expert_linear_bias` | `false` | bool | M1 的 gate/up/down 含 bias；M2/M3 不含 | [P5] [C3] [C5] |
| `models.M3.router_linear_bias` | `false` | bool | M1 router 线性映射有 bias | [P5] [C3] [C5] |
| `models.M3.router_correction_fp32` | `true` | bool | M3 的专家选择 correction bias；合成 trace 设零，真 trace 保留 | [C5]；默认值 D |
| `models.M3.activation` | `"silu_glu"` | — | 专家非线性；禁止把三种模型都改成同一激活 | [P5] [C3] [C5] |
| `models.M3.expert_matrix_params` | `11796480` | parameter | 3*d_model*d_ff，三个矩阵的元素总数 | 派生：[P5] [C3] [C5] |
| `models.M3.expert_params_including_bias` | `11796480` | parameter | 单个专家参数量，M1 加上三个线性 bias | 派生：[P5] [C3] [C5] |
| `models.M3.expert_storage_bytes` | `11799360` | byte | FP8 矩阵/bias 加所有 FP32 block scales | 派生：模型尺寸 + [P6] 的分块规则 |
| `models.M3.router_storage_bytes` | `164176` | byte/die | 完整 router 副本及其 scale/bias/correction 的容量 | 派生：模型配置与本规范量化格式 |
| `models.M3.resident_bytes_per_die` | `63355280` | byte/die | 专家 + router + 固定 4MiB 工作区；必须不超过 64,000,000 | 派生：本规范映射与容量核算 |
| `precision.activation` | `"FP8_E4M3FN"` | — | 所有矩阵输入、层输入/输出、跨 Die activation/partial 的载荷类型 | U；具体格式参考 [P6] |
| `precision.weight` | `"FP8_E4M3FN"` | — | 专家与 router 权重、线性 bias 按 FP8 存储；M1 不沿用原 MXFP4 | U；格式 [P6]；M1 原格式 [P3] |
| `precision.accumulator` | `"FP32"` | — | MAC 累加、softmax、路由概率、Combine 累加的内部精度 | D，参考 [P6] 的高精度累加 |
| `precision.weight_block` | `[128,128]` | element | FP8 权重每个二维 block 共用一个 FP32 scale | [P6] §3.3.2；对本架构的采用为 D |
| `precision.activation_block` | `128` | element | activation 每 token 每 128 元素一个 scale | [P6] §3.3.2 |
| `precision.scale_bytes` | `4` | byte | 每个 scale 的 FP32 字节数 | D |
| `precision.fp8_max` | `448` | — | E4M3FN 最大有限值；amax/448 决定 scale | E4M3FN 格式定义，采用 [P6] |
| `precision.scale_zero_amax` | `1.0` | — | 全零块的 scale，避免除零 | D |
| `precision.rounding` | `"nearest_even_saturate"` | — | 最近偶数舍入、有限范围饱和；不得用 INT8 代替 FP8 | D |
| `hardware.vdies` | `16` | die | Expert V-Die 数量 | U |
| `hardware.base_dies` | `1` | die | Base Die 数量；只作为本层输入/输出端点 | U/按系统描述 |
| `hardware.clock_hz` | `200000000` | Hz | 主数字时钟；一个周期 5ns，不是 THz 载频 | U |
| `hardware.peak_ops_s` | `1000000000000` | OP/s/die | FP8 峰值 1TOPS；乘法和加法各计 1 OP | U；OP 口径 [P2] Table II |
| `hardware.ops_per_mac` | `2` | OP/MAC | 一次乘加计两个操作，防止 2 倍算力错误 | [P2] Table II 脚注 |
| `hardware.edram_bytes` | `64000000` | byte/die | 64MB 按十进制解释，比 64MiB 更保守；所有驻留内容合计受限 | U；单位约定 D |
| `hardware.workspace_bytes` | `4194304` | byte/die | 4MiB 工作区从上述 64MB 内扣除，不额外赠送容量 | D |
| `hardware.buffer_partition_bytes` | `[524288,1048576,1048576,1048576,524288]` | byte | 依次为权重 tile 双缓冲、RX、TX、activation/FP32 partial、控制描述符 | D |
| `hardware.tx_return_reserve_bytes` | `524288` | byte/die | TX缓冲中至少512KiB保留给Return/Output，避免Dispatch堵塞完成路径 | D |
| `hardware.rx_return_reserve_bytes` | `524288` | byte/die | RX缓冲中至少512KiB保留给Return，防止Dispatch占满接收空间 | D |
| `hardware.partial_owner_reserve_bytes` | `262144` | byte/die | activation/partial分区中256KiB保留给owner最终累加器，余下用于专家中间值 | D |
| `hardware.owner_inflight_limit` | `16` | token/die | 每 owner 最多保留 16 个未完成 token，超限在 Base 排队 | D |
| `hardware.vector_lanes` | `64` | element-OP/cycle | Top-k、激活函数、scale、Combine 的向量处理宽度 | D |
| `hardware.compute_utilization` | `1.0` | ratio | tile 内理想计算效率；边界用实际尺寸，不凭空填充全 tile | D；强制做敏感性检验 |
| `hardware.compute_setup_cycles` | `2` | cycle/task | 一项矩阵或向量任务的固定启动开销 | D |
| `hardware.compute_voltage_v` | `1.1` | V | 目标计算域工作电压，按 [P2] Fig.21 约 215MHz 点选择以覆盖 200MHz | [P2] Fig.21 图读近似 + D |
| `hardware.edram_voltage_v` | `0.9` | V | eDRAM 采用独立电压域；不是将两个论文工艺宣称为同一芯片 | [P7] Table 1；集成为 A |
| `hardware.temperature_c` | `27` | degC | eDRAM 参数标定温度；不模拟热反馈 | [P7] Table 1 |
| `compute.tile_m` | `16` | token | 专家一次最多合并 16 个已到达 token | D |
| `compute.tile_n` | `128` | element | 输出维度 tile 宽度 | D，匹配 [P6] 分块 |
| `compute.tile_k` | `128` | element | 归约维度 tile 宽度 | D，匹配 [P6] 分块 |
| `compute.batch_wait_cycles` | `20` | cycle | 最老 token 最多等 100ns 聚批；满 16 或上游窗口关闭可提前发射 | D |
| `compute.activation_ops_per_element` | `8` | vector-OP | 普通 SwiGLU 每中间元素的保守等效成本 | A：非线性电路未提供，固定于全部 baseline |
| `compute.gpt_activation_ops_per_element` | `12` | vector-OP | 含限幅与额外乘加的 GPT-oss SwiGLU 成本 | A；函数依据 [C4] |
| `compute.softmax_ops_per_element` | `8` | vector-OP | softmax/归一化的等效标量成本 | A |
| `compute.quantize_ops_per_element` | `3` | vector-OP | amax、缩放、转换的等效成本；scale 决策按 128 元素组 | A |
| `compute.vector_energy_multiplier` | `4.0` | ratio | FP32 向量操作相对 FP8 等效 OP 的能耗倍率 | A；必须随能耗系数一起检验 |
| `compute.energy_scale` | `1.0` | ratio | FP16 原型迁移到 FP8 的额外修正默认不赠送降耗 | A；1 表示不作位宽线性减半 |
| `compute.extra_idle_power_w` | `0.01` | W/die | 计算/控制域额外待机功耗，独立于 eDRAM 与 RF | A；[P2] 未给本目标 Die 待机功耗 |
| `compute.base_io_idle_power_w` | `0.01` | W | 只计 Base 在本层边界内的 I/O 控制待机 | A；不计 Base 的 Attention 等计算 |
| `memory.banks` | `64` | bank/die | eDRAM 逻辑 bank 数 | D |
| `memory.external_bytes_per_bank_cycle` | `16` | byte/cycle | 每 bank 对计算/DMA 可提供 128bit，读写共享额度 | D |
| `memory.bandwidth_scale` | `1.0` | ratio | 有效用户端口吞吐相对64bank×16B/cycle的倍率；固定访问时延不缩放 | D；敏感性取0.5/1/2 |
| `memory.access_latency_cycles` | `1` | cycle | 最小访问时延 5ns；文献 2ns 访问能力向 200MHz 周期取整 | [P7] §3；向目标时钟换算 D |
| `memory.read_pj_bit` | `0.025` | pJ/bit | 小宏 read 能量，25fJ/bit；非 64MB 成品实测 | [P7] §3，电路仿真；容量扩展 A |
| `memory.write_pj_bit` | `0.04` | pJ/bit | 小宏 write 能量，40fJ/bit | [P7] §3，电路仿真；容量扩展 A |
| `memory.refresh_pj_bit` | `0.065` | pJ/bit/refresh | 一次完整 read+write refresh，65fJ/bit | [P7] §3 |
| `memory.refresh_period_ps` | `11000000` | ps | 11us 的标称保持/刷新周期；6us 作为保守敏感性点 | [P7] Table 1；最坏条件见 §3 |
| `memory.leakage_pw_bit` | `1.25` | pW/bit | 按 Table 1 取泄漏，正文另一工作条件为 0.93；不得混用 | [P7] Table 1 |
| `memory.refresh_blocks_user_port` | `false` | bool | 为 refresh 保留内部端口，用户端口不被阻塞；仍全额计刷新能量 | [P7] internal-refresh 模式；大容量集成 A |
| `memory.refresh_scope` | `"entire_capacity"` | — | 每个已启用 Die 的全部 64MB 保持数据；不靠 baseline 改刷新范围 | D，保守配置 |
| `memory.energy_scale` | `1.0` | ratio | 小宏扩展到64MB的read/write/refresh能量修正；不改变单列的bitcell泄漏，主值1为乐观proxy | A；强制检验 1/2/4 |
| `mapping.routed` | `"expert_id_mod_16"` | — | routed expert e 放在 e mod 16；不复制、不切分 | D |
| `mapping.owner` | `"request_ordinal_mod_16"` | — | 一个请求的所有 token 固定 owner=req_ordinal mod 16 | D |
| `mapping.shared` | `"parity_replica_pair"` | — | M3：S0 在偶数 Die、S1 在奇数 Die各8份；token 只用 owner 和 owner xor 1 两份 | D；原 shared 宽度依据 [C3] [C5] |
| `mapping.router_replicas` | `16` | copy | 每 Die 有完整且相同的 router；每 token 仅 owner 执行一次 Gate | D，符合用户数据流 |
| `mapping.warm_weights` | `true` | bool | 单层全部权重预驻留；不计启动装载，不发生层间换权重 | D，任务只比较一层热态输入到输出 |
| `radio.groups_per_vdie` | `2` | group/die | Base 通信组与 peer 通信组各一组 | U |
| `radio.channels_per_group` | `1` | channel/group | 每组按论文1的一条物理通道计，不把双通道 264Gb/s 给每组 | D；[P1] 通道定义 |
| `radio.base_groups` | `1` | group | Base 有一条聚合接入信道；不是16条独立满带宽链路 | D，避免扩张 Base 带宽 |
| `radio.carrier_hz` | `153800000000` | Hz | 实测锁定本振频率 153.8GHz，约 0.154THz | [P1] §V / Fig.21 |
| `radio.reference_bandwidth_hz` | `45000000000` | Hz | 5层/50um 模拟链路约45GHz带宽，仅作频谱约束背景，不拿载频算速率 | [P1] §II-B，EM 仿真 |
| `radio.ptp_raw_bps` | `132000000000` | bit/s/channel | 有离线 EQ 的单通道点对点峰值 | [P1] §V / Fig.25–26 / Table II |
| `radio.broadcast_raw_bps` | `80000000000` | bit/s/channel | 较长路径广播实测80Gb/s；主实验不使用更乐观124Gb/s | [P1] §V / Fig.27(b)，after EQ |
| `radio.rate_scale` | `1.0` | ratio | 16节点覆盖、耦合/干扰与新PHY带来的未知降额；全部方法一起缩放 | A；[P1] 未验证16节点全互连 |
| `radio.half_duplex_per_group` | `true` | bool | 同组同一时刻 TX 或 RX；不同组可并发，不能将同组当全双工 | D，符合 [P1] TX/RX/OFF 切换 |
| `radio.spatial_reuse` | `false` | bool | 同一介质内不重用相同 RU；16节点共享总频谱预算 | D，保守互扰模型 |
| `radio.independent_groups` | `true` | bool | Base 介质与 peer 介质可并行，二者硬件参数相同 | U；两组足够隔离是 A |
| `radio.rx_pipeline_cycles` | `2` | cycle | 完整包接收后的数字处理延迟 10ns | A |
| `radio.propagation_delay_ps` | `20` | ps | 短距离传播延迟，仅架构占位，非16-Die实测 | A |
| `radio.residual_packet_error` | `0.0` | probability | 主实验假设可靠服务；不从两/四层 BER 外推16节点 BER | A；不做物理链路可靠性结论 |
| `radio.multicast_receivers` | `"destination_mask"` | — | 控制时段所有节点监听；载荷仅目的 Die 启用接收前端 | D；需对 all_non_source 接收方式做敏感性检验 |
| `mac.frame_cycles` | `20` | cycle | TDMA/OFDMA 统一100ns调度帧，全部基线一致 | D |
| `mac.control_cycles` | `4` | cycle/frame | 20ns控制预算下限；降额时按报告/grant字节延长，并从数据窗口扣除 | A |
| `mac.guard_cycles` | `1` | cycle/frame | 5ns 保护/方向切换，从数据窗口扣除 | A |
| `mac.queue_report_bytes` | `8` | byte/node/frame | 每节点每组只报告一个最高优先级HOL候选的mask、bytes、age/type和credit | D |
| `mac.grant_header_bytes` | `8` | byte/frame | 调度grant固定头，包含帧号/状态及CRC | D |
| `mac.grant_bytes_per_ru` | `4` | byte/RU | 每RU发送者/目的mask/方向映射，T模式仍发同长度grant | D |
| `mac.arbitration_cycles` | `1` | cycle/frame | 控制报告结束到grant的数字仲裁预算；作为固定并行硬件假设 | A |
| `mac.peer_arbiter_die` | `0` | die_id | peer介质由Die0仲裁；base介质由Base仲裁，控制通信和固定仲裁延迟显式计费 | D |
| `mac.fft_size` | `256` | bin | OFDMA 服务模型的归一化子载波格点数；不是已有THz芯片参数 | D |
| `mac.data_subcarriers` | `224` | bin | 载荷格点，其余用于 pilot/guard | D |
| `mac.pilot_subcarriers` | `16` | bin | 归一化导频开销 | D |
| `mac.guard_subcarriers` | `16` | bin | 归一化频域保护开销 | D |
| `mac.cp_samples` | `16` | sample | 归一化循环前缀开销；额外效率=224/(256+16) | D |
| `mac.resource_units` | `16` | RU | 每RU含14个数据格点；按帧动态分配，绝不每个RU都给132Gb/s | D |
| `mac.starvation_limit_frames` | `16` | frame | 等待达到16帧的 flow 优先，保证有限队列无饥饿 | D |
| `mac.packet_header_bytes` | `32` | byte/packet | 固定网络头，含token UID/owner/消息类型/目的mask/分片信息 | D |
| `mac.crc_bytes` | `4` | byte/packet | CRC32开销；虽然主实验不注入误码仍计字节 | D |
| `mac.dispatch_entry_bytes` | `8` | byte/expert | expert_id 2B + gate weight FP32 4B + flags 2B | D |
| `mac.mtu_payload_bytes` | `4096` | byte/packet | 最大body，超长分片每片重计header/CRC；不计虚假padding token | D |
| `energy.lo_w_per_node` | `0.032` | W/node | 一个共享VCO覆盖本Die两组；热态测试全程保持锁定 | [P1] §III-B为32mW电路仿真；两组共享采用 D |
| `energy.tx_increment_w_per_group` | `0.00635` | W/active group | (44.7mW−32mW)/2；从L4 TX双通道模块分解 | [P1] §V + 仿真LO功耗，分解值为 A/派生 |
| `energy.rx_increment_w_per_group` | `0.0056` | W/active group | (43.2mW−32mW)/2；从L3 RX双通道模块分解 | [P1] §V + 仿真LO功耗，分解值为 A/派生 |
| `energy.equalizer_extra_w_per_rx` | `0.02` | W/active RX group | 补偿论文1未集成的在线均衡功耗；不能标为实测 | A；另检验0/20/100mW |
| `energy.ofdma_extra_w_per_active_group` | `0.40896428571428567` | W | 完整基带成本压力情景：347mW*(132/112)；不是已测THz OFDMA增量 | [P8] RX参照 + 速率线性外推 A；TX同值为 A |
| `energy.report_modes` | `["frontend_proxy","baseband_stress"]` | — | 必须同时输出：不含未标定EQ/OFDMA基带的能量；加入上述压力proxy的能量 | D，不能把任一值称为实测整机能耗 |
| `energy.compute_pj_op` | `0.06980500749971157` | pJ/OP | 21.4TFLOPS/W @0.9V 按V²外推1.1V；FP8修正默认1 | [P2] Table III + 动态能量V²近似 A |
| `workload.dataset` | `"MohamedRashad/ChatGPT-prompts"` | — | HybriMoE引用的ChatGPT Prompts，不是awesome-chatgpt-prompts | [P9] §VI-A5/参考文献30；[D1] |
| `workload.split` | `"train"` | — | 官方数据划分；本次页面列360行，执行时记录实际版本行数 | [D1] |
| `workload.prompt_column` | `"human_prompt"` | — | Prefill输入文本列 | [D1] |
| `workload.response_column` | `"chatgpt_response"` | — | Decode教师强制重放文本列，不宣称本模型生成 | [D1] |
| `workload.requests_per_seed` | `128` | request | 清洗后无放回采样128个请求；不足时失败，不偷偷复用 | D |
| `workload.prefill_batches` | `[1,8,32]` | request/batch | 原生prompt长度，每批独立释放并排空 | D；区分prefill/decode参考 [P9] |
| `workload.prefill_length_cap` | `1024` | token/request | 仅截断长prompt，不补齐短prompt | D；[P9]也报告至1024 token的prefill工作负载 |
| `workload.decode_batches` | `[1,8,32,128]` | request/batch | 每活跃请求每轮只产生一个MoE输入token | D |
| `workload.decode_steps_cap` | `64` | step/request | 最多重放回复前64个token，短回复自然结束 | D |
| `workload.decode_refill` | `false` | bool | 同一cohort不回填请求，结束时报告真实有效B的变化 | D |
| `workload.arrival` | `"closed_cohort_barrier"` | — | Prefill批内t=0可用；Decode下一轮等本轮全部输出返回再释放 | D，阻止将完整response一次当decode输入 |
| `workload.tokenization` | `"official_plain_text_no_special_tokens"` | — | 每模型用官方tokenizer；编码原字段，不加入聊天模板 | D，固定长度口径 |
| `workload.input_trace_mode` | `"dataset_tokens_synthetic_router"` | — | 真实数据集token/长度 + 受控合成路由；不伪称真实隐藏状态/真实专家命中 | D，CPU默认可完整执行 |
| `workload.routing_profile` | `"uniform"` | — | 主结果：所有专家可交换，无人为热点以保证无偏基线 | D |
| `workload.zipf_alpha` | `1.2` | — | 热点敏感性：先随机置换expert排名，再按r^(-1.2)生成偏置 | D；不是论文测得的特定模型分布 |
| `workload.decode_reuse_probability` | `0.0` | probability | 主结果不强加跨token相关性；敏感性另取0.8 | D |
| `workload.source_revision` | `"resolve_and_freeze"` | — | 首次读取冻结dataset/tokenizer commit及SHA256；后续全部离线复用 | D，禁止随机跟随main变化 |
| `experiment.baseline_ids` | `["T_P_N","T_B_N","T_P_L","T_B_L","O_P_N","O_B_N","O_P_L","O_B_L"]` | — | T=单载波TDMA，O=动态OFDMA，P=逐目的Die点对点，B=广播，N/L=无/有Local Combine | U；采用完整2³消融 D |
| `experiment.strong_tdma_ids` | `["TW_P_N","TW_B_L"]` | — | 无固定帧空尾的包粒度work-conserving TDMA压力对照 | D，防止优势仅来自弱TDMA实现 |
| `experiment.sensitivity_cases` | `["prefill_B32","decode_B32"]` | — | 敏感性与强TDMA用两个代表负载；三模型、五种子全部运行 | D |
| `experiment.sensitivity_baseline_ids` | `["T_P_N","O_P_N","T_B_L","O_B_L"]` | — | 敏感性保留两组可比的T/O配对；完整三因素消融由主实验八组合承担 | D |
| `experiment.rate_scale_values` | `[0.25,0.5,1.0]` | ratio | 16节点覆盖和新PHY的速率降额 | A |
| `experiment.broadcast_rate_values` | `[80000000000,124000000000]` | bit/s | 实测较长/较短广播路径的两个标定点 | [P1] Fig.27 |
| `experiment.compute_utilization_values` | `[0.5,0.75,1.0]` | ratio | 不改变1TOPS峰值的有效计算利用率敏感性 | D |
| `experiment.memory_bandwidth_scale_values` | `[0.5,1.0,2.0]` | ratio | 改变内部bank/DMA有效吞吐，暴露存储瓶颈 | D |
| `experiment.memory_energy_scale_values` | `[1.0,2.0,4.0]` | ratio | eDRAM小宏到大容量的能量外推敏感性 | A |
| `experiment.refresh_period_values_ps` | `[6000000,11000000,100000000]` | ps | 6/11us来自[P7]不同条件；100us来自[P10]另一2T设计，只是替代器件情景 | [P7] [P10]；跨设计采用 A |
| `experiment.compute_energy_scale_values` | `[0.5,1.0,2.0]` | ratio | 检验FP16到FP8、工作电压与电路映射的不确定性 | A |
| `experiment.ofdma_extra_power_values_w` | `[0.0,0.1,0.40896428571428567]` | W/group | 理想下界、中间情景、[P8]速率归一化压力proxy | D/A；高点依据 [P8] |
| `experiment.equalizer_power_values_w` | `[0.0,0.02,0.1]` | W/RX group | 在线EQ未实测，检验共同开销 | A |
| `experiment.frame_cycles_values` | `[10,20,80]` | cycle | 50/100/400ns请求调度粒度；实际帧长由控制报告/grant/仲裁预算和guard共同约束 | D |
| `experiment.multicast_rx_values` | `["destination_mask","all_non_source"]` | — | 只目标接收与全体监听载荷的能量差异 | A |
| `experiment.routing_profiles` | `["uniform","zipf"]` | — | 均匀与热点情景，必须分图报告 | D |
| `experiment.decode_reuse_values` | `[0.0,0.8]` | probability | Decode跨步专家集合相关性的受控敏感性 | D |
| `experiment.idle_power_values_w` | `[0.0,0.01,0.05]` | W/die | 未测待机项敏感性，Base控制项同步采用同值 | A |
| `experiment.bootstrap_resamples` | `1000` | resample | 对配对请求/cohort统计作bootstrap，输出95%区间 | D |
| `experiment.confidence` | `0.95` | probability | 均值/节能率/加速比的置信区间水平 | D |
| `experiment.no_cartesian_sensitivity` | `true` | bool | 每次只改变一项；基线点去重，不跑所有敏感性参数笛卡尔积 | D |


### 论文原型标定值与派生常数（同属参数总表的依据）

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
| FP8等效计算能量 | 0.0698050075pJ/OP | `(1/21.4)*(1.1/0.9)^2`；不是实测FP8能效 | [P2]+A派生 |
| 计算宏面积proxy | 约13.266mm²/Die | `1.53*(1e12/(173e9*200/300))`，只作FP16等效面积参考，不进入延迟/能量公式 | [P2]+A派生 |
| eDRAM 4T bitcell面积 | 0.23µm²/bit | 64MB纯单元几何面积约117.76mm²，尚未计外围和布线 | [P7] Table 1；大容量线性外推A |
| 目标整Die面积 | 不声称有已验证数值 | 本任务不输出未经布局的“实测总PPA”；计算宏proxy与存储单元面积只能分别报告 | 证据边界 |
| eDRAM保持时间保守点 | 6µs | 论文后仿真最坏条件，不与标称11µs混为同一条件 | [P7] §3 |
| eDRAM待机泄漏 | 0.00064W/Die | `64e6*8*1.25e-12`，仅bitcell泄漏；外围另由待机/energy_scale覆盖 | [P7]+A派生 |
| eDRAM刷新功耗proxy | 3.02545W/Die | `64e6*8*65e-15/11e-6`，16Die约48.407W；必须真实保留这个结果，不隐藏刷新以美化能耗 | [P7]+A派生 |
| OFDMA数据效率 | 14/17≈0.823529 | `224/(256+16)`，已经同时包含空载波与CP，只乘一次 | D派生 |
| 每帧控制/数据窗口 | Base 25ns/70ns；peer 20ns/75ns | 按第6.2节控制字节与仲裁公式取整，另有5ns guard；control/guard不是重复收包头 | D派生 |
| 基带压力参考 | 112Gb/s、347mW、5nm | DMT有线接收器的完整接收数据通路，含ADC/DSP等；不能称为28nm THz FFT的实测增量 | [P8] |

## 4. 数据与路由 trace 的唯一生成规则

1. 下载并冻结[D1]的`train`；清除空prompt、空response；按原行序保留首次出现的相同`human_prompt`，报告前后行数。每种子在同一份清洗后的ID集合无放回抽128行；三模型用同一组行ID。用各自官方tokenizer，以`add_special_tokens=False`编码两个字段。不得使用别的数据集替代，不得用字符数冒充分词数。
2. Prefill用prompt实际长度（上限1024）。B=1/8/32表示并发请求数；128条请求划成cohort，最后一组不足B按实际数量。cohort内所有token在Base侧t=0逻辑就绪；Base按请求round-robin选下一个token、保持请求内部顺序，受owner的16-token窗口约束。上一cohort完全排空后再测下一组。不得补齐短prompt并计算padding。
3. Decode用response做**教师强制流量重放**：每个请求每轮只有一个token的本层输入，每个请求最多64步；到其真实回复长度即结束。cohort内下一轮开始时间为上一轮全部活跃请求结果返回时间，无额外全模型计算间隔、无新请求补入。报告每轮真实活跃请求数。这个闭环只表示本层供给约束，所得单层时间不是完整LLM的TBT。
4. 默认路由：以`(seed, model_id, row_id, mode, token_position)`经SHA256产生独立PRNG seed。对每个expert产生`U_e∈(0,1)`，令 `z_e=log(p_e)-log(-log(U_e))`；均匀情景`p_e=1/E`，热点情景`p_e∝rank(e)^(-alpha)`，expert排名先以固定种子随机置换，防止热点特意集中到某个Die。取top-k，输出系数为所选`z_e`的softmax。不同baseline绝不重新抽签。
5. Decode相关性敏感性：以配置概率复用该请求上一步的**整个expert集合和权重**；否则按第4项重新抽样。主结果概率为0。M3两个shared分支每token始终执行，与这些抽样无关；correction向量默认零。这个生成器是公开的工作负载假设，不是声称重现三个模型的预训练router。
6. 若提供真实trace：每条包含`row_id, token_uid, mode, step, d_model, expert_ids[k], gate_weights[k], activation_fp8/scales或payload_hash`，同时记录模型revision、layer_index、量化方法与hidden-state采集来源。真trace必须覆盖本次全部token。只替换路由/数值输入，保留计算和通信计费；不得在时间模型里跳过Gate，因为trace已经离线知道答案。

性能仿真传递张量描述符与字节数，不保存整个大矩阵。数值测试另用固定随机FP8小矩阵验证完整数学流程，不能把这个小测试吞吐作为硬件性能。

## 5. 放置、包格式与层内数据流

### 5.1 输入与Dispatch

请求owner固定为请求在本次128条样本中的序号mod16。Base用base组将`x_t`点对点送到owner，三个机制的开关不改变此输入分配。owner对完整E个候选计算Gate并执行Top-k一次。其他Die不得重复执行Gate。

令`R(t,j)`是token t在Die j上应计算的routed experts与必要shared分支的集合。M3的S0/S1固定分配到owner/buddy，其他副本不激活。owner本地集合不经过无线网络。

- P2P Dispatch：对每个不同的远端目的Die发送一次`x_t + 该Die的expert_id/gate_weight列表`。即使该Die有多个选中专家，也只传一份activation，这是强P2P基线应有的正常去重。
- Broadcast Dispatch：向远端目的集合发送一次`x_t + 全部远端expert列表`，以目的mask过滤。广播信道在物理上一次发射；多接收者各自耗能。若只有一个远端Die，退化为普通单播，用点对点速率；若没有远端Die，零无线dispatch。
- Shared无需router系数，系数固定1，但仍传其expert条目并计输入/输出/计算。若buddy已经是routed目的Die，输入只发送一次。

长度为n的FP8向量序列化长度：`S(n)=n+4*ceil(n/128)` byte。权重矩阵scale数是`ceil(N/128)*ceil(K/128)`，不是先将整个矩阵补齐。

每个包是32B固定头+body+4B CRC。Dispatch body为`S(d)+8*本包expert条目数`；返回body为`S(d)+ceil((E+S_shared)/8)`，末项是contributors位图；Base输入/最终输出body仅为`S(d)`。同一包跨调度帧继续传，不重复收包头；只有MTU分片才每片重新收头和CRC。收到完整包并经接收pipeline后才能消费；不假设免费cut-through。

### 5.2 专家计算与Local Combine

每个Die的专家按已到达token形成最大16-token的任务。不同专家、router和Combine使用同一个Die计算资源；无线可以与计算重叠，eDRAM DMA需争用同一内存接口。禁止每个驻留expert都获得额外1TOPS。

每个专家的输出先在FP32累加；乘以其routed gate权重（shared为1）。Local Combine关闭时每个专家单独产生一个返回向量；开启时该Die等待`R(t,j)`里的全部贡献完成，计算

`p(t,j) = sum_{e in R(t,j)} g(t,e) * f_e(x_t)`。

然后向owner发一份partial。只有同一token在同一Die命中至少两个贡献时，Local Combine才减少返回载荷；不得把不同token相加。owner上的本地专家在两个基线都不产生无线消息。owner根据contributors位图确认k个routed及所有shared均收到且无重复后，做最终求和、转换到FP8，并经base组送回Base。

**FP8量化不是结合律。** 对远端贡献，不做Local Combine时每个加权专家输出分别量化；做Local Combine时局部FP32和只量化一次。owner本地贡献在两种方案均直接加入FP32最终累加器；完整层输出统一量化到FP8。因此线上输出不要求bit-exact。验证分两部分：关闭链路量化时证明数学等价；开启FP8时报告小算例误差，并使用参数表阈值检查。NRMSE定义为`||y−y_ref||₂/max(||y_ref||₂,1e−12)`，余弦相似度使用同一展平输出，参考为相同小矩阵的FP32结果。不得以“本地求和等价”为由宣称量化后的真实模型精度完全不变。

### 5.3 计算、访存与缓冲区时序

采用`M≤16,N≤128,K≤128`的矩阵tile，循环顺序固定为`output-N → reduction-K`，同一输出tile的FP32部分和保留到K归约结束。gate/up/down三次矩阵运算均显式存在。权重在eDRAM中长期驻留，但当前计算tile仍需本地读取；不得把“无片外搬运”等同于“无权重读取”。

- `T_gemm_tile = ceil(2*M*N*K/(5000*utilization)) * 5ns`，任务启动成本每个矩阵调用计一次，不能每个MAC计一次。
- 每个本地DMA burst按实际字节、scale、bias以及`204.8GB/s * memory.bandwidth_scale`服务；一次burst时间为`5ns + ceil(bytes/(1024*memory.bandwidth_scale))*5ns`。DMA粒度按tile打包，禁止每个元素收一次固定开销。
- 使用两个有容量限制的tile缓冲，允许下一tile DMA与当前tile计算重叠；资源被网络RX/TX缓冲读写占用时相应推迟。不得在所有资源彼此冲突时仍简单取`max(total_compute,total_DMA)`。
- 在每次专家聚批调用中，每个权重tile读一次并被该M个token复用；跨专家、跨调用不假定额外常驻权重缓存。MAC内部寄存器/本地计算开销已包含在计算能量proxy，外部eDRAM读取单独计费。
- 普通SwiGLU用`SILU(gate)*up`；GPT-oss按[C4]实现`gate*sigmoid(1.702*gate)*(up+1)`及7.0限幅。M1三个线性bias逐元素相加。gate/up矩阵结果在FP32内进行非线性，得到的中间activation先量化到FP8再输入down矩阵；相关scale、读写和量化操作均计费。
- Gate矩阵乘法`2*d*E OP/token`；Top-k采用确定性k轮扫描，计`k*(E-1)`次比较；概率归一化按softmax等效向量操作数计。真实trace不减少这项计费。
- 向量任务每周期64个等效vector-OP；线性bias、gate系数乘法、最终/局部加法按实际元素个数计。量化处理按每元素3个vector-OP计，另对每128元素组计scale操作。矩阵与向量任务不重叠，合计峰值不超过给定Die计算资源。
- 在每个矩阵tile、向量任务完成时才进行计算资源调度；优先可完成token的Combine，其次已到达owner输入的Gate，再按最老token选专家任务。相同优先级按token_uid/expert_id排序。所有baseline使用相同策略。
- RX/TX/partial空间必须先预留再发grant；队列满时反压而不是丢token。TX中的512KiB只供Return/Output使用，其余512KiB可用于Dispatch；RX同样为Return预留512KiB。partial分区为owner累加器保留256KiB，返回贡献到达即可累加并释放，不积攒k份完整输出；剩余768KiB供专家中间值，按tile回收。不能以Dispatch占满完成路径而形成环形等待。所有预留都在已有4MiB分区内，不额外增加容量；队列账本须给出峰值字节数并检查上限。

## 6. THz介质与OFDMA调度的可实现服务模型

### 6.1 固定资源预算

两种介质分别为`base_medium`（Base与16Die）与`peer_medium`（16Die互连）。每个介质只有**一份**完整频谱，独立仲裁；两者参数相同、可并行。同组半双工：一个Die不能一边发射一边接收，但可在同一TX方向把自己的多个不相交RU用于不同包，也可在RX方向接收多个不相交RU。

所有通信均以132Gb/s点对点或80Gb/s多目的广播的实验值为满带宽锚点。OFDMA是在该锚点之上划分资源的**归一化服务模型**，没有假装论文已验证对应的调制阶数、IQ链路、FFT实现或16用户同步。45GHz只记录频谱参考，不额外用Shannon公式创造另一个吞吐。

对分配到n个RU、持续dt的数据时间：

`delivered_bits = R_packet * rate_scale * (n/16) * (224/272) * dt`。

TDMA对应整带宽单flow，省去OFDMA效率因子。各方案先扣相同control/guard；按实际发送bit累积余数，跨帧保留不足1byte的服务量。广播相同bit只能算一次占用空口，但每个目的端都获得该bit。不得把同一RU授予两个独立TX；不得把广播的一对多能力记成多份物理带宽。

### 6.2 动态grant算法（全部行为确定）

请求的帧长默认100ns，控制预算下限20ns、guard5ns。Base和Die0分别是base/peer介质的仲裁器。实际控制长度为`max(control_cycles*Tclk, ceil((8*Nnodes*queue_report_bytes/(R_ptp*rate_scale) + 8*(grant_header_bytes+resource_units*grant_bytes_per_ru)/(R_broadcast*rate_scale) + arbitration_cycles*Tclk)/Tclk)*Tclk)`，其中`Tclk=5ns`，Nnodes对Base/peer介质分别为17/16。仲裁器也保留一个等长报告时隙，避免偷偷赠送不对称的控制带宽。默认Base控制25ns、数据70ns；peer控制20ns、数据75ns。实际帧长为`max(请求帧长, 实际控制长度+guard+Tclk)`。所有T/O/TW采用同一控制公式。没有任何待发包时跳到下一就绪事件对应的帧边界，不逐空帧推进。所有计算/DMA启动时刻向上对齐5ns主时钟；无线完成事件使用整数ps，进入计算域时再对齐，避免把200MHz错误当作串行链路速率。

1. 收集`ready_time≤frame_start`且目的缓冲有credit的flow；flow键为`(src,destination_mask,message_type)`。包按`ready_time,token_uid,fragment_id`排队。每源每组只把最高优先级的一个HOL候选报告给调度器；需求报告每节点8B，grant为8B+16×4B，不假设固定控制窗口可以传无限队列状态。控制候选不存在时也发送8B空/credit报告；接收credit来自保守预留与上次报告，不能预知未来腾空的缓冲。记录实际帧长和控制利用率，不能忽略控制拥塞。
2. 对flow排序：已等待至少16帧优先；随后按最老包ready_time、message_type（RETURN/OUTPUT优先于DISPATCH/INPUT）、src、mask升序。TDMA/OFDMA使用同一排序，不向某基线提供未来路由或完成时间信息。
3. T：取首个可行flow独占全部频谱一个实际数据窗口，完成包后可继续同flow后续包；该flow本帧尾部没有数据时记录空尾。绝不为不存在的源轮询空时隙。
4. O第一遍：按排序逐候选遍历，方向相容时先授予1个RU，最多接纳16个flow。已标TX的端点不能同时成为接收者，已标RX的端点不能成为发射者；多目的广播要求所有目的均相容。不要让队首大包在第一遍吃掉全部RU，导致OFDMA退化为加了开销的TDMA。
5. O第二遍：把剩余RU逐个分给已接纳且尚有未覆盖需求的flow，选择`剩余待发bits/当前已分配RU`最大的flow，平局按第2项排序；每增加一个RU重算需求，最多授予其本帧可以用掉的RU数。无法再利用则记为空闲。每帧重新分配，不把频带永久绑定expert/Die。发送可以分片跨帧；广播的全部目的同一时刻拿到该分片。

这是明确的动态调度设计，而不是论文1的实验控制算法。可进一步优化算法，但任何修改必须成为另一个具名策略，保留本文默认策略结果，不能用运行结果反向调参。

### 6.3 必跑的强TDMA反证检查

TW为包粒度work-conserving单载波TDMA：沿用相同flow排序、速率、包格式和半双工限制；每次grant收取上述公式的实际控制长度加5ns保护成本，随后按实际包长度发送，完成后立即给下一个flow，不浪费到100ns帧边界。它没有OFDMA导频/CP损失。

在三个模型的Prefill B32、Decode B32上运行TW_P_N、TW_B_L，并分别对比O_P_N、O_B_L。若OFDMA只优于有固定帧空尾的T，却不优于TW，结论必须写明收益来自该调度粒度下的频谱切分/空尾回收，而非“同带宽下OFDMA天然更快”。

## 7. 能耗账本与延迟指标

### 7.1 能量必须逐事件累计并闭合

`E_frontend_proxy = E_compute + E_vector + E_eDRAM_rw + E_refresh + E_mem_leak + E_RF + E_control_idle`。

`E_baseband_stress = E_frontend_proxy + E_online_EQ + E_OFDMA_extra`。

- `E_compute=N_matrix_OP*0.0698050075pJ*compute.energy_scale`，向量操作再乘4。不能把同一矩阵乘法重复算MAC能量和OP能量。
- eDRAM读写逐字节计：`8*read_bytes*0.025pJ + 8*write_bytes*0.040pJ`，再乘`memory.energy_scale`。包含权重/scale读取、activation缓冲、返回/输出缓冲和需要落地的partial。单播/广播多目的接收端的写入分别计费。
- refresh采用各子阵列均匀错相的**持续后台平均**模型，`P_refresh=capacity_bits*refresh_pj_bit*1e−12/(refresh_period_ps*1e−12)*memory.energy_scale`。默认即`capacity_bits*0.065pJ/11us`。整个测量区间积分，不能因一次任务短于11us就令刷新为零。用户端口与内部refresh端口独立，不再从用户带宽重复扣一遍刷新流量；若实现阻塞refresh必须另设实验，不能暗改。bitcell泄漏另按容量×leakage_pw_bit累计，不乘这个动态能量修正。
- 17个节点热态预锁定VCO，整个测量区间每节点32mW；每个活动TX/RX组再加6.35/5.6mW。OFDMA同一组即使占多个RU只算一次前端活动功率；只占一个RU也不能把该组固定功率除以16。控制报告发给仲裁器，grant由仲裁器广播；控制窗口按保守的前端全程唤醒策略计费：每个介质始终1组TX、其余组RX，包含仲裁和取整余量。Base介质为1×TX+16×RX，peer介质为1×TX+15×RX。这个控制窗口功耗约定对T/O/TW相同，LO全局只计一次。
- 广播n个目的时，TX一次，RX n次；`all_non_source`情景中所有非源Die都计载荷RX活动。数据窗口的RF占用按实际有信号的时间积分，空尾只保留LO，不收虚假的载荷前端功率；控制窗口按上一项的唤醒约定，guard只计LO与待机。控制始终用单载波，不把它重复计入OFDMA额外基带活动时间。
- `E_online_EQ=20mW*sum(active_RX_group_seconds)`，只在baseband_stress账本中启用。它是工程占位，不是[P1]实测。
- `E_OFDMA_extra=P_extra*sum(active_OFDMA_TX_or_RX_group_seconds)`；固定吞吐基带不能随RU数线性降功耗。压力默认P_extra≈408.964mW来自[P8]整条DMT RX的速率归一化。它包括ADC/DSP等，**不是纯FFT功耗，更不是目标工艺的准确增量或严格上下界**；TX按同值只是对称成本假设。与0和100mW情景一起报告，求出OFDMA仍可节能的最大P_extra阈值。
- 加入每Die10mW和Base I/O10mW的未标定控制/待机项。VCO、eDRAM泄漏和refresh已单列，不得再包含在此项中。

论文1的0.33pJ/bit只用来对齐“双通道双端点无额外基带”微测试，不作为所有单播/广播/OFDMA的统一能量单价。两个能量账本都依赖部分外推；报告使用“模型估计能耗”，不能使用“目标芯片实测总能耗”。

### 7.2 输出指标与比较口径

| 指标 | 精确定义 |
|---|---|
| 单token层延迟 | Base端输入就绪到Base收到该token层输出；包括Base注入排队 |
| Prefill请求层延迟 | 一个prompt全部输入在Base就绪到其最后一个层输出返回 |
| Prefill cohort完成时间 | cohort首次输入就绪到全部层输出返回 |
| Decode单轮层延迟 | 本轮所有活跃请求输入就绪到其最后一个输出返回 |
| Decode每请求平均层耗时 | 对该请求实际参与轮数取平均；不称整模型TBT |
| 能耗 | 两种账本的J/cohort、µJ/token，以及计算/内存/刷新/无线/基带分项 |
| 加速比 | 同模型、同样本/路由/种子/工况下T_baseline/T_variant |
| 节能率 | 1−E_variant/E_baseline；可以为负 |
| 诊断量 | Dispatch/Return空口bytes、实际接收bytes、RU利用率、空尾、队列等待、compute/DMA占用、峰值buffer、每Die负载、每token不同目的Die数 |

至少给出均值、p50、p95与配对95% bootstrap区间。多Die可重叠时间的简单求和不能冒充总延迟分解：同时输出资源busy-time与一条依赖DAG上的关键路径分解，明确两者区别。对全系统同时活跃的token，避免重复分摊公共待机能量；先算cohort总能量，再除以实际完成token数。

## 8. 实验矩阵（一次命令全部执行）

| ID | 介质访问方式 | Dispatch | Local Combine |
|---|---|---|---|
| T_P_N | 单载波TDMA | 逐目的Die点对点 | 关 |
| T_B_N | 单载波TDMA | 广播 | 关 |
| T_P_L | 单载波TDMA | 逐目的Die点对点 | 开 |
| T_B_L | 单载波TDMA | 广播 | 开 |
| O_P_N | 动态OFDMA | 逐目的Die点对点 | 关 |
| O_B_N | 动态OFDMA | 广播 | 关 |
| O_P_L | 动态OFDMA | 逐目的Die点对点 | 开 |
| O_B_L | 动态OFDMA | 广播 | 开 |

主实验：3模型×8组合×(3种Prefill B+4种Decode B)×5种子=**840个运行配置**；每个配置使用128条请求及各自原生长度，能量两账本从同一事件轨迹计算，不重复改变时序。强TDMA额外60配置。敏感性只在指定的两种代表负载、三模型、五种子上，保持其余参数为默认，每个轴各水平单独改变，保留`T_P_N/O_P_N/T_B_L/O_B_L`四组合，去重默认点；不要做笛卡尔积。只改变能量系数的情景可在已保存的同一事件账本上重算，不必重跑时间仿真。

三项机制应在另外两项固定时配对比较；例如TDMA/OFDMA比较`T_P_N↔O_P_N`、`T_B_N↔O_B_N`、`T_P_L↔O_P_L`、`T_B_L↔O_B_L`。广播与Local Combine也各有4对，不只展示T_P_N→O_B_L的总增益。报告交互项，避免把三项百分比直接相加。

能够验证的预期，而非预先指定的结论：

- 多目的Dispatch广播在空间上复用同一activation；但单目的没有收益，80Gb/s广播也比132Gb/s单播慢，必须按实际fanout计算。
- Local Combine只在同一token多专家落同一Die时减少返回向量。均匀路由且不计shared时，预计不同目的Die数为`16*(1−C(E−E/16,k)/C(E,k))`，M1/M2/M3分别约3.806/6.597/5.316；不得默认收益=k倍。M3需另计owner/buddy上的shared贡献重合。
- OFDMA可以在固定帧粒度下同时服务相容flow、回收尾部频谱并改变计算启动时刻，但总带宽固定且多载波有开销。在饱和等长流或TW对照下，收益可能消失。
- 以M2单专家为例，仅理想MAC就约9.44µs，而2,048B原始activation以80Gb/s传输约0.205µs，实际还有权重读取与队列。固定1TOPS/Die时，整层改善很可能小于纯通信改善。不能通过降低baseline带宽、删除计算/刷新、增加OFDMA总频谱等手段制造显著加速。

## 9. 实现契约、验收与输出

建议Python 3.11+，NumPy、PyYAML、pandas、matplotlib、huggingface_hub、tokenizers/transformers与pytest；允许用heapq或SimPy实现事件队列。不需要CUDA；FP8数值小测试可用NumPy编码/解码表实现。固定并记录实际安装版本。事件队列跳过空闲区间，禁止按200MHz逐周期空转；相同tile序列可以压缩为等价服务段，但必须保留背压、网络与DMA竞争。

项目入口必须是`python -m vdie_moe_sim.run_all --spec VDIE_MOE_SIM_SPEC.md --output results`。入口自动提取唯一的`yaml`代码块、验证参数、获取/复用冻结数据、生成trace、完成测试、执行全部主/强基线/敏感性实验、产出图表和报告。执行是一个连续任务，不能在生成计划或跑完小测试后停止。若网络不能读取所需数据/tokenizer，应明确失败并列出缺失项，不能悄悄用随机文本代替ChatGPT Prompts。

最小模块：`config.py, workload.py, routing.py, placement.py, packets.py, network.py, compute.py, memory.py, energy.py, simulator.py, metrics.py, run_all.py`。真实模型尺寸用于事件/容量核算；小尺寸数值测试与大规模性能仿真分开运行，输出不同标识。

必须通过的验收：

1. **容量与算量**：三模型每Die占用与第2节吻合；1TOPS=5000OP/cycle；一token一expert的三投影为6dh OP；所有原生expert及shared均准确计数。没有缓存超配，没有额外算力。
2. **数学与类型**：GPT-oss特殊SwiGLU/bias正确；M3融合shared沿h拆分后等价；无链路量化时8组合结果在FP32容限内一致；FP8编码确为E4M3FN，小算例误差有记录。数值测试构造正态activation、标准差1/sqrt(d)的权重，避免用全零或全1输入蒙混通过。
3. **token守恒**：每token恰好k个不重复routed贡献，M3再有两个shared贡献；每个贡献恰好使用一次；无丢包、无重复Combine、输出token_uid顺序可恢复。
4. **通信字节**：无远端目的为零dispatch；一个目的时广播等于单播；同Die多个专家P2P也只传一份输入；r个远端目的广播只发一次载荷，接收r次；Local Combine开关的返回向量数与contributors集合一致。
5. **资源守恒**：每介质每帧RU占用≤16；同组无同时TX/RX；两组重叠合法；不能同RU复用独立发送；Base聚合吞吐不超过一信道；网络、计算、DMA及buffer资源不存在超额或负队列。
6. **边界条件**：B=1、空专家、同一Die命中多个专家、全部token集中单专家、仅本地命中、输入短于tile、共享与routed同Die、RX/TX满、跨多帧包、MTU分片均有确定性测试，能排空且无死锁。
7. **能量闭合**：各分项之和与总量一致；广播含所有RX；同一LO两组并发只计一次；保持时间模型不让短任务逃避refresh；0.33pJ/bit仅在对应双通道标定场景重现；待机和后台能量随时间单调增加。
8. **因果公平**：同一实验8组合trace hash完全相同；路由与放置不因baseline变化；专家聚批算法固定但允许由不同实际到达时刻自然产生不同M；没有把观察到的运行收益写入调度参数。
9. **统计与完成性**：主实验840配置无缺失，强TDMA60配置无缺失；敏感性计划在manifest完整枚举并逐项标记成功/失败。全部计划完成才标记`complete=true`。不得把未完成项、推导值或空白图表描述成仿真结果。

输出文件至少包括：

- `results/manifest.json`：本规范/YAML哈希、代码版本、依赖、种子、数据集与tokenizer revision和文件哈希、原始/清洗/选中行数、每模型token长度统计、全部实验清单、证据/假设标签。
- `results/config_resolved.yaml`与`results/parameters.csv`：全部实际参数、单位、含义、文献URL及派生公式，保留本表语义。
- `results/traces/*.jsonl.gz`：冻结后的输入/路由描述符及hash；不得依赖未保存的随机状态。
- `results/raw_runs.csv`、`results/token_metrics.parquet`、`results/paired_effects.csv`、`results/energy_breakdown.csv`、`results/validation.json`。
- `results/figures/`：各模型Prefill/Decode绝对延迟、两种能量账本、8组合归一化对比、三项机制配对效果、计算/通信/内存关键路径、OFDMA额外功耗盈亏阈值和敏感性曲线。使用matplotlib，另存可复现CSV，禁止只保存截图。
- `results/report.md`：给出假设、来源、运行完成性、结果与限制；通信局部收益和整层收益分开写；无收益/负收益必须展示。把“已测参数”“由数据集重放的流量”“受控路由”“PPA外推”分别标明。
- 完整可运行源码与一条复现命令。不要仅提供伪代码、流程图或再次拆分的仿真计划。

## 10. 唯一机器配置

以下YAML是机器配置；参数表是解释。解析后检查所有派生常数，若二者不一致立即报错，不自行选择有利于某baseline的值。

```yaml
spec:
  version: '1.0'
simulation:
  engine: discrete_event
  time_unit_ps: 1
  seed_list:
  - 20261104
  - 20261105
  - 20261106
  - 20261107
  - 20261108
  fp8_numeric_test_tokens: 16
  numeric_test_d: 64
  numeric_test_h: 32
  fp32_rtol: 1.0e-05
  fp32_atol: 1.0e-06
  fp8_nrmse_limit: 0.05
  fp8_cosine_min: 0.99
models:
  M1:
    repo: openai/gpt-oss-20b
    total_params_b: 20.91
    active_params_b: 3.61
    d_model: 2880
    d_ff: 2880
    routed_experts: 32
    top_k: 4
    shared_experts: 0
    shared_d_ff: 0
    transformer_layers: 24
    moe_layers: 24
    layer_index: 1
    expert_linear_bias: true
    router_linear_bias: true
    router_correction_fp32: false
    activation: gpt_oss_swiglu
    swiglu_alpha: 1.702
    swiglu_limit: 7.0
    expert_matrix_params: 24883200
    expert_params_including_bias: 24891840
    expert_storage_bytes: 24898464
    router_storage_bytes: 92288
    resident_bytes_per_die: 54083520
  M2:
    repo: Qwen/Qwen3-30B-A3B
    total_params_b: 30.5
    active_params_b: 3.3
    d_model: 2048
    d_ff: 768
    routed_experts: 128
    top_k: 8
    shared_experts: 0
    shared_d_ff: 0
    transformer_layers: 48
    moe_layers: 48
    layer_index: 1
    expert_linear_bias: false
    router_linear_bias: false
    router_correction_fp32: false
    activation: silu_glu
    expert_matrix_params: 4718592
    expert_params_including_bias: 4718592
    expert_storage_bytes: 4719744
    router_storage_bytes: 262208
    resident_bytes_per_die: 42214464
  M3:
    repo: baidu/ERNIE-4.5-21B-A3B-PT
    total_params_b: 21.0
    active_params_b: 3.0
    d_model: 2560
    d_ff: 1536
    routed_experts: 64
    top_k: 6
    shared_experts: 2
    shared_d_ff: 1536
    transformer_layers: 28
    moe_layers: 27
    layer_index: 1
    expert_linear_bias: false
    router_linear_bias: false
    router_correction_fp32: true
    activation: silu_glu
    expert_matrix_params: 11796480
    expert_params_including_bias: 11796480
    expert_storage_bytes: 11799360
    router_storage_bytes: 164176
    resident_bytes_per_die: 63355280
precision:
  activation: FP8_E4M3FN
  weight: FP8_E4M3FN
  accumulator: FP32
  weight_block:
  - 128
  - 128
  activation_block: 128
  scale_bytes: 4
  fp8_max: 448
  scale_zero_amax: 1.0
  rounding: nearest_even_saturate
hardware:
  vdies: 16
  base_dies: 1
  clock_hz: 200000000
  peak_ops_s: 1000000000000
  ops_per_mac: 2
  edram_bytes: 64000000
  workspace_bytes: 4194304
  buffer_partition_bytes:
  - 524288
  - 1048576
  - 1048576
  - 1048576
  - 524288
  tx_return_reserve_bytes: 524288
  rx_return_reserve_bytes: 524288
  partial_owner_reserve_bytes: 262144
  owner_inflight_limit: 16
  vector_lanes: 64
  compute_utilization: 1.0
  compute_setup_cycles: 2
  compute_voltage_v: 1.1
  edram_voltage_v: 0.9
  temperature_c: 27
compute:
  tile_m: 16
  tile_n: 128
  tile_k: 128
  batch_wait_cycles: 20
  activation_ops_per_element: 8
  gpt_activation_ops_per_element: 12
  softmax_ops_per_element: 8
  quantize_ops_per_element: 3
  vector_energy_multiplier: 4.0
  energy_scale: 1.0
  extra_idle_power_w: 0.01
  base_io_idle_power_w: 0.01
memory:
  banks: 64
  external_bytes_per_bank_cycle: 16
  bandwidth_scale: 1.0
  access_latency_cycles: 1
  read_pj_bit: 0.025
  write_pj_bit: 0.04
  refresh_pj_bit: 0.065
  refresh_period_ps: 11000000
  leakage_pw_bit: 1.25
  refresh_blocks_user_port: false
  refresh_scope: entire_capacity
  energy_scale: 1.0
mapping:
  routed: expert_id_mod_16
  owner: request_ordinal_mod_16
  shared: parity_replica_pair
  router_replicas: 16
  warm_weights: true
radio:
  groups_per_vdie: 2
  channels_per_group: 1
  base_groups: 1
  carrier_hz: 153800000000
  reference_bandwidth_hz: 45000000000
  ptp_raw_bps: 132000000000
  broadcast_raw_bps: 80000000000
  rate_scale: 1.0
  half_duplex_per_group: true
  spatial_reuse: false
  independent_groups: true
  rx_pipeline_cycles: 2
  propagation_delay_ps: 20
  residual_packet_error: 0.0
  multicast_receivers: destination_mask
mac:
  frame_cycles: 20
  control_cycles: 4
  guard_cycles: 1
  queue_report_bytes: 8
  grant_header_bytes: 8
  grant_bytes_per_ru: 4
  arbitration_cycles: 1
  peer_arbiter_die: 0
  fft_size: 256
  data_subcarriers: 224
  pilot_subcarriers: 16
  guard_subcarriers: 16
  cp_samples: 16
  resource_units: 16
  starvation_limit_frames: 16
  packet_header_bytes: 32
  crc_bytes: 4
  dispatch_entry_bytes: 8
  mtu_payload_bytes: 4096
energy:
  lo_w_per_node: 0.032
  tx_increment_w_per_group: 0.00635
  rx_increment_w_per_group: 0.0056
  equalizer_extra_w_per_rx: 0.02
  ofdma_extra_w_per_active_group: 0.40896428571428567
  report_modes:
  - frontend_proxy
  - baseband_stress
  compute_pj_op: 0.06980500749971157
workload:
  dataset: MohamedRashad/ChatGPT-prompts
  split: train
  prompt_column: human_prompt
  response_column: chatgpt_response
  requests_per_seed: 128
  prefill_batches:
  - 1
  - 8
  - 32
  prefill_length_cap: 1024
  decode_batches:
  - 1
  - 8
  - 32
  - 128
  decode_steps_cap: 64
  decode_refill: false
  arrival: closed_cohort_barrier
  tokenization: official_plain_text_no_special_tokens
  input_trace_mode: dataset_tokens_synthetic_router
  routing_profile: uniform
  zipf_alpha: 1.2
  decode_reuse_probability: 0.0
  source_revision: resolve_and_freeze
experiment:
  baseline_ids:
  - T_P_N
  - T_B_N
  - T_P_L
  - T_B_L
  - O_P_N
  - O_B_N
  - O_P_L
  - O_B_L
  strong_tdma_ids:
  - TW_P_N
  - TW_B_L
  sensitivity_cases:
  - prefill_B32
  - decode_B32
  sensitivity_baseline_ids:
  - T_P_N
  - O_P_N
  - T_B_L
  - O_B_L
  rate_scale_values:
  - 0.25
  - 0.5
  - 1.0
  broadcast_rate_values:
  - 80000000000
  - 124000000000
  compute_utilization_values:
  - 0.5
  - 0.75
  - 1.0
  memory_bandwidth_scale_values:
  - 0.5
  - 1.0
  - 2.0
  memory_energy_scale_values:
  - 1.0
  - 2.0
  - 4.0
  refresh_period_values_ps:
  - 6000000
  - 11000000
  - 100000000
  compute_energy_scale_values:
  - 0.5
  - 1.0
  - 2.0
  ofdma_extra_power_values_w:
  - 0.0
  - 0.1
  - 0.40896428571428567
  equalizer_power_values_w:
  - 0.0
  - 0.02
  - 0.1
  frame_cycles_values:
  - 10
  - 20
  - 80
  multicast_rx_values:
  - destination_mask
  - all_non_source
  routing_profiles:
  - uniform
  - zipf
  decode_reuse_values:
  - 0.0
  - 0.8
  idle_power_values_w:
  - 0.0
  - 0.01
  - 0.05
  bootstrap_resamples: 1000
  confidence: 0.95
  no_cartesian_sensitivity: true
```

## 11. 论文与官方数据来源

所有补充论文均为2023–2026年范围内。官方配置/代码用于补齐论文未列出的精确矩阵维度，明确区分它们与论文，不捏造“论文表格已给出所有数字”。附件标题与DOI由附件正文核对。

| 编号 | 来源与年份 | 本规范取用内容 |
|---|---|---|
| [P1] | Jiang et al., *THz-TSI: A Terahertz Wireless Through-Silicon Interconnect Module for Advanced 3-D Integration Applications*, IEEE JSSC, 2026，用户“论文1.pdf” | Fig.21、24–28、Table II；双通道/单通道速率、广播、功耗、EQ限制、模块面积及覆盖限制 |
| [P2] | Li et al., *SLAM-CIM: A Visual SLAM Backend Processor With Dynamic-Range-Driven-Skipping Linear-Solving FP-CIM Macros*, IEEE JSSC 59(11), 2024，用户“论文2.pdf” | Fig.16–21、Table I–III；FP16原型工艺/面积/吞吐/能效与测试条件 |
| [P3] | OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, arXiv:2508.10925, 2025 | Table 1、§2.1–2.2；20.91B/3.61B、32专家top-4、无shared、MXFP4原权重 |
| [P4] | Qwen Team, *Qwen3 Technical Report*, arXiv:2505.09388, 2025 | §2、Table 2；30B量级、128专家top-8、无shared、48层 |
| [P5] | Baidu ERNIE Team, *ERNIE 4.5 Technical Report*, 2025-06-29 | Table 2；21B文本模型、28层、64 routed/top-6、2 shared；不把28B多模态参数当21B文本模型 |
| [P6] | DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437, 2024/2025 | §3.3.2；E4M3、activation 1×128/weight 128×128分块、高精度累加；此处为采用其格式，不是更换三个模型 |
| [P7] | Golman, Giterman & Teman, *Multi-Ported GC-eDRAM Bitcell with Dynamic Port Configuration and Refresh Mechanism*, JLPEA 14(1):2, 2024 | 28nm FD-SOI小宏后仿真；read/write/refresh能量、retention、泄漏、独立refresh端口与bitcell面积 |
| [P8] | Lee et al., *A 112-Gb/s Discrete Multitone Wireline Receiver Datapath With Time-Interleaved Time-Based ADC in 5-nm FinFET*, IEEE JSSC, 2025 | 112Gb/s、347mW完整RX数据通路；仅作为新增多载波基带的成本压力参照 |
| [P9] | Zhong et al., *HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference*, arXiv:2504.05897, 2025 | §VI-A5与参考文献30；确认ChatGPT Prompts数据集身份及Prefill/Decode区分，不搬用其GPU性能 |
| [P10] | Kim et al., *RED: Energy Optimization Framework for eDRAM-based PIM with Reconfigurable Voltage Swing and Retention-aware Scheduling*, arXiv:2502.09007, 2025 | 28nm、200MHz、1V的eDRAM-PIM建模背景；100µs保持时间来自另一设计，仅替代器件敏感性 |
| [C1] / [C4] | GPT-oss官方config / 官方PyTorch实现 | d=h=2880、bias、特殊SwiGLU |
| [C2] | Qwen3-30B-A3B官方config | d=2048、专家h=768、router配置 |
| [C3] / [C5] | ERNIE官方config / Transformers 4.55.4实现 | d=2560、专家h=1536、shared融合宽度3072、correction与归一化语义 |
| [D1] | MohamedRashad/ChatGPT-prompts | train、human_prompt、chatgpt_response；执行时冻结具体revision |

[P1]: https://doi.org/10.1109/JSSC.2026.3720291
[P2]: https://doi.org/10.1109/JSSC.2024.3402808
[P3]: https://arxiv.org/abs/2508.10925
[P4]: https://arxiv.org/abs/2505.09388
[P5]: https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf
[P6]: https://arxiv.org/abs/2412.19437
[P7]: https://doi.org/10.3390/jlpea14010002
[P8]: https://doi.org/10.1109/JSSC.2025.3618188
[P9]: https://arxiv.org/abs/2504.05897
[P10]: https://arxiv.org/abs/2502.09007
[C1]: https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
[C2]: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
[C3]: https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/main/config.json
[C4]: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
[C5]: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py
[D1]: https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts
