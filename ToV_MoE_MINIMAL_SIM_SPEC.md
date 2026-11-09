# ToV-MoE：供 Codex 一次性执行的最小完整仿真规范

版本：1.0；日期：2026-11-08。编码：UTF-8。本文是执行规范，不包含已经测得的性能结果。

> **给 Codex 的任务**：依据本文创建 Python 仿真程序、下载并冻结所需小型数据文件、运行校验、完成全部主实验和规定的敏感性实验，输出原始结果、图表和可用于摘要的结论。不要仅生成计划或代码后停止。无需询问分阶段确认，不预设任何加速倍数。若结果不支持优势，保留结果并明确指出。本文唯一的 JSON 配置块是参数真值来源；正文定义模型语义。不要沿用其他旧版仿真文档的隐含默认值。

## 1. 要回答的问题与结论口径

本实验只模拟**一个 MoE 层**，包括输入传送、Router、专家计算、跨 Die Dispatch/Return 和最终 Combine。目标是回答：

1. THz + OFDM/OFDMA + Broadcast 在相同物理总速率预算下，能否提高有效 D2D 通信吞吐率？
2. 通信改善能否转化为 Prefill/Decode 的单 MoE 层延迟降低？
3. OFDMA、Broadcast、Local Combine 各有多少贡献；优势在哪些 TSV 带宽条件下消失？

摘要中第一处“xxx 倍提升”固定为 **effective D2D communication throughput**，第二处固定为 **single-MoE-layer latency speedup**。额外输出能耗代理值，但不把未知电路成本包装为实测系统能耗。

这两个主指标来自不同实验：吞吐率来自移除专家计算耗时的通信回放；延迟来自包含计算的完整单层仿真。不要把同一次固定工作量的 tokens/s 与其倒数 latency 当作两项独立收益。

### 简化边界

- 使用三个真实模型的结构、真实 ChatGPT Prompts 的 token 长度，以及确定性的受控 Top-k 路由；**不运行完整预训练模型，也不将合成路由称为真实模型路由**。
- 默认只需要 CPU。无需下载专家权重、运行 Attention、生成回答、模拟 NPU，或安装 Ramulator/RTL/电磁/热仿真工具。
- 采用离散事件网络模型与算子级计算/访存模型，不声称周期精确、硅后实测、完整 LLM TTFT/TPOT 或模型准确率。
- 当前不能仅凭这项实验验证散热、制造良率或 16 节点 THz 物理可达性。论文背景中的这些说法需另有文献/物理证据。
- 若以后提供真实路由 trace，可用同一仿真器替换受控路由；不把这一步作为本次默认运行的前置条件。

## 2. 硬件、模型与统一参数

继承最新 CARDS-NMP 版本的硬件：**16 个 V-Die 模块，每模块一层 DRAM Die 与一层 Compute Die 通过 Hybrid Bonding 键合，竖直放置**。通信节点是 16 个，不是 32 个。每模块 4 GiB DRAM、16 PE、每 PE 16×16 MAC、1 GHz。早期 64 MB / 200 MHz / 1 TOPS 配置不混入本实验。

论文参数和当前设计不是同一证据等级：

- **P**：论文报告值；**C**：模型官方配置；
- **D**：本实验设计选择；**A**：尚未实测校准的建模假设。
- FP8 为设计目标；CARDS 原设计的 FP16 MAC 参数迁移到 FP8 是 A，不给 FP8 自动翻倍算力或减半 MAC 能耗。
- THz-TSI 的 132 Gb/s 是离线均衡后的**单通道**实验锚点，264 Gb/s 是两通道之和；广播按较保守的 80 Gb/s/通道取值。不可给每个 Die 各自分配一份完整共享频谱。
- THz-TSI 原型没有实现本文的多用户 OFDMA，也没有验证 16 个双层垂直模块全互连。原文还指出超过 12 层直接耦合存在限制。默认 16 节点直接可达是待验证设计假设，须明确标记。

### 2.1 三个模型

均取 zero-based block 1 的一个 MoE 层，仅仿真该层结构。三模型之间不强行配平算量，只在同模型内比较方案。

| 模型 | d | routed expert 的 h | routed 数 E | Top-k | shared | 专家特点/依据 |
|---|---:|---:|---:|---:|---|---|
| GPT-oss-20B | 2880 | 2880 | 32 | 4 | 0 | 大专家、较少；有专家 bias；[M1][C1] |
| Qwen3-30B-A3B | 2048 | 768 | 128 | 8 | 0 | 小专家、较多；注意 h 不是 dense intermediate_size=6144；[M2][C2] |
| ERNIE-4.5-21B-A3B-PT | 2560 | 1536 | 64 | 6 | 2 个 h=1536 的等价分支 | 较小专家、较多、有 shared；[M3][C3] |

ERNIE 的共享分支按总中间维度 3072 处理。将无 bias 的融合分支沿中间维度拆为两个 h=1536 分支，其实数输出之和与融合分支等价；不要改成两个 h=3072 分支。原生 GPT-oss 检查点使用 MXFP4，本文统一 FP8 是架构比较设置，不是对原生权重格式的描述。

### 2.2 所有可调数值参数

GB/s、Gb/s 使用十进制；GiB、MiB、KiB 使用二进制。时间内核使用整数 ps，所有非整数完成时间向上取整。一个 MAC = 2 operations。

| 参数/配置键 | 默认值 | 含义及来源 |
|---|---|---|
| hardware.dies | 16 | V-Die 模块/通信节点数；D |
| hardware.dram_bytes_per_die | 4294967296 | 每模块 4 GiB；按 [P2] 4 GB/channel 的容量标称作二进制建模；P/A |
| hardware.banks_per_die / pe_per_die | 16 / 16 | bank 与 PE 数；[P2] Table III；P |
| hardware.mac_rows / mac_cols / clock_hz | 16 / 16 / 1000000000 | 每 PE MAC 阵列与主频；[P2]；P |
| 派生峰值算力 | 8.192 TOPS/Die | 2×16×16×16×1 GHz；16 Die 共 131.072 TOPS |
| hardware.hb_pins_per_bank / hb_pin_gbps | 1024 / 0.4 | 模块内部 HB 接口；[P2]；P |
| 派生 HB 峰值 | 819.2 GB/s/Die | 16×1024×0.4 Gb/s÷8，不能写成 51.2 GB/s/Die |
| hardware.compute_efficiency / kernel_memory_efficiency | 0.7 / 0.7 | 算力利用率与 kernel 分配带宽内的有效访存比例；A |
| hardware.dma_bandwidth_fraction | 0.125 | HB 峰值的 1/8 固定留给网络 staging DMA；A |
| 派生 kernel / DMA 带宽 | 501.76 / 102.4 GB/s | kernel=819.2×(1−0.125)×0.7；两者同时使用不超过物理峰值 |
| hardware.vector_elements_per_cycle | 32 | SFE/逐元素操作代理吞吐；A |
| hardware.kernel_startup_ns | 2 | 每个算子任务启动开销；A |
| hardware.staging_bytes_per_die / base_staging_bytes | 8388608 / 8388608 | 每节点 DRAM staging 上限 8 MiB；不是增加 8 MiB SRAM；D |
| hardware.base_dma_GBps | 1024 | Base staging DMA 吞吐代理；不作为有线共享瓶颈；A |
| execution.owner_window | 4 | 每 owner 同时在途的 token 上限；D |
| execution.expert_batch_max / batch_wait_ns | 8 / 200 | 每专家微批上限、最长聚批等待；D |
| precision.weight / activation / accumulation | FP8 E4M3FN / FP8 E4M3FN / FP32 | 权重、线上向量及内部累加精度；D/A |
| precision.weight_block / activation_block / scale_bytes | 128 / 128 / 4 | 权重 128×128 block，向量每 128 元素一个 FP32 scale；D |
| packet.header_bytes / crc_bytes / expert_entry_bytes | 32 / 4 / 8 | 包头、CRC、expert ID+gate 等目录项；D |
| radio.groups_per_die | 2 | Base-facing 与 peer-facing 两个独立频谱域；D |
| radio.unicast_gbps / broadcast_gbps | 132 / 80 | 每域总速率锚点；[P1] 实验段/Fig.25–28；P |
| radio.carrier_GHz | 153.8 | 载波记录，不参与把数字率乘倍的计算；[P1]；P |
| radio.frame_ns / guard_ns / controller_ns | 100 / 5 / 5 | 帧长、保护间隔、仲裁处理代理；D/A |
| radio.report_bytes / grant_base_bytes / grant_bytes_per_ru | 8 / 8 / 4 | 每节点一个 HOL flow 报告、grant 固定项和每 RU grant 项；D |
| radio.rus / fft_bins / data_bins / cp_bins | 16 / 256 / 224 / 16 | RU 数；有效 OFDM 比例 η=224/(256+16)；D |
| radio.endpoint_pipeline_ns / ofdm_extra_endpoint_ns | 5 / 10 | 每次发送端或接收端固定流水延迟；OFDM 每端额外延迟；A |
| radio.rate_scale | 1.0 | 对单播、广播锚点同时缩放；A |
| tsv.normalized_total_gbps | 264 | 等于 THz 两个频谱域的单播速率和；**归一化设计预算，不是 TSV 实测值**；D |
| tsv.high_bw_total_GBps | 896 | [P3] HBM3 实验聚合带宽，用作强有线带宽情景；不是完整 HBM 系统复现；P/A |
| tsv.payload_efficiency / turnaround_ns / base_route_ns | 0.9 / 5 / 5 | 有线有效率、端口换向、Base 存储转发处理；A |
| tsv.endpoint_pipeline_ns | 5 | 每跳发送端、接收端各 5 ns；A |
| workload.dataset / split | MohamedRashad/ChatGPT-prompts / train | 数据集；[D1] |
| workload.requests_per_seed / seeds | 32 / [11,23,37] | 每种子无放回取 32 条请求；D |
| workload.batch_sizes | [1,8] | Prefill 与 Decode 各使用两种并发请求数；D |
| workload.prefill_token_cap / decode_step_cap | 256 / 32 | 每请求输入截断上限、teacher-forced Decode 最大轮数；D |
| workload.routing_profiles | [uniform,hot_shift] | 两种受控路由；不是模型实测路由；D |
| workload.zipf_alpha | 1.2 | hot_shift 的专家排名权重指数；D |
| workload.prefill_hot_epoch_tokens / decode_hot_epoch_steps | 128 / 4 | 按逻辑 token 序号/Decode 轮次改变热专家集；D |
| energy.mac_pJ | 0.982 | 从 [P2] PE MAC 峰值功耗与吞吐推导；FP8 移植为 A |
| energy.hb_pJ_per_bit | 0.88 | [P2] HB I/O 能耗口径；不包含完整 DRAM core；P/A |
| energy.dram_core_pJ_per_bit / scalar_pJ_per_op | 0.5 / 1.0 | DRAM core 动态、逐元素操作的代理；A |
| energy.tsv_pJ_per_bit_hop | 0.2 | TSV 每实际跳、每 bit 的设计代理；不是 [P3] 测量值；A |
| energy.rf_lo_mW_per_node / rf_tx_mW / rf_rx_mW | 32 / 6.35 / 5.6 | [P1] 电路功耗拆分；LO 每节点共用，TX/RX 按活动组；P |
| energy.ofdm_W_per_active_endpoint | 0.1 | 新增 OFDM/EQ 基带功耗代理；必须做 0/0.1/0.4 W 重算；A |
| energy.background_W_per_die / base_W | 0.5 / 0.1 | DRAM/逻辑背景及 Base 控制功耗代理；A |
| experiment.bootstrap_samples / bootstrap_seed | 2000 / 2026 | 几何均值配对 bootstrap 及随机种子；D |
| sensitivity.tsv_budget_multipliers | [0.5,1,2,4,8] | 归一化 TSV 总带宽倍数；其余不变；D |
| sensitivity.thz_rate_scales | [0.5,0.75,1] | 检查理想 THz 锚点下调的影响；D |
| sensitivity.die_counts | [4,8,16] | 相同 N 下配对比较；较小规模不冒充 16 节点物理验证；D |
| sensitivity.memory_efficiencies | [0.4,0.7,1.0] | kernel 访存效率敏感性；保留 DMA 分区；D |

## 3. 最少但充分的对照

所有方法具有相同 NMP、DRAM 容量、精度、专家放置、输入样本、Top-k、微批策略和 admission window。**TSV 对照也执行近存计算和 EP，不在每次计算时人为通过 TSV 重新加载专家权重。**

| ID | 互连 | Dispatch | Local Combine | 目的 |
|---|---|---|---|---|
| TSV_NORM_P2P_LC | 264 Gb/s 聚合 TSV 星形 | P2P | 有 | 测有线 multicast 的收益 |
| **TSV_NORM_MC_LC** | 同上 | Base multicast | 有 | **主基线：有线可用的优化也启用** |
| TSV_HIGH_MC_LC | 896 GB/s 聚合 TSV 带宽情景 | Base multicast | 有 | 强有线对照，必须报告 |
| THZ_TDMA_P2P_LC | 单载波、work-conserving TDMA | P2P | 有 | 不保留空轮询槽的强 TDMA 对照 |
| THZ_TDMA_BCAST_LC | 同上 | Broadcast | 有 | 固定广播，单独检查 OFDMA |
| THZ_OFDMA_P2P_LC | 动态 OFDMA | P2P | 有 | 与 TDMA_P2P 配对 |
| THZ_OFDMA_BCAST_NO_LC | 动态 OFDMA | Broadcast | 无 | Local Combine 消融 |
| **TOV_OFDMA_BCAST_LC** | 动态 OFDMA | Broadcast | 有 | **ToV-MoE 完整方案** |

主基线名称必须保留 NORM：当前没有垂直模块边缘面积、TSV 数量/节距、Base 布线的实物约束，不能声称 264 Gb/s 是 TSV 技术上限或等面积设计。归一化对照检验频谱复用/数据流能力；高带宽对照检验优势适用范围。

### 3.1 有线拓扑的精确定义

- 每 V-Die 一条至 Base 的独立半双工端口；N 条可并行，不限制成单共享串行总线。
- NORM 每端口 raw rate = 264/N Gb/s；HIGH 每端口 raw rate = 8×896/N Gb/s。再乘 0.9 得数据服务速率。
- Base↔owner 是一跳；V-Die a→b 必须 a→Base→b 两跳。Base 采用无额外瓶颈的非阻塞交换，包完整到达后加 base_route_ns 再进入出口队列。
- P2P 按不同目的 Die 分别发送；同 Die 上多个专家共用一次输入。
- MC 的源端只向 Base 发一份输入及完整目录；Base 为不同目的端口并行复制，出口可以删除与该目的无关的专家目录项。每个实际出口传输、DMA 和字节都计费。
- 每端口非抢占、按“已就绪 RETURN/OUTPUT 优先，其后最早 ready_time，再 packet_id”调度；方向改变时加 turnaround_ns。第一次使用无需换向。
- 包长含 header/CRC。串行化时间=bits/服务速率。每跳 TX 与 RX 各加 endpoint_pipeline_ns=5 ns；所有传播影响包含在这项代理中，不另猜传播时间。转发必须在依赖链中；不将每一跳当成零延迟。
- HIGH 只是采用论文带宽的强有线情景，不匹配其完整芯片容量/面积，也不称为复现整套 HBM3。

### 3.2 无线资源与调度

Base-facing 域服务 Base↔owner；peer-facing 域服务专家 Dispatch/Return。每域只有一份共享总频谱，两个域可并行。一个节点在同一域同一时刻只能 TX 或 RX；可在不同 RU 上接收多个源，也可在两个域采用不同方向。无线跨节点无需 Base 转发。所有默认逻辑链路可达且无重传；这是一项明确的理想化假设。

每个有流量的帧先发送控制信息：

C_ns = ceil(8×Nparticipants×report_bytes / Runi + 8×(grant_base_bytes+RUs×grant_bytes_per_ru) / Rbcast + controller_ns)

此处速率换成 bit/ns，并应用 rate_scale。Base 域 Nparticipants=N+1，peer 域 Nparticipants=N。默认 C_Base=21 ns、C_peer=20 ns；数据窗口分别为 74/75 ns。没有流量时可跳至下一个有事件的帧边界，不能每 ps 空转。

- 每源每域每帧只报告一个 HOL flow；flow 键为 (src,destination_mask,message_type)。
- 同源候选顺序：RETURN/OUTPUT 优先、最早 ready_time、flow_id。选择器只看当前已经就绪的数据。
- TDMA：在数据窗口中让一个已报告 flow 占全部频谱。该 flow 空后立即选择下一个已报告且可行的 flow，剩余时间可继续用；帧内不人为浪费空尾。允许节点在不同传输之间改换方向，若新 flow 导致任何参与节点相对其本帧上一次活动发生 TX/RX 反转，则统一收取一次 guard_ns（5 ns），从数据窗口扣除。未传完包留到下一帧。TDMA 不收取 OFDM 的 CP/导频折损。
- OFDMA：按同一候选顺序挑选方向相容的 flows，先各给 1 RU；最多 16 个。其余 RU 逐个给预计排空时间 q_bits/(Rflow×allocated_RU) 最大且仍需要容量的 flow，按 flow_id 打破平局。一个 flow 的“需要 RU 数”是 ceil(q_bits / (Rflow×η×data_window/16))，上限 16。
- OFDMA 每帧固定分配；被选节点在该帧内保持相容的 TX/RX 方向。提前排空的 RU 空闲到帧末，不免费帧内重分配。flow 可在自身 RU 上连续发送本帧已就绪的多个包。
- OFDMA 的数据服务速率 = Rflow×rate_scale×(allocated_RU/16)×224/272。Rflow 为单目的 132 Gb/s，两个及以上目的 80 Gb/s。
- Broadcast 线上只发送一次，所有目的实际接收；不同 token 不能合成一次广播。单目的自动按单播速率。
- 每包端点 TX/RX 各加 5 ns；OFDMA 每端再加 10 ns 基带代理。端点流水属于端点延迟，不占用额外“虚构频谱”，但其忙时计入相应能耗代理。
- 完整就绪顺序为“staging DMA 读完→TX 端点流水→flow 就绪→链路发送→RX 端点流水→staging DMA 写完→下游可用”。端点流水作为可流水化固定延迟，不新增未定义的串行处理瓶颈。
- 服务可跨帧，不能每帧重新加 header/CRC；只有真正新包加包头。本规范最大包小于 4096 B，无需分片。
- 传输在窗口内完成时按实际完成时间产生 RX 事件；下游计算可立即开始，不等整个帧结束。OFDMA 分配的方向/RU 仍保留至帧末；TDMA 按上面的串行复用规则继续调度。
- TDMA 与 OFDMA 使用相同控制开销、可达性、速率锚点和完整接收者计费。不得给 OFDMA 增加总 raw bandwidth。
- 广播 fanout>1 的接收功率按所有目的累加；不能仅按一个 RX 计能耗。

## 4. 工作负载、放置与正确的数据流

### 4.1 小型数据下载与冻结

1. 用 huggingface_hub 读取 [D1] 的 train CSV（官方列为 human_prompt、chatgpt_response）。保存原文件 SHA-256、dataset commit、原始行号；不需要 datasets 大型后端。
2. 丢弃空字段；按完全相同的 human_prompt 去重保留首条。每个种子从剩余记录无放回抽 32 条，三个模型使用相同行号。不足 32 条时失败，不重复凑数。
3. 使用三个仓库的官方 AutoTokenizer，add_special_tokens=False、不套聊天模板。只下载 tokenizer/config 必要文件；禁止下载整模型权重。第一次运行冻结仓库 commit，之后从本地 manifest/cache 重放。
4. 若 tokenizer 需要额外文件，自动安装公开依赖并记录版本。禁止用“字符数÷4”静默替代 token 数。网络失败时使用已校验缓存；缓存不存在则停止该数据实验并报告失败，不输出替代的伪论文结果。
5. 记录 Prefill 长度=min(实际 prompt token 数,256)，Decode 长度=min(实际 response token 数,32)。响应文本只决定 teacher-forced 流的长度，不被称为模型生成结果。
6. 不下载真实权重，不读取 Router 的真实 logits。配置与实际下载 config 的 d/h/E/k/shared 必须一致；若不一致，报告差异，不能悄悄修改模型。

### 4.2 确定性路由

不要使用 Python 内置 hash() 产生随机序列。使用 hashlib SHA-256 派生 64-bit 种子，NumPy PCG64；主清单保存每个 routing trace 的 SHA-256。

派生函数固定为 int.from_bytes(sha256(UTF8(text)).digest()[:8],"little")。每 token 的 text 为 seed|model_id|phase|request_rank|token_index|route；专家排名的 text 为 seed|model_id|perm。各字段用十进制/原 ID 表示，竖线为分隔符。选择专家使用该 token 独立 RNG 的 choice(E,k,replace=False,p=p)，uniform 的 p 为均匀向量，hot_shift 为归一化排名权重。样本抽取使用 PCG64(seed)，不包含 model_id。

- uniform：每 token 从 E 个专家等概率无放回选择 k 个。
- hot_shift：先对 E 个专家做种子固定的随机排列；rank r 的权重 w=(r+1)^(-1.2)。使用加权无放回采样选择 k 个。每个 epoch 将排名循环移位 E/4 位。
- Prefill epoch=floor(本种子内固定请求顺序的累计 token 序号/128)；Decode epoch=floor(step_index/4)。这些是逻辑序号，**不依赖模拟完成时间**。
- gate 权重取选中专家 w 归一化；uniform 使用 1/k。这些是受控权重，不冒充真实 Router 概率。
- 一个 token 必须恰有 k 个不同 routed 专家，shared 分支每个执行一次；任何方法不得丢 token、降低 k、替换冷专家或更改 trace。

可选真实 trace 的导入格式为 JSONL，每行包含 model_id、request_id、phase、step_index、token_index、expert_ids、gate_weights、shared_ids、source_commit；主模式仍是上述可直接运行的受控路由。真实 trace 与合成 trace 结果必须分开命名、汇总。

### 4.3 放置与执行顺序

- routed expert e 永久放在 e mod N 的 DRAM，仿真开始前完成预加载；开始/结束边界不含初始化。
- request owner=该请求在本种子中固定序号 mod N。Prefill 中同请求全部 token 同 owner。
- ERNIE：偶数 Die 各保留 shared-0 副本，奇数 Die 各保留 shared-1 副本。每 token 分别使用 owner 与 owner xor 1 上的那一份；不是执行所有副本。
- 完整路径：Base 输入就绪→owner→Router→远端 Dispatch→各 Die 专家执行→可选 Local Combine→owner 最终 Combine→Base 输出到达。
- owner 本地专家不占 D2D；同一 token、同一远端 Die 上多个专家共用一份输入。shared 目的已包含于 routed 目的时不重复 Dispatch。
- 每 owner 最多 4 个在途 token；只有最终输出到达 Base 后才释放 token admission credit。等待 admission 计入请求层延迟。
- 每 Die 一个 NMP 计算引擎，同一时刻最多执行一个 Router、专家微批或向量合并任务；不同 Die 可并行。
- Router 固定每 token 一个 m=1 的任务；只对专家/共享分支做微批。这样不会由实现者另选 Router 聚批策略。
- 同专家队列达 8 token，或最老 token 等待 200 ns 时可执行；选不超过 8 个最早 token。等待计时从输入真正可用开始，专家输入未到不能凑批。引擎忙期间可继续聚批。
- 计算调度优先就绪 Combine，其次 Router，其次最老专家批；用任务 ID 打破平局。不为每个 expert 分配独立计算单元。
- Local Combine 仅合并同一 token 在同 Die 上的加权完整输出；该 Die 所有应执行贡献齐全后再发送一个 partial。禁止合并不同 token、禁止把矩阵输出列拼接当成求和。
- NO_LC 每个远端 expert 各返一份；LC 每个远端 Die 最多返一份。owner 根据 contributor bitset 检查无遗漏/无重复，再输出。
- 在 DRAM staging 中预留实际占用；跟踪每个 live 输入、FP32 partial、待发包、接收包。完成消费后释放，严禁负值/超限。默认 admission window 应使 8 MiB 足够；若超限属于实现/规格错误，必须报告并修复，不能静默扩大容量或丢弃数据。
- 16 PE 的 weight/output SRAM 和共享输入 SRAM沿用 [P2] 的 32 KiB/PE、16 KiB/PE、128 KiB/Die；只支持流式 tile 和批内权重复用，不缓存完整 expert 跨 batch 复用。

### 4.4 Prefill 与 Decode

- Prefill：同 cohort 的 B 个 prompt 在 t=0 就绪，逐 token admission；完成时间为该 cohort 最后一个有效 token 的输出到 Base。
- Decode：同 cohort 每个仍未结束的请求每轮只产生 1 个 token。下一轮在本轮所有活动请求的单层输出到 Base 后开始；长度不足的请求自然退出，禁止 padding token 参与计算。
- 这是单层闭环 Decode，不是完整模型的自回归延迟。每个 cohort 独立清空队列与时间轴；不将上一 cohort 的完成时间叠加到下一条样本。
- 32 个请求按既定顺序划分 cohort，B=1 或 8；各方案 cohort 完全一致。

## 5. 可直接实现的时间与能耗模型

### 5.1 字节、算量与算子时间

记 Q(v)=v+4×ceil(v/128)，即 FP8 向量及其 scales。

- 单专家权重 W=3dh+12×ceil(d/128)×ceil(h/128)。
- GPT-oss 另加 FP8 biases：2h+d+4×(2ceil(h/128)+ceil(d/128)) B。
- Router 权重 Wrouter=dE+4×ceil(d/128)×ceil(E/128)，GPT-oss 再加 E+4ceil(E/128) B 的 Router bias。
- 总权重、Router 副本及 8 MiB staging 不得超过每 Die 4 GiB。scales/bias 不可漏算。
- 单专家批 m 的 MAC=3dhm；OP=6dhm；Router 批 m 的 MAC=dEm。权重每批流式读取一次，不能给每 token 重复读整批权重，也不能跨批免费命中。
- 专家本地 HB 访问字节=权重 W+m×(Q(d)+4d)；包括读输入和写 FP32 结果。up/gate/down 中间向量采用 SRAM 流式处理，不额外假设整专家可驻留 SRAM。
- Router 本地 HB 字节=Wrouter+m×(Q(d)+4E)。
- Rop=派生峰值算力×compute_efficiency；Bkernel=HB_peak×(1−dma_fraction)×kernel_memory_efficiency；Rscalar=32×clock_hz。

矩阵类任务时间：

Tjob = startup + max(OP/Rop, HB_bytes/Bkernel) + scalar_ops/Rscalar

该 max 是算子内流式重叠的 roofline 代理，不表示先把全部权重读进 SRAM。一个 Die 在整个 Tjob 内占用 NMP 引擎；访存受固定分区约束，不能再为每个专家提供一份 Bkernel。

scalar_ops 统一如下，作为明确的工作量代理：

- Router：m×[k(E−1)+8E]，GPT-oss 再加 mE 的 bias；
- 专家：m×(8h+d)，分别代理非线性/Hadamard 与 gate 加权；GPT-oss 再加 m(2h+d) 的 bias；
- 本地/最终向量累加：每次实际相加 d 次 scalar op；
- FP32→FP8 量化：每输出向量 2d 次 scalar op。LC 只量化合并后的 partial，NO_LC 各 expert 单独量化。

Combine/量化任务使用同一 NMP 引擎；时间=startup+max(本地读写字节/Bkernel,scalar_ops/Rscalar)。其本地读写字节按输入 FP32 partial 的 4d、输出 FP32 的 4d 或 FP8 的 Q(d) 实际记录。别把同一读写同时记入两个任务。

所有网络发送前从 staging 进行一次 DMA 读、接收后一次 DMA 写，按完整包字节计。每 Die 的 DMA 是独立串行资源，带宽 Bdma=HB_peak×dma_fraction；Base 使用 base_dma_GBps。网络和 NMP 可重叠，但 Bdma+Bkernel≤HB_peak。无线/有线都承担相同端点 DMA；Base 中继还承担实际中转 DMA。

### 5.2 线上包定义

- INPUT / OUTPUT：32+Q(d)+4 B。
- DISPATCH：32+Q(d)+8×本包 expert 目录项数+4 B。
- RETURN：32+Q(d)+ceil((E+shared_count)/8)+4 B。
- 目的 Die mask 包含在固定 header 中；expert 目录含 gate。Broadcast 带完整远端目录，P2P/MC 出口可以只带该目的相关目录。
- RETURN 带 contributor bitset。FP32 累加是内部格式，不要把 RETURN 载荷错写成 4d B。
- 网络延迟包括排队、DMA、控制等待、串行化、端点流水和有线中继；不只是 bytes/bandwidth。

### 5.3 能耗：同时给分项与代理总量

所有能量用 J 累加，最终可转 µJ/token。相同活动只能计一次。

E_proxy = E_MAC + E_scalar + E_HB + E_DRAMcore + E_network + E_background

- E_MAC=有效执行 MAC×0.982 pJ；E_scalar=scalar_ops×1 pJ。
- E_HB=本地 kernel/Combine/DMA 实际 HB bits×0.88 pJ；E_DRAMcore=同一组访问 bits×0.5 pJ。两项分别代表接口与阵列代理，不重复加入完整 DRAM 功耗。
- TSV 网络=每实际 hop bits×0.2 pJ；MC 每个出口也计费。
- THz：N+1 个节点 LO 各 32 mW，按完整 cohort 时间计；两组共用 LO。TX/RX 活动组分别 6.35/5.6 mW，按实际发送/接收时长和接收者数计；少 RU 不把固定电路功率简单除以 16。
- 活动时长按 (node,domain,TX/RX) 的时间区间并集计算。一个接收组同时接收不同 RU 的多个 flow 时，只计一次该组功率；两个域分别计。OFDM 基带也按端点/域的并集计时，不按 flow 数重复加同一硬件功率。
- 控制时段保守按各域一个 TX、其余参与者 RX 持续 C_ns 计费；保护间隔仅保留 LO。明确这是控制功率代理。
- 仅 OFDMA 加 0.1 W×活动端点时长的 OFDM/EQ 基带代理；TX 与每个 RX 均计。TDMA 不加 OFDM 基带，但其在线均衡未知成本没有实测校准，报告中必须保留这一边界。
- E_background=(N×0.5+0.1) W×cohort 时间，所有方案相同。不得同时叠加 CARDS 的整芯片峰值功耗，以免重复计费。
- 该能耗模型未包含所有电路细节。表头和论文引用都写 energy proxy；摘要默认不引用它作为已验证节能结论。另将 OFDM 基带系数设为 0/0.1/0.4 W 重算，不重跑时序。

### 5.4 事件实现要求

Python heapq 维护 (time_ps,event_priority,event_id)。同时间先完成/释放，再生成新工作，再仲裁；事件 ID 单调递增。关键事件包括 DMA_DONE、ROUTER_DONE、EXPERT_DONE、COMBINE_DONE、TX_DONE、RX_READY、FRAME_START、BATCH_TIMEOUT、COHORT_DONE。

只处理发生的事件。网络空闲时跳到下一帧边界；不要逐时钟遍历长访存时间。传输未完成前，下游不能提前执行。不要分别计算整层“总通信时间”和“总计算时间”后简单相加或取 max 代替依赖仿真。

## 6. 两种测量、实验规模与统计

### 6.1 完整单层仿真：延迟及能耗

每模型×每 phase×B×routing profile×seed×method 都运行全部 cohort。

主实验共有 3×2×2×2×3×8=**576 个配置**；一个配置内部包括 32/B 个 cohort，Decode cohort 还包含多轮。不是只挑一个最快 cohort。

- Prefill：输出每请求最后 token 的完成时间、cohort makespan、单 token latency 分布。
- Decode：输出每轮单层 latency、每请求各轮 latency、cohort 总完成时间。
- 主延迟统计：Prefill 使用 cohort makespan 的算术均值；Decode 使用所有有效轮次 latency 的算术均值。尾部 p95 另列，不与均值混合成一个指标。
- 同时输出 MAC 数、HB 字节、每 Die kernel/DMA 忙时、网络字节、广播 fanout、微批大小分布、最大 staging 使用、每 token 贡献数、energy proxy 分项。
- kernel/DMA/network 的累计忙时会重叠，不能画成相加必等于端到端延迟的分解柱状图。若无关键路径回溯，只画利用率/累计活动量。

### 6.2 通信回放：有效 D2D 吞吐率

使用完全相同的 token/路由记录，输入视为已经在 owner，Router 已完成；不含 Base 首次输入和最终输出，只保留 Dispatch→Return 的依赖。

专家/Combine 计算设为零，端点 DMA、包长、队列、控制、流水、半双工、owner window 保留。一个目的 Die 收齐该 token 输入后立即产生规定的 Return；LC/NO_LC 的返回数量仍按方法区分。最后一个 Return 到 owner 并完成 DMA 后停止计时。

为所有方法定义同一个有用工作量：

B_useful = Σtoken [2×d×远端目的 Die 数] bytes

这个分子只计算法需要的 FP8 输入与每目的归约结果，不计 header/scales，也不因 NO_LC 发了更多中间包而变大。广播复制到多个实际目标的逻辑载荷分别算“交付”，但物理发送字节只算一次。

G_D2D = 8×B_useful/T_replay

这是 achieved useful-delivery throughput，不是物理线速，也不是宣称广播创造额外频谱。同时输出 physical_tx_bits、每域占用率与 G_D2D。若某 cohort 无远端通信，标记 N/A，不造无穷吞吐率；综合时报告排除数量。

通信回放也有 576 个配置，与完整单层共 **1152 个主运行配置**。两套结果文件分开保存。

### 6.3 配对倍数与摘要数值

主比较为 TOV_OFDMA_BCAST_LC 对 TSV_NORM_MC_LC，按同模型、同 phase、B、profile、seed 配对。

- 吞吐倍数 SG=G_TOV/G_TSV。
- 延迟加速比 ST=T_TSV/T_TOV。
- 延迟减少百分比=100×(1−T_TOV/T_TSV)。
- 能效代理倍数 SE=E_TSV/E_TOV，仅作为辅助、带 proxy 标签。
- 不要求 SG、ST、SE>1，不裁掉负收益或失败样本。

先在每个 model×phase×B×profile 内聚合三个种子的原始观测，计算配对比值；再对预先确定的所有 case 比值取几何均值。Prefill 与 Decode 单独汇总。总汇总每个 model×phase×B×profile 等权，不能让长 prompt 或 Decode 轮数多的配置独占权重。

聚合公式固定：G_case=8×sum(useful_bytes)/sum(replay_time)；T_prefill_case=sum(cohort_makespan)/cohort_count；T_decode_case=sum(round_latency)/round_count；E_case=sum(energy_proxy)/sum(completed_tokens)。先形成双方的 case 值，再计算 SG、ST、SE；不能在同一张汇总表中混用“平均比值”和“均值之比”。

做 2000 次分层配对 bootstrap：先有放回抽取三个 seed，再在各入选 seed 内有放回抽 cohort；同一 phase/B 的模型与方法共享 cohort 索引，同一 cohort 的 Decode 所有轮次作为一个整体。每次重算完整统计流程，给 95% 区间。种子少时如实说明区间只覆盖本受控工作负载的不确定性；禁止把成千 token 当作相互独立模型运行。

摘要建议句式：

“In architectural simulations of the modeled MoE workloads, compared with a TSV-based PNM baseline with the same aggregate raw-link budget, ToV-MoE improves effective D2D communication throughput by SG× and reduces single-MoE-layer latency to 1/ST of the baseline.”

同时提供减少百分比和 Prefill/Decode 分项。不要写语义含混的“latency is reduced by ST×”。若 SG 或 ST≤1，自动改成中性比较，不生成“improves/reduces”夸大句式。即使主比较有收益，也必须在报告邻近位置给出 TSV_HIGH_MC_LC 结果。

### 6.4 必须做的最小敏感性

只使用 Qwen3、B=8、hot_shift、两个 phase、三个 seed；每轴单独变化，不做笛卡尔积：

- N=[4,8,16]；双方同 N，NORM 总带宽仍为 264 Gb/s，端口平分；
- TSV 聚合预算乘 [0.5,1,2,4,8]，仅改变 TSV_NORM_MC_LC；ToV 结果复用；
- THz rate_scale=[0.5,0.75,1]，控制时长随速率重新计算；
- kernel_memory_efficiency=[0.4,0.7,1]，双方同步变化。

只跑主基线与完整 ToV；复用已运行的默认点。输出每个轴的延迟/吞吐曲线，以及 TSV 带宽下收益是否跨过 1 的区间，不在离散点之间声称找到了精确物理阈值。能耗的基带功率敏感性仅重算账本。

## 7. 必须通过的校验

1. **单位**：8.192 TOPS/Die；819.2 GB/s/Die；NORM 总 raw rate=264 Gb/s；HIGH=7168 Gb/s；100 ns 下 C_Base=21、C_peer=20。
2. **容量**：逐 Die 检查权重、scales、bias、Router、shared 副本及 staging；SRAM 不能容纳整专家的事实必须成立。
3. **解析链路例子**：当关闭协议/流水/DMA附加时，一个 4096 B 单播在 132 Gb/s 上的串行化时间为 248.242424… ns；两跳同速 store-and-forward 为其两倍。此处使用虚拟测试包，不代表主包最大值。
4. **资源守恒**：任意域任意时间 RU 合计≤16；不相容 TX/RX 不并发；N 条 TSV 端口可并行；同 Die NMP 任务不重叠；Bkernel+Bdma≤HB_peak。
5. **流量守恒**：P2P 对同目的 Die 去重；broadcast 只一份物理发送但所有目标接收；TSV MC 源一份、出口逐份；RETURN contributors 恰好覆盖 k routed 和所有 shared。
6. **路由公平性**：方法间 input manifest、route trace、placement hash 一致；将这些 hash 写进每条结果。
7. **批处理**：m=1/4/8 的专家字节/MAC 正确；权重每 batch 一次；未到达输入不能提前入批；相同数据全部完成后才结束。
8. **数学小算例**：用 d=32、h=16、16 token 的随机小矩阵，检查 P2P/MC/Broadcast 和 LC/NO_LC 在 FP32 下给出相同加权输出（rtol=1e-5、atol=1e-6）；检查 ERNIE 两 shared 分支之和等价于融合分支。GPT-oss 使用 gate 上限 7、up∈[−7,7]、gate×sigmoid(1.702×gate)×(up+1) 的特定门控；另两模型用标准 SiLU gate。
9. **FP8 小算例**：实现 E4M3FN 有限值量化，block scale=max_abs/448，零 block 的 scale=1，round-to-nearest-even。输出 LC/NO_LC 相对 FP32 的相对 L2 与 cosine；两条量化路径不要求 bit-exact。此测试不是三个真实模型的准确率测试。
10. **负面对照**：一个目的且一个贡献时 broadcast/LC 的物理字节节省应为零；网络无限快时保留非零本地计算时间；提高 kernel 吞吐后应显示真实瓶颈迁移，不能硬编码加速比。
11. **重放**：相同 manifest/config/seed 的 CSV 数值与 trace hash 可重复。浮点服务量允许每服务段≤1e-6 bit 的内部误差，计费总包字节必须为整数、完整守恒。
12. **完成性**：每个预定配置有 completed 或明确 failure_reason；有主实验缺失时不生成可直接用于摘要的总体倍数。不得通过过滤慢 case 让仿真“通过”。

这些测试验证守恒、因果、数值及公平性，不应把“所有优化均快于基线”写成测试断言。

## 8. 实现入口与交付物

由 Codex 在空目录创建以下程序接口；它们不是已经存在的工具：

~~~bash
python -m pip install numpy matplotlib huggingface_hub transformers tokenizers
python -m tov_sim.run_all --spec ToV_MoE_MINIMAL_SIM_SPEC.md --output results
~~~

首次实现后应一次执行 run_all 完成：解析配置→冻结数据→校验→全部主运行→敏感性→图表与报告。允许内部函数分步，不要求用户逐步确认。所有依赖的实际版本写入 requirements.lock；默认不安装 torch、不加载神经网络权重。若官方 tokenizer 需要额外依赖，自动补齐并记录。

建议模块：config.py、workload.py、kernel.py、network.py、simulator.py、metrics.py、validate.py、run_all.py。纯 Python+NumPy 足够；通过事件跳跃、同包服务区间合并优化速度。仿真器的 wall-clock 用时不是芯片 latency。

必须输出：

- config_resolved.json：唯一配置加全部派生量、模型结构、软件版本；
- input_manifest.json：样本原始行号、文件/commit/hash、token 长度、模型 tokenizer 信息；
- routes/*.jsonl 或等价压缩格式：冻结路由，保留解码工具；
- run_manifest.csv：全部计划运行与状态，包含 main/replay/sensitivity 区分；
- layer_results.csv：至少包含 case_id、model、phase、B、profile、seed、cohort、step_index、method、tokens、latency_ns、p95_token_ns、energy_proxy_J、MAC、HB_bytes、physical_tx_bits、peak_buffer_bytes、route_hash。Prefill 每 cohort 一行、step_index=-1；Decode 每轮一行，按轮记录增量能耗/活动量，避免重复累计整 cohort 能量；
- cohort_summary.csv：每 cohort 的总时间、总 token 数和能耗；Decode 的总时间由各轮延迟相加，与最终模拟时钟核对；
- replay_results.csv：同 case 键，另含 useful_bytes、replay_ns、goodput_Gbps、各域物理发送 bit；
- summary.csv：各模型/phase/汇总的 SG、ST、减少百分比、95% CI、样本数、基线全名；
- validation.json：逐项校验结果；
- figures/：单层延迟、D2D goodput、三项机制消融、TSV 带宽敏感性、内存效率敏感性、energy proxy 分项；统一标明归一化基线和代理/假设；
- report.md：方法、结果、反例、计算或通信瓶颈、参数来源与边界；
- abstract_result.txt：只填可支持的结论；包含 prefilling/decoding 分项与等带宽限定；
- source/、requirements.lock、README.md：保证可重跑。

额外规则：失败不可转成 0 延迟；未运行不能填估计结果；不能使用文献报告的 2.3×/2.1× 或任何预设 2×/3× 作为本次输出。不要向用户报告“完成了仿真”而实际上只写了源码。

## 9. 唯一机器配置

下方是本文件唯一 JSON fence；执行程序通过标准 json 模块读取。派生量一律由输入参数计算，禁止表格和代码中各维护一套不同值。

~~~json
{
  "schema_version": "tov-minimal-1.0",
  "hardware": {
    "dies": 16,
    "dram_bytes_per_die": 4294967296,
    "banks_per_die": 16,
    "pe_per_die": 16,
    "mac_rows": 16,
    "mac_cols": 16,
    "clock_hz": 1000000000,
    "hb_pins_per_bank": 1024,
    "hb_pin_gbps": 0.4,
    "compute_efficiency": 0.7,
    "kernel_memory_efficiency": 0.7,
    "dma_bandwidth_fraction": 0.125,
    "vector_elements_per_cycle": 32,
    "kernel_startup_ns": 2,
    "staging_bytes_per_die": 8388608,
    "base_staging_bytes": 8388608,
    "base_dma_GBps": 1024
  },
  "precision": {
    "weight": "fp8_e4m3fn",
    "activation": "fp8_e4m3fn",
    "accumulation": "fp32",
    "weight_block": 128,
    "activation_block": 128,
    "scale_bytes": 4
  },
  "execution": {
    "owner_window": 4,
    "expert_batch_max": 8,
    "batch_wait_ns": 200
  },
  "packet": {
    "header_bytes": 32,
    "crc_bytes": 4,
    "expert_entry_bytes": 8
  },
  "radio": {
    "groups_per_die": 2,
    "unicast_gbps": 132,
    "broadcast_gbps": 80,
    "carrier_GHz": 153.8,
    "frame_ns": 100,
    "guard_ns": 5,
    "controller_ns": 5,
    "report_bytes": 8,
    "grant_base_bytes": 8,
    "grant_bytes_per_ru": 4,
    "rus": 16,
    "fft_bins": 256,
    "data_bins": 224,
    "cp_bins": 16,
    "endpoint_pipeline_ns": 5,
    "ofdm_extra_endpoint_ns": 10,
    "rate_scale": 1.0
  },
  "tsv": {
    "normalized_total_gbps": 264,
    "high_bw_total_GBps": 896,
    "payload_efficiency": 0.9,
    "turnaround_ns": 5,
    "base_route_ns": 5,
    "endpoint_pipeline_ns": 5
  },
  "workload": {
    "dataset": "MohamedRashad/ChatGPT-prompts",
    "split": "train",
    "requests_per_seed": 32,
    "seeds": [11, 23, 37],
    "batch_sizes": [1, 8],
    "phases": ["prefill", "decode"],
    "prefill_token_cap": 256,
    "decode_step_cap": 32,
    "routing_profiles": ["uniform", "hot_shift"],
    "zipf_alpha": 1.2,
    "prefill_hot_epoch_tokens": 128,
    "decode_hot_epoch_steps": 4
  },
  "models": [
    {"id": "gpt_oss_20b", "repo": "openai/gpt-oss-20b", "layer_index": 1, "d": 2880, "h": 2880, "E": 32, "k": 4, "shared": 0, "expert_bias": true, "router_bias": true, "activation_rule": "gpt_oss_swiglu"},
    {"id": "qwen3_30b_a3b", "repo": "Qwen/Qwen3-30B-A3B", "layer_index": 1, "d": 2048, "h": 768, "E": 128, "k": 8, "shared": 0, "expert_bias": false, "router_bias": false, "activation_rule": "silu_gate"},
    {"id": "ernie_21b_a3b", "repo": "baidu/ERNIE-4.5-21B-A3B-PT", "layer_index": 1, "d": 2560, "h": 1536, "E": 64, "k": 6, "shared": 2, "expert_bias": false, "router_bias": false, "activation_rule": "silu_gate"}
  ],
  "methods": [
    "TSV_NORM_P2P_LC",
    "TSV_NORM_MC_LC",
    "TSV_HIGH_MC_LC",
    "THZ_TDMA_P2P_LC",
    "THZ_TDMA_BCAST_LC",
    "THZ_OFDMA_P2P_LC",
    "THZ_OFDMA_BCAST_NO_LC",
    "TOV_OFDMA_BCAST_LC"
  ],
  "primary_baseline": "TSV_NORM_MC_LC",
  "primary_proposal": "TOV_OFDMA_BCAST_LC",
  "energy": {
    "mac_pJ": 0.982,
    "hb_pJ_per_bit": 0.88,
    "dram_core_pJ_per_bit": 0.5,
    "scalar_pJ_per_op": 1.0,
    "tsv_pJ_per_bit_hop": 0.2,
    "rf_lo_mW_per_node": 32,
    "rf_tx_mW": 6.35,
    "rf_rx_mW": 5.6,
    "ofdm_W_per_active_endpoint": 0.1,
    "background_W_per_die": 0.5,
    "base_W": 0.1
  },
  "experiment": {
    "bootstrap_samples": 2000,
    "bootstrap_seed": 2026
  },
  "sensitivity": {
    "model": "qwen3_30b_a3b",
    "B": 8,
    "routing_profile": "hot_shift",
    "tsv_budget_multipliers": [0.5, 1, 2, 4, 8],
    "thz_rate_scales": [0.5, 0.75, 1],
    "die_counts": [4, 8, 16],
    "memory_efficiencies": [0.4, 0.7, 1.0],
    "ofdm_endpoint_power_W": [0, 0.1, 0.4]
  }
}
~~~

## 10. 参数来源与引用链接

这些链接用于核验参数，不要求 Codex 在每次执行时重新检索论文以改变默认值。公开资料无法支持的数值已标 A/D，不能把设计值改写成论文实验值。

- **[P1]** C. Jiang et al., “THz-TSI: A Terahertz Wireless Through-Silicon Interconnect Module for Advanced 3-D Integration Applications,” IEEE JSSC, 2026. [DOI](https://doi.org/10.1109/JSSC.2026.3720291)。依据上传论文的电路功耗、实验段及 Fig.25–28；不把原型已验证为多用户 OFDMA 或 16 模块直接互连。
- **[P2]** Y. Yan et al., “CARDS-MoE: A 3-D Heterogeneous NPU–NMP Architecture with Cost-Aware Expert Routing and Dynamic Attention Scheduler for Edge MoE Inference,” IEEE TVLSI, 2026. [DOI](https://doi.org/10.1109/TVLSI.2026.3743540)。采用 NMP 的 Table III/IV 硬件锚点，不使用其 NPU、CAR、DAS 或完整系统的已发表加速比。原文容量/密度排版存在单位混用，本实验按 Table III 的 4 GB/channel 建模，不按正文误读为 32 GB/channel。
- **[P3]** M.-J. Park et al., “A 192-Gb 12-High 896-GB/s HBM3 DRAM With a TSV Auto-Calibration Scheme and Machine-Learning-Based Layout Optimization,” IEEE JSSC, 58(1):256–269, 2023. [出版社](https://ieeexplore.ieee.org/abstract/document/9858112/)；[DOI](https://doi.org/10.1109/JSSC.2022.3193354)。896 GB/s 为实验聚合带宽，绝非本文 NORM 的 264 Gb/s。
- **[M1]** OpenAI, “gpt-oss-120b & gpt-oss-20b Model Card,” 2025. [论文](https://arxiv.org/abs/2508.10925)。
- **[C1]** [GPT-oss-20B 官方 config](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)；[官方 PyTorch 实现](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)。
- **[M2]** A. Yang et al., “Qwen3 Technical Report,” 2025. [论文](https://arxiv.org/abs/2505.09388)。
- **[C2]** [Qwen3-30B-A3B 官方 config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。
- **[M3]** Baidu-ERNIE-Team, “ERNIE 4.5 Technical Report,” 2025. [官方项目与 Citation 入口](https://github.com/PaddlePaddle/ERNIE#citation)；原报告 PDF 旧链接曾失效，不依赖该旧地址运行。
- **[C3]** [ERNIE-4.5-21B-A3B-PT 官方 config](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/main/config.json)；[官方模型仓库](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT)。
- **[D1]** MohamedRashad, “ChatGPT-prompts,” 2023. [数据集](https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts)。本次检索 train 为 360 条；运行时保存实际 revision、过滤后记录数和选样清单，不把 360 写死为去重后数量。

