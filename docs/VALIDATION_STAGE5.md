# 第五阶段验证记录

环境：Linux x86_64、CPython 3.12.14；NumPy 2.3.5、Pandas 2.2.3、Matplotlib 3.10.8；包版本 0.5.0。依赖未增加，editable 安装和 `pip check` 通过。

## 实际测试顺序

- 开始阶段前运行历史 146 项测试，全部通过。
- 先实现纯扫描/统计规则和独立 broadcast/Local Combine 消融，161 项完整回归通过。
- 加入 CSV/NPY 离线审计、批量运行、续跑和统计重载后，修复序列化问题，再运行完整测试。
- 正式多种子扫描发现 CA 平局处的浮点审计差异，新增真实种子 23 回归并修复审计重放方式。
- 最终 **179 项测试通过，0 失败，0 跳过**。包括原有 146 项和第五阶段新增 33 项。

测试文件：[统计函数](../tests/test_stage5_statistics.py)、[计划/独立消融](../tests/test_stage5_plan.py)、[CSV 离线审计](../tests/test_stage5_offline.py)、[扫描/续跑/CLI](../tests/test_stage5_sweep.py)。

## 发现并修复的问题

| 问题 | 修正及验证 |
|---|---|
| CSV 把 `0.0` 读成 `0`，造成 workload 原样哈希误报 | JSON trace 保留规范输入和类型，再逐项核对 CSV 数值；不将 CSV 的类型推断当作输入变化 |
| JSON 数值列被 Pandas 自动转成 int，绘图时再次 JSON 解析失败 | `views.csv` 的 JSON 列显式按字符串读取；CLI、数值坐标、布尔 grid 和 PNG/SVG 重复导出均测试 |
| manifest 的键排序改变扫描轴遍历顺序，离线汇总出现行次序差异 | 计划编译时规范扫描轴顺序；保存/重载、完整重算和无重跑续跑都逐字节一致 |
| `32000000000` 与 `32000000000.0` 形成两个等价参数点 | 参数点身份规范化等价数值，共享基准只运行一次；新增回归，完整计划由 30 个伪含重复点降为 29 个唯一点 |
| 论文维度、消融 off/off、seed=23 在 80 μs 的 CA 平局审计不一致 | 逐段累加与逐次扣减产生一个 ULP 级差异，改变重算 fractional remainder 的平局顺序。独立核对 Q/W/I/C/priority 后，用记录中的原始 priority 重放离散分配；不改仿真算法、不放松 bit/载波守恒。新增在线及 CSV 重载回归，非法 priority 的拒绝测试保留 |

第一次正式扫描在上述 CA 审计差异处自动停止，未输出成功的完整统计结论。修复后从新的目录运行，未将旧源码指纹的结果混入新结果。

## 独立检查与统计手算

- `[1,2,3]` 的均值为 2、样本标准差为 1；单个观测不生成 std 或区间。
- 两种子 C 延迟为 `[10,100]`，另一方案为 `[1,20]`：同种子加速比分别为 `[10,5]`，均值 **7.5**，不能改成均值比 `110/21`。
- 固定统计 seed 的 bootstrap 可重复；`[0,2]` 的示例区间端点为 `[0,2]`，常量样本的区间退化为该常量。区间标签明确是探索性种子均值区间。
- 重复 seed、重复坐标、不完整/重复的点-种子-方案组合、未知轴、无效拓扑/Top-k 和非法 JSON 拒绝；fixed/trace 用多个 seed 的伪重复拒绝。
- 各参数点的五组完整 PHY、compute/batch/Combine 参数、实际 workload 摘要和数值输出均核对一致；只改变 PHY/compute 时实际路由保持一致。
- 独立开关消融核对四种 broadcast/Local Combine 组合的输入/返回流数及最终向量，三种 PHY 使用明确的 Ablation 标签。
- 从 CSV 独立核对历史 20.5 μs 用例：PEPD=`33.5/20.5`，THz U=`18.5/20.5`，容量 296 bits，payload 256 bits；计算利用率按 Die 含空闲时间统计。
- 修改 summary、segment bits、输出 NPY 或 aggregate 后离线审计会拒绝；校验使用公式/数据重算，文件 hash 只是额外完整性检查。
- 故意在第三个方案中断：前两次成功结果保留，续跑只执行剩余三个方案。完成结果续跑不调用仿真器，最终文件完全一致。改变计划/源码/依赖指纹或篡改已有明细不能混合续跑。
- `--dry-run` 不创建输出目录；非空普通目录不会覆盖。PNG 与 SVG 两次生成均比较内容，SVG 移除时间元数据并固定 hash salt。

## 复现命令

```bash
# 在仓库根目录执行
bash scripts/setup_env.sh
bash scripts/reproduce_stage5.sh
```

快速小模型验证：`bash scripts/reproduce_stage5.sh smoke`。独立检查已有扫描：

```bash
.venv/bin/python -m vdie_moe_sim --stage stage5 --audit --out RESULTS_DIRECTORY
```

脚本在 `results/stage5-*/` 保存 `tests.log`、每次扫描的日志/离线审计 JSON、完整明细和 `reproducibility.json`；测试或审计失败立即停止。统计/模型解释与文件结构见 [STAGE5.md](STAGE5.md)。

## 计划规模与实际执行边界

| 扫描 | 唯一点 | 种子 | 方案 | 每遍仿真 |
|---|---:|---:|---:|---:|
| 小模型 smoke | 5 | 5 | 5 | 125 |
| 论文维度独立开关消融 | 4 | 5 | 3 | 60 |
| 论文维度代表性八类参数 | 18 | 5 | 5 | 450 |
| 完整 PDF 单因素取值计划 | 29 | 5 | 5 | 725，仅预览 |

代表性计划使用 d=7168、h=2048、E=256，基准 8 tokens，token 轴为 1/8/32；还覆盖 Dies 8/16/32/64、Top-k 1/2/4/8、Nsc 16/64/256、B 16/32/64 GHz、ηSE 1/2/4、Zipf 0/1.5/2、FP8/BF16。未把未运行的大 token 完整计划或全笛卡尔积算入已验证实验。

## 本次实际产物

原环境最终成功目录为 `results/stage5-TF8NCJ/`，仓库内保留[可移植的验证记录](evidence/stage5/README.md)。三套计划各完整运行两遍，共 **1,270 次仿真**，六次导出后离线审计全部通过。两遍共 **11,486 对文件逐字节一致**，包括 CSV、JSON、NPY、PNG 和 SVG；这项复现结论限于上述已锁定的软件环境。

| 计划 | 每遍仿真 | 相同文件对 | 每遍离线重算的统计行 |
|---|---:|---:|---:|
| smoke | 125 | 2,266 | 350 |
| ablation | 60 | 1,092 | 168 |
| paper_sample | 450 | 8,128 | 1,260 |

历史归档包括 [tests.log](evidence/stage5/tests.log)、[复现摘要](evidence/stage5/reproducibility.summary.json)和六个离线审计 JSON。六个扫描 manifest 的源码 SHA-256 均为 `f1e1a925f4fd9e6209527e574682da006e3c1d5e59f5d3bce6c02f0645c68876`，与当前 Python 包源码一致。原环境未提交的完整明细未随环境切换保留；2026-11-02 使用相同代码、配置、依赖和种子重新生成，21 个已提交 CSV/manifest 与重建结果逐字节一致。重建后的首遍完整明细现已放入[可下载压缩包](evidence/archives/README.md)，来源和解压审计见[发布校验记录](evidence/archives/publication_validation.json)。

实际查看了[带宽扫描图](evidence/stage5/plots/phy_total_bandwidth_hz.png)、[独立消融图](evidence/stage5/plots/grid.png)及[代表性 CA 通信/计算时间线](evidence/stage5/plots/timeline.png)：坐标、图例、单位与未标定/探索性区间标签可读。时间线展示真实占用的子载波及计算、Combine 区间。

以下为代表性计划基准点 `p_355341e8153208d6`（8 tokens、16 Dies、Top-8、64 子载波、32 GHz、ηSE=2、Zipf=1.5、BF16）的实测模型结果，取路由 seeds 11/23/37/51/71：

| 方案 | 层延迟均值（μs） | 样本 std（μs） | 相对 C 的同种子加速比均值 |
|---|---:|---:|---:|
| A | 2053.465 | 101.388 | 0.565414 |
| B | 2033.509 | 98.477 | 0.571233 |
| C | 1161.563 | 95.477 | 1.000000 |
| D | 968.243 | 72.789 | 1.200019 |
| Proposed | 968.243 | 72.094 | 1.199916 |

数字直接来自 [aggregate.csv](evidence/stage5/paper_sample/aggregate.csv)，加速比先按同 seed 配对再求均值。D 与 Proposed 的延迟均值接近，不能据此声称 CA 优于静态 OFDMA。仅把带宽改为 64 GHz 时，C 的平均层延迟为 931.434 μs，D/Proposed 均为 966.899 μs，配对加速比均为 0.967236；实现允许 OFDMA 比 TDMA 慢。以上仅是参考模型的观测，不是硬件性能预测或统计显著性结论。

## 限制

5 个种子和 percentile bootstrap 只描述本模型下有限的路由变化；没有显著性检验、多重比较校正或硬件不确定性区间。扫描曲线必须允许 CA 与静态近似相同、变慢或加速比小于 1。

硬件、Base/Gate、FIFO Combine、理想多流收发、免费子载波切换和固定净速率等仍未校准；数值为合成 float64，BF16/FP8 仅决定字节数。真实 FFN、量化误差、shared expert、SNR/功率和真实无线错误/重传尚未验证。大规模完整计划的内存/磁盘耗用和其他 Python/平台的逐字节复现尚未验证。
