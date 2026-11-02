# 第五阶段结果、图表和验证证据

历史验收来自 `results/stage5-TF8NCJ/`；2026-11-02 在恢复后的环境按相同代码、锁定依赖、配置和种子重建。历史测试日志仍为 `tests.log`；本次日志为 [rebuild_tests.log](rebuild_tests.log)，完整重复性清单为 [rebuild_reproducibility.json](rebuild_reproducibility.json)。原始指标、配对/统计 CSV 和扫描 manifest 共 21 个历史文件，与本次重建逐字节一致。

[结论报告](../../RESULTS.md) · [全部阶段](../README.md) · [数据包与解压审计](../archives/README.md) · [发布校验](../archives/publication_validation.json)

## 参数扫描图

每张图包含层延迟、吞吐、p95/p99、THz/compute 利用率、PEPD、BWwaste 和配对加速比。置信带为有限路由种子的探索性统计；各点内五方案使用同一总 THz 带宽预算。

| 扫描 | PNG | SVG | 数据 |
|---|---|---|---|
| token 数 | [PNG](plots/workload_num_tokens.png) | [SVG](plots/workload_num_tokens.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| Die 数 | [PNG](plots/topology_num_dies.png) | [SVG](plots/topology_num_dies.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| Top-k | [PNG](plots/model_top_k.png) | [SVG](plots/model_top_k.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| 子载波数 | [PNG](plots/phy_num_subcarriers.png) | [SVG](plots/phy_num_subcarriers.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| 总 THz 带宽 | [PNG](plots/phy_total_bandwidth_hz.png) | [SVG](plots/phy_total_bandwidth_hz.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| 频谱效率 | [PNG](plots/phy_spectral_efficiency.png) | [SVG](plots/phy_spectral_efficiency.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| Zipf 偏斜 | [PNG](plots/workload_zipf_skew.png) | [SVG](plots/workload_zipf_skew.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| activation 字节精度 | [PNG](plots/model_activation_precision.png) | [SVG](plots/model_activation_precision.svg) | [统计](paper_sample/aggregate.csv) / [坐标](paper_sample/views.csv) |
| 广播 / Local Combine 四组合 | [PNG](plots/grid.png) | [SVG](plots/grid.svg) | [统计](ablation/aggregate.csv) / [坐标](ablation/views.csv) |
| 小模型 token 数 | [PNG](plots/smoke/workload_num_tokens.png) | [SVG](plots/smoke/workload_num_tokens.svg) | [统计](smoke/aggregate.csv) / [坐标](smoke/views.csv) |
| 小模型子载波数 | [PNG](plots/smoke/phy_num_subcarriers.png) | [SVG](plots/smoke/phy_num_subcarriers.svg) | [统计](smoke/aggregate.csv) / [坐标](smoke/views.csv) |

## 通信与计算时间线

时间线取每个计划的第一个参数点、seed=11。paper_sample 的首点为 FP8、8 tokens；ablation 首点为 broadcast off / Local Combine off；smoke 使用小模型参考配置。各图中的子载波和计算/Combine 区间来自实际事件。

| 计划 | 时间线 |
|---|---|
| smoke | [A](timelines/smoke/A.png) · [B](timelines/smoke/B.png) · [C](timelines/smoke/C.png) · [D](timelines/smoke/D.png) · [Proposed](timelines/smoke/Proposed.png) |
| ablation | [C](timelines/ablation/C.png) · [D](timelines/ablation/D.png) · [Proposed](timelines/ablation/Proposed.png) |
| paper_sample | [A](timelines/paper_sample/A.png) · [B](timelines/paper_sample/B.png) · [C](timelines/paper_sample/C.png) · [D](timelines/paper_sample/D.png) · [Proposed](timelines/paper_sample/Proposed.png) |

## 数据及校验

- `smoke/`、`ablation/`、`paper_sample/` 保存逐次指标、同 seed 配对数据、统计汇总、共同预算检查、参数点/图坐标和 manifest。
- `*.audit.json` 为历史六次离线审计记录；本次解压包审计另存于发布校验 JSON。
- `reproducibility.summary.json` 保留历史复现计数和原报告摘要。
- `paper_full.preview.json` 只证明完整 725 次计划通过预检，不能当作已执行实验。
- `checksums.json` 记录本目录数据文件的 SHA-256；`original_path` 为相应运行根目录内的路径。

完整首遍 `runs/` 已放入三个压缩包，包含 CSV/JSON/NPY、检查点和原始图片。第二遍与首遍逐字节相同，保留比较清单，省略重复上传。直接对本目录运行 `--audit` 不成立，应按[下载说明](../archives/README.md)先解压完整包。

在仓库根目录校验可浏览数据文件：

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('docs/evidence/stage5')
records = json.loads((root / 'checksums.json').read_text())
for name, record in records.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != record['sha256']:
        raise SystemExit(f'checksum mismatch: {name}')
print(f'{len(records)} evidence files verified')
PY
```

文件摘要只检测文件变化；仿真正确性另由公式手算、因果/资源/数值测试和完整明细的离线重算支撑。
