# 第五阶段验证证据摘录

来自已完成的本地运行 `results/stage5-TF8NCJ/`，包版本 0.5.0。原始记录为 179 项测试通过；smoke、ablation、paper_sample 各运行两遍，共 1,270 次仿真、11,486 对逐字节一致文件，六次离线审计全部通过。解释与限制见 [验证报告](../../VALIDATION_STAGE5.md)。

## 保留内容

- `tests.log`：原始完整测试日志。
- `*.audit.json`：两遍、三套扫描的离线审计结果。
- `reproducibility.summary.json`：完整复现报告的计数摘要及原报告 SHA-256；省略逐文件比较清单。
- `smoke/`、`ablation/`、`paper_sample/`：第一遍原样复制的原始指标、配对结果、统计汇总、共同预算检查、参数点/图坐标和扫描 manifest。
- `plots/`：带宽扫描、独立开关消融的 PNG/SVG，以及 FP8 参数点、seed=11、Proposed 的通信/计算时间线。
- `paper_full.preview.json`：完整 PDF 单因素计划预览，725 次仿真；该计划尚未实际全部执行。
- `checksums.json`：保留数据文件的 SHA-256 和原始相对路径；`original_path: null` 表示由原报告提取的摘要。

约 741 MB 的完整逐事件、逐流、NPY 输出及两遍所有图表保留在原环境，不纳入 Git。本目录是可携带的历史证据摘录，缺少完整 `runs/`，不能直接用 `--audit --out` 执行全量离线审计。应先在仓库根目录重现：

```bash
bash scripts/setup_env.sh
bash scripts/reproduce_stage5.sh
```

脚本创建新的 `results/stage5-*/` 并自动执行两遍实验、完整离线审计及文件比较。目录随机后缀不参与复现比较；统计和源码指纹必须一致。历史证据固定保留，不随代码自动更新。

## 校验摘录完整性

在仓库根目录执行：

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('docs/evidence/stage5')
records = json.loads((root / 'checksums.json').read_text())
for name, record in records.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != record['sha256']:
        raise SystemExit(f'checksum mismatch: {name}')
print(f'{len(records)} evidence files verified')
PY
```

此校验只检测归档文件变化。仿真正确性依据公式手算、因果/资源/数值测试以及原完整明细的离线重算；文件摘要或可重复运行本身不能证明模型正确。
