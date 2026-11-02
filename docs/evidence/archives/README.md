# 完整运行结果下载与离线审计

这 7 个 XZ 压缩包共 **47.94 MB**，含 **11,986 个文件**，保存五阶段共 665 次首遍仿真的完整输出。各阶段另外运行过相同的第二遍，总计 1,330 次；第二遍的文件逐字节相同，比较清单已保存，省略上传重复副本。

数据于 2026-11-02 使用已提交的 0.5.0 代码重新生成。代码、配置和种子未改变；第一至第四阶段是历史入口的当前版本重建。原环境未提交的明细未被保留，来源和验证范围详见[结果报告](../../RESULTS.md)。

## 数据包

| 压缩包 | 包内文件数 | 大小（MB） | 内容 |
|---|---:|---:|---|
| [stage1.tar.xz](stage1.tar.xz) | 17 | 0.09 | tiny / reference |
| [stage2.tar.xz](stage2.tar.xz) | 45 | 0.24 | tiny / batching / trace / zipf |
| [stage3.tar.xz](stage3.tar.xz) | 65 | 7.29 | tiny / no_local_combine / return_feedback / zipf |
| [stage4.tar.xz](stage4.tar.xz) | 373 | 9.08 | tiny / concurrent / limited / zipf，各含五方案 |
| [stage5-ablation.tar.xz](stage5-ablation.tar.xz) | 1,092 | 3.22 | 4 点 × 5 seeds × 3 PHY |
| [stage5-paper_sample.tar.xz](stage5-paper_sample.tar.xz) | 8,128 | 26.64 | 18 点 × 5 seeds × 5 方案 |
| [stage5-smoke.tar.xz](stage5-smoke.tar.xz) | 2,266 | 1.38 | 5 点 × 5 seeds × 5 方案 |

每包保留全部原始文件内容，包括逐事件/逐流 CSV、manifest/trace JSON、输出 NPY、图表和第五阶段检查点。仅规范化 tar 的时间戳及用户信息。所有包均已解压读取并与重建原件逐文件比较；[manifest.json](manifest.json)记录 SHA-256、文件数、字节数和运行目录，[publication_validation.json](publication_validation.json)记录进一步的离线审计。

## 校验下载

在仓库根目录执行，标准库 Python 即可：

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path('docs/evidence/archives')
records = json.loads((root / 'manifest.json').read_text())['archives']
for name, record in records.items():
    with (root / name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != record['sha256']:
        raise SystemExit(f'checksum mismatch: {name}')
print(f'{len(records)} archives verified')
PY
```

## 解压与审计第五阶段

```bash
bash scripts/setup_env.sh
mkdir -p results/published
tar -xJf docs/evidence/archives/stage5-paper_sample.tar.xz -C results/published
.venv/bin/python -m vdie_moe_sim --stage stage5 --audit \
  --out results/published/stage5-t3e3Oc/paper_sample-a
```

smoke 和 ablation 用相应压缩包解压，再将末尾目录换成 `smoke-a` 或 `ablation-a`。审计无需重新仿真，会核对 CSV/NPY、预算、路由一致性、数值等价、配对统计和文件完整性。当前包解压后已实际通过三套审计。

重新生成所有第五阶段数据：`bash scripts/reproduce_stage5.sh`。第一至第四阶段各有 `scripts/reproduce_stage1.sh` 至 `scripts/reproduce_stage4.sh`。完整 PDF 725 次计划只做预览；压缩包没有未执行点的结果。
