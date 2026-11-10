本目录保存本地推理产物，不提交到 Git。

- `enhanced/`：增强结果。
- `comparisons/`：开发时查看的输入与输出拼图。
- `contact_sheet.jpg`：开发样图总览。
- `run.json`、`timings.csv`：开发运行信息与耗时记录。

对比和日志用于开发观察。正式方法评测已完成，原记录保留在 `day5/`，结论见 `../docs/day5.md`。

每次推理或评测使用新的空输出目录，避免覆盖正式证据或混入旧图。评测的 `per_image.csv` 保存逐图方法指标，`strength_sweep.csv` 保存三档强度结果，`summary.json` 保存汇总、协议和运行环境。质量指标可与原记录比较，耗时会受设备和运行状态影响。
