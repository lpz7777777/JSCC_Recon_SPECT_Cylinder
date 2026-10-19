# 代码分类与查询

索引包含1251个现存源码/宏/脚本文件。完整路径及模块说明见[source_index.json](source_index.json)。

| 分类 | 数量 |
|---|---:|
| `analysis-figures` | 86 |
| `calibration-physics` | 45 |
| `execution-control` | 36 |
| `frozen-or-artifact-copy` | 196 |
| `geometry-source` | 35 |
| `reconstruction` | 137 |
| `shared-or-legacy-source` | 578 |
| `transport` | 41 |
| `verification-tests` | 97 |

```powershell
python tools/project_index.py files --group execution-control --topic derenzo
python tools/project_index.py files --group verification-tests --topic 5e10
python tools/project_index.py find single_mlem
```

分类是查找辅助；`frozen-or-artifact-copy`是冻结/绘图副本，不是新的执行入口。查询不导入、调用或提交这些脚本。

## 常用当前入口

| 用途 | 文件 |
|---|---|
| execution-control | [derenzo_campaign_workflow.py](../../experiments/ELLIPSE500x300_H120/derenzo_campaign_workflow.py) |
| transport | [derenzo_campaign_transport.py](../../experiments/ELLIPSE500x300_H120/derenzo_campaign_transport.py) |
| reconstruction | [derenzo_campaign_reconstruction.py](../../experiments/ELLIPSE500x300_H120/derenzo_campaign_reconstruction.py) |
| verification-tests | [derenzo_campaign_verify.py](../../experiments/ELLIPSE500x300_H120/derenzo_campaign_verify.py) |
| execution-control | [resume_reconstruction_controllers_20261019.py](../../experiments/ELLIPSE500x300_H120/resume_reconstruction_controllers_20261019.py) |
| execution-control | [jscc_5e10_5090_production_queue.py](../../experiments/ELLIPSE500x300_H120/jscc_5e10_5090_production_queue.py) |
| execution-control | [jscc_5e10_formal_queue_gate.py](../../experiments/ELLIPSE500x300_H120/jscc_5e10_formal_queue_gate.py) |
| reconstruction | [single_checkpoint_mlem.py](../../experiments/ELLIPSE500x300_H120/single_checkpoint_mlem.py) |
| reconstruction | [torch_active_operator.py](../../experiments/ELLIPSE500x300_H120/torch_active_operator.py) |
| shared-or-legacy-source | [nema_roi_policy.py](../../experiments/ELLIPSE500x300_H120/nema_roi_policy.py) |
| reconstruction | [reconstruction_output_policy.py](../../experiments/ELLIPSE500x300_H120/reconstruction_output_policy.py) |
| analysis-figures | [plot_derenzo_completed.py](../../tools/research/plot_derenzo_completed.py) |
| analysis-figures | [analyze_nema_interior_roi.py](../../tools/research/analyze_nema_interior_roi.py) |
