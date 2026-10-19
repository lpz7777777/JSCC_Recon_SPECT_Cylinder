# 图形与科学可视化

图形工程按`research/<题目_日期>/scripts、data、figures`分层。Git保存脚本、解析几何、小型指标、SHA及说明；PNG/PDF/SVG和NPZ在本地保留。原始重建数组始终位于对应实验generated目录，不复制到图形目录。

| 优先入口 | 内容 | 状态 |
|---|---|---|
| [Derenzo完成三组](research/derenzo_completed_5e9_20261019/README.txt) | JSCC218 10000次、EHE218/440各200次；真值/MIP/切片/迭代图 | 正式完成、独立验收、严格取回 |
| [Derenzo等面积设计](research/derenzo_218_440_design_20261015_equal_regions/design_notes_zh.txt) | 本轮实际采用的218/440源几何 | 唯一当前生产设计 |
| [NEMA结构](research/nema_structure_20261014/README.md) | 六球、肺、218/440分布 | 几何说明 |
| [导师进展图](research/advisor_progress_20261014/README.md) | 成像场景、探头设计、既有NEMA迭代图 | 带日期的展示材料 |
| [全部图形工程](research/README.md) | 历史设计版本与生成入口 | 版本明确分开 |

本轮Derenzo结果使用实际发射数/棒解析体积的固定尺度，白低黑高，无平滑或亮度拟合。主图0–2包含高值显示截顶，另有0–9不截顶版本；边界插值外延保留。EHE伪影和原生6mm网格限制如实展示，不当作分辨率证书。

[工程实验索引](../docs/ac225/EXPERIMENTS.md)可反查数据来源和作业号。新图不能恢复已退休的218+440求和路线。
