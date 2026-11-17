# NEMA5e9：100/80/60% Compton展宽

1814153三档各10000次、600帧严格交付。缩窄展宽未改善整体SSIM/NRMSE；保留旧历史差异未通过标志，不选择80/60%替代生产模型。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4 |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {"formal": 1814153, "failed_predecessor": 1813383} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | polar v5; independently matched S per width |
| 发射量级 | 5e9 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [Analysis_Plots/research/nema5e9_width_sweep_20261115/index.html](../../../../Analysis_Plots/research/nema5e9_width_sweep_20261115/index.html)
- [experiments/ELLIPSE500x300_H120/studies/nema5e9_width_sweep_20261115/README.md](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_width_sweep_20261115/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_width_sweep_20261115/strict_delivery_1814153.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_width_sweep_20261115/strict_delivery_1814153.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_width_sweep_20261115/comparison_visual_qa_20261115.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_width_sweep_20261115/comparison_visual_qa_20261115.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_width_sweep_20261115/freeze_resume_r5.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_width_sweep_20261115/freeze_resume_r5.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/studies/nema5e9_width_sweep_20261115/README.md](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_width_sweep_20261115/README.md)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/nema5e9_width_sweep_20261115](../../../../experiments/ELLIPSE500x300_H120/generated/nema5e9_width_sweep_20261115)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`0063d1f7dc9ddd01`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-width-sweep`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
