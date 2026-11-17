# NEMA5e9原生Cartesian：4090 r2失败保留

有效r2正式1822534于14:10:37失败；没有progress/completed，日志同样触发原80%实际RSS/VRAM守卫。原同名无r2分支为未提交预检失败。全部发布/输入/失败输出保留，未重投。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4 |
| 版本地位 | current |
| 计算 | FAILED |
| 严格交付 | not-delivered |
| 可视化 | 尚未开始 |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {"formal": 1822534, "parent_5090": 1822462} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | shared Cartesian matrices 95022158ffedcdd7; execution a502c70af71dea0e |
| 发射量级 | 5e9 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/README.md](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/freeze.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/freeze.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/formal_registration.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/formal_registration.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/execution_equivalence.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/execution_equivalence.json)
- [docs/organization/20261117/remote_readonly.json](../../../organization/20261117/remote_readonly.json)
- [docs/organization/20261117/remote_logs_readonly.json](../../../organization/20261117/remote_logs_readonly.json)
- [docs/organization/20261117/remote_final_readonly.json](../../../organization/20261117/remote_final_readonly.json)
- [docs/organization/20261117/remote_logs_final_readonly.json](../../../organization/20261117/remote_logs_final_readonly.json)
- [docs/organization/20261117/protection_final.json](../../../organization/20261117/protection_final.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/campaign.py](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/campaign.py)
- [experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/bounded_matrix.py](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/bounded_matrix.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117](../../../../experiments/ELLIPSE500x300_H120/generated/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`a502c70af71dea0e`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-cartesian-4090-r2`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
