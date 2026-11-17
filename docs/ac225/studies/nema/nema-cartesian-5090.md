# NEMA5e9原生Cartesian：5090正式三路

1822462于14:36:55失败退出；进度记录440 SC到10000，但根和三路completed均缺失，不能认定计算完成或完整交付。实际RSS/VRAM超过原80%守卫；输入、输出、原发布和日志保留，未修复/重投。

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
| 作业 | {"formal": 1822462} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | native HE Cartesian4mm; new A218/A440/C and calibrated Compton S |
| 发射量级 | 5e9 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/README.md](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/freeze.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/freeze.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_registration.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/formal_registration.json)
- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/matrix_strict_transfer.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/matrix_strict_transfer.json)
- [docs/organization/20261117/remote_readonly.json](../../../organization/20261117/remote_readonly.json)
- [docs/organization/20261117/remote_logs_readonly.json](../../../organization/20261117/remote_logs_readonly.json)
- [docs/organization/20261117/remote_final_readonly.json](../../../organization/20261117/remote_final_readonly.json)
- [docs/organization/20261117/remote_logs_final_readonly.json](../../../organization/20261117/remote_logs_final_readonly.json)
- [docs/organization/20261117/protection_final.json](../../../organization/20261117/protection_final.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/README.md](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/README.md)
- [experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/run_reconstruction.py](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/run_reconstruction.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/nema5e9_cartesian_he_20261115](../../../../experiments/ELLIPSE500x300_H120/generated/nema5e9_cartesian_he_20261115)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`95022158ffedcdd7`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-cartesian-5090`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
