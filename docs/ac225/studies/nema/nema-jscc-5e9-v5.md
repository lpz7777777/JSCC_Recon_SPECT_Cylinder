# NEMA JSCC 5e9 连续核v5：历史10000次正式基准

历史六路含已退休跨能量求和；后续仅采用获授权的分能量通道和当前ROI。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-dual-energy-proxy |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 历史结果已严格交付 |
| 可视化 | 已按当前ROI复核 |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {"formal": 1669255} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/ACCEPTANCE.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/ACCEPTANCE.md)
- [docs/DUAL_ENERGY_BASELINE.md](../../../DUAL_ENERGY_BASELINE.md)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy](../../../../experiments/ELLIPSE500x300_H120/generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-jscc-5e9-v5`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
