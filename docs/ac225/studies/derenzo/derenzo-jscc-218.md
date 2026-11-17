# Derenzo JSCC 218 keV：独立单能5e9

每个探头/能量独立5e9 gamma；两个源不是同一次225Ac衰变链，218背景为零。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-monoenergy |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 10000 / 50 |
| 作业 | {"validation": 1705505, "validation_verify": 1705507, "formal": 1705508, "formal_verify": 1706056} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_equal_regions_mono_5e9_20261016/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_equal_regions_mono_5e9_20261016/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_equal_regions_mono_5e9_20261016/JSCC_218/formal_acceptance.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_equal_regions_mono_5e9_20261016/JSCC_218/formal_acceptance.json)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/derenzo_equal_regions_mono_5e9_20261016/JSCC_218](../../../../experiments/ELLIPSE500x300_H120/generated/derenzo_equal_regions_mono_5e9_20261016/JSCC_218)

## 图表入口

- [Analysis_Plots/research/derenzo_completed_5e9_20261019/figures/final_mip.png](../../../../Analysis_Plots/research/derenzo_completed_5e9_20261019/figures/final_mip.png)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show derenzo-jscc-218`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
