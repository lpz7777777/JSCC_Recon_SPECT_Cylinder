# Derenzo JSCC440单能实际5e9：SC与SC+Compton正式10000次

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
| 作业 | {"formal": 1722673} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/formal_only_20261022_135623/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/formal_only_20261022_135623/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/formal_only_20261022_135623/final_queue.json](../../../../experiments/ELLIPSE500x300_H120/reports/formal_only_20261022_135623/final_queue.json)
- [experiments/ELLIPSE500x300_H120/reports/resource_rebalance_20261022_v3/derenzo/JSCC_440/formal_job.json](../../../../experiments/ELLIPSE500x300_H120/reports/resource_rebalance_20261022_v3/derenzo/JSCC_440/formal_job.json)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/derenzo_equal_regions_mono_5e9_20261016/JSCC_440](../../../../experiments/ELLIPSE500x300_H120/generated/derenzo_equal_regions_mono_5e9_20261016/JSCC_440)

## 图表入口

- [Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/final_mip.png](../../../../Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/final_mip.png)
- [Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/final_mip_unclipped.png](../../../../Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/final_mip_unclipped.png)
- [Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/iteration_mip_gallery.png](../../../../Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/iteration_mip_gallery.png)
- [Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/final_central_slices.png](../../../../Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/final_central_slices.png)
- [Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/native_convergence.png](../../../../Analysis_Plots/research/derenzo_jscc440_5e9_20261023/figures/native_convergence.png)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show derenzo-jscc-440`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
