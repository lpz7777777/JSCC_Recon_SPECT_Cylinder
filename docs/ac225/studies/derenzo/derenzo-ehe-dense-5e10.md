# Derenzo EHE实际5e10：输运完成、严格传输待修复

两能量各200次、400帧严格交付与数值/视觉QA完成；旧传输中断/未提交状态已被交付证据更新，物理HOLD保留。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-monoenergy |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 200 / 1 |
| 作业 | {"formal": [1769915]} |
| 状态依据日期 | 2026-11-04 |
| 矩阵 | dense65536 |
| 发射量级 | 5e10 |

## 解读边界

- 保留partial和已完成输运；严格补齐传输后才能提交后续两路，不重复蒙卡。

## 报告与原始说明

- [Analysis_Plots/research/ehe_derenzo5e10_1769915_20261104/REPORT.html](../../../../Analysis_Plots/research/ehe_derenzo5e10_1769915_20261104/REPORT.html)

## 阶段证据

- [Analysis_Plots/research/ehe_derenzo5e10_1769915_20261104/data/metadata.json](../../../../Analysis_Plots/research/ehe_derenzo5e10_1769915_20261104/data/metadata.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/ehe_dense_rerun_workflow_20261102.py](../../../../experiments/ELLIPSE500x300_H120/ehe_dense_rerun_workflow_20261102.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/ehe_dense_reruns_20261102/EHE_218](../../../../experiments/ELLIPSE500x300_H120/generated/ehe_dense_reruns_20261102/EHE_218)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show derenzo-ehe-dense-5e10`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
