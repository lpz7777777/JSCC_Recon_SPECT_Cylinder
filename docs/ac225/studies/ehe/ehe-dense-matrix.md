# EHE218/440加密系统矩阵

A218/A440原0–6与唯一补块7完成，16²→256²点/可见面；当前实际EHE重建采用此版本。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | system-matrix |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 矩阵SHA身份已绑定 |
| 可视化 | not-an-image-study |
| 视角 | 20 |
| 迭代 / 保存 | 见原合同 / 见原合同 |
| 作业 | {"formal_original": 1749085, "missing_block_repair": "1749513_7"} |
| 状态依据日期 | 2026-11-03T11:31:24.566092+08:00 |
| 矩阵 | dense65536 |

## 解读边界

- 保持原几何、材料与响应公式；散射均匀板深度近似及物理HOLD仍保留。
- 不重算C440→218。

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_surface_20261029/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_surface_20261029/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_surface_20261029/latest_status.json](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_surface_20261029/latest_status.json)
- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/reconstruction_compact_freeze.json](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/reconstruction_compact_freeze.json)
- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/local_scientific_checks.json](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/local_scientific_checks.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/ehe_dense_surface_workflow.py](../../../../experiments/ELLIPSE500x300_H120/ehe_dense_surface_workflow.py)
- [experiments/ELLIPSE500x300_H120/ehe_dense_surface_runtime.py](../../../../experiments/ELLIPSE500x300_H120/ehe_dense_surface_runtime.py)
- [experiments/ELLIPSE500x300_H120/ehe_dense_surface_repair_workflow.py](../../../../experiments/ELLIPSE500x300_H120/ehe_dense_surface_repair_workflow.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/ehe_dense_surface_20261029](../../../../experiments/ELLIPSE500x300_H120/generated/ehe_dense_surface_20261029)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show ehe-dense-matrix`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
