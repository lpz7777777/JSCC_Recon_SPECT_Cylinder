# NEMA EHE加密矩阵实际5e10

A218/A440提高到65536点/可见面，匹配自身S；200次正式完成，但边缘高值、散斑和小球对比不足仍存在。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4 |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 已严格取回 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 200 / 1 |
| 作业 | {"formal": ["1761244_1"]} |
| 状态依据日期 | 2026-11-03T11:31:24.566092+08:00 |
| 矩阵 | dense65536 |
| 发射量级 | 5e10 |

## 解读边界

- C440→218仍为冻结原版本。
- EHE物理响应尚未认证；加密不等于消除散射模型偏差。

## 报告与原始说明

- [Analysis_Plots/research/ehe_dense_completed_1761244_20261102/REPORT.html](../../../../Analysis_Plots/research/ehe_dense_completed_1761244_20261102/REPORT.html)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/delivery_1761244_strict_fetch.json](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/delivery_1761244_strict_fetch.json)
- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/delivery_1761244_numerical_qa.json](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_reruns_20261102/delivery_1761244_numerical_qa.json)
- [Analysis_Plots/research/ehe_dense_completed_1761244_20261102/data/metadata.json](../../../../Analysis_Plots/research/ehe_dense_completed_1761244_20261102/data/metadata.json)
- [Analysis_Plots/research/ehe_dense_completed_1761244_20261102/data/scientific_qa.json](../../../../Analysis_Plots/research/ehe_dense_completed_1761244_20261102/data/scientific_qa.json)
- [Analysis_Plots/research/ehe_dense_completed_1761244_20261102/data/visual_qa.json](../../../../Analysis_Plots/research/ehe_dense_completed_1761244_20261102/data/visual_qa.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/ehe_dense_rerun_workflow_20261102.py](../../../../experiments/ELLIPSE500x300_H120/ehe_dense_rerun_workflow_20261102.py)
- [experiments/ELLIPSE500x300_H120/ehe_dense_rerun_runtime_20261102.py](../../../../experiments/ELLIPSE500x300_H120/ehe_dense_rerun_runtime_20261102.py)

## 图表入口

- [Analysis_Plots/research/ehe_dense_completed_1761244_20261102/REPORT.html](../../../../Analysis_Plots/research/ehe_dense_completed_1761244_20261102/REPORT.html)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-ehe-dense-5e10`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
