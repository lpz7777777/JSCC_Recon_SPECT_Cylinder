# EHE NEMA20视角：5e9至5e12前投影＋Poisson

匹配模型自生数据在5e9/5e10也出现明显边缘伪影；提高计数显著减弱尖峰，但高剂量仍有模糊。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | matrix-forward-plus-Poisson |
| 版本地位 | current |
| 计算 | 完成 |
| 严格交付 | 本地48选帧；远端1600帧已核对 |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 200 / 1 |
| 作业 | {"formal_array": 1762238} |
| 状态依据日期 | 2026-11-03T11:31:24.566092+08:00 |
| 矩阵 | dense65536 |
| 发射量级 | 5e9, 5e10, 5e11, 5e12 |

## 解读边界

- 匹配原生有限体积源；不是新的真实Geant4输运。
- 完整1600帧远端核对，本地严格取回48选帧，完整本地历史QA为false。

## 报告与原始说明

- [Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/REPORT.html](../../../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/REPORT.html)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_nema_low_noise_20261102/delivery_selected_strict.json](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_nema_low_noise_20261102/delivery_selected_strict.json)
- [experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_nema_low_noise_20261102/final_delivery_qa.json](../../../../experiments/ELLIPSE500x300_H120/reports/EHE/ehe_dense_nema_low_noise_20261102/final_delivery_qa.json)
- [Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/metadata.json](../../../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/metadata.json)
- [Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/scientific_qa.json](../../../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/scientific_qa.json)
- [Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/direct_visual_review.json](../../../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/direct_visual_review.json)
- [Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/independent_selected_display_qa.json](../../../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/data/independent_selected_display_qa.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_workflow_20261102.py](../../../../experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_workflow_20261102.py)
- [experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_runtime_20261102.py](../../../../experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_runtime_20261102.py)
- [experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_source_20261102.py](../../../../experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_source_20261102.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/ehe_dense_nema_low_noise_20261102/visual_results](../../../../experiments/ELLIPSE500x300_H120/generated/ehe_dense_nema_low_noise_20261102/visual_results)

## 图表入口

- [Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/REPORT.html](../../../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/REPORT.html)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show ehe-nema-poisson-20view`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
