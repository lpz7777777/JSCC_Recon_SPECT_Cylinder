# NEMA当前1.5mm内缩球中心ROI及统一完整背景

原3D球内中心距球面至少1.5mm；所有球共享完整体模内部背景，排除六球、肺和接触边界。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | current-evaluation-policy |
| 版本地位 | current |
| 计算 | 分析完成 |
| 严格交付 | 见原阶段证据 |
| 可视化 | 见原专题报告 |
| 视角 | policy-not-acquisition |
| 迭代 / 保存 | 见原合同 / 见原合同 |
| 作业 | {} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/nema_interior_roi_20261010/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/nema_interior_roi_20261010/README.md)
- [experiments/ELLIPSE500x300_H120/nema_roi_policy.py](../../../../experiments/ELLIPSE500x300_H120/nema_roi_policy.py)
- [experiments/ELLIPSE500x300_H120/reconstruction_output_policy.py](../../../../experiments/ELLIPSE500x300_H120/reconstruction_output_policy.py)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/nema_interior_roi_20261010/scientific_acceptance.json](../../../../experiments/ELLIPSE500x300_H120/reports/nema_interior_roi_20261010/scientific_acceptance.json)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-current-roi`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
