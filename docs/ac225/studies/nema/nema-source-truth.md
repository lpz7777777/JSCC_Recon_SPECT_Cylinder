# NEMA H60球体、肺、218/440空间分布

原3mm三维NEMA H60真值：六球与中心肺；218热球10/17/28mm，440热球13/22/37mm。球心及完整能量分布按原manifest/数组。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | analytic-3D-source |
| 版本地位 | current |
| 计算 | authoritative-source |
| 严格交付 | 见原阶段证据 |
| 可视化 | 见原专题报告 |
| 视角 | policy-not-acquisition |
| 迭代 / 保存 | 见原合同 / 见原合同 |
| 作业 | {} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 报告与原始说明

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/README.md](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/README.md)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/manifest.json](../../../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/manifest.json)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/truth_3mm.npz](../../../../experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/truth_3mm.npz)

## 图表入口

- [Analysis_Plots/research/nema_structure_20261014/figures/nema_structure_218_440.png](../../../../Analysis_Plots/research/nema_structure_20261014/figures/nema_structure_218_440.png)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-source-truth`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
