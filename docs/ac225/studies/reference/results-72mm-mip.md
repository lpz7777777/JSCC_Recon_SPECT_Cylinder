# 2026-10-29方法对比画廊：原EHE矩阵快照

保留6组/15路线/90选帧的方法对比；其中EHE为旧矩阵版本，不代表11月加密矩阵结果。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-only |
| 版本地位 | historical-reference |
| 计算 | 分析完成 |
| 严格交付 | 见原阶段证据 |
| 可视化 | 见原专题报告 |
| 视角 | 20 |
| 迭代 / 保存 | 见原合同 / 见原合同 |
| 作业 | {} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 报告与原始说明

- [Analysis_Plots/results/README.md](../../../../Analysis_Plots/results/README.md)
- [Analysis_Plots/results/REPORT.html](../../../../Analysis_Plots/results/REPORT.html)
- [docs/ac225/RESULTS.md](../../RESULTS.md)

## 阶段证据

- [Analysis_Plots/results/data/result_catalog.json](../../../../Analysis_Plots/results/data/result_catalog.json)
- [Analysis_Plots/results/data/source_routes.csv](../../../../Analysis_Plots/results/data/source_routes.csv)
- [Analysis_Plots/results/data/organization_audit.json](../../../../Analysis_Plots/results/data/organization_audit.json)

## 图表入口

- [Analysis_Plots/results/NEMA/5e9/440keV/figures](../../../../Analysis_Plots/results/NEMA/5e9/440keV/figures)
- [Analysis_Plots/results/NEMA/5e9/218keV/figures](../../../../Analysis_Plots/results/NEMA/5e9/218keV/figures)
- [Analysis_Plots/results/NEMA/5e10/440keV/figures](../../../../Analysis_Plots/results/NEMA/5e10/440keV/figures)
- [Analysis_Plots/results/NEMA/5e10/218keV/figures](../../../../Analysis_Plots/results/NEMA/5e10/218keV/figures)
- [Analysis_Plots/results/Derenzo/5e9/440keV/figures](../../../../Analysis_Plots/results/Derenzo/5e9/440keV/figures)
- [Analysis_Plots/results/Derenzo/5e9/218keV/figures](../../../../Analysis_Plots/results/Derenzo/5e9/218keV/figures)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show results-72mm-mip`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
