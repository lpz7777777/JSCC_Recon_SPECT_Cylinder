# EHE单晶体空间响应

原矩阵六个晶体的分能量MIP/切片；晶体响应覆盖及面采样问题另由准直器审计解释。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | read-only-diagnostic |
| 版本地位 | diagnostic |
| 计算 | 分析完成 |
| 严格交付 | source-bound-analysis |
| 可视化 | 图形已复核 |
| 视角 | 20 |
| 迭代 / 保存 | 见原合同 / 见原合同 |
| 作业 | {} |
| 状态依据日期 | 2026-11-03T11:31:24.568093+08:00 |

## 解读边界

- 结论范围以专题报告和冻结输入为准；诊断不授权改变生产模型。

## 报告与原始说明

- [Analysis_Plots/research/ehe_crystal_response_20261029/REPORT.html](../../../../Analysis_Plots/research/ehe_crystal_response_20261029/REPORT.html)

## 阶段证据

- [Analysis_Plots/research/ehe_crystal_response_20261029/data/metadata.json](../../../../Analysis_Plots/research/ehe_crystal_response_20261029/data/metadata.json)
- [Analysis_Plots/research/ehe_crystal_response_20261029/data/visual_qa.json](../../../../Analysis_Plots/research/ehe_crystal_response_20261029/data/visual_qa.json)

## 图表入口

- [Analysis_Plots/research/ehe_crystal_response_20261029/REPORT.html](../../../../Analysis_Plots/research/ehe_crystal_response_20261029/REPORT.html)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show ehe-crystal-response`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
