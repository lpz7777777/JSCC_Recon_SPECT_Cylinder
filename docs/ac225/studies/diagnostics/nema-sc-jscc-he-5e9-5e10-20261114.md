# SC/JSCC/HE综合派生评价

复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | derived-from-strict-delivery |
| 版本地位 | current |
| 计算 | 分析完成 |
| 严格交付 | derived-report |
| 可视化 | see-metadata |
| 视角 | 20 |
| 迭代 / 保存 | 200 / 1 |
| 作业 | {} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | mixed native grids; see metadata |
| 发射量级 | 5e9, 5e10 |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/index.html](../../../../Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/index.html)

## 阶段证据

- [Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/data/metadata.json](../../../../Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/data/metadata.json)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show nema-sc-jscc-he-5e9-5e10-20261114`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
