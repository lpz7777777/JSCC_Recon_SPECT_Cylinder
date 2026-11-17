# Derenzo等面积区域：实际输运采用的唯一设计

当前输运唯一解析几何：218为6/8/10/12/14/16mm、114棒；440为10/12/14/16/18/20mm、53棒。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | authoritative-analytic-geometry |
| 版本地位 | current |
| 计算 | 设计方案 |
| 严格交付 | 本地设计文件 |
| 可视化 | 设计示意图 |
| 视角 | not-applicable |
| 迭代 / 保存 | 见原合同 / 见原合同 |
| 作业 | {} |
| 状态依据日期 | 2026-10-29T12:42:52.493684+08:00 |

## 解读边界

- 218与440采用不同几何；当前径向6mm网格不能直接认证6mm棒分辨率。

## 报告与原始说明

- [Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/design_notes_zh.txt](../../../../Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/design_notes_zh.txt)

## 阶段证据

- [Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/data/derenzo_218keV_geometry.json](../../../../Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/data/derenzo_218keV_geometry.json)
- [Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/data/derenzo_440keV_geometry.json](../../../../Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/data/derenzo_440keV_geometry.json)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/derenzo_equal_regions_mono_5e9_20261016/macros](../../../../experiments/ELLIPSE500x300_H120/generated/derenzo_equal_regions_mono_5e9_20261016/macros)

## 图表入口

- [Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/figures/derenzo_440_218_design.png](../../../../Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/figures/derenzo_440_218_design.png)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show derenzo-design-selected`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
