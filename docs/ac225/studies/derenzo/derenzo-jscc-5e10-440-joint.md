# Derenzo实际5e10：440 JSCC联合64卡补算

1771220失败，保存到9950/10000，共199帧已严格取回并完成预览QA；不是正式10000次完成，不恢复旧64卡请求。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | actual-Geant4-monoenergy |
| 版本地位 | current |
| 计算 | FAILED-at-9950 |
| 严格交付 | strict-partial-199-frames |
| 可视化 | reviewed-preview |
| 视角 | 20 |
| 迭代 / 保存 | 9950 / 50 |
| 作业 | {"formal": 1771220, "parent": 1769020, "historical_64gpu": 1761195} |
| 状态依据日期 | 2026-11-12 |
| 发射量级 | 5e10 |

## 报告与原始说明

- [Analysis_Plots/research/derenzo_joint5e10_1771220_20261112/REPORT.html](../../../../Analysis_Plots/research/derenzo_joint5e10_1771220_20261112/REPORT.html)

## 阶段证据

- [experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_joint_5e10_continue_20261104/preview_20261112.json](../../../../experiments/ELLIPSE500x300_H120/reports/Derenzo/derenzo_jscc_joint_5e10_continue_20261104/preview_20261112.json)
- [Analysis_Plots/research/derenzo_joint5e10_1771220_20261112/data/metadata.json](../../../../Analysis_Plots/research/derenzo_joint5e10_1771220_20261112/data/metadata.json)

## 源码入口

- [experiments/ELLIPSE500x300_H120/derenzo_joint_retry_workflow_20261102.py](../../../../experiments/ELLIPSE500x300_H120/derenzo_joint_retry_workflow_20261102.py)

## 本地数据位置

- [experiments/ELLIPSE500x300_H120/generated/derenzo_jscc_joint_5e10_retry_20261102](../../../../experiments/ELLIPSE500x300_H120/generated/derenzo_jscc_joint_5e10_retry_20261102)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show derenzo-jscc-5e10-440-joint`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
