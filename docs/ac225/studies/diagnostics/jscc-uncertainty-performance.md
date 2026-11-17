# JSCC v5不确定度与隔离等价执行优化

CPU配对计时与真实核局部检查已有证据；合成B/C不等于完整物理算子，CUDA stress未运行，不能推断完整GPU速度或图质改善。

## 条件与完成范围

| 字段 | 内容 |
|---|---|
| 数据来源 | isolated-numerical-study |
| 版本地位 | current |
| 计算 | 分析完成 |
| 严格交付 | local-report |
| 可视化 | 见原专题报告 |
| 视角 | 20 |
| 迭代 / 保存 | 100 / 50 |
| 作业 | {} |
| 状态依据日期 | 2026-11-17 |
| 矩阵 | SHA-bound v5 subset; manufactured benchmark matrices |

## 解读边界

- 数值或视觉QA不代表物理响应、分辨率或图像质量认证。

## 报告与原始说明

- [docs/research/jscc_uncertainty_optimization_20261115/REPORT.html](../../../research/jscc_uncertainty_optimization_20261115/REPORT.html)
- [docs/research/jscc_uncertainty_optimization_20261115/README.md](../../../research/jscc_uncertainty_optimization_20261115/README.md)

## 阶段证据

- [docs/research/jscc_uncertainty_optimization_20261115/delivery_manifest.json](../../../research/jscc_uncertainty_optimization_20261115/delivery_manifest.json)
- [docs/research/jscc_uncertainty_optimization_20261115/verification.json](../../../research/jscc_uncertainty_optimization_20261115/verification.json)
- [docs/research/jscc_uncertainty_optimization_20261115/bench_cpu.json](../../../research/jscc_uncertainty_optimization_20261115/bench_cpu.json)

## 源码入口

- [docs/research/jscc_uncertainty_optimization_20261115/adapters.py](../../../research/jscc_uncertainty_optimization_20261115/adapters.py)
- [docs/research/jscc_uncertainty_optimization_20261115/bench.py](../../../research/jscc_uncertainty_optimization_20261115/bench.py)

图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。

本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。

[返回225Ac总入口](../../README.md) · [研究结论](../../CONCLUSIONS.md)

## 复现与身份

冻结：`见原freeze/release_manifest`。数值QA：`见阶段证据，不推断`。

本页命令只读：`python -B -X utf8 tools/project_index.py show jscc-uncertainty-performance`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。
