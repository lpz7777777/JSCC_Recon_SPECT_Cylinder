# compton_energy_probability_v5

2026-10-05。连续能量响应候选已实现并通过10项本地/冻结发布测试、独立点源评分、固定q域和完整网格数值/资源探针。**159919个固定训练事件的匹配S正在65114 GPU0累计；独立空间/联合类别验证尚未完成，没有新增配对成像。** 状态以[job.json](job.json)、[validation_gate.json](validation_gate.json)和[运行簿](PRODUCTION.md)为准。

- 全10270点源事件评分，遗漏0。七个位置的平均能量密度评分均改善，双端PIT尾部约4.8%–6.9%降至1.3%–2.3%；增益中位数仍为负，主要改善异常尾部，不能称尖峰已经减轻。
- 连续物理转移区间避免了离散支持节点数变化；解析能量积分作为参考。全20540点源上下文的数值实现最大密度误差0.1799%，32事件×132040点响应相对L2=3.68e−5、最大行总变差0.01092%，均通过门槛。
- 224个冻结q上下文、448次能量质量验证，256→512扫描最大变化1.84e−11，原接受判定一致，没有用新核改变事件集合。
- 128事件短探针后来未能预见CUDA缓存增长，旧全量已退出且部分S拒绝使用。修复后8192事件长探针通过，显存峰值27.03%、主存峰值约12.29GB，预计完整累计3673秒（约61分钟）。几何仍float64；仅高斯取值精度优化通过解析参考检查。全部既有GPU尝试计入7200秒预算，当前全量硬上限4195秒，不保存完整事件×体素响应。

[概率合同](CONTRACT.md)明确区分条件评分密度、前向响应和匹配S，以及B/KN代理、端点外推和条件材料训练的限制。独立空间与联合类别任一门槛失败就HOLD，不为了凑成像而放宽标准。

所有输入来自既有first_scatter_v2和v4冻结记录。新增光子、精细A矩阵和重建任务均为0；所有输出在独立generated子目录，原输入和生产核保持只读。原自动任务保持停止。

代码：compton_energy_probability_v5.py、validate_energy_candidate_v5.py、test_compton_energy_probability_v5.py、energy_candidate_v5_workflow.py、collect_energy_candidate_v5.py。工作流禁止重复启动；旧数值/性能失败发布的收据保留在superseded目录。

可视化：[七点源评分及尾部](figures/independent_point_probability.png)、[条件PIT分布](figures/conditional_pit.png)。这是响应诊断，不是重建图；输入SHA和图文件SHA见[figure_manifest.json](figure_manifest.json)。
