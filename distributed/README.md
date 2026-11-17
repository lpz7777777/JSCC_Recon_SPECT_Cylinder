# 通用分布式与历史适配

[Slurm入口](dual_energy_compton_slurm/README.md)与[Python实现](dual_energy_compton_python/README.md)保留通用多能量/Compton接口和原配置。它们不自动等同于当前椭圆225Ac的正式发布；运行方法按原模块说明及所需Factors/CntStat/List匹配，不能据旧配置直接重投。

当前推荐从[完整工作流](../docs/ac225/WORKFLOW.md)及[研究冻结与状态](../docs/ac225/EXPERIMENTS.md)选择具体入口：SC、纯Compton、同能量联合、OpenGATE/PyTomography、HE/EHE与Cartesian分别绑定各自数据和S。旧/tmp跨作业缓存及旧控制器方案不恢复，原程序原位保留供历史追溯。
