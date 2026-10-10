# 新5e10矩阵噪声执行登记

最新freeze.json与各stage_job.json为唯一登记。使用ehe_forward_poisson_5e10_workflow.py status；watch为本地一次8小时有界控制器，不恢复定时任务。

顺序：新前投影/独立Poisson+validation10 → 独立完整源前投影与算子/S/背景验收和strict fetch → 唯一formal200 → 独立验收+strict fetch → 真实图集曲线数值与视觉QA → 小型安全Git交付。

独立验收发布不得覆盖执行发布；所有旧组保持只读。四项小型测试检查有界两处常量适配、发射预算/原科学SHA、新组拒绝旧剂量或study、变更分量即使更新文件SHA也不能绕过PCG64逐值回放。实际完整执行仍必要。

失败/部分目录保存原状态并诊断；不自动覆盖、修改背景、重算矩阵、增加新核或重跑模拟。大矩阵/响应/历史/归档/二进制/敏感文件不入Git。
