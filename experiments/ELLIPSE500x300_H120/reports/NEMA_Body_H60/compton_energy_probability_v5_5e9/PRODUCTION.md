# 5e9独立两核对照运行簿

2026-10-06：按用户新要求开始`compton_energy_probability_v5_5e9`。原始5e9包完整SHA和23个文件逐一通过校验；旧事件关联必须使用`global_legacy_row`，不把ideal-only候选或ideal行号混入legacy集合。四项输入/策略关联测试和原连续核测试在65114冻结发布中通过。

校准发布`9eb47339f4ac1a38`，65114既有agent安全连接，PID2513386。执行目录在本工程`generated/compton_energy_probability_v5_5e9`，单张空闲GPU0有界使用：扫描45分钟、8192事件吞吐/解析一致性探针10分钟、全量匹配S及独立验证120分钟。输入重用已有legacy配对输运；新增光子、规律重训和精细矩阵均为0。`calibration_job.json`和`calibration_deployment.json`记录真实PID/路径/代码/原始输入SHA。正常运行检查日志和实际资源，禁止重复启动。

NEMA全20视角扫描完成：原接受484936复现，stable_float64完整132040圆网格q≤3保留483743、删除1193，约0.246%。旧q保留483768与此相差25，是数值几何修正后的成员变化；未调整3σ阈值。`nema_selection_scan.json`记录逐视角接受/删除/原文件及选择SHA、分块q一致性。校准各数据集继续使用同一规则；只有整个selection_gate完成才进入S生成。

新正式入口已实现：显式validation10/save10和formal2000/save50、8节点×1GPU/bond0、保持原v5响应helpers及原MLEM。当前本地11项检查中10项通过，1项实际校准合同fixture尚未部署而跳过；不能把这些检查当作实际5e9试跑。包含回调保存不改变原MLEM、全部40检查点及SHA损坏拒绝、禁止任意迭代数、没有钉住实际验证authority时禁止正式提交。部署时必须补齐实际校准合同测试，随后进行两核完整事件10次验证及资源门控。

跨机器完整矩阵哈希读取比小型清单耗时长，首次同步连接120秒读取超时，未判定矩阵错误。已将独立CPU审计改为每文件进度和600秒整体硬上限；它不修改矩阵、不占重建GPU、不改变校准核。成功后`factor_identity.json`钉住两台机器三套完整SysMat、Detector、坐标、旋转、体积及清单的27项SHA；正式部署和启动必须重验。

该完整跨机器27项SHA审计已经实际PASSED。启动时rank0流式读取完整三套矩阵；NCCL有界集体通信等待为15分钟，避免旧5分钟上限短于冷共享存储SHA读取。整体启动/响应生成仍限制25分钟。未修改响应或MLEM数值路径。

8192事件实测探针完成：计算188.34秒，全170247个legacy训练事件估计3914秒（约65分钟）；整体探针215.84秒，GPU预留13.65GB/50.90GB（26.83%）、RSS12.30GB。解析融合检查32个完整圆网格事件，L2=3.77e−5、TV=1.09e−4，均低于0.001。此证据仅支持吞吐和数值实现，不替代全量独立圆/椭圆/点源物理验收。全量计算已开始，GPU0利用率约96%、驻留约13.04GiB，未占用scxi717 GPU等待校准。

两核正式入口另外记录每个rank在实际完整柱单元基底上的最小事件行和。非有限或为零时写出原始行号并HOLD，绝不静默丢弃事件；正但小于原1e−12前向下限的行仅计诊断数，不增加筛选、不改原MLEM下限。验收器核对该记录。该初始全1行和不是后续每次迭代触发下限的计数。

本地直接运行工作流前设置模块路径：PowerShell `$env:PYTHONPATH=(Get-Location).Path+';'+(Join-Path (Get-Location) 'experiments/ELLIPSE500x300_H120')`。远端冻结发布包含全部依赖，不依赖本地环境设置。

原1e9正式1666673保持完成，不重复运行。native`compton-v5`心跳已切换为本次5e9独立实验，每30分钟继续校准、实际试跑、正式提交和验收，只在故障/里程碑/科学结果通知。匹配校准HOLD时不得用ideal S代替、放宽门槛或超预算追加输运。正式四条40帧、图集、曲线及科学报告完整交付后暂停。

下一步执行链：

1. `energy_5e9_v5_calibration_workflow.py status`；实际完成后`fetch`，要求独立legacy`calibration_gate`为PASSED。
2. `factor_identity.json`完整跨机器Factors已经通过；部署/启动再次核对，不重复不必要的独立GPU计算。
3. `energy_5e9_v5_workflow.py freeze`、`deploy`、`submit --mode validation`，完成本轮实际两核10次、完整事件/rank/输出/输入/资源闭合。
4. 实际成功完全退出后`fetch`，生成钉住SHA的actual authority；`submit --mode formal`只提交唯一2000/save50顺序配对。
5. 正式严格验收/取回后运行`compare_energy_5e9_v5.py --job JOB`，完成科学和视觉QA；未有正式结果前不生成或展示预测对照图。
