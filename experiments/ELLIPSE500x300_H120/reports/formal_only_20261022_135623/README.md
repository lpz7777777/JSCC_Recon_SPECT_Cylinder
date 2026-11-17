# 当前正式重建任务（2026-10-22）

## 2026-10-23 14:57最新状态

Derenzo440正式1722673已COMPLETED/0:0，所有步骤正常退出，SC和SC+Compton各10000次、各200个完整检查点。完整结果尚未取回本地，未做科学/视觉QA；没有提交额外验收作业。NEMA1722760仍RUNNING：两路单光子各10000次完成，Compton2750/10000。当前队列只剩NEMA。最新原始证据见[状态快照](status_20261023_145743/status.json)，机器入口见[current_jobs.json](current_jobs.json)。

下文保留10月22日初始清理和提交时的快照，历史RUNNING/进度描述不代表现态。

按最新人工指令，仅保留下列两个正式Slurm作业。自动提交控制器已停止，定时任务保持暂停，不再创建单独验证/验收作业。查询时以Slurm实时状态为准；本次快照见[final_queue.json](final_queue.json)，机器入口见[current_jobs.json](current_jobs.json)。

| 正式作业 | 研究与输出通道 | GPU拓扑 | 迭代/保存 | 本次状态 |
|---|---|---|---|---|
| 1722760 `NEMA5e10_formal_3ch` | NEMA JSCC实际5e10；440 SC、218串窗校正SC、440 ComptonOnly | 8节点×2张5090，共16GPU | 各10000次，每50次保存 | RUNNING，无依赖/未hold |
| 1722673 `Derenzo440_5e9_SC_JSCC` | Derenzo单能440实际5e9；440 SC、440 SC+Compton | 4节点×1张5090，共4GPU | 各10000次，每50次保存 | RUNNING；本次读取440 SC为7900/10000 |

两行分别是一个Slurm作业，每个作业顺序完成其全部通道。NEMA218固定加性背景只使用本次最终440 SC，不作跨能量图像相加。实际5e10/5e9是各合同定义的gamma初级数，不是Ac225衰变次数。Derenzo JSCC218、EHE218/440及NEMA历史完成组不重算。

## 清理和保留

已取消12个其他请求：1704572、1713940、1713943、1719262、1719264、1719271、1722665、1722667、1722732、1722735、1722738、1722753。其中1722732原为运行中的验证，全部步骤已退出；其他为held旧正式、验证或验收依赖。见[cancellation_accounting.json](cancellation_accounting.json)、[cancel_intent.json](cancel_intent.json)、[后继验收取消记录](cancel_late_verifier_intent.json)。未删除任何原科学发布、日志、矩阵、原数组或成功输出。

本机PID19764核对命令行、创建时间和可执行文件后已停止，原三份登记字节保存在stopped_registration_*.json；原登记标记stopped。旧自动链入口加入当前正式清单存在时的拒绝保护，防止再次生成验证/验收请求。

## NEMA直接正式入口

一次性入口为[formal_only_reconstruction_20261022.py](../../formal_only_reconstruction_20261022.py)。已存在[nema_formal_job.json](nema_formal_job.json)，重复调用返回同一作业登记，不再次sbatch；提交意图不明则停止。没有watch/后继验收提交。

新独立科学目录由[nema_freeze.json](nema_freeze.json)绑定，复用CUDA闲置缓存修复与40份原筛选数组。完整科学求解主体、原MLEM/核/检查点与原矩阵身份保持；仅将“验证通过authority”启动条件替换为真实的“人工直接正式授权”。该授权明确记录`new_topology_full_input_validation_passed=false`，没有伪造passed证明。所有实际输入SHA检查、有限非负和80%资源守卫仍在正式进程内执行，不产生额外Slurm作业。

新NEMA申请十天有限上限，仅为执行预算，不是实测完成时间或ETA；Derenzo保留原实测得到的45小时53分钟正式时限。显存诊断和CUDA缓存修复原件见[此前显存修复](../nema_vram_repair_20261022/README.md)。正式失败后先确认全退出、保留成功输出，再有界修复重投。

## 唯一结果路径

- NEMA：`generated/formal_only_20261022_135623/nema/formal`（重建超算工程内）。
- Derenzo：`generated/resource_rebalance_20261022_v3/derenzo/formal`（重建超算工程内）。

完整绝对路径、合同/脚本SHA、实际Slurm请求分别在[nema_formal_job.json](nema_formal_job.json)、[Derenzo正式原登记](../resource_rebalance_20261022_v3/derenzo/JSCC_440/formal_job.json)和最终队列证据中。原登记held字段是提交时历史值，现态见final_queue.json。
