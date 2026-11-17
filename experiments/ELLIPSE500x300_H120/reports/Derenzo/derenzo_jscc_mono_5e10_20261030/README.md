# Derenzo JSCC单能实际5e10：SC / JSCC

2026-10-30人工授权“跑JSCC和SC的5e10 Derenzo，从蒙卡模拟到重建”。沿用上一轮两种单能Derenzo几何：440 keV棒径10/12/14/16/18/20 mm、53棒；218 keV棒径6/8/10/12/14/16 mm、114棒。每个能量**独立实际50,000,000,000个gamma初级**，不是Ac225衰变数，不把旧5e9计数放大或Poisson前投影当作新模拟。

| 数据 | 实际输运 | 重建输出 |
|---|---|---|
| JSCC探头440 keV | 1000×50M、全4π、20view×50worker | SC单光子、JSCC同440能量SC+Compton联合 |
| JSCC探头218 keV | 1000×50M、全4π、20view×50worker | SC单光子，单能源无440串窗背景 |

全部重建10000次、每50次保存200帧，全1密度初值、无正则；沿用现有完整120 mm、132040点/78920活动单元、原6 mm径向/3 mm轴向矩阵、v5事件核和匹配S。440联合逐次使用同一当前图像计算两类权重并以两类S之和归一化；没有跨218/440成图相加，也没有EHE新模拟或重建。

独立种子：440为70100101–70101100，218为70200101–70201100。源宏除worker剂量改为50M外，与原冻结Derenzo源逐行一致；每个view实际总2.5e9由50个worker组成，不用超过Geant4单次beamOn整数上限的大宏。CPU复用原已接受的Geant4 binary/CrystalMatrix SHA，不改物理设置。1000个worker及每个step必须成功，全部primary/count/list/source/seed/geometry/Slurm身份和SHA在同一正式收集作业内闭合，严格取回后才上传GPU。

440正式CPU输运 **16191917**、依赖收集 **16191918** 已提交。218首次提交因AssocMaxSubmitJobLimit明确拒绝，原意图按字节保存在JSCC_218/transport_submission_rejected_AssocMaxSubmitJobLimit_20261030.json，未生成作业；CPU协会MaxSubmitJobs=2000已实际查询。流程按真实展开数组元素数检查余量（218输运及收集合计需要1001名额），空出足够名额后只提交一次，不重复已接受的440请求，不取消其他项目。

**14:25最新实际状态**：218输运 **16192523**、依赖收集 **16192524** 已在余量足够后提交，原拒绝已闭合，不再等待初次提交。440已有121/1000根COMPLETED/0:0、199 RUNNING、680 PENDING；218为24 RUNNING、976 PENDING。已独立读取440 worker0实际receipt：seed70100101、50,000,000实际440 gamma、PrimaryCount=[0,50000000,0]、原binary/macro SHA匹配，耗时1568.1秒。完整50B仍需全部1000 worker及收集完成。小型证据见launch_actual_status.json。

GPU正式计划：440为**8节点×4张5090**、每节点32CPU、自动内存；218 SC为**1节点×1张5090**、8CPU、自动内存。上轮440实际接受688618事件、4卡每rank响应约54.35 GB、峰值RSS约62.15 GB。按10倍事件估计响应约2.17 TB；本轮保留原保守16GiB/rank额外主存预算，24卡略超过80%门槛，32卡满足余量。正式作业内部先完成一次实际440筛选，以实际事件分配及实际AllocTRES再次核对主存预算；全部响应无损float32常驻CPU，按原顺序逐32事件送GPU，保持原求解函数，资源/数值守卫仍有效。

原成功Derenzo发布430cdd0d92037c75逐SHA复用；新发布仅改独立研究/剂量/种子身份、直接正式执行授权与作业内阶段衔接，原联合函数及完整科学求解主体AST/字节一致，已有源/矩阵/S不覆盖。100迭代联合分支和全部保存历史的CPU回归逐值一致。全输入GPU预验证未进行，授权明确false；没有单独pilot/validation/验收作业。440正式72小时时限、218正式4小时时限，均为上界而非ETA。

本次完整阶段链由一个**仅此新研究、最多7天的一次性本地流程**衔接：实际输运→正式收集→严格文件传输→正式GPU求解→全部退出及严格结果取回。它不恢复旧研究控制器、不创建周期自动化；登记在新generated目录one_shot_registration.json。提交意图未解决或实际科学作业失败时停止、保留输出，不自动重算成功阶段。入口`derenzo_5e10_workflow_20261030.py run`；独立单步`advance`仅在该新流程PID不存活时使用。

一次性流程PID **42492** 已核对实际创建时间和进程命令并确认运行，stderr当前为空。GPU正式作业尚未提交；实际输入尚未交付时不会提前占用求解GPU。32卡请求只做了Slurm test-only语法/资源检查，没有创建实际作业；其中2080年的预计时间是不可用的调度估计，不作正式排队或ETA。完整矩阵文件SHA由原冻结initial()在正式分配内复核，部署阶段只核对源码、入口和原矩阵manifest身份，不混称完整输入GPU通过。

路径：`generated/derenzo_jscc_mono_5e10_20261030/`（本地与两端对应独立根）。transport_freeze.json、local_transport_checks.json和transport_deployment.json记录实际蒙卡发布；gpu_freeze.json、gpu_deployment.json记录原核、矩阵和直接正式发布身份。每能量transport/collection/formal_job.json与各submission_intent记录唯一作业；latest_progress.json给出当前阶段。结果尚未实际完成时不标记实验交付或图像质量通过。
