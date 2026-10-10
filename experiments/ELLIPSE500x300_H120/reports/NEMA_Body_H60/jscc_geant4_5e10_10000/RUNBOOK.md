# JSCC实际5e10三路运行簿

本实验仅 `jscc_geant4_5e10_10000`，旧研究只读。最新job JSON和实际squeue/sacct/日志为准，不能用Slurm完成状态代替科学验收。

## 2026-10-10 已完成和已提交

- 独立源：20原H60宏只改beamOn、1000worker×50M，新种子35100101–35101100；4π倍数1。CPU冻结dc23f1c14e3e036c，实际复用旧JSCC二进制SHA9de76827814a6ce6f1fa273cbdfc8b353f9fc98bf489507ce5ff94d400fd91e6，探测器SHA4f36ae7b95cfbac647885538bd64d09c991162ae1aa4c5eb46f11e65ab1292fb。
- 启动检查15705111实际COMPLETED/0:0，100000初级/16.023736秒/退出0；收据、原日志、宏和实际分配SHA已取回。它不计正式剂量，不重跑。
- 用户最新要求改为独立CPU调度。15705120在PENDING、RunTime0且无worker目录时仅取消本作业，真实CANCELLED证据与原登记保存；不得将其计入剂量或恢复提交。唯一新输运数组15708389为0–999%1000、每worker1节点/1CPU，不独占节点，可按空闲资源分批启动且最高并行1000。原1000×50M/种子/宏/二进制/50worker每view保持；另冻transport_array_freeze/部署，旧dc23f1c14e3e036c原字节保留。原保守worker进程32348秒/Slurm545分钟不变。只等空位，不取消其他项目或重复submit。
- 独立数组验收需全部1000 root及batch/extern/0成功退出，child实际SLURM_JOB_ID、父ArrayJobId/TaskId与单节点单CPU真实scontrol逐worker绑定；该节点可运行其他worker。collection读取当前transport_job的独立输出目录与完整accounting映射，不读取旧空transport目录。失败保留已成功worker，只冻结有界补缺方案。
- 15708389实际于16:27:37开始，随后accounting及展开squeue均显示1000 RUNNING；worker0真实子JobId15708392、ArrayTaskId0、NumNodes1/NumCPUs1/NumTasks1，实际allocation与Geant4初始化console已生成。transport_array_start_acceptance.json只是启动观察，尚无完整5e10完成/计数验收。控制冻结bd18503c61de2012，Linux及本地3项独立数组身份检查通过。
- 新8×4三路入口和完整输入验证器已准备；正式无联合求解/跨能量和，validation仅保留原Compton分支10次参考用于历史逐值回归。独立源/剂量、三路等式、32GPU身份、节点四rank合计RSS、保存检查点与拒绝损坏的本地9项检查通过。
- 新heartbeat `jscc-5e10` ACTIVE，每15分钟；旧compton-v5继续PAUSED。完整验收/图表/报告实际交付才暂停新任务。

## 2026-10-10 17:22上海：完整输运退出，唯一收集运行

最新sacct显示15708389全部1000个root及各batch/extern/0步骤COMPLETED/0:0，单worker真实1节点1CPU，transport_array_accounting.json完整会计SHA由transport_array_exit_acceptance.json绑定。只读抽查worker0收据为50M、独立种子35100101、实际程序1687.219秒，窗口218/440为113151/48983；它是一个worker的观察，不代表全量计数或总模拟耗时。

当前唯一collection15709588已实际RUNNING，CPU2核、无GPU，原数组发布bd18503c61de2012不变。原worker目录、源registry和二进制全部保留，只核对收据/原字节SHA、全部原始初级数、宏CRLF→LF、实际子Slurm身份，然后统一生成20视角projection/List/worker诊断和新输入archive。必须等待收集退出及严格SHA取回才能出具transport_acceptance、提交新GPU事件筛选；不得重跑已成功1000worker。

## 2026-10-10 17:33上海：实际5e10输运严格交付

1000个独立1节点1CPU worker全部实际COMPLETED/0:0，开始16:27:37、最后退出17:00:13，整批运行32分36秒，不含排队/后续收集。各worker程序1537.065711–1948.995603秒，原注册种子35100101–35101100、每worker50M、20view各50worker闭合。初级218/440/其他为14690247248/35309752752/0，总计实际50000000000个4π光子，倍数1；218份额0.29380494496，对源积分登记份额偏差约−1.40个二项标准差。

唯一collection15709588实际COMPLETED/0:0，root/batch用时4分、extern4分7秒。全部1000收据绑定真实子JobId/父ArrayJobId/TaskId和单节点单CPU分配，8000个worker成员文件原字节SHA远端核对；47个完整输入成员、原source_registry、worker_counts和原始1000收据已逐SHA严格取回。完整压缩包795780575字节，SHA51972f9b4314da5efa10fe332dc7470d6fe825963db6b4a8e8f92ee1d97811a9；transport_acceptance.json严格本地SHA为true。

实际218窗123180986、440窗53642922，原生List共54631328行。窗口总数不等于后续Compton筛选事件数；本次JSCC没有逐窗口初级能量标签，不能由这些总数捏造实测440→218机制比例。实际观测及时间见transport_measurement.json，原日志/会计/收据来源SHA闭合。CPU守护分母是worker启动时节点MemAvailable，不是成像GPU资源证书。

已继续同步已验收输入到scxi717并冻结科学kernel 2a67b816bf25eb7a，未重跑输运、收集或响应。8节点×4GPU事件筛选、完整输入validation10/authority、正式10000及图表报告仍未交付，以后续唯一登记和实际验收为准。

## 2026-10-10 17:46上海：GPU配额拒绝，独立启动控制修复

原selection请求gpu_4090、4GPU/40CPU被集群Lua提交器明确拒绝：每卡最多6CPU。没有返回正式JobId、无selection_job登记、squeue及按作业名sacct为空，原控制器PID33808完全退出/exit1；不是GPU运行/OOM或科学门控失败。原intent、原40CPU脚本及再次test-only拒绝输出保持在selection_cpu40_rejection_acceptance.json/selection_cpu40_rejected_submission_intent.json/launch_scripts中，原GPU输入和科学kernel2a67b816bf25eb7a完整保留。

实际scontrol显示gpu_4090为DefCpuPerGPU6、DefMemPerCPU10000MiB，4GPU/24CPU自动240000MiB；gpu_5090为每卡8CPU、每CPU15750MiB。为保留8节点×4GPU并满足保守全事件RAM预算，新的本地启动控制使用gpu_5090、每节点32CPU、4rank各OMP8，预期自动504000MiB/节点；只读CPU验收申请该队列8CPU/1GPU配额，预期126000MiB。没有显式mem，不修改科学矩阵/输入/算法/保存循环/阈值，实际AllocTRES和RSS/GPU/Slurm≤80%仍是验收要求。

单独控制冻结ce90d678eb32757f保存原/新协调器、原失败脚本、真实配额和修复脚本；原science58成员、kernel60成员和新5e10输入仍用原SHA，只对逐SHA绑定的本地协调器允许这一有界差异，其他科学源码变化拒绝。13项本地科学/拒绝损坏检查及4项实际Linux控制身份检查通过；新的8节点×4GPU/32CPU及只读1GPU/8CPU两项sbatch --test-only和bash语法通过，全控制成员SHA闭合。test-only打印的数字/开始日期只是调度器试算，不是已提交作业或可靠ETA，squeue仍为空。准备通过唯一advance提交修复后selection，不重跑输运、收集或覆盖科学发布。

## 唯一推进入口

先读 `generated/jscc_geant4_5e10_10000/advance_registration.json`；活PID时不并行advance/fetch/submit。

```powershell
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_workflow.py status
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_workflow.py advance
```

正常队列/运行没有变化保持安静；仅新故障、实质里程碑和最终科学结果通知。advance每次只推进已登记状态，等待阶段立即退出；身份/科学/资源失败保留结果并停止，需实际诊断修复。

输运完整退出后唯一collection核对全部1000收据/原始SHA、初级5e10和能量份额，生成20视角projection和原List、源registry原字节、worker诊断及独立archive。严格取回后部署新GPU kernel release，以本次原事件完成q筛选和逐行缓存核验。随后冻结唯一三路production contract。

GPU科学发布7be093113d485805的第一次tar控制器60秒等待超时，远端tar已退出、全部58成员SHA完整；诊断保存science_deployment_timeout_diagnosis.json。不重传/重复解包，继续原字节只读Linux导入/9项检查、8×4 bash语法及全成员SHA，验收保存science_preparation.json。后续archive控制器等待上限调到600秒，不改变科学执行时限；该CPU入口检查不代替实际完整输入GPU validation10。

上述只读恢复已完成：实际Linux9项检查全部通过、无skip，原58成员全SHA及8×4启动语法通过；科学发布保持7be093113d485805，尚未提交GPU计算阶段。orchestration_acceptance.json绑定协调器、两套数组检查、原pilot执行代码快照和实际启动脚本字节；执行/证据小文件安全Git审计后提交，不纳入原始宏/大数据/构建二进制。

每个GPU计算阶段优先 `-N8 -n8 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4`，不指定mem，bond0，LOCAL_RANK0–3。完整源事件CPU数据加8节点四rank历史固定开销的保守估计先检查，实际分配、四rank峰RSS合计、nvidia-smi采样used、PyTorch reserved及Slurm退出MaxRSS再验收。

selection/validation/formal完全成功退出后提交独立CPU只读验收作业；验收代码冻在verification_releases，不能改计算发布或结果。CPU验收使用1GPU配额取得自动真实memory TRES，但不执行GPU核，也不替代源8×4资源证明。原结果前后SHA、收据、原日志、实际分配和所有小/大图像文件严格取回。

validation10实际与原MLEM/Compton分支一致、完整算子/固定背景/资源及strict fetch全部通过才生成authority；随后唯一formal10000/save50。正式ETA仅按实测求解10→10000比例放大，事件准备和SHA/启动只计一次。任何失败/部分目录/未解析submit意向都保留并先诊断，不任意重提、删除或覆盖。

## 交付收尾

formal strict fetch完成后读重建可视化skill，并使用本工程3mm真值及最新 `nema_roi_policy.py`；新三路200帧加入独立5e10组，旧12路线当前ROI参考只读。输出固定真值尺度三维切片/中央72mm MIP、全轨迹及各球CRC/CNR/统一背景/全120mm原生域指标。不得恢复旧叠加图或把EHE200与JSCC10000称为同迭代收敛。

实际科学与视觉QA、图表/原数据SHA及报告完成，安全Git提交推送确认后保存最终delivery证据，再暂停 `jscc-5e10`。`numerical_delivery.json`只表示数值验收完成，不能提前暂停或宣称完整交付。

## 2026-10-10 17:48上海：唯一修复后筛选登记

正常advance PID49728已完整退出/exit0，唯一selection1685272已提交gpu_5090，8节点32GPU、每节点32CPU，科学release2a67b816bf25eb7a和控制ce90d678eb32757f。原40CPU意向保留为selection_cpu40_rejected_submission_intent.json，新intent仅绑定1685272。启动原字节/提交scontrol/Torch构建信息保存；这些只是提交/静态构建观察，实际运行、完整事件数、32GPU身份和80%资源仍需验收。不要重复部署/提交或重跑已交付输运。
