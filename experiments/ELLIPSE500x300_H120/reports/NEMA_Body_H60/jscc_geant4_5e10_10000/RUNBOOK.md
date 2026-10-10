# JSCC实际5e10三路运行簿


## 最新实际状态：4090分批方案等待持久缓存空间

1686694已实际全COMPLETED/0:0（3分23秒），24GPU身份、小块无损往返和80%资源余量通过；整体存储门控未通过。6/8节点/tmp空间不足；1686708只读挂载检查证实/tmp位于当前作业jobcontainer，不能据此承诺验证与正式作业之间复用。此前9b5938545ad3d6c7发布中的/tmp固定节点复用方案已被实际证据否决，原发布及日志保留，不能继续该入口advance或提交validation/formal。

完整响应1530759784160字节（约1.53 TB），现有工作盘约66 GB可用。/ssd全卷空闲不等于账户配额：/ssd/scxi717属于root/0700，scxi717不可读写，目录实际配额仅1 GiB；home也只有1 GiB。已请求用户在平台开通可写、至少2 TiB持久SSD目录并提供路径。需要新独立存储修复冻结和实际I/O验证，不能修改旧发布或重新筛选/模拟。

当前例行入口仅 `python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_4090_storage_status.py`，只读核对六项PID、账户SSD配额/访问和原5090自然状态。自动任务继续ACTIVE，资源未变化时保持安静，不反复执行旧probe/取回。原1685272仍PENDING且job JSON SHA保持；未提交完整输入验证或正式重建。详见报告目录 `STREAMING_STORAGE_HOLD.md`、`streaming_4090_8x3_probe_resource_acceptance.json` 与 `streaming_4090_8x3_storage_diagnosis.json`。


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

## 2026-10-10 20:00上海：用户授权保留主作业并试4090

原5090主筛选1685272仍PENDING(Priority)，用户明确要求保持它不动，另用4090队列提交试验。实际gpu_4090每GPU最多6CPU、每CPU10000MiB；新独立筛选1686107为8节点×4GPU、每节点24CPU/共192CPU，自动请求1875GiB总内存（240000MiB/节点），AllocTRES尚为空，不能称为已分配资源证书。查询工具当时报告47张4090空闲卡，但仅3个节点可各提供至少4张卡；32张卡总数充足不表示8个所需节点可以同时立即启动。真实新作业仍PENDING(Priority)，不使用test-only打印的2051日期作ETA。

独立启动冻结fd8245063b549b78保存控制器/两个隔离检查/实际脚本；kernel2a67b816bf25eb7a、完整5e10输入、原筛选算法/20view/32rank拓扑保持。两项本地隔离检查实际通过：科学调用和拓扑逐字一致、未知启动布局拒绝；真实bash语法、scheduler test-only和三个控制成员远端SHA通过。脚本只更改OMP环境8→6、分配/输出路径及结束标签，不更改科学源码；冻结runtime中的torch.set_num_threads(10)仍保持，不能声称已改为每rank6线程。没有Geant4/PE/Scatter/验证/正式重建新作业。

新目录selection_4090_trial_1686107及allocation/logs/intent/job均独立。selection_job.json SHA b593beff14a554d3e28a9219da26ea58748f1f58be1c2e3c7d028419b5c5efe1提交前后相同，原本地协调器SHA41a4e36f62db5e4e2dd2c6785489a0f0aa986ac514733c32e0a3430682fcb0ca不变；primary_before/after保留实际scontrol，主作业申请8×4/32CPU和原Command不变。不取消、重提、移动或覆盖任一登记。

后续定时推进先检查advance_registration和selection_4090_trial_registration活PID，再读取两个作业登记。运行以下独立只读检查，然后按原主流程推进：

```powershell
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_4090_trial.py status
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_workflow.py status
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_workflow.py advance
```

不要再次执行trial submit。试跑成功退出后只读独立验收其完整筛选/资源/SHA；不可凭COMPLETED直接改主selection_job或跳过完整输入validation10。4090筛选阶段不常驻全部Compton事件响应，能运行不能证明后续完整Compton求解能在该内存配额内保留20%余量。主流程后续预算及5090提交保持原门槛。


## 2026-10-10 21:47上海：新增独立5090 8×2筛选试跑


2026-10-10 21:47上海用户新增授权：保留5090主筛选1685272及4090试跑1686107，另提交独立8节点×2张5090完整输入事件筛选 **1686450**。每节点16CPU、16rank、按配额自动252000MiB/节点，无显式mem。原20view/54631328原List/全部132040点/10496bin/稳定float64全圆q≤3和矩阵科学字节保持；独立试跑发布1516a532adb59f94仅适配16进程身份、资源验收和最终全行汇总。当前PENDING(Resources)，原两项PENDING(Priority)；上限4小时不是ETA。正式重建仍未提交，按实际筛选事件数及后续完整响应内存验证决定，不凭筛选RSS放行Compton重建。
本地3项与实际Linux3项检查均通过：事件数学AST保持、输入/输出和启动隔离、实际16GPU资源身份及重复UUID/主存超80%/错误32GPU分配拒绝。原60成员中56成员原字节不变；3个拓扑副本和kernel_config另冻，源快照见selection_5090_8x2_trial_sources。整个62成员及启动脚本远端SHA闭合。首次CPU预检因测试导入本地协调器缺失失败，尚未sbatch且无intent/job；失败发布341ccb80d8345605保留。修复只使测试助手可独立导入，原错误日志/冻结/无提交证明保留，新发布不覆盖旧发布。成功Linux检查/bash/test-only只是启动验收，不是GPU物理/数值通过。

后续先读三项本地PID登记，再分别运行两个trial helper的status和原workflow status/advance。不要再次执行submit，不取消原两项、不自动更换正式流程。新试跑完全成功退出后需要独立16rank只读逐行/资源验收（原32rank验证器不能原样套用），保留所有原字节和SHA。正式预算由真实选中事件数×78920×4字节/8节点，加实际运行开销推算，并再做完整输入验证；当前筛选不生成和常驻全部Compton响应。


2026-10-10 22:06实际8×2试跑有界恢复

- 1686450 root/batch FAILED/15:0，compute step1 FAILED/1:0，16rank都在DeviceMonitor.sample匹配UUID时失败，未创建筛选输出。原日志/分配/完整sacct远端前后及本地SHA闭合，保存selection_5090_8x2_trial_failure_1686450；旧1516a532adb59f94发布和原job/freeze保持。原执行helper/test另保存于selection_5090_8x2_trial_sources，原身份验收描述的是执行当时的字节。
- 1686107 root/batch FAILED/15:0，step1 CANCELLED/0:15；这表示失败作业终止其计算步骤，不是本轮取消原作业。原4090日志严格SHA取回，确认同一UUID检查失败；保留selection_4090_trial_failure_1686107及原冻结/登记。主1685272仍PENDING，未修改其请求或脚本。
- 远端实际torch2.8.0+cu128；其v2.8.0官方Module.cpp将UUID格式化成完整8-4-4-4-12，nvidia-smi加GPU-前缀。独立监控修复按完整非零UUID严格转换，拒绝缩写/未知/MIG/重复或缺失匹配，继续使用真实指定设备的nvidia-smi内存和原80%门槛。实际新启动16份原/规范UUID对应均通过，证据只证明启动身份，不是完整资源证书。
- 启动控制器两个未提交的预检查候选因已终止作业被squeue/scontrol清除而退出，均未上传或提交新Slurm：本地候选freeze/launch/payload以preflight_squeue或preflight_terminal后缀保存；恢复使用账号队列和最终sacct确认完整退出及保持终态作业。相应小证明记录不代替计算验收。
- 新唯一恢复1686507，冻结75e6ca22ffb21fa4、64份声明成员及原56份非改动成员SHA闭合；完整selection.py和16rank资源contract.py与旧试跑字节相同。7项本地/实际Linux检查通过。新恢复目录独立，原1686450没有完成的筛选阶段可复用；没有重做Geant4/响应。
- 推进时四个PID登记都要检查：advance_registration、selection_4090_trial_registration、selection_5090_8x2_trial_registration、selection_5090_8x2_monitor_repair_registration。status助手自动优先读取最新monitor_repair_job，不再repair/submit。旧FAILED4090仅保存证据，不能让其status异常阻止新的试跑监测；原主作业继续不动。
- 新试跑完整退出后，匹配16rank的独立只读验证发布核对全部20view/原始行/选中缓存/SHA及实际分配、逐设备/逐节点RSS和退出后Slurm MaxRSS。先保留原结果再验收，不用原32rank验证器直接声称通过。依据实际事件数和完整响应实测保留20%余量，再决定正式拓扑；目前未提交validation10或formal10000。


## 2026-10-10 22:23上海：8×2完整筛选退出，独立验收已提交

唯一有界UUID恢复1686507已完整COMPLETED/0:0，root/batch/extern和0/1步骤均成功，Slurm共8分48秒；实际8节点/16GPU、每节点16CPU和自动252000MiB。原List54631328行，原条件接受4860953，稳定float64全圆q≤3移除11866，选中4849087事件；不使用源真值筛选。完整20view/16rank收据及数组均保留原输出。

selection_5090_8x2_monitor_repair_completion_1686507保存原完整out/err、allocation、selection_manifest、sacct及严格SHA。匹配冻结16rank verify_topology检查通过，实际GPU used最大22.7224%、reserved20.1500%、逐节点保守汇总RSS5.24975%、退出Slurm MaxRSS8.28643%；这是完整筛选资源观察，不是全响应CPU缓存或重建资源证书。completion_acceptance的all_row_acceptance仍为false。

独立只读验收1686546、verification freeze4b595d58a50e48e2使用原16rank资源合同，源64声明成员中63份逐字复用，独立verify_stage仅增加执行收据/数组闭合和源作业身份绑定；原source发布没有被改写。8项本地及真实Linux测试覆盖完整闭合、缺失收据、重签SHA后的分区间隙/计数矛盾、缓存差异、数组SHA变化、raw总数不符、执行q自检失败。验收作业1节点8CPU、保留1GPU获得自动126000MiB主存，但实际核验为CPU只读；不把这一作业配额当成16GPU重建资源证书。读取全部20view原始CSV、逐值比较所有选中缓存，归档320收据及640数组和原最终40数组、manifest及分配身份。真实成功退出后严格SHA取回完整1003成员，不能以Slurm退出替代验收。

```powershell
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_8x2_selection_verification.py status
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_8x2_selection_verification.py fetch
```

推进前现在检查5份PID登记：advance、selection_4090_trial、selection_5090_8x2_trial、selection_5090_8x2_monitor_repair、selection_5090_8x2_monitor_repair_verification（均加_registration.json）。任一实际存活时不并行submit/advance/fetch；不要再submit本试跑或已登记验收。原1685272仍PENDING，原4090已FAILED，原job JSON SHA与用户保留要求一致，未取消或修改请求；旧失败输出/源发布继续保留。

初步完整响应预算：4849087×78920×4=1530759784160字节，平均178.2039 GiB/节点。按原生产预算每rank16 GiB保守开销，8×2需210.2039 GiB，超过246.09375 GiB实际自动分配的80%预算196.875 GiB。8×4预期政策配额的80%为393.75 GiB，对应保守242.2039 GiB；这些是预算、均分近似和政策预计值，不是完整响应内存实测。筛选RSS不能替代完整缓存开销，不自动放行16GPU正式重建、不改主selection_job。新试跑严格验收只记录独立结果；原validation10/strict authority/formal10000门槛不变，目前均未提交。


2026-10-10 22:28补充：1686546已实际COMPLETED/0:0、Slurm2分24秒，诊断125.04301974秒。320执行收据及640分区数组、20view最终索引/选中缓存与54631328行原List闭合，所有4849087选中缓存逐值一致。完整1003成员及141201888字节归档、原源/验收日志、实际allocation均严格远端前后/本地SHA取回。selection_5090_8x2_monitor_repair_acceptance.json是真实完整验收，不以先前completion_acceptance代替；后者all_row_acceptance=false保持为原历史初步证据。实际16个唯一GPU UUID及PyTorch原UUID/规范GPU-字符串闭合，resource_acceptance单独保存。

内存评估的原初步JSON保持历史字节，新增formal_memory_assessment_acceptance将实际完整验收事件数与相同预算绑定；没有生成完整Compton响应或提交重建。新helper fetch已complete/exit0、无本地推进器存活。不要重复计算/验收/取回此阶段，不修改或取消原主1685272/原4090登记，不自动让16卡试跑替代主生产selection。后续仍需正式拓扑的完整响应内存验证、完整输入validation10/strict authority及唯一formal10000，整个实验尚未交付，jscc-5e10继续ACTIVE。


## 2026-10-10 最新用户授权：复用筛选、4090 8×3分批响应

用户明确要求复用已完成筛选，将Compton事件响应改成分批载入，用8节点×3张4090推进三路重建，原5090作业保持不动。新的独立入口为 `jscc_5e10_4090_streaming.py status/advance`，登记前缀 `streaming_4090_8x3`。它取代此前后续正式拓扑待选的限制；不调用旧32卡advance、修改旧selection_job或取消1685272。旧FAILED4090/8×2试跑仅保存原证据。

复用1686507/1686546已验收的全部4,849,087事件及40份索引/原行缓存，按24rank重新分区，原数据/SHA保持。新响应存储仅无损压缩完整float32字节，每块最多32事件，节点本地/tmp写入、fsync和完整SHA读回；迭代按原顺序还原当前块调用原事件权重及未正则MLEM。原v5物理、全120mm/78920活动单元、全部20view、矩阵/S、三路和固定本次440末图背景均不改，不丢事件或生成跨能量和。原科学函数调用与检查点AST逐一一致；本地6项测试实际通过，100迭代全部保存历史与常驻内存及原Compton分支逐值一致。该CPU测试不是完整输入GPU资源证书。

4090每节点18CPU/3GPU，自动预计180000MiB/节点，无显式mem；真实24rank/8host/3唯一GPU、UUID/RSS/Slurm MaxRSS/显存均保留20%余量。共享盘约66.6GB可用，完整响应不写共享盘。先实际8×3节点存储/设备/无损小块probe，节点本地盘磁盘余量保护；样本压缩预算不是全部缓存容量保证。成功退出、严格日志/资源/数值证明后进入全输入三路validation10，验证全部响应块/事件、原前向/转置/S及历史回归；独立只读验收、严格取回及新24卡authority通过后直接唯一formal10000/save50。正式固定验收节点以复用相同SHA本地缓存，不重新生成成功缓存；缺失、部分或失败保留并诊断，仅有界修复。实际吞吐和时限据验证实测，不能保证ETA。

定时任务已更新为新入口；检查原五项控制器PID及新增 `streaming_4090_8x3_registration.json`，任一实际存活不并行提交/推进/取回。完整三路数值、科学/视觉QA、报告和安全Git交付之前保持ACTIVE。原5090作业自然状态只读检查，若自行失败保存完整退出和原日志，不借新授权修改它。


最新实际登记：独立streaming预检 **1686694** 已提交gpu_4090，8节点/24GPU、每节点18CPU，无显式mem，30分钟有界时限（不是ETA）。发布 **9b5938545ad3d6c7** 全部成员远端SHA和Linux导入通过，本地与实际Linux各6项测试通过。预检尚需真实退出及存储/资源/无损数值验收；validation10与formal10000尚未提交。主5090 1685272仍PENDING，原job JSON SHA b593beff14a554d3e28a9219da26ea58748f1f58be1c2e3c7d028419b5c5efe1未改；新推进器已complete/exit0。定时任务保持ACTIVE且仅按新独立登记推进，不重复probe提交或旧筛选。
