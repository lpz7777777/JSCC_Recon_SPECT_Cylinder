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

每个GPU计算阶段优先 `-N8 -n8 --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:4`，不指定mem，bond0，LOCAL_RANK0–3。完整源事件CPU数据加8节点四rank历史固定开销的保守估计先检查，实际分配、四rank峰RSS合计、nvidia-smi采样used、PyTorch reserved及Slurm退出MaxRSS再验收。

selection/validation/formal完全成功退出后提交独立CPU只读验收作业；验收代码冻在verification_releases，不能改计算发布或结果。CPU验收使用1GPU配额取得自动真实memory TRES，但不执行GPU核，也不替代源8×4资源证明。原结果前后SHA、收据、原日志、实际分配和所有小/大图像文件严格取回。

validation10实际与原MLEM/Compton分支一致、完整算子/固定背景/资源及strict fetch全部通过才生成authority；随后唯一formal10000/save50。正式ETA仅按实测求解10→10000比例放大，事件准备和SHA/启动只计一次。任何失败/部分目录/未解析submit意向都保留并先诊断，不任意重提、删除或覆盖。

## 交付收尾

formal strict fetch完成后读重建可视化skill，并使用本工程3mm真值及最新 `nema_roi_policy.py`；新三路200帧加入独立5e10组，旧12路线当前ROI参考只读。输出固定真值尺度三维切片/中央72mm MIP、全轨迹及各球CRC/CNR/统一背景/全120mm原生域指标。不得恢复旧叠加图或把EHE200与JSCC10000称为同迭代收敛。

实际科学与视觉QA、图表/原数据SHA及报告完成，安全Git提交推送确认后保存最终delivery证据，再暂停 `jscc-5e10`。`numerical_delivery.json`只表示数值验收完成，不能提前暂停或宣称完整交付。
