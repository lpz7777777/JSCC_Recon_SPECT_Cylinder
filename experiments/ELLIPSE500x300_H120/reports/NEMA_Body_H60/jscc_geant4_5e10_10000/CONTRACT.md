# 独立实际JSCC5e10三路科学合同

本页整理当前有效科学条款，过程状态在[README](README.md)，操作在[RUNBOOK](RUNBOOK.md)。旧说明全部原字节保留于[历史合同](history/CONTRACT_20261019.md)。运行中的可执行身份仍由冻结发布的contract.json、源SHA及匹配authority确定，本次文档整理未改变这些字节。

## 源、观测与剂量

- NEMA H60原双能3mm三维源、原manifest球心、真空环境；全4π，每事件一个gamma，剂量倍数1。原/xcat/angle只旋转源位置。
- 1000×50000000=50000000000实际初级gamma，20view、每视角50worker，新种子35100101–35101100。35100001的100000光子启动检查不计正式剂量。
- 每CPU worker1节点1核、可共享，数组0–999%1000。全部worker及batch/extern/0步骤需COMPLETED/0:0，实际ArrayJobId/TaskId、初级能量计数、10496bin计数、原List、二进制/宏/分配/日志/收据SHA闭合。218期望初级份额0.29380779868182727，源混合5SE检查不放宽。
- 复用原已验收A218、A440、C440→218及坐标/旋转/体积/活动域。完整120mm、132040全圆点、78920活动单元、10496bin、20view保持。

## 固定算法和三路输出

| 顺序 | 通道 | 条件 |
|---|---|---|
| 1 | 440_SinglePhoton | 原单光子MLEM，全1、无正则 |
| 2 | 218_SinglePhoton_CrossTalkCorrected | 本次最终440 single经C前投影得到固定加性背景 |
| 3 | 440_ComptonOnly | 原Compton分支、连续能量v5及匹配自身S |

正式10000/save50，各200帧及200 atomic/fsync检查点，共600。当前不计算生产440联合图，也不生成218/440跨能量求和图；这与Derenzo单独授权的440 SC+Compton不同。

全部本次筛选事件复用1686507/1686546严格交付的4849087事件、40数组，原行与缓存身份保持。筛选为稳定float64全圆q≤3，不依赖真值；禁止重新筛选或重复旧阶段来冒充推进，不丢事件或重训/换核。CPU保存完整响应，原顺序32事件分批转GPU。

## 执行与资源身份

- 当前独立生产冻结ffe08f48be856816：8节点×4 GPU、32rank、每父任务torchrun4子进程，LOCAL_RANK独立绑定、NCCL bond0。
- gpu_5090每节点32CPU，按每CPU15750MiB自动内存，预期504000MiB/节点；无显式mem。所有分配按实际AllocTRES核对。
- 真实32rank/8host/4唯一UUID闭合；每节点四进程峰RSS之和、Slurm MaxRSS以及每GPU reserved/采样used均不超过实际分配/显存80%。独立只读CPU验收申请1节点8CPU/1GPU配额，不执行GPU核，不能替代实际GPU资源证明。
- 正式时限由本次validation各10次求解段×1000推算，准备和输入SHA开销只计一次、留余量并有界，不给ETA。当前gate实际不足且拒绝放行，见原始[时长证据](production_5090_8x4_queue_measured_time_hold_20261019.json)。

## 验证、取回与失败处理

完整输入validation10/save10检查全部20view/全部探测行的原矩阵前向、转置、匹配自身S，L2≤1e-5；原440/218 MLEM及Compton末图L2≤1e-5，历史逐值一致，全部事件使用同一本次输入。Slurm完整成功、独立验收及严格取回后才可生成本次authority。

正式独立验收需核对三路末图、活动域/域外零、有限非负、全部200帧/600检查点/历史SHA；重算三个自身S及本次最终440生成的218全view/bin加性背景，L2≤1e-5。原结果前后SHA不变，退出后逐文件取回原日志、分配和会计。

未解析submission intent禁止重复sbatch；活PID禁止并行advance/fetch/submit。失败或部分目录保留，先确认完整退出、再冻结独立有界修复；成功科学计算不重复，不覆盖发布，不取消其他项目。

## 科学与视觉交付

真值来自原`generated/NEMA_Body_H60/truth_3mm.npz`及manifest三维球心。球中心距球面≥1.5mm的全部3mm体素等权；统一背景为Phantom内部完整体素，排除六球/肺/接触边界。不能宣称去掉全部部分容积效应。

三路0–10000全部保存帧评估CNR/CRC、背景CV、积分与轴向泄漏；固定实际发射密度尺度、crop0、无平滑/单图亮度拟合。EHE0–200和JSCC0–10000分开展示，不把相同迭代视为相同收敛；不生成跨能量和。

代码/文件/图曲SHA、实际资源、科学解释、视觉QA和报告闭合后才算交付。定时任务按最新人工指令保持PAUSED。Git只含审核小文本，排除矩阵、响应、事件/大观测、数组、归档、二进制和凭据。
