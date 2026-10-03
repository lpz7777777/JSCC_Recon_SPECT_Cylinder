# response_mismatch_cut3_v1 运行簿

## 2026-10-04 06:04–06:06：回归和试跑通过，正式迭代开始

1657887分配8个独立节点、各1张RTX4090，启动预检全部通过，13秒完成；确认没有使用已排除的wqd10nba06g6。50次关闭筛选重建接受484936个事件，两路与1644876第50帧相对L2均为0。开启筛选10次完整数据试跑接受483768个事件，20视角及8rank闭合，新Sensi/config/23输入/三套Factors/几何哈希、活动列与完整图对应、椭圆外零、有限非负及末帧一致均通过。

两阶段均由远端verify_response_mismatch.py显式传入mode/config/baseline/geometry核验；取回各自6张数组及run_manifest后逐文件核对SHA，并在本地用相同入口再次验收。小型证据见[回归](regression_1657887_verification.json)、[试跑](pilot_1657887_verification.json)和[门控摘要](gates_1657887.json)，图像数组在忽略的generated/RemoteResults中。

回归GPU预留/物理显存最高44.2315%、进程峰值RSS/实际授予主存最高45.2177%；试跑分别46.7144%、45.1289%，均满足至少20%余量。实际主存授予每节点60000MiB（62914560000bytes），不是用节点物理总内存代替。Slurm回归步骤MaxRSS27858332K、7分55秒；试跑步骤MaxRSS22572412K、2分37秒。包含响应准备与核验，不能直接将阶段总时间除以迭代数作为正式迭代速度。

链内正式Compton/JSCC已启动，仍从全1初值开始，复用同一483768事件响应；06:06快照保存至50/10000，随后日志确认推进至100/10000。实时8节点GPU利用率53–100%，各卡瞬时使用2653–2673MiB；这是运行快照，不代替完整运行峰值验收。后续在2000次只读快照就绪后按共同尺度形成中期图集，不因早期迭代改阈值或提前判定改善。当前没有2000快照或正式verification，不通知科学结果。


## 2026-10-04 05:04–05:12：启动故障定位及修复

1657745于集群时间04:56:10启动、04:57:55退出，Slurm FAILED、ExitCode15:0；全部步骤已终止。stdout仅出现`RESPONSE_PHASE regression 50`，尚无回归输出目录或图像。节点wqd10nba06g6无法切换到工程目录，并报告`torchrun`缺失。原始日志、实际8节点/8GPU/每节点60000MiB分配及终态记账见[failure_1657745](failure_1657745/summary.json)。这与此前1651326/1651944的挂载故障一致；本次新启动脚本遗漏了已记录的绝对Python和/tmp预检修复。

现已修复`reconstruct_response_mismatch.sh`：调用`/data/home/scxi717/.conda/envs/torch/bin/python -m torch.distributed.run`，每个srun在/tmp启动；全部节点先检查发布代码、几何、Factors、首末List和匹配S可读，路径检查最多12次×5秒，CUDA检查最多60秒。预检通过后才初始化进程组。50次回归及10次试跑各有2小时上限，正式阶段受48小时作业上限约束；任一阶段失败即退出释放分配。数据根与冻结代码根分别传递，保持JSCC核导入原冻结发布。

发布`response_mismatch_cut3_v1_a527ffaad51223a9`已逐文件SHA核对及远端`bash -n`通过。修复只改启动和安全重提工具，筛选配置、响应核、MLEM、独立Sensi及所有输入哈希不变。仅排除确认故障节点wqd10nba06g6；不恢复Huber/TV。原失败发布保留、不原地改写。

`submit_response_mismatch.py --replace-failed`先核实旧作业不在队列、sacct为同名终态失败且无阶段verification证据，才归档旧登记；遇到活动作业或验收证据拒绝自动替换。当前账户达到50个作业，工具等待空出的提交位，不取消其它项目任务。重提后当前编号写入job.json；历史编号见job_1657745_failed.json。自动任务已更新为读取当前登记，登记暂缺时按`--exclude wqd10nba06g6 --wait-seconds 0`安全重试。

修复作业已重新提交为 **1657887**（2026-10-03T21:13:56Z本地提交记录），8节点×1GPU、NCCL bond0，当前Priority排队。scontrol确认ExcNodeList=wqd10nba06g6、WorkDir=/tmp及新冻结发布；同名活动作业只有此一个。安全重提策略的4个用例均通过：拒绝活动作业、拒绝COMPLETED、拒绝已存在验收文件的作业，只归档无验收的终态失败；再次执行提交工具直接返回1657887，无重复提交。

此时50次回归、10次试跑、正式图像及科学结论均未通过。重新入队或启动预检通过不算数值验收。

## 2026-10-04：全量扫描与匹配灵敏度完成，首次重建提交（历史）

唯一删除组已冻结；[README](README.md)是实验定义、统计、诊断和验收方法的入口。当前无已完成删除组图像，不宣告尖峰改善。

1. 65114：原20视角、5462327原始行，经原规则接受484936，新增q>3删除1168、保留483768。完整扫描约333秒。
2. 匹配训练170055/1e9，独立验证170239/1e9，平均闭合1.000950729、空间CV0.293412%。新灵敏度保存独立路径，原S不覆盖。
3. 部署12个小型代码文件及5份扫描/Sensi证据到scxi717；逐文件SHA核对，shell语法检查通过。发布路径以内容哈希冻结，见[deployment.json](deployment.json)。
4. 原排队1657719尚未启动即取消（00:00:00），用于补齐2000次只读检查点和完整/活动列一致性检查；[取消记录](job_1657719_cancelled.json)保留。不更改原物理或阈值。
5. 当时登记作业 **1657745**，8节点×1GPU、6CPU/节点，4090/5090分区择可用节点，NCCL固定bond0。顺序50次关闭筛选回归→10次开启筛选资源/数值试跑→10000次正式Compton/JSCC；无依赖检查通过则退出，不提交替代科学组。

初次提交遇到账户50作业上限；提交工具仅等待空出的账号提交位，没有取消其它工程任务。当前作业已成功入队；历史submission_wait.json仅是入队前快照，不代表当前仍未提交。[job.json](job.json)为当前登记。

沿用用户此前授权的定时验收，已建立本聊天每30分钟的自动任务 `nema-5e9-3`，状态ACTIVE，见[登记](automation.json)。正常排队/运行无变化保持安静，只在故障、2000次中期图集或最终交付通知；完整报告、固定尺度图集、曲线及删除清单交付后暂停。不恢复已完成的1e9监控或已取消的正则化任务。任务持久提示覆盖当前冻结文件、SSH安全、验收门控、取回和Git排除规则；采用[官方文档](https://learn.chatgpt.com/docs/automations?surface=app)所述在本聊天继续的定时任务方式。

## 远端位置

scxi717所有执行及输出在本工程 `experiments/ELLIPSE500x300_H120` 下：

```text
/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/
  code_releases/response_mismatch_cut3_v1_749de269c15ebe99/
  generated/response_mismatch_cut3_v1/scan/Sensi_d
  generated/response_mismatch_cut3_v1/regression_1657745/
  generated/response_mismatch_cut3_v1/pilot_1657745/
  generated/response_mismatch_cut3_v1/formal_1657745/
    checkpoint_2000/
  logs/response_cut3.1657745.out
  logs/response_cut3.1657745.err
```

每阶段的verification.json是真实验收依据；不能仅凭COMPLETED或输出目录存在判断成功。formal的checkpoint_2000只是中期观察，禁止用它证明10000次已完成。资源来自run_manifest及Slurm实际分配，另保存allocation/accounting文本。

## 尚待执行

- 50次关闭筛选，484936个事件、两路与基线第50帧相对L2≤1e−5。
- 10次完整数据，483768个事件，逐视角/rank闭合、有限非负值、GPU与主存至少20%余量。
- 唯一正式组10000次、每50次保存，两路各200帧及末帧一致。
- 第2000次只读图像观察，以及正式全部曲线、固定共同尺度图集、完整120mm尖峰/积分/泄漏与球CRC代价。
- 正式报告、取回哈希和安全Git更新；完成后结束本轮验证，无阈值扫描或新增光子量。

若作业失败：先读取日志和实际分配，定位失配/资源/通信原因；旧作业确认退出后才使用新的冻结发布和后续修复编号。不得将已失败目录原地改成“通过”。
