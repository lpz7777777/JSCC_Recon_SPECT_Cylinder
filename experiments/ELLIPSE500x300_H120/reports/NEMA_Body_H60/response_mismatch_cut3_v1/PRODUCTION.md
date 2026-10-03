# response_mismatch_cut3_v1 运行簿

## 2026-10-04：全量扫描与匹配灵敏度完成，重建已提交

唯一删除组已冻结；[README](README.md)是实验定义、统计、诊断和验收方法的入口。当前无已完成删除组图像，不宣告尖峰改善。

1. 65114：原20视角、5462327原始行，经原规则接受484936，新增q>3删除1168、保留483768。完整扫描约333秒。
2. 匹配训练170055/1e9，独立验证170239/1e9，平均闭合1.000950729、空间CV0.293412%。新灵敏度保存独立路径，原S不覆盖。
3. 部署12个小型代码文件及5份扫描/Sensi证据到scxi717；逐文件SHA核对，shell语法检查通过。发布路径以内容哈希冻结，见[deployment.json](deployment.json)。
4. 原排队1657719尚未启动即取消（00:00:00），用于补齐2000次只读检查点和完整/活动列一致性检查；[取消记录](job_1657719_cancelled.json)保留。不更改原物理或阈值。
5. 当前唯一作业 **1657745**，8节点×1GPU、6CPU/节点，4090/5090分区择可用节点，NCCL固定bond0。顺序50次关闭筛选回归→10次开启筛选资源/数值试跑→10000次正式Compton/JSCC；无依赖检查通过则退出，不提交替代科学组。

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
