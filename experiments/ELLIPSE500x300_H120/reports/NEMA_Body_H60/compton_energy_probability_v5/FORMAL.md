# v5独立正式入口与执行合同

2026-10-06最终状态：1666592完整事件双模型10次真实通过后，唯一正式**1666673**已完成顺序A/B2000/save50并完全退出。四条40帧历史、80个持久检查点、91225事件/20视角/200worker种子/1e9初级γ、78920共同活动列、全部输入/S/源码/Factors及实际资源通过严格fetch验收。4节点×1GPU/bond0，实际1:55:30，GPU33.46%/60.86%、进程RSS22.78%/22.76%，Slurm主存23.04%（实际60000MiB/节点）。[正式最终报告](ACCEPTANCE.md)和[图集/全部40帧曲线](comparison_1666673/README.md)完成科学/视觉QA；两路2000次达到双50%极端峰工作判据，其他限制如实报告。本合同保留执行门槛，后文是已执行的流程，不再提交重复配对或恢复旧作业。

## 唯一实验及入口

A为当前角度核，B为连续材料能量核。两组均使用固定91225 stable_float64 q≤3事件、132040完整圆网格、78920完整柱坐标活动单元；各自使用已经独立验收的匹配S，共用原440计数/矩阵和全1初值。每组只运行440 Compton和440 JSCC，共四条历史。原run_reconstruction.py、torch_active_operator.py、共享事件响应和原生产82040基底不修改；旧10000次、Huber/TV、精细场及Geant4均不恢复。

新文件：run_energy_formal_v5.py、energy_formal_v5_contract.py、verify_energy_formal_v5.py、energy_formal_v5_workflow.py、reconstruct_energy_formal_v5.sh。旧run_energy_preflight_v5.py仍为10次入口，其保护不绕过。新入口显式区分validation 10/save10和formal 2000/save50，任意10000或其他保存周期会拒绝。

## 先验证新入口，再授权正式运行

1666592依次运行A和B各完整事件10次，从全1开始，与1666430同模型、同78920列两路第10帧直接比较，相对L2≤1e−5。完整事件/rank身份、20视角、匹配S、全部输入/Factors/代码SHA、有限非负、域外零、末帧一致和实际资源同时通过才有效。

7项新测试包括真实清单的能量分类/worker语义与异常拒绝、任意迭代数/改变已验算子拒绝、无钉住SHA的真实验证authority拒绝、保存回调与原MLEM逐值一致、40个检查点及篡改拒绝、非法快照不发布及禁止覆盖。另使用已验收pilot图像作为验收器协议fixture，以确认全量网格/资源/身份接口；它不是新入口实际重建的证据。新入口完整事件短程仍必须运行。

成功退出后workflow fetch严格验收两路并核对真实Slurm MaxRSS，生成formal_authority.json。Authority包含两模型实际验证结果、图像/清单/分配SHA及当前合同SHA，单独钉住其文件SHA。正式启动只读复验authority及图像，不覆盖旧收据。通过后直接submit --mode formal，无需再次询问用户；避免重复作业。

## 保存、资源和有界运行

每50次由原MLEM已有callback持久化两个通道active/full及checkpoint_manifest：先在同一输出目录写完整临时快照、逐文件fsync，再原子发布，Linux同步目录。既有检查点不覆盖。正式完成应有每模型40个检查点、两路各40帧历史；逐帧与checkpoint、末帧与final严格一致。取消/失败时保留已发布帧，未完成图像不能称为正式结果。

GPU reserved和进程RSS必须≤实际分配80%；响应生成设75%主动限制。实际主存从本次scontrol AllocTRES读取，集群禁止显式--mem类参数。4个rank必须为4个唯一节点，首选已有资源验证的4节点×1GPU，只排除已确认故障wqd10nba06g6。若资源不足，重新冻结/验证8节点，不丢事件或降采样。

节点路径/CUDA预检限60秒，分布式启动超时300秒，响应生成最多25分钟。验证每模型25分钟、整个90分钟。正式walltime使用新验证的准备和求解实测估算2000次，预留50%及额外300秒、阶段最多3小时；超预算先诊断吞吐。两组顺序启动，上一组完全退出释放内存后才运行下一组。

```powershell
python experiments/ELLIPSE500x300_H120/energy_formal_v5_workflow.py status
# 1666592真正完成、内容及Slurm资源通过后：
python experiments/ELLIPSE500x300_H120/energy_formal_v5_workflow.py fetch
python experiments/ELLIPSE500x300_H120/energy_formal_v5_workflow.py submit --mode formal
# 正式成功后同一fetch严格验收并逐文件SHA取回四条历史与全部检查点。
```

## 交付及结论

保留真实NEMA H60双能三维球体真值和原ROI。A/B的100/500/1000/2000次，共同固定尺度、无平滑，中心轴/冠/矢位与中央72mm MIP；完整120mm域的最大密度、峰/背景、位置、高分位、源外|z|>30泄漏、总积分、440三球13/22/37mm CRC/CNR及背景均值/CV，40帧完整曲线。完整单元的f<0.1指标不适用。

同迭代最大值和峰/背景均下降≥50%为工作判据；CRC损失>5个百分点标记代价。2000次不外推10000稳定性，条件代理不称为完整无条件物理证书。无论改善与否如实交付。完整验收、图集、曲线和报告交付后暂停compton-v5，不自动延长或叠加其他响应修正。
