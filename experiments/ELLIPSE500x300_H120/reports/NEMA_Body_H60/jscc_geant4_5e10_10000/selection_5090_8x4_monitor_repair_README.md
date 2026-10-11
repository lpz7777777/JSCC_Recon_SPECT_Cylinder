# 原5090主筛选UUID修复与独立重新提交

用户在查看原1685272失败结果后明确要求“请修正bug，重新提交任务”。2026-10-11 18:40:54上海时间唯一新提交为 **1688717**，8节点×4张RTX5090、32rank、每节点32CPU，自动内存申请504000MiB/节点，无显式mem。最新只读核对仍为 **PENDING(Priority)**，实际GPU启动和完整筛选资源验收尚待运行。

原1685272在setup中的DeviceMonitor.sample按字符串匹配设备时失败。修复将PyTorch完整CUDA UUID严格规范为nvidia-smi的GPU-完整UUID，并同时保留原始及规范UUID执行证据；继续拒绝未知、缩写、零UUID、缺失和不唯一匹配。原32卡setup、NCCL、实际内存采样与80%门槛不变，全部原科学源码、输入绑定和筛选算法逐SHA保持。新的独立冻结发布为 **c49ad9c3a38589e6**；原2a67b816bf25eb7a、1685272请求/登记/失败日志和既有1686507/1686546成功科学结果不覆盖。

实际本地及超算Linux各16项检查通过，涵盖原科学数值、检查点、32卡拓扑与UUID拒绝条件。真实Linux源SHA核对、导入、bash语法和scheduler test-only通过后才提交。只有一个实际sbatch，submission_intent已绑定1688717；test-only不是新作业。

首次本地预检查因测试导入SSH依赖而在连接和提交之前失败，旧候选payload及原测试日志保留。随后把SSH依赖限于控制器submit/status，修复发布通过全部检查。正式提交后的本地控制器又把Slurm等待态NumNodes=8-8误判为资源改变：调度器实际仍请求8节点/256CPU/32GPU。登记已先写入，因而没有重复提交。新的本地资源解析器接受8或8-8精确请求，拒绝范围扩大或GPU/CPU/分区错误；4项回归检查通过。独立submission_recovery_acceptance记录实际scontrol、全部七个PID均已退出和原六登记SHA不变，原exit1登记保留其真实历史。运行中的远端冻结字节没有修改。

本次仅恢复失败的原32卡筛选。定时任务继续PAUSED；完整输入validation10和formal10000仍未提交，也未登记本次authority。4090分批缓存仍受STREAMING_STORAGE_HOLD约束，工作盘334GiB余量不能容纳1.53TB完整响应，本修复不解除持久SSD/I/O门槛。筛选完成或通过静态测试均不代表三路10000重建已交付。

执行身份见同前缀的freeze、job、launch_acceptance、submission_intent、linux_tests和submission_recovery_acceptance；sources保存实际部署runtime/UUID助手/原控制器/原测试及独立只读检查源码。当前本地控制器差异另绑定controller_repair_freeze，原科学SHA和部署身份不混用。原1685272日志及源码见[人工结果检查](../../selection_1685272_manual_review_20261011/README.md)。

手动只读状态入口（不会提交计算）：

```powershell
python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_5090_8x4_monitor_repair.py status
```

不要重复submit或调用旧advance；等待实际退出后才核对日志、全部原行/缓存、32个实际唯一GPU及资源证据。不得因本次提交直接进入validation或formal。
