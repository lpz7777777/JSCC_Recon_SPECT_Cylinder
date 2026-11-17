# 追加4090正式三路：1822534

用户2026-11-17明确要求在4090再提交相同8×1正式作业，保留原5090作业1822462。本目录是唯一已提交的4090分支；相邻无r2目录仅保留未提交的资源预检失败记录。

- 正式任务：**1822534 / NEMA5e9_CartHE4090**，gpu_4090，8节点×1卡，每节点8CPU，72小时。
- 执行冻结：**a502c70af71dea0e**；原科学/矩阵冻结：**95022158ffedcdd7**。
- 同一真实NEMA5e9输入、483743事件、20角、HE原生4mm Cartesian网格、100%展宽v5及独立校准S；依次440 SC、440 JSCC、218 SC，各10000/save50。
- 三矩阵直接只读复用已严格验收的原路径；不重算、不重复传输、不修改原5090代码/登记/输出。新输出独立在本分支远端formal_1822534。
- 2026-11-17 14:03（北京时间）两作业均PENDING / Priority；无实际分配、未开始迭代。不能把提交当作完成或GPU资源验收。

## 为4090内存条件做的等价执行调整

平台拒绝12 CPU/卡，也拒绝显式内存请求；最终仅申请8 CPU/卡、使用平台默认内存。实际ReqTRES为总625GiB，即每节点80000MiB；正式启动仍从AllocTRES解析实际内存。原5090请求保持不动。

新分支只将GPU准入改为4090、把新执行发布与原矩阵release显式关联，并使用只读B矩阵包装器。NumPy高级索引产生独立副本后，调用Linux MADV_DONTNEED释放本进程映射页；不写源文件，不改变副本、批次、算术、浮点精度或共享文件内容。共享切片留到原SC复制/释放时处理。Compton完整FP32响应仍在分布式CPU内存，没有换S、稀疏化或响应截断。

内存预检以实际准备基线（须不超过3GiB）、两份SC源切片、完整历史、有界512MiB staging和额外2GiB预算计算。原RSS/显存80%实际高水位守卫不改；这是新的内存管理方式下的有界预算，不是取消资源检查。完整GPU流程是否通过，必须由正式作业内的实测给出。

## 已有检查与实际限制

58份原冻结文件SHA原样一致；run_reconstruction.py只包含登记的硬件/矩阵来源/I/O入口/预算替换，反向还原后整个原程序源码及AST完全一致，原study全部键值保留。响应核、求解、更新顺序、匹配S和保存函数未改。

本地6项CPU回归通过。实际Linux小矩阵检查确认索引、前向/转置、合成S、4步历史逐值一致，源SHA不变；2MiB映射RSS从2097152B降至0。该检查使用合成小矩阵，不是完整439GB响应缓存、GPU吞吐或物理准确性验收。没有另投Slurm验证任务；实际算子检查仍在正式作业内。

本地数据/证据分别在原generated/reports研究目录下同名子目录。主要证据：freeze.json、execution_equivalence.json、local_cpu_regression.json、linux_bounded_io_regression.json、formal_submission_intent.json、formal_registration.json及initial_actual_slurm_20261117T060302Z.json。

独立有限监测：generated同名目录controller_registration.json，启动PID22172，须核对create_time/cmdline；仅管理1822534，沿用原截止2026-11-22T15:33:42Z。原5090控制器独立保留，不恢复聊天定时任务。全root/batch/extern/step成功后严格取回600帧；科学失败保留全部输出并停止，不盲目重投。

复核命令：`python -B -X utf8 experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/formal_4090_8x1_r2_20261117/campaign.py status`。

已存在formal_registration或submission_intent时禁止重新提交；活跃监测进程存在时禁止并行fetch及修改绑定源码。
