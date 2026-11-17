# 笛卡尔NEMA5e9串窗矩阵I/O补缺（2026-11-16）

原科学冻结`95022158ffedcdd7`保留。修复执行冻结`009433dd7e2b3b50`，
只修改串窗原始文件的选择和有界补缺衔接；不改原GPU二进制、物理参数、
分块、四层标定、体积权重、v5响应或MLEM。

## 原因与成功数据复用

原A440/A218各15块完成，21,285,888,000字节/矩阵。原C首块PE和scatter成功，
随后`build_matrix.py`要求`SysMat_withScatter`，触发FileNotFoundError。
C的实际`Params_Physics[3]=0`，不输出combined；原生产
`run_factor_conversion.py`也是读取`Scatter_SysMat`。C光峰侧窗PE总和为零。
此处不能开启combined或改变响应核来掩盖文件读取错误。

原进程695217和旧有限衔接28856已确认退出。旧28856号被conhost进程复用，
不是旧控制器；没有取消该进程或任何无关任务。旧release、日志、原C partial及
首块原始数组均保留。原C首块资源/计时返回值在旧程序异常前未落盘，不能补造；
成功退出由原run_bounded返回后才执行文件检查的控制流、完整日志及文件闭合证明。

新输出`matrices_repair_20261116`：

- A440/A218逐文件SHA核对后只读硬链接复用，不复制42.6 GB数据。
- C首块核对原参数、网格、完成日志、大小和SHA，直接整理原成功响应。
- 仅计算剩余C分块02、04…28；每块保留原参数和日志，整理后SHA闭合。
- 删除仅限本次新分块的中间数组；原首块和失败目录不删除。
- `completed.json.release`继续表示原科学源，另有`execution_repair`及完整来源区分I/O修复。

新矩阵进程2705558，A6000 GPU1；启动时GPU1使用量0，实际主存/磁盘及20%余量
检查通过。GPU4当时已有其他计算，保持不动。后续仍按原实际RSS/显存80%门槛执行。

## 验证与有限衔接

本地及Linux各5项回归通过：两类输出命名、物理开关一致性、32行/尾块的
float64标定后FP32转换逐值一致、非法值拒绝。合成数组仅验证整理算式，
不冒充新物理模型验证；完整矩阵数值和重建检查仍由正式流程执行。

首次部署在SSH创建之前触发Windows命令行长度限制，没有启动远程计算。
`../launch_recovery_20261116.py`改用SSH stdin传入部署程序，冻结源码不变；
证据`repair_launch_transport_20261116.json`。

新`finish_repair.py`仅替换旧handoff内存中的矩阵状态提供函数，以新补缺输出
作为矩阵来源；原handoff源文件不改。原12个绑定文件及修复代码、修复manifest、
新登记共同SHA绑定。当前有限控制器PID2836，完整身份以generated中的
`completion_repair_registration_20261116.json`为准。仍保留原
2026-11-22T15:33:42.668790Z截止期限，不恢复周期定时器。

矩阵成功且进程完全退出→重新检查超算连接/实际配额→严格流式传输→提交唯一
8节点×1普通5090任务（440 SC、440 JSCC、218 SC，各10000/save50）→成功后
严格取回600帧。科学失败、部分传输、不确定提交均停止保留，不自动重投。
控制器存活期间不得手动并行submit/fetch/transfer或修改绑定代码。

只读状态命令：

```powershell
python -B -X utf8 experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/repair_20261116/repair_workflow.py status
```

截至本次修复，正式重建尚未提交，不宣称新图像已完成。
