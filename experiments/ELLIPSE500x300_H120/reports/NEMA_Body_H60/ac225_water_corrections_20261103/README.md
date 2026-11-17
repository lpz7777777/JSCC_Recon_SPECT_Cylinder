# 带水NEMA：225Ac 单光子AC/能窗散射修正

**2026-11-05 完成子集可视化：** 14路440 SC的完整2800检查点与背景均已严格取回，原数据保留；[最终图、迭代图、BV–CNR和未截高值伴图](../../../../../Analysis_Plots/research/ac225_water_completed_20261105/REPORT.html)。JSCC探头5e10两路散射修正仍未纳入、218未展示；完整32路实验尚未交付。当前有限衔接身份见one_shot_current.json的r11，GPU任务不变。

**2026-11-04 17:42 缓存/依赖修复后当前正式入口：** 4路健康440继续；新440补缺数组1772573（12路）、新218数组1772574（16路）已提交。旧27个PENDING元素已取消，失败_1与成功输出保留。编译缓存改为作业内私有/tmp，科学冻结不变；218等待全部有效440成功并读取同分支最终图。详见[缓存修复与有效映射](cache_repair_20261104/README.md)、[formal_current.json](formal_current.json)。以下早期叙述为历史，各阶段状态以当前入口为准。完整带水实验尚未交付。

**ACTIVE，完整实验尚未交付。** 旧真空数据、A/C矩阵、Compton与联合结果保持原路径和字节。

**2026-11-04修复已提交并实际运行。** 详见[修复记录](repair_20261104/README.md)。有效侧窗数组为1768730；四组有效收集为16280166、16280167、16280171、16280172。旧慢速1764383和依赖1764530已全部退出，成功窗口/原始响应已绑定复用；健康EHE 5e10输运16266031继续。8个EHE修复分块已完成，JSCC分块正在计算，约305MB输出的转换+顺序写盘实测3.6–4.8秒，旧方案同尺寸文件从原始响应完成到窗口回执约7.4–7.6小时；两者不是全流程耗时比较。新衔接身份见`one_shot_r5_registration.json`，不是恢复旧定时器。

2026-11-03晚间只读核对：两组5e9输运数组均20/20 COMPLETED，但收集16265943、16266030均在严格会计校验处FAILED/1:0；一次性衔接进程87956已于16:43停止，登记为`FAILED_STOPPED`，不能继续描述为活跃。已提交的其他计算仍运行，成功输运数据保留，不重复模拟。约21:25的会计快照中，JSCC 5e10为126/200完成、64运行，EHE 5e10为64/200完成、64运行。EHE侧窗首批4个分块各完成`218_lower`，当前处理`218_upper`；JSCC补缺仍Dependency。正式重建未提交。失败原始日志见`collector_failure_logs_*.json`，GPU时间证据见`gpu_eta_snapshot_20261103_212250.json`；原始收集校验未通过，不能据输运退出状态宣称严格交付成功。

[代码与运行说明](../../../ac225_corrections/README.md) · [新研究合同](../../../generated/ac225_water_corrections_20261103/payload/contract.json) · [结果执行状态](../../../../../Analysis_Plots/research/ac225_water_corrections_20261103/STATUS.html)。完整交付后生成同目录`REPORT.html`。

## 已登记正式任务

| 探头 | 总gamma | 输运数组 | 成功依赖收集 |
|---|---:|---:|---:|
| JSCC | 5e9 | 16265923 | 16265943 |
| JSCC | 5e10 | 16265944 | 16266009 |
| EHE | 5e9 | 16266010 | 16266030 |
| EHE | 5e10 | 16266031 | 16266097 |

每组同时含218/440，不是225Ac衰变链预算。5e9每组20worker、5e10每组200worker；各worker真实250M发射、独立新种子。

侧窗正式矩阵数组 **1764383**：16个源能量/探头/z分块，各生成3个侧窗，最多4张普通5090，每元素1卡8CPU。原1764376在所有元素尚PENDING时取消，仅替换准确读取各数组元素AllocTRES的启动设置；[原始记录](side_pending_launch_repair.json)保留，科学代码和旧发布未改。

1764383实际启动后，JSCC_0..7在产生PE值前因`EHE spatial grid capacity exceeded`退出：EHE专用孔索引程序不适用JSCC的500mm无孔参数。EHE_8..15继续运行；仅8个失败JSCC分块采用原无孔PE-v4程序独立编译补缺为 **1764530**，原目录/日志保留。[修复冻结](JSCC_side_repair_freeze.json)与`side_effective_jobs.json`记录补缺身份。补缺依赖整个1764383退出，保持最多4GPU。原一次性进程32264发现错误后停止；新衔接身份见`one_shot_current.json`。

共享盘的小文件元数据等待导致初次部署超时，已保留未完成部署证据。修复源和编译后的二进制合成单个严格SHA包；正式元素只将数MB代码解包到自己的临时目录，矩阵/响应/检查点继续写持久目录。临时代码不作跨作业缓存，与旧/tmp响应持久性方案不同。`JSCC_side_repair_bundle.json`绑定源包和二进制SHA。

32路正式重建尚待带水投影、侧窗矩阵成功退出并严格交付。实际提交以`formal_submitted.json`为准。不能把排队、代码部署或局部检查当成重建完成。

## 检查证据

- [输运冻结](freeze.json)：79a78bf7c0a6a287。
- [侧窗/几何冻结](kernel_freeze.json)：4d39dbed9b03a7e0。
- [执行冻结](execution_freeze.json)：c13b63c15fa6baa6；独立补充，不覆盖前两版。
- [两探头Geant4准备检查](cpu_preparation_checks.json)：各10000事件、同种子和视角，关闭修正及仅开启观测时，主窗与JSCC原List均和原程序逐字节相同。带水CSG/旋转与全部标签闭合通过；不是完整采集的物理验收。
- [本地/Linux数值检查](local_numerical_checks_final.json)：12项均在两端通过，包括完整响应界的随机参考检查。
- [PE代表点](../../../generated/ac225_water_corrections_20261103/kernel_payload/center_ray_audit.json)：加权L2最大约0.161%。
- [局部端点范围](center_endpoint_audit.json)：个别代表点端点透过率最大差约3.09%，如实保留。
- [完整A/C的代表点保守误差界](center_full_response_audit.json)：包括现有A/C的全部非负分量，同一预先选定点集；L2上界最大约1.115%、总响应上界最大约0.714%，低于2%。它约束直接源到晶体端点的近似，不能认证探头多次散射的实际轨迹或整个FOV。
- [材料/μ图](../../../generated/ac225_water_corrections_20261103/material_maps/material_manifest.json)：原活动真值SHA不变，冷球/肺仍为水。

输运首轮CMake误选系统GCC4.8的失败日志和独立GCC12构建目录均保留。没有提交验证/验收Slurm任务，没有恢复旧定时器或改动1761195等其他研究。

## 后续交付

独立一次性衔接进程按`one_shot_current.json`指向的登记记录PID、create_time、cmdline；进程活跃时禁止手动并行推进。它只等待本研究、严格传输新观测、提交32路正式SC、全量取回6400检查点及背景并生成图表。遇失败、SHA不符、部分传输、资源不足或不确定提交立即停止，保留输出，不自动重投。

结果图采用原3D球和统一背景、中心72mm MIP、完整XY、无平滑/拟合/跨能量求和。物理误差、低计数失效和原C偏差分别报告。图表自动生成后仍需实际视觉复核；`physical_calibration_passed`不会由数值通过自动置true。
