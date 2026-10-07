# 执行入口与有界推进

工作目录为仓库根。`ehe_5e9_workflow.py`提供`prepare/deploy/pilots/advance/status/fetch/compare`。

1. `prepare`只执行一次：检查1669255验收、真值SHA、新种子冲突；生成200个worker注册和20份宏；独立EHE源overlay、三套Params、whole geometry和代码冻结。
2. `deploy --host maty/gpu`逐文件SHA校验。模拟主机使用既有Geant4 11.1和GCC12.2、CMake3.25.2；DPAPI连接重建主机，凭证仅内存解密。
3. `pilots`提交独立100000初级gamma吞吐试跑及逐孔/bin几何核对，GPU上编译PE-v4/Scatter并实际测试三种响应各一完整XY层。
4. `advance`读取已登记作业，实际成功退出后按实测吞吐计算运行上限，唯一提交200个worker的5e9输运和三套完整响应。输运数组先限制40并发；只等待，不取消其他项目。阶段登记存在时不重复提交。
5. 全部200worker成功退出，核验6份窗/初级能量分解CSV、primary总数/能量份额/独立种子/源码宏SHA，保存原观测并上传GPU主机。三套完整响应与自身S通过后执行真实源物理审计。
6. 物理无HOLD后唯一提交完整输入validation10；退出后严格fetch，形成实际validation authority，再唯一formal200。
7. 正式完全退出后严格fetch，`compare`生成50/100/150/200主图和JSCC2000/10000参考、全部20帧EHE指标及既有JSCC曲线。图像生成完成仍需科学/视觉QA、文字报告、SHA清单与安全Git审计推送。

正常排队/运行不重复提交，不因尖峰提前停算或调参数。故障先读queue/accounting/logs/failure；旧进程完全退出后，保留其不可变发布和证明。有效完整响应slab有SHA receipt，可只读复用；部分失败目录留存，不自动覆盖。补缺失worker/响应/求解阶段需要登记新修复发布和已经完成阶段的严格SHA及算法身份，不能仅以存在一个末图为依据。当前通用推进器对未完成的部分目录报错停留，避免未经证实地续跑。

所有新增远端数据限定在`generated/ehe_spect_5e9_200`。原输入、Factors、JSCC1669255和旧证明只读。禁止显式mem/mem-per-cpu/mem-per-gpu；只排除已确认故障节点wqd10nba06g6，不取消其他项目；同名活动作业或既有登记阻止重复提交。

初始构建尝试因默认CMake2.8.12.2不支持当前构建入口而退出；已改用主机现有tools/cmake/v3.25.2模块。尚未以这次失败构建启动正式输运。

首次输运15633325在光子发射前因CPU无内存TRES而失败，已完整退出并保留小证据；`repair-transport-pilot`只允许修复这个已诊断启动故障，不触碰正在运行的GPU任务。首次响应1672792在PE-v4拒绝1250孔处失败，尚未产生完成的矩阵；`repair-response-pilot`冻结独立有限圆孔适配和GPU/穷举数值检查，只补响应试跑。两项修复不取消其他项目，不覆盖旧失败输出。

2026-10-07后续真实诊断：15633411停在随机布尔表面重叠检查，15633583停在首轮导航的`G4MultiUnion::InsideWithExclusion`。两次本实验试跑已取消并完全退出，原发布/日志/调试栈保留。Geant4 11.1.0[官方源码](https://raw.githubusercontent.com/Geant4/geant4/v11.1.0/source/geometry/solids/Boolean/src/G4MultiUnion.cc)在空surface列表下的`size_t(size-1)`循环上界会下溢。独立EHE可执行文件内仅把此条件改成`i+1<size`，版本编译门限定1110；系统Geant4安装和JSCC程序均不修改。保留原盒减1250圆柱并集的几何，逐1250孔轴/近切线/域外点共11252处与原`InsideNoVoxels`穷举算法核对。解析尺寸/孔间距/接触面及逐孔/bin Params核验继续保留；禁用冗余随机表面采样。初始化与beam时间分开记录，不能把冷启动按250倍外推。最新CPU冻结`617de71eaed668cd`、试跑15633652，以最新登记状态为准。

GPU试跑1672825的圆孔弦长检查失败，未生成完整响应；诊断发现混合float/double min/max误选了host constexpr重载。独立派生代码改用device double区间比较。最新GPU冻结`74e129c4460163c5`、1672873的5798条射线/全部1250孔数值检查实际通过：最大弦长误差7.09165e-5mm、L2=1.97091e-7，门槛未变。这只证明几何数值，不替代5e9真实计数物理核验。三套响应吞吐试跑仍在执行，正式全量响应尚未提交。`pe_geometry_actual.json`保存实际小证明。

阶段资源最终以实际Slurm MaxRSS/AllocTRES和运行GPU/RSS峰值闭合，保存`*_resource_acceptance.json`。正常排队不重复提交。任何新失败都先保留旧证据，彻底退出后仅修复缺失阶段。数据、矩阵、响应块、压缩包、构建二进制与凭证不进入Git。

1672873已完全COMPLETED/0:0，三个完整XY层的PE/Scatter试跑和资源验收通过，完整响应唯一作业1672966已运行。矩阵吞吐外推约18.90小时，实际进度以receipt/log为准；此时未完成完整5e9输运或成像。严格取回使用单独冻结的只读验收发布，对所有20视角的完整原算子核对S/转置和本次440末图串窗预测，不修改执行发布。原Windows登记宏与Linux实际宏保留双方SHA，只允许CRLF→LF换行转换，任意源命令改动均拒绝。

`ehe_reference_evidence.py`只读统计既有JSCC的两窗观测和三套自身单光子S：每套完整矩阵一次顺序读取，同时SHA核验同一批用于行求和的字节，只写本次EHE evidence，不改原Factors。比较图报告体积归一后的探测概率与密度响应S的分布。旧JSCC没有初级能量标记的窗计数，不能捏造实测串窗比例；对照只列已交付440最终10000图的模型预测背景及其与218实际窗计数之比，并明确标注估计预算。
