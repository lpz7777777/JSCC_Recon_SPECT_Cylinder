# 执行入口与有界推进

2026-10-09 07:21上海：新的本地CPU晶体对筛选复算与标量归一/条件累计能量分析已完成，见physical_scatter_diagnostic_acceptance.json、physical_pair_pruning_read_only.json、physical_scatter_primitives_read_only.json。筛选排除数全为0，标量归一差异约0.0015%；条件多次Compton算例不提供实测缺失机制份额，原HOLD未解除。不得重复这两项已完成诊断当作推进，也不得据此重提physical或提交validation10/formal200。后续只继续有新依据的有限积分、准直器筛选或累计散射历史覆盖只读诊断，不实现下一种核或改冻结模型。

2026-10-09 06:59上海：源/窗身份只读核对与完整三Cartesian的源盒平均敏感性分析已完成并严格SHA取回，原科学HOLD未解除。physical_cuboid_diagnostic_registration.json已complete/exit0；read_only_source_fold_1b9a43962af5c315为CPU只读诊断新目录，不是新的物理/重建作业。不要重复执行/取回该诊断，不将其作为GPU资源或物理PASS。现有插值模型内的C平均变化仅−0.004428%，尚不能解释−24.02%偏差；具体响应原因仍待有依据的只读诊断，参见PHYSICAL_HOLD.md。

2026-10-09 06:17上海：physical1677211真实科学HOLD，原JSON/CSV/日志与完整退出证据已经严格取回，见PHYSICAL_HOLD.md及physical_hold_fetch_acceptance.json/physical_hold_diagnosis.json。不要将FAILED/1:0当成可重提的计算故障，不重复advance旧计算阶段、重新fetch已验收阶段或提交重建；先读HOLD证据，仅做有依据的只读诊断，原门槛、源、模型及所有输出保持。

2026-10-09 04:15上海：完整转换1677140已成功退出，三Factor身份/完整输出SHA/实际Slurm资源验收通过，后续物理门控唯一作业1677211。以physical_job.json及随后physical_gate.json为准；原响应1672966人工取消与新转换成功分别留证，禁止重启原模拟或PE/Scatter阶段。

工作目录为仓库根。`ehe_5e9_workflow.py`提供`prepare/deploy/pilots/advance/status/fetch/compare`。

2026-10-09人工授权的转换恢复：1672966已完整退出且原发布/12份slab收据/两份完整Factor清单/日志留存，见`response_stop_acceptance.json`。`repair-response-conversion`只创建独立`conversion_releases/<key>`，不重新prepare/deploy原科学发布，不调用PE/Scatter或输运。它实际逐SHA核验所有原slab成员、二进制、原发布和A218/A440全矩阵/自身S，然后用真实C440→218第一块测量转换和973MB连续写出/读回吞吐；完整转换时复用这10层，只补其余30层的存储转换。部分原文件保持只读，新目的目录存在任何部分输出即拒绝覆盖。

新输出在`responses_conversion_<key>`；A218/A440只读引用原已完整SHA核验目录，C440→218由RAM连续发布，完整Cartesian/Polar/全20视角自身S/Params/坐标旋转体积保持原身份。`advance`优先读取`response_conversion_probe_job.json`、`response_conversion_job.json`和冻结绑定，实际成功退出与完整证据后才提交原物理门控。新成功转换的资源证书与原12块的计算资源及人工停止记录合并，不能把原CANCELLED解释为整阶段COMPLETED。重建和严格fetch使用同一登记的新响应目录，科学代码仍读取原74e129c4460163c5发布。

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

15633652的实际栈和nm证明CMake采用显式EHE_SOURCES，新增修复源未加入编译目标，因而审计仍调用安装版旧函数。已保留证据并取消本次自己的这项初始化停滞试跑（未开始beam），完全退出后只修复CPU构建接线。发布fdfc7b4a75399b55实际编译src/ehe_G4MultiUnion_11_1.cc且本地text符号检查通过，唯一新试跑15633832。构建符号检查加入deploy，源码列表加入本地合同测试，不能把源码存在或编译退出码当成链接证明。GPU完整响应不重复，系统Geant4和JSCC不修改。

2026-10-08全量启动登记：15633832实际COMPLETED/0:0，11252点分类审计、三套逐孔/bin比较、全部receipt/二进制/冻结文件及原始几何证据SHA核验通过。100000初级gamma试跑的窗计数为0，尚不支持物理效率结论。按实测初始化0.302589秒和beam22.7152秒计算单worker10523秒硬限、181分钟Slurm时限，已唯一提交全量数组15633840（0–199%40）。响应1672966继续运行。不得重复prepare/deploy/pilots或这两项正式输入阶段。

2026-10-08 08:44上海完整输运验收：15633840的200个worker全部成功退出，实际5e9/20视角/种子31100101–31100300及六种标记窗CSV闭合；collection、原source_registry和原/实际宏已逐文件取回并同步到GPU主机。`verify_ehe_transport.py`进一步只读核验两台主机的全部SHA、CPU发布/二进制、所有worker的11252点分类与宏仅CRLF→LF、实际会计退出及能量份额，保存`transport_identity_acceptance.json`和`transport_measurement.json`。不改变执行发布、不重算输运、不执行物理校准。当前1672966仍运行；下一步仅等待完整三响应退出，完成自身S/资源/身份验收，再进入原门槛物理门控。PID24336推进器已正常退出，生成目录下`bounded_advance.json`已标完成；不得重复推进/上传已通过的输运阶段。
