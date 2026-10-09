# 本次实际运行簿

2026-10-09人类最新指令要求停止定时任务并研究解决问题。应用automation_update已实际将compton-v5设为PAUSED。新增RESOLUTION_RESEARCH.md整理冻结局部/晶体间/准直器分支与输运事件/bin累计沉积可观测量的对应，记录未加窗PE生产链、待执行互斥历史标签诊断、分支验证和有界修复/原门控恢复条件。研究不构成已确证根因或科学PASS；没有新GPU/Slurm/模拟/响应/物理/验证/正式作业，原22 HOLD、逐binUNDETERMINED和gate SHA保持。不恢复定时任务或自动advance；新诊断输运/新响应模型等待明确执行授权。

2026-10-09 09:48上海：diagnose_ehe_intercrystal_path_geometry.py实际本地CPU退出0/0.813秒，12份Params/停止前收据/缓存日志和执行代码/结果/完整节点表SHA闭合。原日志为NaI=2312、单层网格路径生成；表面函数沿用固定中心中间段。2311中心段和7042面节点分别对全部2312箱体作独立有限段交集，最大闭合差5.684e−14mm。固定表达式减方向特定总NaI段：近−4.889266至+0.263995mm、远−1.978003至0mm；没有能量、衰减、窗/吸收或源权重，非整体响应误差。见physical_intercrystal_path_geometry_acceptance.json。原HOLD/生产输入/所有数据保持，无GPU/Slurm/响应/重建作业，根因仍UNDETERMINED。

2026-10-09 09:19上海：新的仅几何表面固体角诊断实际退出0/0.891秒，接受12份Params/停止前收据与原运行配置SHA，见physical_surface_solid_angle_acceptance.json。2311类别覆盖5343032对，4522面解析/独立参考最大相对差8.631e−11。近/远每对固体角有符号误差范围−0.143%至+0.265%/+0.012%至+3.831%，其余被积因子保持未评价；不属于完整响应误差界。原HOLD、现有数据和输入保持，无GPU/Slurm/响应/重建新作业，根因仍UNDETERMINED。

2026-10-09 08:47（上海）新增原运行散射摘要与表面采样几何分析：diagnose_ehe_scatter_runtime.py本地CPU实际退出0、0.454秒，复用已缓存原12份scatter日志，没有远端取回、响应矩阵读取、面积/筛选/源盒诊断重算或GPU/Slurm/核评价。每份日志及冻结Detector Params绑定停止前receipt；全部运行记录晶体范围[0,2312)、局部/准直器分量包含、独立分量矩阵关闭，C局部self-Compton-PE开关为0、recoil为1。源码确认日志crystal和collimator为整块2312×85×85×10原Cartesian数组逐元素总和，crystal已包含局部与晶体间项；不是PE+Scatter的完整Factors或真实源前向计数。四块未加权准直器份额：A218=2.43209e−16、A440=0.0808651432、C=0.000231156179；原打印有效位舍入区间已传播，不能外推实测份额。日志记录晶体间近8×8/远1×1表面采样、近距离因子2。三响应/12块几何一致，CPU float32/float64新分类均近167548对（3.135823%）、远5175484对，20mm边界23964对按≤进入近分支。未再执行原5σ支持筛选；几何对数量不是贡献权重、CUDA逐对运行状态或误差界。源码/代码/原日志/Params/结果SHA闭合见physical_scatter_runtime_acceptance.json。实际根因仍UNDETERMINED，原22 HOLD、138720逐binUNDETERMINED和gate SHA保持，未提交physical/validation10/formal200。

2026-10-09 08:19（上海）新增Pb面积权重诊断：diagnose_ehe_collimator_area.py本地CPU实际退出0、0.344秒。新读取原12份scatter日志，其原字节SHA逐一绑定停止前receipt；没有再次验收或计算响应阶段。全部记录1250格、48324.4mm²。冻结默认50×25格、8×8子网格的CPU float32参考面积48324.372372mm²，与日志打印精度一致；未捕获继承环境或运行时逐格数组，不能声称逐位生产状态。全部1250圆孔互不相交/在板内，按圆盘截弦原函数逐格解析积分得Pb面积48314.076848mm²，逐孔面积守恒最大差8.88e−16mm²，另4个独立Gauss积分比较最大差1.5253e−7mm²（数值检查，非严格误差界）。总量误差+0.0213096%，逐格绝对面积误差之和/解析面积1.013804%，逐格最大相对误差7.515904%。固定非负逐格贡献时，单独替换面积权重的误差界为7.515904%（解析总计作分母）或6.990505%（采样总计作分母）；未计算散射贡献、生成响应或把该界限外推到完整物理模型。16/32子网格仅CPU参考诊断，不改生产采样；逐格最大相对误差2.072793%/0.733974%。代码/源码/Params/12日志及收据SHA闭合见physical_collimator_area_acceptance.json。原22 HOLD、138720逐binUNDETERMINED和gate SHA保持，根因仍UNDETERMINED；不重复本面积检查当作推进，也不提交physical/validation10/formal200。

2026-10-09 07:47（上海）新增有限孔准直器路径条件分析：diagnose_ehe_collimator_paths.py本地CPU实际退出0、4.875秒，读取冻结Params/XCOM/源码及原3mm真值，不读取响应矩阵、不运行GPU/Slurm/模拟/PE/Scatter。生产准直器路径使用层中心几何/能窗及完整50.5mm均匀Pb板的深度因子，孔只通过面积权重进入该因子。以确定性正密度源点/中央孔625/NaI bin2295构造4个后表面附近的条件散射点，8条射线各计算完整盒减全1250个有限孔交集并集，另各40000中点独立成员检查，误差均在中点离散界内。首个入射段盒/Pb长50.250489/3.490691mm，出射段6.253630/3.755637mm，条件两段仅Pb存活因子0.0372514对完整盒4.12584e−7；这是未加权条件路径，不是生产响应修正或测得机制份额。相同x/z层中心能量264.259311keV/条件窗概率0.112567，距后表面0.25mm为241.213104keV/0.458525；条件点均未被2σ筛选排除。源码/代码/输入/输出SHA及实际退出闭合见physical_collimator_path_acceptance.json。原HOLD、输入和全部已有结果保持，根因仍UNDETERMINED；不要重复该已完成条件分析冒充推进，后续仅有新依据的只读定量诊断，不改冻结模型或提交重建。

2026-10-09 07:21（上海）新增只读筛选/归一/历史覆盖诊断：diagnose_ehe_pair_pruning.py本地CPU实际退出0、10.609秒，12种响应/10层块Params逐一绑定原保存收据，float32/float64各复算每种5343032个有序非自身晶体对；5σ筛选排除数全为0，最小保留余量41.9670keV，C最小151.0987keV。没有生成响应或采集CUDA运行标志。diagnose_ehe_scatter_primitives.py本地CPU实际退出0，原0.01rad角度归一分母与256/512点cosθ Gauss参考的218/440偏差为−0.0011147%/−0.0014842%，不是全有限积分误差界。条件440→340→222keV两次Compton算例在同一bin沉积100+118=218keV且末光子逃逸时，原窗接受概率参考为0.760968；局部第二Compton分区未计累计反冲窗项，但该算例没有本次测量路径或全局机制份额。完整Scatter还含晶体间/准直器项，具体24.02%偏差根因仍UNDETERMINED。新代码/输出SHA闭合与生产dispatch核对见physical_scatter_diagnostic_acceptance.json。原physical_gate SHA、22项HOLD、138720逐binUNDETERMINED保持；未提交任何GPU/Slurm/物理/验证/正式作业，不重复执行这两项已完成分析冒充推进。

2026-10-09 06:59（上海）新增只读源诊断：diagnose_ehe_source_window.py逐一展开原20视角注册源盒，并核对原旋转、Cartesian坐标和全部2312-bin窗/分辨率。218/440密度相对L2为1.1732e−16/8.1451e−17，插值旋转质心最大误差2.6483e−8mm；窗边界float32/float64最大差1.5871e−5keV，未发现明显配置误配。证明为physical_source_window_basis_audit.json，不代表物理通过。

CPU只读诊断发布1b9a43962af5c315在本实验新目录执行，无GPU/Slurm作业或PE/Scatter/输运重算；全部三Cartesian的2312行/40层在敏感性分析所需读取中核对同批SHA。原20视角中心预测复核最大相对差4.1806e−8；3mm源盒XY采用4×4/8×8积分、z精确分段线性平均，A218/A440/C全局预测变化分别−0.001744%/−0.003804%/−0.004428%。C源盒平均预测7497.844371对9869仍低估24.0263%，现有插值模型内该效应不能解释约24%偏差。两组积分逐视角最大相对差1.6542e−7，不是严格积分误差界，也不排除响应网格/历史覆盖误差。实际退出0、119.231072秒、CPU RSS143282176字节；仅CPU诊断，不能作为成像资源证书。代码/binding/结果/原日志取回前后及本地SHA闭合，见physical_cuboid_read_only_acceptance.json及physical_cuboid_read_only目录。注册physical_cuboid_diagnostic_registration.json已complete/exit0，PID16828已实际退出，不重复启动或取回本分析。原22项HOLD、138720逐binUNDETERMINED及所有生产输入保持；下一步只做有新依据的响应积分/散射覆盖只读诊断。

2026-10-09 06:17（上海）科学HOLD：物理作业1677211于05:46:29–05:52:37实际运行06:08，root/batch FAILED/1:0、extern COMPLETED/0:0并完全退出，异常明确为Physical response HOLD: 22 diagnostics。原JSON/完整CSV/日志取回前后及本地SHA一致，3mm源/原worker/科学执行代码/三Factor清单身份闭合。138783行全部观测与worker标准误差、原充分性/双门槛由verify_ehe_physical_hold.py只读独立复核；22项HOLD、138720逐binUNDETERMINED。C全局7498.176418对9869，−24.0229%/19.6482SE，19视角和全局HOLD；A218全局+3.1693%，视角13/14 HOLD；A440全局−1.8065%且各视角总计通过。Slurm MaxRSS13629532K/实际94500MiB为14.0848%，运行RSS6.2407%/GPU reserved8.0102%；不是OOM/超时。validation10/formal200未提交。已保留HOLD并做冻结源只读审计，具体物理偏差原因仍UNDETERMINED；不追加光子、调整窗/阈值/增益、重跑模拟/响应或开始下一种核。详见PHYSICAL_HOLD.md。

2026-10-09 04:15（上海）完整转换交付：1677140 root/batch/extern实际COMPLETED/0:0，elapsed14:35；转换程序实际853.349197秒。全部12块和原两套完整Factor再次全成员SHA核验，源复用证明SHA仍为ed6a1c86b4aea011fe3cd14b550e186abf7a30387d193c0c38ca982857dbf4a3，与真实探针一致。复用已验收的10层，仅转换剩余30层；40层C440→218完整Cartesian/Polar、S_full、20视角自身S_active、Params、whole geometry及Factor清单均完整生成。A218/A440清单SHA逐字与停止前相同，科学发布74e129c4460163c5及原计算块保持原身份，旧部分C文件继续留存。

2672672000字节Cartesian连续原子写出、fsync及完整读回SHA51.530601秒，1221105920字节Polar同流程21.123843秒；共3893777920字节/72.654444秒。文件SHA分别为8b15b804e0101636acade419eab5de9649da7c13f94d3ed7f9db3ab452c55059与e008242786cd1028837eae4b0a270aed63724208a09db201a3ca6a19281917b3。实际进程RSS4599107584/99090432000为4.6413%；退出后Slurm MaxRSS46647504896/实际94500MiB为47.0757%，均保留20%余量。这是端到端发布及读回耗时，不是共享盘硬件带宽测量。

三个完整Factor清单、原子输出SHA、原Params/geometry身份、有限正自身S范围、完整层/bin/点/view及科学代码身份通过，见response_conversion_identity_acceptance.json；原计算CANCELLED记录与新完整转换成功资源分开保留在response_resource_acceptance.json。真实源物理门控1677211已唯一提交，当前Priority排队；继续原预登记100观测、10%且3SE门槛，不把转换PASS解释为物理校准PASS。尚未提交validation10或formal200，不生成EHE图像科学结论。

2026-10-09 03:38（上海）实际转换探针通过：1677092 root/batch/extern均COMPLETED/0:0，elapsed14:59。全部12块成员文件、两套完整Factor和原发布/实际二进制全SHA通过，并由本地协调器再次逐一绑定停止前保存的12份收据及2份Factor清单SHA。真实C第一块10层/2312bin/33010点与原完整bin插值表达式逐值相等，L2=0；内存转换6.808980秒，668168000字节Cartesian的连续写出/fsync/读回SHA14.355225秒，305276480字节Polar同流程5.915407秒，总实际阶段893.318474秒，其中源数据完整SHA审计848.877507秒。此实测支持顺序发布修复了原跨行小写入模式，但不据此声称共享盘硬件吞吐或整套实验已完成。

探针实际进程RSS峰4344553472字节/实际99090432000为4.3844%，退出后Slurm MaxRSS43976351744字节/同一实际AllocTRES为44.3800%，均保留20%余量。完整转换1677140已唯一登记，39分钟Slurm界限来自实际探针审计/转换/写入时间加有界余量，当前Priority排队；复用探针10层，补其余30层，只读复用原A218/A440。预计启动后约15–20分钟完成转换及完整验收，队列和共享盘波动另计；最终响应、物理门控与成像结论仍待实际证据。收据见response_conversion_probe_acceptance.json、response_compute_reuse_acceptance.json及response_conversion_probe_resource_acceptance.json。

2026-10-09 03:29（上海）：转换探针1677092已结束Priority排队，在wqd10nbj06g4实际RUNNING/0:0，elapsed7:56。日志已逐一确认A218四块和A440前三块成员文件SHA通过，仍在完整源数据验收；数值与连续写入测速尚未结束，不能宣布修复吞吐已通过。原作业1672966不再运行，所有计算结果保持原目录。

2026-10-09 03:18（上海）转换恢复：用户明确要求保留已有结果、考虑停止当前作业并解决共享盘瓶颈。03:00前的只读证据确认当前C440→218映射slab_1（第二个10层块），FUSE页写回等待，输出位于`fuse.juicefs`；原转换每次跨3301行写64列、每行256字节，与9248字节行步长不匹配。反复触碰文件页是明确的不利写入模式；是否另有存储集群拥堵/限速未证实，不据此推断硬件故障。

已保存12份完成slab收据、A218/A440完整Factor清单、原job登记、scontrol及日志，再仅取消1672966。原作业最终无queue项；root为CANCELLED by 1721/0:0，batch为CANCELLED/0:15，extern为COMPLETED/0:0，实际elapsed27:11:12。源计算Slurm MaxRSS25972592K/94500MiB为26.8401%。[停止小证明](response_stop_acceptance.json)记录其人工停止、完全退出和证据SHA，不声称整个响应阶段成功。原输出目录、部分C转换文件、科学发布74e129c4460163c5及5e9输运均保留。

新转换发布b02df58a9d3fd48c独立冻结，仅包含I/O转换实现、原公共资源/原子写入函数和人工停止证明。唯一探针1677092已登记，当前Priority排队；它将逐一核验原12块全部receipt成员文件、实际PE/Scatter二进制和原发布SHA，以及A218/A440完整Cartesian/Polar/自身S/Params/whole geometry SHA。用真实C第一块的全10层、2312bin、33010点核对原插值表达式逐值一致，再连续写出973444480字节Cartesian+Polar探针数据、fsync及独立读回SHA，形成实际吞吐和资源证明。这不是新的PE/Scatter响应试跑。探针退出后完整转换复用该10层，只补剩余30层，内存处理后连续发布；新目录与旧部分输出分开，存在部分新输出则停报。

7项转换/输出字节/不覆盖/路径绑定/停止前SHA绑定测试、原10项科学合同测试及7项状态测试全部实际通过；advance实际退出0并等待最新转换登记。它们不替代真实探针或full-input validation10。新恢复流程的停止/代码/探针登记见response_stop_acceptance.json、response_conversion_freeze.json、response_conversion_probe_job.json和response_conversion_local_acceptance.json；完整响应/物理门控/重建仍待实际证据。

2026-10-08 23:26（上海）更新：完整输运已验收；1672966实际仍RUNNING/0:0，Slurm elapsed23:33:20。12个PE/Scatter完整块已计算完成，A218/A440完整Cartesian/Polar/S/manifest已生成。C440→218的进程当前仍映射slab_0/response.sysmat，正在第一个10层块转换与共享盘写回，尚无自身S/manifest及response_summary。write_bytes从11:01UTC的25711796224增加到15:26UTC的28203347968，但写回计数并非剩余逻辑矩阵字节数，不能据此外推可靠结束时间。Slurm MaxRSS25972592K/实际94500MiB为26.8401%，尚未取得最终退出后的资源证书；物理门控、validation10和formal200尚未开始。

### 只读诊断步骤与本地状态判断修复

23:16上海的只读/proc smaps/stack检查因Python字符串换行语法错误在解析阶段退出，对应1672966.102/FAILED/1:0，elapsed0秒；更正后的同范围只读检查1672966.103实际COMPLETED/0:0。主作业与batch保持RUNNING/0:0，主响应进程没有取消或重启。原失败命令字节及实际完整Slurm记账分别保存在diagnostic_step_1672966_102_command.txt和diagnostic_step_1672966_102_accounting.txt，SHA钉在diagnostic_step_1672966_acceptance.json。

原completed检查把任何失败辅助步骤当成主阶段失败。仅修改本地协调器：读取有SHA闭合的小诊断证明，只排除精确的1672966.102/FAILED/1:0；其他未登记数值步骤、主作业、batch、extern和200个worker的失败仍阻止推进。失败检查先于成功完成判定。7项状态测试实际通过，修复后advance实际退出0，仍未提交下一阶段，见workflow_status_repair_acceptance.json。远端74e129c4460163c5发布清单中全部冻结文件实际SHA一致，没有部署、覆盖生产代码、改变几何/物理/门槛或重复模拟/响应计算。全部原失败记录保留。此前10项科学合同测试不由这7项状态测试替代；完整输入validation10仍待实际运行。

2026-10-08 08:44（上海）：完整EHE 5e9输运已成功退出、严格取回并同步SHA验收。三套完整响应1672966仍运行，真实源物理门控、validation10及formal200尚未完成，尚无EHE成像或物理性能结论。

| 阶段 | 当前登记 | 实际状态与边界 |
|---|---:|---|
| 100000 gamma输运试跑 | 15633832 | 已COMPLETED/0:0，38秒；最新CPU发布fdfc7b4a75399b55、实际修复链接、11252点几何分类及三套逐孔/bin Params核验与原始证据SHA通过 |
| PE-v4/Scatter吞吐试跑 | 1672873 | 最新GPU发布74e129c4460163c5，三响应各一完整XY层，已COMPLETED/0:0；GPU圆孔数值检查与实际资源通过 |
| 完整三套响应 | 1672966 | 同一GPU冻结发布，已RUNNING；12个完整10层slab顺序计算，不重复试跑/提交 |
| 全量5e9输运 | 15633840 | 200个worker全部COMPLETED/0:0，实际5e9；种子31100101–31100300、20视角、6种窗/初级标签及取回/同步SHA闭合；不重算 |
| 物理/validation10/formal200 | 未提交 | 全量输运和三套完整响应实际成功退出后，advance按既定严格门控继续 |
| JSCC对照 | 1669255 | 已完整交付、只读；本轮默认基准验证再次PASS，未改冻结代码/证据/图集 |

GPU几何小证据见`pe_geometry_actual.json`：5798射线/1250孔，最大误差7.091651441e-5mm、L2=1.970908227e-7。完整输运剂量、独立种子、宏内容和初级标签已闭合；真实源前向计数与输运观测的物理门控仍未执行。几何检查不是物理校准。

本次完整输运初级218/440/其他为[1469040802,3530959198,0]，合计5000000000。实际218份额0.2938081604，预期0.29380779868182727，偏差0.0561516个二项标准差。所有200个worker均执行11252点分类审计；各自源宏只允许原Windows CRLF转换为Linux LF，源码发布及二进制身份与通过的15633832试跑一致。

`transport_identity_acceptance.json`保存退出后的完整squeue/sacct及独立SHA审计：本地2424份文件、CPU远端70份发布文件/实际二进制/2200份worker成员文件、GPU同步后的2424份文件均通过；collection原字节SHA为54146b05e864c12963ef6531223ac1cd60d86d86497f408aa26486885caff94a。`verify_ehe_transport.py`为可复核的只读入口，原运行发布和JSCC基准未改。全部原worker CSV、receipt、原注册及实际宏保存在本实验generated数据目录，Git仅保存小验收与测量证据。

| 本次实际窗 | 218初级贡献 | 440初级贡献 | 合计 |
|---|---:|---:|---:|
| 218窗 | 13060 | 9869 | 22929 |
| 440窗 | 0 | 12171 | 12171 |

218窗中实测440串窗占比43.0415631%；逐20视角计数见`transport_measurement.json`。这些是实际标记窗计数，不是图像拟合或响应预测。未匹配JSCC探测计数，物理门控仍待三套完整响应，不据此宣布效率或重建性能通过。

200个worker实际初始化0.300147–0.366133秒、中位0.3258585秒；beam2845.27–3744.09秒、中位3383.785秒，均在10523秒硬限内。CPU内存继续只用开始时MemAvailable作运行保护，不作为成像资源证书。完整响应当前Slurm MaxRSS20816660K/实际94500MiB约21.512%（尚非最终资源验收）；实际资源及完成证据仍须持续检查。本地推进进程PID24336已正常退出，`bounded_advance.json`已标完成，后续不得等待或重复启动该进程。

输运试跑实际100000初级gamma，218/440分类29229/70771；六份窗计数均为0，不能作为灵敏度或串窗物理验收。逐孔/bin几何最大误差6.469726571e-6mm、探测bin误差0，11252点分类审计通过。`transport_pilot_sha_acceptance.json`钉住所有receipt文件、实际可执行文件和原始几何CSV/三套比较报告，原始证据位于`geometry_evidence`。实际初始化0.302589秒、beam22.7152秒：2500万单worker纯beam外推5678.8秒（约1.58小时），采用10523秒进程硬限和181分钟Slurm时限。40并发若连续获配，五批纯beam约7.9小时，排队另计；不以此保证结束时间。CPU无实际内存TRES，使用起始MemAvailable的运行保护，明确不作为GPU成像资源证书。

10项本地合同测试通过：原加权源采样身份、Geant4单一边界修复的官方源SHA、合法迭代合同、实际内存TRES解析、物理双门槛、两种原MLEM保存循环与历史、前向/转置、worker窗/初级标签闭合及源质量旋转，以及独立验收拒绝错误S/串窗背景、宏仅允许CRLF/LF平台换行差异。Python语法/CLI检查通过，比较入口在正式验收缺失时拒绝生成结果。完整输入10次数值验证仍须实际执行。

三套一层响应实际PE/Scatter秒数分别为A218=557.74/64.03，A440=494.15/29.60，440→218=464.70/90.72；完整40层合计吞吐外推约18.90小时，冷启动/转换/排队另计。正式完整响应每个子进程有界11492秒，总walltime2329分钟。试跑Slurm MaxRSS=1130291200字节/实际94500MiB为1.1407%，CUDA进程占用峰值约4.19%或更低；这些是响应试跑资源，不能作为完整矩阵生成或之后成像的证书。

验收脚本单独冻结在verification_releases，运行/完成的响应与模拟源码保持原字节。严格fetch额外逐20视角、全部三响应核对原冻结ActiveGeometry前向/转置及自身S，并从本次440末图重算固定218加性背景，L2门限仍1e-5；保存验证脚本SHA。只读CPU核验使用64探测行分块，不更改求解或充当GPU成像资源证明。

JSCC只读参考量已实际生成：218窗总计12314473、440窗5363190，各20视角原计数与已交付输入SHA匹配；三套完整矩阵用于求和的同一批原字节SHA与原合同匹配，自身S/体积归一概率统计见`jscc_reference_measurement.json`。新统计和小S只写本EHE实验，原JSCC不改。旧JSCC没有初级能量标记窗计数，最终报告须将原44010000模型预测背景和实测串窗分开。

失败与诊断历史均保留，参见PRODUCTION.md和`failed_*`/`stalled_*`小证据；旧试跑不恢复。CPU/响应分别有自己的最新修复冻结，原始freeze.json只是初次发布，不可据此猜当前作业。实际提交记录、receipt、squeue/sacct和日志才是状态依据。

15633652进入几何分类审计后仍停在安装版libG4geometry.so的旧函数，实际nm证明新修复源未列入CMake显式源清单，尚未输运；已取消自己的这项停滞试跑并完全退出，栈/符号/原冻结保留于`unlinked_union_15633652.json`。新CPU发布补齐源清单，并用实际二进制的text符号门控构造器、InsideWithExclusion、InsideNoVoxels，见`transport_link_acceptance.json`。GPU完整响应1672966保持运行，原Geom/采样/物理未改。提交记录和最新冻结比初次准备记录优先。

compton-v5已转为每30分钟推进此EHE实验，保持本聊天、正常排队/推进安静，仅通知故障、关键里程碑和科学结果。全部200次三路结果、图集/曲线、科学与视觉QA、报告与安全Git交付后暂停。原JSCC项目不恢复。

## 后续必须完成

1. 已完成两项实际吞吐试跑、几何/资源/SHA证据和有界全量提交；不得重复提交15633840或1672966。
2. 实际200×25M新EHE输运及种子/视角/标签/取回与同步SHA、完整三响应转换与身份验收均已完成；原1672966人工停止与新转换1677140成功分别留证，不重算或重复转换。
3. 真实源前向计数/串窗与独立worker统计审计已实际HOLD；原证据可信，具体原因仍UNDETERMINED。只做有新依据的只读诊断，不重提1677211或以重算旧证据冒充推进。
4. 仅原物理门控真实解除后，完整输入validation10并严格取回authority，随后唯一formal200/save10；40完整检查点和三路20帧，EHE218背景仅来自本次440最终图。
5. 严格取回，运行真实比较，50/100/150/200主图、JSCC2000/10000参考、全部EHE20帧和JSCC200帧曲线、窗计数/串窗比例/灵敏度/截断、背景估计预算差异。完成科学与视觉QA、报告、SHA清单、审计后的Git推送，再暂停自动推进。
