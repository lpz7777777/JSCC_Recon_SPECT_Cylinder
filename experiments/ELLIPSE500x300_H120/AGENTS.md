# 当前用户约定（2026-10-10）


## 最新实际状态：4090分批方案等待持久缓存空间

1686694已实际全COMPLETED/0:0（3分23秒），24GPU身份、小块无损往返和80%资源余量通过；整体存储门控未通过。6/8节点/tmp空间不足；1686708只读挂载检查证实/tmp位于当前作业jobcontainer，不能据此承诺验证与正式作业之间复用。此前9b5938545ad3d6c7发布中的/tmp固定节点复用方案已被实际证据否决，原发布及日志保留，不能继续该入口advance或提交validation/formal。

完整响应1530759784160字节（约1.53 TB），现有工作盘约66 GB可用。/ssd全卷空闲不等于账户配额：/ssd/scxi717属于root/0700，scxi717不可读写，目录实际配额仅1 GiB；home也只有1 GiB。已请求用户在平台开通可写、至少2 TiB持久SSD目录并提供路径。需要新独立存储修复冻结和实际I/O验证，不能修改旧发布或重新筛选/模拟。

当前例行入口仅 `python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_4090_storage_status.py`，只读核对六项PID、账户SSD配额/访问和原5090自然状态。自动任务继续ACTIVE，资源未变化时保持安静，不反复执行旧probe/取回。原1685272仍PENDING且job JSON SHA保持；未提交完整输入验证或正式重建。详见报告目录 `STREAMING_STORAGE_HOLD.md`、`streaming_4090_8x3_probe_resource_acceptance.json` 与 `streaming_4090_8x3_storage_diagnosis.json`。


- 最新明确授权：完成独立JSCC实际Geant4 5e10及10000次重建，优先8节点×4卡，只保留218 corrected、440 single、440 Compton三路。该实验为 `jscc_geant4_5e10_10000`，独立种子35100101–35101100，1000×5000万实际4π光子；启动检查35100001的100000光子不计入正式剂量。以其 `CONTRACT.md`、最新阶段登记及验收为准。当前实验三路合同优先于下文历史四路默认规则，不计算生产联合图或跨能量相加图。
- 用户最新CPU调度要求：独立worker每作业1节点、1核，可共享节点，数组0–999%1000，完成后统一严格收集。旧18节点15705120未运行且已实际取消；新15708389以最新登记为准。原源、1000worker×50M、种子和Geant4二进制保持，CPU控制修订另冻transport_array_freeze，不覆盖旧发布。GPU仍8节点×4卡。
- 2026-10-10 17:46最新：完整5e10输运15708389/收集15709588已成功退出且严格SHA取回。原GPU4090请求40CPU被每卡6核配额拒绝、未创建selection作业；单独启动控制冻结ce90d678eb32757f保留失败intent/脚本和原科学发布。GPU计算改gpu_5090、每节点32CPU/4卡、8节点不变，自动预期504000MiB/节点、无显式mem；只读验收8CPU/1GPU配额但仅CPU计算。最新launch_control_repair_freeze/acceptance允许只对绑定原/新SHA的本地协调器作配额修复，其他科学源码不得变化；实际80%资源门槛不变。
- `jscc-5e10`为用户授权的新定时推进任务，每15分钟推进本实验；等待计算期间保持ACTIVE，科学/视觉QA、报告和安全Git交付实际完成后暂停。旧compton-v5不恢复。旧实验只读，不重复使用旧计数代替新观测。
- 推进前读 `generated/jscc_geant4_5e10_10000/advance_registration.json`；登记PID仍活跃时不并行advance/fetch/submit。先完整输入32GPU validation10并严格取回生成本次authority，随后直接唯一formal10000/save50。失败保留旧发布和部分输出，诊断后有界补缺；不取消其他项目。
- 2026-10-10 20:00上海用户新增授权：保持5090主筛选1685272不动，另提交4090独立完整输入筛选试跑。唯一试跑1686107，8节点×4卡、每节点24CPU、自动预期240000MiB、无显式mem；单独selection_4090_trial_job/freeze/launch_acceptance和输出目录，原科学kernel及5e10观测不变。这是用户明确允许的并行筛选例外，不是第二套生产重建。每轮同时读取该登记，并运行 `python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_4090_trial.py status`；另外先检查generated中的selection_4090_trial_registration.json，任一提交/推进PID存活时不并行advance/fetch/submit。不重复试跑submit，不取消或改动主作业，不以试跑退出状态替代完整资源/逐行筛选验收，不自动替换生产selection_job。后续重建仍由主流程完整输入验收和实际内存门槛决定；不得把4090筛选能运行当成完整Compton重建内存证明。

- 后续不生成、计算、绘制或纳入比较任何218与440 keV跨能量相加图像，包括 `440SinglePlus218Single` 和 `440SingleComptonPlus218Single`。用户所写218+400按本工程218/440双能解释。440的Single+Compton联合重建仍保留，它不是跨能量叠加。
- EHE保留440单光子和218串扰校正单光子；JSCC保留这两路以及440 Compton、440 Single+Compton。218求解中的固定440→218前投影加性背景仍必须保留。
- 球ROI使用中心距球面至少1.5 mm的全部3 mm体素，等权求均值，不使用分数球体积权重，不再额外内缩。用户在确认严格完整体素会使10 mm球ROI为空之后，明确选择对全部球采用这一规则。它不保证体素角完全位于球内，不得称为完全消除部分容积效应。若ROI为空，指标标记不可计算，不能偷偷扩大ROI。
- 所有球使用同一个背景掩膜：体素完整位于Phantom活动腔体内部，并且与全部六个球、中央零活度肺插入物及其边界均不相交。包含所有合格层和位置，不再使用每球局部背景。接触区域表面的体素也排除。
- 具体实现和验收以 `nema_roi_policy.py`、`reconstruction_output_policy.py` 及最新 `reports/nema_interior_roi_20261010/README.md` 为准。真值必须来自本工程现有3D真值和登记球心。
- 旧六路JSCC和三路EHE的冻结发布、作业登记、数组及验收记录属于历史结果，只读保留。旧比较/报告脚本用于理解历史定义；新的对比使用 `tools/research/analyze_nema_interior_roi.py`，不能重新运行旧脚本产生叠加图或旧ROI指标。新JSCC入口按当前用户明确选择的路线冻结独立合同及匹配验证器，不能沿用历史六路合同提交。
- 先前ROI规则变更本身未授权新增计算；随后新增的JSCC5e10实验按上方最新授权独立推进。compton-v5保持暂停。

- 最新用户新增并行例外：独立8×2张5090完整输入筛选1686450，16rank、每节点16CPU/自动252000MiB、无显式mem，原1685272/1686107均保持不动。最新登记selection_5090_8x2_trial_job/freeze/identity_acceptance及独立输出为准。每轮先检查advance_registration.json、selection_4090_trial_registration.json、selection_5090_8x2_trial_registration.json；任一PID存活不并行advance/fetch/submit。另运行 `python -X utf8 experiments/ELLIPSE500x300_H120/jscc_5e10_5090_8x2_trial.py status`。不重复trial submit，不自动采用试跑结果替换主selection_job；完整退出/16rank真实资源/全部原行与缓存/SHA独立验收后才评估后续方案。原生产32卡冻结未改，正式拓扑以用户后续选择与完整输入验证为准，筛选占用不是全部Compton事件响应的内存证明。

- 2026-10-10 22:06最新有界恢复：1686450和1686107实际在GPU UUID监控处FAILED、已完全退出并保留原证据；原主1685272仍PENDING、原两项作业/登记没有修改或取消。新增独立monitor_repair冻结75e6ca22ffb21fa4及唯一恢复1686507实际RUNNING，仅修正完整CUDA/NVIDIA UUID字符串并捕获实际身份，原科学筛选及80%资源合同不变。四个PID（再加selection_5090_8x2_monitor_repair_registration.json）任一仍活不并行advance/fetch/submit；status助手自动优先最新恢复登记。不重复repair/submit，不覆盖旧发布/登记；只有完整筛选和匹配16rank的严格全行/资源验收才能评估后续正式内存，不能直接提交16GPU正式重建。

- 2026-10-10 22:23最新：1686507已COMPLETED/0:0、8分48秒、4849087选中事件，原完整结果保留。独立只读验收1686546已登记（匹配16rank发布4b595d58a50e48e2），使用 `jscc_5e10_8x2_selection_verification.py status/fetch`，不要重复submit或源筛选。五个PID增加selection_5090_8x2_monitor_repair_verification_registration.json，任一活跃不并行推进/取回/提交。原32GPU主1685272和原4090保持不动；不自动替换production selection或进入16GPU formal。筛选实际资源不是全部Compton响应内存证明：实际事件响应平均178.2039 GiB/节点，按原每rank16 GiB保守开销，8×2需210.2039 GiB超出196.875 GiB的80%预算；完整响应尚未实测。仅严格验收通过后记录结论并按用户后续拓扑选择及原完整输入validation10/authority门槛推进。

- 2026-10-10 22:28该试跑已完整交付：只读验收1686546实际COMPLETED/0:0、2分24秒，全部20view选中缓存逐值等于原始CSV，320收据/640分区数组/最终40数组闭合；1003成员及141201888字节归档/原日志严格SHA取回通过。selection_5090_8x2_monitor_repair_acceptance/resource_acceptance及formal_memory_assessment_acceptance为最新证据；不重复试跑/验收/取回，不自动替换主production登记。正式拓扑仍未改变，完整响应内存未实测，validation10/formal10000未提交。原两项作业保持不动，正常等待安静；仅需要处理的故障/实质里程碑/最终交付通知。


## 2026-10-10 最新用户授权：复用筛选、4090 8×3分批响应

用户明确要求复用已完成筛选，将Compton事件响应改成分批载入，用8节点×3张4090推进三路重建，原5090作业保持不动。新的独立入口为 `jscc_5e10_4090_streaming.py status/advance`，登记前缀 `streaming_4090_8x3`。它取代此前后续正式拓扑待选的限制；不调用旧32卡advance、修改旧selection_job或取消1685272。旧FAILED4090/8×2试跑仅保存原证据。

复用1686507/1686546已验收的全部4,849,087事件及40份索引/原行缓存，按24rank重新分区，原数据/SHA保持。新响应存储仅无损压缩完整float32字节，每块最多32事件，节点本地/tmp写入、fsync和完整SHA读回；迭代按原顺序还原当前块调用原事件权重及未正则MLEM。原v5物理、全120mm/78920活动单元、全部20view、矩阵/S、三路和固定本次440末图背景均不改，不丢事件或生成跨能量和。原科学函数调用与检查点AST逐一一致；本地6项测试实际通过，100迭代全部保存历史与常驻内存及原Compton分支逐值一致。该CPU测试不是完整输入GPU资源证书。

4090每节点18CPU/3GPU，自动预计180000MiB/节点，无显式mem；真实24rank/8host/3唯一GPU、UUID/RSS/Slurm MaxRSS/显存均保留20%余量。共享盘约66.6GB可用，完整响应不写共享盘。先实际8×3节点存储/设备/无损小块probe，节点本地盘磁盘余量保护；样本压缩预算不是全部缓存容量保证。成功退出、严格日志/资源/数值证明后进入全输入三路validation10，验证全部响应块/事件、原前向/转置/S及历史回归；独立只读验收、严格取回及新24卡authority通过后直接唯一formal10000/save50。正式固定验收节点以复用相同SHA本地缓存，不重新生成成功缓存；缺失、部分或失败保留并诊断，仅有界修复。实际吞吐和时限据验证实测，不能保证ETA。

定时任务已更新为新入口；检查原五项控制器PID及新增 `streaming_4090_8x3_registration.json`，任一实际存活不并行提交/推进/取回。完整三路数值、科学/视觉QA、报告和安全Git交付之前保持ACTIVE。原5090作业自然状态只读检查，若自行失败保存完整退出和原日志，不借新授权修改它。
