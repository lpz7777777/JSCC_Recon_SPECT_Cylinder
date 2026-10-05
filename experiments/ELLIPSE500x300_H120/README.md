# 218+440 keV 椭圆柱 FOV 实验

2026-10-06 03:41：预检1666430三阶段实际内容、SHA及资源已通过，角度/连续核GPU峰值35.73%/64.98%，主存约23%。独立正式入口1666534在计算前因新增输运字段语义校验错误失败并退出，输入与响应/MLEM未变；已修复并增加真实清单测试。修复发布2a62bcf35f00e1d4通过7项本地/远端测试及93项SHA，完整事件双模型10次验证 **1666592** 已提交（4节点×1GPU、bond0）。正式2000/save50配对尚未提交，实际短程通过后直接继续。[预检验收](reports/NEMA_Body_H60/compton_energy_probability_v5/PREFLIGHT_ACCEPTANCE.md)、[正式执行合同](reports/NEMA_Body_H60/compton_energy_probability_v5/FORMAL.md)。

**2026-10-05 21:41最新执行：[连续能量响应诊断验收](reports/NEMA_Body_H60/compton_energy_probability_v5/DIAGNOSTIC_ACCEPTANCE.md)完成，DIAGNOSTIC_GATES_PASSED。** 全159919固定训练事件的匹配S、独立空间/联合类别门槛通过，65114候选链已退出释放GPU。144个充分空间区偏差RMS由2.852%降至1.124%；联合类别仅57/504充分，447未判定，不能称整体联合物理已认证。下一步为独立入口的新基底回归与完整事件10次试跑；未提交配对成像，尚不能称尖峰已经解决。原生产核与基底未覆盖，新增Geant4和精细A矩阵均为0，原自动任务保持停止。

**2026-10-05 23:54推进：[完整事件预检](reports/NEMA_Body_H60/compton_energy_probability_v5/PREFLIGHT.md)作业1666205已提交，4节点×1GPU，当前Priority排队。** 独立发布通过17项远端测试及75项文件哈希；链为历史50次回归→角度核完整事件10次→连续能量核完整事件10次。两种核共用91225事件、78920完整柱单元及各自匹配S，不改变原生产基底，不重启精细场。新图像及2000次配对尚未完成。

上一阶段[科学总报告](reports/NEMA_Body_H60/process_list_global_audit_v4/REPORT.md)量化了144个充分空间区、内部多事件峰责任、材料尾部、厚层位置偏移和78920完整柱单元算子。精细场继续停止，约536GB清理及[停止证据](reports/NEMA_Body_H60/compton_response_geometry_v3/STOP_AND_CLEANUP_20261005.md)保留。以下精细场生产和续跑条目是停止前历史，不再执行。

停止前生产记录（已终止）：[完整可续跑A候选场](reports/NEMA_Body_H60/compton_response_geometry_v3/TILED_FULL_PRODUCTION.md)已在65114 GPU0/1/3/4启动，PID2542514/2542646/2542872/2543046。区域168/168精度、21块读取/12公共面及四卡完整索引8块资源试跑均通过，复用29块，目标10720块/2.211TB。20小时tile边界停止、24小时外部上限；全场科学精度仍HOLD，S2和正式2000次未运行，尚无尖峰改善结论。


2026-10-05停止前续作记录（已终止）：[完整边界响应算子与GPU验证](reports/NEMA_Body_H60/compton_response_geometry_v3/FULL_OPERATOR_PREFLIGHT.md)。全部6880部分单元已接入诊断，CPU/GPU全5283840项积分及归一化一致；128固定事件的16→32阶加密全部通过，最大变化0.2137%，GPU细级积分实测约快9.26倍。全轴向0.75mm补点及CPU/CUDA检查完成，细场完整覆盖由6增至80/137600单元—视角对。[分块场试生产](reports/NEMA_Body_H60/compton_response_geometry_v3/TILED_FIELD_PREFLIGHT.md)完成两相邻块，保留完整11520行物理计算，10496晶体合并数据逐值提取和公共面读取通过；完整分块场预计持久化2.21TB，现已作为待验收候选启动。当前关键门槛是全场物理A插值精度；S2和正式2000次配对未完成，未新增输运、正则化或旧任务。这些是响应诊断，尚无尖峰改善结论。

[独立区域精度验证](reports/NEMA_Body_H60/compton_response_geometry_v3/REGIONAL_A_VALIDATION.md)已完成：四横向位置×三轴向层，0.75/0.375mm两个独立物理场共98463点；168项K加权交集/完整参考全部通过，141项有可判别响应，最大采样变化0.5040%、积分变化0.02988%。目前推进可续跑分块场多worker小批，完整场和S2仍未完成，未提交新图像。这些是响应诊断，尚无尖峰改善结论。

2026-10-04较早阶段：[完整单元响应与A加密检查](reports/NEMA_Body_H60/compton_response_geometry_v3/WHOLE_CELL_VALIDATION.md)。补齐支持后的2520积分案例全通过，独立3→1.5→0.75→0.375mm采样最终84/84相邻收敛，最大0.6553%；原A插值场在具体近侧完整单元的差异仍可达38%。这是响应诊断，不是NEMA图像误差或尖峰改善。R1作业1661583以4节点×1GPU完成原q3保留的50次图像回归（两路L2均0）及完整10次试跑，GPU峰值44.58%、主存24.20%。细采样现已接入完整R2诊断，S2、R2试跑和正式2000次配对仍未完成。旧任务/正则化继续停止，原矩阵、单光子链和q网格保持不变。

2026-10-04新实施入口：[compton_response_geometry_v3计划与证据](reports/NEMA_Body_H60/compton_response_geometry_v3/README.md)。共享响应新增默认关闭的stable_float64几何模式，8项CUDA测试及冻结旧核真实事件逐值回归通过。现有输运全量重扫得到NEMA筛选前91385、稳定q3后91225，退出6个、无新增；新S1独立圆/椭圆及九分区、补充边界空间门控通过。初次1260案例发现的端点外推、6外圈支持及精确体积问题已通过独立补点解决；当前R2门槛是A插值精度及实际全算子接入，不再把旧支持缺口当作尚未实施。下文“生产核未改”属于此前研究阶段的历史描述；当前默认legacy算法行为仍兼容，稳定模式须使用新匹配S和冻结配置。

2026-10-04继续研究：[process_list数值与响应审计](reports/NEMA_Body_H60/process_list_followup_20261004/README.md)已全量扫描首散射修正B组91231个保留事件。确认近共线CUDA float32角度梯度错误：实际事件位置σ可由约1.16°放大到91.20°，全圆q从3.35变成0.97。独立稳定原型通过CPU/CUDA及有限差分测试；全量有6个原保留事件转为q>3，但它们对当前Compton/JSCC主峰的事件责任仅约0.064%/0.014%，不能当作主要尖峰已解释。建议先修数值正确性，再优先验证真实椭圆相交体积响应和材料/能量误差分布。生产核、S及原图未改，没有新增输运或重建；旧自动任务仍停止。

最新状态（2026-10-04）：独立[compton_first_scatter_v2最终验收](reports/NEMA_Body_H60/compton_first_scatter_v2/ACCEPTANCE.md)已完成760个worker、3.17e9输运、NEMA旧输出字节回归、两组独立S及1660254配对成像。4节点×1GPU，2:47:44；50次回归两路L2=0，两组10次门控通过，两组各Compton/JSCC2000次、四条历史各40帧及检查点通过远端与本地内容验收。统一q3后A97078、B91231，恢复453条旧漏收、净损失6.023%。固定尺度中央72mm MIP、轴冠矢位及全40帧曲线均交付。**科学判据未达到：Compton最大密度/峰背景下降7.29%/8.27%，JSCC41.56%/41.99%；Compton22/37mm球CRC损失7.68/5.64个百分点。** 主峰仍在全120mm边界，事件定义修正不足以解释主要尖峰。本轮结束，不自动扩展计数/迭代/核。旧1657887取消及nema-5e9-3 PAUSED保持不变。


物理源及所有后续成像限定于物体坐标 `(x/250)^2+(y/150)^2≤1，|z|≤60 mm`，长轴沿 x。探测器保留四层、10496 晶体，准直器前表面距源中心 270 mm，20 个固定半径旋转视角；Geant4 的源中心为世界坐标 `(0,-345,0) mm`。本系列不加入人体材料衰减。**500×300×120 mm 是目标物理范围，不代表该范围的有效成像能力已经通过验收。**

本页是当前入口和文件索引。早期操作、作业号、故障及逐次结果详见 [HISTORY.md](HISTORY.md)；历史中较早的“待运行”等状态可能已过时。远端连接方法见 [docs/REMOTE_COMPUTE_ACCESS.md](../../docs/REMOTE_COMPUTE_ACCESS.md)，不在本实验目录保存密码或私钥。

**已停止的历史对照：NEMA 5e9 `response_mismatch_cut3_v1`。** 全20视角复现484936事件，删除1168、保留483768；独立匹配灵敏度平均闭合1.00095。修复作业1657887通过50次回归（两路相对L2均0）及10次资源试跑，交付2000次图集后按用户要求取消，未完成10000次。2000次JSCC极端峰下降80.81%、Compton29.20%，整体背景噪声和泄漏基本未改善。这是转入事件定义研究的依据，不能当作10000次结论。[历史对照报告](reports/NEMA_Body_H60/response_mismatch_cut3_v1/README.md)及[运行簿](reports/NEMA_Body_H60/response_mismatch_cut3_v1/PRODUCTION.md)继续保留；1657719、1657745、1657887均不恢复。

2026-10-04按用户要求更新显示约定：NEMA轴向MIP默认**仅取中央72mm，即z∈[−36,+36]mm**。40层、3mm间距的网格上下各去掉8层（各24mm），保留24层，层中心−34.5～+34.5mm。重建物理FOV仍500×300×120mm，原始图像及定量指标不改。已完成的MLEM、边界绑定、弱Huber均已重画[中央72mm六路对照和逐迭代MIP](reports/NEMA_Body_H60/spike_ablation/comparison_mip72_20261004/README.md)。`iterations_z20.png`仍是z=+1.5mm单层轴位图，不是MIP。`plot_nema_iterations.py`、`compare_spike_ablation.py`和后续取回流程共用此默认策略；`plot_nema_mip.py --trim-layers 8`可只重画MIP而不重算指标。此前上下各去掉3层的[102mm历史显示](reports/NEMA_Body_H60/NEMA_Body_H60_5e9_1644876/mip_trim3/README.md)及原始全120mm图均保留。

## 当前已形成的基线

**2026-10-04研究方向变更（当前有效）：按用户要求停止 Huber、TV 等正则化重建。1651956 已取消，Slurm 记录 CANCELLED，01:46:21 全部退出；不再自动重提中/强 Huber、TV 或新正则化组。** 保留 MLEM、已完成边界绑定/弱 Huber 和未完成中档的原始证据。后续重点为历史 `process_list`、Geant4 事件定义、Compton 响应和 MLEM 灵敏度的一致性。[响应审计与可复现证据](reports/NEMA_Body_H60/process_list_audit/README.md)为新入口；以下对照运行记录属于取消前历史。

已完成20视角×512候选事件的全网格响应抽样审计，使用65114的8个CPU线程，未运行图像迭代。找到445keV能量和正常、能量角58.7°却与完整网格109.8°–153.5°几何角严重不符的已接受事件；当前最小支持阈值1保留了这类事件，旧strict/sparse阈值50会拒绝部分。真值处得分证实强烈的源外增亮驱动，但最终责任分解、部分体积边界及灵敏度模型表明不能只靠回滚一个参数解释全部尖峰。晶体编号/坐标逐个一致，三路原生产响应未改。详细区别已证实事实与待追踪的Geant4过程，见新审计报告；下一步先做事件轨迹/ARM和响应一致性诊断。

2026-10-02新增用户授权的[尖峰算法对照](reports/NEMA_Body_H60/spike_ablation/README.md)：复用NEMA 5e9已完成MLEM基线，分开比较仅边界小单元密度绑定、MAP-Huber弱/中/强、MAP-TV。核心更新及本地8项数值/几何测试已通过，采用完整数据10次→200次→10000次的验收门控；发布、作业登记和最新证据见对照报告目录。此处新增对照不改变已验收基线、Factors或Geant4输入。

2026-10-04集群01:10进度：作业 **1651956** 仍在8节点×1张4090运行。边界绑定、弱Huber两组已完成10000次并通过六路×200帧正式验收；中档Huber在正式Compton/JSCC **1350/10000**，强Huber/TV尚待执行。[已完成组的统一尺度图像和逐迭代指标](reports/NEMA_Body_H60/spike_ablation/comparison_20261004/README.md)已生成：绑定压低小单元尖峰但主体噪声几乎不变；弱Huber明显降低噪声，同时明显损失热球CRC并增加轴向源外积分。当前不判定最优算法。完整FOV原始指标保留40层，最新[中央72mm MIP显示](reports/NEMA_Body_H60/spike_ablation/comparison_mip72_20261004/README.md)沿用完全相同的重建和定量指标。

2026-10-02算法研究更新：[Compton/JSCC尖峰诊断与改进方案](reports/NEMA_Body_H60/spike_research/README.md)对1e9/5e9原始历史做了体积加权核查。只占椭圆体积0.053%的小相交单元，分别承载约7.3%–7.4%的Compton积分、4.9%的JSCC积分；最大尖峰单元的单位体积Compton灵敏度并非低谷，且尖峰在早期已形成。建议优先核查相交体积核积分、事件责任与空间可辨识性，并对照几何一致的三维MAP-JSCC及边界密度约束。该报告区分已测事实与待测假设；本次没有更改生产算法或提交新重建。

| 环节 | 已完成内容及证据 | 限制 |
|---|---|---|
| 几何 | 完整圆形极坐标计算网格 132040 列；物体坐标椭圆活动列 82040；20 视角旋转映射及严格伴随测试 | 圆网格是计算支持域，不是源的物理 FOV |
| 矩阵 | 218→218、440→440、440→218 三套新距离 Factors 已生产、转换、分层校准；独立圆柱数据核查见 [响应报告](reports/independent_circle_response.json) | 440→218 仅用低能窗散射；距离不同，不能复用旧 Factors |
| Compton 灵敏度 | 新网格的 `Sensi_d` 与独立圆源数据闭合，见 [报告](reports/sensitivity_circle_closure.json) | 椭圆边缘仍需空间质控 |
| Geant4 对照 | 原距离圆柱、新距离圆柱、椭圆均匀源、椭圆热柱与 XCAT 的已有输入；四组新距离/椭圆 1e9 正式数据经实际初级光子数、20 视角和 worker 哈希检查 | 传输完成不等于有效 FOV 验收 |
| 重建 | CircleNewDist、EllipseUniform、EllipseContrast、XCAT 四组 1e9、10000 次、每 50 次保存的六路结果已通过文件完整性检查；[逐迭代图集](reports/iteration_galleries/README.md) | 1e9 高迭代有明显尖峰、短轴热柱欠恢复和 Compton 端区偏低；不得作为通过定量验收的图像 |
| 独立点源 | 10 组中心及长/短轴端区、双能、20 视角点源已完成；[计数效率](reports/selected_point_efficiency.json)、[响应图样](reports/selected_point_factor_alignment.json) | 最近点及 8 邻点插值图样比较仍受低计数 Poisson 噪声影响，不能单独证明响应失配 |

代表性科学指标：椭圆均匀源 `ρ≤0.9、|z|≤30 mm` 的 440 联合图在第 10000 次 CV 约 2.95；`45<|z|≤60 mm` 的 440 联合均值约为中心的 0.575，见 [均匀性报告](reports/EllipseUniform_1e9_1640929_ellipse_uniformity.json)。椭圆热柱五组的 CRC 中位数在第 10000 次包含负值，见 [热柱报告](reports/EllipseContrast_1e9_1641013_contrast_metrics.json)。XCAT 的空间积分恢复也未达到定量验收，见 [XCAT 报告](reports/XCAT_1e9_1641014_xcat_spatial.json)。这些结果指出需要继续核查响应、统计和空间覆盖；不能靠平滑宣称整个椭圆 FOV 可用。

## NEMA Body Phantom H60：1e9与独立5e9已完成验收和比較

用户已验收其几何与双能填充。标准图的相切圆弧截面、六球安排、3 mm 双能真值及预览见 [NEMA 说明与图](reports/NEMA_Body_H60/README.md)。主体高 60 mm，位于椭圆 FOV 正中央。背景同时有 218 和 440 keV，各自相对浓度 1；Ø10/17/28 mm 球仅含 218 keV、浓度 10，Ø13/22/37 mm 球仅含 440 keV、浓度 10。各自单能热球/背景均为 10:1，初级 γ 权重在生成 Geant4 宏时再乘产额 0.114/0.259。

`prepare_nema_simulation.py` 将已验收的 3 mm 真值精确合并成约 3937 个同值 cuboid，使用既有 `/xcat/add` 源接口生成 20 个角度的宏和 200 个唯一种子 worker，总计 1e9 初级 γ。该输运源与**体素真值**的积分逐能闭合；在 3 mm 边界体素内均匀采样，因此解析球壳边界存在至多一个体素量级的近似。此近似应计入小球结果解读。`validate_nema_simulation.py` 检查宏哈希、视角、发射数、初级能量比例和 worker 输出。

2026-09-30：maty 短程 `15506070` 及完整规格的首个 worker `15506288` 均通过。首次正式阵列 `15506077` 在启动 Geant4 前因空 Bash 数组失败，**无生产 worker 输出**；修复后 `15506302` 完成剩余 199 个 worker。合计 1e9 初级 γ，218/440/其他为 293821153/706178847/0，全部 worker、20 视角、输出哈希及能量比例通过。合并的 23 文件输入包已逐文件校验传至 scxi717。4 节点×1 张 4090 的完整事件 10 次重建试跑 `1643079` 已通过，筛选后 Compton 事件为 97299，显存峰值约 11.73 GiB/卡；10000 次正式作业 `1643142` 已启动，结果尚未验收。证据与验收条件见 [NEMA 运行簿](reports/NEMA_Body_H60/PRODUCTION.md) 和 [输运核验](reports/NEMA_Body_H60/transport_1e9.json)。

2026-10-01 更新：`1643142` 于北京时间 04:37:25 完成，总耗时 5:41:33。正式完整性核验通过，六路最终图、每路200帧历史和串窗预测均已逐文件验哈希取回。GPU 最大预留占比51.02%，主存保守55GiB/节点预算下峰值约46.24%，满足20%余量。[完整图集与200帧指标](reports/NEMA_Body_H60/NEMA_Body_H60_1e9_1643142/README.md) 包含固定色标的全FOV轴位迭代图、最终多平面/MIP、CRC/CNR/CV曲线。218校正图的10/17/28mm球CRC为−0.025/0.691/0.880；440联合图13/22/37mm为−0.053/0.208/0.430。背景CV分别为1.505/1.773，高迭代噪声和源外轴向泄漏显著；流程完成不等于小球或整个FOV性能达标。本轮止于1e9验收，未启动1e10。

2026-10-01，用户授权新增独立 **5e9** 实验。20视角、200worker×2500万初级γ，新种子30100101–30100300，与1e9不重叠；体模和全部源宏除发射数外保持一致。maty门控链为短程15514663、首完整worker15514664、其余worker15514665、核验打包15514666；短程与首完整worker均通过。截至北京时间13:00，200个worker和20视角全部完成，实际5e9初级γ及能量比例、全部输出哈希通过；23份输入已逐文件核验部署至scxi717。完整数据试跑1644811已算完，实测接受484936个Compton事件，但GPU预留峰值97.28%未通过资源门槛。已修复此前8节点启动时的rank分配超时（诊断1644806通过），完整数据10次复验1644842（8节点×1张4090）通过：GPU预留峰值42.35%、主存36.94GiB，六路及串窗预测完整、19份数组与修复前数值回归通过。2026-10-02更新：正式1644876已完成，8节点×1张4090、耗时11:49:28；六路10000次及各200帧、输入/Factor哈希通过，GPU42.35%、主存49.62%峰值满足余量。完整[5e9验收与科学报告](reports/NEMA_Body_H60/5e9/ACCEPTANCE.md)、[逐迭代图集](reports/NEMA_Body_H60/NEMA_Body_H60_5e9_1644876/README.md)及[1e9比较](reports/NEMA_Body_H60/5e9/comparison/README.md)已形成。218/440联合背景CV降至0.669/0.951；218的10mm球CRC0.009，440联合源外轴向质量13.60%，定量性能仍有不足。流程验收通过不等于全部FOV可用。详见[5e9运行簿](reports/NEMA_Body_H60/5e9/PRODUCTION.md)。不覆盖1e9，不自动启动1e10。

在本实验目录复现输入：

```powershell
python make_nema_body_h60.py
python -m unittest -v test_nema_body_h60.py
python prepare_nema_simulation.py
python validate_nema_simulation.py generated/NEMA_Body_H60/Simulation_1e9/jobs.json
# 新计数级别（已有冻结目录时不要重跑生成器）
python prepare_nema_simulation.py --level 5e9
python validate_nema_simulation.py generated/NEMA_Body_H60/Simulation_5e9/jobs.json
```

生成器拒绝覆盖已冻结的 `Simulation_1e9` 目录；要重做正式输入须使用新的实验子目录和清单，不应原地改写作业宏。正式重建结束后，应先运行 `verify_formal_result.py` 核对六路最终图、各 200 帧历史、串窗预测和资源余量，再以 NEMA 双能真值评估六球 CRC/CNR、背景噪声与轴向边缘。

## 目录与保留规则

| 路径 | 内容 |
|---|---|
| `config.json`、`geometry.py`、`ellipse_operator.py`、`torch_active_operator.py` | 统一实验几何、椭圆约束和严格伴随算子 |
| `run_matrices.py`、`run_scatter_slabs.py`、`convert_factors.py`、`calibrate.py`、`run_sensitivity.py` | 65114 的新距离三路矩阵、校准及灵敏度生产 |
| `simulation.py`、`xcat.py`、`point_scan.py`、`make_nema_body_h60.py`、`prepare_nema_simulation.py` | maty 的规则源、XCAT、点源和 NEMA 真值/宏生成 |
| `run_reconstruction.py`、`reconstruct.sh`、`resource_budget.py`、`verify_formal_result.py` | scxi717 的椭圆约束六路 JSCC 重建及资源/文件验收 |
| `reports/` | 小型完整性证据、量化 JSON、预览图和图集；按体模/主题查看 |
| `generated/` | 约 15 GB 本地大体模、宏、取回的数据和临时分析输入，受 Git 忽略；历史结果和可复现证据仍需保留，不能仅因体积大而删除 |

已删除不再被引用的旧单次 `smoke_ellipse.mac` 与 `maty_smoke.sh`。保留矩阵切片、数据收集和其他诊断脚本，因为它们记录了生产或失败修复的可复现路径。尚未完成的点源插值诊断见 [POINT_ALIGNMENT.md](reports/POINT_ALIGNMENT.md)，不得把低原始余弦值直接解释为模型错误。

远端独立工作区：maty 为 `/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928`；scxi717 必须在 `/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120` 下运行；65114 的矩阵工作区为 `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120`。不覆盖先前 FOV120 任务、Factors 或重建结果。
