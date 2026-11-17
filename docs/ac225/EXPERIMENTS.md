# 实验与证据目录

当前研究页同时列计算、严格交付、可视化和物理结论；历史记录保留原编号，不能据旧PENDING或held字段恢复任务。

## NEMA实际成像

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [nema-5e10-v5-joint](studies/nema/nema-5e10-v5-joint.md) | NEMA实际5e10：新v5流程440 SC+Compton联合重建 | completed / strict-complete-history | 1740189同440keV联合10000次完成；背景BV约35.94%，低于SC约40.50%，但13mm热球CNR下降；不说明环伪影成因已修正。 |
| [nema-jscc-5e10](studies/nema/nema-jscc-5e10.md) | NEMA实际5e10：SC218/440与纯Compton440 | completed / strict-complete-history | 1722760三路各10000次已交付；纯Compton后期环状、条纹与尖峰明显。纯Compton不命名为JSCC。 |
| [nema-jscc-5e9-v5](studies/nema/nema-jscc-5e9-v5.md) | NEMA JSCC 5e9 连续核v5：历史10000次正式基准 | completed / strict-historical-delivery | 历史六路含已退休跨能量求和；后续仅采用获授权的分能量通道和当前ROI。 |
| [nema-ehe-g4-5e9](studies/nema/nema-ehe-g4-5e9.md) | NEMA原EHE矩阵 5e9：实际Geant4 | delivered / see-original-evidence | 旧矩阵真实Geant4重建已交付；当前EHE实际比较优先使用11月dense65536独立重建。 |
| [nema-ehe-g4-5e10](studies/nema/nema-ehe-g4-5e10.md) | NEMA原EHE矩阵 5e10：实际Geant4 | delivered / see-original-evidence | 旧矩阵真实Geant4重建已交付；当前EHE实际比较优先使用11月dense65536独立重建。 |
| [nema-current-roi](studies/nema/nema-current-roi.md) | NEMA当前1.5mm内缩球中心ROI及统一完整背景 | analysis-complete / see-original-evidence | 原3D球内中心距球面至少1.5mm；所有球共享完整体模内部背景，排除六球、肺和接触边界。 |
| [nema-source-truth](studies/nema/nema-source-truth.md) | NEMA H60球体、肺、218/440空间分布 | authoritative-source / see-original-evidence | 原3mm三维NEMA H60真值：六球与中心肺；218热球10/17/28mm，440热球13/22/37mm。球心及完整能量分布按原manifest/数组。 |
| [nema-ehe-dense-5e9](studies/nema/nema-ehe-dense-5e9.md) | NEMA EHE加密矩阵实际5e9 | completed / strict-delivered | A218/A440提高到65536点/可见面，匹配自身S；200次正式完成，但边缘高值、散斑和小球对比不足仍存在。 |
| [nema-ehe-dense-5e10](studies/nema/nema-ehe-dense-5e10.md) | NEMA EHE加密矩阵实际5e10 | completed / strict-delivered | A218/A440提高到65536点/可见面，匹配自身S；200次正式完成，但边缘高值、散斑和小球对比不足仍存在。 |
| [nema-cartesian-5090](studies/nema/nema-cartesian-5090.md) | NEMA5e9原生Cartesian：5090正式三路 | FAILED / not-delivered | 1822462于14:36:55失败退出；进度记录440 SC到10000，但根和三路completed均缺失，不能认定计算完成或完整交付。实际RSS/VRAM超过原80%守卫；输入、输出、原发布和日志保留，未修复/重投。 |
| [nema-cartesian-4090-r2](studies/nema/nema-cartesian-4090-r2.md) | NEMA5e9原生Cartesian：4090 r2失败保留 | FAILED / not-delivered | 有效r2正式1822534于14:10:37失败；没有progress/completed，日志同样触发原80%实际RSS/VRAM守卫。原同名无r2分支为未提交预检失败。全部发布/输入/失败输出保留，未重投。 |
| [nema-width-sweep](studies/nema/nema-width-sweep.md) | NEMA5e9：100/80/60% Compton展宽 | completed / strict-delivered | 1814153三档各10000次、600帧严格交付。缩窄展宽未改善整体SSIM/NRMSE；保留旧历史差异未通过标志，不选择80/60%替代生产模型。 |
| [water-sc-first](studies/nema/water-sc-first.md) | 带水NEMA：四分支32路SC完整交付 | completed / strict-delivered | 严格交付32路6400帧及背景；原metadata的visual_review_pending=true，14路预览和后续专题不能代替全32路视觉验收。旧README的未交付描述被strict_delivery更新。 |
| [water-scatter-v2](studies/nema/water-scatter-v2.md) | 带水NEMA：独立供体固定散射模型v2 | completed / strict-delivered | 8路1600帧严格交付；独立剂量供体仍含相同体模布局先验。侧窗和440→218串窗偏差单列；并非物理认证。 |
| [opengate-ehe-5e9](studies/nema/opengate-ehe-5e9.md) | OpenGATE EHE NEMA 5e9有效输运 | completed / strict-delivered | 源旋转修复后的真实投影，两能量400帧已严格交付；边缘高值仍存在，MC-CDR近似与4mm网格限制保留。 |
| [opengate-ehe-5e10](studies/nema/opengate-ehe-5e10.md) | OpenGATE EHE NEMA 5e10有效输运 | completed / strict-delivered | 源旋转修复后的真实投影，两能量400帧已严格交付；边缘高值仍存在，MC-CDR近似与4mm网格限制保留。 |
| [opengate-he-nema](studies/nema/opengate-he-nema.md) | OpenGATE HE NEMA 5e9/5e10 | completed / strict-delivered | HEGP孔规格+现有NaI，两剂量两能量800帧交付并完成数值/视觉QA。背景BV降低，部分小球恢复下降；不是完整商用相机或纯单因素比较。 |

## Derenzo实际成像

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [derenzo-jscc-218](studies/derenzo/derenzo-jscc-218.md) | Derenzo JSCC 218 keV：独立单能5e9 | completed / strict-delivered | 每个探头/能量独立5e9 gamma；两个源不是同一次225Ac衰变链，218背景为零。 |
| [derenzo-jscc-440](studies/derenzo/derenzo-jscc-440.md) | Derenzo JSCC440单能实际5e9：SC与SC+Compton正式10000次 | completed / strict-delivered | 每个探头/能量独立5e9 gamma；两个源不是同一次225Ac衰变链，218背景为零。 |
| [derenzo-ehe-218](studies/derenzo/derenzo-ehe-218.md) | 原EHE矩阵：Derenzo EHE 218 keV：独立单能5e9 | delivered / see-original-evidence | 旧矩阵实际5e9结果已交付，当前EHE对比优先使用1761244加密矩阵。 |
| [derenzo-ehe-440](studies/derenzo/derenzo-ehe-440.md) | 原EHE矩阵：Derenzo EHE 440 keV：独立单能5e9 | delivered / see-original-evidence | 旧矩阵实际5e9结果已交付，当前EHE对比优先使用1761244加密矩阵。 |
| [derenzo-completed-figures](studies/derenzo/derenzo-completed-figures.md) | Derenzo已完成三组的真值、MIP、切片和迭代图 | analysis-complete / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [derenzo-ehe-dense-5e9](studies/derenzo/derenzo-ehe-dense-5e9.md) | Derenzo EHE加密矩阵实际5e9 | completed / strict-delivered | A218/A440提高到65536点/可见面，配套自身S；正式200次完成。加密后仍有边缘高值、散斑与棒组分离不足。 |
| [derenzo-jscc-5e10-218](studies/derenzo/derenzo-jscc-5e10-218.md) | Derenzo实际5e10：218 SC | completed / strict-delivered | 218单光子10000次完成并严格取回；本次目录整理不冒充已有统一新版画廊。 |
| [derenzo-jscc-5e10-440-sc](studies/derenzo/derenzo-jscc-5e10-440-sc.md) | Derenzo实际5e10：440 SC成功输出保留 | SC-saved10000;parentFAILED / strict-delivered | 父任务1752461失败但成功SC10000/200保留并于11月12日严格取回；联合失败9950单独登记，不重算成功SC。 |
| [derenzo-jscc-5e10-440-joint](studies/derenzo/derenzo-jscc-5e10-440-joint.md) | Derenzo实际5e10：440 JSCC联合64卡补算 | FAILED-at-9950 / strict-partial-199-frames | 1771220失败，保存到9950/10000，共199帧已严格取回并完成预览QA；不是正式10000次完成，不恢复旧64卡请求。 |
| [derenzo-ehe-dense-5e10](studies/derenzo/derenzo-ehe-dense-5e10.md) | Derenzo EHE实际5e10：输运完成、严格传输待修复 | completed / strict-delivered | 两能量各200次、400帧严格交付与数值/视觉QA完成；旧传输中断/未提交状态已被交付证据更新，物理HOLD保留。 |
| [opengate-he-derenzo](studies/derenzo/opengate-he-derenzo.md) | HE Derenzo：四路完成及PPT交付 | completed / strict-delivered | 4路800帧严格交付，数值/视觉与14页PowerPoint交付完成。复用HE NEMA响应，没有新响应发射；几何/网格/有限板近似仍未获物理认证。 |

## EHE系统矩阵

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [ehe-dense-matrix](studies/ehe/ehe-dense-matrix.md) | EHE218/440加密系统矩阵 | completed / matrix-identity-bound | A218/A440原0–6与唯一补块7完成，16²→256²点/可见面；当前实际EHE重建采用此版本。 |

## 前投影与Poisson对照

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [ehe-nema-poisson-20view](studies/synthetic/ehe-nema-poisson-20view.md) | EHE NEMA20视角：5e9至5e12前投影＋Poisson | completed / strict-selected-48;remote-all1600 | 匹配模型自生数据在5e9/5e10也出现明显边缘伪影；提高计数显著减弱尖峰，但高剂量仍有模糊。 |
| [ehe-nema-poisson-60view](studies/synthetic/ehe-nema-poisson-60view.md) | EHE NEMA60视角：5e9至5e12前投影＋Poisson | completed / strict-selected-48;remote-all1600 | 同总gamma、200次下，60角外围ROI发射量占比8组均降低；440在低两档全域NRMSE略升，高剂量误差仅小幅改善。 |

## 物理与算法诊断

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [compton-v5-legacy](studies/diagnostics/compton-v5-legacy.md) | 440连续材料能量核：legacy独立校准及2000次配对 | accepted-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [compton-v5-ideal](studies/diagnostics/compton-v5-ideal.md) | ideal 1e9连续核对照；与legacy剂量实验分开 | historical-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [first-scatter-v2](studies/diagnostics/first-scatter-v2.md) | 首散射事件定义配对研究 | historical-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [event-physics-audit](studies/diagnostics/event-physics-audit.md) | 全局事件算子、测度及物理响应审计 | accepted-diagnostic / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [nema-prejune-compton](studies/diagnostics/nema-prejune-compton.md) | NEMA实际5e9：旧Compton核回退对照 | completed / strict-delivered | 近期NEMA实际投影配旧核/匹配S/OSEM8完成10000次，明显环纹和尖峰仍存在，回退未消除问题。 |
| [compton-ring](studies/diagnostics/compton-ring.md) | Compton环状伪影 | analysis-complete / source-bound-analysis | 环纹与网格/响应离散化值得隔离研究；观察性关联不足以确定唯一成因。 |
| [compton-transition](studies/diagnostics/compton-transition.md) | Compton角点过渡与体积 | analysis-complete / source-bound-analysis | 未发现简单漏乘/重复乘体积或S约定错误；中心点响应代表有限单元的积分误差仍是候选。 |
| [compton-git-history](studies/diagnostics/compton-git-history.md) | Compton算法Git历史 | analysis-complete / source-bound-analysis | 源码版本、核/S/筛选/求解均有差别；历史图像外观不能代替同源匹配实验。 |
| [compton-legacy-backup](studies/diagnostics/compton-legacy-backup.md) | 旧Compton结果备份审计 | analysis-complete / source-bound-analysis | 备份Image_JSCCD为纯Compton；511keV旧数据与440keV近期NEMA/FOV不同，不当作直接等价基准。 |
| [compton-new-vs-old](studies/diagnostics/compton-new-vs-old.md) | NEMA新旧Compton流程对比 | analysis-complete / source-bound-analysis | 旧流程后期起伏更强，回退未消除环纹；同时改变多步骤，不能唯一归因。 |
| [ehe-crystal-response](studies/diagnostics/ehe-crystal-response.md) | EHE单晶体空间响应 | analysis-complete / source-bound-analysis | 原矩阵六个晶体的分能量MIP/切片；晶体响应覆盖及面采样问题另由准直器审计解释。 |
| [ehe-collimator-audit](studies/diagnostics/ehe-collimator-audit.md) | EHE准直器响应与面积分 | analysis-complete / source-bound-analysis | 256点/可见面的局部积分不足已确认；65536点加密版本独立完成，物理模型仍需验证。 |
| [ehe-fov-clearance](studies/diagnostics/ehe-fov-clearance.md) | EHE准直器FOV间隙 | analysis-complete / source-bound-analysis | 准直器前端298.5mm，完整旋转FOV最小间隙48.5mm，未伸入FOV。 |
| [ehe-dense-edge](studies/diagnostics/ehe-dense-edge.md) | EHE加密矩阵边缘高值 | analysis-complete / source-bound-analysis | 噪声放大与空间响应失配并存；NEMA440→218串窗预测低约25.27%，加密没有消除主要尖峰。 |
| [ehe-edge-original](studies/diagnostics/ehe-edge-original.md) | EHE原矩阵边缘高值历史诊断 | analysis-complete / source-bound-analysis | 保留早期边缘/S/计数诊断；当前结论优先使用11月加密矩阵及匹配Poisson对照。 |
| [jscc-uncertainty-performance](studies/diagnostics/jscc-uncertainty-performance.md) | JSCC v5不确定度与隔离等价执行优化 | analysis-complete / local-report | CPU配对计时与真实核局部检查已有证据；合成B/C不等于完整物理算子，CUDA stress未运行，不能推断完整GPU速度或图质改善。 |
| [nema-roi-audit-20261115](studies/diagnostics/nema-roi-audit-20261115.md) | ROI/源对齐派生评价 | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [nema-bg-out-error-20261115](studies/diagnostics/nema-bg-out-error-20261115.md) | 背景+体外误差派生评价 | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [nema-sc-jscc-he-gaussian-20261115](studies/diagnostics/nema-sc-jscc-he-gaussian-20261115.md) | 五档3D高斯派生评价 | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [nema-sc-jscc-he-cnr-crc-bv-20261114](studies/diagnostics/nema-sc-jscc-he-cnr-crc-bv-20261114.md) | CNR/CRC–BV派生评价 | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [nema-sc-jscc-he-5e9-5e10-20261114](studies/diagnostics/nema-sc-jscc-he-5e9-5e10-20261114.md) | SC/JSCC/HE综合派生评价 | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |

## 体模与系统设计

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [derenzo-design-selected](studies/design/derenzo-design-selected.md) | Derenzo等面积区域：实际输运采用的唯一设计 | design-only / local-design | 当前输运唯一解析几何：218为6/8/10/12/14/16mm、114棒；440为10/12/14/16/18/20mm、53棒。 |
| [ehe-side-shield-design](studies/design/ehe-side-shield-design.md) | EHE NaI四侧10mm铅屏蔽框：加屏蔽前后设计 | design-only / local-design | 以原始报告的研究范围和结论为准。 |

## 展示与历史参考

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [results-72mm-mip](studies/reference/results-72mm-mip.md) | 2026-10-29方法对比画廊：原EHE矩阵快照 | analysis-complete / see-original-evidence | 保留6组/15路线/90选帧的方法对比；其中EHE为旧矩阵版本，不代表11月加密矩阵结果。 |
| [nema-ehe-poisson-5e9](studies/reference/nema-ehe-poisson-5e9.md) | NEMA EHE 5e9：矩阵前投影+Poisson参考 | reference-only / see-original-evidence | 原物理响应HOLD保留；执行验收不代表物理偏差消失。历史求和只读，不进入新比较。 |
| [nema-ehe-poisson-5e10](studies/reference/nema-ehe-poisson-5e10.md) | NEMA EHE 5e10：矩阵前投影+Poisson参考 | reference-only / see-original-evidence | 原物理响应HOLD保留；执行验收不代表物理偏差消失。历史求和只读，不进入新比较。 |
| [advisor-progress](studies/reference/advisor-progress.md) | 成像场景、JSCC/EHE设计及既有NEMA结果说明图 | presentation-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [nema-bv-cnr-reference](studies/reference/nema-bv-cnr-reference.md) | NEMA分能量BV–CNR历史图 | analysis-complete / source-bound-derived-metrics | 既有NEMA结果的BV–CNR曲线；EHE为加密矩阵之前的结果，不冒充新1761244曲线。 |
| [ac225-meeting-slides](studies/reference/ac225-meeting-slides.md) | 225Ac成像汇报图与幻灯片材料 | presentation-only / local-presentation | 汇报材料为对应日期的结果快照；最新数值和矩阵版本以当前结果入口为准。 |

## 归档研究

| 查询ID / 入口 | 研究 | 状态 | 结论 |
|---|---|---|---|
| [spike-ablation](studies/history/spike-ablation.md) | 边界绑定、Huber和尖峰历史消融 | stopped-historical / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [response-cut3-old](studies/history/response-cut3-old.md) | 旧response mismatch门控：保留历史身份 | retired-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [fine-response-v3](studies/history/fine-response-v3.md) | 已停止精细响应场与清理证明 | stopped-not-complete / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [history-5e9](studies/history/history-5e9.md) | 5e9：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-nema-body-h60-1e9-1643142](studies/history/history-nema-body-h60-1e9-1643142.md) | NEMA_Body_H60_1e9_1643142：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-nema-body-h60-5e9-1644876](studies/history/history-nema-body-h60-5e9-1644876.md) | NEMA_Body_H60_5e9_1644876：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-nema-body-h60-5e9-bind-f010-formal-1651956-0](studies/history/history-nema-body-h60-5e9-bind-f010-formal-1651956-0.md) | NEMA_Body_H60_5e9_bind_f010_formal_1651956_0：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-nema-body-h60-5e9-huber-weak-formal-1651956-1](studies/history/history-nema-body-h60-5e9-huber-weak-formal-1651956-1.md) | NEMA_Body_H60_5e9_huber_weak_formal_1651956_1：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-process-list-audit](studies/history/history-process-list-audit.md) | process_list_audit：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-process-list-followup-20261004](studies/history/history-process-list-followup-20261004.md) | process_list_followup_20261004：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-spike-research](studies/history/history-spike-research.md) | spike_research：历史研究/诊断目录 | historical-see-evidence / see-original-evidence | 按原验收和记录时间阅读；目录存在不代表新阶段完成，也不授权恢复。 |
| [history-derenzo-218-440-design-20261014](studies/history/history-derenzo-218-440-design-20261014.md) | derenzo_218_440_design_20261014：未采用的历史设计 | design-not-selected / see-original-evidence | 不是本轮输运真值；无对应新模拟完成声明。 |
| [history-derenzo-218-440-design-20261015](studies/history/history-derenzo-218-440-design-20261015.md) | derenzo_218_440_design_20261015：未采用的历史设计 | design-not-selected / see-original-evidence | 不是本轮输运真值；无对应新模拟完成声明。 |
| [history-derenzo-218-440-design-20261015-sparse36](studies/history/history-derenzo-218-440-design-20261015-sparse36.md) | derenzo_218_440_design_20261015_sparse36：未采用的历史设计 | design-not-selected / see-original-evidence | 不是本轮输运真值；无对应新模拟完成声明。 |

完整字段见[catalog.json](catalog.json)，源码和函数查询见[SOURCE_INDEX.md](SOURCE_INDEX.md)。
