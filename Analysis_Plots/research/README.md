# 按日期保留的图形与诊断专题

当前版本和条件从[结果目录](../results/REPORT.html)查询；这里保留专题原路径。figures为显示产物，data为来源/指标/QA，scripts为实际绘图副本。

2026-11-17新增统一入口：[HE Derenzo完成](../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/COMPLETED_20261117.md)、[展宽600帧](nema5e9_width_sweep_20261115/index.html)、[带水v2综合](ac225_water_correction_overview_20261108/REPORT.html)、[Derenzo9950失败预览](derenzo_joint5e10_1771220_20261112/REPORT.html)。下表为各日期专题范围，不把旧子集/旧ACTIVE提升为当前全组状态。全部源码/报告与阶段证据可从[74项研究索引](../../docs/ac225/EXPERIMENTS.md)反查。

| 专题 | 用途与结论 | 当前地位 |
|---|---|---|
| [NEMA CNR／CRC／BV：ROI与体模对齐审计](nema_roi_audit_20261115/index.html) | 3组实际截面／三维／体素放大图，原六球和共同97604背景mask逐bit一致；132036行指标和实际源核对通过。明确球中心规则、partial cube、10mm仅8体素、背景54mm及无滤波guard band；旧ROI与曲线不变。 | completed-ROI-audit |
| [NEMA CRC／CNR–E_BG+out](nema_bg_out_error_20261115/index.html) | 5e9／5e10、218／440、五档高斯；24组PNG／PDF，10000个E、60000球点、2000原生σ0体外RMS。真背景1，内背景与含壁包络外域各50%；主图采用共同3 mm图像场，原生仅另列E_out；独立数值／视觉通过，物理限制保留。 | completed-derived-diagnostic |
| [NEMA SC／JSCC／HE：五档3D高斯后处理](nema_sc_jscc_he_gaussian_20261115/index.html) | 5e9／5e10、218／440，σ=0／1.5／3／4.5／6 mm；4组末次72mm MIP、16张完整迭代曲线、4张滤波强度轨迹及无截顶伴图。2000帧、60000球点数值／视觉核对通过；旧未滤波结果不变。 | completed-derived-postprocessing |
| [NEMA SC／JSCC／HE：CNR–BV及CRC–BV](nema_sc_jscc_he_cnr_crc_bv_20261114/index.html) | 5e9／5e10、218／440，4张热球主图＋4张冷球补充图；10条路线、12000个保存球点，只读复用原3D ROI。CRC为对比恢复，冷球CNR保留符号，BV含背景非均匀性；各自实际迭代顺序、无平滑。 | completed-curve-comparison |
| [NEMA SC／JSCC／HE：5e9与5e10](nema_sc_jscc_he_5e9_5e10_20261114/index.html) | 440为SC/JSCC/HE，218为SC/HE；中心72mm，4张末次／迭代主图及BV–CNR、球恢复；SC/JSCC10000、HE200，保留模型和网格差异。 | completed-comparison |
| [HE／EHE无水NEMA 5e9及5e10对比](opengate_he_vs_ehe_nema_20261114/index.html) | HE四路各200次、800帧严格交付；同框架EHE与原加密EHE参考，20组末次／迭代／BV–CNR／球恢复图。HE降低BV，部分小球恢复下降。 | completed-with-limitations |
| [OpenGATE/PyTomography无水NEMA实际5e10](opengate_ehe_nema5e10_20261113/index.html) | 独立50B、20角、218/440各200次；含原EHE及5B/50B对比，数值/视觉通过，物理限制保留。 | completed-with-limitations |
| [OpenGATE/PyTomography无水NEMA实际5e9](opengate_ehe_nema5e9_rotfix_20261113/index.html) | 修复源旋转后的有效5B；400帧，边缘高值仍在，未整体改善。 | completed-with-limitations |
| [带水NEMA已完成SC分支](ac225_water_completed_20261105/REPORT.html) | 440 keV：14路/2800帧严格取回；两探头最终及迭代72mm MIP、BV–CNR，含未截高值伴图。灰格为未完成分支，218未展示；无平滑/拟合，尚非完整32路或物理认证。 | completed-subset |
| [nema_joint5e10_1740189_20261029](nema_joint5e10_1740189_20261029/REPORT.html) | 1740189同440keV联合10000次完成；背景BV约35.94%，低于SC约40.50%，但13mm热球CNR下降；不说明环伪影成因已修正。 | current |
| [figures](derenzo_completed_5e9_20261019/figures/final_mip.png) | 每个探头/能量独立5e9 gamma；两个源不是同一次225Ac衰变链，218背景为零。 | current |
| [figures](derenzo_jscc440_5e9_20261023/figures/final_mip.png) | 每个探头/能量独立5e9 gamma；两个源不是同一次225Ac衰变链，218背景为零。 | current |
| [derenzo_218_440_design_20261015_equal_regions](derenzo_218_440_design_20261015_equal_regions/design_notes_zh.txt) | 当前输运唯一解析几何：218为6/8/10/12/14/16mm、114棒；440为10/12/14/16/18/20mm、53棒。 | current |
| [figures](derenzo_218_440_design_20261015_equal_regions/figures/derenzo_440_218_design.png) | 当前输运唯一解析几何：218为6/8/10/12/14/16mm、114棒；440为10/12/14/16/18/20mm、53棒。 | current |
| [derenzo_completed_5e9_20261019](derenzo_completed_5e9_20261019/README.txt) | 以原始报告的研究范围和结论为准。 | historical-reference |
| [figures](nema_structure_20261014/figures/nema_structure_218_440.png) | 原3mm三维NEMA H60真值：六球与中心肺；218热球10/17/28mm，440热球13/22/37mm。球心及完整能量分布按原manifest/数组。 | current |
| [advisor_progress_20261014](advisor_progress_20261014/使用说明.txt) | 以原始报告的研究范围和结论为准。 | historical-reference |
| [figures](advisor_progress_20261014/figures/01_imaging_scene.png) | 以原始报告的研究范围和结论为准。 | historical-reference |
| [derenzo_218_440_design_20261014](derenzo_218_440_design_20261014/README.md) | 不是本轮输运真值；无对应新模拟完成声明。 | historical-reference |
| [derenzo_218_440_design_20261015](derenzo_218_440_design_20261015/README.md) | 不是本轮输运真值；无对应新模拟完成声明。 | historical-reference |
| [derenzo_218_440_design_20261015_sparse36](derenzo_218_440_design_20261015_sparse36/README.md) | 不是本轮输运真值；无对应新模拟完成声明。 | historical-reference |
| [ehe_side_shield_design_20261029](ehe_side_shield_design_20261029/REPORT.html) | 以原始报告的研究范围和结论为准。 | historical-reference |
| [ehe_dense_completed_1761244_20261102](ehe_dense_completed_1761244_20261102/REPORT.html) | A218/A440提高到65536点/可见面，匹配自身S；200次正式完成，但边缘高值、散斑和小球对比不足仍存在。 | current |
| [ehe_dense_nema_low_noise_20261102](ehe_dense_nema_low_noise_20261102/REPORT.html) | 匹配模型自生数据在5e9/5e10也出现明显边缘伪影；提高计数显著减弱尖峰，但高剂量仍有模糊。 | current |
| [ehe_dense_nema_60view_poisson_20261102](ehe_dense_nema_60view_poisson_20261102/REPORT.html) | 同总gamma、200次下，60角外围ROI发射量占比8组均降低；440在低两档全域NRMSE略升，高剂量误差仅小幅改善。 | current |
| [nema_compton_ring_diagnosis_20261026](nema_compton_ring_diagnosis_20261026/REPORT.html) | 环纹与网格/响应离散化值得隔离研究；观察性关联不足以确定唯一成因。 | diagnostic |
| [nema_compton_transition_audit_20261026](nema_compton_transition_audit_20261026/REPORT.html) | 未发现简单漏乘/重复乘体积或S约定错误；中心点响应代表有限单元的积分误差仍是候选。 | diagnostic |
| [nema_compton_git_history_20261026](nema_compton_git_history_20261026/REPORT.html) | 源码版本、核/S/筛选/求解均有差别；历史图像外观不能代替同源匹配实验。 | diagnostic |
| [legacy_compton_backup_audit_20261026](legacy_compton_backup_audit_20261026/REPORT.html) | 备份Image_JSCCD为纯Compton；511keV旧数据与440keV近期NEMA/FOV不同，不当作直接等价基准。 | diagnostic |
| [nema5e9_new_vs_prejune_20261026](nema5e9_new_vs_prejune_20261026/REPORT.html) | 旧流程后期起伏更强，回退未消除环纹；同时改变多步骤，不能唯一归因。 | diagnostic |
| [ehe_crystal_response_20261029](ehe_crystal_response_20261029/REPORT.html) | 原矩阵六个晶体的分能量MIP/切片；晶体响应覆盖及面采样问题另由准直器审计解释。 | diagnostic |
| [ehe_collimator_response_audit_20261029](ehe_collimator_response_audit_20261029/REPORT.html) | 256点/可见面的局部积分不足已确认；65536点加密版本独立完成，物理模型仍需验证。 | diagnostic |
| [ehe_fov_clearance_20261102](ehe_fov_clearance_20261102/REPORT.html) | 准直器前端298.5mm，完整旋转FOV最小间隙48.5mm，未伸入FOV。 | diagnostic |
| [ehe_dense_edge_diagnosis_20261102](ehe_dense_edge_diagnosis_20261102/REPORT.html) | 噪声放大与空间响应失配并存；NEMA440→218串窗预测低约25.27%，加密没有消除主要尖峰。 | diagnostic |
| [ehe_edge_diagnosis_20261019](ehe_edge_diagnosis_20261019/README.txt) | 保留早期边缘/S/计数诊断；当前结论优先使用11月加密矩阵及匹配Poisson对照。 | diagnostic |
| [ac225_meeting_slides_20261029](ac225_meeting_slides_20261029/output) | 汇报材料为对应日期的结果快照；最新数值和矩阵版本以当前结果入口为准。 | historical-reference |
| [derenzo_jscc440_5e9_20261023](derenzo_jscc440_5e9_20261023) | 按原目录与日期说明阅读 | 日期限定专题 |
| [nema_bv_cnr_218_440_20261023](nema_bv_cnr_218_440_20261023) | 按原目录与日期说明阅读 | 日期限定专题 |
| [nema_jscc5e10_20261026](nema_jscc5e10_20261026) | 按原目录与日期说明阅读 | 日期限定专题 |
| [nema_prejune_compton_1739632_20261026](nema_prejune_compton_1739632_20261026) | 按原目录与日期说明阅读 | 日期限定专题 |
| [nema_structure_20261014](nema_structure_20261014) | 按原目录与日期说明阅读 | 日期限定专题 |

未采用的Derenzo设计不作为输运真值。旧BV–CNR中的EHE属于加密前矩阵；新模型BV–CNR只按实际选帧计算。历史图形存在不代表当前物理认证。
