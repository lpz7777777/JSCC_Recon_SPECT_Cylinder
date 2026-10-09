# 500×300×120mm椭圆视野：当前218/440双能基准

2026-10-07：以已严格交付的NEMA H60、既有5e9 legacy、连续能量核v5六路10000次作业1669255为后续实现与执行回归基准。旧试验不自动恢复；后续EHE对比独立执行，JSCC已交付结果保持只读。

- [完整基准：Geant4→三套Factors/匹配S→六路MLEM→验收→图像](../../docs/DUAL_ENERGY_BASELINE.md)
- [7月至10月全部主要测试与正面/负面/未定结论](../../docs/DUAL_ENERGY_RESEARCH_REVIEW.md)
- [源代码与输入/证明SHA登记](../../docs/baselines/dual_energy_20261007/manifest.json)
- [代码分类](../../docs/baselines/dual_energy_20261007/source_inventory.json)与[旧脚本删除/恢复清单](../../docs/baselines/dual_energy_20261007/cleanup_manifest.json)

## 新EHE平行孔SPECT对比

2026-10-09：用户另要求[独立EHE实际Geant4 5e10与原方案200次重建](reports/NEMA_Body_H60/ehe_spect_5e10_200/README.md)。唯一输运15683333已在18节点运行1000个独立worker，每worker5000万，实际总剂量5e10；新种子33100101–33101100。全部20份mac和实际已验收可执行文件对应源码的[角度检查](reports/NEMA_Body_H60/ehe_spect_5e10_200/SOURCE_ANGLE.md)确认全4π发射、一事件一光子，剂量倍数1；`/xcat/angle`仅旋转源位置，没有半立体角限制。已冻结部署原MLEM后续流程，等待全量输运验收后执行validation10→formal200→严格取回和新图集；本实验尚未交付。一次有界本地流程推进，原定时任务保持暂停。

用户已授权[独立EHE 5e9/200次实验](reports/NEMA_Body_H60/ehe_spect_5e9_200/README.md)。保留1250孔/2312 NaI bin及272×136mm探测面，前表面298.5mm；重新模拟EHE观测、生成三套匹配响应与S，先做真实源物理门控及完整输入10次验证，随后正式200次。对照只读复用1669255的218单光子、440单光子/Compton/JSCC和双能结果。2026-10-09已按用户明确授权完成原方案200次及严格取回/科学视觉QA，见[最终报告](reports/NEMA_Body_H60/ehe_spect_5e9_200/RESULTS.md)和[28张图表](reports/NEMA_Body_H60/ehe_spect_5e9_200/comparison_200/gallery.md)。实际EHE有明显斑点和高背景波动，原物理审计仍HOLD，定时任务保持暂停。

2026-10-08：EHE输运15633840的200个worker全部成功退出，实际5e9、20视角、种子31100101–31100300、初级标签及2424份取回/同步文件SHA通过[完整输运身份验收](reports/NEMA_Body_H60/ehe_spect_5e9_200/transport_identity_acceptance.json)。实测218/440窗计数22929/12171，218窗中9869个来自440初级gamma，见[测量证据](reports/NEMA_Body_H60/ehe_spect_5e9_200/transport_measurement.json)。三套响应1672966仍在生成，真实源物理门控、validation10、formal200和图像对比尚未完成；输运身份通过不代表响应物理通过。

2026-10-09 06:17：EHE完整存储转换已交付，但[真实源物理门控1677211触发科学HOLD](reports/NEMA_Body_H60/ehe_spect_5e9_200/PHYSICAL_HOLD.md)：C440→218全局低估24.02%，19视角及全局HOLD，另A218两个视角HOLD。原证据严格取回，重建未提交；保留结果并只读诊断，不将其当成共享盘故障或完整实验交付。

2026-10-09 06:59：源密度、旋转和能窗只读核对未发现明显误配；现有完整响应矩阵内的3mm源盒平均仅使C预测变化−0.004428%，无法解释约24%偏差。新分析实际退出0并严格取回，生产输入和原HOLD保持；更细响应网格/积分/散射历史覆盖仍未确定，见同一HOLD报告。

2026-10-09 07:47：新的有限孔准直器条件路径分析确认，完整均匀Pb板深度衰减与部分有孔射线的实际Pb弦长不同；没有全源/路径权重支持24%缺口归因。分析实际退出0并保存源码/输入/结果SHA，原HOLD与重建停止保持，详见同一HOLD报告。

2026-10-09 08:19：新的Pb逐格面积诊断绑定原12份日志，解析Pb面积48314.076848mm²与原记录48324.4mm²的总量差约0.0213%。默认CPU参考的面积权重在固定非负逐格贡献下有6.9905%的重赋上界（采样总计作分母），不含代表点/深度/角度/历史误差，也未捕获运行时逐格状态；不能据此解除24%串窗缺口。原HOLD保持，未生成响应或提交重建，详见同一HOLD报告。

2026-10-09 08:47：新的原运行散射摘要确认全部12块包含局部/准直器项，C准直器项在未加权Cartesian散射数组总和中的份额约0.023116%，不表示真实源或实测份额。原近8×8/远1×1采样配置下，CPU几何分类远对占96.8642%，没有其响应权重支持积分误差结论。独立分量矩阵未保存，本轮未生成响应或改生产输入；原HOLD保持，详见同一科学报告。

## 已交付的当前结果

2026-10-09：用户授权保留EHE已有结果并停止共享盘转写缓慢的1672966，作业已完全退出。原12个计算块、A218/A440及部分C输出均保留。另冻结仅修改存储转换的b02df58a9d3fd48c，以唯一真实I/O探针1677092核对全SHA、数值及吞吐，随后只补剩余转换；科学发布、输运、物理门控与成像预算不变。最新状态见EHE运行簿和转换作业登记。

[1669255完整执行与科学验收](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/ACCEPTANCE.md)：六路各10000次/200帧、600阶段检查点、3022文件逐SHA取回；8节点×1GPU/bond0，12:00:33；GPU预留63.69%、RSS42.14%、Slurm MaxRSS59.94%。第2000次Compton/JSCC对1667869 B逐值/SHA一致，L2=0。

[六路图集及全部200帧曲线](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/comparison_1669255/README.md)；[218单光子、440单光子、440 Compton、440 JSCC的随迭代图](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/four_channel_iterations_1669255/README.md)，包含真值及100/500/1000/2000/5000/10000的轴/冠/矢位、中央72mm MIP，固定尺度/no smoothing/crop0。

2000→10000使部分大球CRC及泄漏改善，同时所有通道峰值和背景CV上升；Compton13mm、两种双能和10mm有CRC损失>5pp。实现与执行验收通过，不等于整个FOV或小球性能通过。独立legacy校准446联合类别仍未判定。

## 固定流程边界

实际5e9、20视角、200worker×25M、种子30100101–30100300；共同483743 stable_float64 q≤3事件、132040完整圆网格、78920完整活动柱单元。材料规律、legacy连续S、三套完整Factors、原MLEM及全1初值冻结；218固定背景来自本次最终440单光子。六路是218单光子/440单光子/其密度和/440 Compton/440 JSCC/JSCC440与218密度和，不是Ac225母核活度。

当前规范入口为energy_full10000_v5_workflow.py与run_energy_full10000_v5.py。已完成登记禁止重复freeze/deploy/submit；旧2k入口保留拒绝10000的保护，旧分数82040入口仅兼容/历史复现。默认只读检查：

```powershell
python -X utf8 tools/baseline/verify_dual_energy_baseline.py
```

## 历史证据

- [5e9两核2000：1667869](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9/ACCEPTANCE.md)及[legacy独立校准](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9/CALIBRATION_ACCEPTANCE.md)
- [ideal1e9两核2000：1666673](reports/NEMA_Body_H60/compton_energy_probability_v5/ACCEPTANCE.md)，事件策略不同，不能作为纯剂量对照
- [首散射v2配对](reports/NEMA_Body_H60/compton_first_scatter_v2/ACCEPTANCE.md)、[全局物理/数值审计](reports/NEMA_Body_H60/process_list_global_audit_v4/REPORT.md)
- [绑定/Huber结果与停止](reports/NEMA_Body_H60/spike_ablation/README.md)、[精细A停止/清理证据](reports/NEMA_Body_H60/compton_response_geometry_v3/STOP_AND_CLEANUP_20261005.md)
- [本次整理前README原字节快照](HISTORY_20261007.md)及[更早历史](HISTORY.md)

历史快照中的排队/运行/待完成是当时状态，不能作为现在恢复任务的依据。原始Factors、输运数据、图像历史、大归档保持只读并不入Git；未改写任何已验收报告或冻结SHA。

2026-10-09 09:19上海：新的晶体可见面固体角检查覆盖全部有序晶体对，仅几何因子在原近8×8/远1×1下的误差范围分别−0.143%至+0.265%、+0.012%至+3.831%。窗、衰减和吸收的面内变化尚未定量，这不能解释或排除完整串窗响应的24%缺口；未生成响应/提交重建，原HOLD保持，详见[科学报告](reports/NEMA_Body_H60/ehe_spect_5e9_200/PHYSICAL_HOLD.md)。

2026-10-09 09:48上海：新的晶体间路径几何检查发现原表面函数在改变射线后仍使用中心中间段，完整箱体交集参考核对全部7042节点和2311中心段。固定表达式与几何NaI总段差值近−4.889至+0.264mm、远−1.978至0mm；未计算贡献权重，不能归因24%串窗低估。原HOLD保持，未生成响应/提交重建，详见[科学报告](reports/NEMA_Body_H60/ehe_spect_5e9_200/PHYSICAL_HOLD.md)。
