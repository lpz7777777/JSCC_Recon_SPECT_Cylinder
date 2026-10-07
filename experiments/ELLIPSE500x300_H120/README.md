# 500×300×120mm椭圆视野：当前218/440双能基准

2026-10-07：以已严格交付的NEMA H60、既有5e9 legacy、连续能量核v5六路10000次作业1669255为后续实现与执行回归基准。旧试验不自动恢复，compton-v5保持暂停。

- [完整基准：Geant4→三套Factors/匹配S→六路MLEM→验收→图像](../../docs/DUAL_ENERGY_BASELINE.md)
- [7月至10月全部主要测试与正面/负面/未定结论](../../docs/DUAL_ENERGY_RESEARCH_REVIEW.md)
- [源代码与输入/证明SHA登记](../../docs/baselines/dual_energy_20261007/manifest.json)
- [代码分类](../../docs/baselines/dual_energy_20261007/source_inventory.json)与[旧脚本删除/恢复清单](../../docs/baselines/dual_energy_20261007/cleanup_manifest.json)

## 已交付的当前结果

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
