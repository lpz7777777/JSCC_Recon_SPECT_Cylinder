# 结果按条件查询

[可搜索结果](../../Analysis_Plots/results/REPORT.html)与[conditions.csv](../../Analysis_Plots/results/data/conditions.csv)包括完成、运行、失败预览和未交付条件。按体模、gamma总发射量、能量、方法、网格、材料、展宽、视角、投影来源与校正分支查询。

```powershell
python -B -X utf8 tools/project_index.py results --phantom Derenzo --method HE
python -B -X utf8 tools/project_index.py results --phantom NEMA --material water
python -B -X utf8 tools/project_index.py results --grid Cartesian4mm --topic 1822534
```

优先入口：[SC/JSCC/HE NEMA对比](../../Analysis_Plots/research/nema_sc_jscc_he_5e9_5e10_20261114/index.html)、[HE Derenzo完成](../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/COMPLETED_20261117.md)、[展宽研究](../../Analysis_Plots/research/nema5e9_width_sweep_20261115/index.html)、[带水散射v2](../../Analysis_Plots/research/ac225_water_correction_overview_20261108/REPORT.html)。

results/current仅存条件说明和case.json。原研究figures/data/scripts及artifact SHA原位保留。旧15路画廊的EHE为旧矩阵，加密EHE、OpenGATE EHE、HE分别标记。Derenzo联合9950为失败预览；Cartesian两个任务并未产生已交付图像。[阶段依据](CURRENT_STATUS.md)。

合成20/60角Poisson只本地交付48选帧、远端核对1600帧，不等于全历史本地QA。带水JSCC探头的修正分支仍为SC。HE/EHE200次、SC/JSCC10000次、部分9950次分别标明，不把同迭代当同收敛。可视化不构成物理/分辨率认证。
