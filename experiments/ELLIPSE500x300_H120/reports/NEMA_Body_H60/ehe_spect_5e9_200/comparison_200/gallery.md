# EHE / JSCC：200次图集与完整曲线

[结果报告](../RESULTS.md) · [科学QA](../comparison_scientific_qa.json) · [视觉QA](../comparison_visual_qa.json) · [全部文件SHA](artifact_manifest.json)

本次EHE作业1679415三路各20帧；JSCC只读参考1669255。主图50/100/150/200次。使用真实H60三维球体真值；crop0、无平滑、无单图拟合。按实际发射源gamma密度归一到真值尺度，全部图固定显示0–10，白低黑高。坐标为mm。

轴位z=+1.5mm、冠位y=−1.5mm、矢位x=−1.5mm；MIP仅显示中央72mm（z层8–31，边界−36至+36mm），原生定量指标仍使用完整120mm。EHE218固定背景来自本次440200；JSCC218图的固定背景来自旧44010000。

图表忠实呈现实际结果，不意味着图像质量或物理校准通过；EHE有明显斑点和高背景波动，原物理审计仍HOLD，继续计算依据人类明确授权。

| 类别 | 轴位 | 冠位 | 矢位 | 中央72mm MIP |
|---|---|---|---|---|
| 218校正单光子 | [axial](218_axial.png) | [coronal](218_coronal.png) | [sagittal](218_sagittal.png) | [mip72](218_mip72.png) |
| 440单光子/SC Compton/JSCC | [axial](440_axial.png) | [coronal](440_coronal.png) | [sagittal](440_sagittal.png) | [mip72](440_mip72.png) |
| 双能单光子和/JSCC440+218和 | [axial](dual_axial.png) | [coronal](dual_coronal.png) | [sagittal](dual_sagittal.png) | [mip72](dual_mip72.png) |
| JSCC2000/10000参考 | [axial](jscc_long_reference_axial.png) | [coronal](jscc_long_reference_coronal.png) | [sagittal](jscc_long_reference_sagittal.png) | [mip72](jscc_long_reference_mip72.png) |

同迭代不等于同收敛程度；JSCC2000/10000为补充参考，不重跑旧实验。

| 类别 | 原生指标，10–200次 | 球ROI CRC/CNR，10–200次 | 全历史原生指标 | 全历史CRC/CNR |
|---|---|---|---|---|
| 218 | [指标](218_native_curves.png) | [CRC/CNR](218_crc_cnr.png) | [EHE20帧 / JSCC200帧](218_full_history_native_curves.png) | [EHE20帧 / JSCC200帧](218_full_history_crc_cnr.png) |
| 440 | [指标](440_native_curves.png) | [CRC/CNR](440_crc_cnr.png) | [EHE20帧 / JSCC200帧](440_full_history_native_curves.png) | [EHE20帧 / JSCC200帧](440_full_history_crc_cnr.png) |
| dual | [指标](dual_native_curves.png) | [CRC/CNR](dual_crc_cnr.png) | [EHE20帧 / JSCC200帧](dual_full_history_native_curves.png) | [EHE20帧 / JSCC200帧](dual_full_history_crc_cnr.png) |

完整数据：

- [EHE三路原生指标60行](ehe_native_iteration_metrics.csv)
- [EHE三维球ROI指标240行](ehe_sphere_iteration_metrics.csv)
- [JSCC六路原生指标1200行](jscc_native_iteration_metrics.csv)
- [JSCC六路球ROI指标4800行](jscc_sphere_iteration_metrics.csv)
- [20视角实际窗计数与固定背景](ehe_window_and_background_by_view.csv)
- [计数、串扰、灵敏度与显示参数](comparison_report.json)
- [20视角几何投影截断](projection_truncation.json)
- [原物理审计](physical_gate.json)与[继续执行授权](physical_continuation_policy.json)
