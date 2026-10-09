# EHE 0-200 / JSCC 0-10000 整体对比

主总图：[72mm MIP](overall_mip72.png)。PDF为7页完整图集：[整体PDF](overall_comparison.pdf)。
图像总览另有[轴位](overall_axial.png)、[冠状位](overall_coronal.png)、[矢状位](overall_sagittal.png)。
覆盖EHE三路与JSCC六路，按218、440、双能和排列。每行左为实际三维真值，中为自身迭代轨迹，右为完整3D球ROI CRC/CNR。

EHE图像节点为0/10/20/50/100/150/200；JSCC为0/500/1000/2500/5000/7500/10000。
曲线分别使用20/200个实际已保存帧，横轴分别0-200/0-10000。同列只用于排版，不表示同等迭代、进度或收敛。
末帧仅是各自请求的预算终点，不称两者已同等收敛。

**0次来源**：执行冻结源码SHA核对后的全1 gamma/mm³初值；双能相加为2，域外为0。
没有保存0次快照，图中明确标注init，不冒充已保存重建帧。均匀初值CNR的背景标准差为0，因此不定义/不绘制CNR0；峰位置不唯一也留空。CRC0为0。
固定真值背景尺度下初值很淡是实际尺度结果，不调亮。

所有图使用各系统本次实际初级gamma/真实源积分确定的固定尺度，gray_r 0-10，crop0，无平滑/逐图拟合亮度。
轴位z=+1.5mm，冠状y=-1.5mm，矢状x=-1.5mm；MIP中心72mm只用于显示。
原生指标覆盖完整120mm；球指标用manifest现有3D球ROI。三维真值SHA及源尺度见comparison_metadata.json。

原生全轨迹：[密度与噪声](native_density_curves.png)、[积分、泄漏与峰位置](native_integral_position_curves.png)。
左右横轴独立，每项指标纵轴在两系统间保持一致。峰位置变化可能来自不同最大像素，不等同运动。
数据：[原生CSV](all_native_metrics.csv)、[球ROI CSV](all_sphere_metrics.csv)、[各自终点CSV](budget_endpoints.csv)。
所有正迭代指标逐字段保持原验收CSV数值，不重新估计/筛选曲线。
[计数与终点表](data_and_endpoints.png)给出实际计数及各自终点。

EHE218背景来自EHE440末图200；JSCC218背景来自JSCC440末图10000。两边实际计数未匹配，探测结构/材料/覆盖差异不能全归因算法。
原EHE物理审计HOLD保留；用户已授权按原方法继续，该图集不改变科学结论或生产模型。旧JSCC没有初级能量标记的实测串窗计数。
本次只从本地已验收数据生成图表，不运行模拟、系统矩阵、重建或远端作业。compton-v5保持PAUSED。
执行代码、输入SHA、图像节点映射及科学检查随图归档；visual_qa.json为实际图像/PDF渲染复核，artifact_manifest.json封存输出SHA。

已保存帧中的CNR最大值另存[峰值表](cnr_peaks_saved_frames.csv)，不插值未知迭代，也不当作收敛或停迭代证明。
EHE各球/路峰值位于10-40次；JSCC各球/路位于150-10000次，具体取决于路线和球大小。
例如28mm球218校正单光子：EHE保存帧峰值20次，CNR=4.2233；JSCC峰值6500次，CNR=6.7836。
这些是各自完整已保存轨迹上的质量指标，不据此假定两者相同迭代或末帧相同收敛。
独立科学复核逐字段核对59220个正迭代指标值，保持原值，详见scientific_qa.json。
