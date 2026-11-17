NEMA Body 实际5e10 · 正式重建1722760 · 2026-10-26

作业在2026-10-26 01:49:24正常结束；三路各10000/save50，600个检查点。
实际输出：440 keV单光子SC；218 keV串扰校正SC；440 keV Compton独立重建。
本次没有SC+Compton联合图，也没有218/440跨能量求和。

图表（同名PDF为矢量版）
  figures/final_mip.png                  末帧与真值，共同色阶0–10
  figures/final_mip_unclipped.png        同一数据，共同未截顶色阶0–55.3663
  figures/iteration_mip_gallery.png      真值与100/500/1000/2000/5000/10000次
  figures/central_slices.png             z=+1.5mm中央切片
  figures/cnr_bv_iterations.png          3D热球CNR、统一背景BV随迭代
  figures/bv_cnr_trajectory.png          全部200个正迭代保存帧的BV–CNR轨迹

显示与定量
  中心72mm MIP：24层，z中心−34.5..+34.5mm，体素边界±36mm。
  XY完整视野，不平滑、不裁边、不做单图亮度拟合；gray_r白低黑高。
  显示除以“实际本能量发射数/原真值源积分”，是源单位换算，不是图像拟合。
  5e10为实际gamma初级总数，不是225Ac衰变数。
  主图超过色阶10的值显示黑色，原值保留；饱和像素见display_saturation.csv。
  指标来自完整三维当前ROI，不从MIP计算。球中心距球面至少1.5mm，等权。
  全部球共用97604背景体素，排除六球、肺和接触边界的体素。
  BV=100×空间样本标准差/均值（ddof=1），CNR=(热球均值−背景均值)/标准差。
  0次CNR未定义，CSV留空。负CNR和迭代回折均保留。

实际图像情况
  440 Compton独立重建后期出现明显环/条纹及斑点伪影；未平滑或隐藏。
  因而“作业正常完成/输出一致性通过”不等于物理或分辨率认证。
  runtime节点RSS峰值约78.33%、显存约74.52%；Slurm MaxRSS/单节点配额约
  80.7218%，略超80%目标，不能称全部会计源20%余量通过。未发生OOM。
  原直接正式授权new_topology_full_input_validation_passed=false如实保留。

可追溯数据
  data/metadata.json                    源SHA、实际发射定标、显示与资源记录
  data/roi_definition.json              统一三维ROI和掩膜SHA
  data/{sphere,common_background,native}_iteration_metrics.csv
  data/sphere_endpoint_metrics.csv      各球末帧数值
  data/display_arrays_and_curves.npz    实际显示数组和完整曲线
  data/truth_3mm.npz                    原真值字节副本
  data/scientific_qa.json、visual_qa.json、independent_numeric_qa.json
  scripts/                             精确绘图及辅助源码

复现入口（工程根目录）
  python -B -u -X utf8 tools/research/plot_nema_jscc5e10_20261026.py
  前置要求：strict_transfer.json与local_consistency.json均已完成。
  只读原始结果，所有新图表保存到本目录；无超算连接、模拟或重建。
