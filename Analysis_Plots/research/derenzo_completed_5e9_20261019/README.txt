Derenzo已完成重建可视化（2026-10-19）

仅展示已正式完成、独立验收并严格取回的JSCC218（10000次）、EHE218/440（200次）；每组实际5e9单能gamma，不是Ac225衰变次数。

figures/final_mip.png：真值与完整120mm轴向MIP，统一0–2色标；超过2显示黑色，原值未截断。
figures/final_mip_unclipped.png：同一数据的统一、不截顶色标版本。
figures/final_central_slices.png：z=+1.5mm中心切片。
figures/iteration_mip_gallery.png：实际保存帧的迭代变化；JSCC100/500/1000/2000/5000/10000，EHE2/10/20/40/100/200。
figures/native_convergence.png：完整原生三维数组的发射积分与峰值变化。
每张图同时保存PNG和PDF。data/包含精确源几何生成的3mm真值、显示投影NPZ、全部200帧的原生指标、SHA与科学QA。

显示以实际发射数/解析棒体积确定固定尺度，真值棒内=1；不进行单图亮度拟合、平滑或跨能量叠加。白色低、黑色高。XY采用工程既有重心线性显示采样，轴向不插值；显示3mm不改变原生6mm径向分辨单元。完整FOV500×300mm，显示栅格在x方向两边各外延2mm，保留边界插值外延，不施加椭圆显示掩膜，crop0。插值浮点舍入产生的低于1e-8尺度的负数仅在派生显示数组中置零，数量及最小值逐一记录，原生数组未修改。
EHE原响应仍有物理标定限制，峰值和积分不能替代独立绝对活度校准；不同迭代数不等于相同收敛程度。最小6mm棒受本轮粗网格限制，不据此直接宣布6mm分辨率。

复现：python scripts/plot_derenzo_completed.py --project-root <工程绝对路径> --output <本目录绝对路径>
