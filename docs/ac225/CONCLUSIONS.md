# 225Ac代理成像研究结论

以下区分计算完成、数值一致性、图像表现和物理解释。SC为单光子；JSCC为同能量SC+Compton联合。研究使用218/440gamma代理源，未模拟完整225Ac衰变链。

## NEMA与JSCC

NEMA实际5e9的v5基准、实际5e10的SC/纯Compton三路，以及5e10的440联合1740189均有已交付结果。联合图在该条件下背景BV约35.94%，比SC约40.50%降低，但13mm热球CNR下降，说明噪声下降与小目标对比不是同一指标。纯Compton后期环纹、条纹和尖峰仍明显。

[5e10联合结果](../../Analysis_Plots/research/nema_joint5e10_1740189_20261029/REPORT.html) · [原三路](../../Analysis_Plots/research/nema_jscc5e10_20261026/README.txt)

## Compton环纹与不等体积网格

角点数变化的实际单元边界为r=39、75、111mm等；部分峰接近过渡带，并非全部峰恰在边界。当前未知量是密度，Cartesian→Polar时响应已乘单元体积，Compton事件归一化及匹配S沿用同一体积约定。实际审计未发现简单漏乘、重复乘体积或统一倍数错误。角点数突变和中心点近似有限单元响应的积分误差仍是值得隔离的候选机制，不能仅从外观确定成因。

使用近期同一NEMA投影回退旧Compton核/配套S/OSEM8后，后期伪影仍很强。旧OSEM8每轮含8次子集更新，且使用完整S的约定影响总体尺度；它不是与当前MLEM同更新次数的单因素试验，更不是对旧511keV结果的精确重放。

[过渡和体积审计](../../Analysis_Plots/research/nema_compton_transition_audit_20261026/REPORT.html) · [新旧流程对比](../../Analysis_Plots/research/nema5e9_new_vs_prejune_20261026/REPORT.html) · [历史备份审计](../../Analysis_Plots/research/legacy_compton_backup_audit_20261026/REPORT.html)

## EHE响应与边缘高值

准直器前端距中心298.5mm，旋转完整FOV的最小间隙48.5mm，不支持“前端伸入FOV”这一解释。256点/可见面的局部面积分误差确实存在；A218/A440已提高到65536点并重算自身S。但同一Derenzo实际投影的边缘峰和最高点位置变化很小，加密没有消除主要伪影。

低计数、78920活动单元、无正则MLEM和边缘灵敏度较低共同支持噪声放大的解释；实际投影与响应也有明确空间失配。NEMA5e10真实440→218贡献100326，当前C预测74978.5，约低25.27%；重建固定背景同样低约25%。这与面积分加密是不同问题。物理响应尚未闭合，不把数值S匹配当作物理认证。

[完整边缘诊断](../../Analysis_Plots/research/ehe_dense_edge_diagnosis_20261102/REPORT.html) · [FOV间隙](../../Analysis_Plots/research/ehe_fov_clearance_20261102/REPORT.html) · [准直器审计](../../Analysis_Plots/research/ehe_collimator_response_audit_20261029/REPORT.html)

## 前投影与Poisson计数对照

20视角下，生成和重建匹配的原生矩阵合成数据在5e9/5e10仍重现明显边缘高值；提高到5e12后尖峰显著减弱，但仍有模糊。这说明这一类伪影不需要真实Geant4/响应失配作为前提，同时也不排除真实数据已确认的模型偏差。

60视角对照保持每档总期望gamma量不变，曝光按1/60分配。新6°响应来自同环周期线性插值，并重新计算匹配S；原20位置每隔3视角精确保留。这一次独立噪声实现中，外围ROI发射量占比8组均下降；整体误差改善有限，440在5e9/5e10反而略升。相同200次不保证同等收敛。

| 总期望gamma | 440原生全域NRMSE 20→60 | 218原生全域NRMSE 20→60 |
|---|---|---|
| 5e9 | 297.08% → 303.35% | 364.99% → 361.70% |
| 5e10 | 86.83% → 91.74% | 108.47% → 104.84% |
| 5e11 | 35.73% → 35.40% | 38.44% → 37.70% |
| 5e12 | 26.30% → 25.53% | 24.84% → 23.07% |

角采样、响应插值和Poisson实现同时改变，没有重复噪声实验，不能把差异单独归因于角度数量或宣布统计显著性、分辨率提升。两组均远端核对1600帧，本地严格取回48选帧；完整本地历史QA仍为false。

[20角模型对照](../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/REPORT.html) · [20/60角比较和原始指标](../../Analysis_Plots/research/ehe_dense_nema_60view_poisson_20261102/REPORT.html)

## Derenzo与当前边界

5e9的440 SC/JSCC、218 SC及EHE两能量已完成，当前实际EHE图优先采用加密版本。218与440分别使用不同棒径和布局，不能将它们视为同一几何的纯能量对照。原径向6mm网格可能限制最小6mm棒的判断，未认证6mm空间分辨率。

2026-11-17证据更新：实际5e10的218 SC已交付，440 SC成功200帧于11月12日严格取回；联合续算1771220失败后只到9950/10000，共199帧可作预览，不能标正式10000次完成。EHE实际5e10两能量400帧已于11月4日严格交付并完成图形QA，替代旧“传输中断/GPU未提交”判断。原失败、续算与64卡旧请求均保留历史，不恢复。

## 新网格、HE及修正研究的结论范围

HE NEMA及HE Derenzo四路800帧分别已有严格交付与图形QA；HEGP孔规格+现有NaI并非完整商用相机。HE背景BV降低伴随部分小球恢复下降，有限板/孔相位CDR、原生4mm网格等差异不支持纯准直器单变量因果或分辨率认证。Cartesian独立网格1822462和4090 r2的1822534均触发原80%实际资源守卫失败，尚不能对其图质作结论。

带水首轮32路完整交付不代表全组视觉QA完成，原metadata仍为pending；独立剂量供体散射v2的8路严格交付保留同体模布局先验，体内散射和原探头串窗偏差分开解释。展宽100/80/60%三档均完成，缩窄未改善整体SSIM/NRMSE，不据此更换生产物理模型。

[当前证据和限制](CURRENT_STATUS.md) · [HE NEMA](studies/nema/opengate-he-nema.md) · [带水v2](studies/nema/water-scatter-v2.md) · [展宽](studies/nema/nema-width-sweep.md)

[Derenzo设计](studies/design/derenzo-design-selected.md) · [实际结果目录](../../Analysis_Plots/results/REPORT.html) · [待处理项](CURRENT_STATUS.md)
