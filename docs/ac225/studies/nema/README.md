# NEMA实际成像

| 研究 | 版本 | 计算 / 交付 | 结论 |
|---|---|---|---|
| [NEMA实际5e10：新v5流程440 SC+Compton联合重建](nema-5e10-v5-joint.md) | current | completed / strict-complete-history | 1740189同440keV联合10000次完成；背景BV约35.94%，低于SC约40.50%，但13mm热球CNR下降；不说明环伪影成因已修正。 |
| [NEMA实际5e10：SC218/440与纯Compton440](nema-jscc-5e10.md) | current | completed / strict-complete-history | 1722760三路各10000次已交付；纯Compton后期环状、条纹与尖峰明显。纯Compton不命名为JSCC。 |
| [NEMA JSCC 5e9 连续核v5：历史10000次正式基准](nema-jscc-5e9-v5.md) | current | completed / strict-historical-delivery | 历史六路含已退休跨能量求和；后续仅采用获授权的分能量通道和当前ROI。 |
| [NEMA原EHE矩阵 5e9：实际Geant4](nema-ehe-g4-5e9.md) | historical-reference | delivered / see-original-evidence | 旧矩阵真实Geant4重建已交付；当前EHE实际比较优先使用11月dense65536独立重建。 |
| [NEMA原EHE矩阵 5e10：实际Geant4](nema-ehe-g4-5e10.md) | historical-reference | delivered / see-original-evidence | 旧矩阵真实Geant4重建已交付；当前EHE实际比较优先使用11月dense65536独立重建。 |
| [NEMA当前1.5mm内缩球中心ROI及统一完整背景](nema-current-roi.md) | current | analysis-complete / see-original-evidence | 原3D球内中心距球面至少1.5mm；所有球共享完整体模内部背景，排除六球、肺和接触边界。 |
| [NEMA H60球体、肺、218/440空间分布](nema-source-truth.md) | current | authoritative-source / see-original-evidence | 原3mm三维NEMA H60真值：六球与中心肺；218热球10/17/28mm，440热球13/22/37mm。球心及完整能量分布按原manifest/数组。 |
| [NEMA EHE加密矩阵实际5e9](nema-ehe-dense-5e9.md) | current | completed / strict-delivered | A218/A440提高到65536点/可见面，匹配自身S；200次正式完成，但边缘高值、散斑和小球对比不足仍存在。 |
| [NEMA EHE加密矩阵实际5e10](nema-ehe-dense-5e10.md) | current | completed / strict-delivered | A218/A440提高到65536点/可见面，匹配自身S；200次正式完成，但边缘高值、散斑和小球对比不足仍存在。 |
| [NEMA5e9原生Cartesian：5090正式三路](nema-cartesian-5090.md) | current | FAILED / not-delivered | 1822462于14:36:55失败退出；进度记录440 SC到10000，但根和三路completed均缺失，不能认定计算完成或完整交付。实际RSS/VRAM超过原80%守卫；输入、输出、原发布和日志保留，未修复/重投。 |
| [NEMA5e9原生Cartesian：4090 r2失败保留](nema-cartesian-4090-r2.md) | current | FAILED / not-delivered | 有效r2正式1822534于14:10:37失败；没有progress/completed，日志同样触发原80%实际RSS/VRAM守卫。原同名无r2分支为未提交预检失败。全部发布/输入/失败输出保留，未重投。 |
| [NEMA5e9：100/80/60% Compton展宽](nema-width-sweep.md) | current | completed / strict-delivered | 1814153三档各10000次、600帧严格交付。缩窄展宽未改善整体SSIM/NRMSE；保留旧历史差异未通过标志，不选择80/60%替代生产模型。 |
| [带水NEMA：四分支32路SC完整交付](water-sc-first.md) | current | completed / strict-delivered | 严格交付32路6400帧及背景；原metadata的visual_review_pending=true，14路预览和后续专题不能代替全32路视觉验收。旧README的未交付描述被strict_delivery更新。 |
| [带水NEMA：独立供体固定散射模型v2](water-scatter-v2.md) | current | completed / strict-delivered | 8路1600帧严格交付；独立剂量供体仍含相同体模布局先验。侧窗和440→218串窗偏差单列；并非物理认证。 |
| [OpenGATE EHE NEMA 5e9有效输运](opengate-ehe-5e9.md) | current | completed / strict-delivered | 源旋转修复后的真实投影，两能量400帧已严格交付；边缘高值仍存在，MC-CDR近似与4mm网格限制保留。 |
| [OpenGATE EHE NEMA 5e10有效输运](opengate-ehe-5e10.md) | current | completed / strict-delivered | 源旋转修复后的真实投影，两能量400帧已严格交付；边缘高值仍存在，MC-CDR近似与4mm网格限制保留。 |
| [OpenGATE HE NEMA 5e9/5e10](opengate-he-nema.md) | current | completed / strict-delivered | HEGP孔规格+现有NaI，两剂量两能量800帧交付并完成数值/视觉QA。背景BV降低，部分小球恢复下降；不是完整商用相机或纯单因素比较。 |

[返回总入口](../../README.md)
