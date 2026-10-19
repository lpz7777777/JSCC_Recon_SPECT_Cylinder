# 科学语义与当前输出规则

## 剂量与观测

- 5e9/5e10须结合各实验合同解释为实际gamma初级数或模型期望数；不是等效Ac225衰变次数。Derenzo四组各5e9，合计四个独立数据集。
- NEMA双能代理源218/440各有独立空间分布和光子产额权重；当前研究并非完整衰变链或符合事件模拟。
- CntStat与Compton List来自同次输运；不得当成两份独立剂量或重复计算发射数。
- 实际Geant4与`forward_poisson`须单独登记，不能把矩阵自生数据作为真实输运物理通过证明。

## 名称和通道

| 名称 | 含义 |
|---|---|
| SC / SinglePhoton | 单光子图像，JSCC和EHE探头均可有SC输出 |
| JSCC探头 | 具有单光子和Compton观测的探头设计；不表示每张图都用联合求解 |
| 440 SinglePlusCompton | 440 keV同能量的SC+Compton联合重建 |
| 218 CrossTalkCorrected | 在NEMA双能源下保留来自本次最终440图的固定加性串窗预测 |
| 218单能Derenzo | 独立纯218发射，固定背景为零，不引入另一个440体模 |
| 218+440求和 | 历史结果只读保留；后续不生成、计算、绘制或加入比较 |

新增NEMA JSCC5e10仅440 single、218 corrected、440 Compton三路，各10000/save50。Derenzo JSCC为440 single、440 SC+Compton及218 single，10000/save50；Derenzo EHE仅218/440各自single，200/save1。历史六路合同不覆盖这些新合同。

## 几何、求解与单位

当前椭圆FOV为500×300×120mm、20view，原生132040点/78920活动单元，径向6mm、z间距3mm。B=A diag(ΔV)是密度基底；恢复量是gamma密度，不是直接核素活度。响应、筛选、匹配自身S、原MLEM/全1初值及无正则条件分别由冻结合同绑定。

三维真值必须来自NEMA原truth_3mm.npz和manifest，或本轮Derenzo已绑定的解析几何JSON。显示采样、MIP、裁剪和色标均须说明；不使用单图拟合亮度或平滑掩盖伪影。EHE0–200与JSCC0–10000分开展示，同迭代不等同收敛。

## NEMA当前ROI

球ROI采用体素中心距球面至少1.5mm的全部3mm体素、等权均值；不使用分数球体积权重、不再额外内缩。它不保证体素立方体完全在球内。若为空则不可计算，不扩大ROI。

所有球共享一个背景：完整体素严格位于Phantom活动腔体内，排除六球、肺插入物、它们的边界及接触体素。实现以[nema_roi_policy.py](../../experiments/ELLIPSE500x300_H120/nema_roi_policy.py)与[当前分析](../../experiments/ELLIPSE500x300_H120/reports/nema_interior_roi_20261010/README.md)为准。

文件完整性、数值/执行验收、物理响应校准和图像质量是不同层次。EHE物理响应限制、长迭代噪声、网格限制及Compton统计不足类别仍须保留，不从“作业完成”推导分辨率达标。
