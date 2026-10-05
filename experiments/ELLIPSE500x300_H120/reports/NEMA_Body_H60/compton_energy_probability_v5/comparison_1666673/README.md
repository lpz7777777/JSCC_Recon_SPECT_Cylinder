# 1666673：实际A/B正式配对图集与指标

两组各2000次、440 Compton/440 JSCC共四条历史各40帧；实际内容、资源及逐文件SHA通过。A=当前角度核，B=连续材料能量核，各自匹配S；同91225事件、78920完整柱单元，均从全1开始。

2000次最大密度/峰背景：Compton下降73.76%/76.14%，JSCC66.11%/69.53%，达到双50%工作判据。源外轴向积分下降，但背景噪声仅小幅改善，Compton内部仍有23.23倍背景高值，13 mm球恢复仍弱。[完整科学报告与验收](../ACCEPTANCE.md)给出代价、峰位置、资源及结论边界。

## 固定尺度迭代图集

| 图像 | 定义 |
|---|---|
| [中心轴位](iterations_center.png) | z=+1.5 mm单层，不是MIP |
| [冠状位](iterations_coronal.png) | y=−1.5 mm，全120 mm轴向范围 |
| [矢状位](iterations_sagittal.png) | x=−1.5 mm，全120 mm轴向范围 |
| [中央72 mm MIP](iterations_mip72.png) | z边界±36 mm，上下各排除8层 |
| [2000次多平面概览](final_multiplanar.png) | 包含上述四平面及真实三维球体真值 |

每个迭代图集固定显示100/500/1000/2000；行顺序为Compton A/B、JSCC A/B，首列真实440球体真值。每通道两组/全部迭代统一除A2000的背景均值：Compton188.1327667、JSCC173.7997284；共同色阶0–10，灰白低/黑高。超过10仅显示饱和，数据未截断。无平滑、无XY裁边、无逐图拟合。真实H60三维球体与球中心半径57 mm，不使用技能默认二维柱体真值。

MIP排除端层，所以端部尖峰变化要结合冠/矢位和下列**全120 mm原生统计**，不能仅凭中央72 mm图判断。本研究都是完整单元，没有f<0.1部分单元指标。

## 完整40帧曲线与数据

- [尖峰、峰背景、CV与泄漏](spike_noise_leakage_curves.png)
- [13/22/37 mm球CRC/CNR](crc_cnr_curves.png)
- [高值尾部、背景、积分及恢复率](tail_integral_curves.png)
- [原生指标CSV：160行](native_iteration_metrics.csv)
- [球体指标CSV：480行](sphere_iteration_metrics.csv)
- [四个指定迭代的判据与CRC代价：24行](selected_comparison_metrics.csv)
- [共同尺度、最终/逐迭代判断、输入/历史SHA](comparison.json)
- [图像/指标/真值/脚本SHA清单](artifact_manifest.json)
- [独立科学与视觉QA](../final_analysis_qa_1666673.json)

CRC是局部背景定义下的恢复，完整报告给出ROI与公式。40帧中没有CRC损失>5个百分点，但小球绝对恢复仍低；JSCC第100次没有达到双50%尖峰判据。2000次不能外推10000次稳定性或真实设备性能，447项联合类别统计仍不足。

## 实际分析归档

`python -X utf8 experiments/ELLIPSE500x300_H120/compare_energy_formal_v5.py --job 1666673` 已对本次真实正式结果运行成功；该入口拒绝覆盖已有图集，禁止把短程结果标成正式。实际脚本副本保存在[scripts](scripts/compare_energy_formal_v5.py)，另归档analyze_nema_result.py、mip_projection.py及[只读交叉核算脚本](scripts/verify_delivered_analysis.py)。

真值truth_3mm.npz副本本地保留，SHA=2612f0ed6839f9460722711e1017a10102e83adf77cf715d5c2553cfaec948af；遵循仓库忽略规则不提交npz。原始四条历史/final和80检查点完整位于本工程generated/compton_energy_probability_v5/formal_results/1666673（各组约90 MiB），未修改数值，不保存完整事件×体素巨型矩阵。
