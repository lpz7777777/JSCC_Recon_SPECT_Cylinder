# 图形与结果查询

**[当前结果目录](results/REPORT.html)**按真实Geant4、模型Poisson、体模、发射量、角度、能量和SC/JSCC/EHE分类。**[研究结论](../docs/ac225/CONCLUSIONS.md)**统一说明环纹、边缘高值和模型限制。

2026-11-17更新：[条件CSV](results/data/conditions.csv)新增HE、OpenGATE EHE、网格、材料、校正和展宽查询。[HE Derenzo四路与PPT已交付](../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/COMPLETED_20261117.md)；带水首轮32路已严格交付，但原metadata仍有全组视觉QA待完成标记，下表14路是旧子集报告，不是当前完整范围。失败/运行条件与完成图像通过[阶段页](../docs/ac225/CURRENT_STATUS.md)分开判读。

| 内容 | 当前入口 |
|---|---|
| OpenGATE/PyTomography无水NEMA实际5e9/5e10 | [5e10与原EHE、两剂量对照](research/opengate_ehe_nema5e10_20261113/index.html) |
| 带水NEMA衰减/散射修正：已完成14路440 SC | [最终图、迭代图与BV–CNR](research/ac225_water_completed_20261105/REPORT.html) |
| 20/60角、四剂量NEMA合成对比 | [60角及对照报告](research/ehe_dense_nema_60view_poisson_20261102/REPORT.html) |
| 加密EHE实际NEMA5e9/5e10与Derenzo5e9 | [1761244结果](research/ehe_dense_completed_1761244_20261102/REPORT.html) |
| NEMA5e10的440 SC与JSCC联合 | [1740189结果](research/nema_joint5e10_1740189_20261029/REPORT.html) |
| 原Compton与新旧流程诊断 | [专题目录](../docs/ac225/studies/diagnostics/README.md) |
| 体模、探头和汇报设计 | [设计目录](../docs/ac225/studies/design/README.md) |
| 按日期查看全部专题 | [research索引](research/README.md) |

`results/current`是查询指针；`research/<专题_日期>`保存原figures/data/scripts及来源SHA。原6组方法比较画廊保留在results/NEMA和results/Derenzo，其EHE为旧矩阵，已从当前首选结果中分开。

SC＝单光子，JSCC＝同能量SC+Compton联合，纯Compton单列。真实Geant4与forward+Poisson不混作同一实验；不做218/440跨能量图像求和。中心72mm、完整XY、无平滑/亮度拟合按各报告明确。
