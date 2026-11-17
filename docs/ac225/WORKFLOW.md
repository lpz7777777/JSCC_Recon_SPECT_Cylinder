# 从科学定义到可追溯结果

先用[当前阶段](CURRENT_STATUS.md)选择研究，再进入其[研究页](EXPERIMENTS.md)。查询从仓库根目录运行，Python标准库即可：

```powershell
python -B -X utf8 tools/project_index.py list --topic nema
python -B -X utf8 tools/project_index.py show nema-cartesian-5090
python -B -X utf8 tools/project_index.py results --phantom NEMA --grid Cartesian4mm
python -B -X utf8 tools/project_index.py results --material water --method SC
python -B -X utf8 tools/project_index.py results --projection-source forward+Poisson
python -B -X utf8 tools/project_index.py files --group calibration-physics
```

这些命令只读本地JSON，不查询队列、不导入运行器。计量、gamma预算、密度、ROI和显示规则统一引用[科学口径](SCIENCE_CONVENTIONS.md)。

| 阶段 | 实际代码 / 定义 | 输入→输出与身份 |
|---|---|---|
| 探头、准直器、体模、宏 | [Geant4模块](../../Geant4Sim/README.md)、[系统与体模](SYSTEMS_AND_PHANTOMS.md)、[geometry.py](../../experiments/ELLIPSE500x300_H120/geometry.py) | 源几何、材料、旋转、晶体顺序→宏与真值；NEMA与两能量Derenzo不是同一个源 |
| 真实输运、收集、投影 | [Geant4Code](../../Geant4Sim/Geant4Code/README.md)、[EHE](../../Geant4Sim/Geant4Code_EHE/README.md)、[OpenGATE入口](../../experiments/ELLIPSE500x300_H120/opengate_ehe_workflow_20261113.py) | 独立worker宏/种子/实际primary→CntStat、List、collection与成员SHA；OpenGATE也是真实Geant4输运 |
| 系统矩阵、校准、匹配S | [辅助矩阵工程](../../Auxiliary_Studies/README.md)、[加密EHE](../../experiments/ELLIPSE500x300_H120/ehe_dense_surface_workflow.py)、[Cartesian研究](../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/README.md) | A218、A440、C440→218与Compton响应分开；输入/网格变更必须匹配自身S，不能用观测Cᵀ1替代校准S |
| SC / Compton / 联合 | [算子与MLEM](CODE_GUIDE.md)、[NEMA联合](studies/nema/nema-5e10-v5-joint.md)、[Derenzo失败预览](studies/derenzo/derenzo-jscc-5e10-440-joint.md) | 冻结投影+事件+响应+S→检查点、history、run_manifest、最终图。纯Compton与SC+Compton分别命名 |
| HE / EHE / PyTomography | [HE NEMA](studies/nema/opengate-he-nema.md)、[HE Derenzo](studies/derenzo/opengate-he-derenzo.md)、[EHE有效旋转修复](studies/nema/opengate-ehe-5e9.md) | MC-CDR、原生4mm网格与深度/孔相位近似单列；不与65536点/面的柱坐标EHE混称 |
| 衰减与散射修正 | [修正模块](../../experiments/ELLIPSE500x300_H120/ac225_corrections/README.md)、[首轮32路](studies/nema/water-sc-first.md)、[固定供体v2](studies/nema/water-scatter-v2.md) | 带水真实投影、侧窗、衰减算子/匹配S→分支自身440最终图→218背景；JSCC探头在此仍是SC |
| 评价与可视化 | [ROI权威实现](../../experiments/ELLIPSE500x300_H120/nema_roi_policy.py)、[输出规则](../../experiments/ELLIPSE500x300_H120/reconstruction_output_policy.py)、[结果条件目录](../../Analysis_Plots/results/REPORT.html) | 严格交付→原生三维指标→共同网格/滤波派生指标→72mm MIP、曲线和QA；图像完成不等于物理成立 |

真实投影与[GenProj](../../GenProj/README.md)/20–60角前投影+Poisson是不同来源。性能与不确定度研究见[隔离报告](../../docs/research/jscc_uncertainty_optimization_20261115/README.md)；制造矩阵的CPU计时不代替完整GPU或物理验证。Hoffman、XCAT、点源及旧511keV/60mm工程从[历史复现](../../Reproduction/README.md)、[FOV120](../../experiments/FOV120/README.md)进入，不能直接复用当前225Ac条件标签。

## 环境与运行边界

导航工具只需Python 3.10+；现有本地Python为3.12。正式Cartesian日志记录PyTorch 2.8.0+cu128/CUDA12.8，Slurm启动加载cuda/12.9模块；这两个版本字段含义不同。其他研究依赖按其冻结/部署记录核对：Geant4编译器与物理数据、OpenGATE环境、PyTomography/torch/numpy/scipy，不能用导航环境替代。旧MATLAB Factors/前投影按原模块说明运行，不在本次重算。

实际重建命令必须取自对应`formal_submission_intent`、冻结`launch.sh`/run_manifest，并绑定其release及输入。不要把工作区同名脚本拼成新的提交命令。最终只读核对1822462与1822534均失败，原控制器已按原失败策略退出；没有完成或严格交付的Cartesian三路结果，失败证据保留，不修复或重投。

| 命令类型 | 副作用 |
|---|---|
| `tools/project_index.py ...`、`tools/navigation/check.py` | 本地只读 |
| `tools/navigation/build.py`及旧兼容刷新入口 | 仅写导航文本/小索引；不刷新科学状态 |
| 原研究`status` | 必须先读实现；有些会写latest文件，不用于本次整理检查 |
| `prepare`、`advance`、`submit`、`watch`、`fetch`、`deliver`、`handoff` | 可写冻结/数据、创建控制器或改变远端状态；本次禁止调用 |
| 绘图/分析脚本 | 通常写派生结果；被SHA绑定的图集及脚本也须保持原字节 |

[操作维护](OPERATIONS.md) · [源码与冻结关系](CODE_GUIDE.md) · [数据保护](DATA_POLICY.md)
