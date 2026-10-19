# 工程目录图

```text
项目根目录/
├─ README.md                         总入口与只读查询示例
├─ docs/
│  ├─ README.md                      文档导航
│  ├─ ac225/                         当前双能导航、状态、科学约定、机器索引
│  ├─ baselines/dual_energy_20261007/ 固定回归manifest和源码分类，不迁移
│  ├─ history/                       整理前原字节README
│  └─ organization/20261019/          本次整理身份与Git审计
├─ experiments/
│  ├─ ELLIPSE500x300_H120/            当前500×300×120mm、20view三维实验
│  │  ├─ README.md / AGENTS.md        当前入口 / 执行约定
│  │  ├─ *.py / *.cu / *.json         科学代码、控制、测试和配置，保留路径
│  │  ├─ reports/                    小型冻结、作业、验收与解释文件
│  │  │  ├─ NEMA_Body_H60/            双能NEMA；按研究组细分
│  │  │  ├─ Derenzo/                 独立单能Derenzo研究
│  │  │  └─ nema_interior_roi_20261010/ 当前ROI分析
│  │  ├─ generated/                  本地运行数据，Git忽略，保持原路径
│  │  └─ history/                    原README与AGENTS的时间线快照
│  └─ FOV120/                        另一代几何实验，不与当前椭圆混用
├─ Analysis_Plots/                   图形工程入口
│  └─ research/<研究名_日期>/         scripts / data / figures
├─ Geant4Sim/                        JSCC/EHE输运源与宏生成
├─ Auxiliary_Studies/                矩阵引擎、因子转换、事件/响应辅助研究
├─ GenProj/                          极坐标源测度与历史前投影
├─ distributed/                      通用历史分布式入口
├─ Reproduction/                     旧通用复现教程
├─ Results/                          旧本地结果说明，不是当前椭圆结果库
└─ tools/
   ├─ project_index.py               本地只读查询
   └─ research/                      分析、绘图、目录索引生成器
```

## 当前实验的成对目录

| 研究 | 小型证据（Git） | 实际数据（本地） |
|---|---|---|
| NEMA JSCC5e10 | `reports/NEMA_Body_H60/jscc_geant4_5e10_10000/` | `generated/jscc_geant4_5e10_10000/` |
| NEMA JSCC5e9 v5 | `reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/` | 同名`generated/`研究目录 |
| NEMA EHE Geant4 | `reports/NEMA_Body_H60/ehe_spect_<剂量>_200/` | 同名`generated/`研究目录 |
| NEMA EHE模型参考 | `reports/NEMA_Body_H60/ehe_forward_poisson_<剂量>_200/` | 同名`generated/`研究目录 |
| 单能Derenzo | `reports/Derenzo/derenzo_equal_regions_mono_5e9_20261016/` | 同名`generated/`，其下JSCC_218/JSCC_440/EHE_218/EHE_440 |

上表相对`experiments/ELLIPSE500x300_H120/`。源、矩阵、冻结发布、状态登记和数组已有硬编码路径及SHA绑定，因此本轮组织采用**统一入口和分层索引，保留科学运行路径**。仅说明时间线归档为独立历史文件。

根目录`Factors/`、`CntStat/`、`List/`和`Results/`属于旧本地接口。当前椭圆实验应查自己的generated、FactorsCalibrated及阶段合同，不把目录名相似当成身份相同。

[directory_index.json](directory_index.json)列出主要目录，并区分源码/报告与本地运行数据。可用`python tools/project_index.py dirs --topic derenzo`按名称查找；目录盘点不递归读取大数组。
