# ELLIPSE500x300_H120：218/440 keV研究入口

当前三维几何：椭圆500×300mm、完整高度120mm、20view。NEMA体模主体H60；独立Derenzo两种能量也为H60。源、模型、观测、求解和验收按各研究合同区分。

| 查找任务 | 入口 |
|---|---|
| 全工程双能索引 | [225Ac/218/440导航](../../docs/ac225/README.md) |
| 最新带日期状态 | [状态快照](../../docs/ac225/CURRENT_STATUS.md) |
| NEMA实际JSCC5e10 | [README](reports/NEMA_Body_H60/jscc_geant4_5e10_10000/README.md)、[合同](reports/NEMA_Body_H60/jscc_geant4_5e10_10000/CONTRACT.md) |
| 已完成NEMA5e9 v5历史基准 | [正式验收](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/ACCEPTANCE.md) |
| NEMA各EHE组与体模真值 | [NEMA目录](reports/NEMA_Body_H60/README.md) |
| Derenzo四组独立单能5e9 | [Derenzo目录](reports/Derenzo/README.md) |
| 当前球ROI与统一背景 | [当前分析](reports/nema_interior_roi_20261010/README.md) |
| 所有小型证据分类 | [reports目录](reports/README.md) |
| 代码按职责查询 | [源码索引](../../docs/ac225/SOURCE_INDEX.md) |
| 执行约束 | [AGENTS.md](AGENTS.md) |

在仓库根可运行`python tools/project_index.py list --topic nema`或`show nema-jscc-5e10`。查询不会运行生产脚本。

本目录的顶层源码与generated数据保留既有路径，避免破坏导入、运行中任务和SHA合同。当前目录分层及Git保留策略见[目录图](../../docs/ac225/DIRECTORY_MAP.md)和[数据规则](../../docs/ac225/DATA_POLICY.md)。

旧六路JSCC和EHE三路只读保留；新的实验不生成218/440求和图。SC表示单光子；同440能量的SC+Compton与跨能量相加不同。实际gamma发射数不是Ac225衰变数。

[整理前README](history/README_20261019.md)、[整理前AGENTS](history/AGENTS_20261019.md)及[更早历史](HISTORY_20261010.md)保留当时排队/运行语句，不应作为当前状态。
