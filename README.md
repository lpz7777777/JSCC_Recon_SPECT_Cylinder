# JSCC / EHE 三维SPECT重建工程

本工程研究218/440 keV双能、单光子与Compton观测、极坐标密度重建及响应校准。当前三维主实验位于 **ELLIPSE500x300_H120**：椭圆FOV500×300 mm、完整轴向120 mm、20视角。

## 从这里开始

| 要查什么 | 入口 |
|---|---|
| **225Ac标签下的218/440双能成像总览** | [双能研究导航](docs/ac225/README.md) |
| 各组完成情况、作业号和当前阻点 | [带日期的状态快照](docs/ac225/CURRENT_STATUS.md) |
| 按体模、探头、剂量和数据来源找实验 | [实验与证据索引](docs/ac225/EXPERIMENTS.md) |
| 目录职责、原始数据及结果放在哪里 | [工程目录图](docs/ac225/DIRECTORY_MAP.md) |
| 查源码、脚本及冻结副本 | [代码索引](docs/ac225/SOURCE_INDEX.md) |
| 通道、光子数、串窗、ROI和成像单位 | [科学约定](docs/ac225/SCIENCE_CONVENTIONS.md) |
| 查状态、继续既有流程的前置条件 | [操作说明](docs/ac225/OPERATIONS.md) |
| 已完成Derenzo图像 | [绘图与可视化入口](Analysis_Plots/README.md) |
| 后续信息响应与系统优化研究 | [跨工程研究桥接](docs/RESEARCH_STRATEGY.md) |
| 其他说明与历史复现 | [文档目录](docs/README.md) |

```powershell
python tools/project_index.py list
python tools/project_index.py show nema-jscc-5e10
python tools/project_index.py list --topic derenzo
python tools/project_index.py find compton
python tools/project_index.py dirs --topic derenzo
python tools/project_index.py check
```

以上命令仅查询本地目录索引，不连接超算、不提交计算。状态是注明时间的证据快照，实时情况需查看相应登记与验收。

## 重要语义

- 工程中的“225Ac/双能”指文档定义的218/440 gamma代理源；当前模拟光子数不是Ac225衰变次数，也不是完整衰变链符合事件。
- **SC=单光子成像**。JSCC可以指探头或440 keV下SC+Compton重建，须看具体通道；218/440不是同一个通道。
- 后续不生成218与440的跨能量求和图。历史六路结果保留，当前生产以每个实验自己的合同为准。
- EHE实际Geant4投影与矩阵前投影+Poisson是独立实验，索引中分开标注。

## 工程与数据分层

`Geant4Sim/`负责输运；`Auxiliary_Studies/`和`GenProj/`保留系统矩阵/测度研究；`experiments/`保存实验源码与小型证据；`Analysis_Plots/`保存可复现的图形工程；`docs/ac225/`统一导航。运行数据仍在各实验`generated/`，已冻结科学代码、验收和活跃任务路径保持原位。

矩阵、事件数组、完整历史、大型归档、图像二进制和凭据不随本轮Git交付。源码、绘图脚本、几何参数、证据SHA与目录索引进入Git；详见[数据与Git规则](docs/ac225/DATA_POLICY.md)。

[2026-10-19整理前的完整项目README](docs/history/PROJECT_README_20261019.md)保留旧双语教程和兼容入口；旧60 mm/FOV120说明不能替代当前椭圆实验合同。

English: start with the [project map](docs/ac225/DIRECTORY_MAP.md), [experiment catalog](docs/ac225/EXPERIMENTS.md), and [legacy reproduction guide](Reproduction/README.md). Scientific execution paths and frozen results are preserved.
