# JSCC与EHE三维SPECT工程

当前工程研究218/440keV的225Ac代理gamma成像：JSCC探头的单光子及同能量SC+Compton联合重建，EHE、OpenGATE/HE单光子成像，以及物理响应和离散化诊断。

**[打开工程与研究目录](docs/ac225/INDEX.html)** · **[打开成像结果](Analysis_Plots/results/REPORT.html)** · **[查看当前状态](docs/ac225/CURRENT_STATUS.md)**

| 查询内容 | 说明 |
|---|---|
| [225Ac总导航](docs/ac225/README.md) | 研究类别、数据口径与查询示例 |
| [完整工作流与运行边界](docs/ac225/WORKFLOW.md) | 环境、各阶段实际入口、输入输出、只读与有副作用命令 |
| [研究结论](docs/ac225/CONCLUSIONS.md) | 已确认的现象、候选机制和剩余限制 |
| [代码路线](docs/ac225/CODE_GUIDE.md) | 体模→输运→系统矩阵→重建→ROI/图像 |
| [工程目录图](docs/ac225/DIRECTORY_MAP.md) | 源码、证据、运行数据、图形与历史位置 |
| [实验与证据](docs/ac225/EXPERIMENTS.md) | 每个研究的参数、作业及交付状态 |
| [科学口径](docs/ac225/SCIENCE_CONVENTIONS.md) | SC/JSCC命名、gamma预算、密度/体积、72mm MIP |
| [操作和数据管理](docs/ac225/OPERATIONS.md) | 只读查询、冻结保护与目录维护 |
| [其他工程文档](docs/README.md) | 基准、复现与跨工程研究 |

当前三维主工程位于[experiments/ELLIPSE500x300_H120](experiments/ELLIPSE500x300_H120/README.md)，FOV为500×300×120mm；真实输运默认20角，60角仅指独立EHE前投影＋Poisson研究。NEMA及两种单能Derenzo的输入、几何、剂量分别登记。

SC表示单光子，图中JSCC表示同能量SC+Compton联合；纯Compton单列。218和440不做跨能量图像求和。实际Geant4、矩阵合成投影和旧矩阵参考在结果目录中分开。当前光子数不是225Ac衰变数。

```powershell
python tools/project_index.py list --topic nema
python tools/project_index.py show ehe-nema-poisson-60view
python tools/project_index.py find Compton
python tools/project_index.py results --phantom NEMA --grid Cartesian4mm
python tools/project_index.py check
```

原始源码、矩阵、事件、完整历史和已冻结发布保留原路径；新增分层目录保存说明及查询指针。旧本地60mm/FOV120接口继续从[Reproduction](Reproduction/README.md)和[Results](Results/README.md)查询。本轮实际迁移、保护清单与验收见[2026-11-17整理记录](docs/organization/20261117/README.md)。
