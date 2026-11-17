# 工程目录与职责

```text
工程根目录
├─ README.md                         项目入口
├─ docs/ac225/
│  ├─ INDEX.html / README.md          可搜索研究导航
│  ├─ CURRENT_STATUS.md               当前阶段与待处理项
│  ├─ CONCLUSIONS.md                  结论、证据与限制
│  ├─ CODE_GUIDE.md / code/            流程导读与按职责源码列表
│  ├─ studies/<类别>/<研究ID>.md       每个研究条件、代码、数据和证据
│  └─ *_index.json / catalog.json     可查询机器索引
├─ experiments/
│  ├─ ELLIPSE500x300_H120/            当前500×300×120mm工程
│  │  ├─ *.py / *.cu / *.json         原科学代码与工作流位置
│  │  ├─ studies/<研究>/              Cartesian、展宽、HE Derenzo独立源码
│  │  ├─ ac225_corrections/            水衰减/侧窗/背景/SC模块
│  │  ├─ reports/{NEMA_Body_H60,Derenzo,EHE,...}
│  │  ├─ generated/<研究ID>/          原运行数据，位置保持
│  │  └─ history/                     原说明时间线
│  └─ FOV120/                         另一代几何与历史参考
├─ Analysis_Plots/
│  ├─ results/REPORT.html             当前条件目录
│  ├─ results/current/actual/...      真实输运结果指针
│  ├─ results/current/synthetic/...   20/60角模型结果指针
│  ├─ results/current/conditions/...  HE、网格、材料/校正与展宽等独立条件
│  ├─ results/history/20261029/        原EHE方法对比网页快照
│  ├─ results/{NEMA,Derenzo}/...       原6组画廊，保留字节
│  └─ research/<专题_日期>/            原图表、指标、脚本及QA
├─ Geant4Sim/                         JSCC/EHE输运实现
├─ Auxiliary_Studies/                 系统矩阵与响应辅助研究
├─ GenProj/                           极坐标源测度和历史前投影
├─ tools/                             分析、评价、图形和只读查询
│  ├─ navigation/                     当前导航构建、验收与Git审计
│  └─ legacy/ac225_navigation_20261103/ 原字节归档文档工具
├─ distributed/                       通用历史分布式入口
├─ Reproduction/ / Results/           旧本地复现与结果说明
└─ docs/organization/{20261103,20261117}/ 历次状态证据、保护与整理审计
```

目录清单按浅层文件元数据建立；[函数与职责](SOURCE_INDEX.md)、[Python导入关系](source_dependencies.csv)、[完全相同源码组](duplicate_source_groups.json)用于查引用和识别重复实现。相同SHA可能是冻结/图集副本，不能因此删除或合并；[版本关系](VERSION_RELATIONSHIPS.md)列出替代证据与历史保留原因。

## 为什么保留源码和数据路径

正式发布与运行数据包含硬编码相对路径、导入关系、远端路径和SHA绑定。当前整理将职责分类和结果条件体现在查询目录，不移动这些科学文件。`results/current`每个叶目录都有README与case.json，中间层也有入口；原research图集和冻结证据通过指针反查。

每个research专题的figures为图像/PDF，data为指标、来源和显示QA，scripts为当时实际绘图副本。实验reports用于合同与阶段证明，generated用于实际大数据；二者不能互换。

根部Factors、CntStat、List和Results属于历史本地接口，不能据同名目录当作当前椭圆研究输入。当前研究优先读独立generated、reports及具体contract。

[各级目录表](DIRECTORY_INDEX.md)列出每个目录职责、关联研究和可用说明；[机器索引](directory_index.json)另含本地数据标记。`python tools/project_index.py dirs --topic ehe`可检索，不递归读取大数组。
