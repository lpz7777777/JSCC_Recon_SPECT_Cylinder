# 225Ac成像工程导航

本工程以218与440keV的gamma代理源研究JSCC、EHE及HE三维SPECT。主几何为500×300×120mm，NEMA和Derenzo源主体H60；实际输运以20视角为主，另有60视角的EHE模型合成对照。光子预算不是225Ac衰变次数。

当前推荐从[可搜索工程目录](INDEX.html)或[成像结果目录](../../Analysis_Plots/results/REPORT.html)开始。当前EHE实际结果使用65536点/可见面加密矩阵；原EHE矩阵画廊、前投影＋Poisson和纯Compton诊断分别标记，防止把不同来源混作同一组。

| 要查什么 | 入口 |
|---|---|
| 各组图像与数据来源 | [结果说明](RESULTS.md) · [结果网页](../../Analysis_Plots/results/REPORT.html) |
| 新加入工程：环境、运行方法和阶段数据流 | [完整工作流](WORKFLOW.md) |
| 已完成、排队和待处理项 | [当前状态](CURRENT_STATUS.md) |
| 研究得到什么结论 | [结论与限制](CONCLUSIONS.md) |
| 每个研究的参数、代码和证据 | [分类研究页](studies/README.md) · [实验表](EXPERIMENTS.md) |
| 从输运到重建的代码路线 | [代码导读](CODE_GUIDE.md) · [函数与文件索引](SOURCE_INDEX.md) |
| 目录职责与原始数据位置 | [目录图](DIRECTORY_MAP.md) |
| 探头、FOV和两种体模 | [系统与体模](SYSTEMS_AND_PHANTOMS.md) |
| SC/JSCC、剂量、体积、ROI、矩阵版本 | [科学口径](SCIENCE_CONVENTIONS.md) |
| 维护索引及查状态 | [操作说明](OPERATIONS.md) |
| 大数组、冻结、历史和Git边界 | [数据管理](DATA_POLICY.md) |

```powershell
python tools/project_index.py list --topic 60view
python tools/project_index.py show nema-5e10-v5-joint
python tools/project_index.py find ehe_dense
python tools/project_index.py files --group reconstruction
python tools/project_index.py results --material water --method SC
python tools/project_index.py check
```

命令只读本地索引，不导入科学工作流、不连接超算、不提交作业。状态是带时间的证据；源码及数组位置保持稳定。实际文档工具迁移、冻结保护与身份核对见[本次整理记录](../organization/20261117/README.md)。
