# 查询、运行与维护

从仓库根目录运行[工作流中的查询命令](WORKFLOW.md)。`tools/project_index.py results`按体模/发射量/能量/方法/网格/材料/投影来源/校正精确过滤；`find`搜索研究、函数与目录。查询仅读本地索引，不调用科学脚本。

维护时先审定[研究事实](study_registry.json)与[条件事实](result_registry.json)，保留证据日期、冻结和输入身份，再运行：

```powershell
python -B -X utf8 tools/navigation/build.py
python -B -X utf8 tools/navigation/check.py
python -B -X utf8 tools/project_index.py check
```

刷新仅写导航文本、JSON/CSV和指针，不联网、不更新科学阶段、不改AGENTS。旧refresh_ac225_navigation.py、build_project_catalog.py、build_reconstruction_results_index.py兼容转入新实现；原20261103模板与固定36路验证已归档。不得运行归档生成器main或手改生成页代替registry。

实时状态只使用经审查的直接只读squeue/sacct/scontrol和小文件读取。旧研究status也可能写latest；advance/prepare/submit/fetch/deliver/handoff/watch禁止用于目录检查。科学身份以freeze/release_manifest/run_manifest为准，同名工作区源码不自动代表执行版本。

控制器须同时匹配PID、create_time、cmdline。旧ACTIVE与旧PID只作历史；不并行操作相同取回/提交路径。各阶段判据见[状态](CURRENT_STATUS.md)。本次不恢复定时器、不停止控制器、不提交或重投科学任务。

冻结、活跃源码、输入、成功输出、失败证据与严格交付不移动、不改字节。成功步骤直接复用，不为导航重新计算。科研命令只从冻结实际launcher与提交记录取用，并遵守[AGENTS](../../experiments/ELLIPSE500x300_H120/AGENTS.md)及最新人工授权。

[维护工具](../../tools/navigation/README.md) · [数据与Git](DATA_POLICY.md) · [整理清单](../organization/20261117/README.md)
