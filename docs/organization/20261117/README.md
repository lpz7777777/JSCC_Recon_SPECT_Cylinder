# 2026-11-17工程整理记录

本次在`codex/ellipse500x300-h120`原工作区继续，起点`dd54533ebcbf463ef2dc24ca85dda9788b5919ea`。原有未提交研究成果、AGENTS新增授权及失败证据均保留。没有切换分支、回退、取消、重启、重投、重新取回或重算科学任务。

## 实际整理

- 将四份只用于文档的20261103生成/模板/检查/registry脚本移入`tools/legacy/ac225_navigation_20261103`，逐字节SHA一致；旧命令保留兼容入口。旧模板写入API关闭，当前维护实现集中到`tools/navigation`。引用检查见[moves.json](moves.json)。
- 建立74项研究与98条条件索引，增加HE/OpenGATE、Cartesian、带水各校正、展宽、不确定度与派生评价。更新研究页、源码函数索引、浅层目录索引、结果网页/CSV及JSON指针；不复制大数组。
- 增加按体模、gamma量、能量、方法、网格、材料、投影来源与校正查询；关闭旧刷新器覆盖AGENTS/过时状态的路径，移除固定36路验证假设。
- 更新分层导航、完整工作流、模块职责、科学口径、阶段说明与复现边界。旧混合状态原字节保存在[历史状态页](../../ac225/history/CURRENT_STATUS_before_20261117.md)，旧其他说明的整理前SHA另见[navigation_before_sha256](navigation_before_sha256.json)，原字节备份在本地`tmp/organization_20261117/before`。
- 增加node_modules及新媒体/归档忽略规则；依赖、媒体、矩阵、事件、大数组、编译与缓存继续本地保留。没有删除或归档科学数据。

## 证据与保留

[保护基线](protected_baseline.json)含32份freeze及621份小型受保护文件SHA，包括本次三项新研究源码、可定位payload/release及AGENTS。只读查询分别保存在[调度器](remote_readonly.json)和[进度/日志](remote_logs_readonly.json)。这是带时间的观察，最新目录不会让旧证据自动变新。

首次5090作业1822462为RUNNING；最终14:46只读核对两任务均FAILED，错误均为原80%实际RSS/VRAM守卫。440 SC进度到10000但完成manifest缺失，不认定该分支完成。最初与最终快照均保留。最初原控制器18108的PID/create_time/cmdline一致，最终18108与22172均已按原失败策略退出；未由本次整理终止。原同名无r2分支只是未提交预检失败；所有传输r1–r6、原矩阵、有效发布和失败输出均未移动或修改。其他运行任务原位保留。

Derenzo联合1771220只有9950次/199帧失败预览；HE Derenzo四路800帧与PPT交付已完成；带水首轮32路严格交付已完成但原metadata仍声明全组视觉复核待完成。源代码、原投影与前投影Poisson、不同网格/材料/响应/展宽不合并。

## Git审计范围

仅选择与本次目录维护直接相关的既有小型导航修改、归档文档工具、新维护代码、索引、被导航引用的既有研究源码及必要小型证据。既有源码只读纳入并保持字节，补齐其静态本地import依赖，不执行或改写科学实现。既有Derenzo图集/绘图修改、AGENTS新增授权、其他报告合同与未审查源码不混入本次提交。全部候选见[git_candidates.csv](git_candidates.csv)，排除原因见[git_exclusions.json](git_exclusions.json)。

首次暂存核对发现15份原报告/回执会受Git换行规范化影响，已通过收紧.gitattributes的字节保留规则并重新暂存修复，原文件未改。最终逐blob核验应无冻结、归档或原科学文本SHA差异；PNG/PDF/PPTX、node_modules、运行大数据不提交。

未明确外部依赖或冻结关系的顶层科学程序、旧控制器、历史实现、临时文件和原研究README一律原位保留；用途通过索引说明。旧README中的当前/未完成叙述由统一阶段页纠正，原证据不被覆盖。

[构建清单](build_manifest.json) · [当前入口](../../ac225/README.md) · [维护工具](../../../tools/navigation/README.md)
