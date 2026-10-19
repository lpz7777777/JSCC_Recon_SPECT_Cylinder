# NEMA JSCC实际5e10：三路10000次研究

当前仅440单光子、218串窗校正单光子和440 Compton三路。研究未完成正式交付：**完整32GPU validation10及独立验收已通过并严格取回；正式10000次受实测时长门槛阻断**。

## 当前阶段（2026-10-19）

| 阶段 | 作业 | 证据/状态 |
|---|---:|---|
| 实际5e10 Geant4输运/收集 | 15708389 / 15709588 | 已严格交付；1000×50M、20view、全4π |
| 筛选/逐原行独立验收 | 1686507 / 1686546 | 已严格交付；4849087事件、40数组，复用不重算 |
| 完整输入32GPU validation10 | 1704559 | 10月19日06:39上海COMPLETED/0:0 |
| 独立验证验收 | 1704566 | 06:44上海COMPLETED/0:0，已严格取回 |
| 正式10000/save50 | 1704572 | 已登记，仍暂挂，正式尚未开始 |

[实测时长阻断](production_5090_8x4_queue_measured_time_hold_20261019.json)：按实测吞吐外推三路约54.87小时，准备只计一次；留1.5倍余量需4969分钟。现有启动gate的48小时总/24小时阶段上限不足，恢复控制器按门槛退出。这不是平台最大时限声明，也不是ETA；后续须独立执行时限或分段方案修复，不能直接绕过旧gate。

## 首选文件

| 查什么 | 文件 |
|---|---|
| 科学输入、路由和验收条件 | [CONTRACT.md](CONTRACT.md) |
| 当前操作和禁止重复提交的条件 | [RUNBOOK.md](RUNBOOK.md) |
| 原验证完整取回 | [validation_acceptance.json](validation_acceptance.json) |
| 本次32GPU authority | [formal_authority.json](formal_authority.json) |
| 正式请求 | [formal_job.json](formal_job.json) |
| 当前生产冻结/部署 | [production_freeze.json](production_freeze.json)、[production_deployment.json](production_deployment.json) |
| 旧4090存储方案的拒绝依据 | [STREAMING_STORAGE_HOLD.md](STREAMING_STORAGE_HOLD.md) |
| 原UUID修复与1688717证据 | [原修复记录](selection_5090_8x4_monitor_repair_README.md) |
| 全部旧时序 | [history/README.md](history/README.md) |

当前生产冻结ffe08f48be856816，8节点×4张5090，完整float32事件响应CPU常驻、逐32事件转GPU；科学模型保持v5、原MLEM、全1/无正则、完整120mm/78920活动单元。它不依赖旧4090失败的/tmp持久缓存方案，不写1.53TB共享盘响应。

实际大数据在`generated/jscc_geant4_5e10_10000/`，本目录保存小型登记、代码快照与验收。全工程查询见[双能入口](../../../../../docs/ac225/README.md)。定时任务仍暂停；Git提交不代表实验完成。
