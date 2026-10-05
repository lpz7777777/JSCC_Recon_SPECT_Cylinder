# process_list_global_audit_v4 运行簿

2026-10-05。所有本轮有界诊断完成、PID退出，未产生新输运或工程重建。失败/被替代收据只用于追溯，不能恢复旧执行。

## 实际执行

| 阶段 | 最终PID | 发布 | 输出 |
|---|---:|---|---|
| diagnostic | 401512 | `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4/releases/6fb60a8e5394049b` | `主链points/probe/spatial/sampling，见diagnostic_job.json` |
| energy | 447229 | `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4/releases/energy_30f3e6fbd5fae2b3` | `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4/energy_30f3e6fbd5fae2b3` |
| responsibility | 448707 | `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4/releases/responsibility_b5ab267f9830fa60` | `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4/responsibility_b5ab267f9830fa60` |
| crystal | 460680 | `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4/releases/crystal_b1b5a89d81eaafca` | `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4/crystal_b1b5a89d81eaafca` |

主链顺序：CPU点源 → 2048事件GPU探针 → 探针时间/显存门槛 → 全训练空间效率 → K有限单元积分。附加能量/晶体为CPU，固定图像责任为GPU分块；不保存完整事件响应。

## 资源与完整性

最大GPU预留比例 13.7744%，最大RSS 12282716160 bytes，占服务器MemTotal 0.7568%；cgroup上限不可读取，此比例不表示Slurm授予的主存。服务器核验时GPU0为 `0, 0`（MiB、利用率百分比）。

数据集合共3170000000实际初级γ，760唯一独立种子，所有集合种子互不重叠。阶段所消费的CSV按冻结input_manifest逐文件核验；取回tar和每个文件分别核验SHA。440矩阵大小5543567360 bytes，SHA与前轮审计一致。

局部worker编号在视角内重置。最终统计用seed作为全局独立worker身份：NEMA200、每点源20；圆源本来就有200个独立worker，其20个复用旋转按worker合并。

## 修复记录

- 首次点源诊断在float32转换后用固定文字舍入容差，混入半ULP，已改为float64读原文字并按六位有效数字核验；生产仍原float32能量。旧PID395589已退出。
- 能量诊断发现远端没有可选pandas依赖，改为标准CSV，没有安装软件。
- 极小接受概率的CDF相减下溢，改为log-CDF/生存函数稳定差；与scipy truncnorm独立核验。
- worker误合并导致SE非有限值，修正为独立种子身份；能量与责任分别冻结新代码重跑。原输入、事件集合、生产核不变。

## 本地/远端复现

远端Python为 `/home/lipeize/JSCC_FOV120_20260924/.venv/bin/python`。项目诊断根 `/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/process_list_global_audit_v4`。所有命令从独立发布读取冻结脚本及显式路径；不访问scxi717/maty执行新任务。

`process_list_audit_v4_workflow.py start` 启动有界主链；已有job收据时拒绝重复，尚未进入SSH部署。`deploy`只测试/冻结代码，不启动计算。`energy`及`responsibility`是后续有界诊断，已有对应收据时也拒绝重复。

晶体复现使用 `crystal_position_audit_v4.py --inputs FIRST/analysis_inputs --analysis V3/R1_analysis --points POINT_OUTPUT/point_residuals.csv --detector FactorsCalibrated/440keV_RotateNum20/Detector.csv --output NEW_DIR`；FIRST/V3/POINT_OUTPUT完整路径见收据。默认没有覆盖现有目录。

本地分别执行 `build_whole_cell_geometry_v4.py`、`catalog_native_spikes_v4.py`、`compton_identifiability_toy_v4.py` 和 `summarize_process_list_audit_v4.py`；前三者已有目标目录时拒绝覆盖。六项诊断单元测试与完整单元数值检查通过，生产50次回归/10次完整事件门控未运行。

## 当前门控

`OFFLINE_INVESTIGATION_COMPLETED_PRODUCTION_HOLD`。能量诊断支持单项材料尾部方向，但生产概率合同、稀疏角域覆盖、晶体条件积分、固定q接受及匹配S尚未验收。没有新配对结果，也没有定时轮询；旧停止标记与PAUSED任务继续保持。

代码测试、来源哈希见 `final_code_deployment.json`；科学输入取回哈希见各collection收据；小型证据和科学图进入版本管理。大数组、归档和完整历史留在generated。
