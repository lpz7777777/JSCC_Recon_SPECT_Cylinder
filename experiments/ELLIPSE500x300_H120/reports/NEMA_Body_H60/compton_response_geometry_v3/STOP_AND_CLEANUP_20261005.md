# 精细 A 场停止与清理

2026-10-05，按用户明确指令停止计算并清除相关结果。此状态覆盖此前所有“继续生产/续跑/全场待验收”的安排。该候选场未完成全场科学验收，不能登记为成功生产；不再生成依赖它的 S2 或启动对应 R2 成像。

## 停止

65114 四个生产包装进程 2542514、2542646、2542872、2543046 及其子进程，经 `/proc` 路径、父子关系、进程组归属核对后发送 TERM，均退出，无需 KILL。只处理该分支，不取消其他项目任务。远端写入 `generated/compton_response_geometry_v3/USER_STOP_FINE_A.json`；当前工作流入口读取本地 `fine_field_retired.json` 阻止重新启动该辅助场。

证据：[停止记录](fine_field_stop_20261005.json)、[最终复查](fine_field_final_verification_20261005.json)。最后复查无精细场生产进程，12个目标数据目录均已移除。

## 清理范围

仅清理远端本工程 `generated/compton_response_geometry_v3` 下的以下12个目录，核对绝对路径、无符号链接，并逐文件记录身份后删除：

- `A440_guard`、`A440_guard_field`、`A440_near_column`
- `A440_near_patch`、`A440_near_patch_refined`、`A440_near_patch_ultrafine`
- `A440_radial_check`、`A440_tile_pilot`、`A440_tiled_field`
- `regional_a_g0`、`regional_a_g1`、`regional_a_g2`

先删除2552个 `.sysmat/.float32/.float64` 文件，535564319744 bytes；再核对并删除两个未完成 `.sysmat.partial` 文件，434859456 bytes。合计 **2554文件，535999179200 bytes（536.00 GB）**。本地同一分支检查未发现上述后缀的矩阵二进制，无本地矩阵删除。

剩余31379份参数、日志、进度和收据合计3047195566 bytes，压缩为559782141 bytes，并逐文件核验解压后的SHA256、名称和大小后移除散落源目录。保留的远端归档为：

`/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated/compton_response_geometry_v3/retired_fine_field_metadata_20261005.tar.gz`

SHA256：`b36c9674a2e05500db8a66e5629d0f2c29f3028c0d300b044e900d9e4de151a6`。

该归档**没有响应矩阵**，只为追溯保留；未下载到本地或加入Git。最终磁盘可用5799645933568 bytes（约5.80 TB），实时余量会随其他任务变化。

小型证据：[清理前清单](fine_field_cleanup_inventory_20261005.json)、[二进制删除计划](fine_field_cleanup_plan_20261005.json)、[第一阶段收据](fine_field_cleanup_receipt_20261005.json)、[最终归档与清理收据](fine_field_final_cleanup_20261005.json)。第一阶段收据中显示的两个partial和散落元数据已由最终收据覆盖；不是仍待清理。

## 保留与后续

保留原三套Factors、Geant4数据、NEMA/其他体模真值、所有已有重建及历史帧、稳定几何修复、新S1和已完成R1试跑、冻结发布源码、独立诊断摘要及图表。已有数值收敛结果仍是历史局部证据，不提升为被停止完整场的验收。旧自动任务、正则化和旧长迭代均不恢复。

新方向见 [柱坐标全域 process_list 调查计划](../process_list_global_audit_v4/PLAN.md)：允许近似椭圆和完整单元，重点转向内部尖峰的概率核、K×A物理假设、局部灵敏度、位置采样和统计可辨识性。本次未提交新输运或新重建。
