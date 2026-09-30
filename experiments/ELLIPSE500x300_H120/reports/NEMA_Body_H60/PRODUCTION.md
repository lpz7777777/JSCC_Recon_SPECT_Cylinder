# NEMA H60：1e9 Geant4 与椭圆 FOV 重建运行簿

状态日期：2026-09-30。此页是**运行记录和下一步验收条件**；不能仅凭 Slurm `COMPLETED` 推断模拟或重建成功。几何与活度设计见 [README.md](README.md)，椭圆实验基线见 [主 README](../../README.md)。

## 冻结输入

- 本地 3 mm 真值：`generated/NEMA_Body_H60/truth_3mm.npz`，SHA-256 `2612f0ed6839f9460722711e10102e83adf77cf715d5c2553cfaec948af`；完整描述及源配置哈希见 `manifest.json`。
- 20 视角、200 worker 的 `Simulation_1e9/jobs.json`，SHA-256 `d6a45235e38d1217949887eb42c13ec6cb64be83350219ec15933c06ee7e9b2e`。每 worker 5e6，独立种子 30093001–30093200，总计 **1e9** 初级 γ。
- 3 mm 体素真值合并为 218 keV **1945** 个、440 keV **1992** 个 cuboid；体积与 0.114/0.259 光子产额加权后，预计初级 γ 组成 **29.3808% / 70.6192%**。两能量各自的热球/背景**活度浓度**比为 10:1，不能把光子产额差误当作浓度差。
- 20 份宏在 maty 逐份核对 SHA-256、`/xcat/angle`、`/run/beamOn` 和 cuboid 数。与已有矩阵相同的中心 `(0,-345,0) mm`、20 视角、13% FWHM@511 keV 的 List 展宽链，源无人体材料衰减。`/xcat/add` 对每个体素盒均匀采样，边界保留 3 mm 真值的部分体积权重；后续解析球面精度限制须在小球评估中指出。

## maty 作业与故障记录

工作区：`/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928`。所有 NEMA 工件在其 `experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/`，旧体模结果保持原样。

| 作业 | 范围 | 状态与证据 |
|---|---|---|
| `15506070` | worker 0、1e4 初级 γ 短程 | 通过：218/440/其他为 2911/7089/0；worker 哈希及探测器计数形状通过 |
| `15506077` | 首次 0–199 正式阵列 | 批处理脚本在 `set -u` 下使用空 `args[@]` 报错；Geant4 **未启动**，无 `workers/` 输出，不计入生产 |
| `15506288` | worker 0、完整 5e6 | 通过：218/440/其他为 1470218/3529782/0；约 2 分 44 秒；输出哈希、初级比例及探测器计数核验通过 |
| `15506302` | worker 1–199，最多 32 个并行 | 全部完成；连同 worker 0，共 200/200 独立 worker 逐一通过 `--stage complete` 核验 |

完整发射数为 218/440/其他 **293821153/706178847/0**；20 视角、200 种子、全部输出哈希及计数形状通过。收集后 218/440 能窗的 CntStat 总计分别为 **2462927/1073047**，合并 List 共 **1093494** 行；List 行数不作为初级发射数。对应不可变证据在 [`transport_1e9.json`](transport_1e9.json)。收集 manifest SHA-256 为 `3a1aac57da8bc80e5323c8ccc6623942a3beeaf401bee878ce6c60f1e8659b62`。

在 maty 上核查：

```bash
cd /WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928
python3 experiments/ELLIPSE500x300_H120/check_simulation.py experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/jobs.json
python3 experiments/ELLIPSE500x300_H120/validate_nema_simulation.py experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/jobs.json --stage complete
```

第二条只有 200 个 worker 全部完成且实际初级光子数、能量比例、文件 SHA-256、晶体计数形状均通过才会返回成功；本次已通过。`experiments/FOV120/workflow.py collect` 已校验 20 视角和唯一种子，生成 `CntStat`、`List` 和 `collections/NEMA_Body_H60_1e9.json`。收集器拒绝 smoke 混入正式输入。

## scxi717 重建门槛

`package_imaging.py --datasets NEMA_Body_H60 --archive-stem nema_h60_imaging_1e9` 已生成逐文件 SHA-256 包，归档为 23 个文件、15,886,400 字节；`deploy_imaging.py --archive-stem nema_h60_imaging_1e9` 已安装到 scxi717 的**同一椭圆实验** `generated/` 并逐文件验哈希，禁止新建矩形支持域或复用 FOV120 Factors。

完整网格及完整事件的 10 次资源试跑 `1643079` 在 scxi717 用 4 节点×每节点 1 张 4090 完成（Slurm `COMPLETED 0:0`），`check_remote_pilot.py` 对六路 active/full/history、串窗预测、有限非负值及活动列检查通过。重建筛选后的 Compton 事件为 **97299**，4 个 rank 的显存最大预留 **11.734375 GiB**，最小设备容量约 **23.51648 GiB**，实际峰值约为容量的 50%；Slurm 聚合 step MaxRSS **26508596 KiB**，每节点分配 **60000 MB**。按 55 GiB/节点保守预算重新执行 `resource_budget.py`，GPU 和主存均满足 20% 余量。stderr 在进程退出阶段有 TCPStore `RendezvousConnectionError`，但退出码为 0 且全部试跑输出核验通过；正式作业仍需关注该日志是否在计算阶段复现。

10000 次、每 50 次保存的六路 JSCC 正式作业 **`1643142`** 已于 2026-09-30 在 `gpu_4090` 以相同 4 节点×1 GPU、6 CPU/GPU 启动，NCCL 固定 `bond0`，设置 24 小时上限。作业通过资源预检后仍须以实际结果校验，不能仅按 Slurm 状态宣布成像完成。重建继续使用已有新距离的三套 Factors、椭圆活动列和新 `Sensi_d`。

正式结果必须由 `verify_formal_result.py` 检查 1e9 初级 γ、20 视角、200 个独立 worker、六路最终图、各 200 帧历史、串窗预测、有限非负值、几何/灵敏度哈希及资源余量。科学评价再分别对 Ø10/17/28 的 218 通道与 Ø13/22/37 的 440 通道计算真值对照的 CRC/CNR、背景偏差/CV、积分恢复和迭代曲线；必须区分单能 10:1 真值与两能量直接相加后的 5:1 预览。整个实验不应因某个作业退出码为 0 就宣布有效 FOV 通过。
