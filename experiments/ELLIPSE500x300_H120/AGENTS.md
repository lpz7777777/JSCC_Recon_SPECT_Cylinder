# 当前实验执行约定

本文件只保留当前有效规则。2026-10-19整理前原文和全部历史人工授权按原字节保存在[history/AGENTS_20261019.md](history/AGENTS_20261019.md)；不得把其中较早ACTIVE/RUNNING/PENDING描述当成当前状态。当前导航见[双能入口](../../docs/ac225/README.md)、[状态快照](../../docs/ac225/CURRENT_STATUS.md)和各研究合同。

## 科学与数据边界

- 当前几何500×300×120mm、20view、132040原点/78920活动单元；原生径向6mm、z间距3mm。不得用裁剪或跨能量合成替代完整域验收。
- 剂量按每组合同为实际gamma初级数或模型期望数，不是Ac225衰变数，不假称完整衰变链或符合模拟。
- SC是单光子；JSCC可以指探头或同440 keV的SC+Compton求解，按实际channel解释。
- 后续不生成、计算、绘制或加入比较任何218/440跨能量求和图。历史六路只读，不恢复退休路线。
- 原事件权重、v5连续核、匹配自身S、原MLEM、全1初值、无正则和科学源码SHA保持；改变模型或求解需独立版本及相应验证，不能覆盖旧证明。
- 实际Geant4与前投影+Poisson组分开登记，EHE物理响应HOLD和未判定类别保留；执行/数值通过不代表物理或分辨率认证。
- 科学真值来自原NEMA truth_3mm.npz/manifest或本次Derenzo SHA绑定解析JSON。无平滑、无单图亮度拟合；显示采样/色标/MIP范围须披露。

## 当前授权实验

| 研究 | 当前合同 |
|---|---|
| NEMA JSCC实际5e10 | 440 single、218 corrected、440 Compton三路，10000/save50；固定218背景只取本次最终440 single |
| Derenzo JSCC单能各5e9 | 440 single、440 SC+Compton、218 single，10000/save50；纯218背景零 |
| Derenzo EHE单能各5e9 | 440 single、218 single，200/save1 |
| NEMA既有5e9/EHE各组 | 已完成历史参考，按当前输出/ROI规则分析，不重新输运或恢复旧求和路线 |

Derenzo采用20261015_equal_regions设计，218棒径6/8/10/12/14/16、114根；440棒径10/12/14/16/18/20、53根。用户明确本轮复用现有矩阵，不新建细网格；不能据粗网格图像宣布6mm分辨率。四组各200×25M实际全4π光子，启动检查不计正式剂量。

## 既有作业与控制

- 定时任务继续PAUSED。独立Derenzo一次性流程与既有NEMA流程的并行例外保留，不代表可以开启重复控制器。
- 推进前检查该研究全部registration的PID实际进程身份；Windows可能复用旧PID。当前恢复入口`resume_reconstruction_controllers_20261019.py`保留原14/21天截止，只修复过期PID和已知只读SSH通道超时分类。
- 活控制器旁不并行advance/fetch/submit。Derenzo登记在`generated/derenzo_equal_regions_mono_5e9_20261016/controller_registration.json`；NEMA登记在`generated/jscc_geant4_5e10_10000/*registration.json`。
- Derenzo当前链：JSCC440 validation1713939→验收1713940→暂挂formal1713943；其余三组已有formal_acceptance。以实际新证据为准，不从此段推断实时Slurm状态。
- NEMA5e10的32GPU完整验证1704559/验收1704566已成功并严格取回、生成authority；10000次按实测估计54.87h，原48h总/24h阶段gate不满足，1704572仍暂挂。恢复控制器已按门槛退出，后续须独立执行方案修复，不绕过gate或虚称后台仍在推进。
- 原1685272及旧4090/8x2失败、1688717修复、1686507/1686546成功筛选的源码/请求/登记/输出保留；不得复活失败UUID试跑、重复成功筛选或旧阶段取回来代替推进。
- 旧4090 streaming的/tmp跨作业复用不成立，STORAGE_HOLD不解除；不调用旧advance绕过门控，不写满共享盘，不据全卷空闲代替目录配额。
- 失败/OOM/挂载/通信/磁盘/超时须确认完整退出，保留旧发布与成功输出；仅有界补缺，不重复成功科学计算。实际sbatch意图不明先核对，不重投。
- 完整验证、独立验收和严格SHA取回通过才可生成本组authority并放行对应正式任务。正式时限从实测求解吞吐估算，准备只计一次，不承诺ETA；保留20%实际CPU/GPU/内存余量。

## NEMA当前评价

球ROI使用体素中心距球面至少1.5mm的全部3mm体素、等权均值；不使用分数体积权重或额外内缩。它不保证体素立方体完全在球内；若为空则不可计算，不扩大ROI。全部球共享Phantom腔体内完整背景，排除六球、肺和所有接触边界。以`nema_roi_policy.py`、`reconstruction_output_policy.py`和`reports/nema_interior_roi_20261010/README.md`为准。

EHE0–200和JSCC0–10000分别展示，不把同迭代或同图列当成同收敛。数值完成后仍需科学/视觉QA与报告，Git提交不能代替实验交付。

## 目录整理与Git

2026-10-19用户明确要求整理工程并提交推送。当前采用统一导航与分类索引，科学代码、generated数据、冻结发布、作业登记及远端路径保持原位；此次整理不提交新科学作业、不取消或修改原任务。

提交仅限审核的小型文本源码/配置/几何/说明/证据；排除矩阵、事件响应、原始数组、完整历史、归档、PNG/PDF/SVG等新媒体、编译二进制和敏感凭据。历史已跟踪媒体不重写。归档前后SHA、原说明字节及Git审计保存在`docs/organization/20261019/`。不要覆盖预先存在的工作区改动。
