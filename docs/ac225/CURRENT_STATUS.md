# 当前证据状态

2026-11-17整理。最终调度器观察时间与只读原文见[最终远端证据](../organization/20261117/remote_final_readonly.json)，进度/错误另见[最终日志](../organization/20261117/remote_logs_final_readonly.json)。[最初运行快照](../organization/20261117/remote_readonly.json)保留状态演变。本页不是实时看板，生成时间不充当科学证据时间。

| 研究 / 证据日期 | 计算 | 严格交付 | 视觉QA | 事实与限制 |
|---|---|---|---|---|
| [NEMA5e9原生Cartesian：5090正式三路](studies/nema/nema-cartesian-5090.md)<br>2026-11-17 | FAILED | not-delivered | not-started | 1822462于14:36:55失败退出；进度记录440 SC到10000，但根和三路completed均缺失，不能认定计算完成或完整交付。实际RSS/VRAM超过原80%守卫；输入、输出、原发布和日志保留，未修复/重投。 |
| [NEMA5e9原生Cartesian：4090 r2失败保留](studies/nema/nema-cartesian-4090-r2.md)<br>2026-11-17 | FAILED | not-delivered | not-started | 有效r2正式1822534于14:10:37失败；没有progress/completed，日志同样触发原80%实际RSS/VRAM守卫。原同名无r2分支为未提交预检失败。全部发布/输入/失败输出保留，未重投。 |
| [NEMA5e9：100/80/60% Compton展宽](studies/nema/nema-width-sweep.md)<br>2026-11-17 | completed | strict-delivered | reviewed | 1814153三档各10000次、600帧严格交付。缩窄展宽未改善整体SSIM/NRMSE；保留旧历史差异未通过标志，不选择80/60%替代生产模型。 |
| [HE Derenzo：四路完成及PPT交付](studies/derenzo/opengate-he-derenzo.md)<br>2026-11-17 | completed | strict-delivered | reviewed | 4路800帧严格交付，数值/视觉与14页PowerPoint交付完成。复用HE NEMA响应，没有新响应发射；几何/网格/有限板近似仍未获物理认证。 |
| [OpenGATE HE NEMA 5e9/5e10](studies/nema/opengate-he-nema.md)<br>2026-11-17 | completed | strict-delivered | reviewed | HEGP孔规格+现有NaI，两剂量两能量800帧交付并完成数值/视觉QA。背景BV降低，部分小球恢复下降；不是完整商用相机或纯单因素比较。 |
| [带水NEMA：四分支32路SC完整交付](studies/nema/water-sc-first.md)<br>2026-11-17 | completed | strict-delivered | pending-full-32-review | 严格交付32路6400帧及背景；原metadata的visual_review_pending=true，14路预览和后续专题不能代替全32路视觉验收。旧README的未交付描述被strict_delivery更新。 |
| [带水NEMA：独立供体固定散射模型v2](studies/nema/water-scatter-v2.md)<br>2026-11-17 | completed | strict-delivered | see-dated-overview | 8路1600帧严格交付；独立剂量供体仍含相同体模布局先验。侧窗和440→218串窗偏差单列；并非物理认证。 |
| [Derenzo实际5e10：440 JSCC联合64卡补算](studies/derenzo/derenzo-jscc-5e10-440-joint.md)<br>2026-11-12 | FAILED-at-9950 | strict-partial-199-frames | reviewed-preview | 1771220失败，保存到9950/10000，共199帧已严格取回并完成预览QA；不是正式10000次完成，不恢复旧64卡请求。 |
| [Derenzo实际5e10：440 SC成功输出保留](studies/derenzo/derenzo-jscc-5e10-440-sc.md)<br>2026-11-12 | SC-saved10000;parentFAILED | strict-delivered | see-meeting-report | 父任务1752461失败但成功SC10000/200保留并于11月12日严格取回；联合失败9950单独登记，不重算成功SC。 |
| [Derenzo EHE实际5e10：输运完成、严格传输待修复](studies/derenzo/derenzo-ehe-dense-5e10.md)<br>2026-11-04 | completed | strict-delivered | reviewed | 两能量各200次、400帧严格交付与数值/视觉QA完成；旧传输中断/未提交状态已被交付证据更新，物理HOLD保留。 |

## 阶段判据

提交只由唯一registration/intent证明；排队、运行以新鲜Slurm记录为准；计算完成要求root/batch/extern/step及完成manifest闭合；严格交付要求成员大小/SHA和完整范围；数值QA、视觉QA各有独立证据；物理结论只能按研究报告的论证范围成立。任何缺证阶段均保留unknown/pending，不能自动升级。

5090衔接PID18108在本次开始时PID、create_time、cmdline三项一致；最终两原控制器18108/22172均已按原失败策略退出，登记为STOPPED_FAILURE/FORMAL_FAILED。登记原文件未修改。[最初身份](../organization/20261117/protected_baseline.json) · [最终身份与冻结核验](../organization/20261117/protection_final.json)。另一个CRC40任务属于其他研究，本次未触碰。

原11月3–16日混合状态页保存在[历史快照](history/CURRENT_STATUS_before_20261117.md)。旧ACTIVE、未提交、排队等叙述只代表各自日期，已被上表证据更新的不得继续作为当前状态。旧控制器、被拒绝分支及冻结发布继续原位保留。

[完整研究清单](EXPERIMENTS.md) · [机器状态](status_snapshot.json) · [操作边界](OPERATIONS.md)
