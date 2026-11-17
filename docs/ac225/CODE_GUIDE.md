# 从体模到图像的代码路线

科学源码保留既有导入路径。下列链接用于阅读；是否运行、复用哪个冻结版本由研究合同与当前授权决定。

新加入工程先读[完整工作流与环境](WORKFLOW.md)。当前新增研究入口：

| 研究 | 代码 / 条件与证据 |
|---|---|
| 原生Cartesian与4090执行分支 | [研究源码](../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/README.md) · [5090阶段](studies/nema/nema-cartesian-5090.md) · [4090 r2失败](studies/nema/nema-cartesian-4090-r2.md) |
| HE/OpenGATE/PyTomography | [HE NEMA](studies/nema/opengate-he-nema.md) · [有效EHE旋转修复](studies/nema/opengate-ehe-5e9.md) · [HE Derenzo](studies/derenzo/opengate-he-derenzo.md) |
| 展宽与执行优化 | [100/80/60%](studies/nema/nema-width-sweep.md) · [不确定度/CPU配对及局部真实核](studies/diagnostics/jscc-uncertainty-performance.md) |
| 带水固定散射供体v2 | [研究定义及严格交付](studies/nema/water-scatter-v2.md) |

| 阶段 | 当前入口 | 职责 |
|---|---|---|
| 独立带水NEMA单光子修正 | [模块](../../experiments/ELLIPSE500x300_H120/ac225_corrections/README.md)<br>[正式流程](../../experiments/ELLIPSE500x300_H120/ac225_water_gpu_workflow_20261103.py) | 解析水路径、匹配S、同一次展宽侧窗、非负Poisson背景、32路SC；默认不改变旧流程 |
| 几何与支持域 | [geometry.py](../../experiments/ELLIPSE500x300_H120/geometry.py)<br>[config.json](../../experiments/ELLIPSE500x300_H120/config.json)<br>[nema_body_h60_config.json](../../experiments/ELLIPSE500x300_H120/nema_body_h60_config.json) | 全120mm、不等体积、旋转及源分布 |
| 真实Geant4输运 | [README.md](../../Geant4Sim/Geant4Code/README.md)<br>[README.md](../../Geant4Sim/Geant4Code_EHE/README.md)<br>[derenzo_campaign_transport.py](../../experiments/ELLIPSE500x300_H120/derenzo_campaign_transport.py) | JSCC/EHE独立源和初级预算 |
| EHE加密系统矩阵 | [ehe_dense_surface_workflow.py](../../experiments/ELLIPSE500x300_H120/ehe_dense_surface_workflow.py)<br>[ehe_dense_surface_runtime.py](../../experiments/ELLIPSE500x300_H120/ehe_dense_surface_runtime.py) | A218/A440面积分加密；冻结C另列 |
| EHE实际数据重建 | [ehe_dense_rerun_workflow_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_dense_rerun_workflow_20261102.py)<br>[ehe_dense_rerun_runtime_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_dense_rerun_runtime_20261102.py) | 严格输入传输与原MLEM200 |
| 活动密度算子与MLEM | [torch_active_operator.py](../../experiments/ELLIPSE500x300_H120/torch_active_operator.py)<br>[single_checkpoint_mlem.py](../../experiments/ELLIPSE500x300_H120/single_checkpoint_mlem.py)<br>[compton_event_response.py](../../compton_event_response.py)<br>[compton_sparse_ops.py](../../compton_sparse_ops.py) | 体积/ellipse_fraction、匹配前向转置/S和检查点 |
| NEMA440联合 | [nema_5e10_joint_workflow_20261026.py](../../experiments/ELLIPSE500x300_H120/nema_5e10_joint_workflow_20261026.py) | 1740189只复用已完成真实输入与v5响应 |
| Derenzo5e10联合历史与续算 | [原补算](../../experiments/ELLIPSE500x300_H120/derenzo_joint_retry_workflow_20261102.py)<br>[最新失败预览身份](studies/derenzo/derenzo-jscc-5e10-440-joint.md) | 1771220仅保存到9950；原1761195与1769020保留历史，不据旧请求恢复 |
| 20角Poisson | [ehe_nema_low_noise_source_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_source_20261102.py)<br>[ehe_nema_low_noise_workflow_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_workflow_20261102.py)<br>[ehe_nema_low_noise_runtime_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_nema_low_noise_runtime_20261102.py) | 匹配原生有限体积源、独立PCG64分量采样 |
| 60角响应与Poisson | [ehe_60view_operator_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_60view_operator_20261102.py)<br>[ehe_nema_60view_workflow_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_nema_60view_workflow_20261102.py)<br>[ehe_nema_60view_runtime_20261102.py](../../experiments/ELLIPSE500x300_H120/ehe_nema_60view_runtime_20261102.py) | 同环周期响应插值、1/60曝光与新S |
| NEMA评价 | [nema_roi_policy.py](../../experiments/ELLIPSE500x300_H120/nema_roi_policy.py)<br>[reconstruction_output_policy.py](../../experiments/ELLIPSE500x300_H120/reconstruction_output_policy.py)<br>[mip_projection.py](../../experiments/ELLIPSE500x300_H120/mip_projection.py) | 原3D ROI、统一背景和72mm显示 |
| 图像与诊断 | [plot_ehe_dense_completed_1761244.py](../../tools/research/plot_ehe_dense_completed_1761244.py)<br>[plot_ehe_dense_nema_60view_poisson_20261102.py](../../tools/research/plot_ehe_dense_nema_60view_poisson_20261102.py)<br>[diagnose_nema_compton_rings.py](../../tools/research/diagnose_nema_compton_rings.py) | 只读已有原始结果、图形/指标与证据 |
| 目录与查询 | [project_index.py](../../tools/project_index.py)<br>[refresh_ac225_navigation.py](../../tools/research/refresh_ac225_navigation.py) | 本地索引；不导入或提交科学任务 |

## 原始源码与副本

实验顶层为可维护源码；generated中的payload/releases与图集scripts为冻结或产物副本。相似文件名不代表同一SHA，正式身份按冻结release_manifest及合同解析，不用顶层最新文件替代旧发布。

旧resume/watch/advance控制器、streaming4090的/tmp跨作业缓存方案、旧验证/验收提交链保持退休。源码索引列出它们用于追溯，不构成运行授权。

[按职责查看全部源码](SOURCE_INDEX.md)；source_index.json同时提供顶层函数/类名和行号，支持project_index.py find检索。
