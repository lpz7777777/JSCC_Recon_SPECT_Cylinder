# process_list_global_audit_v4

2026-10-05。**首轮离线调查已执行并交付，完整柱单元基底和有限能量原型已经实现；正式响应/配对成像仍为HOLD。** 本次复用既有3.17e9输运，新增光子和精细A矩阵为0；原始输入、生产核和图像保持只读。

[科学总报告](REPORT.md)给出五项调查、概率合同及结论边界；[结构化结果](scientific_summary.json)、[运行簿](PRODUCTION.md)、[预先冻结计划](PLAN.md)及[evidence](evidence/)共同追溯数据与执行。

- 内部完整单元的峰由全部20视角和200种子共同支持，50%责任需约1.26万事件；不支持极少数坏事件为该内部峰的充分解释。
- 点源揭示11.3–13.1 keV真实转移物理残差。独立圆源训练的能量概率原型在七源均改善，PIT尾部约5.5%–8.1%降到1.6%–2.7%；5条稀疏角区记录未评分，未改事件集合。
- 192区空间S验证中144个充分区全部通过，偏差−3.8%至+7.4%；其余48区未判定。43单元K积分全部达到加密门槛，代表点误差RMS约2.36%。
- 厚层真实首交互向靠源侧偏移，值得后续位置模型研究；当前先不同时修改位置和能量。
- 78920完整活动单元在独立目录实现，伴随、旋转、体积、分块及事件常数不变性通过；原82040基底未覆盖。

正式核仍需完整事件域、联合条件/固定q接受合同与匹配独立S验收，因此没有新增2000次配对、自动任务或恢复旧作业。优先实施方向是正规化的能量概率与材料转移尾部，不能把诊断评分改善称为尖峰已解决。

可视化：[独立能量概率](figures/independent_energy_probability.png)、[逐级残差](figures/point_residual_components.png)、[空间效率](figures/fine_spatial_efficiency.png)、[原生3D峰曲线](native_spikes/native_spike_curves.png)、[明确Poisson玩具模型](identifiability_toy/statistical_concentration.png)。全部不平滑；原生峰统计不裁剪。完整CSV/责任数组放在独立generated目录，代码与小证据进入Git。

精细场继续停止：[清理收据](../compton_response_geometry_v3/STOP_AND_CLEANUP_20261005.md)。
