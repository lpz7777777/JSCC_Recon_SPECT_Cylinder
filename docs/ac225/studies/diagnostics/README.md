# 物理与算法诊断

| 研究 | 版本 | 计算 / 交付 | 结论 |
|---|---|---|---|
| [440连续材料能量核：legacy独立校准及2000次配对](compton-v5-legacy.md) | historical-reference | accepted-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [ideal 1e9连续核对照；与legacy剂量实验分开](compton-v5-ideal.md) | historical-reference | historical-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [首散射事件定义配对研究](first-scatter-v2.md) | historical-reference | historical-reference / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [全局事件算子、测度及物理响应审计](event-physics-audit.md) | historical-reference | accepted-diagnostic / see-original-evidence | 以原始报告的研究范围和结论为准。 |
| [NEMA实际5e9：旧Compton核回退对照](nema-prejune-compton.md) | diagnostic | completed / strict-delivered | 近期NEMA实际投影配旧核/匹配S/OSEM8完成10000次，明显环纹和尖峰仍存在，回退未消除问题。 |
| [Compton环状伪影](compton-ring.md) | diagnostic | analysis-complete / source-bound-analysis | 环纹与网格/响应离散化值得隔离研究；观察性关联不足以确定唯一成因。 |
| [Compton角点过渡与体积](compton-transition.md) | diagnostic | analysis-complete / source-bound-analysis | 未发现简单漏乘/重复乘体积或S约定错误；中心点响应代表有限单元的积分误差仍是候选。 |
| [Compton算法Git历史](compton-git-history.md) | diagnostic | analysis-complete / source-bound-analysis | 源码版本、核/S/筛选/求解均有差别；历史图像外观不能代替同源匹配实验。 |
| [旧Compton结果备份审计](compton-legacy-backup.md) | diagnostic | analysis-complete / source-bound-analysis | 备份Image_JSCCD为纯Compton；511keV旧数据与440keV近期NEMA/FOV不同，不当作直接等价基准。 |
| [NEMA新旧Compton流程对比](compton-new-vs-old.md) | diagnostic | analysis-complete / source-bound-analysis | 旧流程后期起伏更强，回退未消除环纹；同时改变多步骤，不能唯一归因。 |
| [EHE单晶体空间响应](ehe-crystal-response.md) | diagnostic | analysis-complete / source-bound-analysis | 原矩阵六个晶体的分能量MIP/切片；晶体响应覆盖及面采样问题另由准直器审计解释。 |
| [EHE准直器响应与面积分](ehe-collimator-audit.md) | diagnostic | analysis-complete / source-bound-analysis | 256点/可见面的局部积分不足已确认；65536点加密版本独立完成，物理模型仍需验证。 |
| [EHE准直器FOV间隙](ehe-fov-clearance.md) | diagnostic | analysis-complete / source-bound-analysis | 准直器前端298.5mm，完整旋转FOV最小间隙48.5mm，未伸入FOV。 |
| [EHE加密矩阵边缘高值](ehe-dense-edge.md) | diagnostic | analysis-complete / source-bound-analysis | 噪声放大与空间响应失配并存；NEMA440→218串窗预测低约25.27%，加密没有消除主要尖峰。 |
| [EHE原矩阵边缘高值历史诊断](ehe-edge-original.md) | diagnostic | analysis-complete / source-bound-analysis | 保留早期边缘/S/计数诊断；当前结论优先使用11月加密矩阵及匹配Poisson对照。 |
| [JSCC v5不确定度与隔离等价执行优化](jscc-uncertainty-performance.md) | current | analysis-complete / local-report | CPU配对计时与真实核局部检查已有证据；合成B/C不等于完整物理算子，CUDA stress未运行，不能推断完整GPU速度或图质改善。 |
| [ROI/源对齐派生评价](nema-roi-audit-20261115.md) | current | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [背景+体外误差派生评价](nema-bg-out-error-20261115.md) | current | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [五档3D高斯派生评价](nema-sc-jscc-he-gaussian-20261115.md) | current | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [CNR/CRC–BV派生评价](nema-sc-jscc-he-cnr-crc-bv-20261114.md) | current | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |
| [SC/JSCC/HE综合派生评价](nema-sc-jscc-he-5e9-5e10-20261114.md) | current | analysis-complete / derived-report | 复用已交付结果的派生指标与图形。原生与共同查询网格、滤波强度、迭代范围分别保留；不改变重建，不证明物理认证。 |

[返回总入口](../../README.md)
