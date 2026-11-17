# 225Ac 单光子衰减 / 能窗散射修正原型

独立研究：`ac225_water_corrections_20261103`。探头包括JSCC和EHE；这里的JSCC探头输出仍是 **SC单光子**，没有运行或修改Compton / SC+Compton联合求解。

## 当前状态

本地12项数值检查通过；Linux检查和两套Geant4编译/关闭模式回归证据见[研究记录](../reports/NEMA_Body_H60/ac225_water_corrections_20261103/README.md)。真实带水数据、侧窗矩阵和正式重建的完成状态分别记录，不能把软件检查当成实验完成。

2026-11-04：[故障修复](../reports/NEMA_Body_H60/ac225_water_corrections_20261103/repair_20261104/README.md)增加`postprocess.py`顺序I/O、`side_recovery.py`逐窗口/原始响应复用；收集校验匹配实际direct-batch启动，修正材料元数据接口。新冻结`7ac34ee47ef2eee3`，本地/GPU Linux各15项和CPU Linux3项检查通过。唯一有效侧窗1768730；首批4块已成功，转换实测3.6–4.8秒。以后补交正式重建使用`ac225_water_repair_20261104.py submit_formal`，不能回到原冻结执行入口。衔接进程存活期间仍禁止手动并行调用。

## 模块

| 文件 | 职责 |
|---|---|
| `material.py` | 原NEMA内部轮廓的水场；解析线段与四个CSG区间的并集；源顺时针旋转、晶体端点逆变换 |
| `windows.py` | 原主窗保持；218相邻上下10.9keV、440相邻下22keV；整数标签闭合 |
| `geant4/`、`geant4_overlay.py` | 只在独立复制的源码上增加水CSG和观测；保留原List格式，新标签另存 |
| `operator.py` | 分晶体块的 `B_v * exp(-mu*length)`；原单元体积和ellipse_fraction各一次；匹配前向、转置、自身S |
| `side_response.py` | 新窗口的完整PE谱尾及探头散射；复用EHE已完成的无窗PE；JSCC保留原分层标定，禁止新拟合系数 |
| `background.py` | 非负Poisson侧窗拟合；原始版和物理3×3邻域版；不跨层/探头面/视角，不填补缺失晶体 |
| `mlem.py` | 全1初值、原MLEM；20次仅AC后，每20次更新背景；atomic/fsync保存图像和实际背景状态 |
| `runtime.py` | 每路一张5090的正式SC执行；同一分支最终440生成218固定C背景；完整矩阵SHA及实际资源核对 |
| `surface_audit.py`、`endpoint_audit.py` | 代表点面积分与完整A/C的正权重端点误差界；保留局部超限，不宣称全空间或多次散射轨迹认证 |
| `transport.py`、`cpu_checks.py` | 新真实输运和逐worker收集；本地Linux准备检查，不另投验收作业 |
| `material_maps.py`、`qa.py` | 独立材料/μ图，体内散射与非体内成分偏差分别报告 |

## 科学约定

- 水填原NEMA内腔、z±30mm、1g/cm³；没有壳。冷球和冷肺也填水。原活度数组和源宏不变。
- 水总衰减系数来自[NIST水表](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)的μ/ρ列，对200/300/400/500keV表值做显式log-log插值，密度为1时除10得到mm⁻¹。218为0.0132864181263mm⁻¹，440为0.0102054652179mm⁻¹。不是μ_en，亦不冒充精确能点的XCOM直接查询。
- 材料图只用于显示/QA；正式水路径积分使用解析轮廓。材料场独立于放射性分布。
- 所有窗口复用一次Gaussian展宽。主窗保持闭区间；下侧窗右端开、上侧窗左端开，避免重复计数。
- 标签轴为发射能量218/440 × clean/body/mixed。晶体内同一计数混合两种祖先贡献时归mixed，不拆分这个计数。水内Compton/Rayleigh及吸收后的再发射继承体内相互作用标签。标签只用于事后QA。
- 已有A/C包含探头响应。侧窗模型先预测这些成分，再拟合额外体内背景；主窗原观测从不被相减或截断。
- 440→218按发射能量使用μ440。218→440下窗在最保守5e10个218初级、全部10496晶体可计数的假设下，期望计数上界约1.114e-5，记录后才忽略。
- 背景首次由x20拟合并用于第21次更新，然后x40用于41次，依此类推。200次最后背景更新时间180；10000次为9980。检查点保存实际使用的背景，而非事后重算。
- `none`分支直接调用原 `single_mlem_checkpointed`；218保留既有探头C背景。AC分支重新生成S，不沿用真空S。
- 不缓存20份完整修正矩阵。CUDA融合只加速同一float32解析式；不采用有损存储。
- 原EHE C已知偏差独立保留；能窗残差可能吸收探头误差，必须结合clean/body标签分别报告，不能据图像变好宣称散射模型正确。

## 执行入口

在工程根目录运行：

```powershell
python -B -X utf8 experiments/ELLIPSE500x300_H120/ac225_water_workflow_20261103.py status
python -B -X utf8 experiments/ELLIPSE500x300_H120/ac225_water_gpu_workflow_20261103.py transfer_inputs
python -B -X utf8 experiments/ELLIPSE500x300_H120/ac225_water_repair_20261104.py submit_formal
python -B -X utf8 experiments/ELLIPSE500x300_H120/ac225_water_gpu_workflow_20261103.py strict_fetch
```

有新研究的一次性衔接进程存活时，不并行运行transfer/submit/fetch。身份以PID、create_time、cmdline三项核对。该进程有期限、遇失败或不确定提交即停，不复活旧定时器，不取消其他研究，不自动重算或补投失败科学任务。

四组采集 × 两能量 × 四分支＝32路，各200个保存帧，总6400个图像/背景检查点。440数组最多4路并行；全部成功后218数组最多4路并行。JSCC SC10000/save50，EHE200/save1。正式作业内检查矩阵/伴随/S/资源；没有独立pilot或validation作业。

## 结果与限制

完整严格取回后，`tools/research/plot_ac225_water_corrections_20261103.py`使用原3D真值与统一ROI，生成72mm MIP、3D活度偏差、六球恢复/BV–CNR、外围假活度和探头/体内成分偏差。显示按实际本能量gamma数与原活度积分归一化；不拟合亮度，不平滑，不叠加能量。重建网格仍是原径向6mm、轴向3mm。

代表点直接端点近似检查不能证明多次探头散射的真实入射路径正确，也不能证明低计数能窗法有效。物理和视觉认证标志保持false/pending，直到真实数据和图像逐项评估。低计数恶化、局部误差或C模型偏差都应原样交付。方法背景参见[225Ac散射修正研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12491142/)。
