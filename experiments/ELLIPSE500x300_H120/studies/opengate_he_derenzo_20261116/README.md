# HE Derenzo 5e9 / 5e10

**2026-11-17完成：** 四路各200帧、共800帧已严格交付；会议PPT第11–14页已替换为HE结果并通过14页真实PowerPoint渲染与视觉检查。见[完成记录](COMPLETED_20261117.md)。下文保留原执行定义；启动登记中的ACTIVE不代表控制器仍存活，实际完成以controller_exit和PPT_delivery_20261117为准。

用户授权：计算HE的Derenzo 5e9与5e10结果，替换会议PPT中的EHE结果。包括独立218与440 keV四组，原SC/JSCC结果保留。

## 科学定义

- 每组分别实际发射5e9或5e10个单能gamma；不是225Ac衰变数。20视角等曝光、全4π、单位权重。
- 原20261015等区域解析圆柱源：218 keV棒径6/8/10/12/14/16 mm、114根；440 keV棒径10/12/14/16/18/20 mm、53根；轴向±30 mm、背景为零。按精确棒体积做预登记multinomial发射数分配；棒内连续均匀采样，不将6 mm棒先栅格化。材料为原无水基线。
- HE采用已有GE HEGP孔规格＋现有NaI探测器：4 mm六边形孔、66 mm孔长、1.8 mm隔板，准直器前端Y=298.5 mm、晶体中心Y=369.5 mm。不是完整商用GE相机，也不是EHE投影改名。
- 复用HE NEMA科学冻结46d438089710a6ce的已严格验收响应，88个worker、4.4e9校准发射不重算。核SHA：`fdb5270e175eb59056db8b27da9166b2699ec74eede20d555e4fc420f83e0be1`。
- 新科学冻结`34a52489b6362ba0`。原探头、物理、读出和一次能量展宽保持；新响应发射数为零。只新增Derenzo源及单能路由。
- HE原4 mm笛卡尔126×126×34网格、完整椭圆500×300×120 mm支持、原体积分数和匹配S、PyTomography MLEM200/save1，全1初值、无正则。独立单能背景为零，不用另一组440图产生218背景。
- 原HE深度插值/四孔相位均值CDR近似仍有有限板边缘及相位依赖限制；数值闭合不等于物理响应认证或分辨率认证。

## 执行与权威

`../../reports/Derenzo/opengate_he_derenzo_20261116/`中的freeze、各路imaging_job/collect_job、status_latest、strict_delivery为权威。所有活动代码/发布不得热改。当前控制器存活须核对PID、create_time和cmdline，不能并行submit/fetch/reconstruct。

正式采集为100/1000个50M worker，两个能量各最多80并行，总最多160。5e10分别依赖同能量5e9收集。平台账户最多2000提交元素，因此5e10的440数组在名额足够后补交；第一次明确拒绝的intent已保留，已查询调度器确认未创建作业。

没有独立验证作业。源旋转/连续圆柱采样检查在正式worker0内执行，额外6144个标签为numerical check的历史不计入成像预算；孔固体检查零额外历史。正式collector逐worker核对全部文件SHA及精确计数。取回前检查全部array root/batch/extern退出0、实际内存80%上限、冻结源码、输入、日志与资源记录。

专属有限控制器`controller.py`最多12小时，仅处理上述四组，完成strict fetch后串行本地8CPU正式MLEM。失败停止并保留，不自动重复成功或失败的科学计算，不恢复周期定时器。它完成计算后仍需要视觉QA及PPT交付，不能把COMPLETED误称PPT已完成。

## 图与PPT

`visualize.py`从strict_delivery读取检查点，制作中心72 mm原生MIP（18个4 mm层），全XY、gray_r、相对热棒活度0–2，无滤波或亮度拟合。PPT中HE选帧20/60/200；SC/JSCC仍500/2000/10000，Derenzo JSCC5e10仍以用户指定9950为终点。

输出研究目录：`Analysis_Plots/research/ac225_results_meeting_he_derenzo_20261116`。修改原会议PPT的第11–14页与条件说明，保留原始备份、NEMA页和12张可编辑CNR–BV图。最终在真实PowerPoint逐页渲染、科学数组/图片/图表SHA验证及视觉检查后才覆盖用户PPT。

OpenGATE圆柱源参数按[官方GenericSource说明](https://opengate-python.readthedocs.io/en/master/user_guide/user_guide_reference_sources_generic_source.html)：`radius`为半径，`dz`为半长；本研究同时在实际Geant4发射顶点中检查该定义。
