# Compton首散射事件定义修正与NEMA 1e9配对验证

本实验按用户2026-10-04批准计划实施，标识`compton_first_scatter_v2`。3.17e9有界输运、旧分支字节回归、两组独立灵敏度及配对成像均已完成。作业1660254（4节点×1GPU，2:47:44）两组各Compton/JSCC 2000次，四条历史各40帧；远端及取回后本地验收通过。**科学工作判据未达到：Compton最大密度/峰背景下降7.29%/8.27%，JSCC下降41.56%/41.99%，未同时达到50%。Compton22/37mm球CRC损失7.68/5.64个百分点。** 详见[最终验收报告及完整图集](ACCEPTANCE.md)。[阶段离线诊断](PRELIMINARY_REPORT.md)保留为前期证据。本轮已结束，不自动延长迭代、增加光子或改变核。

## 固定问题与对照

仅改变事件接受规则及各自匹配灵敏度。A使用旧Geant4事件定义，B使用理想首散射真轨迹定义；两组都应用当前共享核的完整132040点非对称ARM `q>3`筛选。相等于3保留，旧5e9的1168行删除集合不套用。两组使用同一次输运、同一次晶体能量展宽和同一单光子计数，仍以晶体编号及展宽后的累计沉积能量进入重建。

每组只运行440 Compton和440 JSCC，2000次、每50次保存，四条历史各40帧。完整重建域为500×300×120mm椭圆柱、82040活动单元，NEMA主体高60mm。全1初值，当前K*B及MLEM更新不变。能量域原型和边界积分仅作离线诊断。本轮不启用正则化、绑定、平滑、10000次或追加模拟。

## 事件合同与实现

`Geant4Sim/Geant4Code/include/FirstScatterContract.hh`是独立分类器；`FirstScatterRecorder`在原沉积逻辑之前观察step，不杀轨迹、不抽随机数。探测器逻辑体和晶体copy映射共同识别晶体；沉积使用pre-step归属，离散primary交互使用post-step位置。`GetSecondaryInCurrentStep()`给当前产生的次级添加交互来源标记，并沿后代传播，分开累积首次来源和外来来源的C1能量。采用[Geant4 Tracking接口](https://geant4.web.cern.ch/documentation/pipelines/master/bfad_html/ForApplicationDevelopers/TrackingAndPhysics/tracking.html)，不把局部沉积量当作γ交互的前提。

理想接受须满足：primary440；首次物理交互是C1内Compton；下一次primary物理交互在不同C2，中间无其他散射；后续primary不能返回C1；C1真实沉积及首次交互来源沉积都与首次γ转移能量闭合、外来污染不超过容差；测量后恰好两个晶体超过1keV且对应C1/C2。容差固定`max(1e-6MeV,1e-5×transfer)`。C2允许多步和部分吸收，第三测量晶体会违反原两晶体记录合同。原NumCompt计数和原List写入路径在`legacy`/`paired`中保留。

默认`JSCC_COMPTON_POLICY=legacy`、不开诊断，行为兼容。`paired`保留原List并新增`ListIdeal.csv`；`ideal_first_scatter_v2`使主List采用新集合。独立`JSCC_COMPTON_DIAGNOSTICS=1`允许观察legacy。诊断文件不参与接受规则以外的输运，不调用随机数。

## 旁路和身份

`EventContract.csv`按候选事件保存dataset/worker/seed/view/local event ID、原List及ideal List的零基行号、源位置、两套晶体对、首次primary转移、C1/C2真实及测量沉积、首次来源/外来C1能量、primary交互数/过程/位置/方向、接受标志和首个拒绝原因。合并后增加每视角全局行号，原List保持只读。共同、仅旧、仅新分别统计，不假定B为A子集。

`PrimaryTrace.csv`是primary离散交互轨迹的确定性抽样（event ID模1000，最多100个候选事件/worker）。次级的能量来源逐事件累计；该抽样不是所有次级的逐step轨迹。首转移−C1沉积描述首散射能量闭合/逸出，C2累积及primary后能量描述部分吸收；未把两晶体能量缺口错误标为完整事件所有材料的精确逃逸能量。

`EmittedBins.csv`逐440发射记录三档径向×三档轴向总发射数，统计所有发射事件，不用List数代替发射数。粗径向为r/255≤0.5、0.5–0.85、>0.85，轴向为|z|≤30、30–45、45–60mm。

## 有界数据预算

| 数据集 | worker | 每worker初级γ | 总数 | 种子 |
|---|---:|---:|---:|---|
| NEMA配对重放 | 200，20视角 | 5e6 | 1e9 | 30093001–30093200 |
| 圆域440训练 | 200 | 5e6 | 1e9 | 41004001–41004200 |
| 圆域440独立验证 | 200 | 5e6 | 1e9 | 41005001–41005200 |
| 椭圆440独立验证 | 20，20视角 | 5e6 | 1e8 | 41006001–41006020 |
| 七处440点源 | 140，7×20视角 | 5e5 | 7e7 | 41007001–41007140 |

共760worker、3.17e9初级γ。圆训练/验证均匀源半径255、高120mm，中心(0,−345,0)，无人体材料。点源物体坐标为中心、x=±225、y=±135、z=±57mm，逐视角旋转源。短程门控另有三个1e4光子×两种旁路开关，不混入生产数据。训练、验证、成像种子互不重叠；宏与jobs清单冻结且拒绝原地覆盖。

## 数值和科学门控

`analyze_first_scatter.py`共享生产响应的prepare、cone、归一化和q实现。A/B原始List分别全量扫描，先复现A原97299接受数，再分别应用q筛选，记录新删除行号/文件哈希/分数及保留事件身份。生成各自`Sensi_d`，归一化分母仍为训练源实际初级γ，不按接受数改发射数。保留原Factors和旧S，不覆盖。

独立圆平均闭合误差≤max(2%,3SE)，椭圆总效率偏差≤max(5%,3SE)。同时用真实发射位置直接计数效率对比S在九分区的预测；≥400接受事件的分区若偏差>20%且>3SE则HOLD。训练响应不确定度由200独立worker估计，验证计数使用二项误差。统计不足明确未判定，不自动增加预算。归一化行和闭合不能替代空间验收。

`validation_gate.json`不是提示文本：HOLD会阻止输入冻结和成像部署；重建入口还核对该文件哈希、PASSED状态和未改变的共享核哈希。筛选关闭50次对原1e9第50帧L2≤1e-5；两组10次完整事件试跑必须先通过，才顺序运行两组2000次。每组释放前一组响应，不同时驻留。首选4节点×1GPU，必要时同数据8节点×1GPU重试；实际授予主存及GPU预留均≤80%，NCCL固定bond0。

## 离线研究与输出

`diagnose_first_scatter_points.py`额外用七个独立点源实际保留的List晶体对及能量，在真实发射位置计算同一个非对称ARM的覆盖率。它记录dataset/seed/view/event/原始行号和输入哈希，不重新筛选；比较全圆q≤3但真实位置q>3的事件。该诊断和高斯原型互补，不能把物理首晶体的测量残差当作错误legacy晶体对的实际响应残差。

`first_scatter_offline.py`分解真实交互/转移→晶体中心→真实累计→展宽测量；`analyze_first_scatter.py`给出预测沉积尺度的13% FWHM@511高斯残差和1/2/3σ覆盖率。生产Geant4 11.1.0 option4/LowEP的Doppler与原子效应保留，参见[11.1.0模型源码](https://github.com/Geant4/geant4/blob/v11.1.0/source/processes/electromagnetic/lowenergy/src/G4LowEPComptonModel.cc)。自由电子公式的残差不应全归咎于代码错误。

边界诊断在物体坐标真实椭圆交集中积分，先B/完整单元体积得到A，再与K乘积积分，随后按视角旋转查询。选取小重叠单元，比较原代表点、交叠质心和逐级加密r²/角度/z积分，相邻级变化≤1%标为收敛；不收敛如实报告。A使用现有离散场的XY三角插值和z线性插值，端层最多1.5mm显式外推，是诊断近似，不能当作新生产矩阵已验收。

`compare_first_scatter.py`只接受两份已验收正式2000结果。使用本实验冻结三维球体及既有局部背景ROI，中心z=+1.5mm、冠/矢位、中央72mm MIP，同通道两组全部迭代共用A2000背景尺度，无平滑、gray_r。所有尖峰、分位、f<0.1质量、积分和源外泄漏指标使用完整120mm域；440的13/22/37mm球CRC/CNR及背景CV取全部40帧。图集不能套用技能中的另一个全热柱NEMA真值。

最大密度和峰/背景比同迭代均下降≥50%才达到局部尖峰工作判据；CRC损失>5个百分点单独标记，并报告接受效率损失及旧规则漏收的恢复数。2000次不外推10000次，真轨迹理想选择不代表实际设备具备同样可测筛选能力。

## 复现入口

从仓库根运行；prepare/deploy/submit是有副作用阶段，已有冻结清单时不重复：

```powershell
python experiments/ELLIPSE500x300_H120/first_scatter_workflow.py prepare
python experiments/ELLIPSE500x300_H120/first_scatter_workflow.py deploy
# 首个完整worker与短程门控通过后
python experiments/ELLIPSE500x300_H120/first_scatter_workflow.py submit
python experiments/ELLIPSE500x300_H120/first_scatter_pipeline.py queue-collection
python experiments/ELLIPSE500x300_H120/first_scatter_pipeline.py stage-analysis
# collection_ready后
python experiments/ELLIPSE500x300_H120/first_scatter_pipeline.py launch-analysis
python experiments/ELLIPSE500x300_H120/first_scatter_pipeline.py status-analysis
python experiments/ELLIPSE500x300_H120/first_scatter_pipeline.py point-diagnostics
python experiments/ELLIPSE500x300_H120/first_scatter_pipeline.py fetch-analysis
# 必须PASSED；HOLD不执行以下阶段
python experiments/ELLIPSE500x300_H120/first_scatter_imaging.py freeze
python experiments/ELLIPSE500x300_H120/first_scatter_imaging.py deploy
python experiments/ELLIPSE500x300_H120/first_scatter_imaging.py submit --nodes 4
# 若账号50作业上限，单次有界等待；不与已有提交器重复启动
python experiments/ELLIPSE500x300_H120/first_scatter_imaging.py wait-submit --nodes 4
```

大数据只在`generated/compton_first_scatter_v2`及远端对应子目录；小证据在本报告目录。凭证沿用既有SSH agent/DPAPI方法，不打印或保存密码/私钥。旧1657887与旧自动任务保持停止，账号50作业上限时不取消其他工程任务。
