"""Repository documentation templates for the curated225Ac navigation registry."""
from pathlib import Path
import hashlib
import json

def write_documents(root,studies,routes,folders,now,put,js,md,link,status_label):
    D=root/'docs/ac225';H='experiments/ELLIPSE500x300_H120';A='docs/organization/20261103'
    byid={s['id']:s for s in studies}
    def page(path,text):put(root/path,text.strip()+'\n')
    def links(doc,pairs):return '\n'.join('- '+md(p,root/doc,t) for p,t in pairs)
    def table(doc,ids):
        text='| 研究 | 计算 | 交付 / 图像 | 当前结论 |\n|---|---|---|---|\n'
        for ident in ids:
            s=byid[ident];text+=f"| {md(s['guide'],root/doc,s['title'])} | {status_label(s['compute_status'])} | {status_label(s['delivery_status'])} / {status_label(s['visual_status'])} | {s['conclusion']} |\n"
        return text
    overview='''# 225Ac成像工程导航

本工程以218与440keV的gamma代理源研究JSCC和EHE三维SPECT。主几何为500×300×120mm，NEMA和Derenzo源主体H60；实际输运以20视角为主，另有60视角的EHE模型合成对照。光子预算不是225Ac衰变次数。

当前推荐从[可搜索工程目录](INDEX.html)或[成像结果目录](../../Analysis_Plots/results/REPORT.html)开始。当前EHE实际结果使用65536点/可见面加密矩阵；原EHE矩阵画廊、前投影＋Poisson和纯Compton诊断分别标记，防止把不同来源混作同一组。

| 要查什么 | 入口 |
|---|---|
| 各组图像与数据来源 | [结果说明](RESULTS.md) · [结果网页](../../Analysis_Plots/results/REPORT.html) |
| 已完成、排队和待处理项 | [当前状态](CURRENT_STATUS.md) |
| 研究得到什么结论 | [结论与限制](CONCLUSIONS.md) |
| 每个研究的参数、代码和证据 | [分类研究页](studies/README.md) · [实验表](EXPERIMENTS.md) |
| 从输运到重建的代码路线 | [代码导读](CODE_GUIDE.md) · [函数与文件索引](SOURCE_INDEX.md) |
| 目录职责与原始数据位置 | [目录图](DIRECTORY_MAP.md) |
| 探头、FOV和两种体模 | [系统与体模](SYSTEMS_AND_PHANTOMS.md) |
| SC/JSCC、剂量、体积、ROI、矩阵版本 | [科学口径](SCIENCE_CONVENTIONS.md) |
| 维护索引及查状态 | [操作说明](OPERATIONS.md) |
| 大数组、冻结、历史和Git边界 | [数据管理](DATA_POLICY.md) |

```powershell
python tools/project_index.py list --topic 60view
python tools/project_index.py show nema-5e10-v5-joint
python tools/project_index.py find ehe_dense
python tools/project_index.py files --group reconstruction
python tools/project_index.py check
```

命令只读本地索引，不导入科学工作流、不连接超算、不提交作业。状态是带时间的证据；源码及数组位置保持稳定。整理前说明与身份核对见[本次整理记录](../organization/20261103/README.md)。
'''
    page('docs/ac225/README.md',overview)
    page('README.md','''# JSCC与EHE三维SPECT工程

当前工程研究218/440keV的225Ac代理gamma成像：JSCC探头的单光子及同能量SC+Compton联合重建，EHE探头单光子成像，以及物理响应和离散化诊断。

**[打开工程与研究目录](docs/ac225/INDEX.html)** · **[打开成像结果](Analysis_Plots/results/REPORT.html)** · **[查看当前状态](docs/ac225/CURRENT_STATUS.md)**

| 查询内容 | 说明 |
|---|---|
| [225Ac总导航](docs/ac225/README.md) | 研究类别、数据口径与查询示例 |
| [研究结论](docs/ac225/CONCLUSIONS.md) | 已确认的现象、候选机制和剩余限制 |
| [代码路线](docs/ac225/CODE_GUIDE.md) | 体模→输运→系统矩阵→重建→ROI/图像 |
| [工程目录图](docs/ac225/DIRECTORY_MAP.md) | 源码、证据、运行数据、图形与历史位置 |
| [实验与证据](docs/ac225/EXPERIMENTS.md) | 每个研究的参数、作业及交付状态 |
| [科学口径](docs/ac225/SCIENCE_CONVENTIONS.md) | SC/JSCC命名、gamma预算、密度/体积、72mm MIP |
| [操作和数据管理](docs/ac225/OPERATIONS.md) | 只读查询、冻结保护与目录维护 |
| [其他工程文档](docs/README.md) | 基准、复现与跨工程研究 |

当前三维主工程位于[experiments/ELLIPSE500x300_H120](experiments/ELLIPSE500x300_H120/README.md)，FOV为500×300×120mm；真实输运默认20角，60角仅指独立EHE前投影＋Poisson研究。NEMA及两种单能Derenzo的输入、几何、剂量分别登记。

SC表示单光子，图中JSCC表示同能量SC+Compton联合；纯Compton单列。218和440不做跨能量图像求和。实际Geant4、矩阵合成投影和旧矩阵参考在结果目录中分开。当前光子数不是225Ac衰变数。

```powershell
python tools/project_index.py list --topic nema
python tools/project_index.py show ehe-nema-poisson-60view
python tools/project_index.py find Compton
python tools/project_index.py check
```

原始源码、矩阵、事件、完整历史和已冻结发布保留原路径；新增分层目录保存说明及查询指针。旧本地60mm/FOV120接口继续从[Reproduction](Reproduction/README.md)和[Results](Results/README.md)查询。本轮整理说明见[2026-11-03审计](docs/organization/20261103/README.md)。
''')
    page('docs/README.md','''# 工程文档

当前218/440keV成像从[225Ac总入口](ac225/README.md)查找；[工程网页](ac225/INDEX.html)可按研究、能量、剂量、角度和作业号搜索。

| 分类 | 文档 |
|---|---|
| 当前结果与结论 | [结果](ac225/RESULTS.md) · [结论](ac225/CONCLUSIONS.md) · [状态](ac225/CURRENT_STATUS.md) |
| 代码与目录 | [代码路线](ac225/CODE_GUIDE.md) · [源码索引](ac225/SOURCE_INDEX.md) · [目录图](ac225/DIRECTORY_MAP.md) |
| 科学和操作 | [科学口径](ac225/SCIENCE_CONVENTIONS.md) · [操作](ac225/OPERATIONS.md) · [数据管理](ac225/DATA_POLICY.md) |
| 研究回顾 | [双能研究回顾](DUAL_ENERGY_RESEARCH_REVIEW.md) · [双能基准](DUAL_ENERGY_BASELINE.md) |
| 跨工程规划 | [研究策略](RESEARCH_STRATEGY.md) |
| 历史和整理证据 | [历史说明](history) · [整理记录](organization/20261103/README.md) |

带日期的旧状态保留在归档，不作为当前作业指令。当前每个研究的合同、冻结、数据身份和物理结论通过分类研究页反查。
''')
    status_ids=['nema-jscc-5e10','nema-5e10-v5-joint','nema-jscc-5e9-v5','derenzo-jscc-440','derenzo-jscc-218',
      'nema-ehe-dense-5e9','nema-ehe-dense-5e10','derenzo-ehe-dense-5e9','ehe-nema-poisson-20view','ehe-nema-poisson-60view',
      'derenzo-jscc-5e10-218','derenzo-jscc-5e10-440-sc','derenzo-jscc-5e10-440-joint','derenzo-ehe-dense-5e10']
    doc='docs/ac225/CURRENT_STATUS.md'
    observation_times=[]
    for host in ['gpu','cpu']:
        p=root/A/(host+'_readonly_status.json')
        if p.exists():observation_times.append(host.upper()+'观察时间：'+json.loads(p.read_text('utf8'))['checked_at'])
    page(doc,'# 当前研究阶段与待处理项\n\n本地整理时间：'+now+'。已完成项来自严格交付与图像证据；未完成项于本次只读查询GPU/CPU队列确认。状态不自动刷新。\n\n'+'；'.join(observation_times)+'。\n\n'+table(doc,status_ids)+'''
## 尚未交付的部分

- **Derenzo 5e10的440 JSCC联合1761195**：仍PENDING/Priority，8×8张5090；只有正式请求，不能当作已运行或已成图。
- **EHE Derenzo 5e10**：218/440各500个100M实际输运worker及收集均COMPLETED/0:0；本地严格传输存在中断partial，原衔接PID60468已失败退出，后续两路GPU未提交。先有界修复严格传输，不重算成功输运。
- **JSCC Derenzo 5e10的440 SC**：原父任务1752461整体失败，但SC10000/200保存成功；需独立核对并交付成功输出，不重算SC。218 SC1752710已严格取回，统一新版画廊未索引。

## 进程与历史

已登记的旧一次性衔接42492、60468，以及20/60角结果后处理66928、80600均已退出；不据旧JSON状态复活它们。旧定时任务保持停止。目录整理只做只读查询，没有提交、取消或重新取回科学任务。

原始查询见[GPU](../organization/20261103/gpu_readonly_status.json)、[CPU](../organization/20261103/cpu_readonly_status.json)。此前混合排队/运行叙述的状态页按原字节保存在[整理前状态](../organization/20261103/before/docs/ac225/CURRENT_STATUS.md)。
''')
    snapshots={}
    for host in ['gpu','cpu']:
        p=root/A/(host+'_readonly_status.json')
        if p.exists():snapshots[host]=json.loads(p.read_text('utf8'))
    js(D/'status_snapshot.json',dict(snapshot_at=now,semantics='Dated local delivery evidence plus explicit read-only runtime snapshots',
        runtime_snapshots=snapshots,studies=[{k:s[k] for k in ['id','status','compute_status','delivery_status','visual_status','jobs','observed_at','evidence']} for s in studies]))
    oldformal=root/A/'before/docs/ac225/formal_jobs_current.json'
    prior=json.loads((oldformal if oldformal.exists() else D/'formal_jobs_current.json').read_text('utf8'))
    js(D/'formal_jobs_current.json',dict(schema_version=2,snapshot_at=now,execution_policy='formal_only',
        separate_validation_jobs_enabled=False,automatic_submit_controllers_enabled=False,
        cancelled_jobs=prior.get('cancelled_jobs',[]),semantics='Evidence snapshot; completed jobs must not be resubmitted',
        experiments={s['id']:{k:s[k] for k in ['title','status','compute_status','delivery_status','visual_status','jobs','evidence']} for s in studies if s['jobs']}))

    conclusion='''# 225Ac代理成像研究结论

以下区分计算完成、数值一致性、图像表现和物理解释。SC为单光子；JSCC为同能量SC+Compton联合。研究使用218/440gamma代理源，未模拟完整225Ac衰变链。

## NEMA与JSCC

NEMA实际5e9的v5基准、实际5e10的SC/纯Compton三路，以及5e10的440联合1740189均有已交付结果。联合图在该条件下背景BV约35.94%，比SC约40.50%降低，但13mm热球CNR下降，说明噪声下降与小目标对比不是同一指标。纯Compton后期环纹、条纹和尖峰仍明显。

[5e10联合结果](../../Analysis_Plots/research/nema_joint5e10_1740189_20261029/REPORT.html) · [原三路](../../Analysis_Plots/research/nema_jscc5e10_20261026/README.txt)

## Compton环纹与不等体积网格

角点数变化的实际单元边界为r=39、75、111mm等；部分峰接近过渡带，并非全部峰恰在边界。当前未知量是密度，Cartesian→Polar时响应已乘单元体积，Compton事件归一化及匹配S沿用同一体积约定。实际审计未发现简单漏乘、重复乘体积或统一倍数错误。角点数突变和中心点近似有限单元响应的积分误差仍是值得隔离的候选机制，不能仅从外观确定成因。

使用近期同一NEMA投影回退旧Compton核/配套S/OSEM8后，后期伪影仍很强。旧OSEM8每轮含8次子集更新，且使用完整S的约定影响总体尺度；它不是与当前MLEM同更新次数的单因素试验，更不是对旧511keV结果的精确重放。

[过渡和体积审计](../../Analysis_Plots/research/nema_compton_transition_audit_20261026/REPORT.html) · [新旧流程对比](../../Analysis_Plots/research/nema5e9_new_vs_prejune_20261026/REPORT.html) · [历史备份审计](../../Analysis_Plots/research/legacy_compton_backup_audit_20261026/REPORT.html)

## EHE响应与边缘高值

准直器前端距中心298.5mm，旋转完整FOV的最小间隙48.5mm，不支持“前端伸入FOV”这一解释。256点/可见面的局部面积分误差确实存在；A218/A440已提高到65536点并重算自身S。但同一Derenzo实际投影的边缘峰和最高点位置变化很小，加密没有消除主要伪影。

低计数、78920活动单元、无正则MLEM和边缘灵敏度较低共同支持噪声放大的解释；实际投影与响应也有明确空间失配。NEMA5e10真实440→218贡献100326，当前C预测74978.5，约低25.27%；重建固定背景同样低约25%。这与面积分加密是不同问题。物理响应尚未闭合，不把数值S匹配当作物理认证。

[完整边缘诊断](../../Analysis_Plots/research/ehe_dense_edge_diagnosis_20261102/REPORT.html) · [FOV间隙](../../Analysis_Plots/research/ehe_fov_clearance_20261102/REPORT.html) · [准直器审计](../../Analysis_Plots/research/ehe_collimator_response_audit_20261029/REPORT.html)

## 前投影与Poisson计数对照

20视角下，生成和重建匹配的原生矩阵合成数据在5e9/5e10仍重现明显边缘高值；提高到5e12后尖峰显著减弱，但仍有模糊。这说明这一类伪影不需要真实Geant4/响应失配作为前提，同时也不排除真实数据已确认的模型偏差。

60视角对照保持每档总期望gamma量不变，曝光按1/60分配。新6°响应来自同环周期线性插值，并重新计算匹配S；原20位置每隔3视角精确保留。这一次独立噪声实现中，外围ROI发射量占比8组均下降；整体误差改善有限，440在5e9/5e10反而略升。相同200次不保证同等收敛。

| 总期望gamma | 440原生全域NRMSE 20→60 | 218原生全域NRMSE 20→60 |
|---|---|---|
| 5e9 | 297.08% → 303.35% | 364.99% → 361.70% |
| 5e10 | 86.83% → 91.74% | 108.47% → 104.84% |
| 5e11 | 35.73% → 35.40% | 38.44% → 37.70% |
| 5e12 | 26.30% → 25.53% | 24.84% → 23.07% |

角采样、响应插值和Poisson实现同时改变，没有重复噪声实验，不能把差异单独归因于角度数量或宣布统计显著性、分辨率提升。两组均远端核对1600帧，本地严格取回48选帧；完整本地历史QA仍为false。

[20角模型对照](../../Analysis_Plots/research/ehe_dense_nema_low_noise_20261102/REPORT.html) · [20/60角比较和原始指标](../../Analysis_Plots/research/ehe_dense_nema_60view_poisson_20261102/REPORT.html)

## Derenzo与当前边界

5e9的440 SC/JSCC、218 SC及EHE两能量已完成，当前实际EHE图优先采用加密版本。218与440分别使用不同棒径和布局，不能将它们视为同一几何的纯能量对照。原径向6mm网格可能限制最小6mm棒的判断，未认证6mm空间分辨率。

实际5e10的218 SC已交付；440 SC已保存成功输出，联合补算1761195仍排队。EHE两能量实际5e10的输运/收集完成，但严格传输中断，GPU重建未提交。这些条件不能列为已完成图像。

[Derenzo设计](studies/design/derenzo-design-selected.md) · [实际结果目录](../../Analysis_Plots/results/REPORT.html) · [待处理项](CURRENT_STATUS.md)
'''
    page('docs/ac225/CONCLUSIONS.md',conclusion)
    page('docs/ac225/SCIENCE_CONVENTIONS.md','''# 当前科学口径

## 系统、通道和发射量

- SC＝单光子成像；JSCC结果＝同能量SC+Compton联合求解。技术语境中的JSCC探头、历史`JSCC_218`目录不代表联合通道。纯Compton明确标记Compton。
- 不生成218/440跨能量求和图。218没有本轮SC+Compton联合结果。
- 实际NEMA的5e9/5e10为各合同规定的双能gamma初级总数；Derenzo每探头/能量独立单能发射。模型Poisson组是总期望发射gamma，不是探测计数，更不是225Ac衰变数。
- NEMA源积分与产额0.114/0.259用于分配218/440预算；固定理论密度尺度不是亮度拟合。各组精确实际初级数按collection与manifest查证。

## 几何、测度和矩阵

完整FOV500×300×120mm，132040全圆网格点/78920活动单元，径向6mm、轴向3mm。NEMA及本轮Derenzo主体H60，不能以72mm显示裁剪替代120mm响应支持域。

密度未知量ρ乘原单元有效体积得到源总量。矩阵转换的体积、活动算子的ellipse_fraction恰好各一次；前向、转置和自身S须属于同一模型。计数为零的晶体/bin仍参与观测模型及S，不删除零计数bin。

实际输运及主要重建20角：0,18,…342°。60角模型组为0,6,…354°、曝光1/60；同环响应列周期线性插值，并使用匹配60角S。它不等于直接重复20个投影三遍，也不等于独立计算了60角物理矩阵。

EHE A218/A440当前加密到65536点/可见面，原C440→218仍冻结；旧256点EHE结果只作版本参考。晶体物理bin为4×10×4mm，68×34共2312个，无新增DOI分层；面积分采样点不是物理探测器小像素。

## 求解与背景

JSCC当前正式10000/save50；EHE200/save1。全1密度初值、原MLEM、无正则按各冻结合同保持。旧Compton适配对照为OSEM8，不将相同迭代等同相同更新次数或收敛。

NEMA218固定加性背景仅来自同次/同剂量最终440单光子图经C；合成数据生成时的真实C不作为已知重建背景。单能Derenzo背景零。

## 图像和指标

中心72mm MIP为原40层中的24层：中心−34.5至+34.5mm、边界±36mm；保留完整XY。gray_r白低黑高，无平滑、无单图拟合增益。主统一色阶超限显示黑色，另提供共同无截顶图，原数组不截断。

NEMA球ROI采用所有3mm体素中心距原3D球面至少1.5mm的点，等权统计；统一背景为完整体模内部，排除六球、肺和接触边界。不得以MIP二维ROI替代3D指标。实际规则以[nema_roi_policy.py](../../experiments/ELLIPSE500x300_H120/nema_roi_policy.py)为准。

计算结束、严格取回、数值/版式QA和物理/分辨率结论分别登记。48选帧本地检查不冒充1600帧完整本地历史QA。当前EHE物理响应HOLD保持。
''')
    page('docs/ac225/SYSTEMS_AND_PHANTOMS.md','''# 系统与体模

## 共用FOV与成像网格

当前椭圆FOV为500×300mm、完整轴向120mm；132040全圆点中78920个活动单元。径向间距6mm、轴向3mm，角点数随半径分段增加，单元体积不相同。真实采集默认20个旋转角，18°间隔；60角为独立EHE模型合成对照，不改变原生图像基底。

NEMA和Derenzo源主体H60。图中中心72mm MIP只改变显示范围，不把响应域或真实源高度改成72mm。当前体积和重叠定义见[geometry.py](../../experiments/ELLIPSE500x300_H120/geometry.py)及[科学口径](SCIENCE_CONVENTIONS.md)。

## JSCC与EHE

JSCC探头支持单光子计数及Compton双作用点观测；图中SC与同能量SC+Compton联合使用不同观测链，纯Compton单独标记。探头尺寸、晶体顺序和材料按各正式发布中的参数及Geant4源代码身份核对，不能从历史目录名称推断当前二进制。

EHE为准直单光子系统。NaI物理bin为4×10×4mm（X、深度Y、Z），68×34共2312个，正面272×136mm、深度10mm，本轮未细分DOI。65536个点/可见面是响应面积分节点，不是把物理晶体切成65536个读出像素。当前加密A218/A440保持探头几何，C440→218未同步重算。

当前EHE未增加新侧屏蔽。四侧10mm铅框只保留为设计方案；准直器前端298.5mm、完整FOV最小48.5mm间隙来自独立几何审计。

[探头与场景图](../../Analysis_Plots/research/advisor_progress_20261014/使用说明.txt) · [EHE几何间隙](../../Analysis_Plots/research/ehe_fov_clearance_20261102/REPORT.html) · [侧屏蔽方案](../../Analysis_Plots/research/ehe_side_shield_design_20261029/REPORT.html)

## NEMA Body

采用原3mm三维真值和manifest的原3D球心，六球直径10/13/17/22/28/37mm，并包含中心肺结构。218热球为10/17/28mm，440热球为13/22/37mm；精确背景、球、肺及源分布以原数组为准，不从重建图反推。两能场由合同定义的gamma代理源生成，不是假称完整225Ac衰变链。

[分能量体模结构](../../Analysis_Plots/research/nema_structure_20261014/README.md) · [原源manifest](../../experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/manifest.json) · [truth_3mm.npz](../../experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/truth_3mm.npz)

## 独立单能Derenzo

当前唯一设计为20261015_equal_regions，各棒径对应的三角区域面积相近，棒区与FOV中心之间留空。

| 能量 | 棒径/mm | 棒数 | 数据语义 |
|---|---|---:|---|
| 218keV | 6 / 8 / 10 / 12 / 14 / 16 | 114 | 此体模独立单能输运 |
| 440keV | 10 / 12 / 14 / 16 / 18 / 20 | 53 | 此体模独立单能输运 |

两个能量的几何不同，每个探头/能量各自发射5e9或5e10 gamma；单能218不注入另一440体模的背景。现轮复用原矩阵，不据径向6mm网格图像宣布最小6mm棒分辨率。

[当前解析设计与原图](../../Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions/design_notes_zh.txt) · [设计研究页](studies/design/derenzo-design-selected.md)
''')
    page('docs/ac225/RESULTS.md','''# 成像结果按条件查询

[打开可搜索结果目录](../../Analysis_Plots/results/REPORT.html) · [机器索引](../../Analysis_Plots/results/data/current_results.json)

当前结果指针按以下层级组织，文件只引用原数组、图像及证据：

```text
Analysis_Plots/results/current/
  actual/<体模>/<总gamma量>/<视角>/<能量>/<SC或JSCC或EHE>/
  synthetic/NEMA/<总期望gamma量>/<20views或60views>/<能量>/EHE/
    README.md       条件、版本、结果入口与交付范围
    case.json       作业、channel、矩阵、原始来源和查询路径
```

实际结果的NEMA5e9/5e10、Derenzo5e9可看当前SC/JSCC及新加密EHE；Derenzo5e10的未交付条件明确列阶段。模型合成NEMA20/60角覆盖5e9/5e10/5e11/5e12，各含218/440。

`results/NEMA/...`及`results/Derenzo/5e9/...`保存2026-10-29原方法对比画廊，其EHE为旧矩阵；[旧网页快照](../../Analysis_Plots/results/history/20261029/REPORT.html)和[data/result_catalog.json](../../Analysis_Plots/results/data/result_catalog.json)保留原15路线，不能当成加密版本。最新索引为`current_results.json`。

每个原research专题含figures、data、scripts及来源SHA：figures是显示产物，data为指标/显示参数与QA，scripts为实际绘图副本；科学数组仍在各研究generated。新导航不复制矩阵、事件响应或完整历史。

所有结果按[科学口径](SCIENCE_CONVENTIONS.md)解读；[研究结论](CONCLUSIONS.md)说明噪声、环纹、EHE响应失配与60角插值限制。原BV–CNR专题的EHE曲线属于加密前版本，当前合成组的3D BV–CNR仅来自六个真实选帧。
''')
    code_doc='docs/ac225/CODE_GUIDE.md';code_lines=['# 从体模到图像的代码路线','',
      '科学源码保留既有导入路径。下列链接用于阅读；是否运行、复用哪个冻结版本由研究合同与当前授权决定。','',
      '| 阶段 | 当前入口 | 职责 |','|---|---|---|']
    stages=[('几何与支持域',[H+'/geometry.py',H+'/config.json',H+'/nema_body_h60_config.json'],'全120mm、不等体积、旋转及源分布'),
      ('真实Geant4输运',['Geant4Sim/Geant4Code/README.md','Geant4Sim/Geant4Code_EHE/README.md',H+'/derenzo_campaign_transport.py'],'JSCC/EHE独立源和初级预算'),
      ('EHE加密系统矩阵',[H+'/ehe_dense_surface_workflow.py',H+'/ehe_dense_surface_runtime.py'],'A218/A440面积分加密；冻结C另列'),
      ('EHE实际数据重建',[H+'/ehe_dense_rerun_workflow_20261102.py',H+'/ehe_dense_rerun_runtime_20261102.py'],'严格输入传输与原MLEM200'),
      ('活动密度算子与MLEM',[H+'/torch_active_operator.py',H+'/single_checkpoint_mlem.py','compton_event_response.py','compton_sparse_ops.py'],'体积/ellipse_fraction、匹配前向转置/S和检查点'),
      ('NEMA440联合',[H+'/nema_5e10_joint_workflow_20261026.py'],'1740189只复用已完成真实输入与v5响应'),
      ('Derenzo5e10联合补算',[H+'/derenzo_joint_retry_workflow_20261102.py'],'1761195仅补440联合；不重算已成功SC'),
      ('20角Poisson',[H+'/ehe_nema_low_noise_source_20261102.py',H+'/ehe_nema_low_noise_workflow_20261102.py',H+'/ehe_nema_low_noise_runtime_20261102.py'],'匹配原生有限体积源、独立PCG64分量采样'),
      ('60角响应与Poisson',[H+'/ehe_60view_operator_20261102.py',H+'/ehe_nema_60view_workflow_20261102.py',H+'/ehe_nema_60view_runtime_20261102.py'],'同环周期响应插值、1/60曝光与新S'),
      ('NEMA评价',[H+'/nema_roi_policy.py',H+'/reconstruction_output_policy.py',H+'/mip_projection.py'],'原3D ROI、统一背景和72mm显示'),
      ('图像与诊断',['tools/research/plot_ehe_dense_completed_1761244.py','tools/research/plot_ehe_dense_nema_60view_poisson_20261102.py','tools/research/diagnose_nema_compton_rings.py'],'只读已有原始结果、图形/指标与证据'),
      ('目录与查询',['tools/project_index.py','tools/research/refresh_ac225_navigation.py'],'本地索引；不导入或提交科学任务')]
    for stage,paths,role in stages:
        present=[p for p in paths if (root/p).exists()];code_lines.append('| '+stage+' | '+'<br>'.join(md(p,root/code_doc,Path(p).name) for p in present)+' | '+role+' |')
    code_lines+=['','## 原始源码与副本','',
      '实验顶层为可维护源码；generated中的payload/releases与图集scripts为冻结或产物副本。相似文件名不代表同一SHA，正式身份按冻结release_manifest及合同解析，不用顶层最新文件替代旧发布。',
      '', '旧resume/watch/advance控制器、streaming4090的/tmp跨作业缓存方案、旧验证/验收提交链保持退休。源码索引列出它们用于追溯，不构成运行授权。',
      '', '[按职责查看全部源码](SOURCE_INDEX.md)；source_index.json同时提供顶层函数/类名和行号，支持project_index.py find检索。','']
    page(code_doc,'\n'.join(code_lines))
    page('docs/ac225/OPERATIONS.md','''# 查询与目录维护

## 日常查询

```powershell
python tools/project_index.py list --topic nema
python tools/project_index.py show derenzo-jscc-5e10-440-joint
python tools/project_index.py find sensitivity
python tools/project_index.py dirs --topic generated
python tools/project_index.py check
```

这些命令只读本地索引，不连接超算。当前队列和中断项见[状态页](CURRENT_STATUS.md)；正式请求JSON是登记身份，不能仅凭旧held/PENDING字段认定仍在运行。

## 更新说明与索引

研究的统一入口数据为[study_registry.json](study_registry.json)。新增研究时写明数据来源、矩阵版本、剂量、角度、计算/交付/视觉阶段、报告、证据和结论，再运行：

```powershell
python tools/research/refresh_ac225_navigation.py
python tools/project_index.py check
```

刷新只生成文本导航、研究说明与结果指针，不执行任何科学脚本。它检查链接和原源码SHA，并按原字节保存既有被修改的导航。`build_project_catalog.py`和`build_reconstruction_results_index.py`为兼容入口，转到本刷新器，避免恢复旧默认结果或过时状态。

研究说明页由registry自动生成。接入新的结果类型时，还需在刷新器的result_routes中适配其真实metadata和严格交付格式，再核对来源及阶段；不根据目录名称推断完成状态。

## 正式任务边界

只运行已有人工授权范围内的正式任务，不新增独立pilot/验证/验收作业。已经完成的输运、筛选、重建和严格取回不重复；失败后确认全部步骤退出、保留成功输出，仅修复缺失部分。提交不确定先查Slurm和submission_intent，不盲目重复sbatch。

旧周期任务和提交控制器不恢复。新的有界一次性衔接只按其独立授权与实际PID/create_time检查；目录整理不会重启进程。EHE Derenzo5e10目前需要补齐中断严格传输，不能直接跳到GPU重建。

冻结发布、成功检查点、数组和敏感凭据不移动、不改名、不提交到Git。/tmp是JobId隔离容器，不用于跨作业持久缓存；共享卷空闲量不等于用户目录配额。源、响应、S和求解模型变更须独立身份，不能改写旧证明。

[执行约定](../../experiments/ELLIPSE500x300_H120/AGENTS.md) · [数据管理](DATA_POLICY.md)
''')
    page('docs/ac225/DATA_POLICY.md','''# 数据版本与目录保护

## 五层数据

1. **源码和配置**：Geant4Sim、Auxiliary_Studies、实验顶层及tools。可维护版本不等于某正式作业冻结版本。
2. **冻结与小型证据**：各研究reports中的合同、发布SHA、作业登记、strict delivery及QA；原始字节保留。
3. **运行数据**：各研究generated的投影、事件、系统矩阵、事件响应、完整历史与检查点。Git忽略，原路径不动。
4. **图表产物**：Analysis_Plots/research按日期保存figures/data/scripts及artifact SHA；既有产物保持原位置与科学身份。
5. **查询层**：docs/ac225及Analysis_Plots/results/current，按条件连接前四层，只存说明和指针，不复制原始数组。

## 完成与验收

提交、正常退出、严格取回、数值QA、视觉呈现和物理认证分开记录。完整远端历史检查与本地选帧检查分开；本地48选帧不能称为本地1600完整历史QA。失败父任务中成功的SC输出单独标记，不把整个任务伪装成完成，也不重复成功阶段。

目录名、旧checkpoint或partial文件不证明该研究已完成。优先读取原collection、run_manifest、completion、会计与strict delivery；索引只是入口。原稿/旧状态按原字节保存在整理审计的before目录。

## 版本命名

新研究使用独立日期/条件ID；矩阵版本显式区分原256点与dense65536、冻结C。真实Geant4和forward+Poisson分别归类。旧511keV、FOV120、未采用Derenzo设计和停止研究留作历史参考，不覆盖现用真值或自动复活。

## Git与空间

适合Git的小型文本为源码、配置、说明、解析几何、指标与SHA。矩阵、事件响应、大数组、完整历史、归档、PNG/PDF/PPTX等媒体、编译二进制和凭据保留在本地或原计算端，不加入本轮文本交付。

本轮没有移动/删除任何科学数据或发布，没有清理partial或取消任务。空间回收应另按明确路径、成功交付和依赖检查执行；不把“旧”或“失败”当作可删除理由。工程外SSH凭据不复制进目录盘点或报告。
''')
    page('docs/ac225/DIRECTORY_MAP.md','''# 工程目录与职责

```text
工程根目录
├─ README.md                         项目入口
├─ docs/ac225/
│  ├─ INDEX.html / README.md          可搜索研究导航
│  ├─ CURRENT_STATUS.md               当前阶段与待处理项
│  ├─ CONCLUSIONS.md                  结论、证据与限制
│  ├─ CODE_GUIDE.md / code/            流程导读与按职责源码列表
│  ├─ studies/<类别>/<研究ID>.md       每个研究条件、代码、数据和证据
│  └─ *_index.json / catalog.json     可查询机器索引
├─ experiments/
│  ├─ ELLIPSE500x300_H120/            当前500×300×120mm工程
│  │  ├─ *.py / *.cu / *.json         原科学代码与工作流位置
│  │  ├─ reports/{NEMA_Body_H60,Derenzo,EHE,...}
│  │  ├─ generated/<研究ID>/          原运行数据，位置保持
│  │  └─ history/                     原说明时间线
│  └─ FOV120/                         另一代几何与历史参考
├─ Analysis_Plots/
│  ├─ results/REPORT.html             当前条件目录
│  ├─ results/current/actual/...      真实输运结果指针
│  ├─ results/current/synthetic/...   20/60角模型结果指针
│  ├─ results/history/20261029/        原EHE方法对比网页快照
│  ├─ results/{NEMA,Derenzo}/...       原6组画廊，保留字节
│  └─ research/<专题_日期>/            原图表、指标、脚本及QA
├─ Geant4Sim/                         JSCC/EHE输运实现
├─ Auxiliary_Studies/                 系统矩阵与响应辅助研究
├─ GenProj/                           极坐标源测度和历史前投影
├─ tools/                             分析、评价、图形和只读查询
├─ distributed/                       通用历史分布式入口
├─ Reproduction/ / Results/           旧本地复现与结果说明
└─ docs/organization/20261103/         修改前字节、状态查询及整理审计
```

## 为什么保留源码和数据路径

正式发布与运行数据包含硬编码相对路径、导入关系、远端路径和SHA绑定。当前整理将职责分类和结果条件体现在查询目录，不移动这些科学文件。`results/current`每个叶目录都有README与case.json，中间层也有入口；原research图集和冻结证据通过指针反查。

每个research专题的figures为图像/PDF，data为指标、来源和显示QA，scripts为当时实际绘图副本。实验reports用于合同与阶段证明，generated用于实际大数据；二者不能互换。

根部Factors、CntStat、List和Results属于历史本地接口，不能据同名目录当作当前椭圆研究输入。当前研究优先读独立generated、reports及具体contract。

[各级目录表](DIRECTORY_INDEX.md)列出每个目录职责、关联研究和可用说明；[机器索引](directory_index.json)另含本地数据标记。`python tools/project_index.py dirs --topic ehe`可检索，不递归读取大数组。
''')
    page('Analysis_Plots/results/README.md','''# 成像结果目录

[打开当前可搜索结果](REPORT.html) · [条件机器索引](data/current_results.json) · [科学结论](../../docs/ac225/CONCLUSIONS.md)

`current/actual/`按体模/总发射gamma量/角度/能量/方法查询真实Geant4结果，`current/synthetic/`查询EHE20/60视角Poisson对照。每个中间层和叶目录都有说明，叶目录case.json连接原始channel、作业、矩阵版本、图表和证据，不复制科学数组。

实际EHE优先使用1761244的65536点/面加密矩阵结果。NEMA5e9/5e10、Derenzo5e9均可查；Derenzo5e10的未交付条件列出排队/传输阶段，不冒充已完成图像。218只有SC/EHE，440才有同能量JSCC联合。

原`NEMA/...`及`Derenzo/5e9/...`六组画廊属于2026-10-29方法对比，EHE为旧矩阵版本。它们和原[data/result_catalog.json](data/result_catalog.json)、[15条原路线CSV](data/source_routes.csv)保留供追溯；[历史网页](history/20261029/REPORT.html)单独标注。最新查询使用current_results.json。

显示中心72mm MIP、完整XY、gray_r，无平滑/亮度拟合，实际迭代和色阶见各报告。完整远端历史检查、本地选帧与物理认证分别说明。原research图形及全部冻结数组保持原位置。
''')
    page('Analysis_Plots/README.md','''# 图形与结果查询

**[当前结果目录](results/REPORT.html)**按真实Geant4、模型Poisson、体模、发射量、角度、能量和SC/JSCC/EHE分类。**[研究结论](../docs/ac225/CONCLUSIONS.md)**统一说明环纹、边缘高值和模型限制。

| 内容 | 当前入口 |
|---|---|
| 20/60角、四剂量NEMA合成对比 | [60角及对照报告](research/ehe_dense_nema_60view_poisson_20261102/REPORT.html) |
| 加密EHE实际NEMA5e9/5e10与Derenzo5e9 | [1761244结果](research/ehe_dense_completed_1761244_20261102/REPORT.html) |
| NEMA5e10的440 SC与JSCC联合 | [1740189结果](research/nema_joint5e10_1740189_20261029/REPORT.html) |
| 原Compton与新旧流程诊断 | [专题目录](../docs/ac225/studies/diagnostics/README.md) |
| 体模、探头和汇报设计 | [设计目录](../docs/ac225/studies/design/README.md) |
| 按日期查看全部专题 | [research索引](research/README.md) |

`results/current`是查询指针；`research/<专题_日期>`保存原figures/data/scripts及来源SHA。原6组方法比较画廊保留在results/NEMA和results/Derenzo，其EHE为旧矩阵，已从当前首选结果中分开。

SC＝单光子，JSCC＝同能量SC+Compton联合，纯Compton单列。真实Geant4与forward+Poisson不混作同一实验；不做218/440跨能量图像求和。中心72mm、完整XY、无平滑/亮度拟合按各报告明确。
''')
    research_doc='Analysis_Plots/research/README.md'
    text=['# 按日期保留的图形与诊断专题','',
      '当前版本和条件从[结果目录](../results/REPORT.html)查询；这里保留专题原路径。figures为显示产物，data为来源/指标/QA，scripts为实际绘图副本。','',
      '| 专题 | 用途与结论 | 当前地位 |','|---|---|---|']
    covered=set()
    for s in studies:
        reports=[p for p in s['entries']+s['figures'] if p.startswith('Analysis_Plots/research/')]
        for p in reports:
            parent=str(Path(p).parent)
            if parent in covered:continue
            covered.add(parent);text.append(f"| {md(p,root/research_doc,Path(p).parent.name)} | {s['conclusion']} | {s['selection']} |")
    for p in sorted((root/'Analysis_Plots/research').iterdir()):
        if p.is_dir() and str(p.relative_to(root)) not in covered:
            text.append(f'| {md(str(p.relative_to(root)),root/research_doc,p.name)} | 按原目录与日期说明阅读 | 日期限定专题 |')
    text+=['','未采用的Derenzo设计不作为输运真值。旧BV–CNR中的EHE属于加密前矩阵；新模型BV–CNR只按实际选帧计算。历史图形存在不代表当前物理认证。','']
    page(research_doc,'\n'.join(text))
    page('experiments/README.md','''# 实验工程

| 目录 | 职责 |
|---|---|
| [ELLIPSE500x300_H120](ELLIPSE500x300_H120/README.md) | 当前218/440keV三维主工程，500×300×120mm；NEMA、Derenzo、EHE及算法诊断 |
| [FOV120](FOV120/README.md) | 另一代几何和历史工程，不能与当前源/矩阵混用 |

各实验顶层保存代码与配置，reports保存合同/冻结/登记和验收，generated保存大数组及运行历史。当前统一查询见[225Ac导航](../docs/ac225/README.md)和[代码路线](../docs/ac225/CODE_GUIDE.md)。
''')
    page(H+'/README.md','''# ELLIPSE500x300_H120成像工程

当前研究覆盖NEMA、独立单能Derenzo、JSCC单光子/同能量联合、EHE真实输运及模型合成对照。完整FOV500×300×120mm、132040全点/78920活动单元，真实输运主要20角；60角只属于独立EHEPoisson对照。

| 查什么 | 入口 |
|---|---|
| 当前计算与待处理项 | [状态](../../docs/ac225/CURRENT_STATUS.md) |
| 图像与研究结论 | [结果](../../Analysis_Plots/results/REPORT.html) · [结论](../../docs/ac225/CONCLUSIONS.md) |
| 每个研究的条件/代码/数据 | [分类研究页](../../docs/ac225/studies/README.md) |
| 科学源码路线 | [CODE_GUIDE](../../docs/ac225/CODE_GUIDE.md) |
| 小型合同与证据 | [reports入口](reports/README.md) |
| 命名、剂量、体积及3D ROI | [科学口径](../../docs/ac225/SCIENCE_CONVENTIONS.md) |
| 操作约束 | [AGENTS](AGENTS.md) |

顶层代码、generated运行目录和远端发布路径保持稳定。图表中的JSCC只指同能量SC+Compton；纯Compton单列，218/440不求和。所有发射预算按合同定义，不当作225Ac衰变数。

旧README和完整人工授权时间线在[本次整理前备份](../../docs/organization/20261103/before/experiments/ELLIPSE500x300_H120/README.md)及[历史目录](history)；旧排队描述只作日期证据。当前源和矩阵身份始终由研究合同/冻结决定。
''')
    page(H+'/AGENTS.md','''# 当前工程执行约定

## 当前查询与授权身份

主导航见[225Ac总入口](../../docs/ac225/README.md)、[当前状态](../../docs/ac225/CURRENT_STATUS.md)和[分类研究](../../docs/ac225/studies/README.md)。本次目录整理只允许说明、索引及只读状态核对，不新增科学计算或取消任务。

全部原人工授权、撤销与冻结身份按原字节保存在[整理前AGENTS](../../docs/organization/20261103/before/experiments/ELLIPSE500x300_H120/AGENTS.md)。未撤销的授权仍以对应合同和最新人工指令为准；查询索引本身不是运行授权。

## 计算与任务控制

- 只提交获授权的正式任务，不单独创建pilot/validation/验收作业；不恢复已停止定时器、resume/watch/advance旧控制器。
- 已成功输运、收集、筛选、重建、严格取回不重复。失败后确认所有步骤退出，保留原发布与成功输出，仅有界补缺。未解决submission_intent先查Slurm，不盲重投。
- 实际PID须同时核对create_time和cmdline；旧登记不代表存活。本工程一次性进程42492/60468/66928/80600均已退出，不复活。
- 已完成NEMA三路1722760、联合1740189、旧Compton适配1739632、Derenzo5e9两路1722673及EHE/218SC等组保持只读。旧16卡1739454、旧失败4090/8×2请求和已取消验证/依赖作业不恢复。
- Derenzo实际5e10：218SC1752710已交付；原1752461整体失败但440SC10000/200已保存，不重算成功SC。唯一联合补算1761195（8×8张5090）仍PENDING/Priority，只补同440联合。
- EHE Derenzo5e10两能量实际输运/收集完成，60468因中断strict fetch partial退出，GPU后续未登记；先核对并有界补齐严格传输，不重复模拟、不跳过输入验收。
- 普通GPU资源按实际AllocTRES、RSS和显存核对，保留原80%资源守卫和自动内存策略；不假定卷空闲量等于用户配额，不写满共享盘。/tmp为JobId隔离容器，不作跨作业持久缓存。

## 科学约束

- 完整FOV500×300×120mm、132040/78920网格、原事件权重/v5核/匹配自身S/原MLEM/全1密度初值/无正则按各冻结合同保持。科学模型或求解改动须独立版本，不覆盖旧证明。
- SC＝单光子；JSCC结果＝同能量SC+Compton联合。纯Compton不能标JSCC，历史探头前缀不改变通道身份。禁止新的218/440跨能量求和图；本轮218无联合Compton结果。
- 实际NEMA预算为合同定义的双能gamma初级总数，Derenzo按探头和能量独立单能发射；Poisson为总期望gamma，不是225Ac衰变次数或完整衰变链。
- 当前EHE实际A218/A440为65536点/可见面，原C冻结，物理HOLD保留；自身S闭合不代表物理标定通过。NEMA218固定背景只来自本次/同剂量最终440单光子图；单能Derenzo背景零。
- EHE20角1762238和60角1762460合成组均已完成，各8路1600帧远端核对、48选帧严格本地交付。完整本地历史QA仍false，不重新fetch或提交。60角是同环周期线性响应插值、1/60曝光及匹配新S，不是新真实输运，不是纯角度因果或分辨率认证。
- 当前Derenzo唯一几何为20261015_equal_regions：218棒径6/8/10/12/14/16、114根；440为10/12/14/16/18/20、53根。复用原径向6mm矩阵，不能据此宣布6mm分辨率。
- NEMA真值用原truth_3mm.npz与manifest原3D球心；球ROI取中心距球面至少1.5mm的全部体素等权，统一背景为完整内部、排除六球/肺/接触边界。无平滑、无亮度拟合；72mm只是显示MIP，完整XY/120mm响应支持保持。
- JSCC10000/save50与EHE200/save1分别展示，不把同迭代当成同收敛。计算、严格交付、数值/视觉QA和物理/图质认证分别记录。

## 目录与Git

原源码、冻结payload/releases、矩阵、事件、成功检查点、完整历史和研究artifact保留原路径与字节。分层导航用说明及指针，不复制大数组。目录移动/删除不得破坏硬编码、活跃作业、依赖或SHA。

Git只保存审核的小型文本源码/配置/几何/指标/证据SHA与说明，排除矩阵、事件响应、大数组、归档、完整历史、PNG/PDF/PPTX等媒体、二进制和敏感凭据。保留已有工作区改动；目录整理不自动commit/push。
''')
    report_doc=H+'/reports/README.md'
    page(report_doc,'# 实验证据目录\n\n合同、冻结、作业登记、严格交付和科学/图形QA按研究保留，不能用目录存在或旧状态字段代替实际完成证据。\n\n'+links(report_doc,[
      ('docs/ac225/CURRENT_STATUS.md','当前阶段'),('docs/ac225/studies/nema/README.md','NEMA研究说明'),
      ('docs/ac225/studies/derenzo/README.md','Derenzo研究说明'),('docs/ac225/studies/ehe/README.md','EHE矩阵说明'),
      ('docs/ac225/studies/synthetic/README.md','模型Poisson组'),('docs/ac225/studies/diagnostics/README.md','物理/算法诊断'),
      (H+'/reports/NEMA_Body_H60/README.md','NEMA原真值与体模说明'),(H+'/reports/Derenzo/README.md','Derenzo证据入口'),(H+'/reports/EHE/README.md','EHE证据入口')])+'''

研究reports与generated一一对应但职责不同：前者是小型科学身份和阶段证明，后者为实际数组/归档/日志。原研究README及证明不搬移；当前阶段与结论在统一研究页补齐，旧日期状态留作历史。
''')
    for cat,ids in [('EHE',['ehe-dense-matrix','nema-ehe-dense-5e9','nema-ehe-dense-5e10','derenzo-ehe-dense-5e9','derenzo-ehe-dense-5e10','ehe-nema-poisson-20view','ehe-nema-poisson-60view']),
                    ('Derenzo',['derenzo-design-selected','derenzo-jscc-218','derenzo-jscc-440','derenzo-ehe-dense-5e9','derenzo-jscc-5e10-218','derenzo-jscc-5e10-440-sc','derenzo-jscc-5e10-440-joint','derenzo-ehe-dense-5e10'])]:
        doc=H+'/reports/'+cat+'/README.md';page(doc,'# '+cat+'研究证据\n\n'+table(doc,ids)+'\n原报告/冻结位置不变。实际数据与合成投影分开；计算、交付、图形与物理结论分别阅读。\n')
    page('tools/research/README.md','''# 研究分析与目录工具

本目录保存只读分析、绘图、严格取回和一次性衔接脚本。文件名相似时按研究TAG、原metadata与SHA确认，不据文件名猜测生产版本。

| 类别 | 入口 |
|---|---|
| 当前代码路线 | [CODE_GUIDE](../../docs/ac225/CODE_GUIDE.md) |
| 分析与图形 | [职责列表](../../docs/ac225/code/analysis-figures/README.md) |
| 执行/衔接脚本 | [职责列表](../../docs/ac225/code/execution-control/README.md)；历史控制器不恢复 |
| 目录刷新 | [refresh_ac225_navigation.py](refresh_ac225_navigation.py)，只更新文本/索引，不调用科学任务 |
| 已完成图形 | [结果入口](../../Analysis_Plots/results/REPORT.html) |

图集scripts为当时绘图副本，不覆盖这里的可维护源码；源码变化不追认旧artifact身份。取回和one-shot脚本有外部动作，阅读索引不代表授权运行。旧build_project_catalog/build_reconstruction_results_index兼容到当前目录刷新器，避免恢复过时默认状态。
''')
    page('tools/README.md','''# 工程工具

| 目录或命令 | 用途 |
|---|---|
| [project_index.py](project_index.py) | 本地研究/源码/目录查询，绝不导入运行器 |
| [research](research/README.md) | 专题分析、绘图、严格取回及目录刷新 |
| [evaluation](evaluation) | 指标和评价工具 |
| [visualization](visualization) | 通用显示工具 |
| [diagnostics](diagnostics) | 诊断 |
| [baseline](baseline) | 固定回归及身份检查 |
| [runners](runners) | 历史执行适配，按原研究说明使用 |
| [security](security) | 小型文本和敏感信息审计 |
| [legacy](legacy) | 旧接口参考 |

当前225Ac路线见[代码导读](../docs/ac225/CODE_GUIDE.md)。目录查询本身不会提交任务；原始研究的冻结、输入、权限与数据来源不可由工具名推断。
''')
    # Preserve component-specific instructions below the new parent overview.
    for path,title,intro in [('Geant4Sim/README.md','Geant4输运工程',
        '[JSCC实现](Geant4Code/README.md)与[EHE实现](Geant4Code_EHE/README.md)使用各自探头/宏和源身份；当前研究的gamma预算、种子和原二进制由合同绑定。'),
        ('Auxiliary_Studies/README.md','系统矩阵与辅助研究',
        '本目录包括CUDA系统矩阵、因子转换、Compton灵敏度、事件顺序及其他独立研究。相似程序和历史参数不直接等同于当前生产矩阵。')]:
        doc=root/path;arch=root/A/'before'/path
        original=arch.read_text('utf8') if arch.exists() else doc.read_text('utf8') if doc.exists() else ''
        body='# '+title+'\n\n'+intro+'\n\n'+md('docs/ac225/CODE_GUIDE.md',doc,'当前225Ac代码路线')+' · '+md('docs/ac225/DIRECTORY_MAP.md',doc,'目录职责')+'\n'
        if original:body+='\n## 原组件说明\n\n'+original
        page(path,body)
    page(A+'/README.md','''# 2026年11月3日工程整理记录

本次建立统一225Ac研究页、可搜索工程/结果网页、按条件的结果指针，以及当前状态、结论、代码路线和职责索引。旧EHE方法画廊、加密真实结果、20/60角Poisson和未交付任务分别登记。

## 原始内容与身份

- [整理前导航及说明原字节](before)：仅归档被修改的导航文件，不移动原科学文件。
- [baseline.json](baseline.json)：1590项原科学/执行/分析源码与宏SHA、Git HEAD及原已有改动清单。
- [Git工作区原状态](git_status_before.txt)：保留整理前未提交变更，不把它们当成本轮新修改。
- [GPU只读状态](gpu_readonly_status.json)与[CPU只读状态](cpu_readonly_status.json)：1761195排队、EHE两路MC/收集完成。
- [目录与链接验证](link_validation.json)、[整理清单](organization_manifest.json)：刷新器生成的链接/原源码身份验证和文本变更身份。

## 边界与后续

本轮不提交/取消科学作业，不恢复进程，不重取结果，不移动/删除矩阵、事件、数组、归档或冻结发布。只将当前阶段、来源和结论放入分层查询目录。中断partial及成功SC输出保留，待原任务的有界修复流程处理。

新研究维护study_registry.json后运行refresh_ac225_navigation.py。刷新器不导入运行器，不联网；本次状态查询原证据不会被刷新伪装成新的实时状态。未进行Git提交或推送。
''')
