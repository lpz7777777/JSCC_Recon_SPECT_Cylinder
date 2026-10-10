"""Repackage accepted NEMA results without running transport or reconstruction.

data: read immutable histories and bound CSVs, export curated scientific figures.
pdf: typeset the review with ReportLab using those figures and cleanup receipts.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'experiments/ELLIPSE500x300_H120'
R=H/'reports/NEMA_Body_H60'
OUT=H/'reports/dual_energy_review_20261010'
LATEST=R/'ehe_forward_poisson_5e10_200/comparison'
SYS=('EHE','EHE matrix+Poisson','EHE Geant4 5e10','EHE matrix+Poisson 5e10','JSCC')
LABELS=dict(zip(SYS,('EHE G4 5e9','EHE Poisson 5e9','EHE G4 5e10','EHE Poisson 5e10','JSCC 5e9')))
CN=dict(zip(SYS,('EHE Geant4 5e9','EHE 前投影加噪声 5e9','EHE Geant4 5e10','EHE 前投影加噪声 5e10','JSCC 5e9')))
CH=('218_SinglePhoton_CrossTalkCorrected','440_SinglePhoton','440SinglePlus218Single')
JCH=CH+('440_ComptonOnly','440_SinglePlusCompton','440SingleComptonPlus218Single')
CL=dict(zip(JCH,('218 校正单光子','440 单光子','双能单光子和','440 SC Compton','440 JSCC','JSCC 440 与 218 和')))
SHORT=dict(zip(JCH,('218 SC','440 SC','SC sum','440 Compton','440 JSCC','JSCC sum')))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def rows(p):
 with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def energy(c):return '218' if c==CH[0] else 'sum' if 'Plus218' in c else '440'
def data():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 sys.path.insert(0,str(H))
 from analyze_nema_result import xy_interpolator
 from mip_projection import axial_mip
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
 manifest=read(LATEST/'artifact_manifest.json')['files'];bindings={}
 for name in ('reference_native_metrics.csv','reference_sphere_metrics.csv','synthetic_native_metrics.csv','synthetic_sphere_metrics.csv','comparison_report.json'):
  assert sha(LATEST/name)==manifest[name],name
  bindings[str((LATEST/name).relative_to(ROOT))]=manifest[name]
 native=rows(LATEST/'reference_native_metrics.csv')+rows(LATEST/'synthetic_native_metrics.csv')
 spheres=rows(LATEST/'reference_sphere_metrics.csv')+rows(LATEST/'synthetic_sphere_metrics.csv')
 routes=[(s,c) for s in SYS for c in (JCH if s=='JSCC' else CH)]
 endpoint=[];peaks=[]
 for s,c in routes:
  rr=sorted([r for r in native if (r['system'],r['channel'])==(s,c)],key=lambda r:int(r['iteration']))
  assert [int(r['iteration']) for r in rr]==list(range(0,10001,50) if s=='JSCC' else range(0,201,10))
  endpoint.append(rr[-1])
  for d in sorted({int(r['diameter_mm']) for r in spheres if (r['system'],r['channel'])==(s,c)}):
   ss=[r for r in spheres if (r['system'],r['channel'],int(r['diameter_mm']))==(s,c,d) and r['cnr']!='']
   best=max(ss,key=lambda r:float(r['cnr']));last=max(ss,key=lambda r:int(r['iteration']))
   peaks.append(dict(system=s,channel=c,diameter_mm=d,peak_iteration=int(best['iteration']),peak_cnr=float(best['cnr']),last_iteration=int(last['iteration']),last_cnr=float(last['cnr']),last_crc=float(last['crc'])))
 for name,values in [('endpoint_native_metrics.csv',endpoint),('sphere_cnr_peak_and_final.csv',peaks)]:
  keys=list(values[0]);keys=[k for k in keys if all(k in r for r in values)]
  with (OUT/name).open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(values)
 fig,axs=plt.subplots(2,3,figsize=(11,6.1),layout='constrained')
 for k,c in enumerate(CH):
  for s in SYS[:-1]:
   rr=sorted([r for r in native if (r['system'],r['channel'])==(s,c)],key=lambda r:int(r['iteration']))
   for i,key in enumerate(('background_cv','integral_recovery')):
    axs[i,k].plot([int(r['iteration']) for r in rr],[float(r[key]) for r in rr],label=LABELS[s])
  axs[0,k].set_title(SHORT[c]);axs[0,k].set_ylabel('Background CV');axs[1,k].set_ylabel('Integral / emitted dose')
  for ax in axs[:,k]:ax.set_xlabel('EHE actual iterations');ax.grid(alpha=.2);ax.set_xlim(0,200)
 axs[0,0].legend(fontsize=7);fig.savefig(OUT/'figures/ehe_dose_noise.png',dpi=200);plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(11,6.1),layout='constrained')
 for i,key in enumerate(('cnr','crc')):
  for j,(c,d) in enumerate(((CH[0],28),(CH[1],37),('440_SinglePlusCompton',37))):
   systems=SYS[:-1] if j<2 else ('JSCC',)
   channels=(c,) if j<2 else (CH[1],'440_SinglePlusCompton','440_ComptonOnly')
   for s in systems:
    for cc in channels:
     rr=sorted([r for r in spheres if (r['system'],r['channel'],int(r['diameter_mm']))==(s,cc,d)],key=lambda r:int(r['iteration']))
     axs[i,j].plot([int(r['iteration']) for r in rr],[float(r[key]) if r[key]!='' else np.nan for r in rr],label=LABELS[s] if j<2 else SHORT[cc])
   axs[i,j].set(title=f'{d} mm / '+('EHE '+SHORT[c] if j<2 else 'JSCC 440 channels'),xlabel='Actual iterations',ylabel=key.upper(),xlim=(0,200 if j<2 else 10000));axs[i,j].grid(alpha=.2)
 for ax in axs[0]:ax.legend(fontsize=7)
 fig.savefig(OUT/'figures/large_sphere_trajectories.png',dpi=200);plt.close(fig)
 truthp=H/'generated/NEMA_Body_H60/truth_3mm.npz';meta=read(R/'manifest.json');assert sha(truthp)==meta['truth_sha256']
 truth=np.load(truthp);geometry=H/'generated/ehe_spect_5e9_200/payload/whole_geometry.npz'
 assert sha(geometry)=='8e7a5f54e59b7bd7ce0db723a42d71b131e8c840f22982d541deb5e77df72aa4'
 g=np.load(geometry);coords=g['coordinates_mm'];active=g['active_indices'];assert len(active)==78920
 x,y,z=[truth[k+'_mm'] for k in 'xyz'];interp=xy_interpolator(coords[:3301,:2],x,y)
 config=read(LATEST/'comparison_report.json');scales=config['density_scales'];histbindings={}
 datanames={'EHE':'ehe_spect_5e9_200','EHE matrix+Poisson':'ehe_forward_poisson_5e9_200','EHE Geant4 5e10':'ehe_spect_5e10_200','EHE matrix+Poisson 5e10':'ehe_forward_poisson_5e10_200'}
 for s in SYS:
  cs=JCH if s=='JSCC' else CH
  groups=(cs[:3],cs[3:]) if s=='JSCC' else (cs,)
  for groupidx,group in enumerate(groups):
   nodes=(0,2000,5000,10000) if s=='JSCC' else (0,50,100,200)
   fig,axs=plt.subplots(3,5,figsize=(11,6),squeeze=False)
   fig.subplots_adjust(left=.12,right=.92,top=.90,bottom=.08,wspace=.08,hspace=.22)
   for row,c in enumerate(group):
    e=energy(c);source=truth['activity_'+e+'_zyx'] if e!='sum' else (scales[s]['218']*truth['activity_218_zyx']+scales[s]['440']*truth['activity_440_zyx'])/scales[s]['sum']
    if s=='JSCC':
     folder=H/'generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy';proof=read(folder/'verification.json');expected=next(a['sha256']['history'] for a in proof['outputs'] if a['channel']==c);frames=200;step=50
    else:
     folder=H/'generated'/datanames[s]/'results/formal';proof=read(R/datanames[s]/'formal_summary.json');expected=proof['files'][f'Image_{c}_history.float32'];frames=20;step=10
    path=folder/f'Image_{c}_history.float32';assert path.stat().st_size==frames*78920*4 and sha(path)==expected
    histbindings[str(path.relative_to(ROOT))]=expected;hist=np.memmap(path,'<f4','r',shape=(frames,78920));images=[source]
    for n in nodes:
     a=np.full(78920,2 if e=='sum' else 1,dtype='<f4') if n==0 else hist[n//step-1]
     assert np.isfinite(a).all() and (a>=0).all()
     full=np.zeros(132040,dtype='<f4');full[active]=a;images.append(interp(full.reshape(40,3301))/scales[s][e])
    for col,im in enumerate(images):
     ax=axs[row,col];handle=ax.imshow(axial_mip(im,z,8),origin='lower',extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5),cmap='gray_r',vmin=0,vmax=10,interpolation='nearest',aspect='equal')
     ax.set_title('H60 3D truth' if col==0 else f'Iteration {nodes[col-1]}',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
    bbox=axs[row,0].get_position();fig.text(.008,(bbox.y0+bbox.y1)/2,SHORT[c],fontsize=9,va='center')
   cb=fig.add_axes([.945,.22,.015,.48]);fig.colorbar(handle,cax=cb,label='Density / emitted background (0-10)')
   fig.suptitle(LABELS[s]+' | full XY field, central 72 mm MIP\nNo smoothing, no fitted gain; own iteration budget',fontsize=11)
   fig.savefig(OUT/f'figures/gallery_{SYS.index(s)}_{groupidx}.png',dpi=200);plt.close(fig)
 save(OUT/'scientific_sources.json',dict(passed=True,csv_sha256=bindings,histories_sha256=histbindings,truth_sha256=sha(truthp),geometry_sha256=sha(geometry),reconstruction_run=False,transport_run=False,endpoint_rows=len(endpoint),sphere_rows=len(peaks),own_iteration_ranges=True,crop=0,smoothing=0,fitted_gain=False,display_range=[0,10],mip_z_mm=[-36,36],native_metrics_z_mm=[-60,60],figures={p.name:sha(p) for p in (OUT/'figures').glob('*.png')}))
 print('Bound CSVs, complete trajectories, 18 histories, and 8 curated figures exported')

def pdf():
 from reportlab.pdfgen import canvas
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
 from reportlab.lib.styles import ParagraphStyle
 from reportlab.lib import colors
 from reportlab.pdfbase import pdfmetrics
 from reportlab.pdfbase.ttfonts import TTFont
 from reportlab.lib.enums import TA_LEFT,TA_CENTER
 from PIL import Image as PILImage
 pdfmetrics.registerFont(TTFont('ZH','C:/Windows/Fonts/simhei.ttf'))
 width,height=595.2756,841.8898; content=491.2756
 style=ParagraphStyle('body',fontName='ZH',fontSize=10.5,leading=16,spaceAfter=9,wordWrap='CJK',textColor=colors.HexColor('#111111'))
 title=ParagraphStyle('title',parent=style,fontSize=23,leading=31,spaceAfter=17)
 heading=ParagraphStyle('head',parent=style,fontSize=16,leading=22,spaceAfter=13)
 caption=ParagraphStyle('caption',parent=style,fontSize=9,leading=13,textColor=colors.HexColor('#444444'),spaceAfter=10)
 cell=ParagraphStyle('cell',parent=style,fontSize=8.3,leading=12,spaceAfter=0)
 story=[]
 def p(text):story.append(Paragraph(text,style))
 def h(text):story.append(Paragraph(text,heading))
 def page(text):
  if story:story.append(PageBreak())
  h(text)
 def table(headers,rr,widths=None):
  data=[[Paragraph(str(v),cell) for v in r] for r in [headers]+rr]
  t=Table(data,colWidths=widths or [content/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF2')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,12)])
 def figure(name,captiontext):
  path=OUT/'figures'/name
  with PILImage.open(path) as im:iw,ih=im.size
  story.append(Image(str(path),width=content,height=content*ih/iw));story.append(Spacer(1,7));story.append(Paragraph(captiontext,caption))
 source=read(OUT/'scientific_sources.json');assert source['passed']
 ends=rows(OUT/'endpoint_native_metrics.csv');peaks=rows(OUT/'sphere_cnr_peak_and_final.csv')
 story.append(Paragraph('218 与 440 keV 双能重建研究报告',title))
 p('NEMA H60 三维球体模近期结果整理　2026年10月10日')
 p('本报告汇总近期18条已完成的成像路线：JSCC 5e9的六路10000次重建，EHE Geant4 5e9与5e10各三路200次，以及EHE系统矩阵前投影加独立Poisson噪声在两种剂量下各三路200次。重点讨论图像、球体对比恢复、噪声、源外泄漏和完整迭代轨迹。')
 p('主要发现是：EHE剂量从5e9增至5e10后背景波动显著降低，但200次并不总是最佳CNR时点，小球仍较弱；JSCC延长至10000次改善部分大球恢复和轴向泄漏，同时放大背景噪声与尖峰。响应模型自生数据与独立Geant4数据有不同的检验意义，不能合并为一次物理校准。')
 table(['组别','数据含义','末迭代与保存','成像路线'],[
 ['JSCC 5e9','独立Geant4 legacy观测','10000 / 每50保存','六路'],
 ['EHE G4 5e9','实际输运5e9 gamma','200 / 每10保存','218、440、双能和'],
 ['EHE Poisson 5e9','期望发射5e9的模型均值','200 / 每10保存','218、440、双能和'],
 ['EHE G4 5e10','实际输运5e10 gamma','200 / 每10保存','218、440、双能和'],
 ['EHE Poisson 5e10','期望发射5e10的模型均值','200 / 每10保存','218、440、双能和']], [103,150,136,102])
 p('EHE使用0至200、JSCC使用0至10000各自横轴。本报告不把相同迭代次数或相同图列当成相同收敛程度，也不把不同探测器的相同发射预算当成相同探测计数。')
 p('本轮只整理既有数据和只读图表，没有新增模拟、响应或重建。定时任务保持暂停。')
 page('研究对象和共同测量方法')
 p('真值采用工程现有truth_3mm.npz及manifest.json登记的三维球ROI。体模主体为300×230×60 mm，位于500×300×120 mm成像域；3 mm体素表示六个直径10、13、17、22、28、37 mm的球。218 keV热球为10、17、28 mm，440 keV热球为13、22、37 mm。热球为自身能量背景的10倍，异能球在该能量中为零。源为真空代理，未包含人体衰减和散射。本研究的60 mm高度、双能填充和球位是工程适配，不构成完整NEMA标准性能试验。')
 table(['项目','共同约定或系统区别'],[
 ['密度和支持域','gamma/mm3；132040圆网格点、78920完整活动单元；20视角'],
 ['EHE探测器','1250孔、2312 NaI bin、272×136 mm探测面；孔径2.5、隔厚3.4、孔长50.5 mm'],
 ['EHE几何','前表面298.5 mm；源中心Y=-345 mm；Params原点323.75 mm'],
 ['源份额','0.114 / 0.259乘真实两能积分；218期望份额0.2938077987'],
 ['两能源积分','218为3218760.703125、440为3405284.296875 mm3'],
 ['Geant4发射','近期EHE实际宏均全4π；一事件一gamma；beamOn剂量倍数1'],
 ['显示','低白高黑gray_r；源密度背景尺度固定0至10；crop0；无平滑、无拟合亮度'],
 ['空间统计','完整120 mm域；图示MIP为中央72 mm，不能替代完整域指标']], [100,391])
 p('CRC=(球ROI均值/局部背景均值-1)/(真实热球/背景比-1)；CNR=(球ROI均值-局部背景均值)/局部背景标准差。球均值使用现有三维分数体积ROI，背景统计沿用原验收方法；背景CV为标准差/均值。0次为全1初始化，双能和为2，均匀背景使0次CNR和唯一峰位置未定义。')
 p('组合图是同迭代的两能gamma密度相加，组合CRC真值按实际两能背景gamma密度加权。它不是Ac225母核活度。')
 page('观测计数 灵敏度和串扰数据')
 table(['数据组','218窗','440窗','218窗串扰占比'],[
 ['EHE G4 5e9','22929','12171','43.0416% 实测标签'],
 ['EHE Poisson 5e9','20969','11921','35.7003% 模型样本'],
 ['EHE G4 5e10','232888','118846','43.0791% 实测标签'],
 ['EHE Poisson 5e10','210419','119538','35.6204% 模型样本'],
 ['JSCC G4 5e9','12314473','5363190','旧数据缺少初级窗标签']], [146,96,96,153])
 p('EHE G4 5e9的218窗为13060直接事件加9869个440初级串窗计数；5e10为132562加100326。新Poisson 5e10直接218、440和串窗分量分别135467、119538、74952，来自三组独立PCG64种子34100101至34100103。5e10指发射预算，不是探测计数。')
 table(['响应平均S/体积','EHE','JSCC'],[['A218','7.42311889e-6','0.00618843405'],['A440','2.74110204e-6','0.00164926017'],['C440至218','1.701798e-6','0.00117600343']],[170,160,161])
 p('上表是各系统完整矩阵自身S的模型统计，包含探测器结构、材料、覆盖和响应模型的差异，不能直接解释成真实设备效率比。EHE两个能窗的计数远低于JSCC，这使同发射剂量的图像比较包含很大的统计差异。')
 p('EHE逐20视角源活动的正交投影落在探测面外比例最大为218的3.00459%、440的2.84002%。这是几何覆盖比例，不是实测光子拒绝概率。')
 p('原EHE真实源前投影物理审计仍记录C440至218低估24.0229%，预测7498.176418对观测9869；19视角和全局触发HOLD，另A218两个视角HOLD。全部逐bin统计不足100，保持UNDETERMINED。按用户后续要求继续原方法完成重建，原证据保留；执行验收通过没有改变该物理偏差。')
 page('近期方法演变及已经取得的证据')
 p('密度基底修复了极坐标每点代表体积差异造成的中心偏差。用B=A diag(ΔV)对gamma密度建模后，旧1e10案例双能背景中心/中环比由0.607升至1.010；这说明源测度重要，但投影形状残差仅小幅改善，未解决全部响应误差。')
 p('分数边界中的极小体积单元会放大密度尖峰，但完整单元中也存在尖峰。绑定方案使峰迁移、噪声与CRC几乎不变；弱Huber压峰和CV，却使440 JSCC 37 mm CRC从约0.438降至0.086。当前正式对照保留完整单元和无正则化MLEM，不将未完成的中强Huber或TV当成结果。')
 p('更严格的首散射事件定义改善部分JSCC峰值，但SC Compton没有达到既定尖峰下降目标，并造成部分大球CRC损失。ideal1e9与legacy5e9的事件策略不同，不能解释成单纯剂量差。stable_float64消除了近共线几何的少量筛选翻转，却不足以解释主要尖峰。')
 p('连续材料能量核v5在legacy独立校准144个充分空间区把RMS从3.10597%降至1.13606%，椭圆总效率偏差从-1.74664%降至-0.31139%。504个联合类别中446仍未判定。1667869两核2000次对照中Compton与JSCC最大密度分别下降76.63%和62.05%，但背景CV只小幅改善，小球仍较弱。')
 p('1669255采用连续核六路10000次。其2000次与1667869连续核组逐值及SHA一致，证实保存循环和长程入口未改变算法。本轮没有角度核10000次配对。原1e9与最初5e9 NEMA、q筛选、首散射、绑定/Huber、精细场和积分诊断均保留历史证据，详见文末目录。')
 p('EHE三套响应的12个计算块已完成，后续只恢复存储转换，未重复PE或Scatter计算。源盒平均、晶体对筛选、固体角和有限孔路径等只读诊断排除或限定了部分解释，但没有确认24%偏差的实际根因；现有模拟缺少路径分辨标签，不能给出缺失机制的实测份额。')
 page('EHE剂量对背景噪声和积分恢复的影响')
 figure('ehe_dose_noise.png','图1　四组EHE完整保存轨迹。上排为背景CV，下排为重建积分/实际或期望发射预算；两者都是完整120 mm域指标。')
 p('200次时，前投影Poisson剂量增加10倍后218与440背景CV由2.789/2.852降至0.869/0.824，描述性下降约69%/71%；实际Geant4对应2.756/2.744降至0.827/0.821。不同剂量采用不同独立噪声或输运实现，这不是多次重复试验的统计结论。')
 p('5e10的双能和CV约0.60，明显好于EHE 5e9的约2.0。实际G4 5e10的218积分恢复1.2503，高于模型Poisson的1.0658；这提示增加剂量没有消除模型与独立输运的偏差，不能靠匹配计数或图像亮度隐藏。')
 page('CNR和CRC随各自迭代范围变化')
 figure('large_sphere_trajectories.png','图2　218的28 mm、440的37 mm大球完整轨迹；右列JSCC单光子、Compton和联合路线使用0至10000的独立横轴，左两列EHE使用0至200。0次CNR未定义。')
 p('新Poisson 5e10的单能热球CNR峰值在20至100次：218的10/17/28 mm分别在20/60/60次，440的13/22/37 mm在70/100/50次。实际G4 5e10的单能峰值主要在50至60次。200次常有更高对比恢复，但背景标准差也上升；不应仅凭CRC判断质量。')
 p('JSCC 2000至10000的部分大球CRC及轴向泄漏改善，同时所有路线最大密度和背景CV上升。440 Compton CV由0.717升至2.500，峰/背景由19.05升至99.72；13 mm CRC损失约8.11个百分点。两种双能和10 mm CRC也损失超过5个百分点。')
 p('峰值时点只是保存帧中单次实现的描述，不能作为经独立验证的最佳停止策略。不同噪声实现有球间波动；不能择优重抽数据或把某一迭代当系统已收敛的统一证据。')
 page('各路线末帧的完整域噪声和恢复')
 table(['数据组与路线','末迭代','背景CV','积分恢复','轴外泄漏%'],[[CN[r['system']]+' / '+CL[r['channel']],r['iteration'],f"{float(r['background_cv']):.3f}",f"{float(r['integral_recovery']):.4f}",f"{100*float(r['source_z_leakage']):.3f}"] for r in ends],[215,56,70,76,74])
 p('末帧表展示各自预算终点。接近单位积分不保证空间分布准确；高CV和峰值可能与较大的CRC同时出现。EHE与JSCC没有同计数、同材料、同覆盖的控制条件，不能据此作纯算法排名。')
 for group,titletext in [(SYS[:-1],'EHE末帧球体对比恢复与可见性'),(('JSCC',),'JSCC末帧球体对比恢复与可见性')]:
  page(titletext)
  rr=[]
  for s,c in routes_for(group):
   vals=[]
   for d in (10,13,17,22,28,37):
    m=next((r for r in peaks if (r['system'],r['channel'],int(r['diameter_mm']))==(s,c,d)),None)
    vals.append('不适用' if m is None else f"{float(m['last_cnr']):.2f}<br/>{100*float(m['last_crc']):.1f}%")
   rr.append([CN[s]+'<br/>'+CL[c]]+vals)
  table(['路线','10 mm','13 mm','17 mm','22 mm','28 mm','37 mm'],rr,[167]+[54]*6)
  p('每格第一行为CNR，第二行为CRC百分数；单能非对应热球标为不适用。EHE末帧200，JSCC末帧10000。组合图CRC使用两能背景密度加权真值。')
  if group==('JSCC',):
   p('JSCC 440联合末帧对22 mm的CNR为6.404，优于440单光子的5.507；37 mm相近，13 mm低于单光子。SC Compton 22 mm CRC约100.15%，但背景CV为2.5、CNR仅2.324，不意味着完美恢复。')
   p('218的10 mm末帧CNR只有0.115；两种双能和对应球CNR为负。长迭代没有使所有小球稳定可见，合成图把两能密度相加也不等于自动提高每个球的CNR。')
  else:p('EHE 5e10改善多数大球，但10 mm仍敏感于噪声实现：前投影Poisson的218 CNR0.694、G4为2.177。5e9个别球CRC大于1伴随高噪声，不能作为准确超分辨证据。')
 for idx,s in enumerate(SYS):
  for gi in range(2 if s=='JSCC' else 1):
   page(CN[s]+' 的三维真值和迭代图'+(' 单光子路线' if s=='JSCC' and gi==0 else ' 联合及Compton路线' if s=='JSCC' else ''))
   figure(f'gallery_{idx}_{gi}.png','图示中央72 mm MIP，完整XY范围保留；低白高黑，固定源背景密度比0至10。最左为真实三维源MIP；EHE列0/50/100/200，JSCC列0/2000/5000/10000。大于10仅显示截顶，未从数据或指标删除。')
   p('图像横向比较仅表示本路线的实际变化。初值几乎全白是全1密度相对真实发射源尺度较小，未另行放大。各路线自身末440图用于218固定串扰背景，EHE预算440200、JSCC预算44010000。')
   if s=='EHE':p('低探测计数下，200次的背景斑点和源外结构十分明显。该图与完整域CV及球ROI弱结果一致，不能只从少数高值像素判断可见性。')
   elif s=='EHE matrix+Poisson':p('本组是同模型产生均值再加噪声的实现检验，仍受有限计数和源基底与重建基底不同影响。它不证明独立输运的物理响应准确。')
   elif s=='EHE Geant4 5e10':p('增加实际输运剂量改善了大球可见性；218积分偏高和源外泄漏仍存在。实际独立输运与矩阵自生数据的区别应与视觉改善一起阅读。')
   elif s=='EHE matrix+Poisson 5e10':p('本组噪声较5e9明显降低，22与37 mm球更清楚；218 10 mm仍较弱。200次结果保留了迭代后期斑点增加，没有以平滑或单图亮度拟合处理。')
   else:p('JSCC长期迭代保留完整200帧曲线。高峰和小球弱表现与表中指标一致；对比图不把同一图列等同于EHE的收敛程度。')
 page('实际执行代价和结果解释边界')
 table(['阶段','实际登记','实际时长及含义'],[
 ['JSCC六路10000','1669255','8节点×1GPU；Slurm 12:00:33'],
 ['EHE 5e10输运补齐','15684979','1000×50M；18节点；恢复阶段2:21:06'],
 ['EHE G4 5e10正式','1681346','Slurm 2:14；440/218求解50.78/55.70秒'],
 ['EHE Poisson 5e10生成验证','1683295','全输入生成和validation10；Slurm 6:36'],
 ['EHE Poisson 5e10正式','1683357','Slurm 1:57；440/218求解42.83/41.59秒']], [143,94,254])
 p('5e10输运首次作业因srun等待超时退出，保留13个成功worker并补987个，科学输入和种子不变。2:21:06是补齐作业时长，不能作为包含失败及等待的首次启动总耗时。')
 p('三套响应完整SHA、几何、旋转、体积、自身S、全20视角前向/转置验收通过。近期EHE正式各40个atomic/fsync检查点，三路各20帧、130个正式文件严格取回。新Poisson正式固定背景独立L2为3.59e-7，小于1e-5。CPU验收不代表GPU资源证书。')
 p('近期EHE运行以实际AllocTRES 94500 MiB为主存分母，资源峰值保留20%余量。执行正确性、独立响应物理准确性和图像质量是不同证据：模型自生数据没有独立物理校准能力；全局计数一致也不能覆盖逐bin统计不足。')
 p('近期宏全4π发射、剂量倍数1。JSCC原材料与结构保持；旧数据缺少初级标签串窗统计。人体输运、重复噪声下最优停止、小球可靠性及真实设备性能仍没有本轮证据支持。')
 page('空间调查 清理结果和后续回收选择')
 inv=read(OUT/'remote_storage_inventory.json');before=next(r['stdout'] for r in inv['results'] if r['label']=='filesystem')
 p('清理前工作存储为800 GiB，已用约753 GiB、剩余约48 GiB，df使用率95%；账号队列为空。du分配块统计中，本工程约202.33 GiB，另一个lpz旧工程约313.97 GiB，zxc/SCSPECT_EXP约224.35 GiB。du与df统计口径不同，不能要求两者简单相等。')
 cleanp=OUT/'cleanup_acceptance.json'
 if cleanp.exists():
  clean=read(cleanp);p(clean['report_summary'])
 else:p('清理候选和保留规则详见同目录存储记录。唯一原始观测、三套响应、完整迭代历史、冻结发布与科学证明不按目录年龄自动删除。')
 p('最优先的回收对象是可确认重复的传输归档、软件包下载缓存和已完成一次性辅助脚本。移除远端传输包前需确认本地保留归档SHA一致，并逐文件验证远端Factors与Sensitivity全部与原清单相符；运行环境和正式矩阵不属于下载缓存。')
 p('较大空间需要项目级决策：本工程旧511 keV List约57.23 GiB、历史Factors约36.85 GiB可以逐案例核对本地备份后迁出；旧lpz工程和zxc目录必须先确认归属、用途与备份。近期EHE原12块和完整转换响应属于可复现资产，不因存储压力直接移除。')
 p('本地14.6 GB的factors_transfer.tar.zst是当前本地Factors归档保留件，未发现已展开的相同本地成员集，不能把它当无用缓存删除。')
 page('研究结论和完整结果入口')
 p('最清晰的近期改善来自EHE增加发射剂量和连续材料核的独立校准。两者没有消除长迭代噪声、小球弱恢复和模型与Geant4不一致。分析应同时看CRC、CNR、背景CV、积分、峰位置和轴外泄漏，保留各自完整迭代范围。')
 p('近期全部研究结果已经按五组主案例与历史方法证据分层登记。正式原数组和冻结证据仍是权威来源；本报告只读衍生的图像、表格及源SHA登记在scientific_sources.json，末帧与CNR峰值完整表另存CSV。')
 table(['结果入口','对应内容'],[
 ['compton_energy_probability_v5_5e9_full10000','JSCC六路10000；ACCEPTANCE.md；全部200帧曲线'],
 ['ehe_spect_5e9_200','EHE G4 5e9；RESULTS.md；原物理审计及28图'],
 ['ehe_forward_poisson_5e9_200','EHE期望5e9模型Poisson；完整三路线'],
 ['ehe_spect_5e10_200','EHE G4 5e10；RESULTS.md；SOURCE_ANGLE.md'],
 ['ehe_forward_poisson_5e10_200','EHE期望5e10模型Poisson；RESULTS.md；18类总图'],
 ['compton_energy_probability_v5_5e9','连续核与角度核legacy5e9的2000次配对'],
 ['compton_energy_probability_v5','ideal1e9两核2000；非纯剂量对照'],
 ['compton_first_scatter_v2','事件策略与稳定几何比较'],
 ['spike_ablation','绑定与Huber的正面及代价'],
 ['process_list_global_audit_v4','数值 测度 物理响应只读诊断']], [245,246])
 p('以上目录均位于experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60。更早阶段的综合方法回顾在docs/DUAL_ENERGY_RESEARCH_REVIEW.md；固定JSCC回归基准在docs/DUAL_ENERGY_BASELINE.md。当前统一入口为实验README。')
 def footer(c,doc):
  c.setFont('ZH',8);c.setFillColor(colors.HexColor('#666666'));c.drawString(52,27,'218 与 440 keV 双能重建研究报告　2026 10 10');c.drawRightString(width-52,27,str(doc.page))
 dest=OUT/'dual_energy_nema_research_report_20261010.pdf'
 SimpleDocTemplate(str(dest),pagesize=(width,height),rightMargin=52,leftMargin=52,topMargin=47,bottomMargin=47,title='218与440keV双能重建研究报告',author='JSCC reconstruction research').build(story,onFirstPage=footer,onLaterPages=footer)
 print(str(dest))

def routes_for(systems):return [(s,c) for s in systems for c in (JCH if s=='JSCC' else CH)]

def index():
 OUT.mkdir(parents=True,exist_ok=True)
 p=H/'README.md';snapshot=H/'HISTORY_20261010.md'
 if not snapshot.exists():snapshot.write_bytes(p.read_bytes())
 catalog=[]
 definitions=[('compton_energy_probability_v5_5e9_full10000','JSCC实际5e9，六路10000','ACCEPTANCE.md'),('ehe_spect_5e9_200','EHE实际Geant4 5e9，三路200','RESULTS.md'),('ehe_forward_poisson_5e9_200','EHE期望5e9，矩阵前投影加Poisson，三路200','README.md'),('ehe_spect_5e10_200','EHE实际Geant4 5e10，三路200','RESULTS.md'),('ehe_forward_poisson_5e10_200','EHE期望5e10，矩阵前投影加Poisson，三路200','RESULTS.md')]
 for n,label,entry in definitions:
  rp=R/n;fj=read(rp/'formal_job.json');fs=read(rp/'formal_summary.json')
  assert fs['passed'];assert (rp/entry).is_file()
  catalog.append(dict(study=n,description=label,formal_job=fj['job'],entry=str((rp/entry).relative_to(ROOT)),formal_acceptance_sha256=sha(rp/'formal_summary.json'),iterations=10000 if n.startswith('compton') else 200))
 save(OUT/'result_catalog.json',dict(date='2026-10-10',main_completed_studies=catalog,scientific_source_manifest_sha256=sha(OUT/'scientific_sources.json'),automation='compton-v5 PAUSED',reconstruction_launched=False))
 links='\n'.join(f"| [{r['description']}](reports/NEMA_Body_H60/{r['study']}/{Path(r['entry']).name}) | {r['formal_job']} | {r['iterations']} |" for r in catalog)
 p.write_text('''# 218与440 keV双能重建 近期结果总入口

2026-10-10：五组主要NEMA H60研究均已实际完成、独立验收和严格取回。当前没有待继续的本地推进器，compton-v5保持暂停。JSCC既有六路10000次作为实现回归基准；EHE实际输运与模型加噪声是四个独立研究组。

- [完整研究报告 PDF](reports/dual_energy_review_20261010/dual_energy_nema_research_report_20261010.pdf)
- [整理目录及存储清理说明](reports/dual_energy_review_20261010/README.md)
- [各路线末帧指标](reports/dual_energy_review_20261010/endpoint_native_metrics.csv)与[全部球CNR峰值及末帧](reports/dual_energy_review_20261010/sphere_cnr_peak_and_final.csv)
- [18类完整图集和曲线](reports/NEMA_Body_H60/ehe_forward_poisson_5e10_200/RESULTS.md)

## 已完成的主结果

| 实验 | 正式作业 | 末迭代 |
|---|---:|---:|
'''+links+'''

比较保留EHE 0–200、JSCC 0–10000各自范围；同迭代或同图列不表示同收敛。真值为工程现有3mm三维球源，主指标覆盖120mm，MIP图示中央72mm；crop0、无平滑、固定发射源密度尺度，无单图亮度拟合。组合为gamma密度和，不是Ac225活度。

EHE 5e10全4π、一事件一光子；输运15684979复用首次13个成功worker后补齐987个，总实际5e10。旧15683333失败及恢复证据保留。[源发射角核验](reports/NEMA_Body_H60/ehe_spect_5e10_200/SOURCE_ANGLE.md)说明宏的/xcat/angle只旋转源位置，不限半球，剂量倍数1。

原EHE物理审计仍有响应偏差和逐bin统计不足，[HOLD报告](reports/NEMA_Body_H60/ehe_spect_5e9_200/PHYSICAL_HOLD.md)保留。用户随后明确要求继续原方法，正式重建已完成；不得把执行验收或模型自生数据解释为该物理偏差已消失。

## 方法历史与固定基准

- [JSCC完整流程基准](../../docs/DUAL_ENERGY_BASELINE.md)，规范入口energy_full10000_v5_workflow.py及run_energy_full10000_v5.py。
- [截至10月7日的方法研究回顾](../../docs/DUAL_ENERGY_RESEARCH_REVIEW.md)，包括密度基底、首散射、稳定几何、连续核、绑定/Huber及未完成精细场。
- [5e9两核2000对照](reports/NEMA_Body_H60/compton_energy_probability_v5_5e9/ACCEPTANCE.md)；[ideal1e9两核2000](reports/NEMA_Body_H60/compton_energy_probability_v5/ACCEPTANCE.md)不是纯剂量对照。
- [体模定义与几何真值](reports/NEMA_Body_H60/README.md)。

## 数据和代码保留规则

正式响应、观测、完整图像历史、冻结发布、科学诊断与验收证据保留。一次性文档/标题修补器及过时控制器已清出当前源码，恢复提交和逐文件SHA见清理记录；通用工作流、求解器、科学绘图和验证器保留。本次没有新增模拟、响应或重建，也不恢复已经停止的旧研究。

[本次整理前README原字节](HISTORY_20261010.md)、[10月7日快照](HISTORY_20261007.md)及[更早记录](HISTORY.md)包含当时RUNNING/PENDING语句，应按记录时间阅读。
''',encoding='utf-8',newline='\n')
 old=R/'README.md';hs=R/'HISTORY_20261010.md'
 if not hs.exists():hs.write_bytes(old.read_bytes())
 old.write_text('''# NEMA H60 双能研究结果与体模定义

2026-10-10：最新五组主结果已完成。先阅读[研究报告](../dual_energy_review_20261010/dual_energy_nema_research_report_20261010.pdf)、[近期结果总入口](../../README.md)和[18类图集及全部曲线](ehe_forward_poisson_5e10_200/RESULTS.md)。EHE0–200与JSCC0–10000各自比较，不能同迭代等同收敛。

下面保留原几何说明及早期实验记录；其日期只表示当时状态。manifest.json的preview状态描述真值生成阶段，正式输运/重建以各主组最终验收为准。本次整理前原字节另存[历史快照](HISTORY_20261010.md)。

'''+hs.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
 print('Current index and five accepted studies registered; historical README bytes preserved')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('data','pdf','index'));a=ap.parse_args();{'data':data,'pdf':pdf,'index':index}[a.mode]()
