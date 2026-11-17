"""Curate navigation identities from existing evidence; no scientific execution."""
from pathlib import Path
import copy
import json
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
H='experiments/ELLIPSE500x300_H120'
P='Analysis_Plots/research/'
OUT=ROOT/'docs/ac225/study_registry.json'

def load(path):return json.loads((ROOT/path).read_text('utf8'))
def existing(paths):return [p for p in paths if (ROOT/p).exists()]
def main():
    if OUT.exists():print('CURATED_REGISTRY_ALREADY_EXISTS');return
    old=load('docs/ac225/catalog.json');items=copy.deepcopy(old['experiments'])
    byid={e['id']:e for e in items}
    for e in items:
        ident=e['id']
        category=('history' if ident.startswith('history-') or 'stopped' in e['status'] or 'retired' in e['status'] else
                  'derenzo' if ident.startswith('derenzo-') else 'nema' if ident.startswith('nema-') else 'diagnostics')
        if ident in ['results-72mm-mip','advisor-progress']:category='reference'
        if ident=='derenzo-design-selected':category='design'
        if ident=='ehe-side-shield-design':category='design'
        if 'poisson' in ident:category='reference'
        e.update(category=category,selection='historical-reference',observed_at=old.get('formal_overlay_snapshot_at',old['snapshot_at']),
                 conclusion=e.get('note','以原始报告的研究范围和结论为准。'),limitations=[],code=[],views=20,
                 compute_status=e['status'],delivery_status='see-original-evidence',visual_status='see-original-report',physical_status='not-certified')
    for ident in ['nema-jscc-5e10','nema-jscc-5e9-v5','nema-5e10-v5-joint','derenzo-jscc-218','derenzo-jscc-440','nema-current-roi','nema-source-truth','derenzo-design-selected']:
        if ident in byid:byid[ident]['selection']='current'
    byid['results-72mm-mip'].update(title='2026-10-29方法对比画廊：原EHE矩阵快照',
        conclusion='保留6组/15路线/90选帧的方法对比；其中EHE为旧矩阵版本，不代表11月加密矩阵结果。')
    byid['nema-jscc-5e10'].update(title='NEMA实际5e10：SC218/440与纯Compton440',
        compute_status='completed',delivery_status='strict-complete-history',visual_status='reviewed',
        conclusion='1722760三路各10000次已交付；纯Compton后期环状、条纹与尖峰明显。纯Compton不命名为JSCC。',
        code=[H+'/formal_only_reconstruction_20261022.py',H+'/torch_active_operator.py',H+'/single_checkpoint_mlem.py'])
    byid['nema-5e10-v5-joint'].update(category='nema',selection='current',
        compute_status='completed',delivery_status='strict-complete-history',visual_status='reviewed',
        conclusion='1740189同440keV联合10000次完成；背景BV约35.94%，低于SC约40.50%，但13mm热球CNR下降；不说明环伪影成因已修正。',
        code=[H+'/nema_5e10_joint_workflow_20261026.py'],figures=existing([P+'nema_joint5e10_1740189_20261029/REPORT.html']))
    byid['nema-current-roi']['conclusion']='原3D球内中心距球面至少1.5mm；所有球共享完整体模内部背景，排除六球、肺和接触边界。'
    byid['derenzo-design-selected'].update(conclusion='当前输运唯一解析几何：218为6/8/10/12/14/16mm、114棒；440为10/12/14/16/18/20mm、53棒。',
        limitations=['218与440采用不同几何；当前径向6mm网格不能直接认证6mm棒分辨率。'])

    def add(ident,title,category,kind,report=None,evidence=(),data=(),jobs=None,code=(),conclusion='',limitations=(),selection='current',
            status='completed-visualized',compute='completed',delivery='strict-delivered',visual='reviewed',views=20,**extra):
        entries=existing([report] if report else [])
        ev=list(evidence)
        if report and report.startswith(P):
            parent=str(Path(report).parent).replace('\\','/')
            ev+=existing([parent+'/data/'+n for n in ['metadata.json','scientific_qa.json','direct_visual_review.json','visual_qa.json','independent_selected_display_qa.json']])
        e=dict(id=ident,title=title,category=category,topic='ac225 218 440 '+category+' '+title,status=status,data_kind=kind,
               entries=entries,evidence=existing(ev),local_data=existing(list(data)),figures=entries if report and report.startswith(P) else [],
               jobs=jobs or {},code=existing([p if p.startswith(('tools/','Geant4Sim/','Auxiliary_Studies/')) else H+'/'+p for p in code]),
               conclusion=conclusion,limitations=list(limitations),selection=selection,compute_status=compute,
               delivery_status=delivery,visual_status=visual,physical_status='not-certified',views=views,
               observed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**extra)
        if ident in byid:
            olde=byid[ident];olde.update(e)
        else:items.append(e);byid[ident]=e
    denseR=H+'/reports/EHE/ehe_dense_reruns_20261102/'
    denseD=H+'/generated/ehe_dense_reruns_20261102/'
    for phantom,dose,indices in [('NEMA','5e9',[0]),('NEMA','5e10',[1]),('Derenzo','5e9',[2,3])]:
        ident=f'{phantom.lower()}-ehe-dense-{dose}'
        add(ident,f'{phantom} EHE加密矩阵实际{dose}',phantom.lower(),'actual-Geant4',
            P+'ehe_dense_completed_1761244_20261102/REPORT.html',
            [denseR+'delivery_1761244_strict_fetch.json',denseR+'delivery_1761244_numerical_qa.json'],
            [denseD+'delivery_1761244'],{'formal':[f'1761244_{i}' for i in indices]},
            ['ehe_dense_rerun_workflow_20261102.py','ehe_dense_rerun_runtime_20261102.py'],
            conclusion='A218/A440提高到65536点/可见面，配套自身S；正式200次完成。加密后仍有边缘高值、散斑与棒组分离不足。',
            limitations=['C440→218仍为冻结原版本。','EHE物理响应尚未认证；加密不等于消除散射模型偏差。'],
            phantom=phantom,doses=[dose],energies_keV=[218,440],matrix_version='dense65536',iterations=200,save_step=1,
            plot_metadata=P+'ehe_dense_completed_1761244_20261102/data/metadata.json')
    add('ehe-dense-matrix','EHE218/440加密系统矩阵','ehe','system-matrix',
        H+'/reports/EHE/ehe_dense_surface_20261029/README.md',
        [H+'/reports/EHE/ehe_dense_surface_20261029/latest_status.json',denseR+'reconstruction_compact_freeze.json',denseR+'local_scientific_checks.json'],
        [H+'/generated/ehe_dense_surface_20261029'],{'formal_original':1749085,'missing_block_repair':'1749513_7'},
        ['ehe_dense_surface_workflow.py','ehe_dense_surface_runtime.py','ehe_dense_surface_repair_workflow.py'],
        conclusion='A218/A440原0–6与唯一补块7完成，16²→256²点/可见面；当前实际EHE重建采用此版本。',
        limitations=['保持原几何、材料与响应公式；散射均匀板深度近似及物理HOLD仍保留。','不重算C440→218。'],visual='not-an-image-study',delivery='matrix-identity-bound',matrix_version='dense65536')
    for ident,tag,views,job in [('ehe-nema-poisson-20view','ehe_dense_nema_low_noise_20261102',20,1762238),
                                ('ehe-nema-poisson-60view','ehe_dense_nema_60view_poisson_20261102',60,1762460)]:
        r=H+'/reports/EHE/'+tag+'/'
        add(ident,f'EHE NEMA{views}视角：5e9至5e12前投影＋Poisson','synthetic','matrix-forward-plus-Poisson',P+tag+'/REPORT.html',
            [r+'delivery_selected_strict.json',r+('visual_delivery_summary_20261103.json' if views==60 else 'final_delivery_qa.json')],
            [H+'/generated/'+tag+'/visual_results'],{'formal_array':job},
            [f'ehe_nema_{"60view" if views==60 else "low_noise"}_workflow_20261102.py',
             f'ehe_nema_{"60view" if views==60 else "low_noise"}_runtime_20261102.py','ehe_nema_low_noise_source_20261102.py']+
             (['ehe_60view_operator_20261102.py'] if views==60 else []),
            conclusion=('同总gamma、200次下，60角外围ROI发射量占比8组均降低；440在低两档全域NRMSE略升，高剂量误差仅小幅改善。' if views==60 else
                        '匹配模型自生数据在5e9/5e10也出现明显边缘伪影；提高计数显著减弱尖峰，但高剂量仍有模糊。'),
            limitations=['匹配原生有限体积源；不是新的真实Geant4输运。','完整1600帧远端核对，本地严格取回48选帧，完整本地历史QA为false。']+
                (['6°响应为同环周期线性列插值，配套新S；每3视角精确回到原20位置。','插值与独立Poisson实现同时改变，无纯角度因果、重复噪声统计或分辨率认证。'] if views==60 else []),
            views=views,phantom='NEMA',doses=['5e9','5e10','5e11','5e12'],energies_keV=[218,440],iterations=200,save_step=1,
            matrix_version='dense65536',delivery='strict-selected-48;remote-all1600',plot_metadata=P+tag+'/data/metadata.json')
    base=H+'/reports/Derenzo/derenzo_jscc_mono_5e10_20261030/'
    add('derenzo-jscc-5e10-218','Derenzo实际5e10：218 SC','derenzo','actual-Geant4-monoenergy',base+'README.md',
        [base+'JSCC_218/formal_delivery.json',base+'JSCC_218/formal_job.json'],
        [H+'/generated/derenzo_jscc_mono_5e10_20261030/JSCC_218'],{'formal':1752710},
        conclusion='218单光子10000次完成并严格取回；本次目录整理不冒充已有统一新版画廊。',visual='not-yet-indexed-gallery',status='completed-delivered-awaiting-visualization',
        phantom='Derenzo',doses=['5e10'],energies_keV=[218],methods=['SC'],iterations=10000,save_step=50)
    add('derenzo-jscc-5e10-440-sc','Derenzo实际5e10：440 SC成功输出保留','derenzo','actual-Geant4-monoenergy',base+'README.md',
        [base+'failure_1752461_root_cause_20261102.json',base+'JSCC_440/formal_job.json'],[],{'parent_formal':1752461},
        conclusion='1752461整体失败前已保存440 SC10000次/200帧；后续只补联合，不重算成功SC。',
        status='component-saved-parent-failed',compute='SC-saved10000;parentFAILED',delivery='not-certified-full-delivery',visual='not-delivered-gallery',
        phantom='Derenzo',doses=['5e10'],energies_keV=[440],methods=['SC'],iterations=10000,save_step=50)
    add('derenzo-jscc-5e10-440-joint','Derenzo实际5e10：440 JSCC联合64卡补算','derenzo','actual-Geant4-monoenergy',
        H+'/reports/Derenzo/derenzo_jscc_joint_5e10_retry_20261102/README.md',
        [H+'/reports/Derenzo/derenzo_jscc_joint_5e10_retry_20261102/formal_job.json','docs/organization/20261103/gpu_readonly_status.json'],
        [H+'/generated/derenzo_jscc_joint_5e10_retry_20261102'],{'formal':1761195},
        ['../ELLIPSE500x300_H120/derenzo_joint_retry_20261102.py'],
        conclusion='2026-11-03只读Slurm仍PENDING/Priority；8节点×8张5090，只补440联合，不重复SC。',
        status='formal-pending',compute='PENDING/Priority',delivery='not-started',visual='not-started',
        phantom='Derenzo',doses=['5e10'],energies_keV=[440],methods=['JSCC'],iterations=10000,save_step=50)
    add('derenzo-ehe-dense-5e10','Derenzo EHE实际5e10：输运完成、严格传输待修复','derenzo','actual-Geant4-monoenergy',denseR+'README.md',
        ['docs/organization/20261103/cpu_readonly_status.json',denseD+'one_shot_registration.json'],
        [denseD+'EHE_218',denseD+'EHE_440'],{'MC440':16256692,'collect440':16256695,'MC218':16256696,'collect218':16256697},
        ['ehe_dense_rerun_workflow_20261102.py'],
        conclusion='两能量各500个100M输运worker及收集均完成；衔接进程60468因已存在中断取回partial而退出，GPU重建尚未登记。',
        limitations=['保留partial和已完成输运；严格补齐传输后才能提交后续两路，不重复蒙卡。'],
        status='MC-complete-handoff-blocked',compute='MC+collectionCOMPLETED;GPU-unsubmitted',delivery='partial-transfer-retained',visual='not-started',
        phantom='Derenzo',doses=['5e10'],energies_keV=[218,440],matrix_version='dense65536',iterations=200,save_step=1)
    add('nema-prejune-compton','NEMA实际5e9：旧Compton核回退对照','diagnostics','actual-Geant4-reused-projection',
        P+'nema_prejune_compton_1739632_20261026/REPORT.html',
        [H+'/reports/NEMA_Body_H60/nema_5e9_prejune_compton_8x1_20261026/README.md'],
        [H+'/generated/nema_prejune_compton_formal_1739632_20261026'],{'formal':1739632},['prejune_compton_workflow.py'],
        conclusion='近期NEMA实际投影配旧核/匹配S/OSEM8完成10000次，明显环纹和尖峰仍存在，回退未消除问题。',
        limitations=['旧OSEM8每轮8次子集更新，不与MLEM同迭代等同收敛。','旧子集除完整S的约定使总量约新流程1/8；未额外乘8拟合。'],selection='diagnostic')
    diagnostics=[
      ('compton-ring','nema_compton_ring_diagnosis_20261026','Compton环状伪影','环纹与网格/响应离散化值得隔离研究；观察性关联不足以确定唯一成因。'),
      ('compton-transition','nema_compton_transition_audit_20261026','Compton角点过渡与体积','未发现简单漏乘/重复乘体积或S约定错误；中心点响应代表有限单元的积分误差仍是候选。'),
      ('compton-git-history','nema_compton_git_history_20261026','Compton算法Git历史','源码版本、核/S/筛选/求解均有差别；历史图像外观不能代替同源匹配实验。'),
      ('compton-legacy-backup','legacy_compton_backup_audit_20261026','旧Compton结果备份审计','备份Image_JSCCD为纯Compton；511keV旧数据与440keV近期NEMA/FOV不同，不当作直接等价基准。'),
      ('compton-new-vs-old','nema5e9_new_vs_prejune_20261026','NEMA新旧Compton流程对比','旧流程后期起伏更强，回退未消除环纹；同时改变多步骤，不能唯一归因。'),
      ('ehe-crystal-response','ehe_crystal_response_20261029','EHE单晶体空间响应','原矩阵六个晶体的分能量MIP/切片；晶体响应覆盖及面采样问题另由准直器审计解释。'),
      ('ehe-collimator-audit','ehe_collimator_response_audit_20261029','EHE准直器响应与面积分','256点/可见面的局部积分不足已确认；65536点加密版本独立完成，物理模型仍需验证。'),
      ('ehe-fov-clearance','ehe_fov_clearance_20261102','EHE准直器FOV间隙','准直器前端298.5mm，完整旋转FOV最小间隙48.5mm，未伸入FOV。'),
      ('ehe-dense-edge','ehe_dense_edge_diagnosis_20261102','EHE加密矩阵边缘高值','噪声放大与空间响应失配并存；NEMA440→218串窗预测低约25.27%，加密没有消除主要尖峰。'),
      ('ehe-edge-original','ehe_edge_diagnosis_20261019','EHE原矩阵边缘高值历史诊断','保留早期边缘/S/计数诊断；当前结论优先使用11月加密矩阵及匹配Poisson对照。'),
    ]
    for ident,tag,title,conclusion in diagnostics:
        parent=ROOT/P/tag;report=next((p for p in [parent/'REPORT.html',parent/'README.txt',parent/'README.md'] if p.exists()),None)
        if report:add(ident,title,'diagnostics','read-only-diagnostic',report.relative_to(ROOT).as_posix(),conclusion=conclusion,
            limitations=['结论范围以专题报告和冻结输入为准；诊断不授权改变生产模型。'],selection='diagnostic',compute='analysis-complete',delivery='source-bound-analysis')
    add('nema-bv-cnr-reference','NEMA分能量BV–CNR历史图','reference','derived-real-Geant4-results',
        P+'nema_bv_cnr_218_440_20261023/REPORT.html',
        conclusion='既有NEMA结果的BV–CNR曲线；EHE为加密矩阵之前的结果，不冒充新1761244曲线。',selection='historical-reference')
    add('ac225-meeting-slides','225Ac成像汇报图与幻灯片材料','reference','presentation-artifacts',
        P+'ac225_meeting_slides_20261029/README.md',conclusion='汇报材料为对应日期的结果快照；最新数值和矩阵版本以当前结果入口为准。',selection='historical-reference')
    # Remove guessed code paths and use only existing exact identities.
    for e in items:
        e['code']=existing(e.get('code',[]));e['evidence']=existing(e.get('evidence',[]))
        e['entries']=existing(e['entries']) or existing(e.get('figures',[])) or [H+'/README.md']
        if e['selection']=='current' and e['id'].startswith('nema-ehe-g4'):e['selection']='historical-reference'
    record=dict(schema_version=2,updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        purpose='Curated navigation facts; original contracts/registrations remain authorities',studies=items,
        components=old['components'],source_catalog_snapshot_at=old['snapshot_at'])
    OUT.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('CURATED_REGISTRY_CREATED',len(items))

if __name__=='__main__':main()
