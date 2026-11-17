"""Rebuild local225Ac navigation and pointer folders without importing runners.

Only documentation, JSON/CSV indexes and result pointer folders are written.
All original source, frozen releases, data arrays and per-study figure artifacts
retain their locations. Existing changed navigation files are archived by byte.
"""
from __future__ import annotations
import ast
from collections import Counter
import csv
from datetime import datetime
from html import escape
from html.parser import HTMLParser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote,urlsplit
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/ac225'
AUDIT=ROOT/'docs/organization/20261103'
H='experiments/ELLIPSE500x300_H120'
P='Analysis_Plots/research/'
CATEGORIES={'nema':'NEMA实际成像','derenzo':'Derenzo实际成像','ehe':'EHE系统矩阵',
            'synthetic':'前投影与Poisson对照','diagnostics':'物理与算法诊断','design':'体模与系统设计',
            'reference':'展示与历史参考','history':'归档研究'}
MANAGED={}
STATUS_ZH={'completed':'完成','strict-delivered':'已严格取回','reviewed':'图形已复核',
 'strict-complete-history':'完整历史已严格交付','strict-historical-delivery':'历史结果已严格交付',
 'reviewed-current-ROI':'已按当前ROI复核','see-original-evidence':'见原阶段证据','see-original-report':'见原专题报告',
 'not-started':'尚未开始','not-yet-indexed-gallery':'统一新版画廊待整理','not-certified-full-delivery':'成功输出待独立严格交付',
 'not-delivered-gallery':'尚无可交付画廊','completed-delivered-visualized':'完成 · 已交付 · 图形已复核',
 'completed-delivered-awaiting-visualization':'重建已交付；统一新版画廊待整理',
 'completed-selected-delivery-visualized':'完成 · 选帧已交付 · 图形已复核','component-saved-parent-failed':'SC输出保留，父任务失败',
 'formal-pending':'正式任务排队','MC-complete-handoff-blocked':'输运完成，严格传输中断','PENDING/Priority':'排队（Priority）',
 'MC+collectionCOMPLETED;GPU-unsubmitted':'输运/收集完成；GPU未提交','SC-saved10000;parentFAILED':'SC10000次已保存；父任务失败',
 'partial-transfer-retained':'中断文件已保留','strict-selected-48;remote-all1600':'本地48选帧；远端1600帧已核对',
 'matrix-identity-bound':'矩阵SHA身份已绑定','design-only':'设计方案','local-design':'本地设计文件','design-figures':'设计示意图',
 'presentation-only':'汇报材料','local-presentation':'本地汇报文件','see-dated-material':'见当日材料',
 'analysis-complete':'分析完成','source-bound-derived-metrics':'来源已绑定的派生指标','dated-figure-report':'当日图表报告'}
def status_zh(value):return STATUS_ZH.get(value,value)
SAFE_FILES={'README.md','docs/README.md','experiments/README.md',H+'/README.md',H+'/AGENTS.md',
 H+'/reports/README.md',H+'/reports/EHE/README.md',H+'/reports/Derenzo/README.md',
 'Analysis_Plots/README.md','Analysis_Plots/research/README.md','Analysis_Plots/results/README.md',
 'Analysis_Plots/results/REPORT.html','Analysis_Plots/results/data/current_results.json',
 'Analysis_Plots/results/data/organization_audit_20261103.json','Analysis_Plots/results/data/artifact_sha256.json',
 'tools/README.md','tools/research/README.md','Auxiliary_Studies/README.md','Geant4Sim/README.md'}

def read(path):return json.loads(Path(path).read_text('utf8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def link(target,doc):return os.path.relpath(ROOT/target,Path(doc).parent).replace('\\','/')
def md(target,doc,label=None):return f'[{label or Path(target).name}]({link(target,doc)})'
def exists(p):return (ROOT/p).exists()
def backup(path):
    path=Path(path)
    if path.exists():
        saved=AUDIT/'before'/rel(path)
        if not saved.exists():saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,saved)
def put(path,text):
    path=Path(path);name=rel(path)
    allowed=(name in SAFE_FILES or name.startswith(('docs/ac225/','docs/organization/20261103/',
        'Analysis_Plots/results/current/','Analysis_Plots/results/history/20261029/')))
    if not allowed:raise ValueError('Navigation writer cannot touch scientific paths: '+name)
    if path.suffix.lower() not in {'.md','.txt','.json','.csv','.html'}:raise ValueError('Text navigation only')
    raw=text.encode('utf8')
    if not path.exists() or path.read_bytes()!=raw:
        backup(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    MANAGED[name]=sha(path)
def js(path,obj):put(path,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def source_group(path):
    n=path.name.lower();p=path.as_posix()
    if '/scripts/' in p or '/reports/' in p:return 'frozen-or-artifact-copy'
    if p.startswith('tools/baseline/') or any(t in n for t in ['project_index','catalog','navigation','registry','organization','results_index']):return 'documentation-navigation'
    if n.startswith('test_') or n.startswith('verify') or n.endswith('_verify.py'):return 'verification-tests'
    if any(t in n for t in ['workflow','monitor','resume','submit','launch','queue','postprocess']):return 'execution-control'
    if any(t in n for t in ['plot','analy','compare','report','visual','audit','diagnos','review']):return 'analysis-figures'
    if any(t in n for t in ['transport','collect','eventaction','runaction','steppingaction']):return 'transport'
    if any(t in n for t in ['geometry','phantom','macro','config','source']):return 'geometry-source'
    if any(t in n for t in ['response','sensi','calibrat','kernel','probability','measure']):return 'calibration-physics'
    if any(t in n for t in ['mlem','osem','recon','operator','main_','run_']):return 'reconstruction'
    return 'shared-or-legacy-source'

def sources(studies):
    files=set()
    for args in [['git','ls-files','-z'],['git','ls-files','--others','--exclude-standard','-z']]:
        files.update(p for p in subprocess.check_output(args,cwd=ROOT).decode('utf8').split('\0') if p)
    extensions={'.py','.cu','.cc','.cpp','.h','.hpp','.m','.ps1','.sh','.mac'}
    current={p for s in studies if s['selection']=='current' for p in s.get('code',[])}
    rows=[]
    for name in sorted(files):
        p=Path(name)
        if p.suffix.lower() not in extensions or not (ROOT/p).is_file() or name.startswith('docs/organization/') or '/__pycache__/' in name:continue
        group=source_group(p);purpose=p.stem;functions=[]
        if p.suffix=='.py':
            try:
                tree=ast.parse((ROOT/p).read_text('utf-8-sig'));purpose=' '.join((ast.get_docstring(tree) or p.stem).split())[:190]
                functions=[{'name':n.name,'line':n.lineno} for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
            except (SyntaxError,UnicodeError):purpose='历史源码；依赖和解释器按原研究说明'
        role=('frozen-copy' if group=='frozen-or-artifact-copy' else 'current-study-entry' if name in current else
              'retired-controller' if any(t in p.name for t in ['resume_reconstruction_controllers','streaming_4090','formal_queue_gate']) else
              'navigation-only' if group=='documentation-navigation' else 'library-or-historical-tool')
        rows.append(dict(path=name,group=group,purpose=purpose,role=role,definitions=functions))
    return rows

def study_guides(studies,now):
    for s in studies:
        doc=DOC/'studies'/s['category']/(s['id']+'.md');s['guide']=rel(doc)
        lines=['# '+s['title'],'',s['conclusion'],'','## 条件与完成范围','',
          '| 字段 | 内容 |','|---|---|',
          f"| 数据来源 | {s['data_kind']} |",f"| 版本地位 | {s['selection']} |",
          f"| 计算 | {status_zh(s['compute_status'])} |",f"| 严格交付 | {status_zh(s['delivery_status'])} |",f"| 可视化 | {status_zh(s['visual_status'])} |",
          f"| 视角 | {s.get('views','见合同')} |",f"| 迭代 / 保存 | {s.get('iterations','见原合同')} / {s.get('save_step','见原合同')} |",
          f"| 作业 | {json.dumps(s.get('jobs',{}),ensure_ascii=False)} |",f"| 状态依据日期 | {s['observed_at']} |"]
        if s.get('matrix_version'):lines.append(f"| 矩阵 | {s['matrix_version']} |")
        if s.get('doses'):lines.append('| 发射量级 | '+', '.join(s['doses'])+' |')
        if s.get('limitations'):lines+=['','## 解读边界','']+['- '+v for v in s['limitations']]
        for title,key in [('报告与原始说明','entries'),('阶段证据','evidence'),('源码入口','code'),('本地数据位置','local_data'),('图表入口','figures')]:
            paths=list(dict.fromkeys(s.get(key,[])))
            if paths:lines+=['','## '+title,'']+['- '+md(p,doc,p) for p in paths]
        lines+=['','图表中的SC为单光子，JSCC为同能量SC+Compton联合；纯Compton单列。光子预算不是225Ac衰变数。',
                '', '本页为查询说明，原合同、冻结及交付证据保留原路径。目录存在不代表计算完成；本页不调用工作流。',
                '',md('docs/ac225/README.md',doc,'返回225Ac总入口')+' · '+md('docs/ac225/CONCLUSIONS.md',doc,'研究结论')+'\n']
        put(doc,'\n'.join(lines))
    for cat,title in CATEGORIES.items():
        doc=DOC/'studies'/cat/'README.md';rows=[s for s in studies if s['category']==cat]
        text=['# '+title,'','| 研究 | 版本 | 计算 / 交付 | 结论 |','|---|---|---|---|']
        for s in rows:text.append(f"| {md(s['guide'],doc,s['title'])} | {s['selection']} | {s['compute_status']} / {s['delivery_status']} | {s['conclusion']} |")
        text+=['',md('docs/ac225/README.md',doc,'返回总入口'),''];put(doc,'\n'.join(text))
    doc=DOC/'studies/README.md';put(doc,'# 按研究类别查询\n\n'+'\n'.join('- '+md('docs/ac225/studies/'+c+'/README.md',doc,t) for c,t in CATEGORIES.items())+'\n')

def result_routes(studies):
    legacy=read(ROOT/'Analysis_Plots/results/data/result_catalog.json')
    rows=[]
    for r in legacy['routes']:
        if r['method']=='EHE':continue
        doc=f"Analysis_Plots/results/{r['phantom']}/{r['dose']}/{r['energy_keV']}keV/REPORT.html"
        rows.append(dict(id=f"actual-{r['phantom']}-{r['dose']}-{r['energy_keV']}-{r['method']}",
            kind='actual',phantom=r['phantom'],dose=r['dose'],energy_keV=r['energy_keV'],method=r['method'],views=20,
            job=r['source_job'],channel=r['channel'],matrix='frozen JSCC source study',status='completed-delivered-visualized',
            report=doc,source=r.get('source_directory'),evidence=r.get('source_authority'),
            iterations=10000,scope_note='此同图画廊中的EHE行为旧矩阵；当前EHE请查看加密版本。',normalization_density=r.get('normalization_density')))
    dense=P+'ehe_dense_completed_1761244_20261102/'
    d=read(ROOT/(dense+'data/metadata.json'))
    for r in d['routes']:
        rows.append(dict(id=f"actual-{r['phantom']}-{r['dose']}-{r['energy_keV']}-EHE",kind='actual',
            phantom=r['phantom'],dose=r['dose'],energy_keV=r['energy_keV'],method='EHE',views=20,job=r['job'],
            channel=r['channel'],matrix='dense65536;C frozen original',status='completed-delivered-visualized',iterations=200,
            report=dense+'REPORT.html',metadata=dense+'data/metadata.json',evidence=H+'/reports/EHE/ehe_dense_reruns_20261102/delivery_1761244_strict_fetch.json',
            normalization_density=r['normalization_density'],scope_note='实际Geant4投影；物理响应HOLD保留。'))
    for tag,views in [('ehe_dense_nema_low_noise_20261102',20),('ehe_dense_nema_60view_poisson_20261102',60)]:
        b=P+tag+'/';m=read(ROOT/(b+'data/metadata.json'))
        for r in m['routes']:
            rows.append(dict(id=f"synthetic-NEMA-{r['dose']}-{r['energy_keV']}-{views}views",kind='synthetic',phantom='NEMA',
                dose=r['dose'],energy_keV=r['energy_keV'],method='EHE',views=views,job=r['job'],channel=r['channel'],
                matrix='dense65536'+(';periodic linear angle interpolation' if views==60 else ''),status='completed-selected-delivery-visualized',iterations=200,
                report=b+'REPORT.html',metadata=b+'data/metadata.json',evidence=H+'/reports/EHE/'+tag+'/delivery_selected_strict.json',
                source=H+'/generated/'+tag+'/visual_results',normalization_density=r['normalization_density'],
                scope_note='总期望gamma，不是实际输运；全1600帧远端核对，本地48选帧。'+('60角含响应插值，不能作纯角度因果结论。' if views==60 else '')))
    for ident in ['derenzo-jscc-5e10-218','derenzo-jscc-5e10-440-sc','derenzo-jscc-5e10-440-joint','derenzo-ehe-dense-5e10']:
        s=next(s for s in studies if s['id']==ident)
        for e in s['energies_keV']:
            method=s.get('methods',['EHE'])[0]
            rows.append(dict(id=f'actual-Derenzo-5e10-{e}-{method}',kind='actual',phantom='Derenzo',dose='5e10',energy_keV=e,
                method=method,views=20,job=s['jobs'],channel=str(e)+('_SinglePlusCompton' if method=='JSCC' else '_SinglePhoton'),
                matrix=s.get('matrix_version','frozen JSCC study'),status=s['status'],iterations=s['iterations'],
                report=s['guide'],evidence=s['evidence'][0] if s['evidence'] else s['entries'][0],scope_note=s['conclusion']))
    rows.sort(key=lambda r:(r['kind']!='actual',{'NEMA':0,'Derenzo':1}[r['phantom']],float(r['dose']),r['views'],-r['energy_keV'],{'SC':0,'JSCC':1,'EHE':2}[r['method']]))
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:
        root=ROOT/'Analysis_Plots/results/current'/r['kind']/r['phantom']/r['dose']/f"{r['views']}views"/f"{r['energy_keV']}keV"/r['method']
        r['pointer']=rel(root/'README.md');js(root/'case.json',r)
        fields=['kind','phantom','dose','energy_keV','method','views','job','channel','matrix','status','iterations','normalization_density']
        lines=[f"# {r['phantom']} {r['dose']} {r['energy_keV']}keV {r['method']} {r['views']}视角",'',r['scope_note'],'',
            '| 条件 | 内容 |','|---|---|']+[f'| {k} | {r[k]} |' for k in fields if k in r]
        lines+=['','## 查询位置','', '- '+md(r['report'],root/'README.md','图表或当前阶段说明')]
        for key in ['metadata','evidence','source']:
            if r.get(key) and exists(r[key]):lines.append('- '+md(r[key],root/'README.md',key+'：'+r[key]))
        lines+=['','此目录只存查询指针，不复制原始数组或图像。', '',md('Analysis_Plots/results/REPORT.html',root/'README.md','结果总览'),'']
        put(root/'README.md','\n'.join(lines))
    parents=set()
    for r in rows:
        p=(ROOT/r['pointer']).parent.parent
        while p.is_relative_to(ROOT/'Analysis_Plots/results/current'):
            parents.add(p);p=p.parent
    for p in sorted(parents,key=lambda p:len(p.parts),reverse=True):
        children=sorted(q for q in p.iterdir() if q.is_dir())
        put(p/'README.md','# '+p.name+'结果分类\n\n'+'\n'.join('- '+md(rel(q/'README.md'),p/'README.md',q.name) for q in children)+'\n\n'+md('Analysis_Plots/results/REPORT.html',p/'README.md','返回结果总览')+'\n')
    return rows

def html_index(path,studies,routes=None):
    def a(p,title):return f'<a href="{escape(link(p,path),quote=True)}">{escape(title)}</a>'
    title='225Ac工程与结果导航' if routes is None else '225Ac成像结果目录'
    nav=' · '.join(a(p,t) for p,t in [('docs/ac225/README.md','工程入口'),('docs/ac225/CURRENT_STATUS.md','当前阶段'),('docs/ac225/CONCLUSIONS.md','结论与限制'),('docs/ac225/CODE_GUIDE.md','代码路线'),('Analysis_Plots/results/REPORT.html','结果目录')])
    rows=''
    if routes is not None:
        head='<th>数据</th><th>体模 / 总gamma量</th><th>能量 / 方法 / 角度</th><th>矩阵</th><th>状态</th><th>查询</th>'
        for r in routes:
            # Search acquisition fields, not incidental digits in density
            # constants or archive paths (which can make20views match "60").
            raw=' '.join(str(r.get(k,'')) for k in ['id','kind','phantom','dose','energy_keV','method','views','job','channel','matrix','status'])
            raw+=f" {r['views']}views {r['views']}角 {r['energy_keV']}keV {status_zh(r['status'])}";kind=r['kind']
            matrix_zh={'frozen JSCC source study':'原研究冻结JSCC矩阵/响应','frozen JSCC study':'原研究冻结JSCC矩阵/响应','dense65536;C frozen original':'加密65536点/面；C原版本','dense65536':'加密65536点/面','dense65536;periodic linear angle interpolation':'加密65536点/面；同环响应插值'}.get(r['matrix'],r['matrix'])
            rows+=f'<tr data-kind="{kind}" data-search="{escape(raw.casefold(),quote=True)}"><td>{"实际Geant4" if kind=="actual" else "前投影+Poisson"}</td><td>{r["phantom"]}<br>{r["dose"]}</td><td>{r["energy_keV"]} keV · {r["method"]}<br>{r["views"]}角</td><td>{escape(matrix_zh)}</td><td>{escape(status_zh(r["status"]))}</td><td>'+a(r['pointer'],'条件与来源')+'<br>'+a(r['report'],'图表 / 阶段')+'</td></tr>'
        options='<option value="">全部数据</option><option value="actual">实际Geant4</option><option value="synthetic">前投影+Poisson</option>'
    else:
        head='<th>类别</th><th>研究</th><th>计算 / 交付</th><th>结论</th><th>入口</th>'
        for s in sorted(studies,key=lambda s:(s['selection']!='current',list(CATEGORIES).index(s['category']),s['title'])):
            rows+=f'<tr data-kind="{s["category"]}" data-search="{escape(json.dumps(s,ensure_ascii=False).casefold(),quote=True)}"><td>{CATEGORIES[s["category"]]}</td><td>{escape(s["title"])}</td><td>{escape(status_zh(s["compute_status"]))}<br>{escape(status_zh(s["delivery_status"]))}</td><td>{escape(s["conclusion"])}</td><td>'+a(s['guide'],'研究说明')+'<br>'+a(s['entries'][0],'原报告')+'</td></tr>'
        options='<option value="">全部研究</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in CATEGORIES.items())
    body=fr'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px 'Microsoft YaHei',sans-serif;background:#f3f6f8;color:#243746;margin:0}}main{{max-width:1540px;margin:auto;padding:32px}}h1{{font-size:30px}}p{{line-height:1.8}}a{{color:#146c83}}.controls{{display:flex;gap:14px;position:sticky;top:0;background:#f3f6f8;padding:15px 0}}input,select{{padding:12px;font:inherit;border:1px solid #b7cbd3;border-radius:5px}}input{{flex:1}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:13px;border-bottom:1px solid #d9e3e9;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#e6eff3}}td:nth-child(4){{max-width:540px}}.note{{color:#506773}}[hidden]{{display:none!important}}@media(max-width:800px){{main{{padding:16px}}table{{font-size:13px}}td,th{{padding:7px}}}}</style>
<style>th:last-child,td:last-child{{min-width:100px;white-space:nowrap}}</style>
<main><h1>{title}</h1><p>{nav}</p><p>SC＝单光子；JSCC＝同能量SC+Compton联合；纯Compton单列。实际输运与模型合成数据分开。总gamma量不是225Ac衰变次数。</p>
<p class="note">实际EHE成像优先查65536点/面加密矩阵。20/60角Poisson是独立模型对照；60角含响应列插值。状态、交付和物理结论分别阅读。每个条件目录只保存索引，原始数组和冻结证据保持原位。</p>
<div class="controls"><input id="q" aria-label="搜索" placeholder="搜索：NEMA 5e10、Derenzo、60、Compton、作业号…"><select id="kind" aria-label="类别">{options}</select><span id="count"></span></div>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
<p>{a('docs/ac225/DIRECTORY_MAP.md','完整目录职责')} · {a('docs/ac225/SOURCE_INDEX.md','源码索引')} · {a('docs/ac225/OPERATIONS.md','操作与维护')} · {a('Analysis_Plots/results/history/20261029/REPORT.html','2026-10-29旧EHE方法对比快照')}</p></main>
<script>const q=document.querySelector('#q'),k=document.querySelector('#kind'),rs=[...document.querySelectorAll('tbody tr')];function apply(){{const tokens=q.value.toLowerCase().trim().split(/\s+/).filter(Boolean);let n=0;for(const r of rs){{const ok=(!k.value||r.dataset.kind===k.value)&&tokens.every(t=>r.dataset.search.includes(t));r.hidden=!ok;if(ok)n++;}}document.querySelector('#count').textContent=n+' 条';}}q.addEventListener('input',apply);k.addEventListener('change',apply);apply();</script></html>'''
    put(path,body)

def code_guides(code,studies):
    counts=Counter(r['group'] for r in code);doc=DOC/'SOURCE_INDEX.md'
    text=['# 源码索引','',f'现存源码/宏/脚本共{len(code)}个。模块说明、顶层函数/类和行号见[source_index.json](source_index.json)。查询不导入这些模块。','',
      '| 职责 | 数量 | 分类入口 |','|---|---:|---|']
    for group,count in sorted(counts.items()):
        dest=DOC/'code'/group/'README.md'
        text.append(f'| {group} | {count} | {md(rel(dest),doc,"文件列表")} |')
        lines=['# '+group+'源码','', '职责索引用于打开源码；冻结副本和历史控制器不作为新的执行入口。','', '| 文件 | 身份 | 说明 |','|---|---|---|']
        for row in code:
            if row['group']==group:lines.append(f"| {md(row['path'],dest,row['path'])} | {row['role']} | {row['purpose'].replace('|','/')} |")
        lines+=['',md('docs/ac225/CODE_GUIDE.md',dest,'算法与流程路线'),''];put(dest,'\n'.join(lines))
    text+=['','```powershell','python tools/project_index.py files --group reconstruction','python tools/project_index.py find single_mlem','python tools/project_index.py files --group execution-control --topic ehe','```','',
        '[阅读代码路线](CODE_GUIDE.md)。`frozen-or-artifact-copy`为既有发布或图形副本；`retired-controller`保持停止。','']
    put(doc,'\n'.join(text));put(DOC/'code/README.md','# 代码职责分类\n\n'+ '\n'.join('- '+md('docs/ac225/code/'+g+'/README.md',DOC/'code/README.md',g) for g in sorted(counts))+'\n')

def directories(studies):
    parents=[ROOT,ROOT/'experiments',ROOT/H,ROOT/H/'reports',ROOT/H/'reports/NEMA_Body_H60',ROOT/H/'reports/Derenzo',ROOT/H/'reports/EHE',
             ROOT/H/'generated',ROOT/'tools',ROOT/'Geant4Sim',ROOT/'Auxiliary_Studies',ROOT/'Analysis_Plots',ROOT/'Analysis_Plots/research',DOC,DOC/'studies',DOC/'code']
    parents += [p for p in (ROOT/'Analysis_Plots/research').iterdir() if p.is_dir()]
    parents += [p for p in (ROOT/'Analysis_Plots/results/current').rglob('*') if p.is_dir()]
    parents += [p for p in (DOC/'studies').iterdir() if p.is_dir()]
    rows={}
    guides={};related={}
    for s in studies:
        for value in s['local_data']+[str(Path(v).parent).replace('\\','/') for v in s['entries']]:
            p=Path(value)
            while len(p.parts)>2:
                name=p.as_posix()
                related.setdefault(name,set()).add(s['id'])
                guides.setdefault(name,s['guide'])
                if p.parent.as_posix() in [H+'/generated',H+'/reports/NEMA_Body_H60',H+'/reports/Derenzo',H+'/reports/EHE','Analysis_Plots/research']:break
                p=p.parent
    descriptions={
      'Analysis_Plots':'图形工程；当前条件入口与按日期保留的原科学图集。',
      'Geant4Sim':'真实Geant4输运实现、宏和各探头代码；源身份由研究合同绑定。',
      'Auxiliary_Studies':'系统矩阵、因子转换、Compton灵敏度及独立辅助研究。',
      'GenProj':'极坐标源测度与历史前投影，密度/体积约定的参考。',
      'distributed':'历史通用分布式重建入口，不代替当前正式发布。',
      'Reproduction':'历史本地复现教程，与当前500×300×120mm工程分开。',
      'Results':'旧本地60mm结果接口；当前结果查Analysis_Plots/results。',
      'CntStat':'旧本地计数输入位置，不能等同当前独立generated投影。',
      'List':'旧本地事件列表接口，当前事件身份按原研究SHA。',
      'Factors':'旧本地矩阵接口，当前系统矩阵按具体冻结合同。',
      'tmp':'本地临时工作文件；目录存在不表示可自动删除。',
      'experiments/FOV120':'另一代几何/历史实验，不能和当前源、矩阵混用。'}
    for parent in parents:
        if not parent.is_dir():continue
        for p in sorted(parent.iterdir()):
            if not p.is_dir() or p.name.startswith(('.', '__')):continue
            name=rel(p);local='/generated' in name or name in ['Factors','CntStat','List','tmp']
            role=('local-runtime-data' if local else 'current-result-pointer' if '/results/current' in name else
                  'dated-scientific-artifact' if name.startswith(P) else 'stage-evidence' if '/reports/' in name else 'source-or-documentation')
            guide=guides.get(name)
            if guide is None and (p/'README.md').is_file():guide=rel(p/'README.md')
            description=descriptions.get(name,{
              'local-runtime-data':'原运行数据位置；按同名研究reports及manifest查证，不移动或自动清理。',
              'current-result-pointer':'条件分类与指针；README/case.json连接原结果，不复制科学数组。',
              'dated-scientific-artifact':'按日期保留的原图集/指标/脚本，结果与QA范围按metadata。',
              'stage-evidence':'研究合同、冻结、登记与阶段证明，原字节保留。',
              'source-or-documentation':'源码、组件或说明目录；按已有README及相关研究入口阅读。'}[role])
            rows[name]=dict(path=name,local_only=local,role=role,guide=guide,description=description,related_study_ids=sorted(related.get(name,[])))
    return [rows[k] for k in sorted(rows)]

def directory_document(folders,studies):
    doc=DOC/'DIRECTORY_INDEX.md';byid={s['id']:s for s in studies}
    lines=['# 各级目录查询','',f'共{len(folders)}个浅层目录。原科学目录不增加可能改变成员SHA的新文件，说明通过此表、已有README和研究指针关联。','',
           '| 目录 | 职责 | 说明与研究 |','|---|---|---|']
    for row in folders:
        refs=[md(byid[i]['guide'],doc,i) for i in row['related_study_ids'] if i in byid]
        if not refs and row['guide']:refs=[md(row['guide'],doc,'目录说明')]
        lines.append(f"| {md(row['path'],doc,row['path'])} | {row['description']} | "+'<br>'.join(refs)+' |')
    lines+=['','[工程目录总图](DIRECTORY_MAP.md) · [返回225Ac入口](README.md)',''];put(doc,'\n'.join(lines))

def experiments_doc(studies):
    doc=DOC/'EXPERIMENTS.md';lines=['# 实验与证据目录','',
        '当前研究页同时列计算、严格交付、可视化和物理结论；历史记录保留原编号，不能据旧PENDING或held字段恢复任务。','']
    for cat,title in CATEGORIES.items():
        lines+=['## '+title,'','| 查询ID / 入口 | 研究 | 状态 | 结论 |','|---|---|---|---|']
        for s in studies:
            if s['category']==cat:lines.append(f"| {md(s['guide'],doc,s['id'])} | {s['title']} | {s['compute_status']} / {s['delivery_status']} | {s['conclusion']} |")
        lines.append('')
    lines+=['完整字段见[catalog.json](catalog.json)，源码和函数查询见[SOURCE_INDEX.md](SOURCE_INDEX.md)。',''];put(doc,'\n'.join(lines))

def link_check():
    failures=[];checked=0
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.urls=[];self.base=None
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=='base':self.base=attrs.get('href')
            for key in ['href','src']:
                if key in attrs and tag!='base':self.urls.append(attrs[key])
    for name in list(MANAGED):
        p=ROOT/name;urls=[];base=p.parent
        if p.suffix=='.md':urls=re.findall(r'\]\(([^\n]+?)\)',p.read_text('utf8'))
        if p.suffix=='.html':
            parser=Parser();parser.feed(p.read_text('utf8'));urls=parser.urls
            if parser.base:base=(base/parser.base).resolve()
        for u in urls:
            u=u.strip('<>')
            if not u or u.startswith('#') or urlsplit(u).scheme:continue
            target=(base/unquote(u.split('#')[0])).resolve();checked+=1
            if not target.exists():failures.append(dict(document=name,link=u))
    return dict(passed=not failures,links_checked=checked,failures=failures)

def main():
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    registry=read(DOC/'study_registry.json');studies=registry['studies']
    for s in studies:s.setdefault('jobs',{})
    study_guides(studies,now)
    # Keep old source-bound result_catalog and all6 scientific subgalleries intact.
    original=AUDIT/'before/Analysis_Plots/results/REPORT.html'
    if not original.exists():backup(ROOT/'Analysis_Plots/results/REPORT.html')
    legacy=original.read_text('utf8').replace('<meta charset="utf-8">','<meta charset="utf-8"><base href="../../">',1)
    legacy=legacy.replace('<main>','<main><p><strong>2026-10-29历史快照：EHE使用原面采样矩阵。</strong> <a href="REPORT.html">返回当前版本目录</a></p>',1)
    put(ROOT/'Analysis_Plots/results/history/20261029/REPORT.html',legacy)
    routes=result_routes(studies)
    js(ROOT/'Analysis_Plots/results/data/current_results.json',dict(schema_version=2,updated_at=now,routes=routes,
        source_legacy_catalog_sha256=sha(ROOT/'Analysis_Plots/results/data/result_catalog.json'),
        source_registry_sha256=sha(DOC/'study_registry.json'),raw_arrays_moved=False,raw_history_copied=False))
    html_index(DOC/'INDEX.html',studies);html_index(ROOT/'Analysis_Plots/results/REPORT.html',studies,routes)
    code=sources(studies);code_guides(code,studies);folders=directories(studies);directory_document(folders,studies)
    js(DOC/'catalog.json',dict(schema_version=2,snapshot_at=now,snapshot_semantics='Dated evidence; local navigation only; no automatic job actions',
        convention='Ac225 denotes documented gamma proxy sources, not complete decay chains',cross_energy_image_sums_currently_authorized=False,
        experiments=studies,components=registry['components']))
    js(DOC/'source_index.json',dict(schema_version=2,snapshot_at=now,note='Files/definitions only; no module imports or execution',groups=dict(Counter(r['group'] for r in code)),files=code))
    js(DOC/'directory_index.json',dict(schema_version=2,snapshot_at=now,directories=folders,note='Shallow directory discovery; no array relocation or recursive runtime scan'))
    experiments_doc(studies)
    from ac225_navigation_documents import write_documents
    write_documents(ROOT,studies,routes,folders,now,put,js,md,link,status_zh)
    # Parent result index integrity is separate from unchanged per-study media maps.
    index_files=['Analysis_Plots/results/REPORT.html','Analysis_Plots/results/README.md','Analysis_Plots/results/data/current_results.json',
                 'Analysis_Plots/results/data/result_catalog.json','Analysis_Plots/results/data/source_routes.csv']
    js(ROOT/'Analysis_Plots/results/data/artifact_sha256.json',{p:sha(ROOT/p) for p in index_files})
    # Create the two audit targets before resolving links to them on a fresh
    # checkout; they are replaced below by the complete verification records.
    for target in [AUDIT/'link_validation.json',AUDIT/'organization_manifest.json']:
        if not target.exists():put(target,'{}\n')
    validation=link_check()
    js(AUDIT/'link_validation.json',validation)
    baseline=read(AUDIT/'baseline.json');changed=[p for p,h in baseline['protected_source_sha256'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if changed:raise RuntimeError('Protected original science code changed: '+str(changed))
    MANAGED.pop(rel(AUDIT/'organization_manifest.json'),None)
    js(AUDIT/'organization_manifest.json',dict(passed=validation['passed'],updated_at=now,study_count=len(studies),result_route_count=len(routes),
        source_count=len(code),directory_count=len(folders),managed_text_count=len(MANAGED)+1,protected_original_source_files=len(baseline['protected_source_sha256']),
        protected_original_source_SHA_unchanged=True,source_or_array_moves=0,scientific_submissions=0,cancelled_jobs=0,arrays_or_archives_deleted=0,
        navigation_files_sha256=MANAGED.copy(),link_validation=validation,
        curated_registry_sha256=sha(DOC/'study_registry.json'),
        navigation_source_sha256={p:sha(ROOT/p) for p in ['tools/research/refresh_ac225_navigation.py','tools/research/ac225_navigation_documents.py',
          'tools/research/prepare_ac225_registry_20261103.py','tools/research/verify_ac225_navigation.py',
          'tools/research/build_project_catalog.py','tools/research/build_reconstruction_results_index.py']},
        git_commit_or_push_performed=False))
    print(json.dumps(dict(studies=len(studies),routes=len(routes),sources=len(code),directories=len(folders),managed=len(MANAGED),link_validation=validation),ensure_ascii=False))
    if not validation['passed']:raise SystemExit(1)

if __name__=='__main__':main()
