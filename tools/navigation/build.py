"""Build offline navigation from reviewed registries; no scientific execution.

The archived renderer supplies formatting and AST-only source discovery. Its
main(), dated templates, registry bootstrap and AGENTS writer are never called.
"""
from __future__ import annotations
import ast
from collections import Counter
import csv
from datetime import datetime
from html import escape
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import warnings
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/ac225'
AUDIT=ROOT/'docs/organization/20261117'
MANAGED={}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def link(target,doc):return os.path.relpath(ROOT/target,Path(doc).parent).replace('\\','/')
def put(p,text):
    p=Path(p);name=rel(p)
    allowed=name.startswith(('docs/ac225/','Analysis_Plots/results/current/')) or name in {
        'Analysis_Plots/results/REPORT.html','Analysis_Plots/results/data/current_results.json',
        'Analysis_Plots/results/data/conditions.csv','Analysis_Plots/results/data/artifact_sha256.json','docs/organization/20261117/build_manifest.json'}
    if not allowed or p.suffix not in {'.md','.json','.csv','.html'} or p.name=='AGENTS.md':
        raise ValueError('Not a navigation output: '+name)
    raw=text.encode('utf8')
    if not p.exists() or p.read_bytes()!=raw:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    MANAGED[name]=sha(p)
def dump(p,v,compact=False):put(p,json.dumps(v,ensure_ascii=False,indent=None if compact else 2,allow_nan=False)+'\n')

def renderer():
    path=ROOT/'tools/legacy/ac225_navigation_20261103/refresh_ac225_navigation.py'
    spec=importlib.util.spec_from_file_location('_historical_navigation_renderer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT;module.DOC=DOC;module.AUDIT=AUDIT;module.put=put;module.js=dump
    return module

def pointers(routes):
    for row in routes:
        # Original pointer paths remain stable. New conditions use explicit IDs
        # so detector/grid/material/correction branches cannot collide.
        target=row.get('pointer') or 'Analysis_Plots/results/current/conditions/'+row['id']+'/README.md'
        row['pointer']=target;p=ROOT/target
        lines=['# '+row['id'],'','| 条件 | 值 |','|---|---|']
        for key in ['phantom','dose','energy_keV','method','grid','material','projection_source','correction','matrix','width_percent','views','iterations','job','status','delivery','numerical_qa','visual_qa','physical_certification']:
            lines.append('| '+key+' | '+str(row.get(key,'unknown')).replace('|','/')+' |')
        lines+=['',row.get('scope_note',''),'','## 来源','']
        for key in ['report','evidence','source']:
            value=row.get(key)
            if value and (ROOT/value).exists():lines.append('- ['+key+']('+link(value,p)+')')
        lines+=['','仅为条件指针，不复制数组；提交、计算完成、交付与物理结论分别判断。','',
                '[全部结果]('+link('Analysis_Plots/results/REPORT.html',p)+')','']
        put(p,'\n'.join(lines));dump(p.parent/'case.json',row)
    parent=ROOT/'Analysis_Plots/results/current/conditions/README.md'
    put(parent,'# 新增条件索引\n\n'+ '\n'.join('- ['+r['id']+']('+link(r['pointer'],parent)+')' for r in routes if '/conditions/' in r['pointer'])+'\n')

def results_html(routes):
    p=ROOT/'Analysis_Plots/results/REPORT.html'
    labels={'actual':'真实Geant4','synthetic':'前投影+Poisson','RUNNING':'运行中','FAILED':'失败',
      'completed-delivered-visualized':'计算/交付/图形完成','failed-preview-only':'失败；9950次预览',
      'completed-selected-delivery-visualized':'计算完成；选帧交付',
      'component-delivered-parent-failed':'本通道完成；父任务失败',
      'completed-delivered-awaiting-visualization':'已交付；新版图集待整理',
      'strict-delivered-visual-pending':'严格交付；全组视觉QA待完成',
      'passed':'通过','see-original-evidence':'见原证据','not-started':'未开始',
      'not-completed':'未完成','pending-full-32-review':'全32路待复核',
      'reviewed-preview':'预览已复核','native-consistency-only':'原生一致性检查',
      'see-dated-overview':'见专题报告','not-delivered':'未交付',
      'strict-200-frames':'完整200帧','strict-partial-199-frames':'部分199帧'}
    def zh(v):return labels.get(v,v)
    rows=[]
    for r in routes:
        raw=escape(json.dumps(r,ensure_ascii=False).casefold(),quote=True)
        cells=[zh(r['kind']),r['phantom']+' / '+r['dose'],str(r['energy_keV'])+' keV / '+r['method'],
               r['grid']+' / '+r['material']+' / '+r['correction'],
               str(r['views'])+'角 / '+str(r['iterations'])+'次 / '+str(r.get('width_percent',100))+'%',
               zh(r['status'])+'；交付：'+zh(r.get('delivery','见原证据'))+'；数值QA：'+zh(r['numerical_qa'])+'；视觉QA：'+zh(r['visual_qa'])]
        rows.append('<tr data-kind="'+r['kind']+'" data-search="'+raw+'">'+''.join('<td>'+escape(c)+'</td>' for c in cells)+'<td><a href="'+escape(link(r['pointer'],p))+'">条件/来源</a><br><a href="'+escape(link(r['report'],p))+'">报告/阶段</a></td></tr>')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>225Ac结果条件索引</title>
<style>body{font:15px 'Microsoft YaHei',sans-serif;margin:24px;color:#203442;background:#f4f7fa}a{color:#126078}p{line-height:1.7}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #d5dee6;vertical-align:top;overflow-wrap:anywhere}td:last-child,th:last-child{min-width:105px;white-space:nowrap}td:nth-child(6){max-width:400px;line-height:1.6}input,select{padding:10px;font:inherit}input{width:55%}.filters{position:sticky;top:0;background:#f4f7fa;padding:12px 0}[hidden]{display:none!important}.table{overflow-x:auto}th{background:#e7eef4}</style>
<h1>225Ac成像结果与未完成条件</h1><p><a href="../../docs/ac225/README.md">工程入口</a> · <a href="../../docs/ac225/CURRENT_STATUS.md">带时间的状态</a> · <a href="data/conditions.csv">条件CSV</a></p>
<p>SC＝单光子；JSCC＝同能量SC+Compton。HE、OpenGATE EHE、原加密EHE分别列出。gamma发射数不是225Ac衰变数。这里同时收录失败与运行条件，必须阅读状态列。物理认证均未由本索引授予。</p>
<div class="filters"><input id="q" aria-label="搜索条件" placeholder="组合查询：NEMA 5e9 Cartesian4mm / water attenuation / 4090 / 80"><select id="kind" aria-label="投影来源"><option value="">全部来源</option><option value="actual">真实Geant4</option><option value="synthetic">前投影+Poisson</option></select> <span id="count"></span></div>
<div class="table"><table><thead><tr><th>投影</th><th>体模/发射量</th><th>能量/方法</th><th>网格/材料/修正</th><th>视角/迭代/展宽</th><th>阶段</th><th>入口</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p>原研究、冻结、失败证据与大数组保持原路径；本页刷新不查询队列。SC/JSCC10000与HE/EHE200不视为同收敛；选帧交付不等于全历史交付。</p>
<script>const q=document.querySelector('#q'),k=document.querySelector('#kind'),rs=[...document.querySelectorAll('tbody tr')];function apply(){let n=0;const tokens=q.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);for(const r of rs){const ok=(!k.value||r.dataset.kind===k.value)&&tokens.every(t=>r.dataset.search.includes(t));r.hidden=!ok;n+=ok;}document.querySelector('#count').textContent=n+' 条';}q.addEventListener('input',apply);k.addEventListener('change',apply);apply();</script></html>'''
    put(p,content)

def statuses(studies,now):
    fields=['id','title','observed_at','compute_status','delivery_status','visual_status','jobs','evidence','physical_calibration_passed']
    dump(DOC/'status_snapshot.json',dict(schema_version=3,generated_at=now,note='Generation time is not evidence observation time. Unknown stages are not promoted.',experiments=[{k:s.get(k,'not-established') for k in fields} for s in studies]))
    # Replace the misleading old formal-job CURRENT file with dated identities.
    dump(DOC/'formal_jobs_current.json',dict(schema_version=3,generated_at=now,read_only=True,no_automatic_actions=True,experiments={s['id']:{k:s.get(k) for k in fields[2:]} for s in studies if s.get('jobs')}))
    selected=['nema-cartesian-5090','nema-cartesian-4090-r2','nema-width-sweep','opengate-he-derenzo','opengate-he-nema','water-sc-first','water-scatter-v2','derenzo-jscc-5e10-440-joint','derenzo-jscc-5e10-440-sc','derenzo-ehe-dense-5e10']
    by={s['id']:s for s in studies};p=DOC/'CURRENT_STATUS.md'
    lines=['# 当前证据状态','',
      '2026-11-17整理。最终调度器观察时间与只读原文见[最终远端证据](../organization/20261117/remote_final_readonly.json)，进度/错误另见[最终日志](../organization/20261117/remote_logs_final_readonly.json)。[最初运行快照](../organization/20261117/remote_readonly.json)保留状态演变。本页不是实时看板，生成时间不充当科学证据时间。','',
      '| 研究 / 证据日期 | 计算 | 严格交付 | 视觉QA | 事实与限制 |','|---|---|---|---|---|']
    for ident in selected:
        s=by[ident];lines.append('| ['+s['title']+']('+link(s['guide'],p)+')<br>'+s['observed_at']+' | '+s['compute_status']+' | '+s['delivery_status']+' | '+s['visual_status']+' | '+s['conclusion']+' |')
    lines+=['','## 阶段判据','',
      '提交只由唯一registration/intent证明；排队、运行以新鲜Slurm记录为准；计算完成要求root/batch/extern/step及完成manifest闭合；严格交付要求成员大小/SHA和完整范围；数值QA、视觉QA各有独立证据；物理结论只能按研究报告的论证范围成立。任何缺证阶段均保留unknown/pending，不能自动升级。','',
      '5090衔接PID18108在本次开始时PID、create_time、cmdline三项一致；最终两原控制器18108/22172均已按原失败策略退出，登记为STOPPED_FAILURE/FORMAL_FAILED。登记原文件未修改。[最初身份](../organization/20261117/protected_baseline.json) · [最终身份与冻结核验](../organization/20261117/protection_final.json)。另一个CRC40任务属于其他研究，本次未触碰。','',
      '原11月3–16日混合状态页保存在[历史快照](history/CURRENT_STATUS_before_20261117.md)。旧ACTIVE、未提交、排队等叙述只代表各自日期，已被上表证据更新的不得继续作为当前状态。旧控制器、被拒绝分支及冻结发布继续原位保留。','',
      '[完整研究清单](EXPERIMENTS.md) · [机器状态](status_snapshot.json) · [操作边界](OPERATIONS.md)','']
    put(p,'\n'.join(lines))

def main():
    MANAGED.clear();now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    old=renderer();studies=read(DOC/'study_registry.json')['studies'];routes=read(DOC/'result_registry.json')['routes']
    old.study_guides(studies,now)
    # The archived study renderer is augmented with explicit numeric QA/freeze
    # and command safety, rather than duplicating scientific conventions.
    for s in studies:
        p=ROOT/s['guide'];text=p.read_text(encoding='utf8')
        text+='\n## 复现与身份\n\n冻结：`'+str(s.get('freeze','见原freeze/release_manifest'))+'`。数值QA：`'+str(s.get('numerical_qa','见阶段证据，不推断'))+'`。\n\n本页命令只读：`python -B -X utf8 tools/project_index.py show '+s['id']+'`。实际计算需要既有环境、输入与独立授权；prepare/advance/submit/fetch/watch等入口会修改本地或远端状态，目录整理不运行它们。运行版本以冻结payload为准，不用同名工作区源码替代。\n'
        put(p,text)
    pointers(routes);results_html(routes)
    fields=['id','study_id','kind','phantom','dose','energy_keV','method','grid','material','projection_source','correction','width_percent','views','iterations','job','matrix','status','delivery','numerical_qa','visual_qa','physical_certification','report','evidence','pointer']
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(routes)
    put(ROOT/'Analysis_Plots/results/data/conditions.csv',stream.getvalue())
    dump(ROOT/'Analysis_Plots/results/data/current_results.json',dict(schema_version=3,generated_at=now,observed_at=read(DOC/'result_registry.json')['observed_at'],routes=routes))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SyntaxWarning)
        code=old.sources(studies)
    dependencies=[];hashgroups={}
    versioned=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf8').split('\0'))
    for r in code:
        p=ROOT/r['path'];r['bytes']=p.stat().st_size
        r['source_id']=hashlib.sha256(r['path'].encode()).hexdigest()[:12]
        r['git_availability']='in-index' if r['path'] in versioned else 'local-uncommitted; not guaranteed on clone'
        if r['bytes']<=1048576:
            digest=sha(p);hashgroups.setdefault(digest,[]).append(r['path'])
        if p.suffix=='.py' and r['bytes']<=1048576:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',SyntaxWarning)
                    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
                for node in ast.walk(tree):
                    if isinstance(node,ast.Import):
                        dependencies.extend(dict(source_id=r['source_id'],line=node.lineno,module=a.name,kind='import') for a in node.names)
                    elif isinstance(node,ast.ImportFrom):
                        dependencies.append(dict(source_id=r['source_id'],line=node.lineno,module='.'*node.level+(node.module or ''),kind='from'))
            except (SyntaxError,UnicodeError):pass
        if r['path'].startswith('tools/legacy/ac225_navigation_20261103/'):
            r['role']='retired-navigation-renderer'
        if '/studies/' in r['path']:r['role']='study-source-preserved; executed identity in freeze'
    old.code_guides(code,studies);folders=old.directories(studies);old.directory_document(folders,studies);old.experiments_doc(studies);old.html_index(DOC/'INDEX.html',studies)
    p=DOC/'SOURCE_INDEX.md'
    put(p,p.read_text(encoding='utf8')+'\n[Python导入关系](source_dependencies.csv)通过source_id连接[source_index.json](source_index.json)中的完整路径。git_availability标明本地尚未提交文件，克隆后可能缺失；索引不声称它们已交付到Git。完全相同字节的[重复源码组](duplicate_source_groups.json)只用于追溯，不授权合并或删除。\n')
    dump(DOC/'catalog.json',dict(schema_version=3,generated_at=now,snapshot_semantics='Per-study observation date; refresh is not a live query',experiments=studies,components=read(DOC/'study_registry.json')['components']))
    dump(DOC/'source_index.json',dict(schema_version=3,note='AST parsing only; never module import',groups=dict(Counter(r['group'] for r in code)),files=code),compact=True)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['source_id','line','module','kind']);writer.writeheader();writer.writerows(dependencies)
    put(DOC/'source_dependencies.csv',stream.getvalue())
    dump(DOC/'duplicate_source_groups.json',dict(semantics='Exact SHA copies only. Scientific implementations remain in place; copies may be frozen or externally referenced.',groups=[dict(sha256=k,paths=v) for k,v in hashgroups.items() if len(v)>1]))
    dump(DOC/'directory_index.json',dict(schema_version=3,directories=folders,note='Shallow metadata discovery, no recursive TB data content scan'))
    statuses(studies,now)
    index_artifacts=['Analysis_Plots/results/REPORT.html','Analysis_Plots/results/README.md',
                     'Analysis_Plots/results/data/current_results.json','Analysis_Plots/results/data/conditions.csv',
                     'Analysis_Plots/results/data/result_catalog.json']
    dump(ROOT/'Analysis_Plots/results/data/artifact_sha256.json',{p:sha(ROOT/p) for p in index_artifacts})
    outputs=MANAGED.copy()
    dump(AUDIT/'build_manifest.json',dict(studies=len(studies),condition_routes=len(routes),sources=len(code),directories=len(folders),generated_at=now,scientific_execution=False,AGENTS_written=False,inputs_sha256={p:sha(ROOT/p) for p in ['docs/ac225/study_registry.json','docs/ac225/result_registry.json']},outputs_sha256=outputs))
    print(json.dumps(dict(studies=len(studies),routes=len(routes),sources=len(code),directories=len(folders),outputs=len(MANAGED))))

if __name__=='__main__':main()
