"""User-requested completed-branch preview during the remaining 60% solve.

Remote science is read only. Suspend only the identity-checked local waiting
completion process during transfer, then resume that same process in finally.
No full-job delivery is declared until the original completion pipeline exits.
"""
import argparse,datetime,hashlib,json,os,shutil,sys
from pathlib import Path
import numpy as np
import psutil
from prepare import CODE,DATA,REPORT,ROOT,EXP,read,write,sha
from workflow import connect,command,script,now,REMOTE
import compare_results as plots

SELECTED=(100,500,1000,2000,5000,10000)
TARGET=DATA/'preview_100_80_1814153'
OUT=ROOT/'Analysis_Plots/research/nema5e9_width100_vs80_preview_20261115'
RECEIPT=REPORT/'preview_100_80_fetch.json'

def fetch():
    if RECEIPT.exists():
        receipt=read(RECEIPT)
        for name,s in receipt['downloaded_files'].items():assert sha(TARGET/name)==s,name
        print('REUSE_VERIFIED_PREVIEW');return receipt
    reg=read(REPORT/'formal_registration_resume_r5.json')
    local=read(DATA/'completion_once_r5_registration.json');proc=psutil.Process(local['pid'])
    assert abs(proc.create_time()-local['create_time'])<.01 and proc.cmdline()==local['cmdline']
    workers=[p for p in proc.children(recursive=True) if p.name().lower()!='conhost.exe']
    assert local['status']=='ACTIVE' and not workers,'Completion process is already transferring/plotting'
    plan=dict(time=now(),request='80%完成后先展示100%和80%对比',job=reg['job'],local_pid=local['pid'],
              create_time=local['create_time'],cmdline=local['cmdline'],remote_job_touched=False)
    handoff=REPORT/'preview_100_80_handoff.json';write(handoff,plan)
    proc.suspend()
    try:
        with connect() as c:
            accounting=command(c,'sacct -n -P -j '+reg['job']+' --format=JobID,State,ExitCode,Elapsed,AllocTRES',45)
            code='''import pathlib,json,hashlib
p=pathlib.Path(RESULT);r=pathlib.Path(RELEASE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
release=json.loads((r/'release_manifest.json').read_text());assert release['key']==KEY
for n,s in release['files'].items():assert sha(r/n)==s,n
files={};sizes={};branch_manifests={};remote_verified={}
for pct in (100,80):
 b=p/f'width{pct}';m=json.loads((b/'completed.json').read_text())
 assert (m['iterations'],m['save_step'],m['frames'],m['release'])==(10000,50,200,KEY)
 assert len(list((b/'checkpoints').glob('*.float32')))==200
 for n,s in m['files'].items():assert sha(b/n)==s,n
 cp=json.loads((b/'checkpoint_manifest.json').read_text());assert cp['iteration']==10000 and len(cp['sha256'])==200
 for n,s in cp['sha256'].items():assert m['files'][n]==s,n
 names=['completed.json','checkpoint_manifest.json','S_SC_active.float32','S_Compton_active.float32','S_Compton_full.float32','Image_JSCC_active.float32','Image_JSCC_full.float32']
 names += [f'rank{i}.json' for i in range(8)]+[f'numerical_rank{i}.json' for i in range(8)]
 names += [f'checkpoints/iteration_{i:05d}.float32' for i in SELECTED]
 names += ['S_provenance.json'] if pct==80 else ['original_regression_00050.json','original_regression_10000.json','benchmark_choice.json']
 for n in names:
  name=f'width{pct}/'+n;files[name]=sha(b/n);sizes[name]=(b/n).stat().st_size
 branch_manifests[str(pct)]=m;remote_verified[str(pct)]=len(m['files'])
files['resume_lineage.json']=sha(p/'resume_lineage.json');sizes['resume_lineage.json']=(p/'resume_lineage.json').stat().st_size
print(json.dumps(dict(files=files,bytes=sizes,release=release,remote_all_completed_members_verified=remote_verified,branch_manifests=branch_manifests)))
'''.replace('RESULT',repr(reg['result'])).replace('RELEASE',repr(reg['release'])).replace('KEY',repr(reg['execution_key'])).replace('SELECTED',repr(SELECTED))
            remote=json.loads(script(c,code,120));assert remote['release']==read(DATA/'execution_manifest_resume_r5.json')
            TARGET.mkdir(exist_ok=True)
            with c.open_sftp() as s:
                for index,(name,digest) in enumerate(remote['files'].items()):
                    dest=TARGET/name;dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():assert sha(dest)==digest,name;continue
                    partial=dest.with_name(dest.name+'.partial')
                    s.get(reg['result']+'/'+name,str(partial),prefetch=True,max_concurrent_prefetch_requests=8)
                    assert partial.stat().st_size==remote['bytes'][name] and sha(partial)==digest,name
                    partial.replace(dest)
                    if index%10==0:print('PREVIEW_FETCH',index+1,len(remote['files']),flush=True)
                # Re-read completed fences after transfer; never preview changing output.
                for pct in (100,80):
                    name=f'width{pct}/completed.json'
                    with s.open(reg['result']+'/'+name,'rb') as f:raw=f.read()
                    assert hashlib.sha256(raw).hexdigest()==remote['files'][name]
            receipt=dict(time=now(),job=reg['job'],execution_key=reg['execution_key'],accounting_at_preview=accounting,
                downloaded_files=remote['files'],downloaded_bytes=sum(remote['bytes'].values()),
                remote_all_completed_members_verified=remote['remote_all_completed_members_verified'],
                selected_iterations=list(SELECTED),completed_branch_frames=400,downloaded_checkpoint_frames=12,
                full_job_exit_accepted=False,full_three_branch_delivery=False,scope='Completed 100%/80% preview only; 60% continues')
            validate(receipt);write(RECEIPT,receipt)
            return receipt
    finally:
        if proc.is_running() and abs(proc.create_time()-local['create_time'])<.01:proc.resume()
        plan.update(resumed=now(),same_local_process_resumed=proc.is_running());write(handoff,plan)

def validate(receipt):
    frozen=read(REPORT/'freeze_r3.json');geo=np.load(DATA/frozen['payload_dir']/'whole_geometry.npz');active=geo['active_indices'];ss0=None
    for pct in (100,80):
        folder=TARGET/f'width{pct}';m=read(folder/'completed.json');cp=read(folder/'checkpoint_manifest.json')
        assert m['release']==receipt['execution_key'] and m['frames']==200 and cp['iteration']==10000
        for n,digest in receipt['downloaded_files'].items():
            if n.startswith(f'width{pct}/'):
                leaf=n.split('/',1)[1]
                if leaf!='completed.json':assert m['files'][leaf]==digest
                assert sha(TARGET/n)==digest
        for i in SELECTED:
            n=f'checkpoints/iteration_{i:05d}.float32';v=np.fromfile(folder/n,'<f4')
            assert v.shape==(78920,) and np.isfinite(v).all() and (v>=0).all() and sha(folder/n)==cp['sha256'][n]
        final=np.fromfile(folder/'Image_JSCC_active.float32','<f4');full=np.fromfile(folder/'Image_JSCC_full.float32','<f4')
        assert np.array_equal(final,np.fromfile(folder/'checkpoints/iteration_10000.float32','<f4'))
        assert full.shape==(132040,) and np.array_equal(full[active],final)
        mask=np.ones(132040,bool);mask[active]=False;assert np.count_nonzero(full[mask])==0
        ss=np.fromfile(folder/'S_SC_active.float32','<f4');sc=np.fromfile(folder/'S_Compton_active.float32','<f4');sf=np.fromfile(folder/'S_Compton_full.float32','<f4')
        assert ss.shape==sc.shape==(78920,) and sf.shape==(132040,) and np.isfinite(sf).all() and (sf>=0).all()
        assert np.array_equal(sc,sf[active]*geo['ellipse_fraction'][active].astype(np.float32))
        if ss0 is None:ss0=ss
        else:assert np.array_equal(ss,ss0)
        if pct==100:
            assert sha(folder/'S_Compton_full.float32')==frozen['files']['continuous_energy_Sensi_full']
            lineage=read(TARGET/'resume_lineage.json');assert lineage==read(REPORT/'resume_plan_r5.json')
            for i in SELECTED[:-1]:
                n=f'checkpoints/iteration_{i:05d}.float32';assert sha(folder/n)==lineage['parent_files']['width100/'+n]
        else:
            prov=read(folder/'S_provenance.json');assert prov['scale']==.8 and prov['events']==170247 and prov['primary_gamma_440']==1000000000 and abs(prov['integral_relative_error'])<=1e-5
        ranks=[]
        for rank in range(8):
            r=read(folder/f'rank{rank}.json');g=r['resources'];ranks.append(r)
            assert r['rank']==rank and r['scale']==pct/100
            assert g['rss_peak_bytes']<=.8*g['host_allocated_bytes'] and g['gpu_reserved_peak_bytes']<=.8*g['gpu_total_bytes']
            n=read(folder/f'numerical_rank{rank}.json');assert n['chunk_relative_L2']<=1e-5 and max(n['adjoint_errors'])<=1e-5
        assert len({r['node'] for r in ranks})==8
        for view,expected in enumerate(frozen['study']['events_per_view']):
            chunks=[r['partitions'][view] for r in ranks]
            assert chunks[0]['first']==0 and chunks[-1]['last']==expected and sum(x['events'] for x in chunks)==expected
            assert all(a['last']==b['first'] for a,b in zip(chunks,chunks[1:]))

def render():
    receipt=read(RECEIPT);validate(receipt);c=plots.context();bg=[];spheres=[];mips={}
    for name in ('data','figures','scripts'):(OUT/name).mkdir(parents=True,exist_ok=True)
    for pct in (100,80):
        for i in SELECTED:
            v=np.fromfile(TARGET/f'width{pct}/checkpoints/iteration_{i:05d}.float32','<f4')
            b,s,m=plots.analyze(v,c,pct,i);bg.append(b);spheres.extend(s);mips[pct,i]=m
    import matplotlib.pyplot as plt
    figures=[]
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=190,facecolor='white')
        plt.close(fig);figures.append(name)
    truth=c['truth']['activity_440_zyx'][c['slab']].max(0);maximum=max(10.,max(float(x.max()) for x in mips.values()))
    for suffix,limit in [('',10.),('_unclipped',maximum)]:
        fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
        plots.draw(axes[0],truth,c,'NEMA truth',limit)
        for ax,pct in zip(axes[1:],(100,80)):h=plots.draw(ax,mips[pct,10000],c,f'{pct}% width · iteration 10000',limit)
        fig.colorbar(h,ax=axes.tolist(),label='Relative activity density',shrink=.74)
        fig.suptitle('NEMA 5e9 · 440 keV JSCC · central 72 mm MIP\nIdentical events · matched sensitivity · no smoothing',fontsize=13)
        save(fig,'final_100_vs_80'+suffix)
        fig,axes=plt.subplots(2,7,figsize=(20,6.4),layout='constrained')
        for row,pct in enumerate((100,80)):
            plots.draw(axes[row,0],truth,c,f'{pct}% · truth',limit)
            for ax,i in zip(axes[row,1:],SELECTED):h=plots.draw(ax,mips[pct,i],c,str(i),limit)
        fig.colorbar(h,ax=axes.ravel().tolist(),label='Relative activity density',shrink=.7)
        fig.suptitle('JSCC 100% vs 80% width · actual saved iterations · central 72 mm MIP',fontsize=14)
        save(fig,'iterations_100_vs_80'+suffix)
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.6),layout='constrained')
    for col,d in enumerate((13,22,37)):
        for pct in (100,80):
            rows=[s for s in spheres if s['width_percent']==pct and s['diameter_mm']==d]
            for row,key in enumerate(('CRC_percent','CNR')):
                ax=axes[row,col];ax.plot([x['BV_percent'] for x in rows],[x[key] for x in rows],'-o',label=f'{pct}%',color=plots.COLORS[pct],markersize=4)
                ax.scatter(rows[-1]['BV_percent'],rows[-1][key],s=65,marker='s',color=plots.COLORS[pct])
                ax.set(xlabel='BV (%)',ylabel='CRC (%)' if row==0 else 'CNR',title=f'{d} mm hot sphere');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Six saved samples: 100, 500, 1000, 2000, 5000, 10000\nSquares = final; connecting lines are guides (not all 200 frames)',fontsize=12)
    save(fig,'sampled_crc_cnr_bv_100_vs_80')
    plots.csvwrite(OUT/'data/background_metrics.csv',bg);plots.csvwrite(OUT/'data/sphere_metrics.csv',spheres)
    endpoints=[s for s in spheres if s['iteration']==10000];plots.csvwrite(OUT/'data/final_sphere_metrics.csv',endpoints)
    np.savez_compressed(OUT/'data/display_arrays.npz',truth=truth,**{f'width{pct}_iter{i}':v for (pct,i),v in mips.items()})
    sources=plots.sources();sources.update({str(TARGET/n):v for n,v in receipt['downloaded_files'].items()});sources[str(Path(__file__))]=sha(Path(__file__));sources[str(CODE/'compare_results.py')]=sha(CODE/'compare_results.py')
    write(OUT/'data/input_sha256.json',sources);shutil.copyfile(Path(__file__),OUT/'scripts'/Path(__file__).name);shutil.copyfile(CODE/'compare_results.py',OUT/'scripts'/'compare_results_reference.py')
    metadata=dict(job=receipt['job'],widths=[100,80],selected_iterations=list(SELECTED),selected_frames=12,
        completed_branch_frames=400,preview_only=True,full_job_delivery=False,full_60_percent_result_included=False,
        center_MIP_mm=72,full_XY=True,normalization_density=c['norm'],no_fitted_gain=True,no_smoothing=True,
        ROI_policy=plots.POLICY_ID,background_voxels=97604,figures=figures,main_vmax=10.,shared_unclipped_vmax=maximum,
        numerical_QA_passed=True,visual_QA_passed=False,physical_calibration_passed=False)
    write(OUT/'data/metadata.json',metadata)
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>NEMA JSCC 100%与80%对比</title><style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;line-height:1.6;color:#223}img{width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>NEMA 5e9 · 440 keV JSCC：100%与80%宽度</h1>'
    page+='<p>两组均完成10000次。复用同一批真实事件和筛选；80%统一缩放电子学、位置和材料残差宽度，保留材料均值，并使用独立匹配的sensitivity。60%仍在同一作业中运行，本页是已完成分支预览，不等同三档完整验收。</p><p>中心72 mm MIP、完整XY、gray_r；无平滑、无亮度拟合。采用原3D球心ROI与97604体素共同背景。物理单位用同一原初级gamma数归一化，不按图像亮度调整。主图色标0–10，另有共同全幅色标图；5e9是双能gamma总发射量。</p>'
    page+='<h2>最终指标</h2><table><tr><th>宽度</th><th>热球直径</th><th>CRC</th><th>CNR</th><th>BV</th></tr>'
    for s in endpoints:
        if s['region_kind']=='hot':page+=f"<tr><td>{s['width_percent']}%</td><td>{s['diameter_mm']} mm</td><td>{s['CRC_percent']:.2f}%</td><td>{s['CNR']:.3f}</td><td>{s['BV_percent']:.2f}%</td></tr>"
    page+='</table>'
    for name in figures:page+=f'<h2>{name}</h2><a href="figures/{name}.png"><img src="figures/{name}.png"></a>'
    page+='<p>曲线仅包含明确列出的6个真实保存迭代点，方块为10000次；连线仅为导引，不替代完整200帧曲线。球指标在3D图像计算，不在MIP上计算。不同响应模型的同迭代对比不证明同等收敛或真实物理分辨率改善。</p><p><a href="data/background_metrics.csv">背景指标</a> · <a href="data/sphere_metrics.csv">球指标</a> · <a href="data/metadata.json">元数据</a></p></html>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    write(OUT/'data/artifact_sha256.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'})
    print(json.dumps(dict(report=str(OUT/'index.html'),background_final=[b for b in bg if b['iteration']==10000],hot_final=[s for s in endpoints if s['region_kind']=='hot']),ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fetch','render']);a=p.parse_args()
    if a.action=='fetch':fetch()
    else:render()
