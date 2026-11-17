"""Freeze side responses and 32 single-photon routes; formal-only submissions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from ac225_corrections.io import digest,read,write,verify,hashes,immutable
from ac225_corrections.material import WaterNEMA
from ac225_corrections.windows import energy_windows
from ac225_corrections.surface_audit import reference_case,summarize
from ac225_water_workflow_20261103 import DATA,REPORT,TAG,CPU,GPU,prepare,submit_registered
from campaign_20261102_io import execute,upload_files,python_json,sftp
from derenzo_campaign_common import HERE,ROOT,GPU_PROJECT,GPU_PYTHON
from ehe_5e9_workflow import q
from ac225_transfer_io_20261104 import download_serial,sftp_bounded,download_cpu
from campaign_reliable_io_20261104 import python_json as reliable_python_json,cpu_member,accounting as reliable_accounting


def prepare_gpu():
    marker=REPORT/'kernel_freeze.json';payload=DATA/'kernel_payload'
    if marker.exists():
        f=read(marker);verify(payload,f['files']);return f
    parent=prepare();payload.mkdir(exist_ok=False)
    for n,p in hashes(HERE/'ac225_corrections').items():
        target=payload/'ac225_corrections'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/'ac225_corrections'/n,target)
    for n in ('whole_geometry.npz','contract.json','nema_body_h60_config.json','torch_active_operator.py','single_checkpoint_mlem.py','ehe_common.py'):
        shutil.copyfile(DATA/'payload'/n,payload/n)
    for n in ('test_ac225_corrections.py','make_nema_body_h60.py'):shutil.copyfile(HERE/n,payload/n)
    params_root=ROOT/'Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/FileGenerater_3D_Unified/output'
    kernel=read(HERE/'generated/jscc_geant4_5e10_10000/production_payload/kernel_config.json')
    dense=read(HERE/'generated/ehe_dense_reruns_20261102/reconstruction_payload/contract.json')['factors']
    factors={'JSCC':{},'EHE':dense};parents={}
    for e,name in ((218,'218keV_RotateNum20'),(440,'440keV_RotateNum20'),('C440to218','440keV_to218win_RotateNum20')):
        key='A'+str(e) if isinstance(e,int) else e
        files={n[len(name)+1:]:h for n,h in kernel['factor_payload_sha256'].items() if n.startswith(name+'/')}
        factors['JSCC'][key]=dict(root=GPU_PROJECT+'/generated/FactorsCalibrated/'+name,
            manifest_sha256=kernel['factor_manifest_sha256'][name],files=files)
    # Only small original manifests/coordinate tables fetched; complete matrix SHA
    # is checked in the new formal process, rather than downloading old matrices.
    with sftp('gpu') as s:
        for system in ('JSCC','EHE'):
            for key,spec in factors[system].items():
                target=payload/'parent_evidence'/system/key;target.mkdir(parents=True)
                p=target/'factor_manifest.json';s.get(spec['root']+'/factor_manifest.json',str(p))
                if digest(p)!=spec['manifest_sha256']:raise ValueError('Parent manifest changed')
                parents[system+'/'+key]=read(p)
            if system=='JSCC':
                p=payload/'parent_evidence/JSCC/Detector.csv';s.get(factors[system]['A440']['root']+'/Detector.csv',str(p))
                if digest(p)!=factors[system]['A440']['files']['Detector.csv']:raise ValueError('Physical detector table differs')
    audit_cases=[];tasks={}
    for system in ('JSCC','EHE'):
        for energy in (218,440):
            target=payload/'params'/system/str(energy);target.mkdir(parents=True)
            source=(HERE/'generated/ehe_dense_surface_20261029/payload/params'/('A'+str(energy)) if system=='EHE'
                    else params_root/('JSCC_'+str(energy)+'keV'))
            for p in source.glob('Params_*.dat'):shutil.copyfile(p,target/p.name)
            np.array([85,85,40,6,6,3,1,0,0,0,0,323.75 if system=='EHE' else 270],'<f4').tofile(target/'Params_Image.dat')
            params={n:np.fromfile(target/('Params_'+n.title()+'.dat'),'<f4') for n in ('image','detector','collimator','physics')}
            records=params['detector'][1:].reshape(-1,12);sel=np.flatnonzero(records[:,11]==1)
            if len(sel)!=(2312 if system=='EHE' else 10496):raise ValueError('Original detector count required')
            scale=np.ones(len(sel),np.float64)
            if system=='JSCC':
                m=parents['JSCC/A'+str(energy)]
                if digest(target/'Params_Detector.dat')!=m['input_sha256']['params_detector']:raise ValueError('Baseline detector/material params differ')
                y=records[sel,1]+270
                for r in m['calibration']['metrics']:scale[np.isclose(y,r['layer_mm'])]=r['scale']
                if np.any(scale==1):raise ValueError('Every original calibration layer must be bound')
            np.save(target/'frozen_layer_scale.npy',scale)
            # Readable physical topology; layers have different pitch/sparsity.
            xyz=records[sel,:3].astype(float);xyz[:,1]+=float(params['image'][11])
            layout=[]
            for i,p in enumerate(xyz):
                layer=int(np.flatnonzero(np.isclose(np.unique(xyz[:,1]),p[1]))[0])
                group=xyz[np.isclose(xyz[:,1],p[1])]
                u=np.unique(group[:,0]);v=np.unique(group[:,2])
                # Integer physical lattice coordinates preserve missing positions.
                dx=np.diff(u);dz=np.diff(v)
                pitchx=float(dx.min()) if len(dx) else 1.;pitchz=float(dz.min()) if len(dz) else 1.
                ux=(p[0]-u[0])/pitchx;vz=(p[2]-v[0])/pitchz
                if abs(ux-round(ux))>1e-3 or abs(vz-round(vz))>1e-3:raise ValueError('Irregular physical lattice')
                layout.append([0,layer,round(ux),round(vz)])
            np.save(target/'detectors_mm.npy',xyz.astype('<f4'));np.save(target/'physical_layout.npy',np.array(layout,np.int64))
            for slab in range(4):
                i=len(tasks);tasks[str(i)]=dict(system=system,emission_keV=energy,slab=slab,
                    PE_parent_root=GPU_PROJECT+'/generated/ehe_dense_surface_20261029/formal_1749085/A'+str(energy)+'/slab_'+str(slab) if system=='EHE' else None)
            # Independent face integral checks run locally, no validation job.
            center=int(sel[np.argmin(np.sum(xyz[:,[0,2]]**2,axis=1))])
            edge=int(sel[np.argmax(xyz[:,2])])
            material=WaterNEMA(read(payload/'nema_body_h60_config.json'))
            for ident,src,view in [(center,[0,0,-1.5],0),(center,[138,-35,-1.5],5),
                                   (edge,[-108,-45,25.5],3),(center,[0,0,58.5],7)]:
                audit_cases.append(reference_case(system,params,ident,src,energy,view,material,n=64))
    audits={}
    for system in ('JSCC','EHE'):
        for energy in (218,440):audits[system+'_'+str(energy)]=summarize([x for x in audit_cases if x['system']==system and x['emission_keV']==energy])
    write(payload/'center_ray_audit.json',dict(passed=all(x['passed'] for x in audits.values()),groups=audits,
        full_instrument_scatter_certificate=False,representative_only=True))
    windows={n:v for n,v in energy_windows().items() if not n.endswith('main')}
    binaries=read(HERE/'generated/ehe_dense_surface_20261029/parent_gpu_binary_manifest.json')['files']
    side=dict(study=TAG,tasks=tasks,side_windows_keV=windows,
        binary_root=GPU_PROJECT+'/generated/ehe_spect_5e9_200/releases/74e129c4460163c5',binary_files=binaries,
        minimum_available_bytes=110*(1<<30),side_response_storage_peak_bound_bytes=100*(1<<30),
        maximum_concurrent_GPU_tasks=4,original_factors=factors,
        original_calibration_preserved_no_new_gain_fit=True,source_energy_for_mu=True,
        known_original_C_bias_not_absorbed_as_calibration=True,physical_calibration_passed=False,
        transport_freeze_key=parent['key'],transport_contract_sha256=digest(DATA/'payload/contract.json'))
    write(payload/'side_contract.json',side)
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(payload/'release_manifest.json',dict(key=key,files=files))
    f=dict(key=key,release=GPU+'/kernel_releases/'+key,files=hashes(payload),transport_freeze_key=parent['key'])
    write(marker,f);return f


def deploy_gpu():
    f=prepare_gpu();marker=REPORT/'kernel_deployment.json'
    if marker.exists():return read(marker)
    upload_files('gpu',f['release'],{n:DATA/'kernel_payload'/n for n in f['files']})
    execute('gpu','cd '+q(f['release'])+' && '+q(GPU_PYTHON)+' -B -m unittest test_ac225_corrections -v > '+q(f['release']+'/local_checks.log')+' 2>&1',timeout=180)
    proof=dict(passed=True,key=f['key'],release=f['release'],local_Linux_numerical_checks=True)
    write(marker,proof);return proof


def submit_side_responses():
    f=prepare_gpu();deploy_gpu()
    if not read(DATA/'kernel_payload/center_ray_audit.json')['passed']:raise ValueError('Center ray approximation failed; no masked model change')
    code='''import os,sys,json,pathlib
p=json.loads(sys.argv[1])['path'];pathlib.Path(p+'/logs').mkdir(parents=True,exist_ok=True);s=os.statvfs(p)
print('CAMPAIGN_JSON '+json.dumps(dict(available=s.f_bavail*s.f_frsize,total=s.f_blocks*s.f_frsize,writable=os.access(p,os.W_OK))))
'''
    quota=python_json('gpu',GPU_PYTHON,code,dict(path=GPU))
    if not quota['writable'] or quota['available']<125*(1<<30):raise OSError('Side responses need actual directory headroom plus 20% reserve')
    body='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\nexport OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\ncd '+q(f['release'])+'\n'+q(GPU_PYTHON)+' -B -m ac225_corrections.side_response --release '+q(f['release'])+' --output '+q(GPU+'/side_responses')+' --index "$SLURM_ARRAY_TASK_ID"\n'
    return submit_registered('gpu','ACW_side_response',body,'-p gpu_5090 --account=scxi717 --qos=gpugpu -N1 -n1 -c8 --gres=gpu:1 --array=0-15%4 --time=4320',GPU+'/side_responses',GPU)


def freeze_execution():
    """Independent execution supplement; preserves published transport/kernel."""
    old=prepare_gpu();marker=REPORT/'execution_freeze.json'
    if marker.exists():
        f=read(marker);verify(DATA/'execution_payload',f['files']);return f
    payload=DATA/'execution_payload';shutil.copytree(DATA/'kernel_payload',payload)
    # New, unpublished derived release: parent copies remain byte-identical.
    (payload/'release_manifest.json').unlink()
    for n,p in hashes(HERE/'ac225_corrections').items():
        target=payload/'ac225_corrections'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/'ac225_corrections'/n,target)
    shutil.copyfile(HERE/'test_ac225_corrections.py',payload/'test_ac225_corrections.py')
    from ac225_corrections.windows import upward_leakage_bound
    bound={s:upward_leakage_bound(50_000_000_000,n) for s,n in [('JSCC',10496),('EHE',2312)]}
    if max(bound.values())>=.001:raise ValueError('Cannot neglect 218->440 lower-window leakage')
    write(payload/'execution_contract.json',dict(parent_kernel_key=old['key'],
        parent_kernel_manifest_sha256=digest(DATA/'kernel_payload/release_manifest.json'),
        negligible_218_to_440_lower_expected_count_bound_at_full_5e10=bound,
        bound_assumes_all_primaries_are218_and_all_bins_may_count=True,
        ief_440_excludes_218_only_after_analytic_bound=True,
        formal_routes=32,formal440_elements=16,formal218_elements=16,
        maximum_concurrent_GPU_routes=4,physical_calibration_passed=False))
    files=hashes(payload);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    write(payload/'release_manifest.json',dict(key=key,files=files))
    f=dict(key=key,release=GPU+'/execution_releases/'+key,files=hashes(payload),parent_kernel_key=old['key'])
    write(marker,f);return f


def active_side_job():
    if (REPORT/'side_effective_jobs.json').exists():
        effective=read(REPORT/'side_effective_jobs.json')
        return dict(job=effective['jobs'][-1]['job'],output=effective['output'],effective_jobs=effective['jobs'])
    repaired=REPORT/'jobs/ACW_side_response_r2.json'
    return read(repaired if repaired.exists() else REPORT/'jobs/ACW_side_response.json')


def repair_pending_side_launch():
    """Bounded launch-only change before science starts; frozen model unchanged."""
    old=read(REPORT/'jobs/ACW_side_response.json');j=str(old['job'])
    marker=REPORT/'side_pending_launch_repair.json'
    if not marker.exists():
        queue=execute('gpu','squeue -h -r -j '+j+' -o "%i|%T"',readonly=True)
        intent=REPORT/'side_pending_cancel_intent.json'
        if not intent.exists():
            rows=[r.split('|') for r in queue.splitlines()]
            if len(rows)!=16 or any(r[1]!='PENDING' for r in rows):raise RuntimeError('Launch-only replacement requires all original elements still pending')
            write(intent,dict(job=old['job'],queue_before=queue,
                  reason='Use exact array-element AllocTRES, preserving all frozen scientific code'))
            execute('gpu','scancel '+j,readonly=False)
        elif read(intent)['job']!=old['job'] or queue.strip():
            raise RuntimeError('Unresolved cancellation intent; do not repeat mutation')
        for _ in range(30):
            if not execute('gpu','squeue -h -j '+j,readonly=True).strip():break
            time.sleep(1)
        else:raise RuntimeError('Original side array not fully exited')
        accounting=execute('gpu','sacct -n -P --array -j '+j+' -o JobID,State,ExitCode',readonly=True)
        roots=[r.split('|') for r in accounting.splitlines() if r.split('|')[0] in [j+'_'+str(i) for i in range(16)]]
        if len(roots)!=16 or any(not r[1].startswith('CANCELLED') for r in roots):raise RuntimeError('Confirm original pending request cancelled before replacement')
        write(marker,dict(original_job=old['job'],accounting=accounting,original_scientific_release_preserved=True,
              output_root_new=GPU+'/side_responses_r2',change='Only exact array-element memory identity',no_other_job_cancelled=True))
    f=prepare_gpu();module=HERE/'ac225_corrections/resources.py';sha=digest(module)
    root=GPU+'/launch_resources/'+sha[:16];upload_files('gpu',root,{'resources.py':module})
    code='''import sys,os,pathlib,hashlib,importlib.util
r=pathlib.Path(sys.argv[1]);p=pathlib.Path(sys.argv[2]);sys.path.insert(0,str(r))
assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[3]
spec=importlib.util.spec_from_file_location('bounded_array_resources',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import ehe_common;ehe_common.allocation=m.allocation
from ac225_corrections.side_response import run
run(r,pathlib.Path(sys.argv[4]),int(os.environ['SLURM_ARRAY_TASK_ID']))
'''
    body='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\nexport OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
    body+=q(GPU_PYTHON)+' -B -c '+q(code)+' '+q(f['release'])+' '+q(root+'/resources.py')+' '+q(sha)+' '+q(GPU+'/side_responses_r2')+'\n'
    return submit_registered('gpu','ACW_side_response_r2',body,'-p gpu_5090 --account=scxi717 --qos=gpugpu -N1 -n1 -c8 --gres=gpu:1 --array=0-15%4 --time=4320',GPU+'/side_responses_r2',GPU)


def transfer_inputs():
    """Transfer only new water observations/QA. Old/List scientific data stay put."""
    result={};contract=read(DATA/'payload/contract.json')
    for dataset,spec in contract['acquisitions'].items():
        marker=REPORT/(dataset+'_input_transfer.json')
        if marker.exists():result[dataset]=read(marker);continue
        effective=REPORT/'effective_collections.json'
        collect=read(effective)['datasets'][dataset] if effective.exists() else read(REPORT/'jobs'/('ACW_'+dataset+'_collect.json'))
        rows=[r for r in reliable_accounting('maty',[collect['job']],'python3') if r[0]==str(collect['job'])]
        if len(rows)!=1 or rows[0][1:3]!=['COMPLETED','0:0']:
            result[dataset]=dict(ready=False,reason='new water transport/collection not complete');continue
        remote=collect['output'];local=DATA/'water_inputs'/dataset;local.mkdir(parents=True,exist_ok=True)
        code='''import sys,json,pathlib,hashlib
p=pathlib.Path(json.loads(sys.argv[1])['root'])/'collection.json'
print('CAMPAIGN_JSON '+json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),collection=json.loads(p.read_text()))))
'''
        meta=reliable_python_json('maty','python3',code,dict(root=remote))
        collection=meta['collection']
        if collection['total_gamma']!=spec['total_gamma'] or not collection['actual_Geant4']:raise ValueError('New actual water dataset identity differs')
        members=dict(collection['files'],**{'collection.json':meta['sha256']})
        for n,h in members.items():
            p=local/n
            if p.exists():
                if digest(p)!=h:raise FileExistsError('Existing transfer mismatch preserved')
                continue
            cpu_member(remote+'/'+n,p,h)
        verify(local,members)
        ready=dict(passed=True,dataset=dataset,collection_sha256=meta['sha256'],all_transferred_members_SHA=members,
                   actual_gamma=spec['total_gamma'],all_worker_and_tag_closure=True)
        write(local/'gpu_input_ready.json',ready)
        destination=GPU+'/inputs/'+dataset
        upload_files('gpu',destination,{n:local/n for n in hashes(local)})
        code='''import sys,json,pathlib,hashlib
c=json.loads(sys.argv[1]);p=pathlib.Path(c['root'])
for n,h in c['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True)))
'''
        proof=python_json('gpu',GPU_PYTHON,code,dict(root=destination,files=hashes(local)))
        ready.update(GPU_root=destination,GPU_verification=proof);write(marker,ready);result[dataset]=ready
    return result


def submit_formal():
    if (REPORT/'formal_submitted.json').exists():return read(REPORT/'formal_submitted.json')
    gate=read(REPORT/'center_full_response_audit.json')
    if (not gate['passed'] or gate['source_sha256']!=digest(HERE/'ac225_corrections/endpoint_audit.py')
            or gate['endpoint_audit_sha256']!=digest(REPORT/'center_endpoint_audit.json')):
        raise ValueError('Representative full A/C endpoint approximation not passed or evidence changed')
    f=freeze_execution();inputs=transfer_inputs()
    if not all(v.get('passed') for v in inputs.values()):return dict(submitted=False,waiting=inputs)
    side=active_side_job()
    for bundle in side.get('effective_jobs',[dict(job=side['job'],indices=list(range(16)))]):
        text=execute('gpu','sacct --array -n -P -j '+str(bundle['job'])+' -o JobID%40,State,ExitCode',readonly=True)
        roots=[r.split('|') for r in text.splitlines() if r.split('|')[0] in [str(bundle['job'])+'_'+str(i) for i in bundle['indices']]]
        if len(roots)!=len(bundle['indices']) or any(r[1:3]!=['COMPLETED','0:0'] for r in roots):return dict(submitted=False,waiting='all effective formal side-response blocks incomplete')
        ids={str(bundle['job'])+'_'+str(i) for i in bundle['indices']}
        steps=[r.split('|') for r in text.splitlines() if r.split('|')[0].split('.')[0] in ids]
        required=ids|{n+'.batch' for n in ids}|{n+'.extern' for n in ids}
        if not required.issubset({r[0] for r in steps}) or any(r[1:3]!=['COMPLETED','0:0'] for r in steps):
            return dict(submitted=False,waiting='all effective side root/batch/extern steps complete')
    upload_files('gpu',f['release'],{n:DATA/'execution_payload'/n for n in f['files']})
    execute('gpu','cd '+q(f['release'])+' && '+q(GPU_PYTHON)+' -B -m unittest test_ac225_corrections -v > '+q(f['release']+'/local_checks.log')+' 2>&1',timeout=180)
    result={}
    for energy in (440,218):
        python='''import os,sys,pathlib,json
release=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(release))
from ac225_corrections.io import read
from ac225_corrections.runtime import run
energy=int(sys.argv[2]);index=int(os.environ['SLURM_ARRAY_TASK_ID']);base=pathlib.Path(sys.argv[3])
conditions=[(n,c) for n,c in read(release/'contract.json')['conditions'].items() if c['energy_keV']==energy]
name,c=conditions[index]
run(release,pathlib.Path(sys.argv[4]),base/'inputs'/c['dataset'],base/'formal'/name,name,
    base/'formal'/c['final440_from_same_branch'] if energy==218 else None)
'''
        body='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\nexport OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
        body+=q(GPU_PYTHON)+' -B -c '+q(python)+' '+q(f['release'])+' '+str(energy)+' '+q(GPU)+' '+q(side['output'])+'\n'
        options='-p gpu_5090 --account=scxi717 --qos=gpugpu -N1 -n1 -c8 --gres=gpu:1 --array=0-15%4 --time=4320'
        if energy==218:options+=' --dependency=afterok:'+str(result['440']['job'])
        result[str(energy)]=submit_registered('gpu','ACW_formal'+str(energy),body,options,GPU+'/formal',GPU)
    write(REPORT/'formal_submitted.json',result);return result


def full_center_audit():
    from ac225_corrections.endpoint_audit import full_response_bounds
    endpoint=read(REPORT/'center_endpoint_audit.json');side=read(DATA/'kernel_payload/side_contract.json')
    samples=[]
    for i,case in enumerate(endpoint['cases']):
        system,energy=case['group'].split('_');energy=int(energy)
        records=np.fromfile(DATA/'kernel_payload/params'/system/str(energy)/'Params_Detector.dat','<f4')[1:].reshape(-1,12)
        selected=np.flatnonzero(records[:,11]==1);d=int(np.flatnonzero(selected==case['detector'])[0])
        # Recover the explicitly preselected view from the original PE case.
        original=next(c for c in read(DATA/'kernel_payload/center_ray_audit.json')['groups'][case['group']]['cases']
                      if c['source_object_mm']==case['source'] and c['detector_parameter_index']==case['detector'])
        for factor in ['A'+str(energy)]+(['C440to218'] if energy==440 else []):
            spec=side['original_factors'][system][factor]
            samples.append(dict(case_index=i,factor=system+'/'+factor,detector=d,source=case['source'],
                view=original['view'],root=spec['root'],manifest_sha256=spec['manifest_sha256'],bins=len(selected)))
    code='''import sys,json,pathlib,hashlib,math,numpy as np
from scipy.spatial import Delaunay
c=json.loads(sys.argv[1]);g=np.load(c['geometry']);xy=g['coordinates_mm'][:3301,:2];tri=Delaunay(xy);out=[]
for s in c['samples']:
 root=pathlib.Path(s['root']);assert hashlib.sha256((root/'factor_manifest.json').read_bytes()).hexdigest()==s['manifest_sha256']
 point=np.array(s['source'],float);t=2*math.pi*s['view']/20
 x=point[0]*math.cos(t)+point[1]*math.sin(t);y=point[1]*math.cos(t)-point[0]*math.sin(t)
 simplex=tri.find_simplex([x,y]);assert simplex>=0
 b=tri.transform[simplex,:2]@(np.array([x,y])-tri.transform[simplex,2]);w=np.r_[b,1-b.sum()];assert w.min()>-1e-12
 z=int(np.argmin(abs(g['coordinates_mm'][::3301,2]-point[2])));assert abs(g['coordinates_mm'][z*3301,2]-point[2])<1e-6
 ids=tri.simplices[simplex]+z*3301
 raw=np.memmap(root/'SysMat_polar','<f4',mode='r',shape=(132040,s['bins']))
 s['air_probability']=float(w@(raw[ids,s['detector']].astype(float)/g['cell_volume_mm3'][ids]))
 out.append(s)
print('CAMPAIGN_JSON '+json.dumps(out))
'''
    f=prepare_gpu();values=python_json('gpu',GPU_PYTHON,code,dict(samples=samples,geometry=f['release']+'/whole_geometry.npz'))
    result=full_response_bounds(endpoint,values)
    result.update(endpoint_audit_sha256=digest(REPORT/'center_endpoint_audit.json'),
                  source_sha256=digest(HERE/'ac225_corrections/endpoint_audit.py'),
                  source_model='frozen full Polar point field / original cell volume; barycentric point query; no matrix regeneration')
    write(REPORT/'center_full_response_audit.json',result);return result


def strict_fetch():
    """All 32 routes, every saved state and background; no selected-frame claim."""
    marker=REPORT/'strict_delivery.json'
    if marker.exists():return read(marker)
    jobs=read(REPORT/'formal_submitted.json')
    accounting={}
    for energy,j in jobs.items():
        text=execute('gpu','sacct --array -n -P -j '+str(j['job'])+' -o JobID%40,State,ExitCode,AllocTRES%160,MaxRSS,Elapsed',readonly=True)
        rows=[r.split('|') for r in text.splitlines() if r]
        expected={str(j['job'])+'_'+str(i) for i in range(16)}
        if {r[0] for r in rows if r[0] in expected}!=expected:return dict(ready=False,waiting='full formal array accounting')
        effective=[r for r in rows if r[0].split('.')[0] in expected]
        required=expected|{r+'.batch' for r in expected}|{r+'.extern' for r in expected}
        if not required.issubset({r[0] for r in effective}):return dict(ready=False,waiting='all root/batch/extern accounting records')
        if any(r[1:3]!=['COMPLETED','0:0'] for r in effective):return dict(ready=False,waiting='all formal roots/steps successful and exited')
        accounting[energy]=text
    code='''import sys,json,pathlib,hashlib
r=pathlib.Path(json.loads(sys.argv[1])['root']);out={}
for p in r.glob('*/completion.json'):
 c=json.loads(p.read_text());out[p.parent.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=c,
 sizes={n:(p.parent/n).stat().st_size for n in c['files']})
print('CAMPAIGN_JSON '+json.dumps(out))
'''
    metadata=python_json('gpu',GPU_PYTHON,code,dict(root=GPU+'/formal'))
    conditions=read(DATA/'payload/contract.json')['conditions']
    if set(metadata)!=set(conditions):raise ValueError('Exactly all 32 approved routes required')
    target=DATA/'formal_delivery';target.mkdir(exist_ok=True)
    need=sum(sum(v['sizes'].values()) for v in metadata.values())
    if shutil.disk_usage(target).free<need/0.8:raise OSError('Local complete strict delivery needs 20% storage reserve')
    with sftp_bounded('gpu') as s:
        for name,cfg in conditions.items():
            meta=metadata[name];r=meta['receipt']
            if not r['passed'] or r['condition']!=name or (r['iterations'],r['save_step'],r['checkpoint_count'])!=(cfg['iterations'],cfg['save_step'],200):
                raise ValueError('Formal condition/save loop differs')
            files=dict(r['files'],**{'completion.json':meta['sha256']})
            for n,h in files.items():
                p=target/name/n;p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists():
                    if digest(p)!=h:raise FileExistsError('Different local output preserved')
                    continue
                partial=p.with_name(p.name+'.partial')
                if partial.exists():raise FileExistsError('Interrupted strict output fetch preserved')
                download_serial(s,GPU+'/formal/'+name+'/'+n,partial,h)
                if digest(partial)!=h:raise ValueError('Formal output SHA mismatch')
                partial.replace(p)
            verify(target/name,files)
    checks={}
    for name,cfg in conditions.items():
        folder=target/name
        history=np.load(folder/'active_history.npy',mmap_mode='r')
        if history.shape!=(200,78920) or not np.isfinite(history).all() or np.any(history<0):raise ValueError('Full active history failed')
        for k,iteration in enumerate(range(cfg['save_step'],cfg['iterations']+1,cfg['save_step'])):
            with np.load(folder/'checkpoints'/f'iter_{iteration:05d}.npz') as a:
                if int(a['iteration'])!=iteration or not np.array_equal(a['image'].reshape(-1),history[k]):raise ValueError('All checkpoint/history state closure failed')
                expected=((iteration-1)//20)*20 if 'tew' in cfg['branch'] and iteration>20 else 0
                if int(a['background_last_update'])!=expected:raise ValueError('Wrong actual background checkpoint epoch')
                for key in ('body_background','instrument_cross_background'):
                    if not np.isfinite(a[key]).all() or np.any(a[key]<0):raise ValueError('Saved additive background invalid')
        checks[name]=dict(all200_states_verified=True,completion_sha256=metadata[name]['sha256'])
    result=dict(passed=True,local=str(target),routes=32,all_checkpoints=6400,all_files_SHA_verified=True,
        full_histories_and_backgrounds_checked=True,conditions=checks,accounting=accounting,
        physical_calibration_passed=False,visual_review_pending=True)
    write(marker,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare_gpu','deploy_gpu','submit_side_responses','freeze_execution','transfer_inputs','submit_formal','repair_pending_side_launch','strict_fetch','full_center_audit']);a=p.parse_args()
    print(json.dumps(globals()[a.action](),indent=2))
