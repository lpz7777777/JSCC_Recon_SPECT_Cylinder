"""Freeze an independent matched-S width study; no remote operation."""
import ast, hashlib, io, json, shutil, subprocess, sys, tarfile
from pathlib import Path
import numpy as np

CODE=Path(__file__).resolve().parent
EXP=CODE.parents[1]
ROOT=EXP.parents[1]
TAG='nema5e9_width_sweep_20261115'
DATA=EXP/'generated'/TAG
REPORT=EXP/'reports/NEMA_Body_H60'/TAG
PARENT=EXP/'generated/compton_energy_probability_v5_5e9_full10000/formal_payload'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def derive_kernel(source):
    """Keep alpha=1 literally delegated; dilate each conditional mixture."""
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='energy_log_density')
    body='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
    body=body.replace('def energy_log_density(', 'def scaled_energy_log_density(')
    body=body.replace("gaussian_float32=False):", "gaussian_float32=False, width_scale=1.):",1)
    anchor='    beta,sp=geometry_arrays(prepared,coordinates)'
    body=body.replace(anchor,"""    if width_scale == 1.:
        return original.energy_log_density(prepared,coordinates,law,node_chunk=node_chunk,
            material=material,compiled_bins=compiled_bins,backend=backend,gaussian_float32=gaussian_float32)
    if width_scale not in (.8,.6) or not material or backend!='tail16_mid2' or compiled_bins:
        raise ValueError('Only frozen material tail16_mid2 and declared scales')
"""+anchor)
    anchor='        total=torch.full_like(b,-torch.inf);mass_total=torch.zeros_like(b)'
    body=body.replace(anchor,"""        # Exact mean of the ORIGINAL physically clipped uniform-bin law.
        # Dilation about it preserves material bias and scales all three
        # conditional variance components by alpha**2. No data energy changes.
        mu_n=torch.zeros_like(b);mu_d=torch.zeros_like(b)
        for start,end,_ in chunks:
            delta=n[lower,start:end+1]*(1-alpha[...,None])+n[upper,start:end+1]*alpha[...,None]
            left=(t[...,None]+delta[...,:-1]).clamp_min(0)
            right=(t[...,None]+delta[...,1:]).clamp_max(E0)
            retained=(right-left).clamp_min(0)/(delta[...,1:]-delta[...,:-1])
            mu_n+=(retained*(left+right)/2).sum(-1);mu_d+=retained.sum(-1)
        if bool((mu_d<=0).any()):raise ValueError('Empty original material law')
        mu=mu_n/mu_d
"""+anchor)
    anchor='                    mean_eval=safe.to(dtype);sigma_eval=sigma.to(dtype)'
    assert body.count(anchor)==1
    body=body.replace(anchor,'                    mean_eval=(mu[...,None]+width_scale*(safe-mu[...,None])).to(dtype)\n                    sigma_eval=(width_scale*sigma).to(dtype)')
    header='''"""Scientific width candidate; separate from equivalent storage optimization.
All three widths scale about the original physically clipped material mean.
Electronic sigma is scaled at its original latent transfer, not re-calibrated.
Mean, weights, KN, B, q selection and event data are retained. Alpha=1 delegates.
"""
import math
import numpy as np
import torch
import compton_energy_probability_v5 as original
from compton_energy_probability_v5 import E0,MASS,ENERGY_VARIANCE_SLOPE,geometry_arrays
'''
    tail='''
def scaled_response(prepared,coordinates,B,law,width_scale):
    if width_scale==1.:
        return original.normalized_proxy_response(prepared,coordinates,B,law,node_chunk=16,
            backend='tail16_mid2',gaussian_float32=True)
    logp,beta=scaled_energy_log_density(prepared,coordinates,law,node_chunk=16,
        backend='tail16_mid2',gaussian_float32=True,width_scale=width_scale)
    e=prepared.e1.double();kn=E0/(E0-e)+(E0-e)/E0
    logweights=logp+(kn[:,None]-beta.sin().square()).log()+B[prepared.cpnum1-1].double().log()
    norm=torch.logsumexp(logweights,dim=1,keepdim=True)
    if not bool(torch.isfinite(norm).all()):raise ValueError('Empty frozen event response')
    response=(logweights-norm).exp().to(B.dtype)
    if not bool(torch.isfinite(response).all()):raise ValueError('Invalid scaled response')
    return response
'''
    return header+body+tail

def main():
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    target=DATA/'payload_r3'
    if (REPORT/'freeze_r3.json').exists():
        f=read(REPORT/'freeze_r3.json')
        for n,s in f['files'].items():assert sha(target/n)==s,n
        print('EXISTING FREEZE VERIFIED',f['key']);return
    target.mkdir(exist_ok=True)
    contract=read(PARENT/'contract.json')
    receipt=read(EXP/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/formal_summary.json')
    delivered=EXP/'generated/compton_energy_probability_v5_5e9_full10000/formal_results/1669255/continuous_energy'
    assert sha(delivered/'run_manifest.json')==receipt['models']['continuous_energy']['sha256']['run_manifest.json']
    assert sha(PARENT/'contract.json')==receipt['contract_sha256']
    for suffix in ('history','active'):
        name=f'Image_440_SinglePlusCompton_{suffix}.float32'
        assert sha(delivered/name)==receipt['models']['continuous_energy']['sha256'][name]
    first=np.fromfile(delivered/'Image_440_SinglePlusCompton_history.float32','<f4',count=78920)
    first.tofile(target/'original_iteration_00050.float32')
    shutil.copyfile(delivered/'Image_440_SinglePlusCompton_active.float32',target/'original_iteration_10000.float32')
    for n,s in contract['files'].items():
        p=PARENT/n
        if p.exists():assert sha(p)==s,n
    names=[p for p in PARENT.glob('*.py')]+[PARENT/n for n in ['config.json','whole_geometry.npz','transfer_training_summary.json','continuous_energy_Sensi_full']]
    for p in names:shutil.copyfile(p,target/p.name)
    shutil.copytree(PARENT/'selections',target/'selections',dirs_exist_ok=True)
    for name in ['jscc_5e10_joint_mlem.py']:
        p=EXP/'generated/nema_joint5e10_formal_1740189_20261029/evidence/source'/name
        m=read(p.parent.parent.parent/'result/run_manifest.json');assert sha(p)==m['helper_sha256'][name]
        shutil.copyfile(p,target/name)
    adapter=ROOT/'docs/research/jscc_uncertainty_optimization_20261115/adapters.py'
    v=read(adapter.parent/'verification.json');assert v['passed'] and sha(adapter)==v['local_code_sha256']['adapters.py']
    shutil.copyfile(adapter,target/'adapters.py')
    detector=EXP/'generated/Diagnostics/process_list_audit/Detector.csv'
    assert sha(detector)==contract['factor_payload_sha256']['440keV_RotateNum20/Detector.csv']
    shutil.copyfile(detector,target/'Detector.csv')
    (target/'width_kernel.py').write_text(derive_kernel((target/'compton_energy_probability_v5.py').read_text()),encoding='utf8',newline='\n')
    cal=EXP/'generated/compton_energy_probability_v5_5e9/calibration_results/selections'
    calreceipt=read(EXP/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9/calibration_fetch.json')['files']
    idxpath=cal/'circle_train_legacy_v01_kept_rows.npy';assert sha(idxpath)==calreceipt['selections/'+idxpath.name]
    indices=np.load(idxpath);assert len(indices)==170247
    manifest=read(EXP/'generated/compton_first_scatter_v2/analysis_inputs/input_manifest.json')['files']
    archive=EXP/'generated/compton_first_scatter_v2/analysis_inputs.tar.gz';found={}
    if (DATA/'calibration_extract.json').exists():
        extract=read(DATA/'calibration_extract.json')
        if not (target/'calibration_rows.npy').exists():
            shutil.copyfile(DATA/'payload/calibration_rows.npy',target/'calibration_rows.npy')
        assert sha(target/'calibration_rows.npy')==extract['selected_rows_sha256']
    else:
        with tarfile.open(archive,'r|gz') as t:
            for item in t:
                name=item.name.removeprefix('./').removeprefix('analysis_inputs/')
                if name not in ('circle_train/legacy_v01.csv','circle_train/collection.json'):continue
                raw=t.extractfile(item).read();assert hashlib.sha256(raw).hexdigest()==manifest[name]
                if name.endswith('.csv'):
                    rows=np.loadtxt(io.BytesIO(raw),delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
                    np.save(target/'calibration_rows.npy',rows[indices]);found['raw_shape']=list(rows.shape)
                else:found['collection']=json.loads(raw)
                found[name]=manifest[name]
                if 'collection' in found and 'raw_shape' in found:break
        assert 'collection' in found and 'raw_shape' in found
        assert found['collection']['primary_counts'][1]==1000000000
        extract=dict(found,selected_rows_sha256=sha(target/'calibration_rows.npy'),selection_sha256=sha(idxpath),events=len(indices))
        write(DATA/'calibration_extract.json',extract)
    shutil.copyfile(DATA/'calibration_extract.json',target/'calibration_extract.json')
    for name in ('run_study.py','test_study.py'):
        shutil.copyfile(CODE/name,target/name)
    protocol=dict(study=TAG,scales=[1.,.8,.6],definition='Conditional-mixture dilation about original clipped-law mean; material/electronic/position SD all multiplied by alpha; material mean bias unchanged',
        energy_keV=440,total_dual_energy_gamma=5000000000,events=483743,iterations=10000,save_step=50,nodes=8,gpus_per_node=1,
        input_sha256={n:s for n,s in contract['input_sha256'].items() if '/218keV_' not in n},
        factor_sha256={n:s for n,s in contract['factor_payload_sha256'].items() if n.startswith('440keV_RotateNum20/')},
        events_per_view=contract['events_per_view'],parent_contract_sha256=sha(PARENT/'contract.json'),
        parent_run_manifest_sha256=sha(delivered/'run_manifest.json'),adapter_sha256=sha(adapter),
        matched_S='Independent original 170247 circle_train events, 1e9 primary 440 photons, full volume and rotation mean; alpha=1 reuses delivered S',
        randomization='Same recorded MC realization, same original selections, no random re-smearing, no new transport',
        solve='Frozen joint_mlem: ones, original view/32-event accumulation and matched S; no smoothing/regularization',
        engineering_gate='GPU paired 50 iterations, bitwise requirement; same alpha100 matrices/S/topology; 3 timing repetitions; choose prefetch only for >5% median end-to-end gain over cache',
        allocation_walltime_hours=72,phase_limit_hours=22,calibration_limit_hours=4,output_budget_bytes=2<<30,
        old_work_quota_bytes=800<<30,storage_policy='No event response written to shared disk; all C is CPU RAM. New outputs capped 2GiB; require 20% reserve on available write budget.',
        baseline_reference='Previously delivered 1669255; no repeat original full 10000 solve. New optimized alpha100 is this explicitly requested paired reconstruction.')
    write(target/'study.json',protocol)
    for p in target.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    files={str(p.relative_to(target)).replace('\\','/'):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    test=subprocess.run([sys.executable,'-B','-X','utf8',str(target/'test_study.py')],capture_output=True,text=True,encoding='utf8')
    (REPORT/'local_tests_r3.txt').write_text(test.stdout+test.stderr,encoding='utf8')
    if test.returncode or 'WIDTH_SWEEP_LOCAL_PASS' not in test.stdout:raise RuntimeError('Local test failed; draft not frozen: '+test.stderr[-2000:])
    write(REPORT/'freeze_r3.json',dict(key=key,files=files,study=protocol,payload_dir=target.name,local_tests_passed=True,local_tests_sha256=sha(REPORT/'local_tests_r3.txt'),supersedes_unsubmitted_drafts=['3620477d7d2952fc','035b648e3e8508f0']))
    print('FROZEN',key,len(files),'files')

if __name__=='__main__':main()
