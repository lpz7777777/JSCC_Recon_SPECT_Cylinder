"""Explicit freeze/deploy/one-formal-submit; no watch or automatic resubmit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ehe_5e9_workflow import connection, command, q, put_tree
from jscc_5e10_common import GPU_PROJECT, GPU_PYTHON

HERE = Path(__file__).resolve().parent
TAG = 'nema_5e9_prejune_compton_20261026'
DATA = HERE/'generated'/TAG
REPORT = HERE/'reports/NEMA_Body_H60'/TAG
REMOTE = GPU_PROJECT+'/generated/'+TAG


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text('utf8'))


def write(path,data):
    path=Path(path)
    with path.open('x',encoding='utf8') as f: json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')


def freeze():
    marker=REPORT/'freeze.json'
    if marker.exists():
        f=read(marker)
        for rel,sha in f['files'].items():assert digest(DATA/'payload'/rel)==sha,rel
        return f
    tests=subprocess.run([sys.executable,'-X','utf8',str(HERE/'test_prejune_compton.py')],capture_output=True,text=True)
    (REPORT/'local_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf8')
    if tests.returncode:raise RuntimeError('Historical numerical tests failed')
    source=HERE/'generated/compton_energy_probability_v5_5e9/formal_payload'
    payload=DATA/'payload';payload.mkdir(exist_ok=False)
    shutil.copytree(DATA/'legacy_original',payload/'legacy_original')
    shutil.copytree(DATA/'calibration',payload/'calibration')
    for name in ('prejune_compton_model.py','run_prejune_compton.py'):
        shutil.copy2(HERE/name,payload/name)
    shutil.copy2(HERE.parents[1]/'detector_csv.py',payload/'detector_csv.py')
    shutil.copy2(source/'whole_geometry.npz',payload/'whole_geometry.npz')
    shutil.copy2(DATA/'recent_inputs/nema_h60_imaging_5e9_files.json',payload/'projection_manifest.json')
    raw=read(DATA/'recent_inputs/collections/NEMA_Body_H60_5e9.json')
    cal=read(payload/'calibration/collection.json')
    assert sum(raw['primary_counts'])==5_000_000_000 and cal['primary_counts']==[0,1_000_000_000,0]
    assert not set(raw['seeds']).intersection(cal['seeds'])
    projection=read(payload/'projection_manifest.json')
    factors=read(source/'factor_identity.json')['files']
    inputs={'generated/'+name:item['sha256'] for name,item in projection.items()}
    for name in ('SysMat_polar','Detector.csv','factor_manifest.json'):
        rel='440keV_RotateNum20/'+name
        inputs['generated/FactorsCalibrated/'+rel]=factors[rel]['sha256']
    contract=dict(study=TAG,historical_commit=read(REPORT/'archive_manifest.json')['historical_commit'],
        current_archive_manifest_sha256=digest(REPORT/'archive_manifest.json'),
        channels=['440_Compton'],iterations=10000,save_every=50,subsets=8,
        nodes=8,gpus_per_node=2,ranks=16,actual_primary_photons=5_000_000_000,
        actual_primary_counts=raw['primary_counts'],new_transport=False,
        new_selection='April strict kernel Neff>=50 on all current raw List rows; no v5 selection reused',
        kernel='April-14 angular Gaussian, Klein-Nishina, asymmetric energy and full position uncertainty',
        basis='point activity x=V_effective*rho; current density matrix B divided by full cell V',
        initializer='ones at 78920 active cells, zero outside; old point-activity ones',
        geometry=dict(dimensions_mm=[500,300,120],full_points=132040,active_cells=78920,views=20),
        resolution_fwhm_at_440=.13*math.sqrt(.511/.440),input_already_smeared=True,
        response_parameter_adapters=['440 keV instead of historical 511','13% FWHM at 511 keV current detector instead of old 10% at 662',
            'current detector coordinates, full 120 mm geometry and active domain','no extra random energy smearing'],
        sensitivity='April backprojection/global count-ratio method on independent uniform circle data, t/(t@V) varying-volume adapter; sum(S)=sum(Ssc)*Nd/Ns',
        historical_S_generator_provenance='Not fully recoverable; old 511 file is NOT reused and exact historical S reproduction is NOT claimed',
        iteration='Old safe_em_update, full S per subset (no divide by 8), original per-rank contiguous event subsets, no regularizer',
        output='Both native point activity and density; no intensity fit or smoothing',
        local_tests_passed=True,full_input_gpu_run_passed=False,separate_validation_job=False,
        gpu_memory_limit_fraction=.8,host_memory_limit_fraction=.8,
        walltime_cap_seconds=259200,walltime_is_eta=False,input_sha256=inputs,
        release_files={str(p.relative_to(payload)).replace('\\','/'):digest(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(payload/'contract.json',contract)
    files={str(p.relative_to(payload)).replace('\\','/'):digest(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f=dict(release=REMOTE+'/releases/'+key,key=key,files=files,contract_sha256=digest(payload/'contract.json'),local_tests_sha256=digest(REPORT/'local_tests.txt'))
    write(marker,f)
    return f


def launcher(f):
    allocation_code="import re,sys; s=open(sys.argv[1]).read(); n=int(re.search(r'\\bNumNodes=(\\d+)',s)[1]); assert n==8; t=re.search(r'\\bAllocTRES=([^\\s]+)',s)[1]; m=re.search(r'(?:^|,)mem=([0-9.]+)([KMGT])(?:,|$)',t); print(int(float(m[1])*1024**('KMGT'.index(m[2])+1)/n))"
    child='''set -euo pipefail
export TMPDIR="/tmp/nema_prejune_${SLURM_JOB_ID}_${SLURM_PROCID}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" CUDA_CACHE_PATH="$TMPDIR/cuda" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
exec '''+q(GPU_PYTHON)+''' -B -m torch.distributed.run --nnodes=8 --nproc_per_node=2 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_backend=static --max_restarts=0 '''+q(f['release']+'/run_prejune_compton.py')+' --release '+q(f['release'])+' --base '+q(GPU_PROJECT)+' --output '+q(REMOTE+'/formal_')+'"${SLURM_JOB_ID}"'
    return '''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
'''+ 'cd '+q(f['release'])+'\n'+'allocation='+q(REMOTE+'/allocation_')+'"${SLURM_JOB_ID}.txt"\nscontrol show job "$SLURM_JOB_ID" > "$allocation"\n'+ 'export JSCC_HOST_BYTES_NODE=$('+q(GPU_PYTHON)+' -B -c '+q(allocation_code)+' "$allocation")\n'+ 'export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)\nexport MASTER_PORT=$((40000+SLURM_JOB_ID%10000))\n'+ 'timeout --signal=TERM --kill-after=60s 258600s srun --label --kill-on-bad-exit=1 bash -c '+q(child)+'\n'


def deploy_submit():
    f=freeze()
    if (REPORT/'formal_job.json').exists():return read(REPORT/'formal_job.json')
    if (REPORT/'submission_intent.json').exists():raise RuntimeError('Existing submission intent: inspect scheduler; NEVER blindly retry')
    body=launcher(f);script=DATA/'formal.sh'
    if script.exists():assert script.read_text('utf8')==body
    else:script.write_text(body,encoding='utf8',newline='\n')
    with connection('gpu') as c:
        before=command(c,"squeue -h -u scxi717 -o '%i|%j|%T|%R'",timeout=60)
        if 'NEMA5e9_preJune_C' in before:raise RuntimeError('Same formal already in queue')
        # Space is for small code/S/images only; never materialize event matrices on disk.
        disk=json.loads(command(c,GPU_PYTHON+' -c '+q('import os,json;s=os.statvfs('+repr(GPU_PROJECT)+');print(json.dumps({"available":s.f_bavail*s.f_frsize}))')))
        if disk['available']<16*1024**3:raise RuntimeError('Less than 16GiB available; refuse submission')
        command(c,'mkdir -p -- '+q(f['release'])+' '+q(REMOTE+'/logs'))
        with c.open_sftp() as s:
            put_tree(s,DATA/'payload',f['release'])
            try:
                with s.open(REMOTE+'/formal.sh','rb') as stream:assert hashlib.sha256(stream.read()).hexdigest()==digest(script)
            except FileNotFoundError:s.put(str(script),REMOTE+'/formal.sh')
        checks='import hashlib,json,pathlib; r=pathlib.Path('+repr(f['release'])+'); h='+repr(f['files'])+'; assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v for k,v in h.items()); print("RELEASE_SHA_OK",len(h))'
        proof=command(c,GPU_PYTHON+' -B -c '+q(checks),timeout=120)
        command(c,'bash -n '+q(REMOTE+'/formal.sh'))
        opts='-p gpu_5090 --qos=gpugpu -N 8 -n 8 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:2 --time=3-00:00:00 --job-name=NEMA5e9_preJune_C --chdir='+q(f['release'])+' --output='+q(REMOTE+'/logs/formal.%j.out')+' --error='+q(REMOTE+'/logs/formal.%j.err')+' '+q(REMOTE+'/formal.sh')
        feasibility=command(c,'sbatch --test-only '+opts+' 2>&1',timeout=60)
        if 'error' in feasibility.lower():raise RuntimeError(feasibility)
        write(REPORT/'deployment.json',dict(release=f['release'],proof=proof,script_sha256=digest(script),queue_before=before,disk=disk,scheduler_test_only=feasibility))
        write(REPORT/'submission_intent.json',dict(command='sbatch --parsable '+opts,time=time.time(),resolved=False))
        job=command(c,'sbatch --parsable '+opts,timeout=60).strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Uncertain submission outcome: '+job)
        rec=dict(job=int(job),study=TAG,stage='formal',release=f['release'],contract_sha256=f['contract_sha256'],output=REMOTE+'/formal_'+job,
                 nodes=8,gpus_per_node=2,channel='440 keV pure Compton',iterations=10000,save_every=50,subsets=8,new_transport=False,separate_validation_jobs=0)
        write(REPORT/'formal_job.json',rec)
        status=command(c,'scontrol show job '+job,timeout=60)
        (REPORT/'submitted_status.txt').write_text(status,encoding='utf8')
        print(status,flush=True)
        return rec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','submit']);a=p.parse_args()
    print(json.dumps(freeze() if a.action=='freeze' else deploy_submit(),ensure_ascii=False,indent=2))
