"""One 64-GPU joint-only recovery; successful MC, selection and SC are read-only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
from campaign_20261102_io import execute, python_json, upload_files
from ehe_5e9_workflow import q
from derenzo_campaign_common import HERE, GPU_PROJECT, GPU_PYTHON, read, write, digest, hashes, verify_files

TAG='derenzo_jscc_joint_5e10_retry_20261102'
DATA=HERE/'generated'/TAG
REPORT=HERE/'reports/Derenzo'/TAG
REMOTE=GPU_PROJECT+'/generated/'+TAG
PARENT=HERE/'generated/derenzo_jscc_mono_5e10_20261030/gpu_payload'
OLD=GPU_PROJECT+'/generated/derenzo_jscc_mono_5e10_20261030/JSCC_440'
NODES,GPUS=8,8


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Nonunique source anchor: '+old[:100])
    return text.replace(old,new)


def function(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def prepare():
    marker=REPORT/'freeze.json';payload=DATA/'payload'
    if marker.exists():
        f=read(marker);verify_files(payload,f['files']);return f
    REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    parent_manifest=read(PARENT/'release_manifest.json');verify_files(PARENT,parent_manifest['sha256'])
    if payload.exists():raise FileExistsError('Inspect incomplete retry freeze before continuing')
    shutil.copytree(PARENT,payload)
    original=(PARENT/'derenzo_campaign_reconstruction.py').read_text('utf8');source=original
    begin=source.index("    channel=info['channels'][0];began=time.monotonic()")
    end=source.index('    prep_seconds=0.;',begin)
    source=source[:begin]+"    if rank==0:sensitivity.cpu().numpy().astype('<f4').tofile(a.output/'S_single.float32')\n"+source[end:]
    source=once(source,"info=dataset_info(a.dataset);iterations,step=", "info=dict(dataset_info(a.dataset),channels=['440_SinglePlusCompton']);iterations,step=")
    source=once(source,"channel=info['channels'][1];began=", "channel=info['channels'][0];began=")
    source=once(source,"        from derenzo_5e10_formal_runtime import check_authorization\n        authority=check_authorization(a.authority,a.authority_sha256,a.plan,a.input/'collection.json',a.release)",
        "        if digest(a.authority)!=a.authority_sha256:raise ValueError('Recovery authorization SHA differs')\n        authority=read(a.authority)\n        if (authority.get('policy')!='human_joint_repair_20261102' or authority.get('execution_authorized') is not True or authority.get('full_input_validation_passed') is not False\n                or authority['plan_sha256']!=digest(a.plan) or authority['collection_sha256']!=digest(a.input/'collection.json')\n                or authority['release_sha256']!=digest(a.release/'release_manifest.json')):raise ValueError('Current input-bound direct recovery authorization required')")
    source=once(source,"    if a.dataset=='JSCC_440':\n        manifest=read(a.selection/'selection_manifest.json');verify_files(a.selection,manifest['files'])",
        "    if a.dataset=='JSCC_440':\n        from jscc_5e10_gpu_memory_policy import release_unused_cuda,install_rows_for_policy\n        import run_energy_preflight_v5 as energy_module\n        global rows_for\n        rows_for=install_rows_for_policy(energy_module,a.output,rank,device)\n        release_unused_cuda(a.output,rank,device,'before_full_response')\n        manifest=read(a.selection/'selection_manifest.json');verify_files(a.selection,manifest['files'])\n        if digest(a.selection/'selection_manifest.json')!=authority['selection_manifest_sha256']:raise ValueError('Read-only original selection differs')")
    for name in ['joint_mlem','setup','resource_record','verify_resources','initial','save_npy']:
        if function(original,name)!=function(source,name):raise ValueError('Original scientific/resource helper changed: '+name)
    (payload/'derenzo_campaign_reconstruction.py').write_text(source,encoding='utf8',newline='\n')
    energy=(PARENT/'run_energy_preflight_v5.py').read_text('utf8')
    changed=once(energy,"    assigned=geometry.compact(rows_for(raw[partition],detector,variance,coords,B,law,model),0).double()",
        "    assigned=(geometry.compact(rows_for(raw[partition],detector,variance,coords,B,law,model),0).double()\n              if len(partition) else compact[:0].clone())")
    if function(energy,'rows_for')!=function(changed,'rows_for'):raise ValueError('Frozen response function changed')
    (payload/'run_energy_preflight_v5.py').write_text(changed,encoding='utf8',newline='\n')
    shutil.copyfile(HERE/'jscc_5e10_gpu_memory_policy.py',payload/'jscc_5e10_gpu_memory_policy.py')
    frozen_science={n:digest(PARENT/n) for n in ['compton_energy_probability_v5.py','compton_event_response.py',
        'torch_active_operator.py','continuous_energy_Sensi_full','transfer_training_summary.json','whole_geometry.npz']}
    verify_files(payload,frozen_science)
    for path in payload.glob('*.py'):compile(path.read_text('utf8'),str(path),'exec')
    manifest=dict(parent_release_sha256=digest(PARENT/'release_manifest.json'),sha256=hashes(payload))
    manifest['sha256'].pop('release_manifest.json',None)
    write(payload/'release_manifest.json',manifest)
    key=digest(payload/'release_manifest.json')[:16]
    f=dict(key=key,release=REMOTE+'/releases/'+key,files=hashes(payload),nodes=NODES,gpus_per_node=GPUS,
        world_size=NODES*GPUS,source_failed_job=1752461,new_MC=False,new_selection=False,repeat_SC=False,
        scientific_files_unchanged=frozen_science,original_joint_function_AST_identical=True,
        original_rows_for_function_AST_identical=True,original_32_event_batch_unchanged=True,
        memory_repair='Release only unused CUDA cache before response and above 60%; retain all peak counters and 80% guards',
        embedded_check_repair='Ranks without any of the 32 sample events contribute an empty transpose term',
        iterations=10000,save_step=50,channels=['440_SinglePlusCompton'],
        full_input_prevalidation_passed=False,no_separate_validation_jobs=True)
    write(marker,f);return f


def deploy():
    f=prepare()
    if (REPORT/'deployment.json').exists():return f
    upload_files('gpu',f['release'],{n:DATA/'payload'/n for n in f['files']})
    code='''import sys,json,pathlib,hashlib,subprocess,shutil
c=json.loads(sys.argv[1]);root=pathlib.Path(c['release'])
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
for n,h in c['files'].items():assert sha(root/n)==h,n
old=pathlib.Path(c['old']);selected=json.loads((old/'run/selection/selection_manifest.json').read_text())
assert selected['passed'] and selected['accepted_events']==6872376
assert len(list((old/'run/formal/checkpoints/440_SinglePhoton').glob('checkpoint_*/manifest.json')))==200
assert not (old/'run/formal/checkpoints/440_SinglePlusCompton').exists()
account=subprocess.check_output(['sacct','-j','1752461','-nP','-o','JobID,State,ExitCode'],text=True)
assert '1752461|FAILED|1:0' in account
queue=subprocess.check_output(['squeue','-h','-u','scxi717','-o','%i'],text=True).splitlines()
assert not any(v.split('_')[0]=='1752461' for v in queue)
for p in ['scripts','logs','allocations']:(pathlib.Path(c['remote'])/p).mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(c['remote']).free>30*(1<<30)
events=selected['events_per_view'];world=64;gpus=8
ranks=[sum(n*(r+1)//world-n*r//world for n in events) for r in range(world)]
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True,selected_manifest_sha256=sha(old/'run/selection/selection_manifest.json'),
 events_per_view=events,accepted_events=selected['accepted_events'],original_SC_files={p.name:sha(p) for p in (old/'run/formal').glob('*') if p.is_file()},
 source_accounting=account,maximum_events_per_rank=max(ranks),
 conservative_host_bytes_node=[sum(ranks[i:i+gpus])*78920*4+gpus*(16<<30) for i in range(0,world,gpus)])))
'''
    proof=python_json('gpu',GPU_PYTHON,code,dict(f,old=OLD,remote=REMOTE),timeout=180)
    write(REPORT/'deployment.json',proof);return f


def submit():
    path=REPORT/'formal_job.json';intent=REPORT/'submission_intent.json'
    if path.exists():return read(path)
    if intent.exists():raise RuntimeError('Unresolved retry intent; inspect scheduler, never repeat')
    f=deploy();proof=read(REPORT/'deployment.json')
    # A single setup process binds the new topology to unchanged inputs, then
    # launches only the missing joint solve, without selection or SC repetition.
    driver='''import os,sys,json,pathlib,subprocess,shlex
release=pathlib.Path(sys.argv[1]);output=pathlib.Path(sys.argv[2]);old=pathlib.Path(sys.argv[3]);alloc=pathlib.Path(sys.argv[4])
sys.path.insert(0,str(release))
from derenzo_campaign_common import read,write,digest,hashes
from jscc_5e10_common import host_allocated_bytes
output.mkdir(parents=True,exist_ok=False)
selected=read(old/'run/selection/selection_manifest.json');events=selected['events_per_view'];world=64;gpus=8
if int(os.environ['SLURM_NNODES'])!=8:raise ValueError('Eight actual nodes required')
ranks=[sum(n*(r+1)//world-n*r//world for n in events) for r in range(world)]
node_bytes=[sum(ranks[i:i+gpus])*78920*4+gpus*(16<<30) for i in range(0,world,gpus)]
actual=host_allocated_bytes(alloc.read_text(),8)
if max(node_bytes)>.8*actual:raise MemoryError('Full event responses plus16GiB/rank exceed actual node budget')
plan=dict(dataset='JSCC_440',nodes=8,gpus_per_node=8,world_size=64,accepted_events=selected['accepted_events'],
 collection_sha256=digest(old/'input/collection.json'),release_sha256=digest(release/'release_manifest.json'),
 channels=['440_SinglePlusCompton'],iterations=10000,save_step=50,conservative_host_bytes_node=node_bytes,actual_host_bytes_node=actual)
write(output/'plan.json',plan)
auth=dict(passed=True,execution_authorized=True,full_input_validation_passed=False,dataset='JSCC_440',
 plan_sha256=digest(output/'plan.json'),collection_sha256=plan['collection_sha256'],release_sha256=plan['release_sha256'],
 selection_manifest_sha256=digest(old/'run/selection/selection_manifest.json'),source_SC_job=1752461,
 source_SC_output=str(old/'run/formal'),repeat_SC=False,repeat_selection=False,policy='human_joint_repair_20261102')
write(output/'authorization.json',auth)
argv=[sys.executable,'-B','-m','torch.distributed.run','--nnodes=8','--nproc_per_node=8','--node_rank=$SLURM_PROCID',
 '--master_addr='+os.environ['MASTER_ADDR'],'--master_port='+os.environ['MASTER_PORT'],'--rdzv_backend=static',
 '--rdzv_id='+os.environ['SLURM_JOB_ID'],'--max_restarts=0',str(release/'derenzo_campaign_reconstruction.py'),
 '--release',str(release),'--dataset','JSCC_440','--mode','formal','--input',str(old/'input'),
 '--output',str(output/'formal'),'--selection',str(old/'run/selection'),'--allocation',str(alloc),
 '--plan',str(output/'plan.json'),'--plan-sha256',digest(output/'plan.json'),
 '--authority',str(output/'authorization.json'),'--authority-sha256',digest(output/'authorization.json')]
invocation=' '.join('"'+v+'"' if v=='--node_rank=$SLURM_PROCID' else shlex.quote(v) for v in argv)
subprocess.run(['srun','--label','--kill-on-bad-exit=1','--chdir=/tmp','bash','-c','exec '+invocation],check=True)
m=read(output/'formal/run_manifest.json')
if not m['passed'] or m['channels']!=['440_SinglePlusCompton']:raise ValueError('Only complete joint result required')
if len(list((output/'formal/checkpoints/440_SinglePlusCompton').glob('checkpoint_*/manifest.json')))!=200:raise ValueError('All200 checkpoints required')
write(output/'completion.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],source_SC_job=1752461,files=hashes(output)))
print('JOINT_ONLY_64GPU_COMPLETE',flush=True)
'''
    driver_path=DATA/'joint_driver.py';driver_path.write_text(driver,encoding='utf8',newline='\n')
    upload_files('gpu',REMOTE+'/scripts',{'joint_driver.py':driver_path})
    body='#!/usr/bin/env bash\nset -euo pipefail\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\n'
    body+='export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0\n'
    body+='export JSCC_PROJECT_ROOT='+q(f['release'])+' DERENZO_PHASE_SECONDS=244800 DERENZO_PREPARE_SECONDS=14400\n'
    body+='export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)\nexport MASTER_PORT=$((35000+SLURM_JOB_ID%20000))\n'
    body+='allocation="'+REMOTE+'/allocations/${SLURM_JOB_ID}.txt"\nscontrol show job -o "$SLURM_JOB_ID" > "$allocation"\n'
    body+='timeout --signal=TERM --kill-after=60s 258900s '+q(GPU_PYTHON)+' -B '+q(REMOTE+'/scripts/joint_driver.py')+' '+q(f['release'])+' '+q(REMOTE+'/run')+' '+q(OLD)+' "$allocation"\n'
    script=DATA/'formal.sh';script.write_text(body,encoding='utf8',newline='\n');upload_files('gpu',REMOTE+'/scripts',{'formal.sh':script})
    cmd='sbatch --parsable -p gpu_5090 --qos=gpugpu -N8 -n8 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:8 --time=4320 --job-name=DR5e10_joint64 --comment='+f['key']+' --output='+q(REMOTE+'/logs/formal_%j.out')+' --error='+q(REMOTE+'/logs/formal_%j.err')+' '+q(REMOTE+'/scripts/formal.sh')
    execute('gpu','bash -n '+q(REMOTE+'/scripts/formal.sh'),readonly=True)
    request=dict(command=cmd,release=f['release'],output=REMOTE+'/run',nodes=8,gpus_per_node=8,world_size=64,
        script_sha256=digest(script),driver_sha256=digest(driver_path),source_selection_sha256=proof['selected_manifest_sha256'],submitted_epoch=time.time())
    write(intent,request)
    result=execute('gpu',cmd,timeout=60).strip();number=result.split(';')[0]
    if not number.isdigit():raise RuntimeError('Uncertain retry submission; intent retained')
    request.update(job=int(number),sbatch_result=result);write(path,request)
    print('JOINT64_FORMAL_SUBMITTED',number,flush=True);return request


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','deploy','submit']);a=p.parse_args();globals()[a.action]()
