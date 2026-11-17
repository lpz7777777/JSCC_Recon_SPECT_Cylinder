"""Authorized resource rebalance; independent full-input validation before formal.

No transport, collection or selection is repeated. Original publications, jobs,
receipts and source arrays remain intact. Only pending 1719262 is held to prevent
two production chains; actual running production prevents its replacement.
"""
import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import reconstruction_execution_repair_20261020 as old
import derenzo_campaign_workflow as dw
import jscc_5e10_reconstruction_workflow as nw
from ehe_5e9_workflow import connection, command, q
from jscc_5e10_common import HERE, GPU_PROJECT, GPU_PYTHON, read, write, digest, hashes, verify_files
from resource_rebalance_gate_20261022 import measured, check, MAX_MINUTES, TOPOLOGY
from resume_reconstruction_controllers_20261019 import live_registration

TAG='resource_rebalance_20261022_v3'
ROOT=HERE/'reports'/TAG; DATA=HERE/'generated'/TAG; REMOTE=GPU_PROJECT+'/generated/'+TAG
REPORTS={k:ROOT/k for k in TOPOLOGY}; DATAS={k:DATA/k for k in TOPOLOGY}; REMOTES={k:REMOTE+'/'+k for k in TOPOLOGY}
GATE='resource_rebalance_gate_20261022.py'
REG=DATA/'controller_registration.json'
NEMA_OLD=HERE/'reports/NEMA_Body_H60/jscc_geant4_5e10_10000'
DERENZO_OLD=dw.REPORT


def immutable(path, data):
    old.immutable(path,data)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Expected exactly one reviewed execution edit: '+before)
    return text.replace(before,after)


def ast_function(text, name):
    tree=ast.parse(text); node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    return ast.dump(node,include_attributes=False)


def prepare_payload():
    """Reuse exact frozen source/arrays; make three explicit topology-only edits."""
    source=HERE/'generated/jscc_geant4_5e10_10000/production_payload'; dest=DATA/'nema_payload'
    freeze=read(NEMA_OLD/'production_freeze.json'); verify_files(source,freeze['sha256'])
    if dest.exists():
        f=read(ROOT/'nema_freeze.json');verify_files(dest,f['files']);return f
    shutil.copytree(source,dest)
    edits={
      'jscc_5e10_runtime.py': [('world!=32 or not 0<=local<4 or torch.cuda.device_count()!=4','world!=16 or not 0<=local<2 or torch.cuda.device_count()!=2'),('result=[None]*32','result=[None]*16'),('torch.set_num_threads(10)','torch.set_num_threads(6)')],
      'jscc_5e10_contract.py': [("cfg['gpus_per_node']!=4","cfg['gpus_per_node']!=2"),("cfg['world_size']!=32","cfg['world_size']!=16"),('len(resources)!=32','len(resources)!=16'),('list(range(32))','list(range(16))'),('gres/gpu=32','gres/gpu=16'),('list(range(4))','list(range(2))'),("for r in group})!=4","for r in group})!=2")],
      'run_jscc_5e10.py': [("int(os.environ['JSCC_HOST_BYTES_NODE'])/4","int(os.environ['JSCC_HOST_BYTES_NODE'])/2")]
    }
    changed={}
    for name, replacements in edits.items():
        original=(source/name).read_text(encoding='utf8'); updated=original
        for before,after in replacements:updated=replace_once(updated,before,after)
        ast.parse(updated);(dest/name).write_bytes(updated.encode())
        changed[name]={'original_sha256':digest(source/name),'new_sha256':digest(dest/name),'exact_replacements':replacements}
    for name,functions in {'jscc_5e10_contract.py':('write_checkpoint','verify_checkpoints'),
                           'run_jscc_5e10.py':('operator_checks','relative','publish_images')}.items():
        for function in functions:
            if ast_function((source/name).read_text(),function)!=ast_function((dest/name).read_text(),function):
                raise ValueError('Science/checkpoint function changed: '+function)
    cfg=read(source/'contract.json'); old_cfg=dict(cfg)
    for name in cfg['files']:
        if name not in edits and digest(dest/name)!=cfg['files'][name]:raise ValueError('Original scientific payload changed: '+name)
    accepted=read(NEMA_OLD/'validation_acceptance.json'); proof=read(Path(accepted['local_result'])/'verification.json')
    overhead=max(r['host_peak_rss_bytes']-r['event_bytes_cpu'] for r in proof['resources'])*1.25
    event_counts=cfg['events_per_view']; ranks=[sum(n*(r+1)//16-n*r//16 for n in event_counts) for r in range(16)]
    node_bytes=[sum(ranks[i:i+2])*78920*4+2*overhead for i in range(0,16,2)]
    if max(node_bytes)>.8*(252000<<20):raise MemoryError('Measured response/overhead prediction exceeds new node budget')
    cfg.update(nodes=8,gpus_per_node=2,world_size=16,files={n:digest(dest/n) for n in cfg['files']},
               conservative_expected_memory_per_node=max(node_bytes))
    write(dest/'contract.json',cfg)
    scientific_cfg_keys=set(old_cfg)-{'nodes','gpus_per_node','world_size','files','conservative_expected_memory_per_node'}
    if any(cfg[k]!=old_cfg[k] for k in scientific_cfg_keys):raise ValueError('Scientific configuration changed')
    scientific_identity={n:sha for n,sha in old_cfg['files'].items() if n not in edits}
    files=hashes(dest);write(dest/'verification_release.json',dict(sha256=files,scientific_release_sha256=freeze['sha256'],read_only=True))
    files=hashes(dest);key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    f=dict(release=REMOTE+'/science/'+key,files=files,source_release=read(NEMA_OLD/'production_deployment.json')['release'],
           contract_sha256=digest(dest/'contract.json'),changed=changed,unchanged_scientific_files=scientific_identity,
           predicted_node_bytes=node_bytes,prediction_overhead_bytes_per_rank=overhead,requested_memory_MiB=252000,
           full_input_resource_validation_required=True,original_science_configuration_keys_unchanged=sorted(scientific_cfg_keys))
    write(ROOT/'nema_freeze.json',f);return f


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    for k in TOPOLOGY:
        REPORTS[k].mkdir(exist_ok=True);DATAS[k].mkdir(exist_ok=True)
    f=prepare_payload()
    for name in ('gpu_deployment.json','gpu_freeze.json'):
        immutable(REPORTS['derenzo']/name,(DERENZO_OLD/name).read_bytes())
    for name in ('selection_job.json','selection_acceptance.json'):
        immutable(REPORTS['derenzo']/'JSCC_440'/name,(DERENZO_OLD/'JSCC_440'/name).read_bytes())
    plan=read(DERENZO_OLD/'JSCC_440/validation_plan.json');plan.update(nodes=4,gpus_per_node=1,world_size=4)
    planfile=REPORTS['derenzo']/'JSCC_440/validation_plan.json';immutable(planfile,(json.dumps(plan,indent=2)+'\n').encode())
    pd=dict(remote=REMOTES['derenzo']+'/plan.json',sha256=digest(planfile))
    immutable(REPORTS['derenzo']/'JSCC_440/validation_plan_deployment.json',(json.dumps(pd,indent=2)+'\n').encode())
    execution=DATA/'execution_payload';execution.mkdir(exist_ok=True)
    for name in (Path(__file__).name,GATE,'test_resource_rebalance_20261022.py','jscc_5e10_common.py','reconstruction_execution_gate_20261020.py'):
        immutable(execution/name,(HERE/name).read_bytes())
    for k in TOPOLOGY:
        text=old.tmp_environment(REMOTE+'/execution',REMOTES[k]).replace(old.GATE_NAME,GATE)
        immutable(execution/(k+'_tmp.sh'),text.encode())
    execution_files=hashes(execution)
    with connection('gpu') as c:
        for root in [REMOTE+'/execution',f['release']]+list(REMOTES.values()):
            command(c,'mkdir -p '+' '.join(q(root+'/'+n) for n in ('','scripts','logs','allocations','JSCC_440')))
        marker=ROOT/'deployment.json'
        if not marker.exists():
            # Read-only reuse source arrays; destination is a new independent release.
            command(c,'test ! -e '+q(f['release']+'/contract.json')+' && cp -a '+q(f['source_release']+'/.')+' '+q(f['release'])+'/',timeout=180)
            with c.open_sftp() as s:
                for name in list(f['changed'])+['contract.json','verification_release.json']:
                    s.put(str(DATA/'nema_payload'/name),f['release']+'/'+name)
            remotecheck='from pathlib import Path;from jscc_5e10_common import *;r=Path('+repr(f['release'])+');verify_files(r,read(r/"verification_release.json")["sha256"]);print("NEW_TOPOLOGY_ALL_PAYLOAD_SHA_PASS")'
            identity=command(c,'cd '+q(f['release'])+' && '+q(GPU_PYTHON)+' -B -c '+q(remotecheck),timeout=180)
            write(marker,dict(release=f['release'],identity=identity,files=f['files']))
        for name in execution_files:old.upload(c,execution/name,REMOTE+'/execution/'+name)
        old.upload(c,planfile,pd['remote'])
        if not (ROOT/'linux_tests.txt').exists():
            tests=command(c,'cd '+q(REMOTE+'/execution')+' && REBALANCE_SCIENCE='+q(f['release'])+' '+q(GPU_PYTHON)+' -B -m unittest -v test_resource_rebalance_20261022 2>&1',timeout=180)
            if '\nOK' not in tests or 'skipped' in tests:raise RuntimeError('Actual Linux new-topology checks failed: '+tests)
            immutable(ROOT/'linux_tests.txt',tests.encode())
    write(ROOT/'execution_freeze.json',dict(files=execution_files,policy_max_minutes=MAX_MINUTES,topology=TOPOLOGY))
    return f


@contextlib.contextmanager
def namespace(kind):
    mod=nw if kind=='nema' else dw; previous=mod.REPORT,mod.DATA,mod.GPU_BASE
    mod.REPORT,mod.DATA,mod.GPU_BASE=REPORTS[kind],DATAS[kind],REMOTES[kind]
    try:yield mod
    finally:mod.REPORT,mod.DATA,mod.GPU_BASE=previous


def template(kind, stage, source=None):
    verify=stage.endswith('_verification') or stage.endswith('_verify');mode=stage.split('_')[0]
    nodes,gpus,world=TOPOLOGY[kind]; env=REMOTE+'/execution/'+kind+'_tmp.sh'
    root=REMOTES[kind]; output=(root+'/acceptance_'+mode if verify else root+'/'+mode)
    allocation=root+'/allocations/'+mode+'_"${SLURM_JOB_ID}.txt"'
    script='#!/usr/bin/env bash\nset -euo pipefail\nsource '+q(env)+'\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\nmodule load miniforge3/25.11.0-1\n'
    script+='export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0\n'
    science=read(ROOT/'nema_freeze.json')['release'] if kind=='nema' else read(DERENZO_OLD/'gpu_deployment.json')['release']
    script+='export JSCC_PROJECT_ROOT='+q(science)+'\ncd '+q(science)+'\n'
    if kind=='nema':
        if verify:
            args=[science+'/jscc_5e10_verify_stage.py','--release',science,'--input',old.ORIGINAL['nema'][2]+'/input','--factors',nw.FACTORS,'--result',source['output'],'--allocation',source['allocation'],'--output',output,'--mode',mode]
        else:
            args=[science+'/run_jscc_5e10.py','--contract',science+'/contract.json','--input-root',old.ORIGINAL['nema'][2]+'/input','--factors',nw.FACTORS,'--output',output,'--allocation','$allocation','--mode',mode]
    else:
        pd=read(REPORTS[kind]/'JSCC_440/validation_plan_deployment.json')
        args=[science+('/derenzo_campaign_verify.py' if verify else '/derenzo_campaign_reconstruction.py'),'--release',science,'--dataset','JSCC_440','--input',old.ORIGINAL['derenzo'][2]+'/JSCC_440/input','--plan',pd['remote']]
        if verify:args+=['--result',source['output'],'--allocation',source['allocation'],'--output',output,'--mode',mode]
        else:args+=['--selection',read(DERENZO_OLD/'JSCC_440/selection_job.json')['output'],'--output',output,'--allocation','$allocation','--mode',mode,'--plan-sha256',pd['sha256']]
    if verify:
        script+='timeout --signal=TERM --kill-after=60s 10500s '+q(GPU_PYTHON)+' -B -u '+' '.join(map(q,args))+'\n'
        return script,output,None
    prefix='JSCC' if kind=='nema' else 'DERENZO'
    if mode=='formal':
        auth=root+'/authority.json'
        script+='gate_exports=$('+q(GPU_PYTHON)+' -B '+q(REMOTE+'/execution/'+GATE)+' --authority '+q(auth)+')\neval "$gate_exports"\n'
        args+=['--authority',auth,'--authority-sha256','$'+prefix+'_AUTHORITY_SHA256']
    else:script+='export '+prefix+'_PHASE_SECONDS=10800 '+prefix+'_PREPARE_SECONDS=10800 '+prefix+'_TOTAL_SECONDS=21420\n'
    script+='[[ "$SLURM_NNODES" == '+str(nodes)+' && "$SLURM_NTASKS" == '+str(nodes)+' ]]\n'
    script+='allocation='+allocation+'\nscontrol show job "$SLURM_JOB_ID" > "$allocation"\n'
    if kind=='nema':
        code='from jscc_5e10_common import host_allocated_bytes;import sys;print(host_allocated_bytes(open(sys.argv[1]).read(),8))'
        script+='export JSCC_HOST_BYTES_NODE=$('+q(GPU_PYTHON)+' -B -c '+q(code)+' "$allocation")\n'
    script+='export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)\nexport MASTER_PORT=$((40000+SLURM_JOB_ID%10000))\n'
    inv=' '.join('"'+a+'"' if a.startswith('$') else q(a) for a in args)
    script+='timeout --signal=TERM --kill-after=60s "${'+prefix+'_TOTAL_SECONDS}s" srun --label --chdir='+q(root)+' --kill-on-bad-exit=1 bash -c \'source '+q(env)+'; exec '+q(GPU_PYTHON)+' -B -m torch.distributed.run --nnodes='+str(nodes)+' --nproc_per_node='+str(gpus)+' --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_backend=static --rdzv_id="${SLURM_JOB_ID}_r22" --max_restarts=0 "${@:1}"\' bash '+inv+'\n'
    return script,output,root+'/allocations/'+mode+'_{job}.txt'


def folder(kind):
    return REPORTS[kind]/('JSCC_440' if kind=='derenzo' else '')


def submit(kind, stage, dependency=None, source=None):
    if (HERE.parents[1]/'docs/ac225/formal_jobs_current.json').exists():
        raise RuntimeError('Retired by human formal-only instruction; use current formal job list, never recreate old validation/acceptance chains')
    dest=folder(kind);path=dest/(stage+'_job.json')
    if path.exists():return read(path)
    intent=dest/(stage+'_submission_intent.json')
    if intent.exists():raise RuntimeError('Unresolved submission intent; inspect scheduler')
    verify=stage.endswith('_verification') or stage.endswith('_verify');formal=stage=='formal'
    script,output,allocation=template(kind,stage,source);local=DATAS[kind]/(stage+'.sh');immutable(local,script.encode())
    remote=REMOTES[kind]+'/scripts/'+local.name;nodes,gpus,_=TOPOLOGY[kind]
    if verify:nodes,gpus=1,1
    # Actual submit plugin enforces at most eight CPU cores per GPU.
    # NEMA runtime caps each worker at six threads, below its eight-core share.
    cpus=gpus*8
    minutes=180 if verify else (MAX_MINUTES if formal else 360)
    log=REMOTES[kind]+'/logs/'+('JSCC_440_' if kind=='derenzo' else '')+stage
    opts='-p gpu_5090 --qos=gpugpu -N '+str(nodes)+' -n '+str(nodes)+' --ntasks-per-node=1 --cpus-per-task='+str(cpus)+' --gres=gpu:'+str(gpus)+' --time='+str(minutes)+' --chdir='+q(REMOTES[kind])+' --job-name=R22_'+kind+'_'+stage+' --output='+q(log+'.%j.out')+' --error='+q(log+'.%j.err')
    # The actual submit plugin rejects explicit --mem, including the default
    # amount. Preserve the tested automatic 15750 MiB per allocated CPU.
    if dependency:opts+=' --dependency=afterok:'+str(dependency)
    if formal:opts+=' --hold'
    opts+=' '+q(remote)
    with connection('gpu') as c:
        old.upload(c,local,remote);command(c,'bash -n '+q(remote));test=command(c,'sbatch --test-only '+opts+' 2>&1')
        write(intent,dict(command='sbatch --parsable '+opts,resolved=False,script_sha256=digest(local),test_only=test,epoch=time.time()))
        job=command(c,'sbatch --parsable '+opts).split(';')[0].strip()
        if not job.isdigit():raise RuntimeError('Ambiguous actual sbatch; preserve intent')
        rec=dict(job=int(job),stage=stage,host='gpu',nodes=nodes,gpus_per_node=gpus,output=output,
                 allocation=allocation.replace('{job}',job) if allocation else None,dependency=dependency,
                 held=formal,script_sha256=digest(local),submitted_epoch=time.time(),walltime_minutes=minutes)
        if kind=='derenzo':rec.update(dataset='JSCC_440',plan=read(dest/'validation_plan.json'))
        write(path,rec);write(intent,dict(read(intent),resolved=True,registered_job=int(job)))
        snap=command(c,'scontrol show job -o '+job);immutable(dest/(stage+'_submitted.txt'),snap.encode())
        if ('TresPerNode=gres/gpu:'+str(gpus)) not in snap:raise ValueError('Actual GPU request differs')
    print('REBALANCE_REGISTERED',kind,stage,job,flush=True);return rec


def supersede_pending():
    marker=ROOT/'superseded_pending.json'
    if marker.exists():return
    with connection('gpu') as c:
        before=command(c,'scontrol show job -o 1719262')
        if 'JobState=PENDING' not in before or old.ORIGINAL['nema'][2]+'/execution_repair_20261020/scripts/formal.sh' not in before:
            raise RuntimeError('Original formal is no longer the exact pending request; do not replace running science')
        write(ROOT/'hold_intent.json',dict(job=1719262,before=before,reason='Prevent duplicate production after user-authorized resource rebalance'))
        command(c,'scontrol hold 1719262');after=command(c,'scontrol show job -o 1719262')
        if 'JobState=PENDING' not in after or 'Reason=JobHeldUser' not in after:raise RuntimeError('Cannot prove old production held')
        write(marker,dict(job=1719262,before=before,after=after,epoch=time.time(),source_and_outputs_unchanged=True))


def fetch(kind,stage):
    with namespace(kind) as workflow:
        return workflow.fetch_stage(stage) if kind=='nema' else workflow.fetch_gpu('JSCC_440',stage)


def release(kind,formal):
    dest=folder(kind);marker=dest/'formal_release.json'
    if marker.exists():return
    accepted_path=dest/'validation_acceptance.json';accepted=read(accepted_path);proof_path=Path(accepted['local_result'])/'verification.json';proof=read(proof_path)
    bound=measured(kind,proof);source=read(dest/'validation_job.json');verifier=read(dest/('validation_verification_job.json' if kind=='nema' else 'validation_verify_job.json'))
    if time.time()+bound['minutes']*60>deadline(kind):raise ValueError('Measured formal would exceed original finite campaign deadline')
    cfg=read(ROOT/'nema_freeze.json') if kind=='nema' else read(dest/'validation_plan_deployment.json')
    science=cfg['release']+'/contract.json' if kind=='nema' else cfg['remote'];sha=cfg['contract_sha256'] if kind=='nema' else cfg['sha256']
    a=dict(passed=True,strict_fetch_passed=True,kind=kind,formal_job=formal['job'],validation_job=source['job'],
           policy_max_minutes=MAX_MINUTES,limits=bound,science_file=science,science_sha256=sha,
           strict_acceptance_file=REMOTES[kind]+'/strict_acceptance.json',strict_acceptance_sha256=digest(accepted_path),
           validation_proof_file=verifier['output']+'/verification.json',validation_proof_sha256=digest(proof_path))
    if kind=='nema':a.update(contract_sha256=sha,result=source['output'],evidence_sha256=proof['result_files_sha256'])
    else:a.update(dataset='JSCC_440',plan_sha256=sha,collection_sha256=proof['collection_sha256'],release_sha256=proof['release_sha256'],validation_result=source['output'],validation_files_sha256=proof['result_files_sha256'])
    auth=DATAS[kind]/'authority.json';immutable(auth,(json.dumps(a,indent=2)+'\n').encode())
    with connection('gpu') as c:
        old.upload(c,accepted_path,a['strict_acceptance_file']);old.upload(c,auth,REMOTES[kind]+'/authority.json')
        current=command(c,'scontrol show job -o '+str(formal['job']))
        if 'JobState=PENDING' not in current or 'JobHeldUser' not in current:raise RuntimeError('Formal release state ambiguous; inspect without resubmitting')
        command(c,'scontrol update JobId='+str(formal['job'])+' TimeLimit='+str(bound['minutes']))
        code='from resource_rebalance_gate_20261022 import check;check('+repr(REMOTES[kind]+'/authority.json')+','+str(formal['job'])+');print("STRICT_NEW_TOPOLOGY_AUTHORITY_PASS")'
        gate=command(c,'cd '+q(REMOTE+'/execution')+' && '+q(GPU_PYTHON)+' -B -c '+q(code),timeout=180)
        write(dest/'formal_release_intent.json',dict(job=formal['job'],gate=gate,authority_sha256=digest(auth)))
        command(c,'scontrol release '+str(formal['job']));after=command(c,'scontrol show job -o '+str(formal['job']))
        write(marker,dict(job=formal['job'],limits=bound,scontrol=after,epoch=time.time()))
    print('REBALANCE_FORMAL_RELEASED',kind,formal['job'],bound,flush=True)


def deadline(kind):
    recovery=DERENZO_OLD/'recovery_20261019'
    name='nema_production_5090_8x4_queue_registration.json' if kind=='nema' else 'derenzo_controller_registration.json'
    return read(recovery/name)['started_epoch']+(14 if kind=='nema' else 21)*86400


def run(watch):
    if (HERE.parents[1]/'docs/ac225/formal_jobs_current.json').exists():
        raise RuntimeError('Legacy automatic chain retired; only current formal jobs are authorized')
    old.no_other_controller()
    if live_registration(REG):raise RuntimeError('New rebalance controller already alive')
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    lock=DATA/'exclusive.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    write(REG,dict(pid=os.getpid(),started_epoch=time.time(),status='running',command=__file__,deadlines={k:deadline(k) for k in TOPOLOGY}))
    code=1
    try:
        prepare();supersede_pending();states={k:'waiting' for k in TOPOLOGY}
        while True:
            for kind in TOPOLOGY:
                if states[kind]!='waiting':continue
                try:
                    if time.time()>deadline(kind):raise TimeoutError('Original campaign deadline reached')
                    validation=submit(kind,'validation')
                    submit(kind,'validation_verification' if kind=='nema' else 'validation_verify',validation['job'],validation)
                    formal=submit(kind,'formal')
                    if not fetch(kind,'validation'):continue
                    release(kind,formal)
                    submit(kind,'formal_verification' if kind=='nema' else 'formal_verify',formal['job'],formal)
                    if fetch(kind,'formal'):states[kind]='numerically_accepted'
                except (TimeoutError,ConnectionError) as e:
                    if isinstance(e,TimeoutError) and 'campaign deadline' in str(e):raise
                    write(REPORTS[kind]/'read_only_wait.json',dict(error=str(e),epoch=time.time()))
                except Exception as e:
                    states[kind]='needs_attention';write(REPORTS[kind]/'controller_failure.json',dict(type=type(e).__name__,error=str(e),epoch=time.time(),all_outputs_preserved=True))
                    print('REBALANCE_CHAIN_NEEDS_ATTENTION',kind,str(e),flush=True)
            write(ROOT/'latest_progress.json',dict(epoch=time.time(),states=states,scientific_visual_qa_completed=False,automations_remain_paused=True))
            print('REBALANCE_PROGRESS',states,flush=True)
            if not watch or all(s!='waiting' for s in states.values()):break
            time.sleep(300)
        code=1 if 'needs_attention' in states.values() else 0
    finally:
        write(REG,dict(read(REG),status='complete',exit_code=code,finished_epoch=time.time()));lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if args.prepare_only:old.no_other_controller();prepare()
    else:run(args.watch)
