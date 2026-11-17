"""One-shot NEMA formal submission; never creates validation/acceptance jobs.

Human instruction on 2026-10-22 supersedes the former separate validation gate.
Retains the cache repair, numerical kernels, all selected rows and checkpoints.
Existing running Derenzo formal1722673 is retained without resubmission.
"""
import argparse
import ast
import hashlib
import json
import shutil
import time
from pathlib import Path
from ehe_5e9_workflow import connection, command, q
from jscc_5e10_common import HERE, GPU_PROJECT, GPU_PYTHON, read, write, digest, hashes, verify_files
from reconstruction_execution_repair_20261020 import immutable, upload, tmp_environment, GATE_NAME
from resource_rebalance_20261022_v3 import replace_once, ast_function

TAG = 'formal_only_20261022_135623'
REPORT = HERE/'reports'/TAG
DATA = HERE/'generated'/TAG
REMOTE = GPU_PROJECT+'/generated/'+TAG
SOURCE = HERE/'generated/nema_vram_repair_20261022/nema_payload'
PRIOR = HERE/'reports/nema_vram_repair_20261022/nema_freeze.json'
AUTH_HELPER = 'formal_only_authorization_20261022.py'


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA/'nema_payload'
    marker = REPORT/'nema_freeze.json'
    if marker.exists():
        freeze = read(marker); verify_files(dest, freeze['files']); return freeze
    prior = read(PRIOR); verify_files(SOURCE, prior['files'])
    shutil.copytree(SOURCE, dest)
    original = (SOURCE/'run_jscc_5e10.py').read_text()
    before = """    if a.mode=='formal':
        if digest(a.authority)!=a.authority_sha256:raise ValueError('Full-input authority SHA differs')
        authority=read(a.authority)
        if not authority['passed'] or authority['contract_sha256']!=digest(a.contract):raise ValueError('Actual current input validation authority required')
        for n,sha in authority['evidence_sha256'].items():
            if digest(Path(authority['result'])/n)!=sha:raise ValueError('Accepted validation evidence changed')"""
    after = """    if a.mode=='formal':
        from formal_only_authorization_20261022 import check as direct_formal_check
        authority=direct_formal_check(a.authority,a.authority_sha256,a.contract)"""
    updated = replace_once(original, before, after)
    updated = replace_once(updated, "actual_primary_photons=TOTAL,", "actual_primary_photons=TOTAL,execution_authorization='human_direct_formal_only_20261022',new_topology_full_input_validation_passed=False,")
    ast.parse(updated)
    for name in ('relative','operator_checks','publish_images'):
        if ast_function(original,name) != ast_function(updated,name):
            raise ValueError('Scientific helper changed')
    # The complete main after authorization and before manifest must be identical.
    body = "    rank,world,local,device=setup();error=[None]"
    stop = "        write(a.output/'run_manifest.json'"
    if original.split(body)[1].split(stop)[0] != updated.split(body)[1].split(stop)[0]:
        raise ValueError('Scientific solve body changed')
    (dest/'run_jscc_5e10.py').write_bytes(updated.encode())
    shutil.copy2(HERE/AUTH_HELPER, dest/AUTH_HELPER)
    cfg = read(dest/'contract.json'); old_cfg = dict(cfg)
    cfg['files'] = dict(cfg['files'], **{name:digest(dest/name) for name in ('run_jscc_5e10.py', AUTH_HELPER)})
    if any(cfg[k] != old_cfg[k] for k in cfg if k != 'files'):
        raise ValueError('Scientific configuration changed')
    write(dest/'contract.json', cfg)
    unchanged = {n:s for n,s in prior['files'].items() if n not in ('run_jscc_5e10.py','contract.json','verification_release.json')}
    verify_files(dest, unchanged)
    files = hashes(dest); files.pop('verification_release.json')
    write(dest/'verification_release.json',dict(sha256=files,source_freeze_sha256=digest(PRIOR),
          execution_authorization='human_direct_formal_only_20261022',new_topology_full_input_validation_passed=False))
    files = hashes(dest)
    key = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    freeze = dict(release=REMOTE+'/science/'+key,source_release=prior['release'],files=files,
        changed=['run_jscc_5e10.py','contract.json','verification_release.json',AUTH_HELPER],
        contract_sha256=digest(dest/'contract.json'),unchanged_files=unchanged,
        complete_scientific_solve_body_unchanged=True,all_40_selection_arrays_unchanged=True,
        new_topology_full_input_validation_passed=False,separate_validation_required=False)
    write(marker,freeze)
    auth = {k:cfg[k] for k in ('study','channels','iterations','save_step','nodes','gpus_per_node','world_size')}
    auth.update(policy='human_direct_formal_only_20261022',contract_sha256=freeze['contract_sha256'],
        separate_validation_required=False,new_topology_full_input_validation_passed=False,
        human_instruction='Only formal jobs; cancel other jobs; resubmit formal if it fails.',
        actual_primary_photons=cfg['total_primary_photons'],event_count=cfg['accepted_events'],
        input_sha256=cfg['input_sha256'],factor_payload_sha256=cfg['factor_payload_sha256'],
        bounded_walltime_minutes=14400,walltime_basis='conservative ten-day cap; no new 16-GPU solve timing claimed')
    write(DATA/'authorization.json',auth)
    return freeze


def script(freeze):
    root=REMOTE+'/nema';science=freeze['release'];env=REMOTE+'/execution/nema_tmp.sh'
    auth=REMOTE+'/authorization.json'
    s='#!/usr/bin/env bash\nset -euo pipefail\nsource '+q(env)+'\nsource /etc/profile.d/modules.sh\nmodule load cuda/12.9\nmodule load miniforge3/25.11.0-1\n'
    s+='export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0\n'
    s+='export JSCC_PHASE_SECONDS=850000 JSCC_PREPARE_SECONDS=10800 JSCC_TOTAL_SECONDS=863400\n'
    s+='export JSCC_PROJECT_ROOT='+q(science)+'\ncd '+q(science)+'\n'
    s+='[[ "$SLURM_NNODES" == 8 && "$SLURM_NTASKS" == 8 ]]\n'
    s+='allocation='+q(root+'/allocations/formal_')+'"${SLURM_JOB_ID}.txt"\nscontrol show job "$SLURM_JOB_ID" > "$allocation"\n'
    code='from jscc_5e10_common import host_allocated_bytes;import sys;print(host_allocated_bytes(open(sys.argv[1]).read(),8))'
    s+='export JSCC_HOST_BYTES_NODE=$('+q(GPU_PYTHON)+' -B -c '+q(code)+' "$allocation")\n'
    s+='export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)\nexport MASTER_PORT=$((40000+SLURM_JOB_ID%10000))\n'
    args=[science+'/run_jscc_5e10.py','--contract',science+'/contract.json','--input-root',GPU_PROJECT+'/generated/jscc_geant4_5e10_10000/input',
          '--factors',GPU_PROJECT+'/generated/FactorsCalibrated','--output',root+'/formal','--allocation','$allocation',
          '--mode','formal','--authority',auth,'--authority-sha256',digest(DATA/'authorization.json')]
    inv=' '.join('"'+a+'"' if a.startswith('$') else q(a) for a in args)
    s+='timeout --signal=TERM --kill-after=60s "${JSCC_TOTAL_SECONDS}s" srun --label --chdir='+q(root)+' --kill-on-bad-exit=1 bash -c \'source '+q(env)+'; exec '+q(GPU_PYTHON)+' -B -m torch.distributed.run --nnodes=8 --nproc_per_node=2 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_backend=static --rdzv_id="${SLURM_JOB_ID}_formal_only" --max_restarts=0 "${@:1}"\' bash '+inv+'\n'
    return s


def deploy_and_submit():
    f=prepare();root=REMOTE+'/nema';record=REPORT/'nema_formal_job.json'
    if record.exists(): return read(record)
    intent=REPORT/'nema_submission_intent.json'
    if intent.exists(): raise RuntimeError('Unresolved submission; inspect Slurm before any retry')
    execution=DATA/'execution';execution.mkdir(exist_ok=True)
    # Reuse only the job-private TMP check; no authority/validation gate is called.
    for name in ('reconstruction_execution_gate_20261020.py',):
        immutable(execution/name,(HERE/name).read_bytes())
    immutable(execution/'nema_tmp.sh',tmp_environment(REMOTE+'/execution',root).encode())
    launch=DATA/'nema_formal.sh';immutable(launch,script(f).encode())
    opts='-p gpu_5090 --qos=gpugpu -N 8 -n 8 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:2 --time=14400 --chdir='+q(root)+' --job-name=NEMA5e10_formal_3ch --output='+q(root+'/logs/formal.%j.out')+' --error='+q(root+'/logs/formal.%j.err')+' '+q(root+'/scripts/formal.sh')
    with connection('gpu') as c:
        queue=command(c,"squeue -h -u scxi717 -o '%i|%j|%T'",timeout=60)
        if not queue or any(line.split('|')[0]!='1722673' for line in queue.splitlines() if line.strip()):
            raise RuntimeError('Expected only existing Derenzo formal before unique NEMA submit: '+queue)
        for d in (f['release'],REMOTE+'/execution',root,root+'/logs',root+'/scripts',root+'/allocations'):
            command(c,'mkdir -p '+q(d))
        if not (REPORT/'deployment.json').exists():
            command(c,'if [ ! -e '+q(f['release']+'/contract.json')+' ]; then cp -a '+q(f['source_release']+'/.')+' '+q(f['release'])+'/; fi',timeout=180)
            # This new, unused destination initially contains the old release's
            # copied bytes. Replace only those exact source bytes or accept the
            # already-deployed target; never change any prior release in place.
            prior=read(PRIOR)
            with c.open_sftp() as sftp:
                for name in f['changed']:
                    remote=f['release']+'/'+name
                    try:
                        with sftp.open(remote,'rb') as stream: current=hashlib.sha256(stream.read()).hexdigest()
                    except FileNotFoundError: current=None
                    if current==f['files'][name]: continue
                    if current!=prior['files'].get(name): raise ValueError('Unexpected new-release member: '+name)
                    sftp.put(str(DATA/'nema_payload'/name),remote)
        for p in execution.iterdir(): upload(c,p,REMOTE+'/execution/'+p.name)
        upload(c,DATA/'authorization.json',REMOTE+'/authorization.json');upload(c,launch,root+'/scripts/formal.sh')
        code='from pathlib import Path;from jscc_5e10_common import *;from formal_only_authorization_20261022 import check;r=Path('+repr(f['release'])+');verify_files(r,read(r/"verification_release.json")["sha256"]);check('+repr(REMOTE+'/authorization.json')+','+repr(digest(DATA/'authorization.json'))+',r/"contract.json");print("DIRECT_FORMAL_BINDINGS_OK")'
        proof=command(c,'cd '+q(f['release'])+' && '+q(GPU_PYTHON)+' -B -c '+q(code)+' 2>&1',timeout=180)
        if 'DIRECT_FORMAL_BINDINGS_OK' not in proof: raise RuntimeError('Deployment binding failed: '+proof)
        write(REPORT/'deployment.json',dict(release=f['release'],proof=proof,execution_files=hashes(execution),script_sha256=digest(launch)))
        command(c,'bash -n '+q(root+'/scripts/formal.sh'))
        test=command(c,'sbatch --test-only '+opts+' 2>&1',timeout=60)
        if 'error' in test.lower(): raise RuntimeError('Scheduler request rejected: '+test)
        write(intent,dict(command='sbatch --parsable '+opts,epoch=time.time(),resolved=False,test_only=test,script_sha256=digest(launch)))
        job=command(c,'sbatch --parsable '+opts,timeout=60).split(';')[0].strip()
        if not job.isdigit():raise RuntimeError('Ambiguous submit, inspect scheduler; do not retry: '+job)
        rec=dict(job=int(job),stage='formal',condition='NEMA_JSCC_actual5e10_three_channels',nodes=8,gpus_per_node=2,
             channels=read(DATA/'authorization.json')['channels'],iterations=10000,save_step=50,output=root+'/formal',
             release=f['release'],contract_sha256=f['contract_sha256'],authorization_sha256=digest(DATA/'authorization.json'),
             held=False,dependency=None,separate_validation_required=False,walltime_minutes=14400,submitted_epoch=time.time(),
             script_sha256=digest(launch))
        write(record,rec);write(intent,dict(read(intent),resolved=True,registered_job=int(job)))
        snap=command(c,'scontrol show job -o '+job,timeout=60)
        immutable(REPORT/'nema_formal_submitted.txt',snap.encode())
        if 'TresPerNode=gres/gpu:2' not in snap or 'Dependency=(null)' not in snap:
            raise RuntimeError('Actual formal scheduler request differs')
    return rec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(deploy_and_submit() if a.submit else prepare(),indent=2))
