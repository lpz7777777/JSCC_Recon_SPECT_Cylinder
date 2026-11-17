"""One human-authorized v5 SC+Compton formal job using delivered 5e10 inputs.

Keeps previous releases/results unchanged, no transport/selection repetition,
no separate GPU validation jobs, no automation or automatic resubmission.
"""
import argparse
import ast
import hashlib
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from ehe_5e9_workflow import connection, command, q
from jscc_5e10_common import HERE, GPU_PROJECT, GPU_PYTHON, read, write, digest, hashes, verify_files
from prejune_8x1_archive_submit import pipe_command

TAG = 'nema_5e10_v5_joint_20261026'
DATA = HERE/'generated'/TAG
PAYLOAD = DATA/'payload'
REPORT = HERE/'reports/NEMA_Body_H60'/TAG
REMOTE = GPU_PROJECT+'/generated/'+TAG
PARENT = HERE/'generated/formal_only_20261022_135623/nema_payload'
PARENT_FREEZE = HERE/'reports/formal_only_20261022_135623/nema_freeze.json'
DELIVERED = HERE/'generated/nema_jscc5e10_formal_1722760_20261026'
CHANNEL = '440_SinglePlusCompton'
JOB_NAME = 'NEMA5e10_v5_joint'


def once(text, old, new):
    if text.count(old) != 1: raise ValueError('Source anchor not unique: '+old[:100])
    return text.replace(old, new)


def function_ast(text, name):
    return ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    marker = REPORT/'freeze.json'
    if marker.exists():
        frozen = read(marker); verify_files(PAYLOAD, frozen['files']); return frozen
    prior = read(PARENT_FREEZE); verify_files(PARENT, prior['files'])
    if PAYLOAD.exists(): raise RuntimeError('Incomplete local preparation exists; inspect before continuing')
    shutil.copytree(PARENT, PAYLOAD)
    original = (PARENT/'run_jscc_5e10.py').read_text('utf8')
    runner = once(original, '"""Actual 5e10, 8x4 GPUs, only 440 single/218 corrected/440 Compton."""',
        '"""Actual 5e10, v5 440 SC+Compton only, 8 nodes x 3 GPUs."""')
    runner = once(runner, 'from jscc_5e10_compton_mlem import compton_mlem',
        'from jscc_5e10_joint_mlem import joint_mlem')
    # Remove already-completed standalone SC, 218, and cross-prediction solves.
    start = runner.index("    if a.mode=='validation':all_operator_checks['A440']=operator_checks(response)")
    end = runner.index("    release_unused_cuda(a.output,rank,device,'before_compton_response')", start)
    runner = runner[:start]+"""    projection440=observations(440)
    if rank==0:sensitivity440.cpu().numpy().astype('<f4').tofile(a.output/'S_440.float32')
"""+runner[end:]
    runner = once(runner, "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>.8*int(os.environ['JSCC_HOST_BYTES_NODE'])/2",
        "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>.8*int(os.environ['JSCC_HOST_BYTES_NODE'])/3")
    runner = once(runner, 'Conservative four-rank node RSS operational guard fails', 'Conservative three-rank node RSS operational guard fails')
    start = runner.index('    prepare_seconds=time.monotonic()-prep_started;solve_started=time.monotonic();')
    end = runner.index('    record=resource_record(rank,local,device,started)', start)
    runner = runner[:start]+"""    if rank==0:(sensitivity440+sensitivity).cpu().numpy().astype('<f4').tofile(a.output/'S_Joint.float32')
    prepare_seconds=time.monotonic()-prep_started;solve_started=time.monotonic();progress('440_joint',0)
    imageJ,hJ=joint_mlem(response,projection440,blocks,sensitivity440,sensitivity,iterations,save_step,rank==0,
        (lambda i,h:checkpoint('440_joint',i,h)) if rank==0 else None,limit)
    phase_times['440_joint']=time.monotonic()-solve_started;compton_regression=None
    publish_images(a.output,CHANNELS[0],imageJ,hJ,geometry,rank)
"""+runner[end:]
    runner = once(runner, "joint_solver_enabled=False", "joint_solver_enabled=True")
    runner = once(runner, "cross_prediction_source='440_SinglePhoton_final'", "cross_prediction_source=None")
    runner = once(runner, "execution_authorization='human_direct_formal_only_20261022'",
        "execution_authorization='human_v5_joint_20261026_formal_only'")
    # Bind unchanged response generation, helper/checkpoint function ASTs.
    for name in ('relative', 'operator_checks', 'publish_images'):
        assert function_ast(original, name) == function_ast(runner, name)
    response_begin = '    B=full_rows(rawB,device);'
    response_end = '    prepare_seconds=time.monotonic()-prep_started;'
    before_response = original.split(response_begin)[1].split(response_end)[0]
    after_response = runner.split(response_begin)[1].split("    if rank==0:(sensitivity440+sensitivity)")[0]
    after_response = after_response.replace("/3:\n                    raise MemoryError('Conservative three-rank", "/2:\n                    raise MemoryError('Conservative four-rank")
    assert before_response == after_response
    (PAYLOAD/'run_jscc_5e10.py').write_text(runner, encoding='utf8', newline='\n')
    shutil.copyfile(HERE/'jscc_5e10_joint_mlem.py', PAYLOAD/'jscc_5e10_joint_mlem.py')
    common = (PARENT/'jscc_5e10_common.py').read_text('utf8')
    common = once(common, "STUDY = 'jscc_geant4_5e10_10000'", "STUDY = "+repr(TAG)+"\nTRANSPORT_STUDY = 'jscc_geant4_5e10_10000'")
    common = once(common, "CHANNELS = ('440_SinglePhoton', '218_SinglePhoton_CrossTalkCorrected', '440_ComptonOnly')", "CHANNELS = ('440_SinglePlusCompton',)")
    common = once(common, "PHASE_CHANNELS = dict(zip(('440_single', '218_corrected', '440_compton'), ((c,) for c in CHANNELS)))", "PHASE_CHANNELS = {'440_joint': CHANNELS}")
    common = once(common, "c['study']!=STUDY", "c['study']!=TRANSPORT_STUDY")
    (PAYLOAD/'jscc_5e10_common.py').write_text(common, encoding='utf8', newline='\n')
    runtime = (PARENT/'jscc_5e10_runtime.py').read_text('utf8')
    runtime = once(runtime, 'world!=16 or not 0<=local<2 or torch.cuda.device_count()!=2', 'world!=24 or not 0<=local<3 or torch.cuda.device_count()!=3')
    runtime = once(runtime, 'Frozen 8-node x4-GPU topology required', 'Frozen 8-node x3-GPU topology required')
    runtime = once(runtime, 'result=[None]*16', 'result=[None]*24')
    (PAYLOAD/'jscc_5e10_runtime.py').write_text(runtime, encoding='utf8', newline='\n')
    validator = (PARENT/'jscc_5e10_contract.py').read_text('utf8')
    validator = once(validator, "cfg['gpus_per_node']!=2", "cfg['gpus_per_node']!=3")
    validator = once(validator, "cfg['world_size']!=16", "cfg['world_size']!=24")
    validator = once(validator, "cfg['cross_prediction_source']!='440_SinglePhoton_final' or cfg['joint_solver_enabled']", "cfg['cross_prediction_source'] is not None or not cfg['joint_solver_enabled']")
    validator = once(validator, "    if (iterations,save_step)!=execution_policy(mode):", "    if mode!='formal':raise ValueError('Only explicitly authorized formal joint reconstruction')\n    if (iterations,save_step)!=execution_policy(mode):")
    validator = once(validator, 'len(resources)!=16 or sorted(r[\'rank\'] for r in resources)!=list(range(16))', 'len(resources)!=24 or sorted(r[\'rank\'] for r in resources)!=list(range(24))')
    validator = once(validator, 'gres/gpu=16', 'gres/gpu=24')
    validator = once(validator, "list(range(2)) or len({r['gpu_uuid'] for r in group})!=2", "list(range(3)) or len({r['gpu_uuid'] for r in group})!=3")
    validator = validator.replace('All 32 rank identities', 'All 24 rank identities').replace('contain 32 GPUs', 'contain 24 GPUs').replace('Four different actual GPUs', 'Three different actual GPUs')
    assert function_ast(validator, 'write_checkpoint') == function_ast((PARENT/'jscc_5e10_contract.py').read_text('utf8'), 'write_checkpoint')
    (PAYLOAD/'jscc_5e10_contract.py').write_text(validator, encoding='utf8', newline='\n')
    cfg = read(PAYLOAD/'contract.json')
    cfg.update(study=TAG, source_study='jscc_geant4_5e10_10000', channels=[CHANNEL], joint_solver_enabled=True,
        cross_prediction_source=None, nodes=8, gpus_per_node=3, world_size=24,
        source_formal_job=1722760, new_transport=False, new_selection=False,
        conservative_expected_memory_per_node=cfg['actual_event_rows_cpu_bytes']/8+3*(12<<30),
        authorization='2026-10-26 human request: new-flow NEMA 5e10 SC+Compton joint reconstruction',
        joint_update='x *= allreduce(SC_weight(x) + Compton_weight(x)) / (S_SC + S_Compton)',
        parent_contract_sha256=prior['files']['contract.json'])
    changed = ['run_jscc_5e10.py', 'jscc_5e10_common.py', 'jscc_5e10_runtime.py', 'jscc_5e10_contract.py', 'jscc_5e10_joint_mlem.py']
    for name in changed: cfg['files'][name] = digest(PAYLOAD/name)
    write(PAYLOAD/'contract.json', cfg)
    for p in PAYLOAD.glob('*.py'): compile(p.read_text('utf8'), str(p), 'exec')
    unchanged = {n:s for n,s in prior['files'].items() if n not in changed+['contract.json','verification_release.json']}
    verify_files(PAYLOAD, unchanged)
    payload_hashes = hashes(PAYLOAD); payload_hashes.pop('verification_release.json')
    write(PAYLOAD/'verification_release.json', dict(sha256=payload_hashes, parent_freeze_sha256=digest(PARENT_FREEZE),
        new_topology_full_input_validation_passed=False, separate_validation_required=False))
    files = hashes(PAYLOAD); key = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    old_manifest = read(DELIVERED/'result/run_manifest.json')
    times = old_manifest['resources']
    compton = max(x['phase_solve_seconds']['440_compton'] for x in times)
    single = max(x['phase_solve_seconds']['440_single'] for x in times)
    prep = max(x['prepare_seconds'] for x in times)
    phase_limit = math.ceil(1.5*(compton+single))
    # No speedup credited to 24 vs 16 GPUs; preparation budget counted once.
    prepare_limit = max(10800, math.ceil(2*prep))
    initialization_limit = 7200
    total_limit = phase_limit+prepare_limit+initialization_limit
    minutes = math.ceil((total_limit+600)/60)
    frozen = dict(key=key, files=files, unchanged_files=unchanged, parent_release=prior['release'],
        parent_freeze_sha256=digest(PARENT_FREEZE), contract_sha256=digest(PAYLOAD/'contract.json'),
        unchanged_response_generation=True, unchanged_checkpoint_function_ast=True,
        all_40_selection_arrays_unchanged=True, phase_seconds=phase_limit, prepare_seconds=prepare_limit,
        total_seconds=total_limit, walltime_minutes=minutes,
        walltime_basis=dict(source_job=1722760, compton_seconds=compton, single_seconds=single,
            preparation_seconds=prep, solver_factor=1.5, assumed_24gpu_speedup=1, initialization_budget_seconds=initialization_limit,
            note='Bounded scheduler request, not completion ETA; one response preparation'))
    auth = {k:cfg[k] for k in ('study','channels','iterations','save_step','nodes','gpus_per_node','world_size')}
    auth.update(policy='human_direct_formal_only_20261022', human_instruction='新流程下的5e10 NEMA SC+Compton联合重建',
        scope_authorized_at='2026-10-26', contract_sha256=frozen['contract_sha256'], separate_validation_required=False,
        new_topology_full_input_validation_passed=False, source_formal_job=1722760,
        actual_primary_photons=50000000000, accepted_events=4849087, input_sha256=cfg['input_sha256'],
        factor_payload_sha256=cfg['factor_payload_sha256'])
    write(DATA/'authorization.json',auth); write(marker,frozen)
    verify_files(PARENT, prior['files'])
    return frozen


def build_delivery(frozen):
    archive=DATA/'frozen_payload.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in frozen['files']:
                info=zipfile.ZipInfo(name,date_time=(2026,10,26,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,(PAYLOAD/name).read_bytes())
            z.writestr('authorization.json',(DATA/'authorization.json').read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(frozen['files'])|{'authorization.json'}
        for n,s in frozen['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==s
        assert hashlib.sha256(z.read('authorization.json')).hexdigest()==digest(DATA/'authorization.json')
    archive_sha=digest(archive);remote_archive=REMOTE+'/releases/'+frozen['key']+'.zip'
    unpack="""import hashlib,json,pathlib,sys,zipfile,os
p=pathlib.Path(sys.argv[1]);target=pathlib.Path(sys.argv[2])
assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[3]
assert not target.exists()
with zipfile.ZipFile(p) as z:
 for n in z.namelist():
  assert not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts
 z.extractall(target)
assert hashlib.sha256((target/'contract.json').read_bytes()).hexdigest()==sys.argv[4]
c=json.loads((target/'contract.json').read_text())
for n,s in c['files'].items():assert hashlib.sha256((target/n).read_bytes()).hexdigest()==s,n
print('NODE_SOURCE_UNPACK_SHA_OK',os.environ['SLURM_PROCID'],flush=True)
"""
    node='''set -euo pipefail
export TMPDIR="/tmp/nema5e10_joint_${SLURM_JOB_ID}_${SLURM_PROCID}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" CUDA_CACHE_PATH="$TMPDIR/cuda" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
'''
    node+=q(GPU_PYTHON)+' -B -c '+q(unpack)+' '+q(remote_archive)+' "$TMPDIR/release" '+q(archive_sha)+' '+q(frozen['contract_sha256'])+'\n'
    node+='export JSCC_PROJECT_ROOT="$TMPDIR/release"\ncd "$TMPDIR/release"\n'
    node+='exec '+q(GPU_PYTHON)+' -B -m torch.distributed.run --nnodes=8 --nproc_per_node=3 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_backend=static --rdzv_id="${SLURM_JOB_ID}_v5joint" --max_restarts=0 "$TMPDIR/release/run_jscc_5e10.py" --mode formal --contract "$TMPDIR/release/contract.json" --authority "$TMPDIR/release/authorization.json" --authority-sha256 '+q(digest(DATA/'authorization.json'))+' --input-root '+q(GPU_PROJECT+'/generated/jscc_geant4_5e10_10000/input')+' --factors '+q(GPU_PROJECT+'/generated/FactorsCalibrated')+' --output '+q(REMOTE)+'/formal_"${SLURM_JOB_ID}" --allocation '+q(REMOTE)+'/allocations/formal_"${SLURM_JOB_ID}".txt\n'
    memory_parser="import re,sys;s=sys.stdin.read();n=int(re.search(r'\\bNumNodes=(\\d+)',s)[1]);assert n==8;t=re.search(r'\\bAllocTRES=([^\\s]+)',s)[1];m=re.search(r'(?:^|,)mem=([0-9.]+)([KMGT])(?:,|$)',t);print(int(float(m[1])*1024**('KMGT'.index(m[2])+1)/n))"
    script='''#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
module load miniforge3/25.11.0-1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 NCCL_SOCKET_IFNAME=bond0
'''
    script+='[[ "$SLURM_NNODES" == 8 && "$SLURM_NTASKS" == 8 ]]\n'
    script+='scontrol show job "$SLURM_JOB_ID" > '+q(REMOTE)+'/allocations/formal_"${SLURM_JOB_ID}".txt\n'
    script+='export JSCC_HOST_BYTES_NODE=$(scontrol show job "$SLURM_JOB_ID" | '+q(GPU_PYTHON)+' -B -c '+q(memory_parser)+')\n'
    script+='export JSCC_PHASE_SECONDS='+str(frozen['phase_seconds'])+' JSCC_PREPARE_SECONDS='+str(frozen['prepare_seconds'])+' JSCC_TOTAL_SECONDS='+str(frozen['total_seconds'])+'\n'
    script+='export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)\nexport MASTER_PORT=$((40000+SLURM_JOB_ID%10000))\n'
    script+='timeout --signal=TERM --kill-after=60s "${JSCC_TOTAL_SECONDS}s" srun --label --kill-on-bad-exit=1 bash -c '+q(node)+'\n'
    launch=DATA/'formal.sh'
    if launch.exists():assert launch.read_text('utf8')==script
    else:launch.write_text(script,encoding='utf8',newline='\n')
    result=dict(remote_archive=remote_archive,archive_sha256=archive_sha,archive_bytes=archive.stat().st_size,
        script_sha256=digest(launch),contract_sha256=frozen['contract_sha256'],authorization_sha256=digest(DATA/'authorization.json'))
    write(REPORT/'delivery.json',result)
    return archive,launch,result


def submit():
    f=prepare()
    if (REPORT/'formal_job.json').exists():return read(REPORT/'formal_job.json')
    if (REPORT/'submission_intent.json').exists():raise RuntimeError('Unresolved submission; inspect Slurm, never blindly retry')
    if not (REPORT/'local_joint_regression.json').exists() or not read(REPORT/'local_joint_regression.json')['passed']:
        raise RuntimeError('Require local comparison against frozen v5 joint branch before deployment')
    assert read(REPORT/'local_joint_regression.json')['joint_helper_sha256']==digest(PAYLOAD/'jscc_5e10_joint_mlem.py')
    archive,launch,d=build_delivery(f)
    with connection('gpu') as c:
        queue=command(c,"squeue -h -u scxi717 -o '%i|%j|%T|%R'",45)
        if JOB_NAME in queue:raise RuntimeError('Same named job exists; reconcile before submission')
        (REPORT/'queue_before.txt').write_text(queue,encoding='utf8')
        command(c,'mkdir -p '+q(REMOTE+'/releases')+' '+q(REMOTE+'/logs')+' '+q(REMOTE+'/allocations'),45)
        code="""import hashlib,os,pathlib,sys
p=pathlib.Path(%r);expected=%r
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()==expected
s=os.statvfs(p.parent);assert s.f_bavail*s.f_frsize>max(16*1024**3,2*len(data))
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
else:
 with p.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print('ARCHIVE_PERSISTENT_SHA_OK',len(data),flush=True)
"""%(d['remote_archive'],d['archive_sha256'])
        proof,_=pipe_command(c,q(GPU_PYTHON)+' -B -c '+q(code),archive.read_bytes(),300)
        # Same local regression runs on actual Linux CPU inside the archive;
        # no extra Slurm job and no assertion of full-input GPU validation.
        checkcode="""import ast,hashlib,json,sys,zipfile,pathlib
p=pathlib.Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(p) as z:
 for n in z.namelist():
  if n.endswith('.py'):compile(z.read(n).decode('utf8'),n,'exec')
 assert hashlib.sha256(z.read('contract.json')).hexdigest()==%r
sys.path.insert(0,str(p))
import run_jscc_5e10,jscc_5e10_joint_mlem
print('LINUX_IMPORT_COMPILE_SHA_OK',sys.version,flush=True)
"""%(d['remote_archive'],d['archive_sha256'],f['contract_sha256'])
        check,_=pipe_command(c,'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+q(GPU_PYTHON)+' -B -c '+q(checkcode),b'',120)
        # Regression source receives the immutable archive import path.
        regression=(HERE/'test_jscc_5e10_joint_mlem.py').read_text('utf8')
        regression=regression.replace("SOURCE = Path(__file__).resolve().parent", "SOURCE = Path("+repr(d['remote_archive'])+")")
        linux,_=pipe_command(c,'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+q(GPU_PYTHON)+' -B -c '+q(regression),b'',120)
        if 'JOINT_NUMERICAL_REGRESSION_PASS' not in linux:raise RuntimeError('Linux regression failed')
        pipe_command(c,'bash -n',launch.read_bytes(),30)
        opts='-p gpu_5090 --qos=gpugpu -N 8 -n 8 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:3 --time='+str(f['walltime_minutes'])+' --job-name='+JOB_NAME+' --chdir='+q(GPU_PROJECT)+' --output='+q(REMOTE+'/logs/formal.%j.out')+' --error='+q(REMOTE+'/logs/formal.%j.err')
        out,err=pipe_command(c,'sbatch --test-only '+opts,launch.read_bytes(),60)
        write(REPORT/'deployment.json',dict(passed=True,archive_proof=proof,linux_import=check,linux_cpu_regression=linux,
            scheduler_test_only=out+err,full_input_gpu_validation_performed=False,separate_validation_jobs=0,delivery=d))
        write(REPORT/'submission_intent.json',dict(command='sbatch --parsable '+opts,script_sha256=digest(launch),time=time.time(),resolved=False))
        reply,_=pipe_command(c,'sbatch --parsable '+opts,launch.read_bytes(),60)
        job=reply.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('Ambiguous submission, inspect scheduler; never retry: '+reply)
        rec=dict(job=int(job),study=TAG,stage='formal',nodes=8,gpus_per_node=3,world_size=24,cpus_per_node=24,
            channels=[CHANNEL],iterations=10000,save_step=50,initial_density=1,regularization='none',
            actual_primary_photons=50000000000,selected_events=4849087,model='continuous_energy_v5',
            output=REMOTE+'/formal_'+job,release_archive=d['remote_archive'],release_archive_sha256=d['archive_sha256'],
            contract_sha256=f['contract_sha256'],script_sha256=digest(launch),authorization_sha256=digest(DATA/'authorization.json'),
            separate_validation_required=False,full_input_gpu_validation_performed=False,walltime_minutes=f['walltime_minutes'],
            source_jobs=[1722760,1686507,1686546],new_transport=False,new_selection=False,submitted_epoch=time.time())
        write(REPORT/'formal_job.json',rec)
        write(REPORT/'submission_intent.json',dict(read(REPORT/'submission_intent.json'),resolved=True,registered_job=int(job)))
        status=command(c,'scontrol show job '+job,45)
        (REPORT/'submitted_status.txt').write_text(status,encoding='utf8')
        assert 'TresPerNode=gres/gpu:3' in status and 'Dependency=(null)' in status
        spool=command(c,'scontrol write batch_script '+job+' -',45)
        if spool.startswith('Batch script for job '):spool=spool.split('\n',1)[1]
        assert spool.encode()==launch.read_bytes(), 'Actual scheduler script differs'
        (REPORT/'slurm_spooled_script.sh').write_bytes(spool.encode())
        write(REPORT/'scheduler_script_check.json',dict(passed=True,job=int(job),sha256=digest(launch)))
        print(status,flush=True)
    return rec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    result=submit() if a.submit else prepare()
    print(json.dumps(result if a.submit else {k:v for k,v in result.items() if k not in ('files','unchanged_files')},indent=2))
