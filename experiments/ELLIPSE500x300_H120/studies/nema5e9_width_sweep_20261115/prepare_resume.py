"""Freeze a distinct continuation; old files and acceptance evidence stay immutable."""
import ast,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from prepare import CODE,DATA,REPORT,sha,read,write
from workflow import connect,command,script,now,FREEZE

def replace(source,old,new):
    if source.count(old)!=1:raise ValueError('Expected unique source anchor: '+old[:100])
    return source.replace(old,new)

def audit():
    old=read(REPORT/'formal_registration_r4.json')
    with connect() as c:
        state=command(c,'sacct -n -P -j '+old['job']+' --format=JobID,State,ExitCode,Elapsed',45)
        rows=[s.split('|') for s in state.splitlines() if s.strip()]
        assert {old['job'],old['job']+'.batch',old['job']+'.extern',old['job']+'.0'}<={r[0] for r in rows}
        assert all(r[1] in ('FAILED','COMPLETED','CANCELLED') for r in rows),state
        queue=command(c,"squeue -u scxi717 -h -o '%i|%j|%T|%D|%b'",45)
        assert 'NEMA5e9_width' not in queue,queue
        code='''import pathlib,json,hashlib,numpy as np
p=pathlib.Path(PARENT);r=pathlib.Path(RELEASE)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'release_manifest.json').read_text())
for n,s in m['files'].items():assert sha(r/n)==s,n
cp=json.loads((p/'width100/checkpoint_manifest.json').read_text());assert cp['iteration']==9950 and len(cp['sha256'])==199
assert len(list((p/'width100/checkpoints').glob('*.float32')))==199
assert not (p/'width80').exists() and not (p/'width60').exists() and not (p/'completed.json').exists()
for step in range(50,9951,50):
 n=f'checkpoints/iteration_{step:05d}.float32';f=p/'width100'/n
 assert sha(f)==cp['sha256'][n]
 a=np.fromfile(f,'<f4');assert a.shape==(78920,) and np.isfinite(a).all() and (a>=0).all()
choice=json.loads((p/'width100/benchmark_choice.json').read_text());assert choice['chosen']=='grouped_sync' and all(choice['bitwise_gates'].values())
files={f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
rf=dict(m['files']);rf['release_manifest.json']=sha(r/'release_manifest.json')
print(json.dumps(dict(parent_files=files,parent_release_files=rf,checkpoint_manifest=cp,benchmark_choice=choice)))
'''.replace('PARENT',repr(old['result'])).replace('RELEASE',repr(old['release']))
        proof=json.loads(script(c,code,120))
    return dict(proof,time=now(),parent_job=old['job'],parent_result=old['result'],parent_release=old['release'],
        parent_execution_key=old['execution_key'],parent_accounting=state,other_queue_readonly=queue,
        completed_iterations=9950,active_points=78920,save_step=50,remaining_iterations={'100':50,'80':10000,'60':10000},
        historical_comparison_policy='Record original 1e-5 threshold and actual error; nonfinite remains fatal. User explicitly accepts the historical discrepancy; do not relabel it as a passing regression.',
        user_authorization='不纠结这个差异。继续任务，把三个百分比的都算完之后对比',
        authorization_local_date='2026-11-15',old_benchmark_reused=True,old_successful_iterations_recomputed=False)

def helper(source):
    source=replace(source,'checkpoint_callback=None, phase_limit_seconds=None):',
        'checkpoint_callback=None, phase_limit_seconds=None, *, initial_image=None,\n               initial_history=None, completed_iterations=0):')
    source=replace(source,'    history = []\n    started = time.monotonic()', '''    history = []
    if completed_iterations:
        if completed_iterations < 0 or completed_iterations >= iterations or completed_iterations % save_step:
            raise ValueError('Invalid completed iteration boundary')
        if initial_image is None or initial_image.shape != (n,1) or initial_image.dtype != torch.float32:
            raise ValueError('Invalid resume image')
        if not bool(torch.isfinite(initial_image).all()) or bool((initial_image<0).any()):
            raise ValueError('Invalid resume density')
        image=initial_image.detach().to(device).clone()
        if save_history:
            if initial_history is None or len(initial_history)!=completed_iterations//save_step:
                raise ValueError('Incomplete initial history')
            history=[h.detach().cpu().clone() for h in initial_history]
            if any(h.shape!=(n,1) or h.dtype!=torch.float32 or not bool(torch.isfinite(h).all()) or bool((h<0).any()) for h in history):
                raise ValueError('Invalid initial history')
            if not torch.equal(history[-1],image.cpu()):raise ValueError('Resume image/history differ')
    elif initial_image is not None or initial_history:
        raise ValueError('Initial state supplied without completed iterations')
    started = time.monotonic()''')
    return replace(source,'for iteration in range(iterations):','for iteration in range(completed_iterations, iterations):')

def runtime(source):
    source=replace(source,'from jscc_5e10_joint_mlem import joint_mlem','from joint_mlem_resume import joint_mlem\nfrom resume_support import verify_parent,inherit_branch,inherited_sensitivity')
    source=replace(source,"    rank=int(os.environ['RANK']);", "    plan=json.loads((root/'resume_plan.json').read_text())\n    rank=int(os.environ['RANK']);")
    source=replace(source,'            a.output.mkdir(parents=True,exist_ok=False)',
        "            verify_parent(plan)\n            a.output.mkdir(parents=True,exist_ok=False)\n            save_json(a.output/'resume_lineage.json',plan)")
    source=replace(source,"    counts=np.loadtxt(","    ss=inherited_sensitivity(plan,ss,rank,a.output)\n    counts=np.loadtxt(")
    source=replace(source,'    choice=None',"    choice=plan['benchmark_choice']['chosen']")
    source=replace(source,"        if scale==1.:choice=benchmark(root,out,response,projection,blocks,ss,sc,geo,rank,device,a.host_bytes)",
        "        # Previous real GPU paired benchmark is inherited, not repeated.")
    source=replace(source,'        def checkpoint(iteration,history):',
        "        initial=None;inherited=[];completed=0\n        if scale==1.:\n            initial,inherited,checksums=inherit_branch(plan,out,rank,device);completed=plan['completed_iterations']\n        def checkpoint(iteration,history):")
    source=replace(source,"dict(relative_L2=rel,bitwise=np.array_equal(frame.reshape(-1),reference),threshold=1e-5,reference_job=1669255)",
        "dict(relative_L2=rel,bitwise=np.array_equal(frame.reshape(-1),reference),threshold=1e-5,reference_job=1669255,within_original_threshold=bool(rel<=1e-5),continuation_authorized=True,policy=plan['historical_comparison_policy'])")
    source=replace(source,"                if not np.isfinite(rel) or rel>1e-5:raise ValueError('Alpha100 differs from delivered original')",
        "                if not np.isfinite(rel):raise ValueError('Nonfinite historical difference')")
    source=replace(source,"cfg['phase_limit_hours']*3600)\n        sync();elapsed=", "cfg['phase_limit_hours']*3600,initial_image=initial,initial_history=inherited,completed_iterations=completed)\n        sync();elapsed=")
    source=replace(source,'seconds_per_iteration=elapsed/10000,','seconds_per_iteration=elapsed/(10000-completed),new_iterations=10000-completed,inherited_iterations=completed,')
    source=replace(source,'        del blocks,items,events,im,h,sc,cache,pool;', '        del blocks,items,events,im,h,sc,cache,pool,initial,inherited;')
    return source

def main():
    freeze=REPORT/'freeze_resume_r5.json';target=DATA/'payload_resume_r5'
    if freeze.exists():
        f=read(freeze)
        for n,s in f['files'].items():assert sha(target/n)==s,n
        print('EXISTING RESUME FREEZE',f['key']);return
    planpath=REPORT/'resume_plan_r5.json'
    if planpath.exists():plan=read(planpath)
    else:plan=audit();write(planpath,plan)
    source=DATA/read(FREEZE)['payload_dir']
    for n,s in read(FREEZE)['files'].items():assert sha(source/n)==s,n
    target.mkdir(exist_ok=True)
    for n in read(FREEZE)['files']:
        p=target/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/n,p)
    shutil.copyfile(planpath,target/'resume_plan.json')
    for name in ('resume_support.py','test_resume.py'):shutil.copyfile(CODE/name,target/name)
    (target/'joint_mlem_resume.py').write_text(helper((source/'jscc_5e10_joint_mlem.py').read_text()),encoding='utf8',newline='\n')
    (target/'run_resume.py').write_text(runtime((source/'run_study.py').read_text()),encoding='utf8',newline='\n')
    for p in target.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    tests=subprocess.run([sys.executable,'-B','-X','utf8',str(target/'test_resume.py')],capture_output=True,text=True,encoding='utf8')
    (REPORT/'local_resume_tests_r5.txt').write_text(tests.stdout+tests.stderr,encoding='utf8')
    if tests.returncode or 'WIDTH_RESUME_PASS' not in tests.stdout:raise RuntimeError(tests.stdout+tests.stderr)
    files={p.relative_to(target).as_posix():sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    assert all(files[n]==s for n,s in read(FREEZE)['files'].items())
    write(freeze,dict(key=key,files=files,payload_dir=target.name,parent_scientific_freeze=read(FREEZE)['key'],
                     inherited_original_files_unchanged=True,resume_plan_sha256=sha(planpath),local_resume_tests_passed=True,
                     local_tests_sha256=sha(REPORT/'local_resume_tests_r5.txt')))
    print('FROZEN RESUME',key,len(files),'files; original science files unchanged; 100% resumes at 9950')

if __name__=='__main__':main()
