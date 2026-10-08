"""One bounded CPU read-only sensitivity run, with immutable inputs and strict retrieval."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ehe_5e9_workflow import connection,command,q
from ehe_common import HERE,DATA,REPORT,GPU_BASE,GPU_PYTHON,digest,read,write


def main():
    registration=REPORT/'physical_cuboid_diagnostic_registration.json'
    assert not registration.exists(),'Existing registration retained; inspect before any action'
    accepted=read(REPORT/'physical_hold_fetch_acceptance.json')
    assert accepted['science_gate_passed'] is False and accepted['job']==1677211
    view_file=REPORT/'physical_1677211/view_global_audit.csv'
    rows=list(csv.DictReader(view_file.open(newline='',encoding='utf-8')))
    names=['A218','A440','C440to218']
    predicted={n:[float(r['predicted']) for r in rows if r['response']==n and r['scope']=='view_total'] for n in names}
    assert all(len(p)==20 for p in predicted.values())
    observed={n:float(next(r['observed'] for r in rows if r['response']==n and r['scope']=='global')) for n in names}
    binding=dict(truth_path=GPU_BASE+'/releases/74e129c4460163c5/truth_3mm.npz',
        truth_sha256=accepted['remote_before']['truth_3mm.npz']['sha256'],
        workers_path=GPU_BASE+'/counts/worker_counts.npz',
        workers_sha256=accepted['remote_before']['worker_counts.npz']['sha256'],
        response_root=accepted['response_root'],
        factor_manifest_sha256={n:accepted['remote_before'][n+'/factor_manifest.json']['sha256'] for n in names},
        original_view_predicted=predicted,original_global_observed=observed,
        original_physical_gate_sha256=accepted['remote_before']['physical_gate.json']['sha256'],
        original_audit_csv_sha256=accepted['data_csv_sha256'],view_global_reference_sha256=digest(view_file),
        authorization_scope='Read-only interpolation sensitivity; existing original HOLD remains binding')
    code=HERE/'diagnose_ehe_cuboid_fold.py'
    encoded=(json.dumps(binding,indent=2,allow_nan=False)+'\n').encode()
    key=hashlib.sha256(code.read_bytes()+encoded).hexdigest()[:16]
    folder=DATA/'read_only_source_fold_releases'/key
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'binding.json').write_bytes(encoded)
    shutil.copy2(code,folder/code.name)
    files={p.name:digest(p) for p in folder.iterdir()}
    write(folder/'diagnostic_manifest.json',dict(key=key,files=files,science_status='HOLD',kind='CPU read-only',hard_limit_seconds=540))
    remote=GPU_BASE+'/read_only_source_fold_'+key
    reg=dict(kind='CPU read-only diagnostic, no Slurm submission',science_job=1677211,key=key,
        local_launcher_pid=os.getpid(),status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        output=remote+'/output',immutable_execution_files=files,launcher_sha256=digest(__file__),
        no_production_input_modified=True,no_stage_resubmitted=True,hard_limit_seconds=540)
    write(registration,reg)
    try:
        with connection('gpu') as c:
            command(c,'mkdir '+q(remote))
            with c.open_sftp() as s:
                for p in folder.iterdir():s.put(str(p),remote+'/'+p.name)
            verify='import hashlib,json,pathlib;d=pathlib.Path('+repr(remote)+');m=json.loads((d/"diagnostic_manifest.json").read_text());assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in m["files"].items());print("DIAGNOSTIC_CODE_BINDING_SHA_PASS")'
            print(command(c,q(GPU_PYTHON)+' -c '+q(verify)),flush=True)
            inner='set -o pipefail\ntimeout --signal=TERM --kill-after=10s 540s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+q(GPU_PYTHON)+' -u '+q(remote+'/'+code.name)+' --binding '+q(remote+'/binding.json')+' --output '+q(reg['output'])+' 2>&1 | tee '+q(remote+'/execution.log')
            shell='bash -c '+q(inner)
            reg['exact_command']=shell;write(registration,reg)
            _,stdout,stderr=c.exec_command(shell,timeout=600)
            channel=stdout.channel;began=time.monotonic()
            with (folder/'local_execution.log').open('wb') as f:
                while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                    for ready,receive in ((channel.recv_ready,channel.recv),(channel.recv_stderr_ready,channel.recv_stderr)):
                        if ready():
                            b=receive(65536);f.write(b);f.flush();print(b.decode(errors='replace'),end='',flush=True)
                    if time.monotonic()-began>600:raise TimeoutError('Bounded remote read-only command did not exit')
                    time.sleep(.1)
            exit_code=channel.recv_exit_status();reg['exit_code']=exit_code
            assert exit_code==0,'Read-only diagnostic retained its evidence after failure'
            paths={'cuboid_fold.json':reg['output']+'/cuboid_fold.json','execution.log':remote+'/execution.log',
                'binding.json':remote+'/binding.json',code.name:remote+'/'+code.name,'diagnostic_manifest.json':remote+'/diagnostic_manifest.json'}
            check='import hashlib,json,pathlib;print(json.dumps({n:{"sha256":hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(),"bytes":pathlib.Path(p).stat().st_size} for n,p in '+repr(paths)+'.items()}))'
            before=json.loads(command(c,q(GPU_PYTHON)+' -c '+q(check)))
            evidence=REPORT/'physical_cuboid_read_only'
            evidence.mkdir(exist_ok=False)
            with c.open_sftp() as s:
                for n,p in paths.items():s.get(p,str(evidence/n))
            after=json.loads(command(c,q(GPU_PYTHON)+' -c '+q(check)))
            assert before==after
            for n,h in before.items():
                assert digest(evidence/n)==h['sha256'] and (evidence/n).stat().st_size==h['bytes']
            proof=read(evidence/'cuboid_fold.json')
            assert proof['passed'] and proof['scientific_status']=='HOLD' and not proof['physical_gate_pass_claimed']
            assert proof['code_sha256']==files[code.name] and proof['binding_sha256']==files['binding.json']
            assert digest(code)==proof['code_sha256']
            shutil.copy2(__file__,evidence/'launch_and_fetch_read_only.py')
            write(REPORT/'physical_cuboid_read_only_acceptance.json',dict(passed=True,
                scope='Only identity, CPU interpolation sensitivity execution, and strict retrieval; original physical HOLD unchanged',
                key=key,science_job=1677211,scientific_status='HOLD',no_physical_or_reconstruction_submission=True,
                remote_before=before,remote_after_equal=True,launcher_sha256=digest(__file__),
                factor_manifest_sha256=binding['factor_manifest_sha256'],exit_code=exit_code,
                elapsed_seconds=proof['elapsed_seconds'],cpu_readonly_rss_peak_bytes=proof['cpu_readonly_rss_peak_bytes'],
                imaging_allocation_certificate=False,original_physical_gate_sha256=binding['original_physical_gate_sha256'],
                original_audit_csv_sha256=binding['original_audit_csv_sha256']))
            reg['status']='complete';reg['exit_code']=0
    except Exception as ex:
        reg['status']='failed_preserved';reg['error_type']=type(ex).__name__;write(registration,reg);raise
    finally:
        reg['ended_utc']=datetime.now(timezone.utc).isoformat();write(registration,reg)


if __name__=='__main__':main()
