"""Hash-checked handoff from maty to 65114, then gated paired reconstruction.

Never stores credentials. Collection and analysis are separate bounded stages;
HOLD is a hard stop before preparing or submitting an imaging release.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/"experiments/FOV120")]
from first_scatter_workflow import DATA,REPORT,REMOTE,HOST,digest,write,remote

SERVER="65114_lipeize"
SERVER_ROOT="/home/lipeize/JSCC_FOV120_20260924"
SERVER_BASE=SERVER_ROOT+"/experiments/ELLIPSE500x300_H120"
SERVER_STUDY=SERVER_BASE+"/generated/compton_first_scatter_v2"

def ssh(host,command):
    p=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",host,command],
                     capture_output=True,text=True,encoding="utf-8",errors="replace")
    if p.returncode:raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout.strip()

def transfer(host,source,target):
    subprocess.run(["scp","-o","BatchMode=yes",str(source),host+":"+target],check=True)
    if ssh(host,"sha256sum -- "+shlex.quote(target)).split()[0]!=digest(source):
        raise ValueError("Transferred payload hash differs")

def queue_collection():
    record=json.loads((REPORT/"transport_jobs.json").read_text())
    if (REPORT/"collection_job.json").exists():raise ValueError("Collection already queued")
    source=HERE/"first_scatter_workflow.py";sha=digest(source)
    script=REMOTE+"/first_scatter_collect_"+sha[:16]+".py"
    transfer(HOST,source,script)
    command="/apps/soft/anaconda3/bin/python3 "+shlex.quote(script)+" collect --base "+shlex.quote(REMOTE)
    job=remote(f"sbatch --parsable -p cnmix --cpus-per-task=1 --time=00:30:00 "
               f"--dependency=afterok:{record['array_job']} --job-name=FIRST_SCATTER_COLLECT "
               f"--output={REMOTE}/logs/collect.%j.out --error={REMOTE}/logs/collect.%j.err "
               "--wrap="+shlex.quote(command))
    if not job.isdigit():raise ValueError("Unexpected collection job ID")
    write(REPORT/"collection_job.json",dict(job=int(job),depends_on=record["array_job"],code_sha256=sha))
    print("COLLECTION_JOB",job)

def stage_analysis():
    names=("analyze_first_scatter.py","first_scatter_offline.py","geometry.py","config.json",
           "test_first_scatter.py","torch_active_operator.py","validate_factors.py")
    paths={name:HERE/name for name in names}
    paths["geometry.npz"]=HERE/"generated/Geometry/geometry.npz"
    paths.update({name:ROOT/name for name in ("compton_event_response.py","detector_csv.py")})
    hashes={name:digest(path) for name,path in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=SERVER_STUDY+"/releases/"+key
    archive=DATA/"analysis_code.tar.gz"
    with tarfile.open(archive,"w:gz") as f:
        for name,path in paths.items():f.add(path,arcname=name)
    ssh(SERVER,"mkdir -p -- "+shlex.quote(release))
    transfer(SERVER,archive,release+"/code.tar.gz")
    ssh(SERVER,"tar --no-same-owner -xzf "+shlex.quote(release+"/code.tar.gz")+" -C "+shlex.quote(release)+
        "; cd "+shlex.quote(release)+" && "+SERVER_ROOT+"/.venv/bin/python -m unittest test_first_scatter -v")
    write(REPORT/"analysis_deployment.json",dict(release=release,code_sha256=hashes,archive_sha256=digest(archive)))
    print("ANALYSIS_CODE_TESTED",release)

def launch_analysis():
    if (REPORT/"analysis_job.json").exists():raise ValueError("Analysis already launched")
    ready=json.loads(remote("cat "+shlex.quote(REMOTE+"/collection_ready.json")))
    if ready["status"]!="READY":raise ValueError("Transport collection is not ready")
    archive=DATA/"analysis_inputs.tar.gz"
    if not archive.exists():
        subprocess.run(["scp","-o","BatchMode=yes",HOST+":"+REMOTE+"/analysis_inputs.tar.gz",str(archive)],check=True)
    if digest(archive)!=ready["archive_sha256"]:raise ValueError("Collection archive hash differs")
    ssh(SERVER,"mkdir -p -- "+shlex.quote(SERVER_STUDY))
    transfer(SERVER,archive,SERVER_STUDY+"/analysis_inputs.tar.gz")
    ssh(SERVER,"test ! -d "+shlex.quote(SERVER_STUDY+"/analysis_inputs")+" && tar --no-same-owner -xzf "+
        shlex.quote(SERVER_STUDY+"/analysis_inputs.tar.gz")+" -C "+shlex.quote(SERVER_STUDY))
    deployment=json.loads((REPORT/"analysis_deployment.json").read_text());release=deployment["release"]
    # A free device is checked before launching. Nothing touches another user's process.
    memory=ssh(SERVER,"nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0")
    free,util=map(int,memory.split(","))
    if free<30000 or util>5:raise ValueError("GPU 0 is busy; leave analysis pending")
    script=DATA/"run_analysis.sh"
    body=f'''#!/usr/bin/env bash
set -euo pipefail
exec 9>{SERVER_STUDY}/analysis.lock
flock -n 9
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd {release}
python={SERVER_ROOT}/.venv/bin/python
timeout --signal=TERM --kill-after=30s 4h "$python" analyze_first_scatter.py \\
 --inputs {SERVER_STUDY}/analysis_inputs --factors {SERVER_BASE}/generated/FactorsCalibrated \\
 --geometry {release}/geometry.npz --output {SERVER_STUDY}/analysis --device cuda:0
timeout --signal=TERM --kill-after=30s 2h "$python" first_scatter_offline.py \\
 --inputs {SERVER_STUDY}/analysis_inputs --factors {SERVER_BASE}/generated/FactorsCalibrated \\
 --geometry {release}/geometry.npz --output {SERVER_STUDY}/offline
echo FIRST_SCATTER_ANALYSIS_AND_OFFLINE_FINISHED > {SERVER_STUDY}/analysis_finished.txt
'''
    script.write_text(body,encoding="ascii",newline="\n")
    transfer(SERVER,script,release+"/run_analysis.sh")
    pid=ssh(SERVER,"nohup bash "+shlex.quote(release+"/run_analysis.sh")+" > "+shlex.quote(SERVER_STUDY+"/analysis.log")+
            " 2>&1 < /dev/null & echo $!")
    if not pid.isdigit():raise ValueError("Invalid analysis PID")
    write(REPORT/"analysis_job.json",dict(pid=int(pid),server_study=SERVER_STUDY,release=release,
          inputs_archive_sha256=digest(archive),launcher_sha256=digest(script)))
    print("ANALYSIS_PID",pid)

def analysis_status():
    text=ssh(SERVER,"tail -n 3 "+shlex.quote(SERVER_STUDY+"/analysis.log")+
        "; if test -f "+shlex.quote(SERVER_STUDY+"/analysis/validation_gate.json")+
        "; then "+SERVER_ROOT+"/.venv/bin/python -c "+shlex.quote(
        "import json; x=json.load(open('"+SERVER_STUDY+"/analysis/validation_gate.json')); "
        "print('GATE',x['status']); print([(r['group'],r['test'],r.get('bin'),r.get('relative_error',r.get('value'))) "
        "for r in x['gates'] if not r['passed']])")+"; fi")
    print(text)

def repair_analysis():
    """Restart a stopped analyzer on exactly the existing immutable transport."""
    old=json.loads((REPORT/"analysis_job.json").read_text())
    deployment=json.loads((REPORT/"analysis_deployment.json").read_text())
    if deployment["release"]==old["release"]:raise ValueError("A new frozen repair release is required")
    active=ssh(SERVER,"pgrep -af "+shlex.quote("[a]nalyze_first_scatter.py.*"+SERVER_STUDY)+
        " || pgrep -af "+shlex.quote("[f]irst_scatter_offline.py.*"+SERVER_STUDY)+" || true")
    if active:raise ValueError("The previous analysis has not exited; never duplicate it")
    if ssh(SERVER,"sha256sum "+shlex.quote(SERVER_STUDY+"/analysis_inputs.tar.gz")).split()[0]!=old["inputs_archive_sha256"]:
        raise ValueError("Repair must use the same frozen transport archive")
    memory=ssh(SERVER,"nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0")
    free,util=map(int,memory.split(","))
    if free<30000 or util>5:raise ValueError("GPU 0 is busy; leave repair pending")
    suffix="_failed_"+str(old["pid"])
    ssh(SERVER,"set -e; test ! -f "+shlex.quote(SERVER_STUDY+"/analysis_finished.txt")+
        " && test ! -e "+shlex.quote(SERVER_STUDY+"/analysis"+suffix)+
        " && mv -- "+shlex.quote(SERVER_STUDY+"/analysis")+" "+shlex.quote(SERVER_STUDY+"/analysis"+suffix)+
        " && mv -- "+shlex.quote(SERVER_STUDY+"/analysis.log")+" "+shlex.quote(SERVER_STUDY+"/analysis"+suffix+".log")+
        "; if test -d "+shlex.quote(SERVER_STUDY+"/offline")+"; then test ! -e "+
        shlex.quote(SERVER_STUDY+"/offline"+suffix)+" && mv -- "+shlex.quote(SERVER_STUDY+"/offline")+" "+
        shlex.quote(SERVER_STUDY+"/offline"+suffix)+"; fi")
    script=DATA/"run_analysis.sh"
    body=script.read_text().replace(old["release"],deployment["release"])
    script.write_text(body,encoding="ascii",newline="\n")
    transfer(SERVER,script,deployment["release"]+"/run_analysis.sh")
    pid=ssh(SERVER,"nohup bash "+shlex.quote(deployment["release"]+"/run_analysis.sh")+
            " > "+shlex.quote(SERVER_STUDY+"/analysis.log")+" 2>&1 < /dev/null & echo $!")
    if not pid.isdigit():raise ValueError("Invalid repair PID; inspect before retry")
    history=old.pop("previous_runs",[]);history.append(old)
    write(REPORT/"analysis_job.json",dict(pid=int(pid),server_study=SERVER_STUDY,release=deployment["release"],
        inputs_archive_sha256=old["inputs_archive_sha256"],launcher_sha256=digest(script),previous_runs=history,
        transport_replayed=False,failed_evidence_suffix=suffix))
    print("REPAIRED_ANALYSIS_PID",pid)

def point_diagnostics():
    """Observe the current kernel at held-out true points without changing cuts."""
    ssh(SERVER,"test -f "+shlex.quote(SERVER_STUDY+"/analysis/validation_gate.json"))
    if (REPORT/"point_deployment.json").exists():raise ValueError("Point diagnostic already registered")
    paths={"diagnose_first_scatter_points.py":HERE/"diagnose_first_scatter_points.py",
        "compton_event_response.py":ROOT/"compton_event_response.py","detector_csv.py":ROOT/"detector_csv.py"}
    hashes={name:digest(p) for name,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=SERVER_STUDY+"/point_releases/"+key
    ssh(SERVER,"mkdir -p -- "+shlex.quote(release))
    for name,p in paths.items():transfer(SERVER,p,release+"/"+name)
    write(REPORT/"point_deployment.json",dict(release=release,code_sha256=hashes,diagnostic_only=True))
    print(ssh(SERVER,"timeout --signal=TERM --kill-after=20s 10m "+SERVER_ROOT+"/.venv/bin/python "+
        shlex.quote(release+"/diagnose_first_scatter_points.py")+" --inputs "+shlex.quote(SERVER_STUDY+"/analysis_inputs")+
        " --analysis "+shlex.quote(SERVER_STUDY+"/analysis")+" --factors "+shlex.quote(SERVER_BASE+"/generated/FactorsCalibrated")+
        " --output "+shlex.quote(SERVER_STUDY+"/point_diagnostics")))

def fetch_analysis():
    if "FINISHED" not in ssh(SERVER,"cat "+shlex.quote(SERVER_STUDY+"/analysis_finished.txt")):
        raise ValueError("Analysis/offline have not both finished")
    target=DATA/"analysis_evidence.tar.gz"
    extra=""
    if (REPORT/"point_deployment.json").exists():
        ssh(SERVER,"test -f "+shlex.quote(SERVER_STUDY+"/point_diagnostics/point_consistency.json"))
        extra=" point_diagnostics"
    ssh(SERVER,"tar -czf "+shlex.quote(SERVER_STUDY+"/analysis_evidence.tar.gz")+" -C "+shlex.quote(SERVER_STUDY)+
        " analysis offline"+extra)
    sha=ssh(SERVER,"sha256sum "+shlex.quote(SERVER_STUDY+"/analysis_evidence.tar.gz")).split()[0]
    subprocess.run(["scp","-o","BatchMode=yes",SERVER+":"+SERVER_STUDY+"/analysis_evidence.tar.gz",str(target)],check=True)
    if digest(target)!=sha:raise ValueError("Analysis evidence transfer hash differs")
    with tarfile.open(target) as f:
        f.extractall(DATA,filter="data")
    gate=json.loads((DATA/"analysis/validation_gate.json").read_text())
    write(REPORT/"validation_gate.json",gate)
    write(REPORT/"offline_summary.json",json.loads((DATA/"offline/offline_summary.json").read_text()))
    if extra:write(REPORT/"point_consistency.json",json.loads((DATA/"point_diagnostics/point_consistency.json").read_text()))
    write(REPORT/"analysis_collection.json",dict(archive_sha256=sha,gate_status=gate["status"],
          validation_gate_sha256=digest(DATA/"analysis/validation_gate.json")))
    print("FETCHED_VALIDATION",gate["status"])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("queue-collection","stage-analysis","launch-analysis","repair-analysis","status-analysis","point-diagnostics","fetch-analysis"))
    a=p.parse_args()
    {"queue-collection":queue_collection,"stage-analysis":stage_analysis,"launch-analysis":launch_analysis,
     "repair-analysis":repair_analysis,"status-analysis":analysis_status,
     "point-diagnostics":point_diagnostics,"fetch-analysis":fetch_analysis}[a.mode]()
if __name__=="__main__":main()
