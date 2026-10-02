"""Safely deploy an immutable ablation release and submit a gated serial sweep.

Uses the established DPAPI/SSH helper; contains no credentials. A remote job
registry blocks duplicate submissions and records each sbatch immediately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"experiments/FOV120"))
from reconstruction_ssh import connect

REMOTE_ROOT="/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor"
BASE=REMOTE_ROOT+"/experiments/ELLIPSE500x300_H120"
PYTHON="/data/home/scxi717/.conda/envs/torch/bin/python"
STUDY="NEMA_5e9_SPIKE_ABLATION_V1"
REPORT=HERE/"reports/NEMA_Body_H60/spike_ablation"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(client,command,timeout=180,capture_stderr=False):
    _,stdout,stderr=client.exec_command(command,timeout=timeout)
    out=stdout.read().decode(errors="replace"); err=stderr.read().decode(errors="replace")
    code=stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote operation failed ({code}): {err or out}")
    return (out+("\n"+err if capture_stderr else "")).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submit",action="store_true")
    p.add_argument("--replace-pending",action="store_true",help="Replace only this study's pending older release; preserve its job ledger")
    p.add_argument("--replace-failed",action="store_true",help="Replace only a terminated study with no passed gates; preserve its ledger")
    p.add_argument("--exclude",default="",help="Slurm nodes excluded after a confirmed node preflight failure")
    a=p.parse_args()
    if a.exclude and not re.fullmatch(r"[A-Za-z0-9,._\[\]-]+",a.exclude):
        raise ValueError("Invalid Slurm node exclusion list")
    names=("run_reconstruction.py","torch_active_operator.py","regularized_update.py",
           "verify_spike_ablation.py","reconstruct_spike_ablation.sh","resource_budget.py",
           "test_spike_ablation.py","test_spike_ablation_dist.py","validate_factors.py",
           "geometry.py","config.json")
    payload={name:(HERE/name).read_bytes() for name in names}
    payload.update({name:(ROOT/name).read_bytes() for name in
                    ("compton_sparse_ops.py","process_list_plane_sparse.py","compton_event_response.py","detector_csv.py")})
    # Unix script bytes are canonical and covered by the release manifest.
    payload["reconstruct_spike_ablation.sh"]=payload["reconstruct_spike_ablation.sh"].replace(b"\r\n",b"\n")
    release={"study":STUDY,"files":{name:{"bytes":len(data),"sha256":digest(data)} for name,data in payload.items()}}
    release_bytes=(json.dumps(release,indent=2)+"\n").encode()
    release_id="spike_ablation_v1_"+digest(release_bytes)[:16]
    remote_release=BASE+"/code_releases/"+release_id
    remote_study=BASE+"/generated/SpikeAblation/"+STUDY
    local_study=HERE/"generated/SpikeAblation"/STUDY
    report_release={**release,"release_id":release_id,"remote_release":remote_release,"manifest_sha256":digest(release_bytes)}
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/"release.json").write_text(json.dumps(report_release,indent=2)+"\n")
    with connect() as client:
        # Common response kernels must match the unchanged remote baseline code.
        checker='import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); expected=json.loads(sys.argv[2]); assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in expected.items()), "Baseline response kernels differ"'
        common={name:digest(payload[name]) for name in ("compton_sparse_ops.py","process_list_plane_sparse.py","compton_event_response.py","detector_csv.py")}
        run(client,shlex.quote(PYTHON)+" -c "+shlex.quote(checker)+" "+shlex.quote(REMOTE_ROOT)+" "+shlex.quote(json.dumps(common)))
        run(client,"mkdir -p -- "+" ".join(shlex.quote(path) for path in (remote_release,remote_study,BASE+"/logs")))
        with client.open_sftp() as sftp:
            files={**{remote_release+"/"+name:data for name,data in payload.items()},
                   remote_release+"/release.json":release_bytes,
                   **{remote_study+"/"+name:(local_study/name).read_bytes() for name in ("study.json","spatial_model.npz")}}
            for path,data in files.items():
                try:
                    with sftp.open(path,"rb") as stream:
                        present=stream.read()
                except FileNotFoundError:
                    with sftp.open(path,"wx") as stream:
                        stream.write(data)
                else:
                    if digest(present)!=digest(data):
                        raise ValueError("Refusing to overwrite differing remote release/study file: "+path)
            for path,data in files.items():
                with sftp.open(path,"rb") as stream:
                    if digest(stream.read())!=digest(data):
                        raise ValueError("Remote payload verification failed: "+path)
        # Small CPU tests are bounded and run no production reconstruction.
        prefix="export JSCC_PROJECT_ROOT="+shlex.quote(REMOTE_ROOT)+" OMP_NUM_THREADS=1; "
        checks=[]
        for command in (
            shlex.quote(PYTHON)+" "+shlex.quote(remote_release+"/test_spike_ablation.py"),
            shlex.quote(PYTHON)+" -m torch.distributed.run --standalone --nproc_per_node=2 "+shlex.quote(remote_release+"/test_spike_ablation_dist.py"),
            "bash -n "+shlex.quote(remote_release+"/reconstruct_spike_ablation.sh")):
            checks.append(run(client,prefix+command,capture_stderr=True))
        (REPORT/"remote_tests.txt").write_text("\n\n".join(checks).rstrip()+"\n")
        print("IMMUTABLE_RELEASE_VERIFIED",release_id)
        print(checks[-2])
        if not a.submit:
            return
        registry=remote_study+"/jobs.json"
        # Check before any mutation. Existing registry is an authoritative submission ledger.
        existing=run(client,"if test -f "+shlex.quote(registry)+"; then cat "+shlex.quote(registry)+"; fi")
        if existing:
            old=json.loads(existing)
            if (not (a.replace_pending or a.replace_failed) or
                (not a.replace_failed and old["release_sha256"]==digest(release_bytes) and
                 old["topology"].get("excluded_nodes","")==a.exclude)):
                (REPORT/"jobs.json").write_text(existing+"\n")
                print("EXISTING_ABLATION_JOBS",existing)
                return
            if old["study"]!=STUDY or len(old["jobs"])!=1:
                raise ValueError("Unexpected existing study ledger")
            job=old["jobs"][0]["job"]
            if not job.isdigit():
                raise ValueError("Unsafe recorded job ID")
            queue_rows=run(client,"squeue -u scxi717 -h -o '%i|%T|%j'")
            snapshot=next((line.split("|",1)[1] for line in queue_rows.splitlines()
                           if line.split("|",1)[0]==job),"")
            disposition="cancelled"
            if a.replace_pending and snapshot=="PENDING|NEMA_ABL_sweep":
                run(client,"scancel "+job)
            elif a.replace_failed and not snapshot:
                state=run(client,"sacct -X -n -P -j "+job+" --format=State,JobName%100").strip().strip("|")
                if state not in {name+"|NEMA_ABL_sweep" for name in
                                 ("FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY")}:
                    raise RuntimeError("Existing study is not a confirmed terminated failure: "+state)
                with client.open_sftp() as sftp:
                    try:
                        gates=sftp.listdir(remote_study+"/gates")
                    except FileNotFoundError:
                        gates=[]
                if any(name.endswith(".json") for name in gates):
                    raise RuntimeError("Passed gates exist; inspect release compatibility before repairing")
                disposition="failed"
            else:
                raise RuntimeError("Existing study cannot be safely replaced: "+snapshot)
            archived=registry+"."+disposition+"_"+job
            with client.open_sftp() as sftp:
                sftp.rename(registry,archived)
            (REPORT/f"jobs.{disposition}_{job}.json").write_text(existing+"\n")
        queue=run(client,"squeue -u scxi717 -h -o '%i|%j|%T'")
        if any("NEMA_ABL_" in line for line in queue.splitlines()):
            raise RuntimeError("Existing ablation jobs without ledger; inspect before submitting")
        record={"study":STUDY,"release":remote_release,"release_sha256":digest(release_bytes),
                "topology":{"nodes":8,"gpus_per_node":1,"cpus_per_node":6,
                            "excluded_nodes":a.exclude,
                            "memory":"scheduler assigns per GPU; actual grant recorded; at least 55GiB required"},
                "jobs":[],"gate":"Each variant: 10 -> 200 -> 10000, next variant only after formal gate; failure stops the sweep"}
        # One account submission slot. 270h is only a ceiling equal to all phase
        # timeouts (5*(2+4+48)), not a runtime estimate or a requirement to wait.
        args=["sbatch","--parsable","--job-name=NEMA_ABL_sweep","--time=270:00:00",
              "--output="+BASE+"/logs/ablation_sweep.%j.out",
              "--error="+BASE+"/logs/ablation_sweep.%j.err",
              "--export=ALL,ABLATION_RELEASE="+remote_release+",ABLATION_PHASE=pipeline",
              remote_release+"/reconstruct_spike_ablation.sh"]
        if a.exclude:
            args.insert(-1,"--exclude="+a.exclude)
        job=run(client," ".join(shlex.quote(arg) for arg in args)).split(";")[0]
        if not job.isdigit():
            raise ValueError("Unexpected sbatch job ID")
        record["jobs"].append({"phase":"pipeline","job":job,
            "variants":[v["id"] for v in json.loads((local_study/"study.json").read_text())["variants"]],
            "phase_timeout_hours":{"pilot":2,"short":4,"formal":48}})
        text=json.dumps(record,indent=2)+"\n"
        with client.open_sftp() as sftp:
            with sftp.open(registry,"w") as stream:
                stream.write(text.encode())
        (REPORT/"jobs.json").write_text(text)
        print("GATED_ABLATION_SUBMITTED",json.dumps(record,indent=2))


if __name__=="__main__":
    main()
