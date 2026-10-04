"""Freeze/deploy/submit only the independently accepted 1e9 paired study."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
import re
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/"experiments/FOV120")]
from first_scatter_workflow import DATA,REPORT,digest,write
from reconstruction_ssh import connect

REMOTE_ROOT="/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor"
REMOTE=REMOTE_ROOT+"/experiments/ELLIPSE500x300_H120"
STUDY=REMOTE+"/generated/compton_first_scatter_v2"
CHANNELS=["440_ComptonOnly","440_SinglePlusCompton"]

def command(client,text):
    _,out,err=client.exec_command(text,timeout=120)
    data=out.read().decode(errors="replace");error=err.read().decode(errors="replace")
    if out.channel.recv_exit_status():raise RuntimeError(error or data)
    return data.strip()

def freeze():
    gatepath=DATA/"analysis/validation_gate.json";gate=json.loads(gatepath.read_text())
    if gate["status"]!="PASSED":raise ValueError("Independent sensitivity validation is on HOLD; imaging remains stopped")
    inputs=DATA/"analysis_inputs/NEMA";collection=json.loads((inputs/"collection.json").read_text())
    baseline=json.loads((HERE/"generated/RemoteResults/NEMA_Body_H60_1e9_1643142/run_manifest.json").read_text())
    if gate["geometry_sha256"]!=baseline["geometry_sha256"]:raise ValueError("Paired geometry differs from baseline")
    if digest(ROOT/"compton_event_response.py")!=gate["kernel_sha256"]:raise ValueError("Response kernel changed")
    if sum(collection["primary_counts"])!=10**9 or collection["seeds"]!=list(range(30093001,30093201)):
        raise ValueError("NEMA paired replay photon/seed closure differs")
    output=DATA/"recon_inputs";output.mkdir(exist_ok=False)
    configs=DATA/"configs";configs.mkdir(exist_ok=False)
    shutil.copy2(gatepath,configs/"validation_gate.json")
    for group in ("legacy","ideal"):
        target=output/group
        lp=target/"List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_1e9"
        lp.mkdir(parents=True)
        for view in range(1,21):shutil.copy2(inputs/f"{group}_v{view:02d}.csv",lp/f"{view}.csv")
        for e in (218,440):
            cp=target/f"CntStat/{e}keV_RotateNum20_Geant4JSCC";cp.mkdir(parents=True)
            rows=[np.loadtxt(inputs/f"counts_{e}_v{v:02d}.csv",delimiter=",",dtype=np.int64,ndmin=1) for v in range(1,21)]
            np.savetxt(cp/"CntStat_NEMA_Body_H60_1e9.csv",np.stack(rows),fmt="%d",delimiter=",")
        c=target/"collections";c.mkdir()
        write(c/"NEMA_Body_H60_1e9.json",dict(dataset="NEMA_Body_H60",level="1e9",views=list(range(1,21)),
            seeds=collection["seeds"],worker_indices=list(range(200)),primary_counts=collection["primary_counts"],
            event_policy=group,paired_collection_sha256=digest(inputs/"collection.json")))
        hashes={p.relative_to(target).as_posix():digest(p) for p in target.rglob("*") if p.is_file()}
        scan=gate["scans"][f"NEMA_{group}"]
        cfg=dict(study="compton_first_scatter_v2",group="legacy" if group=="legacy" else "ideal_first_scatter_v2",
            iterations=2000,save_step=50,channels=CHANNELS,max_min_standardized_arm=3.,quality_domain="full_circle_132040",
            validation_gate_passed=True,validation_gate_sha256=digest(gatepath),kernel_sha256=gate["kernel_sha256"],
            geometry_sha256=gate["geometry_sha256"],baseline_input_sha256=hashes,
            baseline_factor_manifest_sha256=gate["factor_manifest_sha256"],baseline_sensi_d_sha256=baseline["sensi_d_sha256"],
            sensi_d_sha256=digest(DATA/f"analysis/{group}/Sensi_d"),scan_manifest_sha256=digest(gatepath),
            kept_compton_events=scan["kept"],removed_compton_events=scan["uncut"]-scan["kept"],
            baseline_accepted_compton_events=97299,baseline_result="NEMA_Body_H60_1e9_1643142",
            baseline_per_view=[r["uncut"] for r in gate["scans"]["NEMA_legacy"]["records"]],
            kept_per_view=[r["kept"] for r in scan["records"]])
        if len(hashes)!=23:raise ValueError("Exactly 23 reconstruction inputs required")
        write(configs/f"{group}.json",cfg)
    for energy in (218,440):
        relative=f"CntStat/{energy}keV_RotateNum20_Geant4JSCC/CntStat_NEMA_Body_H60_1e9.csv"
        if digest(output/"legacy"/relative)!=digest(output/"ideal"/relative):raise ValueError("Paired single-photon counts differ")
    print("FIRST_SCATTER_IMAGING_INPUTS_FROZEN")

def deploy():
    configs=DATA/"configs";gate=json.loads((configs/"validation_gate.json").read_text())
    if gate["status"]!="PASSED":raise ValueError("HOLD blocks imaging deployment")
    names=("run_reconstruction.py","torch_active_operator.py","validate_factors.py","geometry.py","config.json",
           "verify_first_scatter.py","reconstruct_first_scatter.sh")
    paths={n:HERE/n for n in names}
    paths.update({n:ROOT/n for n in ("compton_event_response.py","compton_sparse_ops.py","process_list_plane_sparse.py","detector_csv.py")})
    paths.update({n:configs/n for n in ("legacy.json","ideal.json","validation_gate.json")})
    paths["geometry.npz"]=HERE/"generated/Geometry/geometry.npz"
    hashes={n:digest(p) for n,p in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+"/code_releases/compton_first_scatter_v2_"+key
    archive=DATA/"imaging_payload.tar.gz"
    with tarfile.open(archive,"w:gz") as f:
        for group in ("legacy","ideal"):
            f.add(DATA/f"recon_inputs/{group}",arcname="recon_inputs/"+group)
            for name in ("Sensi_d","Sensi_d_provenance.json"):
                f.add(DATA/f"analysis/{group}/{name}",arcname=f"analysis/{group}/{name}")
    with connect() as client:
        command(client,"mkdir -p -- "+shlex.quote(release)+" "+shlex.quote(STUDY))
        with client.open_sftp() as sftp:
            for name,p in paths.items():
                sftp.put(str(p),release+"/"+name)
                if command(client,"sha256sum -- "+shlex.quote(release+"/"+name)).split()[0]!=hashes[name]:
                    raise ValueError("Frozen code transfer differs")
            sftp.put(str(archive),STUDY+"/imaging_payload.tar.gz")
        if command(client,"sha256sum "+shlex.quote(STUDY+"/imaging_payload.tar.gz")).split()[0]!=digest(archive):
            raise ValueError("Imaging payload transfer differs")
        command(client,"test ! -e "+shlex.quote(STUDY+"/recon_inputs")+" && tar --no-same-owner -xzf "+
                shlex.quote(STUDY+"/imaging_payload.tar.gz")+" -C "+shlex.quote(STUDY))
        command(client,"bash -n "+shlex.quote(release+"/reconstruct_first_scatter.sh"))
        for group in ("legacy","ideal"):
            cfg=json.loads((configs/f"{group}.json").read_text())
            for rel,sha in cfg["baseline_input_sha256"].items():
                if command(client,"sha256sum "+shlex.quote(STUDY+"/recon_inputs/"+group+"/"+rel)).split()[0]!=sha:
                    raise ValueError("Remote per-file input differs")
            for name,sha in cfg["baseline_factor_manifest_sha256"].items():
                if command(client,"sha256sum "+shlex.quote(REMOTE+"/generated/FactorsCalibrated/"+name+"/factor_manifest.json")).split()[0]!=sha:
                    raise ValueError("Remote existing Factors differ from independent validation")
            if command(client,"sha256sum "+shlex.quote(STUDY+"/analysis/"+group+"/Sensi_d")).split()[0]!=cfg["sensi_d_sha256"]:
                raise ValueError("Remote matching Sensi differs")
    write(REPORT/"imaging_deployment.json",dict(release=release,code_sha256=hashes,payload_sha256=digest(archive),gated=True))
    print("FIRST_SCATTER_IMAGING_DEPLOYED",release)

def submit(nodes,exclude):
    if nodes not in (4,8):raise ValueError("Only validated candidate topologies are allowed")
    if (REPORT/"imaging_job.json").exists():raise ValueError("A registered job exists; inspect its state before replacement")
    deployment=json.loads((REPORT/"imaging_deployment.json").read_text());release=deployment["release"]
    with connect() as client:
        queue=command(client,"squeue -u scxi717 -h -o '%i %j %T'")
        if "NEMA_first_scatter_v2" in queue:raise ValueError("Duplicate paired reconstruction refused")
        if len(queue.splitlines())>=50:
            write(REPORT/"imaging_submission_wait.json",dict(status="account_50_job_limit",other_project_jobs_untouched=True))
            print("ACCOUNT_LIMIT_WAIT");return
        cmd=(f"sbatch --parsable -N {nodes} --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=6 "
            "-p gpu_4090,gpu_5090 --qos=gpugpu --time=12:00:00 --chdir=/tmp "
            "--job-name=NEMA_first_scatter_v2 --exclude="+shlex.quote(exclude)+" --export="+
            shlex.quote("ALL,FIRST_SCATTER_RELEASE="+release)+
            " --output="+shlex.quote(REMOTE+"/logs/first_scatter.%j.out")+
            " --error="+shlex.quote(REMOTE+"/logs/first_scatter.%j.err")+" "+shlex.quote(release+"/reconstruct_first_scatter.sh"))
        job=command(client,cmd).split(";")[0]
        if not job.isdigit():raise ValueError("Ambiguous submission result; inspect before retry")
    write(REPORT/"imaging_job.json",dict(job=int(job),nodes=nodes,gpus_per_node=1,release=release,
        submitted_utc=datetime.now(timezone.utc).isoformat(),nccl_interface="bond0",
        phases=["legacy regression 50","legacy full-data pilot 10","ideal full-data pilot 10","legacy 2000","ideal 2000"]))
    print("PAIRED_IMAGING_JOB",job)

def status():
    registration=json.loads((REPORT/"imaging_job.json").read_text());job=str(registration["job"])
    with connect() as client:
        print(command(client,f"squeue -j {job} -h -o '%i %T %M %N'; sacct -X -j {job} -n -P --format=JobIDRaw,State,ExitCode"))
        print(command(client,"tail -n 6 "+shlex.quote(REMOTE+f"/logs/first_scatter.{job}.out")+
                      "; tail -n 3 "+shlex.quote(REMOTE+f"/logs/first_scatter.{job}.err")))

def fetch():
    registration=json.loads((REPORT/"imaging_job.json").read_text());job=str(registration["job"])
    target=DATA/"RemoteResults";target.mkdir(exist_ok=False)
    with connect() as client:
        accounting=command(client,f"sacct -j {job} -n -P --format=JobIDRaw,State,ExitCode,MaxRSS,ReqMem,AllocTRES")
        overall=[r.split("|") for r in accounting.splitlines() if r.split("|")[0]==job]
        if len(overall)!=1 or overall[0][1]!="COMPLETED" or overall[0][2]!="0:0":
            raise ValueError("Paired job did not finish successfully")
        resource=[];minimum_host=None
        for group in ("legacy","ideal"):
            result=STUDY+f"/formal_{group}_{job}"
            verify=json.loads(command(client,"cat "+shlex.quote(result+"/verification.json")))
            if not verify["passed"] or verify["mode"]!="formal":raise ValueError("Remote numerical verification missing")
            minimum_host=min([r["host_allocated_bytes"] for r in verify["resources"]]+([minimum_host] if minimum_host else []))
            name=f"formal_{group}_{job}.tar.gz"
            command(client,"tar -czf "+shlex.quote(STUDY+"/"+name)+" -C "+shlex.quote(result)+" .")
            sha=command(client,"sha256sum "+shlex.quote(STUDY+"/"+name)).split()[0]
            with client.open_sftp() as sftp:sftp.get(STUDY+"/"+name,str(target/name))
            if digest(target/name)!=sha:raise ValueError("Result archive transfer differs")
            folder=target/group;folder.mkdir()
            with tarfile.open(target/name) as tar:tar.extractall(folder,filter="data")
            resource.append(dict(group=group,archive_sha256=sha,verification_sha256=digest(folder/"verification.json")))
        rss=[]
        for line in accounting.splitlines():
            value=line.split("|")[3]
            match=re.fullmatch(r"([0-9.]+)([KMGT])",value)
            if match:rss.append(float(match[1])*1024**("KMGT".index(match[2])+1))
        if not rss:raise ValueError("Slurm actual MaxRSS is missing")
        if max(rss)/minimum_host>.8:raise ValueError("Slurm MaxRSS exceeds 80% of actual granted host memory")
        write(REPORT/"paired_collection.json",dict(job=int(job),groups=resource,accounting=accounting,
            slurm_max_rss_bytes=max(rss),minimum_actual_host_bytes=minimum_host,
            passed=True,result_root=str(target)))
    print("FIRST_SCATTER_PAIRED_RESULTS_FETCHED",target)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("freeze","deploy","submit","status","fetch"));p.add_argument("--nodes",type=int,default=4)
    p.add_argument("--exclude",default="wqd10nba06g6")
    a=p.parse_args()
    if a.mode=="freeze":freeze()
    elif a.mode=="deploy":deploy()
    elif a.mode=="status":status()
    elif a.mode=="fetch":fetch()
    else:submit(a.nodes,a.exclude)
if __name__=="__main__":main()
