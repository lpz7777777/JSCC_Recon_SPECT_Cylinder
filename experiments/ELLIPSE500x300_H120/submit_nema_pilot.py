"""Submit a bounded full-data NEMA pilot after transfer/primary closure checks."""
import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from resource_budget import estimate

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"FOV120"))
from reconstruction_ssh import connect

REMOTE_ROOT="/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor"
REMOTE=REMOTE_ROOT+"/experiments/ELLIPSE500x300_H120"


def command(client,text):
    _,out,err=client.exec_command(text,timeout=120)
    output=out.read().decode()
    if out.channel.recv_exit_status():
        raise RuntimeError(err.read().decode(errors="replace") or output)
    return output.strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level",choices=("1e9","5e9","1e10"),required=True)
    parser.add_argument("--nodes",type=int,default=8)
    parser.add_argument("--gpus-per-node",type=int,default=1)
    parser.add_argument("--accepted-budget",type=int,default=550000)
    parser.add_argument("--attempt",type=int,default=1)
    parser.add_argument("--validate-only",action="store_true",
                        help="Check the full transferred input and budget without submitting")
    args=parser.parse_args()
    if not 1<=args.nodes<=8 or args.gpus_per_node not in (1,2,3,4,6,8) or args.attempt<1:
        parser.error("Invalid topology or attempt")
    budget=estimate(args.accepted_budget,args.nodes*args.gpus_per_node,
                    args.gpus_per_node,22,55*args.gpus_per_node)
    if not budget["gpu_20_percent_margin_pass"] or not budget["host_20_percent_margin_pass"]:
        raise ValueError("Pilot resource estimate fails the 20% margin")
    record_path=HERE/f"reports/NEMA_Body_H60/{args.level}/pilot_job_{args.attempt}.json"
    if record_path.exists():
        print(record_path.read_text()); return  # Never submit the same attempt twice.
    job_name=f"NEMA_{args.level}_pilot_a{args.attempt}"
    total={"1e9":10**9,"5e9":5*10**9,"1e10":10**10}[args.level]
    checker=r'''
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); level=sys.argv[2]; expected=int(sys.argv[3])
def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): sha.update(block)
    return sha.hexdigest()
files=json.loads((root/f'nema_h60_imaging_{level}_files.json').read_text())
if len(files)!=23: raise ValueError('Expected 23 imaging inputs')
for relative,item in files.items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Unsafe input path')
    if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
        raise ValueError(f'Transferred input changed: {relative}')
path=root/f'collections/NEMA_Body_H60_{level}.json'; row=json.loads(path.read_text())
if (row['dataset']!='NEMA_Body_H60' or row['level']!=level or
    row['views']!=list(range(1,21)) or sum(row['primary_counts'])!=expected or
    sorted(row['worker_indices'])!=list(range(200)) or len(row['seeds'])!=200 or len(set(row['seeds']))!=200):
    raise ValueError('Primary/view/seed closure failed')
print(json.dumps({'primaries':sum(row['primary_counts']),'collection_sha256':digest(path),'files_verified':len(files)}))
'''
    with connect() as client:
        closure=json.loads(command(client,"/data/home/scxi717/.conda/envs/torch/bin/python -c "+shlex.quote(checker)+" "+
            shlex.quote(REMOTE+"/generated")+" "+args.level+" "+str(total)))
        if args.validate_only:
            print(json.dumps({"input_closure":closure,"planning_budget":budget,
                              "submitted":False},indent=2)); return
        queued=command(client,"squeue -u scxi717 -h -o '%j'").splitlines()
        if any(name.startswith(f"NEMA_{args.level}_") for name in queued):
            raise RuntimeError("A NEMA job for this dose is already queued or running")
        env=(f"ALL,ELLIPSE_ACCEPTED_EVENTS={args.accepted_budget},ELLIPSE_GPU_GIB=22,"
             f"ELLIPSE_HOST_GIB={55*args.gpus_per_node},ELLIPSE_GPUS_PER_NODE={args.gpus_per_node},"
             f"ELLIPSE_DATASET=NEMA_Body_H60,ELLIPSE_LEVEL={args.level},ELLIPSE_PILOT=1,NCCL_SOCKET_IFNAME=bond0")
        submit=(f"cd {shlex.quote(REMOTE_ROOT)} && sbatch --parsable -N {args.nodes} "
            f"--gres=gpu:{args.gpus_per_node} --cpus-per-task={6*args.gpus_per_node} "
            f"-p gpu_4090,gpu_5090 --time=02:00:00 --job-name={job_name} "
            f"--export={env} {shlex.quote(REMOTE+'/reconstruct.sh')}")
        job_id=command(client,submit).split(';')[0]
        if not job_id.isdigit(): raise ValueError("Invalid sbatch response")
        record={"job_id":job_id,"job_name":job_name,"level":args.level,"pilot_only":True,
            "submitted_utc":datetime.now(timezone.utc).isoformat(),"nodes":args.nodes,
            "gpus_per_node":args.gpus_per_node,"world_size":args.nodes*args.gpus_per_node,
            "iterations":10,"save_step":10,"accepted_budget_not_measured":args.accepted_budget,
            "planning_budget":budget,"input_closure":closure,
            "result_name":f"NEMA_Body_H60_{args.level}_{job_id}","nccl_interface":"bond0"}
        record_path.parent.mkdir(parents=True,exist_ok=True)
        record_path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(record,indent=2))


if __name__=="__main__": main()
