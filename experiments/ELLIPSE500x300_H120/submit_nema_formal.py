"""Submit formal NEMA imaging only after a completed full-data resource pilot."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from submit_nema_pilot import HERE, REMOTE, REMOTE_ROOT, command, connect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level",choices=("1e9","5e9","1e10"),required=True)
    parser.add_argument("--pilot-job",required=True)
    args=parser.parse_args()
    if not args.pilot_job.isdigit(): parser.error("Invalid pilot job")
    folder=HERE/f"reports/NEMA_Body_H60/{args.level}"
    record_path=folder/"formal_job.json"
    if record_path.exists():
        print(record_path.read_text()); return
    matching=[json.loads(p.read_text()) for p in folder.glob("pilot_job_*.json")
              if json.loads(p.read_text())["job_id"]==args.pilot_job]
    if len(matching)!=1: raise ValueError("Expected one locally recorded pilot")
    pilot=matching[0]
    # This checker reads all six active/full/history images and cross-talk prediction.
    report=json.loads(subprocess.check_output([sys.executable,str(HERE/"check_remote_pilot.py"),
                                               pilot["result_name"]],text=True))
    if report["peak_reserved_gib"]>22*.8:
        raise ValueError("Pilot reserve exceeds 80% of the conservative 4090 capacity")
    with connect() as client, client.open_sftp() as sftp:
        manifest_path=f'{REMOTE}/generated/Results/{pilot["result_name"]}/run_manifest.json'
        with sftp.open(manifest_path) as stream: raw=stream.read()
        manifest=json.loads(raw)
        if (manifest["dataset"]!="NEMA_Body_H60" or manifest["count_level"]!=args.level or
            manifest["world_size"]!=pilot["world_size"] or
            manifest.get("cuda_allocator_config")!="expandable_segments:True"):
            raise ValueError("Pilot/settings mismatch")
        accounting=command(client,f"sacct -j {args.pilot_job} -n -P "
                           "--format=JobID,State,ExitCode,MaxRSS,AllocTRES")
        rows=[row.split("|") for row in accounting.splitlines()]
        main_row=next(row for row in rows if row[0]==args.pilot_job)
        if main_row[1:3]!=["COMPLETED","0:0"]: raise ValueError("Pilot has not completed successfully")
        rss=[]
        for row in rows:
            if row[3]:
                value=row[3]
                if value[-1] in "KMGT":
                    rss.append(float(value[:-1])*{"K":1024,"M":2**20,"G":2**30,"T":2**40}[value[-1]])
                else:
                    rss.append(float(value))  # Slurm can report an unsuffixed zero.
        if not rss: raise ValueError("No measured Slurm MaxRSS")
        # One Slurm task/node starts gpus_per_node ranks: MaxRSS is checked per task/node.
        host_gib=55*pilot["gpus_per_node"]
        host_fraction=max(rss)/(host_gib*2**30)
        if host_fraction>.8: raise ValueError("Measured host memory exceeds 80% conservative budget")
        checker="""
import hashlib,json,pathlib,sys
base=pathlib.Path(sys.argv[1]); run=json.loads(pathlib.Path(sys.argv[2]).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
for relative,expected in run['input_sha256'].items():
 if sha(base/'generated'/relative)!=expected: raise ValueError('Changed input: '+relative)
if sha(base/'generated/Geometry/geometry.npz')!=run['geometry_sha256']: raise ValueError('Changed geometry')
if sha(base/'generated/FactorsCalibrated/440keV_RotateNum20/Sensi_d')!=run['sensi_d_sha256']: raise ValueError('Changed sensitivity')
for name,expected in run['factor_manifest_sha256'].items():
 if sha(base/'generated/FactorsCalibrated'/name/'factor_manifest.json')!=expected: raise ValueError('Changed Factors')
print('PILOT_INPUTS_UNCHANGED')
"""
        if command(client,"/data/home/scxi717/.conda/envs/torch/bin/python -c "+shlex.quote(checker)+
                   " "+shlex.quote(REMOTE)+" "+shlex.quote(manifest_path))!="PILOT_INPUTS_UNCHANGED":
            raise ValueError("Input recheck failed")
        queued=command(client,"squeue -u scxi717 -h -o '%j'").splitlines()
        if any(name.startswith(f"NEMA_{args.level}_") for name in queued):
            raise RuntimeError("A NEMA job for this dose is already active")
        accepted=manifest["accepted_compton_events"]
        env=(f"ALL,ELLIPSE_ACCEPTED_EVENTS={accepted},ELLIPSE_GPU_GIB=22,ELLIPSE_HOST_GIB={host_gib},"
             f'ELLIPSE_GPUS_PER_NODE={pilot["gpus_per_node"]},ELLIPSE_DATASET=NEMA_Body_H60,'
             f"ELLIPSE_LEVEL={args.level},ELLIPSE_PILOT=0,ELLIPSE_DRY_RUN=0,NCCL_SOCKET_IFNAME=bond0,"
             "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        submission=(f"cd {shlex.quote(REMOTE_ROOT)} && sbatch --parsable -N {pilot['nodes']} "
                    f"--gres=gpu:{pilot['gpus_per_node']} --cpus-per-task={6*pilot['gpus_per_node']} "
                    "-p gpu_4090 --time=48:00:00 "
                    f"--job-name=NEMA_{args.level}_formal --export={env} {shlex.quote(REMOTE+'/reconstruct.sh')}")
        job_id=command(client,submission).split(';')[0]
        if not job_id.isdigit(): raise ValueError("Invalid sbatch response")
        record={"job_id":job_id,"level":args.level,"dataset":"NEMA_Body_H60",
                "submitted_utc":datetime.now(timezone.utc).isoformat(),"pilot_job":args.pilot_job,
                "nodes":pilot["nodes"],"gpus_per_node":pilot["gpus_per_node"],
                "world_size":pilot["world_size"],"iterations":10000,"save_step":50,
                "accepted_events":accepted,"result_name":f"NEMA_Body_H60_{args.level}_{job_id}",
                "pilot_image_gpu_check":report,"pilot_manifest_sha256":hashlib.sha256(raw).hexdigest(),
                "pilot_slurm_accounting":accounting,"host_peak_fraction_conservative":host_fraction,
                "host_20_percent_margin_pass":True,"nccl_interface":"bond0",
                "memory_policy":"Cluster auto-grants 60GB per GPU; explicit --mem flags prohibited",
                "cuda_allocator_config":"expandable_segments:True"}
        record_path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(record,indent=2))


if __name__=="__main__": main()
