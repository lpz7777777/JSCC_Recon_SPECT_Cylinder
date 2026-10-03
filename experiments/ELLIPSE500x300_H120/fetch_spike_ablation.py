"""Fetch small ablation gate evidence; optionally retrieve verified formal data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"FOV120"))
from reconstruction_ssh import connect

BASE="/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120"
REPORT=HERE/"reports/NEMA_Body_H60/spike_ablation"
STUDY=HERE/"generated/SpikeAblation/NEMA_5e9_SPIKE_ABLATION_V1/study.json"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",action="store_true",help="Download passed formal histories and prediction (large)")
    a=p.parse_args()
    study=json.loads(STUDY.read_text())
    study_sha=hashlib.sha256(STUDY.read_bytes()).hexdigest()
    completed=[]
    status={"study":study["study"],"variants":{}}
    gates=REPORT/"gates"; gates.mkdir(exist_ok=True)
    with connect() as client,client.open_sftp() as sftp:
        for variant in study["variants"]:
            name=variant["id"]; status["variants"][name]={}
            for phase in ("pilot","short","formal"):
                path=BASE+"/generated/SpikeAblation/"+study["study"]+"/gates/"+name+"."+phase+".json"
                try:
                    with sftp.open(path,"rb") as stream:
                        data=stream.read()
                except FileNotFoundError:
                    status["variants"][name][phase]="not_verified"
                    continue
                gate=json.loads(data)
                if (not gate["passed"] or gate["study_sha256"]!=study_sha or gate["variant"]!=name or gate["phase"]!=phase):
                    raise ValueError("Remote gate does not match frozen study")
                (gates/f"{name}.{phase}.json").write_bytes(data)
                status["variants"][name][phase]={k:gate[k] for k in
                    ("result","elapsed_max_seconds","gpu_reserved_fraction","host_peak_fraction","slurm_host_fraction")}
                result=gate["result"]
                if not result.startswith(BASE+"/generated/Results/") or "/" in result[len(BASE+"/generated/Results/"):]:
                    raise ValueError("Unexpected result directory")
                for filename,key in (("run_manifest.json","run_manifest_sha256"),("optimization.json","optimization_sha256")):
                    with sftp.open(result+"/"+filename,"rb") as stream:
                        blob=stream.read()
                    if hashlib.sha256(blob).hexdigest()!=gate[key]:
                        raise ValueError("Evidence checksum failed")
                    (REPORT/f"{name}.{phase}.{filename}").write_bytes(blob)
                if phase=="formal":
                    with sftp.open(result+"/integrity_report.json","rb") as stream:
                        blob=stream.read()
                    formal=json.loads(blob)
                    if formal["job_result"]!=result.rsplit("/",1)[1] or formal["accepted_compton_events"]!=484936:
                        raise ValueError("Formal result identity mismatch")
                    if {r["channel"]:r["sha256"] for r in formal["outputs"]}!=gate["outputs_sha256"]:
                        raise ValueError("Formal and ablation integrity outputs disagree")
                    (HERE/"reports"/f'{formal["job_result"]}_integrity.json').write_bytes(blob)
                    completed.append((formal["job_result"],name))
    (REPORT/"progress.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status,indent=2))
    if a.results:
        for result,name in completed:
            subprocess.run([sys.executable,str(HERE/"fetch_nema_results.py"),result],check=True)
            dest=HERE/"generated/RemoteResults"/result
            for filename in ("optimization.json","run_manifest.json"):
                (dest/filename).write_bytes((REPORT/f"{name}.formal.{filename}").read_bytes())
            subprocess.run([sys.executable,str(HERE/"analyze_nema_result.py"),result],check=True)
            subprocess.run([sys.executable,str(HERE/"plot_nema_iterations.py"),result],check=True)


if __name__=="__main__":
    main()
