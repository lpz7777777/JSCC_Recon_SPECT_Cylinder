"""Isolated, bounded paired Geant4 production; no reconstruction submission here."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STUDY="compton_first_scatter_v2"
DATA=HERE/"generated"/STUDY
REPORT=HERE/"reports/NEMA_Body_H60"/STUDY
REMOTE_ROOT="/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928"
REMOTE=f"{REMOTE_ROOT}/experiments/ELLIPSE500x300_H120/generated/{STUDY}"
HOST="maty@192.168.11.1"
OUTPUTS=("List.csv","ListIdeal.csv","EventContract.csv","PrimaryTrace.csv","EmittedBins.csv",
         "CntStat_218.csv","CntStat_440.csv","PrimaryCount.csv")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)

def gps(point=None):
    body="/gps/source/multiplevertex false\n/gps/particle gamma\n/gps/energy 440 keV\n/gps/number 1\n"
    p=point or (0,-345,0)
    body+=f"/gps/pos/centre {p[0]:.12g} {p[1]:.12g} {p[2]:.12g} mm\n"
    if point is None:
        body+="/gps/pos/type Volume\n/gps/pos/shape Cylinder\n/gps/pos/radius 255 mm\n/gps/pos/halfz 60 mm\n"
    else: body+="/gps/pos/type Point\n"
    return body+"/gps/ang/type iso\n/gps/ang/mintheta 0 deg\n/gps/ang/maxtheta 180 deg\n"

def prepare():
    if (DATA/"jobs.json").exists(): raise FileExistsError("Immutable task manifest already exists")
    (DATA/"macros").mkdir(parents=True,exist_ok=True)
    jobs=[]
    def add(dataset,view,worker,photons,seed,body,object_point=None):
        name=f"macros/{dataset}_v{view:02d}.mac"
        target=DATA/name
        macro=re.sub(r"/run/beamOn\s+\d+", "",body).rstrip()+f"\n/run/beamOn {photons}\n"
        if target.exists() and target.read_text()!=macro: raise ValueError("Conflicting macro")
        target.write_text(macro,encoding="ascii")
        jobs.append(dict(index=len(jobs),dataset=dataset,view=view,worker=worker,
                         photons=photons,seed=seed,macro=name,macro_sha256=digest(target),object_mm=object_point))
    old=HERE/"generated/NEMA_Body_H60/Simulation_1e9"
    for j in json.loads((old/"jobs.json").read_text())["jobs"]:
        add("NEMA",j["view"],j["worker"],j["photons"],j["seed"],(old/j["macro"]).read_text())
    for family,seed0 in (("circle_train",41004001),("circle_validation",41005001)):
        for worker in range(200): add(family,1,worker,5_000_000,seed0+worker,gps())
    for v in range(1,21):
        body=f"/ellipse/clear\n/ellipse/centerY -345\n/ellipse/angle {(v-1)*18}\n/ellipse/add 440 250 150 60 1\n"
        add("ellipse_validation",v,0,5_000_000,41006000+v,body)
    sites=((0,0,0),(225,0,0),(-225,0,0),(0,135,0),(0,-135,0),(0,0,57),(0,0,-57))
    for site,p in enumerate(sites):
        for v in range(1,21):
            t=(v-1)*math.pi/10; x,y,z=p
            world=(x*math.cos(t)+y*math.sin(t),-345+y*math.cos(t)-x*math.sin(t),z)
            add(f"point_{site}",v,0,500_000,41007000+site*20+v,gps(world),list(p))
    if len({j["seed"] for j in jobs})!=len(jobs): raise ValueError("Duplicate production seed")
    totals=collections.Counter()
    for j in jobs: totals[j["dataset"]]+=j["photons"]
    if sum(totals.values())!=3_170_000_000: raise ValueError("Photon budget mismatch")
    write(DATA/"jobs.json",dict(study=STUDY,jobs=jobs,photon_totals=dict(totals),
          total_photons=sum(totals.values()),policy="paired",primary_per_event=1,
          expected_legacy_NEMA_compton_uncut=97299))
    write(REPORT/"configuration.json",dict(study=STUDY,iterations=2000,save_step=50,
          groups=["legacy","ideal_first_scatter_v2"],max_min_standardized_arm=3,
          quality_points=132040,active_columns=82040,first_energy_tolerance_MeV=[1e-6,1e-5],
          channels=["440_ComptonOnly","440_SinglePlusCompton"],mip_height_mm=72,
          transport_budget=dict(totals),total_photons=sum(totals.values()),
          old_jobs_remain_stopped=True,automatic_extensions=False))
    print(json.dumps(dict(workers=len(jobs),total_photons=sum(totals.values()))))

def validate_worker(folder,job,photons=None,paired=True):
    expected=photons or job["photons"]
    with (folder/"PrimaryCount.csv").open() as f: primary=[list(map(int,r)) for r in csv.reader(f)]
    if len(primary)!=1 or len(primary[0])!=3 or sum(primary[0])!=expected or primary[0][2]:
        raise ValueError("PrimaryCount does not close")
    if job["dataset"]!="NEMA" and primary[0]!=[0,expected,0]: raise ValueError("Non-440 calibration")
    for energy in (218,440):
        with (folder/f"CntStat_{energy}.csv").open() as f: rows=list(csv.reader(f))
        if len(rows)!=1 or len(rows[0])!=10496 or any(int(x)<0 for x in rows[0]):
            raise ValueError("Invalid detector counts")
    counts={}; lists={}
    for name in (("List.csv","ListIdeal.csv") if paired else ("List.csv",)):
        values=[]
        with (folder/name).open() as f:
            for r in csv.reader(f):
                if len(r)!=5 or int(r[4])!=1: raise ValueError("Invalid List row")
                a,e1,b,e2,_=map(float,r)
                if not(1<=a<=10496 and 1<=b<=10496 and a==int(a) and b==int(b)
                       and a!=b and math.isfinite(e1) and math.isfinite(e2) and e1>0 and e2>0):
                    raise ValueError("Nonfinite or invalid List")
                values.append((int(a),e1,int(b),e2))
        counts[name]=len(values); lists[name]=values
    reasons=collections.Counter(); intersections=collections.Counter()
    if paired:
        next_legacy=next_ideal=0
        with (folder/"EventContract.csv").open() as f:
            previous=-1
            for r in csv.DictReader(f):
                event=int(r["event_id"])
                if event<=previous or event>=expected: raise ValueError("Duplicate or invalid event ID")
                previous=event
                if (r["dataset"],int(r["worker"]),int(r["seed"]),int(r["view"]))!=(job["dataset"],job["worker"],job["seed"],job["view"]):
                    raise ValueError("Event identity differs from worker")
                l=bool(int(r["legacy"])); i=bool(int(r["ideal"]))
                intersections[f"legacy_{int(l)}_ideal_{int(i)}"]+=1; reasons[r["reason"]]+=1
                if l:
                    if int(r["legacy_row"])!=next_legacy: raise ValueError("Legacy row mapping gap")
                    if lists["List.csv"][next_legacy][::2]!=(int(r["legacy_c1"]),int(r["legacy_c2"])):
                        raise ValueError("Legacy pair mapping mismatch")
                    next_legacy+=1
                if i:
                    if int(r["ideal_row"])!=next_ideal: raise ValueError("Ideal row mapping gap")
                    actual=lists["ListIdeal.csv"][next_ideal]
                    c1,c2=int(r["c1"]),int(r["c2"])
                    if actual[::2]!=(c1,c2): raise ValueError("Ideal pair mapping mismatch")
                    for x,key in zip(actual[1::2],("measured_e1","measured_e2")):
                        if abs(x-float(r[key]))>5e-6*max(abs(x),1e-3): raise ValueError("Measured energy mapping mismatch")
                    transfer=float(r["transfer_mev"]); tol=max(1e-6,1e-5*transfer)
                    if r["reason"]!="accepted" or abs(float(r["true_e1"])-transfer)>tol or float(r["foreign_e1"])>tol:
                        raise ValueError("Accepted ideal event violates energy contract")
                    next_ideal+=1
        if (next_legacy,next_ideal)!=(counts["List.csv"],counts["ListIdeal.csv"]):
            raise ValueError("Unmapped List rows")
        with (folder/"EmittedBins.csv").open() as f: bins=list(csv.DictReader(f))
        if len(bins)!=9 or sum(int(r["primary_440"]) for r in bins)!=primary[0][1]:
            raise ValueError("Emitted source histogram does not close")
    return dict(primary_counts=primary[0],list_rows=counts,reasons=dict(reasons),intersections=dict(intersections))

def run_worker(index,executable,base=DATA,photons=None,policy="paired",folder=None):
    manifest=json.loads((base/"jobs.json").read_text()); job=manifest["jobs"][index]
    macro=base/job["macro"]
    if digest(macro)!=job["macro_sha256"]: raise ValueError("Macro hash mismatch")
    folder=folder or base/"workers"/f"{index:05d}"
    if (folder/"worker.json").exists():
        old=json.loads((folder/"worker.json").read_text())
        if old.get("status")=="complete":
            for n,h in old["output_sha256"].items():
                if digest(folder/n)!=h: raise ValueError("Completed worker output changed")
            print("ALREADY_VERIFIED",index); return old
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(base/"source/CrystalMatrix.txt",folder/"CrystalMatrix.txt")
    body=macro.read_text(); actual=photons or job["photons"]
    (folder/"run.mac").write_text(re.sub(r"/run/beamOn\s+\d+",f"/run/beamOn {actual}",body),encoding="ascii")
    env=os.environ.copy();env.update(JSCC_RANDOM_SEED=str(job["seed"]),JSCC_COMPTON_POLICY=policy,
        JSCC_DATASET=job["dataset"],JSCC_WORKER=str(job["worker"]),JSCC_VIEW=str(job["view"]))
    env.pop("JSCC_COMPTON_DIAGNOSTICS",None)
    start=time.monotonic()
    with (folder/"console.log").open("w") as log:
        p=subprocess.run([str(executable.resolve()),"run.mac"],cwd=folder,env=env,
                         stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    if p.returncode: raise RuntimeError(f"Geant4 failed ({p.returncode}): {folder}")
    result=validate_worker(folder,job,actual,policy!="legacy")
    result.update(job,status="complete",simulated_photons=actual,elapsed_seconds=time.monotonic()-start,
                  executable_sha256=digest(executable),policy=policy)
    names=OUTPUTS if policy!="legacy" else ("List.csv","CntStat_218.csv","CntStat_440.csv","PrimaryCount.csv")
    result["output_sha256"]={n:digest(folder/n) for n in names}
    write(folder/"worker.json",result);print(json.dumps(result));return result

def smoke(base):
    executable=base/"build/gamma01"
    checks=[]
    for index in (0,10,20):
        for policy in ("legacy","paired"):
            run_worker(index,executable,base,10000,policy,base/"smoke"/f"{index}_{policy}")
        for name in ("List.csv","CntStat_218.csv","CntStat_440.csv","PrimaryCount.csv"):
            if digest(base/"smoke"/f"{index}_legacy"/name)!=digest(base/"smoke"/f"{index}_paired"/name):
                raise ValueError("Recorder changed legacy random stream: "+name)
        checks.append(dict(index=index,seed=json.loads((base/"jobs.json").read_text())["jobs"][index]["seed"],
                           identical_outputs=True))
    # Full first worker must replay the already accepted legacy production exactly.
    result=run_worker(0,executable,base)
    old=Path(REMOTE_ROOT)/"experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/workers/00000"
    for name in ("List.csv","CntStat_218.csv","CntStat_440.csv","PrimaryCount.csv"):
        if digest(base/"workers/00000"/name)!=digest(old/name):
            raise ValueError("Full legacy replay differs from archived production: "+name)
    gate=dict(status="PASSED",smoke=checks,first_worker=result,
              legacy_full_worker_bitwise_replay=True,manifest_sha256=digest(base/"jobs.json"))
    write(base/"transport_gate.json",gate);print("FIRST_SCATTER_TRANSPORT_GATE_OK")

def remote(command):
    p=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",HOST,command],
                     capture_output=True,text=True,check=True)
    return p.stdout.strip()

def deploy():
    if not (DATA/"jobs.json").exists(): prepare()
    archive=DATA/"deployment.tar.gz"
    sources=ROOT/"Geant4Sim/Geant4Code"
    with tarfile.open(archive,"w:gz") as tar:
        for path in sorted(sources.rglob("*")):
            if path.is_file() and path.suffix in (".hh",".cc",".txt",".mac"):
                tar.add(path,arcname="source/"+path.relative_to(sources).as_posix())
        tar.add(Path(__file__),arcname="first_scatter_workflow.py")
        tar.add(HERE/"first_scatter_build.sh",arcname="first_scatter_build.sh")
        tar.add(HERE/"first_scatter_array.sh",arcname="first_scatter_array.sh")
        tar.add(DATA/"jobs.json",arcname="jobs.json")
        tar.add(DATA/"macros",arcname="macros")
    remote(f"test ! -e {REMOTE}/jobs.json && mkdir -p {REMOTE}/logs")
    subprocess.run(["scp","-o","BatchMode=yes",str(archive),HOST+":"+REMOTE+"/deployment.tar.gz"],check=True)
    if remote(f"sha256sum {REMOTE}/deployment.tar.gz").split()[0]!=digest(archive): raise ValueError("Transfer hash mismatch")
    remote(f"tar --no-same-owner -xzf {REMOTE}/deployment.tar.gz -C {REMOTE}")
    text=remote(f"cd {REMOTE} && sbatch --parsable --output={REMOTE}/logs/build.%j.out --error={REMOTE}/logs/build.%j.err first_scatter_build.sh")
    if not re.fullmatch(r"\d+",text): raise ValueError("Invalid build job ID")
    write(REPORT/"deployment.json",dict(remote=REMOTE,build_job=int(text),archive_sha256=digest(archive),
          manifest_sha256=digest(DATA/"jobs.json"),status="BUILD_SMOKE_AND_FIRST_WORKER_SUBMITTED"))
    print("BUILD_JOB",text)

def submit():
    gate=json.loads(remote(f"cat {REMOTE}/transport_gate.json"))
    if gate["status"]!="PASSED" or gate["manifest_sha256"]!=digest(DATA/"jobs.json"):
        raise ValueError("Transport gate not verified")
    active=remote("squeue -u maty -h -o '%i %j %T'")
    if "FIRST_SCATTER_V2" in active: raise RuntimeError("This experiment already has an active array")
    text=remote(f"cd {REMOTE} && sbatch --parsable --array=1-759%32 --output={REMOTE}/logs/worker.%A_%a.out --error={REMOTE}/logs/worker.%A_%a.err first_scatter_array.sh")
    if not re.fullmatch(r"\d+",text): raise ValueError("Invalid array job ID")
    write(REPORT/"transport_jobs.json",dict(array_job=int(text),gate=gate,total_workers=760,
          maximum_concurrency=32,total_primary_photons=3_170_000_000))
    print("TRANSPORT_ARRAY",text)

def check(base):
    manifest=json.loads((base/"jobs.json").read_text()); records=[]; missing=[]
    for job in manifest["jobs"]:
        folder=base/"workers"/f'{job["index"]:05d}'
        if not (folder/"worker.json").exists(): missing.append(job["index"]);continue
        record=json.loads((folder/"worker.json").read_text())
        if record.get("status")!="complete": raise ValueError("Failed worker")
        for n,h in record["output_sha256"].items():
            if digest(folder/n)!=h: raise ValueError("Worker hash changed")
        records.append(record)
    totals=collections.Counter()
    for r in records: totals[r["dataset"]]+=r["simulated_photons"]
    result=dict(completed_workers=len(records),missing_workers=missing,photon_totals=dict(totals),
                status="COMPLETE" if not missing else "IN_PROGRESS")
    write(base/"transport_progress.json",result);print(json.dumps(result))

def collect(base):
    check(base)
    if json.loads((base/"transport_progress.json").read_text())["status"]!="COMPLETE":
        raise ValueError("Cannot collect incomplete transport")
    out=base/"analysis_inputs";out.mkdir(exist_ok=False)
    jobs=json.loads((base/"jobs.json").read_text())["jobs"]
    # Every NEMA worker, not only the initial gate worker, must replay the
    # original production. Compare bytes before forming any derived inputs.
    old=Path(REMOTE_ROOT)/"experiments/ELLIPSE500x300_H120/generated/NEMA_Body_H60/Simulation_1e9/workers"
    replay=[]
    for job in jobs:
        if job["dataset"]!="NEMA":continue
        folder=base/"workers"/f'{job["index"]:05d}'
        original=old/f'{job["index"]:05d}'
        for name in ("List.csv","CntStat_218.csv","CntStat_440.csv","PrimaryCount.csv"):
            if digest(folder/name)!=digest(original/name):
                raise ValueError(f"NEMA legacy replay differs: worker {job['index']}, {name}")
        replay.append(job["index"])
    if replay!=list(range(200)):raise ValueError("NEMA replay worker coverage differs")
    write(base/"legacy_replay.json",dict(passed=True,workers=replay,bitwise_outputs=True))
    groups=collections.defaultdict(list)
    for j in jobs: groups[j["dataset"]].append(j)
    for dataset,selected in groups.items():
        dst=out/dataset;dst.mkdir()
        totals=collections.Counter();raw_counts=collections.Counter();reasons=collections.Counter()
        for view in sorted({j["view"] for j in selected}):
            offsets={"legacy":0,"ideal":0};counts={218:[0]*10496,440:[0]*10496}
            emitted=[0]*9
            with (dst/f"legacy_v{view:02d}.csv").open("w") as legacy, \
                 (dst/f"ideal_v{view:02d}.csv").open("w") as ideal, \
                 (dst/f"events_v{view:02d}.csv").open("w",newline="") as metadata:
                writer=None
                for j in sorted((j for j in selected if j["view"]==view),key=lambda x:x["worker"]):
                    folder=base/"workers"/f'{j["index"]:05d}'
                    record=json.loads((folder/"worker.json").read_text())
                    for e,n in zip((218,440,"other"),record["primary_counts"]): totals[e]+=n
                    for key,n in record["reasons"].items(): reasons[key]+=n
                    for name,stream in (("List.csv",legacy),("ListIdeal.csv",ideal)):
                        with (folder/name).open() as f: shutil.copyfileobj(f,stream)
                    with (folder/"EventContract.csv").open() as f:
                        reader=csv.DictReader(f)
                        if writer is None:
                            writer=csv.DictWriter(metadata,fieldnames=reader.fieldnames+["global_legacy_row","global_ideal_row"])
                            writer.writeheader()
                        for r in reader:
                            for group in ("legacy","ideal"):
                                row=int(r[group+"_row"])
                                r["global_"+group+"_row"]=offsets[group]+row if row>=0 else -1
                            writer.writerow(r)
                    for group,name in (("legacy","List.csv"),("ideal","ListIdeal.csv")):
                        offsets[group]+=record["list_rows"][name]
                        raw_counts[group]+=record["list_rows"][name]
                    for e in counts:
                        with (folder/f"CntStat_{e}.csv").open() as f: row=list(map(int,next(csv.reader(f))))
                        counts[e]=[a+b for a,b in zip(counts[e],row)]
                    with (folder/"EmittedBins.csv").open() as f:
                        for r in csv.DictReader(f): emitted[int(r["radial_bin"])*3+int(r["axial_bin"])]+=int(r["primary_440"])
            for e,row in counts.items():
                with (dst/f"counts_{e}_v{view:02d}.csv").open("w",newline="") as f:csv.writer(f).writerow(row)
            write(dst/f"emitted_v{view:02d}.json",emitted)
        write(dst/"collection.json",dict(dataset=dataset,views=sorted({j["view"] for j in selected}),
              primary_counts=[totals[218],totals[440],totals["other"]],seeds=[j["seed"] for j in selected],
              worker_indices=[j["index"] for j in selected],raw_counts=dict(raw_counts),reasons=dict(reasons)))
    files={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write(out/"input_manifest.json",dict(study=STUDY,files=files,jobs_sha256=digest(base/"jobs.json"),
          legacy_replay_sha256=digest(base/"legacy_replay.json")))
    with tarfile.open(base/"analysis_inputs.tar.gz","w:gz") as tar:
        tar.add(out,arcname="analysis_inputs")
    write(base/"collection_ready.json",dict(status="READY",archive_sha256=digest(base/"analysis_inputs.tar.gz"),
          input_manifest_sha256=digest(out/"input_manifest.json")))
    print("FIRST_SCATTER_COLLECTION_READY")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","deploy","submit","run","smoke","check","collect"))
    p.add_argument("--base",type=Path,default=DATA);p.add_argument("--index",type=int)
    a=p.parse_args()
    if a.mode=="prepare": prepare()
    elif a.mode=="deploy": deploy()
    elif a.mode=="submit": submit()
    elif a.mode=="smoke": smoke(a.base)
    elif a.mode=="check": check(a.base)
    elif a.mode=="collect": collect(a.base)
    elif a.mode=="run":
        if json.loads((a.base/"transport_gate.json").read_text())["status"]!="PASSED": raise ValueError("Gate not passed")
        run_worker(a.index,a.base/"build/gamma01",a.base)
if __name__=="__main__": main()
