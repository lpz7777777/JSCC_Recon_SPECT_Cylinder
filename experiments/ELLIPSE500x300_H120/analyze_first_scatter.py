"""Shared-kernel event scan, two matched sensitivities, independent efficiency gates."""
from __future__ import annotations
import argparse
import collections
import csv
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
from compton_event_response import (ComptonEventSettings,PreparedComptonEvents,build_compton_cone_weights,
    build_detector_position_variance,prepare_compton_events,min_standardized_compton_arm,
    select_normalized_response_rows)
from detector_csv import load_detector_coordinates
import compton_event_response as response_module
from validate_factors import validate as validate_factors

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2)+"\n")
def subset(p,index):
    return PreparedComptonEvents(**{f.name:None if getattr(p,f.name) is None else getattr(p,f.name)[index]
                                   for f in fields(p)})
def source_bin(r):
    radius=math.hypot(float(r["source_x"]),float(r["source_y"])+345)/255
    z=abs(float(r["source_z"]));return (0 if radius<=.5 else 1 if radius<=.85 else 2)*3+(0 if z<=30 else 1 if z<=45 else 2)
def rotation_average(values,rotation):
    return sum(values[rotation[:,v]] for v in range(20))/20

def analyze(args):
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    device=torch.device(args.device)
    if device.type=="cuda":torch.cuda.set_device(device)
    inputs=args.inputs;out=args.output;out.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((inputs/"input_manifest.json").read_text())
    for name,h in provenance["files"].items():
        if digest(inputs/name)!=h:raise ValueError("Input manifest hash mismatch: "+name)
    validate_factors(args.factors)
    factor=args.factors/"440keV_RotateNum20"
    geo=np.load(args.geometry)
    coordinates=geo["coordinates_mm"].astype(np.float32);volumes=geo["cell_volume_mm3"]
    frac=geo["ellipse_fraction"];rotation=geo["rotation"]
    if coordinates.shape!=(132040,3) or len(geo["active_indices"])!=82040:raise ValueError("Geometry mismatch")
    coords=torch.tensor(coordinates,device=device)
    detector=torch.tensor(load_detector_coordinates(factor/"Detector.csv",10496),device=device)
    var=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
    memory=np.memmap(factor/"SysMat_polar",dtype="<f4",mode="r",shape=(132040,10496))
    b=torch.tensor(np.array(memory.T,copy=True),device=device);del memory
    radius=np.hypot(coordinates[:,0],coordinates[:,1])/255;z=np.abs(coordinates[:,2])
    bins=(np.where(radius<=.5,0,np.where(radius<=.85,1,2))*3+np.where(z<=30,0,np.where(z<=45,1,2)))
    masks=torch.tensor(np.eye(9,dtype=np.float32)[bins],device=device)
    binvol=np.bincount(bins,weights=volumes,minlength=9)
    groups=("legacy","ideal");scans={};sums={};accepted_meta={};gates=[]
    all_records=[];offline=[];start=time.monotonic()
    seen_seeds=set()
    for folder in sorted(p for p in inputs.iterdir() if p.is_dir()):
        dataset=folder.name;collection=json.loads((folder/"collection.json").read_text())
        seeds=collection["seeds"]
        if len(set(seeds))!=len(seeds) or seen_seeds.intersection(seeds):
            raise ValueError("Calibration/imaging/validation seeds overlap")
        seen_seeds.update(seeds)
        if dataset!="NEMA" and collection["primary_counts"][0]!=0:
            raise ValueError("Calibration must contain only primary 440 keV photons")
        for group in groups:
            accumulator=torch.zeros(132040,dtype=torch.float64,device=device)
            total=uncut=0;true_counts=np.zeros(9,dtype=np.int64);records=[];per_worker_bin={}
            emitted=np.zeros(9,dtype=np.int64)
            accepted_identities=set();rejected_rows=[]
            for view in collection["views"]:
                emitted+=np.array(json.loads((folder/f"emitted_v{view:02d}.json").read_text()))
                path=folder/f"{group}_v{view:02d}.csv"
                input_sha=provenance["files"][path.relative_to(inputs).as_posix()]
                raw=(np.loadtxt(path,delimiter=",",usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
                     if path.stat().st_size else np.empty((0,4),dtype=np.float32))
                p,diagnostics=prepare_compton_events(torch.tensor(raw,device=device),settings,detector,var,var,
                    input_energies_already_smeared=True)
                kept=[];scores=[];uncut_view=0;bin_contributions=[]
                if p is not None:
                    for offset in range(0,p.count,args.batch_size):
                        batch=subset(p,slice(offset,offset+args.batch_size))
                        cone=build_compton_cone_weights(batch,coords,settings)
                        response=cone*b[batch.cpnum1-1];norm=response.sum(1)
                        valid=torch.isfinite(response).all(1)&torch.isfinite(norm)&(norm>0)
                        if not valid.any():continue
                        valid_p=subset(batch,valid);normal=response[valid]/norm[valid,None]
                        q=min_standardized_compton_arm(valid_p,coords,settings)
                        old,_,_=select_normalized_response_rows(normal,1)
                        keep,_,_=select_normalized_response_rows(normal,1,q,3)
                        rejected_rows.extend(dict(view=view,row=int(row),q=float(score),file_sha256=input_sha)
                            for row,score in zip(valid_p.source_row_indices[old&~keep].cpu().tolist(),
                                                 q[old&~keep].cpu().tolist()))
                        uncut_view+=int(old.sum())
                        selected=normal[keep]
                        accumulator+=selected.double().sum(0)
                        kept.extend(valid_p.source_row_indices[keep].cpu().tolist())
                        scores.extend(q[keep].cpu().tolist())
                        bin_contributions.extend((selected@masks).cpu().numpy())
                selected_metadata={};wanted=set(kept)
                with (folder/f"events_v{view:02d}.csv").open() as f:
                    for r in csv.DictReader(f):
                        row=int(r[f"global_{group}_row"])
                        if row in wanted:selected_metadata[row]=r
                if len(selected_metadata)!=len(kept):raise ValueError("Selected response has no event identity")
                for row,q,bc in zip(kept,scores,bin_contributions):
                    r=selected_metadata[row];true_counts[source_bin(r)]+=1
                    accepted_identities.add((int(r["seed"]),int(r["event_id"])))
                    worker=int(r["worker"])
                    per_worker_bin.setdefault(worker,np.zeros(9));per_worker_bin[worker]+=bc
                if dataset.startswith("point_"):
                    # Deterministic diagnostic sample from held-out point sources only.
                    for row in kept[:256]:offline.append(dict(group=group,view=view,**selected_metadata[row]))
                records.append(dict(view=view,raw_rows=len(raw),uncut=uncut_view,kept=len(kept),removed=uncut_view-len(kept)))
                total+=len(kept);uncut+=uncut_view
                np.save(out/f"{dataset}_{group}_v{view:02d}_kept_rows.npy",np.asarray(kept,dtype=np.int64))
                if dataset=="NEMA":accepted_meta[(group,view)]=dict(kept=len(kept),uncut=uncut_view)
                print(json.dumps(dict(dataset=dataset,group=group,view=view,uncut=uncut_view,kept=len(kept),
                                      elapsed_seconds=time.monotonic()-start)),flush=True)
            key=f"{dataset}_{group}"
            with (out/f"{key}_q_rejected.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=("view","row","q","file_sha256"))
                writer.writeheader();writer.writerows(rejected_rows)
            np.save(out/f"{key}_event_identities.npy",np.asarray(sorted(accepted_identities),dtype=np.int64).reshape(-1,2))
            sums[key]=accumulator.cpu().numpy()
            scans[key]=dict(primary_counts=collection["primary_counts"],uncut=uncut,kept=total,
                emitted_bins=emitted.tolist(),accepted_source_bins=true_counts.tolist(),records=records,
                worker_response_bins={str(k):v.tolist() for k,v in per_worker_bin.items()})
            all_records.append(dict(dataset=dataset,group=group,uncut=uncut,kept=total))
            write(out/"scan_progress.json",all_records)
    if scans["NEMA_legacy"]["uncut"]!=97299:raise ValueError("Original 1e9 accepts must reproduce 97299")
    circle_volume=float(volumes.sum());ellipse_volume=float(np.dot(volumes,frac))
    for group in groups:
        target=out/group;target.mkdir()
        train=scans[f"circle_train_{group}"];validation=scans[f"circle_validation_{group}"]
        n=train["primary_counts"][1];nv=validation["primary_counts"][1]
        s=rotation_average(sums[f"circle_train_{group}"]*circle_volume/n,rotation)
        sv=rotation_average(sums[f"circle_validation_{group}"]*circle_volume/nv,rotation)
        if np.any(s<=0) or not np.isfinite(s).all():raise ValueError("Nonpositive matched sensitivity")
        s.astype("<f4").tofile(target/"Sensi_d")
        mean=float(np.dot(sv/s,volumes)/circle_volume)
        se=math.sqrt(1/train["kept"]+1/validation["kept"])
        gates.append(dict(group=group,test="independent_circle_closure",value=mean,limit=max(.02,3*se),
                          passed=abs(mean-1)<=max(.02,3*se)))
        ellipse=scans[f"ellipse_validation_{group}"]
        ne=ellipse["primary_counts"][1];eff=ellipse["kept"]/ne
        predicted=float(np.dot(s,frac)/ellipse_volume)
        ese=math.sqrt(1/ellipse["kept"]+1/train["kept"])
        discrepancy=predicted/eff-1
        gates.append(dict(group=group,test="independent_ellipse_efficiency",observed=eff,predicted=predicted,
                          relative_error=discrepancy,limit=max(.05,3*ese),passed=abs(discrepancy)<=max(.05,3*ese)))
        predicted_bins=np.bincount(bins,weights=s,minlength=9)/binvol
        emitted=np.asarray(validation["emitted_bins"]);counts=np.asarray(validation["accepted_source_bins"])
        observed_bins=counts/emitted
        # Independent worker estimates of the training response uncertainty.
        contributions=np.array([train["worker_response_bins"].get(str(i),[0.]*9) for i in range(200)])
        worker_eff=contributions*circle_volume/(5_000_000*binvol)
        model_se=worker_eff.std(axis=0,ddof=1)/math.sqrt(200)
        observed_se=np.sqrt(observed_bins*(1-observed_bins)/emitted)
        for i in range(9):
            relative=float(predicted_bins[i]/observed_bins[i]-1)
            relative_se=float(math.hypot(model_se[i],observed_se[i])/observed_bins[i])
            adequate=int(counts[i])>=400
            passes=adequate and not(abs(relative)>.2 and abs(relative)>3*relative_se)
            gates.append(dict(group=group,test="direct_source_bin_efficiency",bin=i,
                emitted=int(emitted[i]),accepted=int(counts[i]),observed=float(observed_bins[i]),
                predicted=float(predicted_bins[i]),relative_error=relative,relative_standard_error=relative_se,
                statistically_adequate=adequate,passed=passes))
        sensitivity_provenance=dict(experiment="ELLIPSE500x300_H120",pixel_count=132040,operator="K*B",
            event_policy="legacy" if group=="legacy" else "ideal_first_scatter_v2",
            resolution_fwhm=.13,reference_keV=511,sum_threshold_MeV=.350,input_already_smeared=True,
            max_min_standardized_arm=3,source_photons=n,independent_validation_photons=nv,
            geometry_sha256=digest(args.geometry),input_manifest_sha256=digest(inputs/"input_manifest.json"))
        write(target/"Sensi_d_provenance.json",sensitivity_provenance)
        np.save(target/"independent_circle_ratio.npy",sv/s)
    gate=dict(study="compton_first_scatter_v2",status="PASSED" if all(x["passed"] for x in gates) else "HOLD",
        gates=gates,scans=scans,input_manifest_sha256=digest(inputs/"input_manifest.json"),
        geometry_sha256=digest(args.geometry),factor_manifest_sha256={p.name:digest(p/"factor_manifest.json")
            for p in args.factors.iterdir() if p.is_dir() and (p/"factor_manifest.json").exists()},
        kernel_sha256=digest(Path(response_module.__file__)),nema_pair=accepted_meta_to_json(accepted_meta))
    write(out/"validation_gate.json",gate)
    offline_energy_diagnostics(offline,out)
    print("FIRST_SCATTER_ANALYSIS",gate["status"],flush=True)
    return gate

def accepted_meta_to_json(values):return {g:{str(v):r for (gg,v),r in values.items() if gg==g} for g in ("legacy","ideal")}

def offline_energy_diagnostics(rows,out):
    results=[]
    for r in rows:
        if r["first_process"]!="compt" or int(r["c1"])<=0 or int(r["c2"])<=0:continue
        source=np.array([float(r["source_"+k]) for k in "xyz"])
        p1=np.array([float(r["p1_"+k]) for k in "xyz"]);p2=np.array([float(r["p2_"+k]) for k in "xyz"])
        a=p1-source;b=p2-p1
        if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:continue
        cosine=np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)
        predicted=.440-.440/(1+(.440/.511)*(1-cosine))
        sigma=.13/2.35482*np.sqrt(.511*max(predicted,1e-12))
        results.append(dict(group=r["group"],view=r["view"],event_id=r["event_id"],
            predicted_transfer=predicted,true_transfer=float(r["transfer_mev"]),
            accumulated=float(r["true_e1"]),measured=float(r["measured_e1"]),
            predicted_sigma=sigma,pull=(float(r["measured_e1"])-predicted)/sigma,
            reason=r["reason"],legacy_pair_matches_first=(int(r["legacy_c1"])==int(r["c1"]) and
                                                       int(r["legacy_c2"])==int(r["c2"]))))
    with (out/"energy_domain_diagnostics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]) if results else ["group"]);writer.writeheader();writer.writerows(results)
    summary=[]
    for group in ("legacy","ideal"):
        p=np.array([r["pull"] for r in results if r["group"]==group])
        summary.append(dict(group=group,samples=len(p),coverage={str(k):float(np.mean(np.abs(p)<=k)) for k in (1,2,3)}
                            if len(p) else {},note="Diagnostic Gaussian prototype; no production kernel changes"))
    write(out/"energy_domain_diagnostics.json",summary)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--factors",type=Path,required=True)
    p.add_argument("--geometry",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=64)
    a=p.parse_args()
    with torch.no_grad():analyze(a)
if __name__=="__main__":main()
