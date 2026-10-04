"""Distributed six-output active-ellipse JSCC MLEM.

Inputs must be calibrated Factors, an independently validated full-circle
Sensi_d, and collected Geant4 data from this experiment. Large raw Factors
are memory mapped; only detector shards are copied to each GPU. K*B events
are compacted to active object columns immediately after materialization.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import io
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
try:
    import resource as resource_usage
except ImportError:  # Windows local preflight; production resource gates require Linux.
    resource_usage=None

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("JSCC_PROJECT_ROOT", str(HERE.parents[1])))
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "distributed/dual_energy_compton_python")]
from compton_sparse_ops import build_compton_sparse_projector, materialize_sparse_event_rows_to_fine
from detector_csv import load_detector_coordinates
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from torch_active_operator import (ActiveGeometry, ViewResponse, forward_project,
                                   single_mlem, compton_and_joint_mlem)
from validate_factors import NAMES, validate


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8 << 20),b""):
            sha.update(block)
    return sha.hexdigest()


def setup(backend):
    for name in ("RANK","LOCAL_RANK","WORLD_SIZE"):
        if name not in os.environ:
            raise RuntimeError("Use torchrun for the distributed entry")
    rank=int(os.environ["RANK"])
    local=int(os.environ["LOCAL_RANK"])
    world=int(os.environ["WORLD_SIZE"])
    if backend=="nccl":
        torch.cuda.set_device(local)
        device=torch.device(f"cuda:{local}")
    else:
        device=torch.device("cpu")
    print(f"[rank {rank}] initializing {backend} process group",flush=True)
    dist.init_process_group(backend=backend,init_method="env://",
                            timeout=timedelta(minutes=5))
    print(f"[rank {rank}] process group ready",flush=True)
    return rank,world,device


def split_bins(total,rank,world):
    start=total*rank//world
    stop=total*(rank+1)//world
    if stop==start:
        raise ValueError("More ranks than detector bins")
    return start,stop


def read_event_partition(path,rank,world):
    size=path.stat().st_size
    lower=size*rank//world
    upper=size*(rank+1)//world
    with path.open("rb") as stream:
        if lower:
            stream.seek(lower-1)
            if stream.read(1)!=b"\n":
                stream.readline()
        start=stream.tell()
        chunks=[]
        while stream.tell()<upper:
            line=stream.readline()
            if not line: break
            chunks.append(line)
        end=stream.tell()
    if chunks:
        rows=np.loadtxt(io.BytesIO(b"".join(chunks)),delimiter=",",
                        usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
    else:
        rows=np.empty((0,4),np.float32)
    return torch.from_numpy(np.ascontiguousarray(rows)),[start,end,len(rows)]


def load_matrix(root,name,pixels,detectors):
    folder=root/name
    record=json.loads((folder/"factor_manifest.json").read_text())
    if not record.get("calibration",{}).get("enabled"):
        raise ValueError(f"Uncalibrated factor: {folder}")
    if (folder/"SysMat_polar").stat().st_size!=pixels*detectors*4:
        raise ValueError(f"Factor size mismatch: {folder}")
    return np.memmap(folder/"SysMat_polar",mode="c",dtype="<f4",
                     shape=(pixels,detectors))


def local_rows(raw,start,stop,device):
    return torch.from_numpy(np.ascontiguousarray(raw[:,start:stop].T)).to(device)


def full_rows(raw,device):
    # This read-only strided view shares the host page cache between ranks.
    return torch.from_numpy(raw.T).to(device)


def collect_response(output,name,image,history,geometry,rank):
    if rank: return
    active=image.detach().cpu().numpy().reshape(-1).astype("<f4")
    active.tofile(output/f"Image_{name}_active.float32")
    full=np.zeros(geometry.full_count,dtype="<f4")
    full[geometry.object_active.cpu().numpy()]=active
    full.tofile(output/f"Image_{name}_full.float32")
    if history is not None:
        history.numpy().astype("<f4").tofile(output/f"Image_{name}_history.float32")


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factors",type=Path,required=True)
    p.add_argument("--geometry",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--level",choices=("1e9","5e9","1e10"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--iterations",type=int,default=10000)
    p.add_argument("--save-step",type=int,default=50)
    p.add_argument("--backend",choices=("nccl","gloo"),default="nccl")
    p.add_argument("--cache-views",choices=("none","cpu","device"),default="none")
    p.add_argument("--event-chunks",type=int,default=16)
    p.add_argument("--event-block",type=int,default=512)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--pilot-only",action="store_true",
                   help="Run full events/grid for 10 iterations to measure peak memory")
    p.add_argument("--study-json",type=Path)
    p.add_argument("--variant")
    p.add_argument("--study-short",action="store_true",
                   help="Frozen spike-ablation 200-iteration numerical check")
    p.add_argument("--channels",choices=("six","compton-jscc"),default="six")
    p.add_argument("--response-filter-config",type=Path)
    p.add_argument("--compton-sensitivity",type=Path)
    p.add_argument("--baseline-regression",action="store_true",
                   help="Filter-off 50-iteration check for the response-mismatch study")
    args=p.parse_args()
    if args.iterations<=0 or args.iterations%args.save_step:
        raise ValueError("Iterations must divide by save-step")
    cfg=json.loads((HERE/"config.json").read_text())
    if args.pilot_only and args.iterations!=10:
        raise ValueError("Resource pilot must run exactly 10 iterations")
    if args.study_short and (args.iterations!=200 or args.study_json is None):
        raise ValueError("Study short run requires frozen study and 200 iterations")
    if args.baseline_regression and (args.iterations!=50 or args.save_step!=50 or
        args.channels!="compton-jscc" or args.response_filter_config is None):
        raise ValueError("Baseline regression requires response study, two channels and 50 iterations")
    response_study=None
    first_scatter_study=False
    max_min_arm=None
    if args.response_filter_config is not None:
        response_study=json.loads(args.response_filter_config.read_text())
        first_scatter_study=response_study["study"]=="compton_first_scatter_v2"
        if (response_study["study"] not in ("response_mismatch_cut3_v1","compton_first_scatter_v2") or
            response_study["max_min_standardized_arm"]!=3.0 or
            response_study["quality_domain"]!="full_circle_132040" or
            args.channels!="compton-jscc" or args.study_json is not None or
            args.dataset!="NEMA_Body_H60" or args.level!=("1e9" if first_scatter_study else "5e9")):
            raise ValueError("Response mismatch study configuration mismatch")
        if first_scatter_study and (response_study["iterations"]!=2000 or
            response_study["save_step"]!=50 or
            response_study["group"] not in ("legacy","ideal_first_scatter_v2") or
            not response_study["validation_gate_passed"]):
            raise ValueError("First-scatter paired study has not passed its frozen validation gate")
        if first_scatter_study:
            if not (args.baseline_regression or args.pilot_only or args.dry_run) and (
                args.iterations!=2000 or args.save_step!=50):
                raise ValueError("First-scatter formal imaging is bounded to 2000 iterations, save every 50")
            gate_path=args.response_filter_config.parent/"validation_gate.json"
            if (digest(gate_path)!=response_study["validation_gate_sha256"] or
                json.loads(gate_path.read_text())["status"]!="PASSED"):
                raise ValueError("First-scatter independent validation evidence differs or is on HOLD")
            if digest(ROOT/"compton_event_response.py")!=response_study["kernel_sha256"]:
                raise ValueError("First-scatter study must use the frozen, unchanged response kernel")
        max_min_arm=None if args.baseline_regression else 3.0
        if max_min_arm is not None and args.compton_sensitivity is None:
            raise ValueError("Filtered events require an explicit matched sensitivity")
    elif args.compton_sensitivity is not None or args.baseline_regression:
        raise ValueError("Sensitivity override requires a frozen response study")
    if args.iterations!=cfg["iterations"] and not (args.dry_run or args.pilot_only or args.study_short or
        args.baseline_regression or (first_scatter_study and args.iterations==2000 and args.save_step==50)):
        raise ValueError("Formal experiment requires 10000 iterations, or the frozen 2000-iteration first-scatter study")
    if args.event_chunks<=0 or args.event_block<=0:
        raise ValueError("Invalid event chunk sizes")
    study=None
    variant=None
    spatial_path=None
    if args.study_json is not None:
        study=json.loads(args.study_json.read_text())
        variant=next((v for v in study["variants"] if v["id"]==args.variant),None)
        spatial_path=args.study_json.parent/"spatial_model.npz"
        if (variant is None or args.dataset!=study["dataset"] or args.level!=study["level"] or
            digest(args.geometry)!=study["geometry_sha256"] or
            digest(spatial_path)!=study["spatial_model_sha256"]):
            raise ValueError("Frozen ablation config/geometry mismatch")
    elif args.variant is not None or args.study_short:
        raise ValueError("Variant requires study-json")
    rank,world,device=setup(args.backend)
    if rank==0: print("ELLIPSE_STAGE validate_factors",flush=True)
    validation_error=None
    input_hashes={}
    if rank==0:
        try:
            validate(args.factors)
            collection_path=args.data_root/"collections"/f"{args.dataset}_{args.level}.json"
            collection=json.loads(collection_path.read_text())
            expected={"1e9":10**9,"5e9":5*10**9,"1e10":10**10}[args.level]
            if (collection["dataset"]!=args.dataset or collection["level"]!=args.level or
                collection["views"]!=list(range(1,21)) or
                sum(collection["primary_counts"])!=expected or
                len(collection["seeds"])!=200 or len(set(collection["seeds"]))!=200 or
                sorted(collection["worker_indices"])!=list(range(200))):
                raise ValueError("Input collection dose/view/seed closure failed")
            input_paths=[collection_path]
            input_paths.extend(args.data_root/"CntStat"/f"{energy}keV_RotateNum20_Geant4JSCC"/
                               f"CntStat_{args.dataset}_{args.level}.csv" for energy in (218,440))
            input_paths.extend(args.data_root/"List/218-440keV_RotateNum20_Geant4JSCC"/
                               f"List_{args.dataset}_{args.level}"/f"{view}.csv" for view in range(1,21))
            input_hashes={path.relative_to(args.data_root).as_posix():digest(path)
                          for path in input_paths}
            if study is not None:
                if input_hashes!=study["baseline_input_sha256"]:
                    raise ValueError("Ablation must use precisely the frozen baseline inputs")
                for name,expected_hash in study["baseline_factor_manifest_sha256"].items():
                    if digest(args.factors/name/"factor_manifest.json")!=expected_hash:
                        raise ValueError("Ablation Factor provenance differs from baseline")
            if response_study is not None:
                if input_hashes!=response_study["baseline_input_sha256"]:
                    raise ValueError("Response study inputs differ from baseline")
                if digest(args.geometry)!=response_study["geometry_sha256"]:
                    raise ValueError("Response study geometry differs from baseline")
                for name,expected_hash in response_study["baseline_factor_manifest_sha256"].items():
                    if digest(args.factors/name/"factor_manifest.json")!=expected_hash:
                        raise ValueError("Response study Factors differ from baseline")
        except Exception as error:
            validation_error=f"{type(error).__name__}: {error}"
        print("ELLIPSE_STAGE factors_validated",flush=True)
    shared_error=[validation_error]
    if rank==0: print("ELLIPSE_STAGE broadcast_validation",flush=True)
    dist.broadcast_object_list(shared_error,src=0)
    if shared_error[0]:
        raise ValueError(f"Factors validation failed: {shared_error[0]}")
    if rank==0: print("ELLIPSE_STAGE geometry",flush=True)
    geometry=ActiveGeometry.from_npz(args.geometry,device)
    if rank==0: print("ELLIPSE_STAGE geometry_loaded",flush=True)
    if geometry.full_count!=132040 or geometry.active_count!=82040 or geometry.views!=20:
        raise ValueError("Ellipse geometry mismatch")
    total_bins=cfg["detector_count"]
    begin,end=split_bins(total_bins,rank,world)
    factor_paths={key:args.factors/folder for key,folder in NAMES.items()}
    for path in factor_paths.values():
        if not path.is_dir(): raise FileNotFoundError(path)
    data=args.data_root
    projections={}
    count_totals={}
    for energy in (218,440):
        path=data/"CntStat"/f"{energy}keV_RotateNum20_Geant4JSCC"/f"CntStat_{args.dataset}_{args.level}.csv"
        counts=np.loadtxt(path,delimiter=",",dtype=np.float32)
        if counts.shape!=(20,total_bins) or not np.isfinite(counts).all() or np.any(counts<0):
            raise ValueError(f"CntStat shape or values invalid: {path}")
        count_totals[energy]=float(counts.sum(dtype=np.float64))
        projections[energy]=torch.from_numpy(np.ascontiguousarray(counts[:,begin:end].T)).to(device)
    list_dir=data/"List"/"218-440keV_RotateNum20_Geant4JSCC"/f"List_{args.dataset}_{args.level}"
    for view in range(1,21):
        if not (list_dir/f"{view}.csv").is_file(): raise FileNotFoundError(list_dir/f"{view}.csv")
    sensi_path=(args.compton_sensitivity if args.compton_sensitivity is not None
                else factor_paths["A440"]/"Sensi_d")
    provenance_path=sensi_path.parent/"Sensi_d_provenance.json"
    if not sensi_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Validated new-distance Compton sensitivity required")
    provenance=json.loads(provenance_path.read_text())
    if study is not None and digest(sensi_path)!=study["baseline_sensi_d_sha256"]:
        raise ValueError("Ablation Compton sensitivity differs from baseline")
    if (provenance["experiment"]!=cfg["experiment_id"] or
        provenance["pixel_count"]!=geometry.full_count or
        provenance["resolution_fwhm"]!=.13 or provenance["reference_keV"]!=511 or
        provenance["sum_threshold_MeV"]!=.350 or
        not provenance["input_already_smeared"]):
        raise ValueError("Compton sensitivity provenance mismatch")
    if response_study is not None:
        expected_sensi=(response_study["baseline_sensi_d_sha256"] if args.baseline_regression
                        else response_study["sensi_d_sha256"])
        if digest(sensi_path)!=expected_sensi:
            raise ValueError("Response study sensitivity SHA256 mismatch")
        if not args.baseline_regression and provenance.get("max_min_standardized_arm")!=3.0:
            raise ValueError("Sensitivity does not match the 3-sigma event filter")
    if args.dry_run:
        if rank==0: print("ELLIPSE_RECON_PREFLIGHT_OK",flush=True)
        dist.destroy_process_group()
        return
    output_error=None
    if rank==0:
        try:
            args.output.mkdir(parents=True,exist_ok=False)
        except Exception as error:
            output_error=f"{type(error).__name__}: {error}"
    shared_error=[output_error]
    dist.broadcast_object_list(shared_error,src=0)
    if shared_error[0]:
        raise ValueError(f"Output creation failed: {shared_error[0]}")
    update_rule=None
    if study is not None:
        from regularized_update import AblationUpdate
        update_rule=AblationUpdate(study,spatial_path,variant,device,args.output)
    raw440=load_matrix(args.factors,NAMES["A440"],geometry.full_count,total_bins)
    if rank==0: print("ELLIPSE_STAGE loading_440_detector_shard",flush=True)
    response440=ViewResponse(local_rows(raw440,begin,end,device),geometry,args.cache_views)
    if rank==0: print("ELLIPSE_STAGE 440_detector_shard_ready",flush=True)
    sensi440=response440.sensitivity()
    dist.all_reduce(sensi440,op=dist.ReduceOp.SUM)
    if update_rule:
        update_rule.configure("440_single",sensi440,count_totals[440])
    if args.channels=="six":
        raw218=load_matrix(args.factors,NAMES["A218"],geometry.full_count,total_bins)
        rawcross=load_matrix(args.factors,NAMES["C440to218"],geometry.full_count,total_bins)
        image440,h440=single_mlem(response440,projections[440],sensi440,
                              args.iterations,args.save_step,save_history=rank==0,
                              progress_label="440_single",update_rule=update_rule)
        if rank==0: print("ELLIPSE_STAGE loading_cross_and_218_shards",flush=True)
        cross=ViewResponse(local_rows(rawcross,begin,end,device),geometry,args.cache_views)
        predicted=forward_project(cross,image440)
        response218=ViewResponse(local_rows(raw218,begin,end,device),geometry,args.cache_views)
        sensi218=response218.sensitivity()
        dist.all_reduce(sensi218,op=dist.ReduceOp.SUM)
        if update_rule:
            update_rule.configure("218_corrected",sensi218,count_totals[218])
        image218,h218=single_mlem(response218,projections[218],sensi218,
                                  args.iterations,args.save_step,predicted,rank==0,
                                  progress_label="218_corrected",update_rule=update_rule)
        if rank==0:
            predicted_parts=[torch.empty_like(predicted) for _ in range(world)]
        else:
            predicted_parts=None
        # Unequal detector shards: gather padded arrays on all ranks.
        max_rows=(total_bins+world-1)//world
        padded=torch.zeros((max_rows,20),device=device)
        padded[:end-begin]=predicted
        gathered=[torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(gathered,padded)
        if rank==0:
            np.concatenate([part[:split_bins(total_bins,k,world)[1]-split_bins(total_bins,k,world)[0]].cpu().numpy()
                            for k,part in enumerate(gathered)]).astype("<f4").tofile(
                            args.output/"PredictedCntStat_218_From440.float32")
        del response218,cross,raw218,rawcross
        torch.cuda.empty_cache() if device.type=="cuda" else None
    detector=torch.from_numpy(load_detector_coordinates(factor_paths["A440"]/"Detector.csv",
                                                   expected_count=total_bins)).to(device)
    coordinates=torch.from_numpy(np.loadtxt(factor_paths["A440"]/"coor_polar_full.csv",
                                            delimiter=",",dtype=np.float32))
    projector=build_compton_sparse_projector(coordinates,theta_stride=1,z_stride=1,
                                              rotate_num=20,dtype=torch.float32).to(device)
    if rank==0: print("ELLIPSE_STAGE loading_full_440_compton_response",flush=True)
    sysfull=full_rows(raw440,device)
    if rank==0: print("ELLIPSE_STAGE preparing_compton_event_blocks",flush=True)
    resolution=.13*(511/440)**.5
    threshold_max=2*.440**2/(.511+2*.440)-.001
    blocks=[]
    accepted=0
    byte_ranges=[]
    for view in range(20):
        events,byte_range=read_event_partition(list_dir/f"{view+1}.csv",rank,world)
        byte_ranges.append(byte_range)
        packed=[]
        for part in torch.chunk(events,args.event_chunks,dim=0):
            if not part.numel(): continue
            result,_,_=get_compton_backproj_list_single_sparse(
                sysfull,detector,projector,part.to(device),0.0,0.0,
                .440,resolution,threshold_max,.05,.350,device,
                input_energies_already_smeared=True,
                max_min_standardized_arm=max_min_arm)
            if result.numel(): packed.append(result)
        view_blocks=[]
        if packed:
            accepted+=sum(item.size(0) for item in packed)
            for group in packed:
                for start in range(0,group.size(0),args.event_block):
                    fine,_=materialize_sparse_event_rows_to_fine(
                        group[start:start+args.event_block].to(device),sysfull,projector)
                    compact=geometry.compact(fine,view).cpu()
                    view_blocks.append(compact)
        blocks.append(view_blocks)
        print(f"[rank {rank}] view {view+1} accepted={sum(item.size(0) for item in view_blocks)}",flush=True)
    del sysfull,projector,detector,raw440
    torch.cuda.empty_cache() if device.type=="cuda" else None
    accepted_tensor=torch.tensor([accepted],dtype=torch.int64,device=device)
    dist.all_reduce(accepted_tensor,op=dist.ReduceOp.SUM)
    accepted_per_view=torch.tensor([sum(item.size(0) for item in view) for view in blocks],
                                  dtype=torch.int64,device=device)
    dist.all_reduce(accepted_per_view,op=dist.ReduceOp.SUM)
    if int(accepted_tensor.item())<=0:
        raise ValueError("No accepted 440-keV Compton events")
    if study is not None and int(accepted_tensor.item())!=study["baseline_accepted_compton_events"]:
        raise ValueError("Ablation accepted events differ from frozen baseline")
    if response_study is not None:
        expected_events=(response_study.get("baseline_accepted_compton_events",484936) if args.baseline_regression
                         else response_study["kept_compton_events"])
        if int(accepted_tensor.item())!=expected_events:
            raise ValueError("Response-filter accepted event closure failed")
    if rank==0: print(f"ELLIPSE_STAGE compton_events_ready accepted={int(accepted_tensor.item())}",flush=True)
    full_sensi=np.fromfile(sensi_path,dtype="<f4")
    if len(full_sensi)!=geometry.full_count or not np.isfinite(full_sensi).all():
        raise ValueError("Invalid full Compton sensitivity")
    sensid=geometry.compton_sensitivity(torch.from_numpy(full_sensi))
    if update_rule:
        update_rule.configure("440_compton",sensid,int(accepted_tensor.item()))
        update_rule.configure("440_jscc",sensi440+sensid,count_totals[440]+int(accepted_tensor.item()))
    checkpoint_callback=None
    if response_study is not None and not (args.baseline_regression or args.pilot_only) and rank==0:
        def checkpoint_callback(iteration, history_d, history_j):
            if first_scatter_study:
                folder=args.output/f"checkpoint_{iteration:06d}"
                folder.mkdir(exist_ok=False)
                outputs={}
                for channel,history in (("440_ComptonOnly",history_d),("440_SinglePlusCompton",history_j)):
                    frame=history[-1]
                    if not torch.isfinite(frame).all() or bool((frame<0).any()):
                        raise ValueError("Nonfinite or negative persistent checkpoint")
                    collect_response(folder,channel,frame,None,geometry,0)
                    outputs[channel]={kind:digest(folder/f"Image_{channel}_{kind}.float32")
                                      for kind in ("active","full")}
                (folder/"checkpoint_manifest.json").write_text(json.dumps({
                    "study":response_study["study"],"group":response_study["group"],
                    "iteration":iteration,"outputs":outputs,
                    "config_sha256":digest(args.response_filter_config)},indent=2)+"\n")
                return
            if iteration!=2000:
                return
            folder=args.output/"checkpoint_2000"
            folder.mkdir(exist_ok=False)
            outputs=[]
            for channel,history in (("440_ComptonOnly",history_d),("440_SinglePlusCompton",history_j)):
                collect_response(folder,channel,history[-1],torch.stack(history),geometry,0)
                paths={kind:folder/f"Image_{channel}_{kind}.float32" for kind in ("active","full","history")}
                outputs.append({"channel":channel,"frames":len(history),
                                "sha256":{kind:digest(path) for kind,path in paths.items()}})
            checkpoint={"study":response_study["study"],"mode":"interim","iteration":iteration,
                "save_step":args.save_step,"input_sha256":input_hashes,"geometry_sha256":digest(args.geometry),
                "sensi_d_sha256":digest(sensi_path),"config_sha256":digest(args.response_filter_config),
                "accepted_events":int(accepted_tensor.item()),"outputs":outputs,
                "warning":"Read-only interim observation, not formal acceptance; continue to 10000 with fixed threshold"}
            # Write the ready marker last, after all immutable snapshot files exist.
            (folder/"checkpoint_manifest.json").write_text(json.dumps(checkpoint,indent=2)+"\n")
            print("RESPONSE_CHECKPOINT_READY 2000",flush=True)
    (image_d,hd),(image_j,hj)=compton_and_joint_mlem(
        response440,projections[440],blocks,sensi440,sensid,
        args.iterations,args.save_step,save_history=rank==0,
        progress_label="440_compton_and_jscc",update_rule=update_rule,checkpoint_callback=checkpoint_callback)
    collect_response(args.output,"440_ComptonOnly",image_d,hd,geometry,rank)
    collect_response(args.output,"440_SinglePlusCompton",image_j,hj,geometry,rank)
    if args.channels=="six":
        collect_response(args.output,"440_SinglePhoton",image440,h440,geometry,rank)
        collect_response(args.output,"218_SinglePhoton_CrossTalkCorrected",image218,h218,geometry,rank)
        collect_response(args.output,"440SinglePlus218Single",image440+image218,
                         h440+h218 if rank==0 else None,geometry,rank)
        collect_response(args.output,"440SingleComptonPlus218Single",image_j+image218,
                         hj+h218 if rank==0 else None,geometry,rank)
    if update_rule:
        update_rule.save()
    resource={"rank":rank,"device":str(device),"accepted_events":accepted,
              "event_bytes_cpu":sum(item.numel()*item.element_size() for view in blocks for item in view),
              "host_peak_rss_bytes":(int(resource_usage.getrusage(resource_usage.RUSAGE_SELF).ru_maxrss)*1024
                                     if resource_usage is not None else 0),
              "node":os.environ.get("SLURMD_NODENAME", "unknown"),
              "host_allocated_bytes":int(os.environ.get("ABLATION_HOST_ALLOCATED_BYTES",
                   str(int(os.environ.get("SLURM_MEM_PER_NODE", "0"))*1024**2))),
              "elapsed_seconds":time.monotonic()-started}
    if device.type=="cuda":
        resource.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                        total_device_bytes=torch.cuda.get_device_properties(device).total_memory)
    resources=[None]*world
    dist.all_gather_object(resources,resource)
    if rank==0:
        (args.output/"run_manifest.json").write_text(json.dumps({
          "experiment":cfg["experiment_id"],"dataset":args.dataset,"count_level":args.level,
          "algorithm":("JSCC joint MLEM; active ellipse density; fixed additive 440-to-218" if study is None else
                       f"JSCC {variant['method']}; global descent-checked update; active ellipse density; fixed additive 440-to-218"),
          "gamma_channels_not_parent_ac225_activity":True,
          "pixels_full":geometry.full_count,"pixels_active":geometry.active_count,
          "iterations":args.iterations,"save_step":args.save_step,"world_size":world,
          "pilot_only":args.pilot_only,
          "study_short":args.study_short,
          "channels":args.channels,
          "response_mismatch":None if response_study is None else {
              "study":response_study["study"],"filter_enabled":not args.baseline_regression,
              "max_min_standardized_arm":max_min_arm,
              "config_sha256":digest(args.response_filter_config),
              "scan_manifest_sha256":response_study["scan_manifest_sha256"],
              "removed_compton_events":0 if args.baseline_regression else response_study["removed_compton_events"],
              "baseline_accepted_compton_events":response_study.get("baseline_accepted_compton_events",484936),
              "baseline_result":response_study.get("baseline_result","NEMA_Body_H60_5e9_1644876"),
              "event_policy":response_study.get("group","legacy"),
              "validation_gate_sha256":response_study.get("validation_gate_sha256")},
          "code_sha256":{name:digest(HERE/name if (HERE/name).is_file() else ROOT/name) for name in
                         ("run_reconstruction.py","torch_active_operator.py",
                          "compton_sparse_ops.py","process_list_plane_sparse.py","compton_event_response.py")},
          "spike_ablation":None if study is None else {"study":study["study"],"variant":variant,
             "study_sha256":digest(args.study_json),"spatial_model_sha256":digest(spatial_path),
             "baseline":study["baseline"],"independent_parameters":(int(update_rule.group_count)
               if variant["method"]=="binding" else geometry.active_count),
             "code_sha256":{name:digest(HERE/name if (HERE/name).is_file() else ROOT/name) for name in
                            ("run_reconstruction.py","torch_active_operator.py","regularized_update.py",
                             "compton_sparse_ops.py","process_list_plane_sparse.py",
                             "compton_event_response.py","detector_csv.py")}},
          "cuda_allocator_config":os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
          "accepted_compton_events":int(accepted_tensor.item()),
          "accepted_compton_events_per_view":accepted_per_view.cpu().tolist(),
          "energy_resolution_fwhm_at_511keV":.13,"compton_sum_threshold_MeV":.350,
          "list_energies_already_smeared":True,"geometry_sha256":digest(args.geometry),
          "sensi_d_sha256":digest(sensi_path),"input_sha256":input_hashes,
          "factor_manifest_sha256":{name:digest(args.factors/name/"factor_manifest.json")
                                    for name in NAMES.values()},
          "resources":resources},indent=2)+"\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__=="__main__":
    with torch.no_grad(): main()
