"""Apply unchanged legacy stable-float64 full-circle q<=3 to fresh events."""
import argparse, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from jscc_5e10_common import *
from jscc_5e10_runtime import setup,resource_record,gather_resources
from jscc_5e10_contract import verify_topology
from run_reconstruction import read_event_partition,load_matrix,full_rows
from run_energy_preflight_v5 import settings,validate_whole_geometry
from prepare_energy_5e9_v5 import subset
from compton_event_response import prepare_compton_events,build_compton_cone_weights,min_standardized_compton_arm,build_detector_position_variance
from detector_csv import load_detector_coordinates

def save(path,array):
    with Path(path).open('wb') as f:np.save(f,array);f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser()
    for n in ('release','input','factors','output','allocation'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();began=time.monotonic();cfg=read(a.release/'kernel_config.json');verify_files(a.release,cfg['files'])
    coll=validate_collection(read(a.input/'collection.json'))
    if digest(a.input/'collection.json')!=cfg['transport_collection_sha256']:raise ValueError('Fresh transport collection differs')
    rank,world,local,device=setup()
    if rank==0:
        verify_files(a.input,coll['files']);verify_files(a.factors,cfg['factor_payload_sha256'])
        a.output.mkdir(exist_ok=False);(a.output/'parts').mkdir();(a.output/'selections').mkdir();(a.output/'selected_rows').mkdir()
    dist.barrier()
    geo=validate_whole_geometry(a.release/'whole_geometry.npz',cfg['whole_geometry_sha256'])
    coords=torch.tensor(geo['coordinates_mm'],dtype=torch.float32,device=device)
    folder=a.factors/'440keV_RotateNum20'
    detector=torch.tensor(load_detector_coordinates(folder/'Detector.csv',10496),dtype=torch.float32,device=device)
    variance=build_detector_position_variance(detector,0)
    B=full_rows(load_matrix(a.factors,'440keV_RotateNum20',132040,10496),device)
    for v in range(1,21):
        raw,bytepart=read_event_partition(a.input/'List'/f'{v}.csv',rank,world)
        counts=[None]*world;dist.all_gather_object(counts,len(raw));offset=sum(counts[:rank])
        prepared,diag=prepare_compton_events(raw.to(device),settings(),detector,variance,variance,input_energies_already_smeared=True)
        kept=[];removed=0;original=0;max_difference=0.
        if prepared is not None:
            for start in range(0,prepared.count,32):
                e=subset(prepared,slice(start,start+32))
                w=build_compton_cone_weights(e,coords,settings())*B[e.cpnum1-1];s=w.sum(1)
                valid=torch.isfinite(w).all(1)&torch.isfinite(s)&(s>0)
                diag.invalid_kernel_events+=int((~valid).sum());e=subset(e,valid);del w,s
                if not e.count:continue
                original+=e.count;qscore=min_standardized_compton_arm(e,coords,settings())
                if not bool(torch.isfinite(qscore).all()):raise ValueError('Nonfinite q')
                if start==0:
                    individual=torch.cat([min_standardized_compton_arm(subset(e,slice(i,i+1)),coords,settings()) for i in range(e.count)])
                    max_difference=float((individual-qscore).abs().max())
                    if max_difference>1e-10 or not torch.equal(individual<=3,qscore<=3):raise ValueError('Frozen q changes with partition')
                mask=qscore<=3;kept.extend((e.source_row_indices[mask].cpu().numpy()+offset).tolist());removed+=int((~mask).sum())
                if torch.cuda.max_memory_reserved(device)>.8*torch.cuda.get_device_properties(device).total_memory:raise MemoryError('Selection GPU margin fails')
        name=f'{v:02d}_rank{rank:02d}';save(a.output/'parts'/(name+'.npy'),np.asarray(kept,dtype='<i8'))
        save(a.output/'parts'/(name+'_rows.npy'),raw[np.asarray(kept,dtype=np.int64)-offset].numpy())
        write(a.output/'parts'/(name+'.json'),dict(rank=rank,view=v,raw_rows=len(raw),raw_global_offset=offset,
            byte_partition=bytepart,original_accepted=original,kept=len(kept),removed=removed,q_chunk_max_absolute_difference=max_difference,
            diagnostics=diag.to_dict(),selection_sha256=digest(a.output/'parts'/(name+'.npy')),
            selected_rows_sha256=digest(a.output/'parts'/(name+'_rows.npy'))))
        print('JSCC_FRESH_SELECTION',rank,v,original,len(kept),flush=True)
        del prepared,raw
    resources=gather_resources(resource_record(rank,local,device,began));verify_topology(resources,a.allocation)
    dist.barrier()
    if rank==0:
        per_view=[];original_total=removed_total=0;parts={}
        for v in range(1,21):
            values=[];row_values=[];cursor=0
            for ranknum in range(world):
                name=f'{v:02d}_rank{ranknum:02d}';r=read(a.output/'parts'/(name+'.json'))
                if digest(a.output/'parts'/(name+'.npy'))!=r['selection_sha256'] or digest(a.output/'parts'/(name+'_rows.npy'))!=r['selected_rows_sha256']:
                    raise ValueError('Partition arrays differ from their executed receipts')
                indices=np.load(a.output/'parts'/(name+'.npy'))
                if r['raw_global_offset']!=cursor or len(indices)!=r['kept'] or r['kept']+r['removed']!=r['original_accepted']:
                    raise ValueError('Raw/accepted event rank partition does not close')
                cursor+=r['raw_rows'];original_total+=r['original_accepted'];removed_total+=r['removed'];values.append(indices)
                row_values.append(np.load(a.output/'parts'/(name+'_rows.npy')))
                parts[name]=digest(a.output/'parts'/(name+'.json'))
            selected=np.concatenate(values)
            if np.any(selected[1:]<=selected[:-1]):raise ValueError('Event selections overlap/out of order')
            save(a.output/'selections'/f'{v}.npy',selected);save(a.output/'selected_rows'/f'{v}.npy',np.concatenate(row_values));per_view.append(len(selected))
        if sum(read(a.output/'parts'/f'{v:02d}_rank{r:02d}.json')['raw_rows'] for v in range(1,21) for r in range(32))!=coll['raw_list_rows']:
            raise ValueError('Raw event partitions do not close to the actual collected list')
        files=hashes(a.output/'selections')
        write(a.output/'selection_manifest.json',dict(passed=True,study=STUDY,events_per_view=per_view,accepted_events=sum(per_view),
            original_accepted=original_total,removed=removed_total,selection_files=files,
            selected_rows_files=hashes(a.output/'selected_rows'),part_receipt_sha256=parts,
            selection_policy='unchanged legacy positive support + full-circle stable_float64 q <= 3',resources=resources,
            input_collection_sha256=cfg['transport_collection_sha256'],kernel_config_sha256=digest(a.release/'kernel_config.json'),
            actual_primary_photons=TOTAL,source_or_truth_used_in_selection=False))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
