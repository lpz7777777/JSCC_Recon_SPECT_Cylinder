"""Original sequential density MLEM for separately registered matrix-Poisson data."""
import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from ehe_common import *
from reconstruction_output_policy import require_separate_channels, POLICY_ID as OUTPUT_POLICY
from torch_active_operator import ActiveGeometry,ViewResponse,forward_project,single_mlem
from single_checkpoint_mlem import single_mlem_checkpointed
from ehe_forward_poisson_data import verify_counts,save_npy

def reconstruct(a):
    policy(a.mode,a.iterations,10);r=a.release.resolve();root=a.responses.resolve();out=a.output.resolve()
    freeze=read(r/'release_manifest.json');verify_files(r,freeze['sha256'])
    collection=verify_counts(a.counts,r)
    config=read(r/'config.json')
    channels=require_separate_channels(config)
    if a.mode=='formal':
        authority=read(a.authority)
        if not authority['passed'] or authority['mode']!='validation' or authority['release_key']!=freeze['release_key']:
            raise ValueError('Actual validation authority differs')
        if authority['counts_sha256']!=digest(a.counts/'collection.json') or authority['factor_sha256']!=config['factor_sha256']:
            raise ValueError('Validation input identity differs')
        if not authority['forward_mean_closure']['passed'] or not authority['operator_closure']['passed']:
            raise ValueError('Complete-input validation closure required')
    out.mkdir(parents=True,exist_ok=False);alloc=allocation();write(out/'allocation.json',alloc)
    geometry=ActiveGeometry.from_npz(r/'whole_geometry.npz','cuda');g=np.load(r/'whole_geometry.npz');active=g['active_indices']
    if len(active)!=78920:raise ValueError('Whole active basis differs')
    resources_seen=[];tests={};times={};started=time.time()
    def response(name):
        f=read(root/name/'factor_manifest.json');verify_files(root/name,f['files'])
        if digest(root/name/'factor_manifest.json')!=config['factor_sha256'][name]:raise ValueError('Registered synthetic factor identity differs')
        raw=np.memmap(root/name/'SysMat_polar','<f4',mode='r',shape=(132040,2312))
        return ViewResponse(torch.as_tensor(np.array(raw.T),device='cuda'),geometry,cache='none')
    def checkpoint(channel):
        def save(iteration,history):
            values=history[-1].numpy().reshape(-1).astype('<f4');full=np.zeros(132040,'<f4');full[active]=values
            folder=out/'checkpoints'/channel/f'{iteration:04d}';folder.mkdir(parents=True,exist_ok=False)
            array_write(folder/'active.float32',values);array_write(folder/'full.float32',full)
            write(folder/'manifest.json',dict(iteration=iteration,channel=channel,release_key=freeze['release_key'],
                    files={n:digest(folder/n) for n in ('active.float32','full.float32')}))
            usage=resources(alloc);resources_seen.append(usage)
            write(out/'progress.json',dict(mode=a.mode,channel=channel,iteration=iteration,target=a.iterations,elapsed_seconds=time.time()-started,resource=usage))
        return save
    y={e:torch.as_tensor(np.load(a.counts/f'projection_{e}.npy'),dtype=torch.float32,device='cuda') for e in (218,440)}
    for e,name,channel in ((440,'A440',CHANNELS[0]),(218,'A218',CHANNELS[1])):
        model=response(name);s=model.sensitivity()
        if not bool(torch.isfinite(s).all()) or bool((s<=1e-12).any()):raise ValueError('HOLD sensitivity below original update floor')
        bg=None if e==440 else torch.as_tensor(np.load(out/'fixed_cross_background.npy'),device='cuda')
        if a.mode=='validation':
            # The actual complete matrix is exercised in both directions, not a 32-event substitute.
            v=model.matrix(0);x=torch.linspace(.2,1.2,78920,device='cuda')[:,None];z=torch.linspace(.1,.9,2312,device='cuda')[:,None]
            lhs=torch.sum((v@x)*z).double();rhs=torch.sum(x*(v.T@z)).double();adj=float(abs(lhs-rhs)/max(float(abs(lhs)),1e-30))
            original,original_h=single_mlem(model,y[e],s,10,10,additive_background=bg)
            tests[channel]={'adjoint_relative_error':adj}
        began=time.monotonic()
        final,h=single_mlem_checkpointed(model,y[e],s,a.iterations,10,additive_background=bg,
            checkpoint_callback=checkpoint(channel),progress_label=channel,phase_limit_seconds=a.limit)
        torch.cuda.synchronize();times[channel]=time.monotonic()-began
        hist=h.numpy().reshape(a.iterations//10,78920).astype('<f4');array_write(out/f'Image_{channel}_history.float32',hist)
        array_write(out/f'Image_{channel}_final.float32',final.cpu().numpy().astype('<f4'))
        if a.mode=='validation':
            tests[channel].update(l2=float(torch.linalg.vector_norm(final-original)/torch.linalg.vector_norm(original).clamp_min(1e-30)),
                                 history_equal=bool(torch.equal(h,original_h)))
            if tests[channel]['l2']>1e-5 or tests[channel]['adjoint_relative_error']>1e-5 or not tests[channel]['history_equal']:raise ValueError('Original actual-input MLEM regression failed')
            del original,original_h,v,x,z
        if e==440:
            cross=response('C440to218');b=forward_project(cross,final).cpu().numpy()
            save_npy(out/'fixed_cross_background.npy',b);del cross,b
        del model,s,final,h;gc.collect();torch.cuda.empty_cache()
    record=dict(mode=a.mode,iterations=a.iterations,save_step=10,channels=channels,output_policy=OUTPUT_POLICY,release_key=freeze['release_key'],
        counts_sha256=digest(a.counts/'collection.json'),factor_sha256={n:digest(root/n/'factor_manifest.json') for n in RESPONSES},
        data_kind=collection['data_kind'],physical_calibration_claim=False,geometry_sha256=digest(r/'whole_geometry.npz'),tests=tests,
        phase_seconds=times,resources=resources_seen,allocation=alloc,started_epoch=started,finished_epoch=time.time(),
        background_source='this EHE final440 single image; fixed additive Poisson term',baseline_helper_sha256={n:digest(r/n) for n in ('torch_active_operator.py','single_checkpoint_mlem.py')})
    write(out/'run_manifest.json',record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--responses',type=Path,required=True);p.add_argument('--counts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['validation','formal'],required=True);p.add_argument('--iterations',type=int,required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--authority',type=Path)
    a=p.parse_args()
    try:reconstruct(a)
    except BaseException as e:
        if a.output.is_dir():write(a.output/'failure.json',dict(passed=False,error=str(e)))
        raise
