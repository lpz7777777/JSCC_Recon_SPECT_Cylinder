"""Four doses of matched-native EHE forward+Poisson and original MLEM200."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from ehe_common import read,write,digest,verify_files,hashes,array_write,allocated_bytes
from torch_active_operator import ActiveGeometry,forward_project
from single_checkpoint_mlem import single_mlem_checkpointed
from ehe_dense_rerun_runtime_20261102 import load_response,resource


def save_npy(path,a):
    path=Path(path);temp=path.with_name(path.name+'.writing')
    with temp.open('wb') as f:np.save(f,a);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def run(release,output,index):
    freeze=read(release/'release_manifest.json');verify_files(release,freeze['sha256'])
    cfg=read(release/'contract.json');spec=cfg['conditions'][str(index)]
    output.mkdir(parents=True,exist_ok=False);began=time.time()
    identity=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']
    allocation=subprocess.check_output(['scontrol','show','job','-o',identity],text=True)
    allocated=allocated_bytes(allocation);write(output/'allocation.json',dict(identity=identity,scontrol=allocation,allocated_bytes=allocated))
    torch.cuda.set_device(0);torch.set_grad_enabled(False);torch.set_num_threads(8)
    device=torch.device('cuda',0);g=np.load(release/'whole_geometry.npz');active=g['active_indices']
    geometry=ActiveGeometry.from_npz(release/'whole_geometry.npz',device)
    source=np.load(release/'source_native.npz');source_info=read(release/'source_native_manifest.json')
    original=source_info['original_integral_mm3'];native=source_info['native_integral_mm3']
    mix={e:original[e]*cfg['gamma_yields'][e] for e in original};denom=sum(mix.values())
    N={e:spec['expected_emitted_photons']*mix[e]/denom for e in original}
    density={e:N[e]/native[e] for e in original}
    times={};resources=[];checks={};summary={};means={};samples={}
    def true_image(e):return torch.tensor(source[f'activity_{e}_full'][active]*density[str(e)],dtype=torch.float32,device=device)[:,None]
    def sample(name,mean):
        assert mean.shape==(2312,20) and np.isfinite(mean).all() and (mean>=0).all()
        values=np.random.Generator(np.random.PCG64(spec['noise_seeds'][name])).poisson(mean)
        save_npy(output/f'mean_{name}.npy',mean);save_npy(output/f'sampled_{name}.npy',values)
        means[name]=mean;samples[name]=values
        summary[name]=dict(expected_counts=float(mean.sum()),sampled_counts=int(values.sum()),
            total_poisson_z=float((values.sum()-mean.sum())/np.sqrt(mean.sum())))
        return values
    def solve(e,model,s,y,bg=None):
        channel=f'{e}_SinglePhoton'+('_CrossTalkCorrected' if e==218 else '')
        def checkpoint(iteration,history):
            a=history[-1].numpy().reshape(-1).astype('<f4');full=np.zeros(132040,'<f4');full[active]=a
            if not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid scientific snapshot')
            final=output/'checkpoints'/channel/f'checkpoint_{iteration:06d}';temp=final.with_name('.'+final.name+'.partial')
            temp.mkdir(parents=True,exist_ok=False)
            array_write(temp/'active.float32',a);array_write(temp/'full.float32',full)
            write(temp/'manifest.json',dict(iteration=iteration,channel=channel,release_sha256=digest(release/'release_manifest.json'),files=hashes(temp)))
            temp.rename(final);fd=os.open(final.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            usage=resource(allocated,device);resources.append(usage)
            write(output/'progress.json',dict(channel=channel,iteration=iteration,target=200,resource=usage,updated_epoch=time.time()))
        t=time.monotonic()
        final,h=single_mlem_checkpointed(model,torch.tensor(y,dtype=torch.float32,device=device),s,200,1,
            additive_background=bg,checkpoint_callback=checkpoint,progress_label=channel,phase_limit_seconds=1800)
        times[channel]=time.monotonic()-t
        array_write(output/f'Image_{channel}_history.float32',h.numpy().reshape(200,78920).astype('<f4'))
        a=final.cpu().numpy().reshape(-1).astype('<f4');full=np.zeros(132040,'<f4');full[active]=a
        array_write(output/f'Image_{channel}_active.float32',a);array_write(output/f'Image_{channel}_full.float32',full)
        array_write(output/f'S_{channel}.float32',s.cpu().numpy().astype('<f4'))
        prediction=forward_project(model,final)
        if bg is not None:prediction+=bg
        save_npy(output/f'final_prediction_{e}.npy',prediction.cpu().numpy())
        checks[channel]['final_predicted_counts']=float(prediction.sum())
        return final
    # Generate and solve440 first; then generateC and freeze background from this final440.
    model,s,error=load_response(cfg['factors']['A440'],geometry,device)
    checks['440_SinglePhoton']=dict(matched_s_relative_L2=error)
    mean=forward_project(model,true_image(440)).double().cpu().numpy();y=sample('A440',mean);save_npy(output/'projection_440.npy',y)
    final440=solve(440,model,s,y);del model,s;gc.collect();torch.cuda.empty_cache()
    cross,sc,error=load_response(cfg['factors']['C440to218'],geometry,device,require_positive=False)
    sample('C440to218',forward_project(cross,true_image(440)).double().cpu().numpy())
    background=forward_project(cross,final440).detach();save_npy(output/'fixed_cross_background.npy',background.cpu().numpy())
    checks['fixed_background']=dict(source='this dose own final440200 image',cross_s_relative_L2=error,
        generated_true_C_not_used_as_background=True)
    del cross,sc,final440;gc.collect();torch.cuda.empty_cache()
    model,s,error=load_response(cfg['factors']['A218'],geometry,device)
    checks['218_SinglePhoton_CrossTalkCorrected']=dict(matched_s_relative_L2=error)
    y=sample('A218',forward_project(model,true_image(218)).double().cpu().numpy())+samples['C440to218'];save_npy(output/'projection_218.npy',y)
    final218=solve(218,model,s,y,background);del model,s,final218,background;gc.collect();torch.cuda.empty_cache()
    collection=dict(passed=True,study=cfg['study'],data_kind='matched_native_matrix_forward_plus_independent_Poisson',
        expected_emitted_photons=spec['expected_emitted_photons'],expected_primary_photons=N,normalization_density=density,
        noise_seeds=spec['noise_seeds'],bit_generator='PCG64',numpy_version=np.__version__,components=summary,
        window_counts={str(e):int(np.load(output/f'projection_{e}.npy').sum()) for e in [218,440]},
        source_native_sha256=digest(release/'source_native.npz'),truth_3mm_sha256=source_info['truth_3mm_sha256'],
        physical_calibration_passed=False,transport_performed=False,cross_energy_image_sum=False,
        basis='Same native finite-volume source and reconstruction basis; isolates Poisson noise within the frozen model',
        files={p.name:digest(p) for p in output.glob('*.npy')})
    write(output/'collection.json',collection)
    write(output/'run_manifest.json',dict(passed=True,condition=spec,channels=['440_SinglePhoton','218_SinglePhoton_CrossTalkCorrected'],
        iterations=200,save_step=1,initial_density=1,regularization='none',study=cfg['study'],
        data_kind=collection['data_kind'],collection_sha256=digest(output/'collection.json'),
        release_sha256=digest(release/'release_manifest.json'),numerical_checks=checks,resources=resources,
        job=identity,phase_solve_seconds=times,started_epoch=began,completed_epoch=time.time(),
        full_input_prevalidation_passed=False,separate_validation_job=False,physical_calibration_passed=False,
        matched_model_test_is_not_independent_Geant4_calibration=True))
    write(output/'completion.json',dict(passed=True,job=identity,condition_index=index,files=hashes(output)))
    print('LOW_NOISE_FORMAL_COMPLETE',index,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    try:run(a.release,a.output,a.index)
    except BaseException as exc:
        if a.output.is_dir():write(a.output/'failure.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
