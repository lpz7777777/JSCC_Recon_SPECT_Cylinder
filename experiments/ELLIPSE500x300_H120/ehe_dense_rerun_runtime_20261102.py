"""Formal EHE reruns: new dense A218/A440, original unregularized MLEM."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from ehe_common import read, write, digest, verify_files, hashes, array_write, allocated_bytes
from torch_active_operator import ActiveGeometry, ViewResponse, forward_project
from single_checkpoint_mlem import single_mlem_checkpointed


def resource(allocated, device):
    import resource as posix_resource
    total = torch.cuda.get_device_properties(device).total_memory
    result = dict(host_peak_rss_bytes=posix_resource.getrusage(posix_resource.RUSAGE_SELF).ru_maxrss * 1024,
                  host_allocated_bytes=allocated, gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  gpu_total_bytes=total)
    if result['host_peak_rss_bytes'] > .8 * allocated or result['gpu_peak_reserved_bytes'] > .8 * total:
        raise MemoryError('Original 80% host/GPU resource margin failed: ' + str(result))
    return result


def load_response(spec, geometry, device, require_positive=True):
    root = Path(spec['root'])
    if digest(root/'factor_manifest.json') != spec['manifest_sha256']:
        raise ValueError('Dense/retained cross matrix manifest differs')
    verify_files(root, spec['files'])
    if digest(root/'whole_geometry.npz') != spec['geometry_sha256']:
        raise ValueError('Matrix geometry differs')
    raw = np.memmap(root/'SysMat_polar', '<f4', mode='r', shape=(132040, 2312))
    model = ViewResponse(torch.tensor(np.array(raw.T), device=device), geometry, cache='none')
    sensitivity = model.sensitivity()
    # Producer S_active covers the full cells; the unchanged active operator
    # applies the object-frame ellipse fraction, exactly once, to both B and S.
    full_s = np.fromfile(root/'S_full.float64', '<f8')
    expected = torch.zeros_like(sensitivity, dtype=torch.float64)
    full_s = torch.tensor(full_s, device=device)
    for view in range(20):
        expected += full_s[geometry.indices[view]][:, None] * geometry.fraction[:, None] / 20
    error = float(torch.linalg.vector_norm(sensitivity.double()-expected) /
                  torch.linalg.vector_norm(expected).clamp_min(1e-300))
    if error > 1e-5 or not bool(torch.isfinite(sensitivity).all()) or (require_positive and bool((sensitivity <= 1e-12).any())):
        raise ValueError('Actual complete matrix and matched active S differ: ' + str(error))
    return model, sensitivity, error


def run(release, output, index):
    frozen = read(release/'release_manifest.json');verify_files(release, frozen['sha256'])
    contract = read(release/'contract.json');spec = contract['conditions'][str(index)]
    if spec.get('requires_transport'):
        ready = read(Path(spec['input_root']).parent/'gpu_input_ready.json')
        if not ready['passed'] or ready['actual_primary_photons'] != 50_000_000_000:
            raise ValueError('Full actual EHE5e10 transport must be strictly transferred first')
        collection_sha = ready['collection_sha256']
    else:
        collection_sha = spec['collection_sha256']
    inp = Path(spec['input_root'])
    if digest(inp/'collection.json') != collection_sha:
        raise ValueError('Actual Geant4 observation manifest changed')
    collection = read(inp/'collection.json');verify_files(inp, collection['files'])
    if (not collection['passed'] or collection['total_primary_photons'] != spec['actual_primary_photons']
            or collection['views'] != 20 or sum(collection['primary_counts']) != spec['actual_primary_photons']):
        raise ValueError('Original actual complete emission count differs')
    output.mkdir(parents=True, exist_ok=False)
    job = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ['SLURM_JOB_ID'])
    identity = job+'_'+os.environ['SLURM_ARRAY_TASK_ID'] if 'SLURM_ARRAY_TASK_ID' in os.environ else job
    allocation = subprocess.check_output(['scontrol','show','job','-o',identity], text=True)
    allocated = allocated_bytes(allocation)
    write(output/'allocation.json', dict(identity=identity, scontrol=allocation, allocated_bytes=allocated))
    torch.cuda.set_device(0);torch.set_grad_enabled(False);torch.set_num_threads(8)
    device = torch.device('cuda', 0)
    geometry = ActiveGeometry.from_npz(release/'whole_geometry.npz', device)
    if (geometry.full_count, geometry.active_count, geometry.views) != (132040, 78920, 20):
        raise ValueError('Full original reconstruction domain required')
    times = {};checks = {};resources = [];started = time.time()
    channels = []
    for energy in spec['energies_keV']:
        channel = str(energy)+'_SinglePhoton'
        if energy == 218 and spec['phantom'] == 'NEMA':channel += '_CrossTalkCorrected'
        channels.append(channel)
        model, sensitivity, error = load_response(contract['factors']['A'+str(energy)], geometry, device)
        if spec['projection_format'] == 'npy':
            observed = np.load(inp/f'projection_{energy}.npy')
        else:
            observed = np.loadtxt(inp/f'projection_{energy}.csv', delimiter=',', dtype=np.float32).T
        if observed.shape != (2312,20) or not np.isfinite(observed).all() or np.any(observed < 0):
            raise ValueError('All actual Geant4 detector/view bins required')
        y = torch.tensor(observed, dtype=torch.float32, device=device)
        background = None
        if energy == 218 and spec['phantom'] == 'NEMA':
            background = torch.tensor(np.load(output/'fixed_cross_background.npy'), device=device)
        def checkpoint(iteration, history):
            active = history[-1].numpy().reshape(-1).astype('<f4')
            full = np.zeros(132040, '<f4');full[geometry.object_active.cpu().numpy()] = active
            if not np.isfinite(active).all() or np.any(active < 0):raise ValueError('Nonfinite/negative reconstruction')
            folder = output/'checkpoints'/channel/f'checkpoint_{iteration:06d}'
            temporary = folder.with_name('.'+folder.name+'.partial');temporary.mkdir(parents=True, exist_ok=False)
            array_write(temporary/'active.float32', active);array_write(temporary/'full.float32', full)
            write(temporary/'manifest.json', dict(iteration=iteration, channel=channel,
                  release_manifest_sha256=digest(release/'release_manifest.json'), files=hashes(temporary)))
            if folder.exists():raise FileExistsError('Successful checkpoint retained')
            temporary.rename(folder)
            fd=os.open(folder.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            usage=resource(allocated,device);resources.append(usage)
            write(output/'progress.json',dict(channel=channel,iteration=iteration,target=200,resource=usage,updated_epoch=time.time()))
        began=time.monotonic()
        final, history=single_mlem_checkpointed(model,y,sensitivity,200,1,additive_background=background,
            checkpoint_callback=checkpoint,progress_label=channel,phase_limit_seconds=10800)
        times[channel]=time.monotonic()-began
        array_write(output/f'Image_{channel}_history.float32',history.numpy().reshape(200,78920).astype('<f4'))
        active=final.cpu().numpy().reshape(-1).astype('<f4');full=np.zeros(132040,'<f4');full[geometry.object_active.cpu().numpy()]=active
        array_write(output/f'Image_{channel}_active.float32',active);array_write(output/f'Image_{channel}_full.float32',full)
        array_write(output/f'S_{channel}.float32',sensitivity.cpu().numpy().astype('<f4'))
        checks[channel]=dict(matched_s_relative_L2=error,finite_nonnegative=True,domain_outside_zero=True,checkpoints=200)
        if energy==440 and spec['phantom']=='NEMA':
            del model;gc.collect();torch.cuda.empty_cache()
            cross,unused,cross_error=load_response(contract['factors']['C440to218'],geometry,device,require_positive=False)
            np.save(output/'fixed_cross_background.npy',forward_project(cross,final).cpu().numpy())
            checks['fixed_background']=dict(source='this new dense-A440 EHE final200 image',
                cross_response='original frozen C440to218 retained unchanged',matched_s_relative_L2=cross_error)
            del cross,unused
        else:del model
        del final,history,sensitivity,y,background;gc.collect();torch.cuda.empty_cache()
    write(output/'run_manifest.json',dict(passed=True,study=contract['study'],condition=spec,channels=channels,
        iterations=200,save_step=1,initial_density=1,regularization='none',collection_sha256=collection_sha,
        release_sha256=digest(release/'release_manifest.json'),factor_sha256={k:v['manifest_sha256'] for k,v in contract['factors'].items()},
        phase_solve_seconds=times,numerical_checks=checks,resources=resources,job=identity,
        physical_calibration_passed=False,full_input_prevalidation_passed=False,separate_validation_job=False,
        started_epoch=started,completed_epoch=time.time()))
    write(output/'completion.json',dict(passed=True,job=identity,condition_index=index,channels=channels,files=hashes(output)))
    print('EHE_DENSE_FORMAL_COMPLETE',index,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    try:run(args.release,args.output,args.index)
    except BaseException as error:
        if args.output.is_dir():write(args.output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise
