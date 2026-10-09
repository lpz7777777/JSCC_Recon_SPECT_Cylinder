"""Independent matrix-generated H60 data; never a Geant4 observation or calibration."""
import argparse, gc, time
from pathlib import Path
import numpy as np
from ehe_common import digest, read, write, verify_files, hashes, allocation, resources, RESPONSES
from ehe_gpu_pipeline import source_grid

STUDY = 'ehe_forward_poisson_5e9_200'
COMPONENT_ENERGY = {'A218': 218, 'A440': 440, 'C440to218': 440}


def dose_budget(truth, config):
    integrals = {str(e): float(truth[f'activity_{e}_zyx'].astype(np.float64).sum()*27) for e in (218,440)}
    if any(not np.isclose(integrals[e], config['relative_activity_integral_mm3'][e], rtol=1e-13) for e in integrals):
        raise ValueError('Actual 3D truth integral differs')
    weights = {e: integrals[e]*config['gamma_yields'][e] for e in integrals}
    total = sum(weights.values())
    photons = {e: config['expected_emitted_photons']*weights[e]/total for e in weights}
    return dict(integrals_mm3=integrals, expected_primary_photons=photons,
                density_gamma_per_mm3={e: photons[e]/integrals[e] for e in integrals},
                expected_emitted_photons=sum(photons.values()))


def sample_components(means, seeds):
    samples = {}
    for name in RESPONSES:
        mean = np.asarray(means[name], np.float64)
        if mean.shape != (2312,20) or not np.isfinite(mean).all() or np.any(mean<0):
            raise ValueError('Complete nonnegative 2312-bin/20-view mean required')
        samples[name] = np.random.Generator(np.random.PCG64(seeds[name])).poisson(mean)
    return samples, {218: samples['A218']+samples['C440to218'], 440: samples['A440'].copy()}


def save_npy(path, array):
    import os
    path=Path(path); temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:
        np.save(f,array);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def generate(release, responses, output):
    import torch
    release=Path(release);responses=Path(responses);output=Path(output)
    freeze=read(release/'release_manifest.json');verify_files(release,freeze['sha256'])
    cfg=read(release/'config.json');truth=np.load(release/'truth_3mm.npz');budget=dose_budget(truth,cfg)
    output.mkdir(parents=True,exist_ok=False);alloc=allocation();write(output/'allocation.json',alloc)
    began=time.monotonic();means={};factor_sha={};checks={}
    for name in RESPONSES:
        folder=responses/name;manifest=read(folder/'factor_manifest.json')
        if digest(folder/'factor_manifest.json')!=cfg['factor_sha256'][name]:raise ValueError('Frozen factor manifest differs')
        verify_files(folder,manifest['files'])
        raw=np.memmap(folder/'SysMat_cartesian','<f4',mode='r',shape=(2312,40*85*85))
        matrix=torch.as_tensor(np.array(raw),device='cuda');del raw
        if not bool(torch.isfinite(matrix).all()) or bool((matrix<0).any()):raise ValueError('Invalid full Cartesian matrix')
        e=COMPONENT_ENERGY[name];values=np.empty((2312,20),np.float64)
        for view in range(20):
            source=source_grid(truth,e,view)
            if not np.isclose(source.sum(),1,rtol=1e-12) or np.any(source<0):raise ValueError('Source mass conservation failed')
            weights=torch.as_tensor(source.reshape(-1),dtype=torch.float32,device='cuda')
            values[:,view]=(matrix@weights).cpu().numpy().astype(np.float64)*budget['expected_primary_photons'][str(e)]/20
        means[name]=values;factor_sha[name]=digest(folder/'factor_manifest.json')
        del matrix,weights;gc.collect();torch.cuda.empty_cache()
        checks[name]=resources(alloc)
        print('FORWARD_MEAN',name,float(values.sum()),flush=True)
    sampled,projection=sample_components(means,cfg['noise_seeds'])
    summary={}
    for name in RESPONSES:
        save_npy(output/f'mean_{name}.npy',means[name]);save_npy(output/f'sampled_{name}.npy',sampled[name])
        mean=float(means[name].sum());count=int(sampled[name].sum())
        summary[name]=dict(expected_counts=mean,sampled_counts=count,global_poisson_z=(count-mean)/np.sqrt(mean),
                           expected_by_view=means[name].sum(axis=0).tolist(),sampled_by_view=sampled[name].sum(axis=0).tolist())
    for e in (218,440):save_npy(output/f'projection_{e}.npy',projection[e])
    record=dict(passed=True,study=STUDY,data_kind='matrix_forward_plus_independent_Poisson',
        release_key=freeze['release_key'],config_sha256=digest(release/'config.json'),truth_sha256=digest(release/'truth_3mm.npz'),
        geometry_sha256=digest(release/'whole_geometry.npz'),factor_sha256=factor_sha,views=20,bins=2312,
        budget=budget,noise_seeds=cfg['noise_seeds'],bit_generator='PCG64',numpy_version=np.__version__,
        components=summary,window_counts={str(e):int(projection[e].sum()) for e in (218,440)},
        generated_cross_fraction=float(sampled['C440to218'].sum()/projection[218].sum()),
        physical_calibration_claim=False,transport_performed=False,
        source_basis='All actual 3mm truth voxel masses, original Cartesian bilinear stencil and clockwise 20-view rotation',
        reconstruction_basis='Original complete volume-weighted Polar operator; source and reconstruction grids differ',
        allocation=alloc,resources=checks,elapsed_seconds=time.monotonic()-began,files=hashes(output))
    write(output/'collection.json',record)


def verify_counts(counts, release):
    counts=Path(counts);release=Path(release);record=read(counts/'collection.json');cfg=read(release/'config.json')
    verify_files(counts,record['files'])
    if record['study']!=STUDY or record['data_kind']!='matrix_forward_plus_independent_Poisson' or not record['passed']:
        raise ValueError('Independent synthetic study identity required')
    if record['physical_calibration_claim'] or record['transport_performed'] or (record['views'],record['bins'])!=(20,2312):
        raise ValueError('Synthetic measurement contract differs')
    if record['config_sha256']!=digest(release/'config.json') or record['truth_sha256']!=digest(release/'truth_3mm.npz') or record['geometry_sha256']!=digest(release/'whole_geometry.npz'):
        raise ValueError('Source/config/geometry identity differs')
    if record['factor_sha256']!=cfg['factor_sha256'] or record['noise_seeds']!=cfg['noise_seeds']:
        raise ValueError('Original matrices or preregistered seeds differ')
    budget=dose_budget(np.load(release/'truth_3mm.npz'),cfg)
    if record['budget']!=budget or cfg['expected_emitted_photons']!=5_000_000_000:raise ValueError('Expected source dose differs')
    means={n:np.load(counts/f'mean_{n}.npy') for n in RESPONSES}
    sampled,projection=sample_components(means,cfg['noise_seeds'])
    for name in RESPONSES:
        saved=np.load(counts/f'sampled_{name}.npy')
        if saved.dtype!=np.dtype('int64') or not np.array_equal(saved,sampled[name]):raise ValueError('Seeded component replay differs')
    for e in (218,440):
        saved=np.load(counts/f'projection_{e}.npy')
        if saved.dtype!=np.dtype('int64') or not np.array_equal(saved,projection[e]):raise ValueError('Poisson component/window addition differs')
        if record['window_counts'][str(e)]!=int(saved.sum()):raise ValueError('Recorded count total differs')
    return record


def verify_forward_means(counts, release, responses):
    """Independent CPU float64 multiplication of every Cartesian row and all views."""
    counts=Path(counts);release=Path(release);responses=Path(responses)
    record=verify_counts(counts,release);truth=np.load(release/'truth_3mm.npz');proof={};started=time.monotonic()
    for name in RESPONSES:
        e=COMPONENT_ENERGY[name]
        weights=np.column_stack([source_grid(truth,e,v).reshape(-1) for v in range(20)])
        raw=np.memmap(responses/name/'SysMat_cartesian','<f4',mode='r',shape=(2312,40*85*85));prediction=np.empty((2312,20))
        for start in range(0,2312,64):
            rows=np.array(raw[start:start+64],dtype=np.float64)
            if not np.isfinite(rows).all() or np.any(rows<0):raise ValueError('Invalid Cartesian rows')
            prediction[start:start+64]=rows@weights*record['budget']['expected_primary_photons'][str(e)]/20
        saved=np.load(counts/f'mean_{name}.npy');delta=prediction-saved
        l2=float(np.linalg.norm(delta)/max(np.linalg.norm(prediction),1e-30))
        view_l2=np.linalg.norm(delta,axis=0)/np.maximum(np.linalg.norm(prediction,axis=0),1e-30)
        if l2>1e-5 or np.any(view_l2>1e-5):raise ValueError('All-row/all-view source forward identity failed')
        proof[name]=dict(l2=l2,l2_by_view=view_l2.tolist(),cpu_float64_counts=float(prediction.sum()))
        print('CPU_FORWARD_VERIFIED',name,l2,flush=True)
    return dict(passed=True,method='Independent float64 Cartesian multiplication, all 2312 rows and 20 views; same registered 3mm source stencil',
                components=proof,elapsed_seconds=time.monotonic()-started,gpu_resource_certificate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--responses',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:generate(a.release,a.responses,a.output)
    except BaseException as exc:
        if a.output.is_dir():write(a.output/'failure.json',dict(passed=False,error=str(exc)))
        raise
