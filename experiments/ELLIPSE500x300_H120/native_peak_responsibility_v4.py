"""Stream frozen events through a fixed historical image; no new MLEM update."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from process_list_global_audit_v4 import settings,metadata,vector,digest,write,table
from compton_event_response import prepare_compton_events,build_detector_position_variance,build_compton_cone_weights,_stable_event_angles
from detector_csv import load_detector_coordinates

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    gate=json.loads((a.analysis/'validation_gate.json').read_text())
    assert digest(a.geometry)==gate['geometry_sha256']
    assert digest(Path(__file__).parent/'compton_event_response.py')==gate['kernel_sha256']
    assert digest(a.inputs/'input_manifest.json')==gate['input_manifest_sha256']
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())['files']
    for name,sha in manifest.items():
        if name.startswith('NEMA/') and name.split('/')[-1].startswith(('ideal_v','events_v')):
            assert digest(a.inputs/name)==sha,name
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.cuda.set_device(0)
    geo=np.load(a.geometry);coords=geo['coordinates_mm'];active=geo['active_indices'];inv=geo['inverse_rotation']
    fraction=geo['ellipse_fraction'];vol=geo['cell_volume_mm3']
    cells=np.array(json.loads(a.probes.read_text())['cells'],dtype=int)
    lookup={int(j):i for i,j in enumerate(active)}
    assert all(int(j) in lookup for j in cells)
    ai=np.array([lookup[int(j)] for j in cells])
    images={c:np.fromfile(a.images/f'Image_{c}_active.float32','<f4') for c in ['440_ComptonOnly','440_SinglePlusCompton']}
    assert all(len(im)==len(active) and np.all(np.isfinite(im)) and np.all(im>=0) for im in images.values())
    detector=torch.tensor(load_detector_coordinates(a.factors/'440keV_RotateNum20/Detector.csv',10496),device='cuda',dtype=torch.float32)
    var=build_detector_position_variance(detector,0);coordinate=torch.tensor(coords,device='cuda',dtype=torch.float32)
    matrix=np.memmap(a.factors/'440keV_RotateNum20/SysMat_polar','<f4',mode='r',shape=(132040,10496))
    B=torch.tensor(np.array(matrix.T,copy=True),device='cuda');del matrix
    V=torch.tensor(vol,device='cuda',dtype=torch.float32)
    rows=[];event_records=[];spectrum=[];scores={c:[] for c in images};only_scores={c:[] for c in images}
    for view in range(1,21):
        chosen=np.load(a.analysis/f'NEMA_ideal_v{view:02d}_kept_rows.npy')
        raw=np.loadtxt(a.inputs/f'NEMA/ideal_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        wanted=set(map(int,chosen));meta={int(r['global_ideal_row']):r for r in metadata(a.inputs/'NEMA',view,wanted)}
        mapped=inv[active,view-1];probe=inv[cells,view-1]
        detector_images={}
        for c,image in images.items():
            full=np.zeros(len(coords),np.float32);full[mapped]=image*fraction[active]
            detector_images[c]=torch.tensor(full,device='cuda')
        peak_coord=coordinate[probe]
        for offset in range(0,len(chosen),32):
            indices=chosen[offset:offset+32]
            p,_=prepare_compton_events(torch.tensor(raw[indices],device='cuda'),settings(),detector,var,var,input_energies_already_smeared=True)
            assert p is not None and p.count==len(indices)
            K=build_compton_cone_weights(p,coordinate,settings());R=K*B[p.cpnum1-1]
            beta,theta,sigma=_stable_event_angles(p,peak_coord,settings())
            q=((beta-theta[:,None]).abs()/sigma).cpu().numpy()
            pair=np.column_stack((p.cpnum1.cpu().numpy(),p.cpnum2.cpu().numpy()))
            for c,image in images.items():
                prediction=R@detector_images[c]
                if not bool(torch.all(prediction>0)):raise ValueError('Frozen image has impossible events')
                responsibility=(R[:,probe]*torch.tensor(image[ai]*fraction[cells],device='cuda')/prediction[:,None]).cpu().numpy()
                prediction_k=K@(detector_images[c]*V)
                only=(K[:,probe]*torch.tensor(image[ai]*fraction[cells]*vol[cells],device='cuda')/prediction_k[:,None]).cpu().numpy()
                scores[c].append(responsibility);only_scores[c].append(only)
            angle=(view-1)*np.pi/10
            for index in indices:
                r=meta[int(index)];src=vector(r,'source_')+np.array([0,345,0])
                obj=src.copy();obj[0]=src[0]*np.cos(angle)-src[1]*np.sin(angle);obj[1]=src[0]*np.sin(angle)+src[1]*np.cos(angle)
                event_records.append(dict(view=view,input_row=int(index),worker=int(r['worker']),seed=int(r['seed']),event_id=int(r['event_id']),
                    measured_e1_keV=float(r['measured_e1'])*1000,measured_e2_keV=float(r['measured_e2'])*1000,
                    c1=int(r['c1']),c2=int(r['c2']),source_x_mm=float(obj[0]),source_y_mm=float(obj[1]),source_z_mm=float(obj[2])))
            spectrum.extend(q.tolist())
        print('NATIVE_RESPONSIBILITY_VIEW',view,'events',len(chosen),flush=True)
    total=len(event_records);assert total==91225
    metadata_array=np.array([[r['source_x_mm'],r['source_y_mm'],r['source_z_mm']] for r in event_records])
    seeds,seed_labels=np.unique([r['seed'] for r in event_records],return_inverse=True)
    assert len(seeds)==200,'Global worker identity is seed; per-view worker indices reset'
    q=np.array(spectrum);sens=np.fromfile(a.analysis/'ideal/Sensi_d','<f4')
    top=[]
    for c,image in images.items():
        weights=np.concatenate(scores[c]);only=np.concatenate(only_scores[c])
        for k,j in enumerate(cells):
            w=weights[:,k].astype(float);order=np.argsort(w)[::-1];totalw=float(w.sum())
            cumulative=np.cumsum(w[order])/totalw
            per_view=np.array([w[np.array([r['view']==v for r in event_records])].sum() for v in range(1,21)])
            per_worker=np.bincount(seed_labels,weights=w,minlength=200)
            rows.append(dict(channel=c,full_cell_index=int(j),x_mm=float(coords[j,0]),y_mm=float(coords[j,1]),z_mm=float(coords[j,2]),
                density=float(image[ai[k]]),fraction=float(fraction[j]),effective_volume_mm3=float(vol[j]*fraction[j]),
                assigned_event_mass=totalw,effective_contributing_events=float(totalw**2/(w@w)),
                events_for_50_percent=int(np.searchsorted(cumulative,.5)+1),events_for_90_percent=int(np.searchsorted(cumulative,.9)+1),
                top_one_mass_fraction=float(w.max()/totalw),top_ten_mass_fraction=float(w[order[:10]].sum()/totalw),
                participating_views=int(np.sum(per_view>totalw*.001)),participating_workers=int(np.sum(per_worker>totalw*.001)),
                source_distance_weighted_mean_mm=float(w@np.linalg.norm(metadata_array-coords[j],axis=1)/totalw),
                q_at_cell_above_3_mass_fraction=float(w[q[:,k]>3].sum()/totalw),
                K_only_assigned_mass=float(only[:,k].sum()),
                full_circle_stable_S=float(sens[j]),matched_S_times_historical_density=float(sens[j]*fraction[j]*image[ai[k]])))
            for i in order[:128]:top.append(dict(channel=c,full_cell_index=int(j),responsibility=float(w[i]),q_at_cell=float(q[i,k]),**event_records[int(i)]))
        np.savez_compressed(a.output/(c+'_responsibility.npz'),response_KA=weights,response_K_only=only,cells=cells)
    table(a.output/'native_peak_responsibility.csv',rows);table(a.output/'top_responsible_events.csv',top)
    write(a.output/'execution.json',dict(events=total,probe_cells=cells.tolist(),elapsed_seconds=time.monotonic()-start,
        geometry_sha256=digest(a.geometry),source_sha256=digest(__file__),probe_sha256=digest(a.probes),
        image_sha256={c:digest(a.images/f'Image_{c}_active.float32') for c in images},
        selection_sha256={f'NEMA_v{v:02d}':digest(a.analysis/f'NEMA_ideal_v{v:02d}_kept_rows.npy') for v in range(1,21)},
        independent_workers=200,worker_identity='seed; local worker indices reset in each view',
        input_manifest_sha256=digest(a.inputs/'input_manifest.json'),sensitivity_sha256=digest(a.analysis/'ideal/Sensi_d'),
        kernel_sha256=digest(Path(__file__).parent/'compton_event_response.py'),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        baseline_image='Historical ideal 1660254 at iteration 2000',
        response='Current stable R1 kernel and 91225 frozen events evaluated at a fixed historical image; not the exact old MLEM iterate',
        K_only='Diagnostic posterior with A=1 and full volumes; does not include physical collimator/detection efficiency',
        inference='Responsibilities are conditional on the fixed reconstructed image, not independent causal evidence',
        new_transport_photons=0,new_reconstruction=False))
    print('NATIVE_RESPONSIBILITY_COMPLETE',total,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','analysis','factors','geometry','probes','images','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
