"""Full-event 3-sigma study scan and independent matched sensitivity, no MLEM."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(Path(os.environ.get('JSCC_PROJECT_ROOT', HERE.parents[1])))]
from compton_event_response import (ComptonEventSettings, PreparedComptonEvents,
    build_compton_cone_weights, build_detector_position_variance,
    min_standardized_compton_arm, prepare_compton_events,
    select_normalized_response_rows)
from detector_csv import load_detector_coordinates


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def subset(prepared, index):
    return PreparedComptonEvents(**{f.name: (None if getattr(prepared,f.name) is None
        else getattr(prepared,f.name)[index]) for f in fields(prepared)})


def rotation_average(values, rotation):
    source = torch.from_numpy(np.asarray(values, dtype=np.float32))
    result = torch.zeros_like(source)
    for view in range(20):
        result += source[torch.from_numpy(rotation[:,view]-1).long()]
    return (result/20).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--truth', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    factor = args.base/'generated/FactorsCalibrated/440keV_RotateNum20'
    baseline = json.loads((args.baseline_root/'run_manifest.json').read_text())
    if digest(args.geometry) != baseline['geometry_sha256']:
        raise ValueError('Baseline geometry SHA256 differs')
    if digest(factor/'Sensi_d') != baseline['sensi_d_sha256']:
        raise ValueError('Baseline sensitivity differs')
    if digest(factor/'factor_manifest.json') != baseline['factor_manifest_sha256']['440keV_RotateNum20']:
        raise ValueError('Baseline factor manifest differs')
    geo = np.load(args.geometry)
    coords = torch.tensor(np.loadtxt(factor/'coor_polar_full.csv', delimiter=',', dtype=np.float32), device=device)
    if coords.shape != (132040,3) or not np.allclose(coords.cpu(), geo['coordinates_mm'], atol=1e-5):
        raise ValueError('Complete calculation grid differs')
    active = geo['active_indices']
    fraction = torch.tensor(geo['ellipse_fraction'][active], device=device, dtype=torch.float32)
    truth = torch.tensor(np.load(args.truth)['density_64'][active], device=device, dtype=torch.float64)
    volumes = np.fromfile(factor/'polar_cell_volume_mm3.float64', '<f8')
    rotation = np.loadtxt(factor/'RotMat_full.csv', delimiter=',', dtype=np.int64)
    if rotation.shape != (132040,20) or len(volumes) != 132040:
        raise ValueError('Rotation/volume shape differs')
    detector = torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    variance = build_detector_position_variance(detector,0)
    settings = ComptonEventSettings(.440,.13*np.sqrt(511/440),
        2*.440**2/(.511+2*.440)-.001,.05,.350,max_min_standardized_arm=3.0)
    raw_b = np.memmap(factor/'SysMat_polar','<f4',mode='r',shape=(132040,10496))
    b = torch.tensor(np.array(raw_b.T,copy=True),device=device)
    del raw_b
    response_sum = torch.zeros(len(active),device=device,dtype=torch.float64)
    truth_score_sum = torch.zeros_like(response_sum)
    records, rejected = [], []
    sensitivity_sums = {}
    group_totals = {}
    input_hashes = {}
    known_removed = False
    def scan(path, group, view):
        nonlocal known_removed
        sha = digest(path)
        input_hashes[str(path)] = sha
        if group=='nema':
            relative = f'List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/{view}.csv'
            if sha != baseline['input_sha256'][relative]: raise ValueError('NEMA input SHA differs')
        raw = np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        prepared, diagnostics = prepare_compton_events(torch.tensor(raw,device=device),settings,
            detector,variance,variance,input_energies_already_smeared=True)
        accumulator = torch.zeros(132040,device=device,dtype=torch.float64)
        original_accumulator = torch.zeros_like(accumulator)
        accepted = removed = 0
        kept_rows = []
        if prepared is not None:
            indices = (torch.tensor(geo['inverse_rotation'][active,view-1],device=device)
                       if group=='nema' else None)
            for start in range(0,prepared.count,args.batch_size):
                e = subset(prepared,slice(start,start+args.batch_size))
                cone = build_compton_cone_weights(e,coords,settings)
                weight = cone*b[e.cpnum1-1]
                sums = weight.sum(1)
                valid = torch.isfinite(weight).all(1)&torch.isfinite(sums)&(sums>0)
                diagnostics.invalid_kernel_events += int((~valid).sum())
                if not bool(valid.any()): continue
                ev = subset(e,valid)
                normalized = weight[valid]/sums[valid,None]
                q = min_standardized_compton_arm(ev,coords,settings)
                old_keep, low, _ = select_normalized_response_rows(normalized,1.0)
                new_keep, _, ncut = select_normalized_response_rows(normalized,1.0,q,3.0)
                diagnostics.low_support_rejected_events += low
                diagnostics.mismatch_rejected_events += ncut
                accepted += int(old_keep.sum()); removed += ncut
                original_accumulator += normalized[old_keep].double().sum(0)
                accumulator += normalized[new_keep].double().sum(0)
                if group=='nema':
                    kept_rows.extend(ev.source_row_indices[new_keep].cpu().tolist())
                    cut = old_keep&~new_keep
                    for i in torch.nonzero(cut,as_tuple=True)[0].cpu().tolist():
                        source_row = int(ev.source_row_indices[i])
                        # Prepared index is tracked from the complete per-view preparation.
                        preindex = int(torch.searchsorted(prepared.source_row_indices,ev.source_row_indices[i]))
                        record = {'view':view,'raw_row_0based':source_row,'csv_line_1based':source_row+1,
                            'input_sha256':sha,'pre_kernel_index':preindex,'q':float(q[i]),
                            'c1':int(ev.cpnum1[i]),'c2':int(ev.cpnum2[i]),
                            'e1_MeV':float(ev.e1[i]),'e2_MeV':float(ev.e2[i]),
                            'first_y_mm':float(ev.pos1[i,1]),'second_y_mm':float(ev.pos2[i,1])}
                        rejected.append(record)
                        if view==7 and preindex==10254 and record['c1']==7629 and record['c2']==1936:
                            known_removed = True
                    if bool(cut.any()):
                        compact = normalized[cut][:,indices]*fraction
                        response_sum.add_(compact.double().sum(0))
                        prediction = compact.double()@truth
                        truth_score_sum.add_((compact.double()/prediction.clamp_min(1e-12)[:,None]).sum(0))
                del cone,weight,normalized,q
        diagnostics.kept_events = accepted-removed
        row = {'group':group,'view':view,'raw_rows':len(raw),'baseline_accepted':accepted,
            'removed':removed,'kept':accepted-removed,'input_sha256':sha,'diagnostics':diagnostics.to_dict()}
        records.append(row)
        if group=='nema':
            np.save(args.output/f'kept_raw_rows_view_{view}.npy',np.array(kept_rows,dtype=np.int64))
        else:
            sensitivity_sums[group] = accumulator.cpu().numpy()
            sensitivity_sums[group+'_original'] = original_accumulator.cpu().numpy()
        print(json.dumps({**row,'elapsed_seconds':time.monotonic()-started}),flush=True)
        (args.output/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    nema_root = args.baseline_root/'inputs/generated'
    for view in range(1,21):
        scan(nema_root/f'List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/{view}.csv','nema',view)
    group_totals['nema'] = {key:sum(r[key] for r in records) for key in ('raw_rows','baseline_accepted','removed','kept')}
    if group_totals['nema']['baseline_accepted']!=484936 or not known_removed:
        raise ValueError('Baseline event closure or known-event rejection failed')
    if group_totals['nema']['removed']/484936 > .01:
        raise ValueError('More than 1% removed: investigation required before reconstruction')
    with (args.output/'rejected_events.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rejected[0]) if rejected else ['view','raw_row_0based','q'])
        writer.writeheader(); writer.writerows(rejected)
    np.save(args.output/'removed_backprojection_active.npy',response_sum.cpu().numpy())
    old_s = np.fromfile(factor/'Sensi_d','<f4')[active]*geo['ellipse_fraction'][active]
    np.save(args.output/'removed_truth_score_active.npy',truth_score_sum.cpu().numpy()/old_s)
    source_counts, seed_sets = {}, []
    for dataset, group in [('sensitivity_440','train'),('sensitivity_validation_440','validation')]:
        collection = args.base/f'generated/collections/{dataset}_all.json'
        record = json.loads(collection.read_text())
        source_counts[group] = record['primary_counts'][1]
        if source_counts[group]<10**9 or sum(record['primary_counts'])!=source_counts[group]:
            raise ValueError('Independent source primary closure failed')
        seed_sets.append(set(record['seeds']))
        input_hashes[str(collection)] = digest(collection)
        scan(args.base/f'generated/List/218-440keV_RotateNum20_Geant4JSCC/List_{dataset}_all/1.csv',group,1)
        row = records[-1]; group_totals[group] = {k:row[k] for k in ('raw_rows','baseline_accepted','removed','kept')}
    if seed_sets[0]&seed_sets[1]: raise ValueError('Independent sensitivity seeds overlap')
    volume = float(volumes.sum())
    full_s = rotation_average(sensitivity_sums['train']*volume/source_counts['train'],rotation)
    full_s.astype('<f4').tofile(args.output/'Sensi_d')
    original_s = rotation_average(sensitivity_sums['train_original']*volume/source_counts['train'],rotation)
    old_full = np.fromfile(factor/'Sensi_d','<f4')
    old_regression = float(np.linalg.norm(original_s-old_full)/np.linalg.norm(old_full))
    validation_s = rotation_average(sensitivity_sums['validation']*volume/source_counts['validation'],rotation)
    ratio = validation_s.astype(np.float64)/full_s
    closure = {'volume_weighted_mean':float(ratio@volumes/volume),'cv':float(ratio.std()/ratio.mean()),
        'min':float(ratio.min()),'max':float(ratio.max()),
        'relative_l2_original_sensitivity':old_regression,
        'sensitivity_total_efficiency':float(full_s.sum(dtype=np.float64)/volume),
        'train_accepted_efficiency':group_totals['train']['kept']/source_counts['train']}
    if not np.isfinite(full_s).all() or np.any(full_s<=0) or not np.isfinite(ratio).all():
        raise ValueError('Sensitivity finite/positive check failed')
    if abs(closure['volume_weighted_mean']-1)>.02 or closure['cv']>.02 or old_regression>1e-4:
        raise ValueError('Independent uniform closure or old-Sensi regression failed')
    np.save(args.output/'sensitivity_new_old_ratio.npy',full_s/old_full)
    np.save(args.output/'uniform_closure_ratio.npy',ratio)
    provenance = {'experiment':'ELLIPSE500x300_H120','pixel_count':132040,'operator':'K*B',
        'resolution_fwhm':.13,'reference_keV':511,'sum_threshold_MeV':.350,
        'input_already_smeared':True,'max_min_standardized_arm':3.0,
        'source_photons':source_counts['train'],'independent_validation_photons':source_counts['validation'],
        'independent_validation_seeds':sorted(seed_sets[1]),'closure':closure,
        'sensi_d_sha256':digest(args.output/'Sensi_d'),'baseline_sensi_d_sha256':digest(factor/'Sensi_d'),
        'geometry_sha256':digest(args.geometry),'factor_manifest_sha256':digest(factor/'factor_manifest.json')}
    (args.output/'Sensi_d_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    report = {'study':'response_mismatch_cut3_v1','max_min_standardized_arm':3.0,
        'criterion_domain':'all 132040 complete-circle centers; no truth used in selection',
        'groups':group_totals,'known_event_removed':known_removed,'input_sha256':input_hashes,
        'geometry_sha256':digest(args.geometry),'factor_manifest_sha256':digest(factor/'factor_manifest.json'),
        'matrix_sha256':digest(factor/'SysMat_polar'),'sensi_d_sha256':digest(args.output/'Sensi_d'),
        'closure':closure,'device':str(device),'elapsed_seconds':time.monotonic()-started,
        'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
        'code_sha256':{Path(__file__).name:digest(__file__),'compton_event_response.py':digest(Path(sys.modules['compton_event_response'].__file__))}}
    (args.output/'scan_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
