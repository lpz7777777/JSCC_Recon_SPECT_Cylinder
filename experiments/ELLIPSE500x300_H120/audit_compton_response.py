"""Read-only stratified event-response audit; no image update or input mutation.

Run on an allocated GPU or bounded CPU allocation, with the production grid.
Historical kernels here are controlled *kernel-only* variants on identical
already-broadened events, not reconstructions of an undocumented old experiment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from compton_event_response import (ComptonEventSettings, PreparedComptonEvents,
    build_detector_position_variance, prepare_compton_events,
    build_compton_cone_weights, compton_theta_from_e1,
    _energy_angle_sigma, _position_angle_sigma)
from detector_csv import load_detector_coordinates
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from compton_sparse_ops import build_compton_sparse_projector, materialize_sparse_event_rows_to_fine


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def subset(events, indices):
    return PreparedComptonEvents(**{k: getattr(events, k)[indices]
                                  for k in events.__dataclass_fields__})


def kernel_details(e, coords, settings):
    v01 = e.pos1[:, None] - coords[None]
    v12 = (e.pos2-e.pos1)[:, None]
    d01, d12 = v01.norm(dim=2), v12.norm(dim=2)
    beta = torch.acos(((v01*v12).sum(2)/(d01*d12)).clamp(-1+1e-7, 1-1e-7))
    theta = compton_theta_from_e1(e.e1, settings.energy_mev)
    se = _energy_angle_sigma(e.e1, settings, beta, theta)
    sp = _position_angle_sigma(v01, v12, e.sigma_pos1_sq,
                              e.sigma_pos2_sq, True)
    sigma = (se**2+sp**2).sqrt()
    arm = (beta-theta[:, None]).abs()/sigma
    # Exact historical Taylor/2 mm formula, with current energy resolution and
    # current measured energies held fixed to isolate the kernel change.
    er = e.e1*settings.energy_resolution/2.355*(.440/e.e1).sqrt()
    first = .511/((.440-e.e1)**2*theta.sin().abs())
    second = .511/(theta.sin()*(.440-e.e1)**3)*(2-theta.cos()/theta.sin()**2*.511/(.440-e.e1))
    se_old = torch.where(beta >= theta[:, None],
                        (first*er+.5*second*er**2).abs()[:, None],
                        (first*er-.5*second*er**2).abs()[:, None])
    ratio = d01/d12
    sp_old = torch.atan(2/d01)*(1+2*ratio**2+2*ratio*theta.cos()[:, None]).sqrt()
    kn = (.440/(.440-e.e1)+(.440-e.e1)/.440)[:, None]-beta.sin()**2
    old = torch.exp(-(beta-theta[:, None])**2/(2*(se_old**2+sp_old**2)))*kn
    return arm, old


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample-per-view', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--truth', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    device = args.device
    base = args.base
    factors = base/'generated/FactorsCalibrated/440keV_RotateNum20'
    result = base/'generated/Results/NEMA_Body_H60_5e9_1644876'
    run = json.loads((result/'run_manifest.json').read_text())
    gp = base/'generated/Geometry/geometry.npz'
    assert digest(gp) == run['geometry_sha256']
    assert digest(factors/'Sensi_d') == run['sensi_d_sha256']
    assert digest(factors/'factor_manifest.json') == run['factor_manifest_sha256']['440keV_RotateNum20']
    geo = np.load(gp)
    coords = torch.tensor(geo['coordinates_mm'], dtype=torch.float32, device=device)
    active = geo['active_indices']
    fraction = torch.tensor(geo['ellipse_fraction'][active], dtype=torch.float32, device=device)
    tiny = fraction < .1
    outer = torch.tensor(np.abs(geo['coordinates_mm'][active, 2]) > 51, device=device)
    source_z = torch.tensor(np.abs(geo['coordinates_mm'][active, 2]) < 30, device=device)
    c = np.fromfile(result/'Image_440_ComptonOnly_full.float32', '<f4')[active]
    x = torch.tensor(c, device=device)
    peak = int(c.argmax())
    sensitivity = np.fromfile(factors/'Sensi_d', '<f4')[active]*geo['ellipse_fraction'][active]
    truth = None
    gradients = {}
    floored_gradients = {}
    if args.truth:
        truth = {k: torch.tensor(v[active], device=device, dtype=torch.float64)
                 for k, v in np.load(args.truth).items()}
        gradients = {k: torch.zeros(len(active), device=device, dtype=torch.float64) for k in truth}
        floored_gradients = {k: torch.zeros_like(value) for k,value in gradients.items()}
    detector = torch.tensor(load_detector_coordinates(factors/'Detector.csv', 10496), device=device)
    var = build_detector_position_variance(detector, 0)
    settings = ComptonEventSettings(.440, .13*np.sqrt(511/440),
                                   2*.440**2/(.511+2*.440)-.001, .050, .350)
    matrix = np.memmap(factors/'SysMat_polar', '<f4', mode='r', shape=(132040, 10496))
    b = torch.tensor(np.array(matrix.T, copy=True), device=device)
    rng = np.random.default_rng(args.seed)
    rows, views = [], []
    adapter_check = None
    for view in range(20):
        path = base/'generated/List/218-440keV_RotateNum20_Geant4JSCC'/f'List_NEMA_Body_H60_5e9/{view+1}.csv'
        rel = path.relative_to(base/'generated').as_posix()
        assert digest(path) == run['input_sha256'][rel]
        raw = np.loadtxt(path, delimiter=',', usecols=(0,1,2,3), dtype=np.float32, ndmin=2)
        prepared, diag = prepare_compton_events(torch.tensor(raw, device=device), settings,
                                               detector, var, var,
                                               input_energies_already_smeared=True)
        n = prepared.count
        chosen = np.sort(rng.choice(n, size=min(args.sample_per_view, n), replace=False))
        view_info = {'view': view+1, 'pre_kernel_events': n, 'filter': diag.to_dict(),
                     'sample': len(chosen), 'sum_below_400_keV': int(((prepared.e1+prepared.e2)<.4).sum()),
                     'first_energy_over_250_keV': int((prepared.e1>.25).sum())}
        views.append(view_info)
        indices = torch.tensor(geo['inverse_rotation'][active, view], device=device, dtype=torch.long)
        for offset in range(0, len(chosen), 16):
            chosen_batch = chosen[offset:offset+16]
            e = subset(prepared, torch.tensor(chosen_batch, device=device))
            current = build_compton_cone_weights(e, coords, settings)
            arm, old = kernel_details(e, coords, settings)
            variants = {'current': current, 'legacy_taylor_2mm_same_energy': old,
                        'no_first_source_leg': build_compton_cone_weights(e, coords,
                                                replace(settings, include_first_hit_source_leg_uncertainty=False))}
            br = b[e.cpnum1-1]
            if view==0 and offset==0:
                projector = build_compton_sparse_projector(coords,theta_stride=1,z_stride=1,rotate_num=20).to(device)
                inp = torch.stack((e.cpnum1[:2],e.e1[:2],e.cpnum2[:2],e.e2[:2]),dim=1)
                packed,_,_ = get_compton_backproj_list_single_sparse(b,detector,projector,inp,
                    0,0,.440,settings.energy_resolution,settings.energy_threshold_max_mev,.05,.35,
                    device,input_energies_already_smeared=True)
                official,keep = materialize_sparse_event_rows_to_fine(packed,b,projector)
                direct = current[:2]*br[:2]
                direct /= direct.sum(1)[:,None]
                assert official.shape==direct.shape and bool(keep.all())
                error = float((official-direct).abs().amax())
                assert error<1e-7
                adapter_check={'events':2,'full_columns':132040,'max_absolute_error':error,
                               'criterion':'direct shared K*B matches official sparse/materialized full-stride chain'}
                del projector,packed,official,direct
            for variant, k in variants.items():
                raw_response = k*br
                total = raw_response.sum(1)
                norm = raw_response/torch.where(total>0,total,torch.ones_like(total))[:, None]
                compact = norm[:, indices]*fraction
                active_sum = compact.sum(1)
                posterior = compact/torch.where(active_sum>0,active_sum,torch.ones_like(active_sum))[:, None]
                responsibility = compact*x
                final_prediction = responsibility.sum(1)
                rsum = responsibility.sum(1)
                responsibility /= torch.where(rsum>0,rsum,torch.ones_like(rsum))[:, None]
                stats = {'raw_sum': total, 'effective_support_full': 1/(norm**2).sum(1),
                         'effective_support_active': 1/(posterior**2).sum(1),
                         'ellipse_row_mass': active_sum, 'cone_max': k.amax(1),
                         'current_min_standardized_arm_full': arm.amin(1),
                         'current_min_standardized_arm_source_z': arm[:, indices][:, source_z].amin(1),
                         'uniform_tiny_fraction': posterior[:, tiny].sum(1),
                         'final_tiny_responsibility': responsibility[:, tiny].sum(1),
                         'final_outer_responsibility': responsibility[:, outer].sum(1),
                         'final_peak_responsibility': responsibility[:, peak],
                         'final_prediction':final_prediction,
                         'e1': e.e1, 'e2': e.e2, 'cp1': e.cpnum1, 'cp2': e.cpnum2}
                if variant == 'current' and truth:
                    for truth_key, truth_density in truth.items():
                        comp64 = compact.double()
                        predicted = comp64@truth_density
                        production_valid = torch.isfinite(total)&(total>0)
                        production_valid &= (1/(norm**2).sum(1))>=1
                        zero32 = (predicted<=0)&production_valid
                        b_truth = (br[:, indices].double()*fraction.double())@truth_density
                        # The actual production floor acts on float32 T@x. Its
                        # effect is separate from the mathematical likelihood.
                        floor_contribution = comp64/predicted.clamp_min(1e-12)[:,None]
                        floor_contribution[~production_valid] = 0
                        floored_gradients[truth_key] += floor_contribution.sum(0)*(n/len(chosen))
                        if bool(zero32.any()):
                            ee = subset(e, zero32)
                            ee64 = PreparedComptonEvents(**{name:(value.double() if value.is_floating_point() else value)
                                for name, value in ee.__dict__.items()})
                            k64 = build_compton_cone_weights(ee64, coords.double(), settings)
                            r64 = k64*br[zero32].double()
                            n64 = r64/r64.sum(1)[:, None]
                            comp64[zero32] = n64[:, indices]*fraction.double()
                            predicted = comp64@truth_density
                        positive = (predicted>0)&production_valid
                        contribution = torch.zeros_like(comp64)
                        contribution[positive] = comp64[positive]/predicted[positive, None]
                        gradients[truth_key] += contribution.sum(0)*(n/len(chosen))
                        stats[truth_key+'_peak_gradient_per_event'] = contribution[:, peak]/sensitivity[peak]
                        stats[truth_key+'_zero_response_float32'] = zero32.float()
                        stats[truth_key+'_zero_response_after_float64'] = ((predicted<=0)&production_valid).float()
                        stats[truth_key+'_zero_B_template'] = ((b_truth<=0)&production_valid).float()
                        stats[truth_key+'_peak_gradient_production_floor'] = floor_contribution[:,peak]/sensitivity[peak]
                values = {key: value.cpu().numpy() for key, value in stats.items()}
                for i, sampled_index in enumerate(chosen_batch):
                    row = {'view': view+1, 'pre_kernel_index': int(sampled_index),
                           'variant': variant, 'stratum_weight': n/len(chosen)}
                    row.update({key: float(value[i]) for key, value in values.items()})
                    rows.append(row)
            del variants, current, old, arm, br, raw_response, norm, compact, posterior, responsibility
        print(f'view {view+1}/20: {n} events; {time.monotonic()-start:.1f}s', flush=True)
    with (args.output/'events.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for variant in {r['variant'] for r in rows}:
        all_rr = [r for r in rows if r['variant']==variant]
        rr = [r for r in all_rr if r['raw_sum']>0 and np.isfinite(r['raw_sum']) and r['ellipse_row_mass']>0]
        w = np.array([r['stratum_weight'] for r in rr])
        summary[variant] = {'sample':len(all_rr), 'valid_sample':len(rr),
            'invalid_raw_rows':sum(not (r['raw_sum']>0 and np.isfinite(r['raw_sum'])) for r in all_rr)}
        for key in ['effective_support_full','effective_support_active','raw_sum', 'cone_max',
                    'ellipse_row_mass','current_min_standardized_arm_full','current_min_standardized_arm_source_z']:
            a = np.array([r[key] for r in rr])
            summary[variant][key+'_quantiles'] = np.quantile(a, [0,.01,.05,.5,.95,.99,1]).tolist()
        for name, predicate in {'full_support_lt50': lambda r:r['effective_support_full']<50,
                                 'active_support_lt50':lambda r:r['effective_support_active']<50,
                                 'arm_full_gt3':lambda r:r['current_min_standardized_arm_full']>3,
                                 'arm_source_z_gt3':lambda r:r['current_min_standardized_arm_source_z']>3,
                                 'sum_lt400':lambda r:r['e1']+r['e2']<.4}.items():
            mask = np.array([predicate(r) for r in rr])
            resp = np.array([r['final_tiny_responsibility'] for r in rr])
            summary[variant][name] = {'count':int(mask.sum()), 'weighted_fraction':float(w[mask].sum()/w.sum()),
                'fraction_of_tiny_responsibility':float((w[mask]*resp[mask]).sum()/(w*resp).sum())}
        for key in ['uniform_tiny_fraction','final_tiny_responsibility','final_outer_responsibility','final_peak_responsibility']:
            summary[variant][key+'_weighted_mean'] = float(np.average([r[key] for r in rr], weights=w))
    gradient_summary = {}
    for key, grad in gradients.items():
        gain = grad.cpu().numpy()/sensitivity
        np.save(args.output/(key+'_truth_score.npy'), gain)
        gradient_summary[key] = {'peak_gain':float(gain[peak]),
            'peak_gain_production_floor':float(floored_gradients[key][peak].cpu())/sensitivity[peak],
            'tiny_cell_median_gain':float(np.median(gain[geo['ellipse_fraction'][active]<.1])),
            'active_gain_quantiles':np.quantile(gain,[0,.5,.9,.99,1]).tolist()}
        rr = [r for r in rows if r['variant']=='current']
        ww = [r['stratum_weight'] for r in rr]
        for suffix in ['zero_response_float32','zero_response_after_float64','zero_B_template']:
            gradient_summary[key][suffix+'_fraction'] = float(np.average([r[key+'_'+suffix] for r in rr],weights=ww))
            gradient_summary[key][suffix+'_count'] = int(sum(r[key+'_'+suffix] for r in rr))
        gradient_summary[key]['scope'] = 'Finite-likelihood events only; if any zero-response event exists, full truth likelihood is invalid and this is not a complete KKT test.'
    if device == 'cuda':
        torch.cuda.synchronize()
    report = {'purpose':'response diagnostic only; no MLEM or regularization',
        'sample_seed':args.seed, 'per_view':views, 'summary':summary,
        'geometry_sha256':digest(gp), 'baseline_run_manifest_sha256':digest(result/'run_manifest.json'),
        'script_sha256':digest(__file__), 'kernel_sha256':digest(HERE/'compton_event_response.py'),
        'factor_manifest_sha256':digest(factors/'factor_manifest.json'),
        'adapter_check':adapter_check,
        'baseline_compton_image_sha256':digest(result/'Image_440_ComptonOnly_full.float32'),
        'truth_sha256':digest(args.truth) if args.truth else None,
        'truth_score_diagnostic':gradient_summary,
        'peak_full_index':int(active[peak]), 'peak_coordinates_mm':geo['coordinates_mm'][active[peak]].tolist(),
        'peak_fraction':float(fraction[peak]), 'elapsed_seconds':time.monotonic()-start,
        'device': device, 'threads': args.threads,
        'max_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,
        'host_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
        'limits':['Stratified random sample, not all event kernels.',
                  'Alternative kernels evaluated at unchanged baseline image; not new ML solutions.',
                  'ARM source test constrains z only, not full NEMA body.',
                  'Legacy variant holds present energy resolution, data and B fixed; not an old-run reproduction.']}
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'done':True,'seconds':report['elapsed_seconds']}), flush=True)


if __name__=='__main__':
    main()
