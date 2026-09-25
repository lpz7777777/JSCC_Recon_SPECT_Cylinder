"""Check full-grid noiseless fixed points and unfiltered contrast convergence.

Uses the frozen polar truth and fractional rod masks, never image-derived ROIs.
Fixed-point checks use the production local MLEM update and exact true cross-talk.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_local_cntstat import load_full_sysmat, compute_sensitivity_local
from recon_osem_local_cntstat import osem_bin_mode_local


def weighted_mean(x, weights):
    return float(np.dot(x, weights) / weights.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'generated')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = args.root.resolve()
    factors = root / 'Factors'
    truth = np.load(root / 'Truth_Contrast_1e9.npz')
    rods = json.loads((root / 'Simulation/Contrast_truth.json').read_text())['rods']
    reference = factors / '440keV_RotateNum20'
    volume = np.fromfile(reference / 'polar_cell_volume_mm3.float64', dtype='<f8')
    coords = np.loadtxt(reference / 'coor_polar_full.csv', delimiter=',')
    rot = torch.from_numpy(np.loadtxt(reference / 'RotMat_full.csv', delimiter=',', dtype=np.int64))
    inverse = torch.from_numpy(np.loadtxt(reference / 'RotMatInv_full.csv', delimiter=',', dtype=np.int64))
    n = len(volume)
    report = {'scope': 'Exact-model numerical consistency and convergence; not physical validation',
              'fixed_points': {}, 'histories': {},
              'roi': 'Fractional rod masks; background r<=120, same rod z interval, excluding all rod cells'}
    # Independent NumPy forward projections are compared to production Torch MLEM.
    with torch.no_grad():
        for energy in (440, 218):
            factor = factors / f'{energy}keV_RotateNum20'
            matrix, _ = load_full_sysmat(str(factor / 'SysMat_polar'), n, 1.0)
            sensitivity = compute_sensitivity_local(matrix, inverse, 20).to(args.device)
            matrix = matrix.to(args.device)
            image = torch.tensor(truth[f'rho{energy}'], dtype=torch.float32, device=args.device).reshape(-1, 1)
            projection = np.loadtxt(root / f'GenProj_Contrast_1e9/Noiseless/{energy}keV_RotateNum20/CntStat_Closure_1e9.csv', delimiter=',')
            y = torch.tensor(projection.T, dtype=torch.float32, device=args.device)
            bg = None if energy == 440 else torch.tensor(np.load(root / 'GenProj_Contrast_1e9/cross_projection.npy').T, dtype=torch.float32, device=args.device)
            updated = osem_bin_mode_local([[matrix]], [[y]], [rot.to(args.device)],
                                          [inverse.to(args.device)], image, sensitivity, [[bg]])
            delta = (updated - image).cpu().numpy().ravel()
            t = truth[f'rho{energy}']
            report['fixed_points'][str(energy)] = {
                'weighted_relative_l2': float(np.sqrt(np.dot(delta**2, volume) / np.dot(t*t, volume))),
                'max_relative_voxel_change': float(np.max(np.abs(delta) / np.maximum(t, 1e-12))),
            }
            del matrix, image, updated, y, bg, sensitivity
    all_rods = np.maximum.reduce([truth[f'mask_rod_{i+1}'] for i in range(len(rods))])
    for kind in ('Noiseless', 'Poisson'):
        paths = list((root / 'ClosedLoop' / kind).rglob('run_manifest.json'))
        if len(paths) != 1:
            raise ValueError(f'Expected one {kind} manifest, found {len(paths)}')
        manifest = json.loads(paths[0].read_text())
        report['histories'][kind] = {}
        for task in manifest['tasks']:
            energy = {'Direct440': 440, 'CrossTalkCorrected218': 218}.get(task['type'])
            if energy is None:
                continue
            frames = np.fromfile(paths[0].parent / task['history_file'], dtype='<f4').reshape(task['history_shape'])
            if frames.shape[1] != n or not np.isfinite(frames).all() or np.any(frames < 0):
                raise ValueError('Invalid history')
            t = truth[f'rho{energy}']
            metrics = []
            for k, x in enumerate(frames):
                row = {'iteration': (k+1)*task['save_iter_step'],
                       'weighted_relative_l2': float(np.sqrt(np.dot((x-t)**2, volume)/np.dot(t*t, volume))),
                       'rods': []}
                for i, rod in enumerate(rods):
                    if rod['energy'] != energy:
                        continue
                    roi = truth[f'mask_rod_{i+1}'] * volume
                    background = ((np.hypot(coords[:, 0], coords[:, 1]) <= 120)
                                  & (np.abs(coords[:, 2]-rod['center_mm'][2]) <= rod['height_mm']/2)
                                  & (all_rods == 0)) * volume
                    hot, back = weighted_mean(x, roi), weighted_mean(x, background)
                    true_contrast = weighted_mean(t, roi) / weighted_mean(t, background)-1
                    row['rods'].append({'index': i+1, 'diameter_mm': 2*rod['radius_mm'],
                                        'z_mm': rod['center_mm'][2], 'crc': (hot/back-1)/true_contrast})
                metrics.append(row)
            report['histories'][kind][str(energy)] = metrics
    out = root / 'closedloop_diagnostics.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['fixed_points'], indent=2))
    print(out)
    if any(v['max_relative_voxel_change'] > 2e-5 for v in report['fixed_points'].values()):
        raise ValueError('Noiseless truth is not a numerical fixed point; inspect operator before production')


if __name__ == '__main__':
    main()
