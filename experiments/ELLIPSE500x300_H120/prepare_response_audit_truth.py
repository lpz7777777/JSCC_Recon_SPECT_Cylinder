"""Integrate the actual transported 3-mm cuboid truth over polar cells for audit.

Two deterministic r-squared/angle quadratures expose mapping discretization.
The z grids coincide exactly. This writes diagnostic inputs, never source truth.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from geometry import grid

HERE = Path(__file__).resolve().parent


def main():
    truth_path = HERE/'generated/NEMA_Body_H60/truth_3mm.npz'
    truth = np.load(truth_path)
    geom = np.load(HERE/'generated/Geometry/geometry.npz')
    _, cells, _ = grid(json.loads((HERE/'config.json').read_text()))
    assert np.allclose(np.unique(geom['coordinates_mm'][:, 2]), truth['z_mm'])
    source = truth['activity_440_zyx'].astype(np.float64)
    target_integral = float(source.sum()*27)
    arrays, stats = {}, {}
    for order in (32, 64):
        centers = (np.arange(order)+.5)/order
        parts = []
        for start in range(0, len(cells), 64):
            lo, hi, a, b = cells[start:start+64].T
            radius = np.sqrt(lo[:, None, None]**2+(hi**2-lo**2)[:, None, None]*centers[None, :, None])
            theta = a[:, None, None]+(b-a)[:, None, None]*centers[None, None, :]
            ix = np.floor((radius*np.cos(theta)+252)/3).astype(int)
            iy = np.floor((radius*np.sin(theta)+150)/3).astype(int)
            inside = (ix>=0)&(ix<168)&(iy>=0)&(iy<100)
            value = source[:, np.clip(iy,0,99), np.clip(ix,0,167)]*inside
            parts.append(value.mean(axis=(-1,-2)))
        density = np.concatenate(parts, axis=1).reshape(-1)
        integrated = float(density@geom['cell_volume_mm3'])
        # Total number of emitted 440-keV gamma photons, from transport closure.
        arrays[f'density_{order}'] = density*(3530946267/target_integral)
        stats[str(order)] = {'integral':integrated,
                            'relative_integral_error':integrated/target_integral-1}
    output = HERE/'generated/Diagnostics/process_list_audit'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'polar_truth.npz', **arrays)
    report = {'source':'actual piecewise-constant transported cuboid activity_440_zyx',
              'truth_sha256':hashlib.sha256(truth_path.read_bytes()).hexdigest(),
              'quadratures':stats, 'actual_440_primaries':3530946267,
              'polar_truth_sha256':hashlib.sha256((output/'polar_truth.npz').read_bytes()).hexdigest(),
              'note':'Mapping discretization check, not a proof of response accuracy.'}
    (HERE/'reports/NEMA_Body_H60/process_list_audit/truth_mapping.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__=='__main__':
    main()
