"""Independent precise R2 measure; never rewrites the baseline geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from geometry import grid
from compton_boundary_quadrature import angular_breakpoints, cell_quadrature


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_coordinates(regenerated, frozen):
    # libm trigonometric evaluations differ by a few ULP across Windows/Linux.
    # This checks physical coordinates; immutable input identity is its SHA.
    np.testing.assert_allclose(regenerated,frozen,rtol=0,atol=1e-10)


def precise_cell_volume(cell, semi_axes=(250., 150.), height=3.):
    lo, hi, start, end = cell; a, b = semi_axes
    breaks = angular_breakpoints(cell, semi_axes)
    def integrand(theta):
        radius2 = 1 / ((np.cos(theta)/a)**2 + (np.sin(theta)/b)**2)
        return max(0., min(hi*hi, radius2) - lo*lo) * height/2
    values = [quad(integrand, u, v, epsabs=1e-10, epsrel=1e-12)
              for u, v in zip(breaks[:-1], breaks[1:])]
    return sum(v for v, _ in values), sum(e for _, e in values)


def prepare(geometry_path, config_path, output):
    cfg = json.loads(config_path.read_text()); geo = np.load(geometry_path)
    coords, cells, _ = grid(cfg); n = cfg['points_per_layer']
    validate_coordinates(coords, geo['coordinates_mm'])
    old = geo['ellipse_fraction'][:n]; full = geo['cell_volume_mm3'][:n]
    partial = np.flatnonzero((old > 1e-12) & (old < 1-1e-12))
    exact = full*old; cases = []
    for cell in partial:
        value, error = precise_cell_volume(cells[cell], tuple(cfg['semi_axes_mm']), cfg['z_spacing_mm'])
        coarse = cell_quadrature(cells[cell], 1.5, 16, 4, 4, semi_axes=tuple(cfg['semi_axes_mm']))[1].sum()
        fine = cell_quadrature(cells[cell], 1.5, 32, 4, 4, semi_axes=tuple(cfg['semi_axes_mm']))[1].sum()
        exact[cell] = value
        cases.append(dict(cell=int(cell), old_fraction=float(old[cell]), precise_volume_mm3=value,
            scipy_absolute_error_bound_mm3=error, frozen_relative_error=float(full[cell]*old[cell]/value-1),
            quadrature16_relative_error=float(coarse/value-1), quadrature32_relative_error=float(fine/value-1)))
    fraction = exact/full
    if not np.array_equal(np.flatnonzero(np.tile(fraction,40) > 1e-13), geo['active_indices']):
        raise ValueError('Precise overlap changes active identities; hold for review')
    theoretical = np.pi*np.prod(cfg['semi_axes_mm'])*cfg['height_mm']
    error = float(exact.sum()*40/theoretical-1)
    if abs(error) > 1e-10 or any(abs(c['quadrature32_relative_error']) > .001 for c in cases):
        raise ValueError('Precise overlap volume contract failed')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'measure.npz', effective_volume_mm3=np.tile(exact,40),
        ellipse_fraction=np.tile(fraction,40), active_indices=geo['active_indices'],
        partial_indices=np.concatenate([partial+layer*n for layer in range(40)]))
    result = dict(status='PASSED', baseline_geometry_sha256=sha(geometry_path), config_sha256=sha(config_path),
        measure_sha256=sha(output/'measure.npz'), original_geometry_unchanged=True, active_points=82040,
        partial_points=len(partial)*40, ellipse_volume_mm3=float(exact.sum()*40),
        theoretical_volume_mm3=float(theoretical), relative_volume_error=error,
        maximum_quadrature32_relative_error=max(abs(c['quadrature32_relative_error']) for c in cases),
        normalization='Physical object intersection dV; no duplicate fraction or full volume.',
        single_photon_factors_unchanged=True, cells=cases)
    (output/'measure_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=prepare(a.geometry,a.config,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
