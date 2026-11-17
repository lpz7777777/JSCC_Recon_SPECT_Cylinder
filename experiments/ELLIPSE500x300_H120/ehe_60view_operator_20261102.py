"""60 distinct views on the retained unequal polar basis; linear angular response.

No interpolation of measurements or reconstructed images. Matrix columns on a
ring already contain the same cell volume. Interpolate only these columns at
theta-view*6deg, then apply object-frame ellipse fraction exactly once. Three
full-grid phase tables avoid interpolating 730MB of rows at every MLEM view.
"""
from pathlib import Path
import numpy as np
import torch
from ehe_common import digest, verify_files


def angular_maps(g, views=60):
    if views != 60:
        raise ValueError('Only the preregistered 60-view geometry is supported')
    xyz = g['coordinates_mm']; old = g['inverse_rotation']
    z0 = xyz[:, 2] == xyz[0, 2]; nxy = int(z0.sum())
    if (len(xyz), nxy, old.shape) != (132040, 3301, (132040, 20)):
        raise ValueError('Original complete geometry required')
    lo = np.empty((len(xyz), views), np.int32); hi = np.empty_like(lo)
    alpha = np.empty((len(xyz), views), np.float32)
    radii = np.rint(np.hypot(xyz[:nxy, 0], xyz[:nxy, 1])).astype(int)
    for radius in np.unique(radii):
        ring = np.flatnonzero(radii == radius); n = len(ring)
        if not np.array_equal(ring, np.arange(ring[0], ring[0]+n)):
            raise ValueError('Noncontiguous ring')
        # Integer division avoids floating-point wrap/rounding at exact nodes.
        numerator = np.arange(n)[:, None]*views - np.arange(views)[None, :]*n
        a = numerator//views; rem = numerator % views
        if radius == 0:
            rem[:] = 0  # The central disk has no angular coordinate.
        for z in range(len(xyz)//nxy):
            target = ring+z*nxy
            lo[target] = ring[0]+a % n+z*nxy
            hi[target] = ring[0]+(a+1) % n+z*nxy
            alpha[target] = rem/views
    if not np.array_equal(lo[:, ::3], old) or np.any(alpha[:, ::3]):
        raise ValueError('60-view original20 subset is not exactly identical')
    vol = g['cell_volume_mm3']
    if not (np.array_equal(vol[lo], np.broadcast_to(vol[:, None], lo.shape))
            and np.array_equal(vol[hi], np.broadcast_to(vol[:, None], hi.shape))):
        # Producer volumes may have harmless last-bit trig differences within a
        # ring, so require a tolerance far below the matrix precision.
        if not (np.allclose(vol[lo], vol[:, None], rtol=1e-13, atol=1e-10)
                and np.allclose(vol[hi], vol[:, None], rtol=1e-13, atol=1e-10)):
            raise ValueError('Angular interpolation crosses unequal cell volumes')
    return lo, hi, alpha


class Geometry60:
    def __init__(self, path, device):
        with np.load(path) as g:
            lo, hi, alpha = angular_maps(g)
            active = g['active_indices']; frac = g['ellipse_fraction']
            old = g['inverse_rotation']
            if not np.array_equal(active, np.flatnonzero(frac > 1e-13)):
                raise ValueError('Original active domain differs')
            self.object_active = torch.tensor(active.astype(np.int64), device=device)
            self.fraction = torch.tensor(frac[active].astype(np.float32), device=device)
            self.indices = [torch.tensor(old[active, k].astype(np.int64), device=device) for k in range(20)]
            self.phase_maps = [(torch.tensor(lo[:, p].astype(np.int64), device=device),
                                torch.tensor(hi[:, p].astype(np.int64), device=device),
                                torch.tensor(alpha[:, p], device=device)) for p in (1, 2)]
            self.lo = [torch.tensor(lo[active, v].astype(np.int64), device=device) for v in range(60)]
            self.hi = [torch.tensor(hi[active, v].astype(np.int64), device=device) for v in range(60)]
            self.alpha = [torch.tensor(alpha[active, v], device=device) for v in range(60)]
        self.views = 60; self.active_count = len(active); self.full_count = len(frac)


class Response60:
    def __init__(self, rows, geometry):
        self.full_rows = rows; self.geometry = geometry; self.phases = [rows]
        for lo, hi, a in geometry.phase_maps:
            phase = torch.index_select(rows, 1, lo)
            phase.mul_(1-a)
            other = torch.index_select(rows, 1, hi); other.mul_(a)
            phase.add_(other); del other
            self.phases.append(phase)

    def matrix(self, view):
        if not 0 <= view < 60:
            raise ValueError('View out of range')
        return torch.index_select(self.phases[view % 3], 1,
                                  self.geometry.indices[view//3])*self.geometry.fraction

    def sensitivity(self):
        s = torch.zeros((self.geometry.active_count, 1), device=self.full_rows.device)
        for v in range(60):
            s += self.matrix(v).sum(dim=0)[:, None]
        return s/60


def load_response(spec, geometry, device, require_positive=True):
    root = Path(spec['root'])
    if digest(root/'factor_manifest.json') != spec['manifest_sha256']:
        raise ValueError('Frozen matrix manifest differs')
    verify_files(root, spec['files'])
    if digest(root/'whole_geometry.npz') != spec['geometry_sha256']:
        raise ValueError('Frozen matrix geometry differs')
    raw = np.memmap(root/'SysMat_polar', '<f4', mode='r', shape=(132040, 2312))
    response = Response60(torch.tensor(np.array(raw.T), device=device), geometry)
    s = response.sensitivity()
    full_s = torch.tensor(np.fromfile(root/'S_full.float64', '<f8'), device=device)
    expected = torch.zeros_like(s, dtype=torch.float64)
    for v in range(60):
        a = geometry.alpha[v].double()
        expected += ((1-a)*full_s[geometry.lo[v]] + a*full_s[geometry.hi[v]])[:, None]/60
    expected *= geometry.fraction[:, None]
    error = float(torch.linalg.vector_norm(s.double()-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-300))
    if error > 1e-5 or not bool(torch.isfinite(s).all()) or (require_positive and bool((s <= 1e-12).any())):
        raise ValueError('60-view matrix and independent matched S differ: '+str(error))
    return response, s, error


def local_checks(path):
    """Meaningful CPU regression: all indices, phase composition, adjoint/dose/S."""
    with np.load(path) as g:
        lo, hi, a = angular_maps(g); old = g['inverse_rotation']; active = g['active_indices']
        frac = g['ellipse_fraction'][active]; vol = g['cell_volume_mm3']
    rng = np.random.default_rng(6020261102)
    rows = rng.random((3, 132040)); x = rng.random(len(active)); y = rng.random(3)
    phase = [rows]+[rows[:, lo[:, p]]*(1-a[:, p])+rows[:, hi[:, p]]*a[:, p] for p in (1, 2)]
    mass_error = 0.; phase_error = 0.; adjoint_error = 0.
    s = np.zeros(len(active)); expected_s = np.zeros_like(s)
    row_s = rows.sum(axis=0)
    for v in range(60):
        m = phase[v % 3][:, old[active, v//3]]*frac
        direct = (rows[:, lo[active, v]]*(1-a[active, v])+rows[:, hi[active, v]]*a[active, v])*frac
        phase_error = max(phase_error, float(np.max(abs(m-direct))))
        adjoint_error = max(adjoint_error, float(abs(y@(m@x)-(m.T@y)@x)/abs(y@(m@x))))
        s += m.sum(axis=0)/60
        expected_s += (row_s[lo[active, v]]*(1-a[active, v])+row_s[hi[active, v]]*a[active, v])*frac/60
        weights = np.zeros(132040)
        np.add.at(weights, lo[active, v], x*frac*(1-a[active, v])/60)
        np.add.at(weights, hi[active, v], x*frac*a[active, v]/60)
        expected_mass = (x*frac)@vol[active]/60
        mass_error = max(mass_error, float(abs(weights@vol-expected_mass)/expected_mass))
        assert np.allclose(rows@weights, m@x/60, rtol=1e-13, atol=1e-10)
    s_error = float(np.linalg.norm(s-expected_s)/np.linalg.norm(s))
    assert phase_error < 1e-13 and adjoint_error < 1e-13 and mass_error < 1e-7 and s_error < 1e-13, (phase_error, adjoint_error, mass_error, s_error)
    return dict(passed=True,views=60,old20_subset_exact=True,phase_composition_max_abs=phase_error,
                max_adjoint_relative_error=adjoint_error,max_source_mass_relative_error=mass_error,
                independent_s_relative_L2=s_error,full_input_GPU_prevalidation=False,
                approximation='Periodic linear interpolation of same-ring volume-weighted response columns')
