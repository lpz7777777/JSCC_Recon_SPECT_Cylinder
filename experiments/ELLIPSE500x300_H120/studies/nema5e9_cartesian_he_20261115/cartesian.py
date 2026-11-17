"""Native HE-aligned density basis, with a separate Cartesian detector field.

No polar response or reconstructed image is interpolated. The detector field
is newly calculated at 4 mm centres; only its view rotation is interpolated.
The same explicit matrix serves forward, transpose and sensitivity.
"""
from pathlib import Path
import math
import numpy as np
from scipy.integrate import quad
import torch

SHAPE = (126, 126, 34)  # native storage is xyz, exactly as delivered HE
SPACING = 4.
FIELD_SHAPE_ZYX = (30, 130, 130)


def axes():
    return tuple((np.arange(n)-(n-1)/2)*SPACING for n in SHAPE)


def area_fraction(x, y, a, b, spacing=4.):
    """Independent exact cell/ellipse area integration, no centre-only mask."""
    result = np.zeros((len(x), len(y)), dtype=np.float64)
    for i, xc in enumerate(x):
        xl, xh = max(-a, xc-spacing/2), min(a, xc+spacing/2)
        if xh <= xl:
            continue
        for j, yc in enumerate(y):
            yl, yh = yc-spacing/2, yc+spacing/2
            def length(xx):
                yy = b*math.sqrt(max(0., 1-(xx/a)**2))
                return max(0., min(yh, yy)-max(yl, -yy))
            points = []
            for yy in (yl, yh):
                if abs(yy) <= b:
                    xx = a*math.sqrt(1-(yy/b)**2)
                    points.extend(q for q in (-xx, xx) if xl < q < xh)
            result[i, j] = quad(length, xl, xh, points=sorted(set(points)),
                                epsabs=1e-9, epsrel=1e-11)[0]/spacing**2
    return result


def field_coordinates():
    xy = np.arange(130)*4.-258.
    z = np.arange(30)*4.-58.
    zz, yy, xx = np.meshgrid(z, xy, xy, indexing='ij')
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


class Geometry:
    def __init__(self, file, device='cpu'):
        with np.load(file) as f:
            self.fractions_xyz = f['fraction_xyz'].copy()
            self.active_numpy = f['active_indices'].copy()
            self.coords_numpy = f['active_coordinates_mm'].copy()
            self.reference_fraction_numpy = f['reference_circle_fraction_zyx'].ravel().copy()
        self.active_count = len(self.active_numpy)
        self.full_count = int(np.prod(SHAPE))
        self.views = 20
        self.fraction = torch.tensor(self.fractions_xyz.ravel()[self.active_numpy],
                                     dtype=torch.float32, device=device)
        self.reference_fraction = torch.tensor(self.reference_fraction_numpy,
                                               dtype=torch.float64, device=device)
        self.indices, self.weights = [], []
        for view in range(self.views):
            indices, weights = stencil(self.coords_numpy, view)
            self.indices.append(torch.tensor(indices, dtype=torch.long, device=device))
            self.weights.append(torch.tensor(weights, dtype=torch.float32, device=device))

    def compact(self, rows, view):
        """Rows contain full-cell volume exactly once; overlap applied here once."""
        idx, w = self.indices[view], self.weights[view]
        answer = rows[:, idx[0]]*w[0]
        for corner in range(1, 4):
            answer = answer+rows[:, idx[corner]]*w[corner]
        return answer*self.fraction

    def full_image(self, active):
        full = np.zeros(self.full_count, dtype=np.float32)
        full[self.active_numpy] = np.asarray(active).reshape(-1)
        return full.reshape(SHAPE)


def stencil(coordinates, view):
    """Object to detector rotation: -view*18deg, matching frozen inverse map."""
    phi = view*math.pi/10
    c, s = math.cos(phi), math.sin(phi)
    x, y, z = np.asarray(coordinates).T
    u = (x*c+y*s+258)/4
    v = (y*c-x*s+258)/4
    iz = np.rint((z+58)/4).astype(np.int64)
    if not np.allclose(z, iz*4-58, atol=1e-10, rtol=0):
        raise ValueError('Unexpected HE active z centres')
    if min(u.min(), v.min()) < 0 or max(u.max(), v.max()) >= 129 or iz.min()<0 or iz.max()>29:
        raise ValueError('Rotation would extrapolate the detector field')
    ix, iy = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    tx, ty = u-ix, v-iy
    base = (iz*130+iy)*130+ix
    return np.stack((base, base+1, base+130, base+131)), np.stack(
        ((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty))


def energy_rows(prepared, coords, response_memmap, law, reference_fraction,
                tile=32768, energy_module=None):
    """Original v5 likelihood/KN; new Cartesian quadrature and row normalizer.

    Preserve measured energies, widths and selection. Source tiling bounds GPU
    temporaries; normalization happens once across the complete reference disk.
    Reference fraction is used in the normalizer, not multiplied into the field.
    The image's ellipse fraction is subsequently applied by Geometry.compact.
    """
    if energy_module is None:
        import compton_energy_probability_v5 as energy_module
    n = len(coords)
    device = coords.device
    crystal = (prepared.cpnum1-1).cpu().numpy()
    logweights = torch.empty((prepared.count, n), dtype=torch.float64, device=device)
    e = prepared.e1.double()
    kn = energy_module.E0/(energy_module.E0-e)+(energy_module.E0-e)/energy_module.E0
    for start in range(0, n, tile):
        end = min(n, start+tile)
        logp, beta = energy_module.energy_log_density(prepared, coords[start:end], law,
            node_chunk=16, backend='tail16_mid2', gaussian_float32=True)
        b = torch.as_tensor(np.array(response_memmap[crystal, start:end], copy=True), device=device)
        logweights[:, start:end] = logp+(kn[:, None]-beta.sin().square()).log()+b.double().log()
    normalizer = torch.logsumexp(logweights+reference_fraction.log()[None, :], dim=1, keepdim=True)
    if not bool(torch.isfinite(normalizer).all()):
        raise ValueError('Frozen event has zero Cartesian reference response; cannot discard it')
    result = (logweights-normalizer).exp().float()
    integral = (result.double()*reference_fraction).sum(1)
    if not bool(torch.isfinite(result).all()) or not bool(torch.allclose(integral,
            torch.ones_like(integral), rtol=1e-5, atol=1e-7)):
        raise ValueError('Cartesian event normalization failed')
    return result


class ViewResponse:
    """Bounded fixed view cache; no change to frozen MLEM update/save routines."""
    def __init__(self, rows, geometry, reserve_bytes=2<<30):
        self.full_rows, self.geometry = rows, geometry
        self.cached = {}
        if rows.is_cuda:
            free, total = torch.cuda.mem_get_info(rows.device)
            available = max(0, int(.8*total)-(total-free)-reserve_bytes)
        else:
            available = 64<<20  # small local tests only
        per_view = rows.shape[0]*geometry.active_count*rows.element_size()
        count = min(20, max(0, available//max(1, per_view)-2))
        for v in range(count):
            self.cached[v] = geometry.compact(rows, v)

    def matrix(self, view):
        return self.cached[view] if view in self.cached else self.geometry.compact(self.full_rows, view)

    def sensitivity(self):
        s = torch.zeros((self.geometry.active_count, 1), dtype=self.full_rows.dtype,
                        device=self.full_rows.device)
        for view in range(20):
            s += self.matrix(view).sum(0)[:, None]
        return s/20

    def close(self):
        if self.full_rows.is_cuda:
            torch.cuda.synchronize(self.full_rows.device)
        self.cached.clear()

