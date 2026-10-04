"""Bounded R2 object/full-cell integration using the unchanged shared kernel.

Event acceptance belongs to the frozen R1 scan, never to this integration
layer. The A provider selects a field for an entire reference cell before
evaluating either domain. Production requires its independent accuracy gate.
"""
from collections import OrderedDict
import time
import numpy as np
import torch
from compton_boundary_quadrature import cell_quadrature, rotate_to_detector
from compton_event_response import build_compton_cone_weights


def reference_cell_bounds(cell, z, view):
    """Exact axis-aligned bounds of a rotated annular sector, including faces."""
    lo, hi, start, end = cell
    start -= view * np.pi / 10
    end -= view * np.pi / 10
    angles = [start, end]
    for k in range(int(np.floor(start / (np.pi / 2))),
                   int(np.ceil(end / (np.pi / 2))) + 1):
        t = k * np.pi / 2
        if start < t < end:
            angles.append(t)
    points = np.array([[r*np.cos(t), r*np.sin(t)]
                       for r in (lo, hi) for t in angles])
    return np.array([points[:, 0].min(), points[:, 1].min(), z-1.5]), np.array(
        [points[:, 0].max(), points[:, 1].max(), z+1.5])


def array_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(v) for v in value)
    if isinstance(value, dict):
        return sum(array_bytes(v) for v in value.values())
    return 0


class OverlapIntegrator:
    def __init__(self, cells, coordinates, partial_indices, points_per_layer,
                 provider, settings, *, node_chunk=32768, cells_per_block=8,
                 cache_bytes=128 << 20, diagnostic_only=False,
                 kernel=build_compton_cone_weights):
        if settings.geometry_mode != 'stable_float64':
            raise ValueError('R2 requires the frozen stable geometry kernel')
        if min(node_chunk, cells_per_block) < 1 or cache_bytes < 0:
            raise ValueError('Invalid bounded integration allocation')
        if not diagnostic_only:
            provider.assert_production_ready()
        self.cells = np.asarray(cells, dtype=float)
        self.coords = np.asarray(coordinates, dtype=float)
        self.partial = np.asarray(partial_indices, dtype=np.int64)
        self.n = int(points_per_layer)
        if self.cells.shape != (self.n, 4) or self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError('Wrong physical cell geometry')
        if (len(self.partial) == 0 or np.any(np.diff(self.partial) <= 0)
                or self.partial[0] < 0 or self.partial[-1] >= len(self.coords)):
            raise ValueError('Partial identities must be nonempty, unique and sorted')
        self.provider, self.settings, self.kernel = provider, settings, kernel
        self.node_chunk, self.cells_per_block = int(node_chunk), int(cells_per_block)
        self.limit = int(cache_bytes)
        self.cache, self.cache_size = OrderedDict(), 0
        self.diagnostic_only = bool(diagnostic_only)
        self.statistics = {}

    def _block(self, view, order, offset):
        key = (int(view), tuple(order), int(offset))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        nodes, weights, slots, fields = [], [], [], []
        for local, index in enumerate(self.partial[offset:offset+self.cells_per_block]):
            cell, z = self.cells[index % self.n], self.coords[index, 2]
            # The same selected A field serves object and full reference.
            source = self.provider.choose(reference_cell_bounds(cell, z, view))
            for domain in (0, 1):
                p, w = cell_quadrature(cell, z, *order, ellipse=domain == 0)
                nodes.append(rotate_to_detector(p, view)); weights.append(w)
                slots.append(np.full(len(w), 2*local+domain, dtype=np.int32))
                fields.append(np.full(len(w), source, dtype=np.int32))
        p, w = np.concatenate(nodes), np.concatenate(weights)
        s, fields = np.concatenate(slots), np.concatenate(fields)
        compiled = self.provider.cache(p, fields)
        entry = (p, w, s, compiled)
        size = array_bytes(entry)
        if size <= self.limit:
            while self.cache and self.cache_size+size > self.limit:
                _, old = self.cache.popitem(last=False)
                self.cache_size -= array_bytes(old)
            self.cache[key] = entry; self.cache_size += size
        return entry

    def integrate(self, prepared, view, order=(16, 8, 8), *, progress=None):
        if not 0 <= view < 20 or len(order) != 3 or min(order) < 1:
            raise ValueError('Invalid view or quadrature order')
        if prepared.count < 1:
            raise ValueError('Empty accepted event block')
        start = time.monotonic(); total_nodes = 0
        integrals = np.zeros((prepared.count, len(self.partial), 2), dtype=np.float64)
        cp = prepared.cpnum1.detach().cpu().numpy().astype(int)-1
        unique = np.unique(cp)
        maximum_chunk = 0; field_counts = {}
        for offset in range(0, len(self.partial), self.cells_per_block):
            p, w, slots, compiled = self._block(view, order, offset)
            count = min(self.cells_per_block, len(self.partial)-offset)
            result = np.zeros((prepared.count, 2*count), dtype=np.float64)
            a_rows = {c: self.provider.evaluate(c, compiled) for c in unique}
            for c, values in a_rows.items():
                if values.shape != w.shape or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('Invalid sampled A; no event may be removed')
            for lo in range(0, len(w), self.node_chunk):
                hi = min(lo+self.node_chunk, len(w)); maximum_chunk = max(maximum_chunk, hi-lo)
                xyz = torch.as_tensor(p[lo:hi], dtype=torch.float64, device=prepared.e1.device)
                k = self.kernel(prepared, xyz, self.settings).double().cpu().numpy()
                if k.shape != (prepared.count, hi-lo) or not np.isfinite(k).all() or np.any(k < 0):
                    raise ValueError('Invalid shared event kernel')
                for event, crystal in enumerate(cp):
                    result[event] += np.bincount(slots[lo:hi],
                        weights=k[event]*a_rows[crystal][lo:hi]*w[lo:hi], minlength=2*count)
            integrals[:, offset:offset+count] = result.reshape(prepared.count, count, 2)
            total_nodes += len(w)
            for name, value in self.provider.field_counts(compiled).items():
                field_counts[name] = field_counts.get(name, 0)+int(value)
            if progress is not None:
                progress(dict(completed_cells=offset+count, total_cells=len(self.partial),
                              elapsed_seconds=time.monotonic()-start, nodes=total_nodes))
        if not np.isfinite(integrals).all() or np.any(integrals < 0):
            raise ValueError('Invalid complete integrals; HOLD')
        self.statistics = dict(events=prepared.count, partial_cells=len(self.partial),
            view_zero_based=int(view), order=list(order), quadrature_nodes=total_nodes,
            maximum_point_chunk=maximum_chunk, cache_bytes=self.cache_size,
            cache_limit_bytes=self.limit, provider_node_counts=field_counts,
            elapsed_seconds=time.monotonic()-start, diagnostic_only=self.diagnostic_only,
            event_filter_applied=False, density_volume_applied_once=True)
        device = prepared.e1.device
        return (torch.as_tensor(integrals[:, :, 0].copy(), device=device),
                torch.as_tensor(integrals[:, :, 1].copy(), device=device))
