"""Whole-reference-cell A selection, with an explicit production HOLD.

Fine patches are used only if they contain the entire rotated reference cell.
Object intersections and normalization references cannot choose different A.
The current local patches are diagnostic evidence, not a certified full FOV.
"""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
from compton_boundary_quadrature import GuardedPolarResponseField
from compton_cartesian_patch import CartesianPatch
from generate_compton_a_guard import digest, axes, PE_HASH, SCATTER_HASH
from build_compton_a_guard_field import combined


class GuardedCellCache:
    """Reuse the unchanged XY stencil across axial nodes and all z layers."""
    def __init__(self, guard, limit=64 << 20):
        self.guard, self.limit = guard, int(limit)
        self.xy = OrderedDict(); self.bytes = 0

    def compile(self, points, key, axial_order):
        if len(points)%axial_order:
            raise ValueError('Quadrature tensor-product ordering differs')
        transverse = points[::axial_order,:2]
        if not np.array_equal(points[:,:2].reshape(-1,axial_order,2),
                              np.broadcast_to(transverse[:,None,:],(len(transverse),axial_order,2))):
            raise ValueError('Axial grouping changes transverse nodes')
        if key not in self.xy:
            support=np.column_stack((transverse,np.zeros(len(transverse))))
            vertices,bary,_,_=self.guard.cache(support)
            size=vertices.nbytes+bary.nbytes
            while self.xy and self.bytes+size>self.limit:
                _,old=self.xy.popitem(last=False);self.bytes-=sum(a.nbytes for a in old)
            if size>self.limit:raise ValueError('XY cache entry exceeds bounded allocation')
            self.xy[key]=(vertices,bary);self.bytes+=size
        self.xy.move_to_end(key);vertices,bary=self.xy[key]
        z=points[:,2]
        if np.any(z<self.guard.z[0]-1e-10) or np.any(z>self.guard.z[-1]+1e-10):
            raise ValueError('Axial extrapolation forbidden')
        lower=np.clip(np.searchsorted(self.guard.z,z,side='right')-1,0,len(self.guard.z)-2)
        t=(z-self.guard.z[lower])/(self.guard.z[lower+1]-self.guard.z[lower])
        if np.any(t < -1e-10) or np.any(t > 1+1e-10):raise ValueError('Axial extrapolation forbidden')
        return (np.repeat(vertices,axial_order,axis=0),np.repeat(bary,axial_order,axis=0),
                lower,np.clip(t,0,1))


class CellResponseField:
    def __init__(self, field_folder, patch_folder=None, column_folder=None):
        field_folder = Path(field_folder)
        fm = json.loads((field_folder/'field_manifest.json').read_text())
        for name, expected in [('A_field.float64', fm['A_field_sha256']),
                               ('field_geometry.npz', fm['field_geometry_sha256'])]:
            if digest(field_folder/name) != expected:
                raise ValueError('Immutable A field differs')
        fg = np.load(field_folder/'field_geometry.npz')
        self.base = np.memmap(field_folder/'A_field.float64', mode='r', dtype='<f8', shape=tuple(fm['shape']))
        self.guard = GuardedPolarResponseField(fg['xy_mm'], int(fg['original_points']), fg['z_mm'])
        self.cell_cache = GuardedCellCache(self.guard)
        self.selected = fg['selected_raw_detectors'].copy()
        self.scales = np.asarray(fm['calibration_scales'])
        self.patches = []
        self.provenance = dict(field_manifest_sha256=digest(field_folder/'field_manifest.json'),
            baseline_geometry_sha256=fm['baseline_geometry_sha256'],
            baseline_B_sha256=fm['baseline_B_sha256'], patch_ready_sha256=None,
            policy='finest complete reference-cell box; fallback guarded A',
            calibration_refitted=False, original_data_read_only=True)
        if patch_folder is not None:
            patch_folder = Path(patch_folder)
            ready = json.loads((patch_folder/'patch_ready.json').read_text())
            if ready['status'] != 'COMPLETE_DIAGNOSTIC_ONLY':
                raise ValueError('Fine A patch is incomplete')
            if ready['pe_binary_sha256'] != PE_HASH or ready['scatter_binary_sha256'] != SCATTER_HASH:
                raise ValueError('Fine A physical model differs')
            for part in ready['parts']:
                name = part['spec']['name']; spec = part['spec']
                if json.loads((patch_folder/name/'complete.json').read_text()) != part:
                    raise ValueError('Fine A patch receipt changed')
                xyz = axes(spec)
                self.patches.append(dict(name=name, axes=xyz, interpolation=CartesianPatch(xyz),
                    values=combined(patch_folder, name), spacing=max(spec['spacing'])))
            self.patches.sort(key=lambda value: value['spacing'])
            self.provenance['patch_ready_sha256'] = digest(patch_folder/'patch_ready.json')
        if column_folder is not None:
            column_folder=Path(column_folder)
            gate=json.loads((column_folder/'column_gate.json').read_text())
            if gate['status']!='COLUMN_COMMON_POINTS_PASSED_GLOBAL_ACCURACY_HOLD':
                raise ValueError('Full-axis column common-point gate failed')
            if gate['pe_binary_sha256']!=PE_HASH or gate['scatter_binary_sha256']!=SCATTER_HASH:
                raise ValueError('Column physical model differs')
            part=gate['part'];spec=part['spec'];name=spec['name']
            if json.loads((column_folder/name/'complete.json').read_text())!=part:
                raise ValueError('Column receipt changed')
            xyz=axes(spec)
            self.patches.append(dict(name=name,axes=xyz,interpolation=CartesianPatch(xyz),
                values=combined(column_folder,name),spacing=max(spec['spacing'])))
            self.patches.sort(key=lambda value:value['spacing'])
            self.provenance['column_gate_sha256']=digest(column_folder/'column_gate.json')
        self.provenance['production_gate'] = 'HOLD_UNCERTIFIED_GLOBAL_A_ACCURACY'

    def assert_production_ready(self):
        raise ValueError('Local A patches cannot certify global R2 or S2 production')

    def choose(self, bounds):
        lo, hi = bounds
        for index, patch in enumerate(self.patches, 1):
            if all(a[0]-1e-10 <= l and h <= a[-1]+1e-10
                   for a, l, h in zip(patch['axes'], lo, hi)):
                return index
        return 0

    def cache(self, points, fields):
        groups = {}
        for source in np.unique(fields):
            indices = np.flatnonzero(fields == source)
            interp = self.guard if source == 0 else self.patches[source-1]['interpolation']
            groups[int(source)] = (indices, interp.cache(points[indices]))
        return dict(count=len(points), groups=groups)

    def cache_grouped(self, points, fields, segments):
        guard_indices=[];guard_parts=[[],[],[],[]];groups={}
        for start,end,key,source in segments:
            if source==0:
                idx=np.arange(start,end);guard_indices.append(idx)
                cache=self.cell_cache.compile(points[idx],key,key[-1][-1])
                for arrays,value in zip(guard_parts,cache):arrays.append(value)
        if guard_indices:
            groups[0]=(np.concatenate(guard_indices),tuple(np.concatenate(a) for a in guard_parts))
        for source in np.unique(fields):
            if source==0:continue
            idx=np.flatnonzero(fields==source)
            groups[int(source)]=(idx,self.patches[source-1]['interpolation'].cache(points[idx]))
        return dict(count=len(points),groups=groups)

    def evaluate(self, crystal, compiled):
        if not 0 <= crystal < len(self.base):
            raise ValueError('Unknown selected crystal')
        out = np.empty(compiled['count'], dtype=float)
        for source, (indices, cache) in compiled['groups'].items():
            if source == 0:
                out[indices] = self.guard.evaluate(self.base[crystal], cache)
            else:
                patch = self.patches[source-1]
                out[indices] = patch['interpolation'].evaluate(
                    patch['values'][self.selected[crystal]], cache)*self.scales[crystal]
        return out

    def field_counts(self, compiled):
        return {('guarded' if source == 0 else self.patches[source-1]['name']): len(indices)
                for source, (indices, _) in compiled['groups'].items()}
