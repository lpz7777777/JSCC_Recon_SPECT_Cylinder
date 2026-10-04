"""Diagnostic compact A reader with deterministic shared-vertex ownership.

All tiles share the original physical model and one auxiliary .75 mm lattice.
Missing samples never fall back to a different field or extrapolate. This
reader cannot certify global physical accuracy or authorize S2/imaging.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from generate_compton_a_guard import digest


class CompactCartesianTiles:
    origin=np.array([-258.,-258.,-60.])
    spacing=.75
    intervals=np.array([688,688,160])
    tile_intervals=16

    def __init__(self,specs,values,scales):
        self.specs=specs;self.values=values;self.scales=np.asarray(scales,dtype=float)
        self.lookup={tuple(s['grid_index_xyz']):i for i,s in enumerate(specs)}
        if len(self.lookup)!=len(specs) or len(specs)!=len(values):raise ValueError('Duplicate/missing tile')
        for s,v in zip(specs,values):
            index=np.array(s['grid_index_xyz'])
            if (s['shape']!=[17]*3 or s['spacing']!=[.75]*3 or v.shape!=(len(self.scales),17,17,17)
                    or not np.array_equal(s['shift'],self.origin+(index+.5)*12)):
                raise ValueError('Tile lattice or crystal mapping differs')

    @classmethod
    def from_pilot(cls,folder,scales):
        folder=Path(folder);gate=json.loads((folder/'tile_pilot_gate.json').read_text())
        if gate['status']!='TILE_INTERFACE_AND_COMPACTION_PASSED_ACCURACY_HOLD':
            raise ValueError('Physical tile interface/extraction gate failed')
        specs=[];values=[]
        for raw,compact in zip(gate['receipts'],gate['compact_extraction']):
            spec=raw['spec'];path=folder/spec['name']/'A_selected.float32'
            if digest(path)!=compact['sha256'] or path.stat().st_size!=compact['bytes']:
                raise ValueError('Immutable compact A changed')
            specs.append(spec);values.append(np.memmap(path,mode='r',dtype='<f4',shape=tuple(compact['shape_selected_zyx'])))
        return cls(specs,values,scales)

    def assert_production_ready(self):
        raise ValueError('Pilot tile coverage/continuity is not global physical A validation')

    @classmethod
    def from_production(cls,folder,scales):
        """Diagnostic subset; every expected pilot tile and exact bytes verified."""
        folder=Path(folder);plan_path=folder/'production_plan.json';plan=json.loads(plan_path.read_text())
        plan_sha=digest(plan_path);specs=[];values=[];mapping=None
        if plan['status']!='FROZEN_TILED_PHYSICAL_PRODUCTION_PLAN_ACCURACY_HOLD':
            raise ValueError('Unknown frozen tiled production plan')
        for spec in plan['pilot_specs']:
            receipt=json.loads((folder/spec['name']/'compact_receipt.json').read_text())
            compact=receipt['compact'];path=folder/spec['name']/'A_selected.float32'
            if (receipt['status']!='PHYSICAL_TILE_STORED_ACCURACY_HOLD' or receipt['spec']!=spec
                    or receipt['plan_sha256']!=plan_sha or not compact['all_selected_rows_bitwise_verified']
                    or path.stat().st_size!=compact['bytes'] or digest(path)!=compact['sha256']):
                raise ValueError('Stored diagnostic tile identity differs')
            if mapping is None:mapping=compact['raw_selected_mapping_sha256']
            elif mapping!=compact['raw_selected_mapping_sha256']:raise ValueError('Different crystal row mapping')
            specs.append(spec);values.append(np.memmap(path,mode='r',dtype='<f4',shape=tuple(compact['shape_selected_zyx'])))
        obj=cls(specs,values,scales);obj.selected_mapping_sha256=mapping
        return obj

    def owner(self,node):
        candidates=[]
        for n in node:
            i=int(n)//16;candidates.append([i-1,i] if n>0 and n%16==0 else [i])
        for index in itertools.product(*candidates):
            if index in self.lookup:
                local=np.asarray(node)-np.asarray(index)*16
                if np.any(local<0) or np.any(local>16):raise ValueError('Invalid shared-vertex owner')
                return self.lookup[index],int((local[2]*17+local[1])*17+local[0])
        raise ValueError('Required physical tile vertex missing; no fallback/extrapolation')

    def cache(self,points):
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('Invalid physical sample coordinates')
        q=(points-self.origin)/self.spacing
        if np.any(q<0) or np.any(q>self.intervals):raise ValueError('A sampling support extrapolation forbidden')
        lower=np.minimum(np.floor(q).astype(int),self.intervals-1);t=q-lower
        ids=np.empty((len(points),8),dtype=int);flat=np.empty_like(ids);weights=np.empty((len(points),8))
        vertex_cache={}
        for k,corner in enumerate(itertools.product((0,1),repeat=3)):
            corner=np.asarray(corner);nodes=lower+corner
            weights[:,k]=np.prod(np.where(corner,t,1-t),axis=1)
            for i,node in enumerate(nodes):
                # Exact grid faces need no zero-weight neighbour outside a
                # provided tile. Positive-weight vertices still require real
                # coverage; this does not extrapolate or change interpolation.
                if weights[i,k]==0:
                    ids[i,k]=0;flat[i,k]=0
                    continue
                key=tuple(node)
                if key not in vertex_cache:vertex_cache[key]=self.owner(node)
                ids[i,k],flat[i,k]=vertex_cache[key]
        if not np.allclose(weights.sum(axis=1),1,atol=1e-14,rtol=0):raise ValueError('Interpolation weights do not close')
        return ids,flat,weights

    def evaluate(self,crystal,cache):
        if not 0<=crystal<len(self.scales):raise ValueError('Unknown crystal')
        ids,flat,weights=cache;samples=np.empty(ids.shape)
        for tile in np.unique(ids):
            where=ids==tile;row=self.values[int(tile)][crystal].reshape(-1)
            samples[where]=row[flat[where]]
        out=np.sum(samples*weights,axis=1)*self.scales[crystal]
        if not np.isfinite(out).all() or np.any(out<0):raise ValueError('Invalid compact field response')
        return out
