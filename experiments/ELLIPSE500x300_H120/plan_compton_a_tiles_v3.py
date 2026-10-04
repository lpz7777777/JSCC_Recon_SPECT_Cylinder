"""Exact bounded Cartesian tile coverage for the frozen R2 reference cells.

This is an A sampling/storage plan, not a new imaging or event-selection grid.
No matrix production or accuracy permission is implied by geometric coverage.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from geometry import grid
from compton_overlap_integrator import reference_cell_bounds
from generate_compton_a_guard import digest,write

SPACING=.75
XY_ORIGIN=-258.
XY_INTERVALS=688
Z_ORIGIN=-60.
Z_INTERVALS=160
TILE_INTERVALS=16


def cover_xy_boxes(boxes):
    """Allocate interpolation cubes, including all vertices of each bbox."""
    needed=np.zeros((XY_INTERVALS,XY_INTERVALS),dtype=bool)
    for low,high in boxes:
        lo=np.floor((np.asarray(low[:2])-XY_ORIGIN)/SPACING+1e-10).astype(int)
        hi=np.ceil((np.asarray(high[:2])-XY_ORIGIN)/SPACING-1e-10).astype(int)
        if np.any(lo<0) or np.any(hi>XY_INTERVALS) or np.any(hi<=lo):
            raise ValueError('Reference cell outside the declared A lattice')
        needed[lo[1]:hi[1],lo[0]:hi[0]]=True
    tiles=needed.reshape(43,16,43,16).any(axis=(1,3))
    y,x=np.nonzero(tiles)
    return needed,np.column_stack((x,y))


def tile_spec(ix,iy,iz):
    if not (0<=ix<43 and 0<=iy<43 and 0<=iz<10):raise ValueError('Unknown A tile')
    shift=[XY_ORIGIN+(ix+.5)*12,XY_ORIGIN+(iy+.5)*12,Z_ORIGIN+(iz+.5)*12]
    return dict(name=f'tile_x{ix:02d}_y{iy:02d}_z{iz:02d}',shape=[17,17,17],
        spacing=[SPACING]*3,shift=shift,grid_index_xyz=[int(ix),int(iy),int(iz)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','geometry','measure','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());geo=np.load(a.geometry)
    coords,cells,_=grid(cfg);np.testing.assert_allclose(coords,geo['coordinates_mm'],atol=1e-10,rtol=0)
    mm=json.loads((a.measure/'measure_manifest.json').read_text())
    if (mm['status']!='PASSED' or mm['baseline_geometry_sha256']!=digest(a.geometry)
            or mm['measure_sha256']!=digest(a.measure/'measure.npz')):
        raise ValueError('Verified physical measure differs')
    m=np.load(a.measure/'measure.npz');partial=m['partial_indices'];n=cfg['points_per_layer']
    if len(partial)!=6880 or len(geo['active_indices'])!=82040:raise ValueError('Frozen identities differ')
    required=np.unique(geo['inverse_rotation'][partial].reshape(-1));transverse=np.unique(required%n)
    if len(required)!=52800 or len(transverse)!=1320:raise ValueError('Frozen reference coverage differs')
    boxes=[reference_cell_bounds(cells[i],0.,0) for i in transverse]
    cubes,xy=cover_xy_boxes(boxes)
    # Validate allocation from actual bbox corners/interior, not just volume.
    for low,high in boxes:
        for t in (.001,.25,.5,.75,.999):
            q=np.asarray(low[:2])*(1-t)+np.asarray(high[:2])*t
            ix,iy=np.floor((q-XY_ORIGIN)/SPACING).astype(int)
            if not cubes[iy,ix] or not np.any((xy==[ix//16,iy//16]).all(axis=1)):
                raise ValueError('Required full-reference interpolation cube missing')
    points_per_tile=17**3;count=len(xy)*10
    result=dict(status='GEOMETRY_AND_STORAGE_PLAN_ONLY_ACCURACY_HOLD',
        geometry_sha256=digest(a.geometry),config_sha256=digest(a.config),
        measure_manifest_sha256=digest(a.measure/'measure_manifest.json'),
        measure_npz_sha256=digest(a.measure/'measure.npz'),
        unique_detector_reference_cells=len(required),unique_transverse_reference_cells=len(transverse),
        required_xy_interpolation_cubes=int(cubes.sum()),xy_tiles=len(xy),z_tiles=10,total_tiles=count,
        sampling_spacing_mm=SPACING,tile_shape_xyz=[17,17,17],tile_core_extent_mm=12,
        xy_origin_mm=XY_ORIGIN,z_origin_mm=Z_ORIGIN,xy_intervals=XY_INTERVALS,z_intervals=Z_INTERVALS,
        points_per_tile=points_per_tile,points_with_shared_face_duplicates=count*points_per_tile,
        four_raw_physical_11520_row_bytes_per_tile=points_per_tile*11520*4*4,
        retained_combined_10496_row_bytes_per_tile=points_per_tile*10496*4,
        retained_combined_all_tiles_bytes=count*points_per_tile*10496*4,
        raw_all_tiles_bytes=count*points_per_tile*11520*4*4,
        xy_tile_indices=xy.tolist(),
        pilot_tiles=[tile_spec(21,41,5),tile_spec(21,42,5)],
        new_transport_photons=0,imaging_grid_unchanged=True,production_started=False,
        reconstruction_permitted=False,S2_generated=False,
        interpretation='Tiles preserve all 11520 physical source/target rows during PE/Scatter calculation. '
            'Only selected 10496 combined float32 rows are persisted after verified extraction. '
            'All tiles use one physical lattice; shared faces need deterministic ownership and validation. '
            'This exact allocation includes bbox padding/shared vertices, unlike a volume-only estimate.')
    write(a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('xy_tile_indices','pilot_tiles')},indent=2))


if __name__=='__main__':main()
