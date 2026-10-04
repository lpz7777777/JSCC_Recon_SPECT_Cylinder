"""Freeze geometrically selected independent A control cells before production."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from geometry import grid
from compton_overlap_integrator import reference_cell_bounds
from generate_compton_a_guard import digest,write


def main():
    here=Path(__file__).resolve().parent;data=here/'generated/compton_response_geometry_v3'
    report=here/'reports/NEMA_Body_H60/compton_response_geometry_v3'
    geometry=here/'generated/Geometry/geometry.npz';geo=np.load(geometry)
    config=here/'config.json';cfg=json.loads(config.read_text());coords,cells,_=grid(cfg)
    n=cfg['points_per_layer'];m=np.load(data/'precise_measure/measure.npz');partial=m['partial_indices']
    required=np.unique(geo['inverse_rotation'][partial].reshape(-1));xy=np.unique(required%n)
    targets=(('inner_near',156,np.pi/2),('middle_oblique',204,np.pi/4),
        ('outer_side',252,0),('outer_far',252,-np.pi/2))
    cases=[]
    for name,radius,angle in targets:
        target=np.array([radius*np.cos(angle),radius*np.sin(angle)])
        chosen=int(xy[np.argmin(np.sum((coords[xy,:2]-target)**2,axis=1))])
        origin,view=next((int(i%n),v) for v in range(20) for i in partial[:172]
                        if int(geo['inverse_rotation'][i,v]%n)==chosen)
        for layer in (0,20,39):
            z=float(coords[layer*n+origin,2]);lo,hi=reference_cell_bounds(cells[origin],z,view)
            # Shared .75 mm-aligned faces; .375 mm is an independent calculation.
            low=np.floor(lo/.75+1e-10)*.75;high=np.ceil(hi/.75-1e-10)*.75
            parts=[]
            for step,label in ((.75,'coarse'),(.375,'fine')):
                shape=np.rint((high-low)/step).astype(int)+1
                if 11520*int(np.prod(shape))>=2**31:raise ValueError('Unsafe original ScatterGen index budget')
                parts.append(dict(name=f'{name}_z{layer:02d}_{label}',shape=shape.tolist(),
                    spacing=[step]*3,shift=((high+low)/2).tolist()))
            cases.append(dict(name=f'{name}_z{layer:02d}',group=(0,20,39).index(layer),
                desired_detector_xy_mm=target.tolist(),actual_detector_xy_mm=coords[chosen,:2].tolist(),
                detector_xy_index=chosen,object_xy_index=origin,view=view+1,layer=layer,z_mm=z,
                full_reference_bounds_mm=[lo.tolist(),hi.tolist()],parts=parts))
    if len(cases)!=12 or len(set(c['detector_xy_index'] for c in cases))!=4:
        raise ValueError('Regional control identities are not distinct')
    value=dict(status='FROZEN_REGIONAL_GEOMETRY_DIAGNOSTIC_ONLY',geometry_sha256=digest(geometry),
        config_sha256=digest(config),measure_npz_sha256=digest(data/'precise_measure/measure.npz'),
        source_sha256=digest(Path(__file__)),cases=cases,
        matrix_points=sum(int(np.prod(p['shape'])) for c in cases for p in c['parts']),
        raw_response_bytes=sum(int(np.prod(p['shape']))*11520*4*4 for c in cases for p in c['parts']),
        physical_detector_rows=11520,new_transport_photons=0,imaging_grid_unchanged=True,
        reconstruction_permitted=False,S2_generated=False,
        interpretation='Control selection uses only frozen full-reference geometry. Nearest eligible '
            'detector-frame cells are recorded when a desired radius/angle has no exact eligible cell. '
            'Original object cell and acquisition view are explicit for later intersection/K analysis. '
            'Common-point production gates alone do not certify interpolation precision.')
    destination=report/'regional_a_plan.json'
    if destination.exists():raise ValueError('Control plan is already frozen; do not silently overwrite')
    write(destination,value);print(json.dumps({k:v for k,v in value.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
