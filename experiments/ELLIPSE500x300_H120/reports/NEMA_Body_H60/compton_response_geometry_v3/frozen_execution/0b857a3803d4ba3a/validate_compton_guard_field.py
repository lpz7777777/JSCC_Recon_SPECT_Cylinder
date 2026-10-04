"""Independent A440 midpoint interpolation gate, using unchanged production physics.

Tests cover centre, axial endpoints and the long/short-axis neighbourhood.
This is an interpolation check, not independent transport or proof of S2.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from generate_compton_a_guard import digest, axes, specs
from build_compton_a_guard_field import combined
from compton_boundary_quadrature import GuardedPolarResponseField


def validate(field_dir, guard_dir, output):
    manifest=json.loads((field_dir/'field_manifest.json').read_text())
    if digest(field_dir/'A_field.float64')!=manifest['A_field_sha256']:raise ValueError('A field hash differs')
    if digest(field_dir/'field_geometry.npz')!=manifest['field_geometry_sha256']:raise ValueError('A support hash differs')
    geo=np.load(field_dir/'field_geometry.npz');xy=geo['xy_mm'];z=geo['z_mm']
    field=np.memmap(field_dir/'A_field.float64',mode='r',dtype='<f8',shape=tuple(manifest['shape']))
    interpolator=GuardedPolarResponseField(xy,int(geo['original_points']),z)
    selected=geo['selected_raw_detectors'];scale=np.asarray(manifest['calibration_scales'])
    rows=[];omitted=[]
    for name in ('mid_minus','mid_plus'):
        raw=combined(guard_dir,name);spec=next(s for s in specs() if s['name']==name)
        xs,ys,zs=axes(spec)
        for yi,yv in enumerate(ys):
            for xi,xv in enumerate(xs):
                if np.hypot(xv,yv)>255:
                    omitted.append(dict(part=name,xyz=[float(xv),float(yv),float(zs[0])],reason='Outside full circle support'))
                    continue
                point=np.array([[xv,yv,zs[0]]]);cache=interpolator.cache(point)
                actual=raw[selected,0,yi,xi].astype(float)*scale
                predicted=np.array([interpolator.evaluate(field[c],cache)[0] for c in range(10496)])
                delta=predicted-actual;floor=1e-8*max(actual.max(),1e-30)
                rel_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(actual),1e-30))
                rel_eff=float(predicted.sum()/actual.sum()-1) if actual.sum()>0 else None
                elementwise=bool(np.all(np.abs(delta)<=.01*np.abs(actual)+floor))
                rows.append(dict(part=name,xyz_mm=point[0].tolist(),relative_l2=rel_l2,
                    total_response_relative_error=rel_eff,elementwise_passed=elementwise,
                    max_abs_error=float(np.abs(delta).max()),absolute_floor=floor,
                    bins_exceeding_tolerance=int(np.count_nonzero(np.abs(delta)>.01*np.abs(actual)+floor)),
                    passed=elementwise and rel_l2<=.01 and rel_eff is not None and abs(rel_eff)<=.01))
    result=dict(status='PASSED' if all(r['passed'] for r in rows) and len(rows)==10 else 'HOLD',
        field_manifest_sha256=digest(field_dir/'field_manifest.json'),
        field_sha256=manifest['A_field_sha256'],cases=rows,omitted_outside_support=omitted,
        interpolation_relative_tolerance=.01,new_transport_photons=0,
        covers='Five transverse positions at each independently simulated axial midpoint.',
        limitation='Radial refinement and boundary event integrals remain separately gated; this is not S2 validation.',
        reconstruction_permitted=False)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('field','guard','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();validate(a.field,a.guard,a.output)
