"""One bounded physical A column for full-axis fine-field validation.

Original PE-v4 and full physical ScatterGen binaries/parameters, all 11520
computational detector rows. Old matrices and calibration stay read-only.
This isolated column does not certify the full FOV or authorize S2/imaging.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from generate_compton_a_guard import digest,write,axes,run_one,ENGINE_REL,SOURCE_RUN,NDET,PE_HASH,SCATTER_HASH


def comparison(actual,expected):
    delta=actual.astype(float)-expected
    floor=float(expected.max())*1e-8
    l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(expected.ravel()),1e-30))
    elements=bool(np.all(abs(delta)<=1e-5*abs(expected)+floor))
    return dict(relative_l2=l2,bitwise_equal=bool(np.array_equal(actual,expected)),
        maximum_absolute_error=float(abs(delta).max()),numerical_floor=floor,
        elementwise_passed=elements,passed=l2<=1e-5 and elements)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','patches','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cuda',type=int,default=0);a=p.parse_args()
    engine=a.root/ENGINE_REL;source=engine/SOURCE_RUN
    pe=engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter=engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe)!=PE_HASH or digest(scatter)!=SCATTER_HASH:raise ValueError('Original physical model differs')
    provenance=json.loads((source/'ELLIPSE_inputs.json').read_text())
    for name,sha in provenance['parameters'].items():
        if digest(source/name)!=sha:raise ValueError('Original geometry/material parameters changed')
    reference=json.loads((a.patches/'patch_ready.json').read_text())
    if reference['pe_binary_sha256']!=PE_HASH or reference['scatter_binary_sha256']!=SCATTER_HASH:
        raise ValueError('Fine reference model differs')
    spec=dict(name='near_column',shape=(17,11,161),spacing=(.75,.75,.75),shift=(0,252,0))
    bytes_four=int(np.prod(spec['shape']))*NDET*4*4
    if shutil.disk_usage(a.output.parent).free<bytes_four*1.25+(10<<30):raise ValueError('Insufficient bounded column storage')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'plan.json',dict(spec=spec,additional_sampling_points=int(np.prod(spec['shape'])),
        raw_output_bytes=bytes_four,rows=NDET,pe_binary_sha256=PE_HASH,scatter_binary_sha256=SCATTER_HASH,
        original_params_sha256=provenance['parameters'],reference_ready_sha256=digest(a.patches/'patch_ready.json'),
        new_transport_photons=0,imaging_grid_unchanged=True,calibration_refitted=False))
    receipt=run_one(spec,a.output,source,pe,scatter,a.cuda);xyz=axes(spec)
    matrices={}
    kinds=('pe.sysmat','pe_windowed.sysmat','Scatter_SysMat','SysMat_withScatter')
    parent_kinds=('PE_SysMat','PE_Windowed_SysMat','Scatter_SysMat','SysMat_withScatter')
    for kind,parent_kind in zip(kinds,parent_kinds):
        name=next(n for n in receipt['matrices'] if n.startswith(kind))
        values=np.memmap(a.output/spec['name']/name,mode='r',dtype='<f4',shape=(NDET,161,11,17))
        suffix='_v4' if parent_kind.startswith('PE_') else ''
        parent=source/f'{parent_kind}_shift_0.000000_0.000000_0.000000{suffix}.sysmat'
        old=np.memmap(parent,mode='r',dtype='<f4',shape=(NDET,40,85,85))
        iz=np.searchsorted(xyz[2],np.arange(40)*3-58.5)
        anchors=comparison(np.array(values[:,iz,5,8],copy=True),np.array(old[:,:,84,42],copy=True))
        references={}
        for part in reference['parts']:
            folder=a.patches/part['spec']['name'];file=next(n for n in part['matrices'] if n.startswith(kind))
            if digest(folder/file)!=part['matrices'][file]['sha256']:raise ValueError('Fine reference changed')
            axes_old=axes(part['spec']);axes_sub=[v[::2] for v in axes_old]
            indices=[np.searchsorted(new,v) for new,v in zip(xyz,axes_sub)]
            for new,i,want in zip(xyz,indices,axes_sub):np.testing.assert_allclose(new[i],want,rtol=0,atol=1e-10)
            ix,iy,iz=indices
            fine=np.memmap(folder/file,mode='r',dtype='<f4',shape=(NDET,*reversed(part['spec']['shape'])))
            references[part['spec']['name']]=comparison(
                np.array(values[:,iz[:,None,None],iy[None,:,None],ix[None,None,:]],copy=True),
                np.array(fine[:,::2,::2,::2],copy=True))
        matrices[kind]=dict(original_40_axial_anchor=anchors,independent_fine_common_points=references)
    passed=all(r['original_40_axial_anchor']['passed'] and all(v['passed'] for v in r['independent_fine_common_points'].values()) for r in matrices.values())
    result=dict(status='COLUMN_COMMON_POINTS_PASSED_GLOBAL_ACCURACY_HOLD' if passed else 'HOLD_COLUMN_COMMON_POINTS',
        part=receipt,common_point_regression=matrices,code_sha256=digest(Path(__file__)),
        pe_binary_sha256=PE_HASH,scatter_binary_sha256=SCATTER_HASH,
        reference_ready_sha256=digest(a.patches/'patch_ready.json'),new_transport_photons=0,
        imaging_grid_unchanged=True,calibration_refitted=False,reconstruction_permitted=False,S2_generated=False,
        interpretation='A .75 mm near-side column spanning -60..60 mm; all source/target detector rows. '
                       'Common physical samples checked; global interpolation and column interfaces are not yet certified.')
    write(a.output/'column_gate.json',result);print(json.dumps({k:v for k,v in result.items() if k!='part'},indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
