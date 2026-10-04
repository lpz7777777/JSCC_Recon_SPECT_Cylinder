"""Complete-6880-cell R2 throughput and mixed-Z diagnostic; no MLEM or S2.

Uses predetermined first two R1-accepted NEMA events in each of views 1/11.
Sampling controls runtime diagnostics only; it is not a new imaging dataset.
Local fine A boxes leave global accuracy and cell interfaces uncertified.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from geometry import grid
from generate_compton_a_guard import digest
from compton_cell_response_field import CellResponseField
from compton_overlap_integrator import OverlapIntegrator
from compton_overlap_assembly import OverlapAssembly
from compton_event_response import (ComptonEventSettings, prepare_compton_events,
    build_detector_position_variance, build_compton_cone_weights, min_standardized_compton_arm)
from detector_csv import load_detector_coordinates


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('field','patches','geometry','config','measure','inputs','factors','scan','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--node-chunk',type=int,default=32768)
    p.add_argument('--cells-per-block',type=int,default=8)
    p.add_argument('--event-count',type=int,default=2)
    p.add_argument('--views',type=int,nargs='+',default=[1,11])
    p.add_argument('--reference-benchmark',type=Path)
    a=p.parse_args()
    if not 1 <= a.event_count <= 64:raise ValueError('Bounded diagnostic event budget exceeded')
    if len(a.views)>4 or len(set(a.views))!=len(a.views) or any(v not in (1,6,11,16) for v in a.views):
        raise ValueError('Unregistered diagnostic views')
    torch.set_num_threads(8);torch.set_grad_enabled(False);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    start=time.monotonic();cfg=json.loads(a.config.read_text());geo=np.load(a.geometry)
    mm=json.loads((a.measure/'measure_manifest.json').read_text())
    gate=json.loads((a.scan/'validation_gate.json').read_text())
    if (gate['status']!='PASSED' or gate['geometry_sha256']!=digest(a.geometry)
            or gate['scans']['NEMA_ideal']['kept']!=91225 or mm['status']!='PASSED'
            or mm['baseline_geometry_sha256']!=digest(a.geometry)
            or mm['measure_sha256']!=digest(a.measure/'measure.npz')):
        raise ValueError('Frozen R1/measure contract differs')
    if gate['kernel_sha256']!=digest(Path(__import__('compton_event_response').__file__)):
        raise ValueError('Frozen R1 kernel differs')
    coords,cells,_=grid(cfg)
    np.testing.assert_allclose(coords,geo['coordinates_mm'],rtol=0,atol=1e-10)
    measure=np.load(a.measure/'measure.npz');partial=measure['partial_indices']
    if len(partial)!=6880 or len(geo['active_indices'])!=82040:raise ValueError('Partial/active identities differ')
    field=CellResponseField(a.field,a.patches)
    if field.provenance['baseline_geometry_sha256']!=digest(a.geometry):raise ValueError('A field geometry differs')
    factor=a.factors/'440keV_RotateNum20'
    if digest(factor/'SysMat_polar')!=field.provenance['baseline_B_sha256']:raise ValueError('Frozen B differs')
    detector=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    variance=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35,
        geometry_mode='stable_float64')
    operator=OverlapIntegrator(cells,coords,partial,cfg['points_per_layer'],field,settings,
        node_chunk=a.node_chunk,cells_per_block=a.cells_per_block,diagnostic_only=True)
    assembler=OverlapAssembly(geo['active_indices'],partial,geo['inverse_rotation'],132040)
    B=np.memmap(factor/'SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    grid_coords=torch.tensor(geo['coordinates_mm'],dtype=torch.float32,device=device)
    a.output.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((a.inputs/'input_manifest.json').read_text());cases=[]
    orders=((8,4,4),(16,8,8),(32,12,12))
    for view in a.views:
        path=a.inputs/'NEMA'/f'ideal_v{view:02d}.csv'
        if digest(path)!=provenance['files'][path.relative_to(a.inputs).as_posix()]:raise ValueError('NEMA List changed')
        ids=np.load(a.scan/f'NEMA_ideal_v{view:02d}_kept_rows.npy')[:a.event_count]
        raw=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)[ids]
        prepared,_=prepare_compton_events(torch.tensor(raw,device=device),settings,detector,variance,variance,
                                        input_energies_already_smeared=True)
        if prepared is None or prepared.count!=len(ids):raise ValueError('Frozen accepted event lost in preparation')
        q=min_standardized_compton_arm(prepared,grid_coords,settings)
        if bool((q>3).any()):raise ValueError('Frozen R1 acceptance differs; do not silently drop')
        crystals=prepared.cpnum1.cpu().numpy()-1
        raw_kb=build_compton_cone_weights(prepared,grid_coords,settings)*torch.tensor(
            np.array(B[:,crystals].T,copy=True),device=device)
        common_Z=raw_kb.double().sum(1).cpu().numpy()
        previous=None;previous_rows=None
        for oi,order in enumerate(orders):
            def progress(value):
                if value['completed_cells']%512==0 or value['completed_cells']==6880:
                    print(json.dumps(dict(view=view,order=list(order),**value)),flush=True)
            obj,ref=operator.integrate(prepared,view-1,order,progress=progress)
            rows,norm=assembler.assemble(raw_kb,obj,ref,view-1)
            x=torch.linspace(.1,1.1,82040,device=device)[:,None]
            y=torch.linspace(.3,.9,len(ids),device=device)[:,None]
            lhs=(rows@x).T@y;rhs=x.T@(rows.T@y)
            adjoint=float((lhs-rhs).abs()/torch.maximum(lhs.abs(),rhs.abs()).clamp_min(1e-20))
            if adjoint>1e-5:raise ValueError('R2 full active-column transpose failed')
            rec=dict(view=view,order=list(order),events=len(ids),input_sha256=digest(path),
                kept_rows_sha256=digest(a.scan/f'NEMA_ideal_v{view:02d}_kept_rows.npy'),
                raw_rows=ids.tolist(),q=q.cpu().tolist(),active_points=rows.shape[1],
                inner_product_relative_error=adjoint,mixed_reference=norm.cpu().tolist(),
                common_R1_reference=common_Z.tolist(),statistics=operator.statistics.copy())
            actual=np.stack((obj.cpu().numpy(),ref.cpu().numpy()),axis=-1)
            if previous is not None:
                floor=1e-10*common_Z[:,None,None]
                change=np.abs(actual-previous)
                scaled=change/np.maximum(np.abs(actual),floor)
                passed=change <= .01*np.abs(actual)+floor
                rec.update(quadrature_entries=passed.size,quadrature_entries_passed=int(passed.sum()),
                    maximum_integral_relative_change=float(scaled.max()),
                    active_row_relative_l2=[float(np.linalg.norm(u-v)/max(np.linalg.norm(u),1e-30))
                        for u,v in zip(rows.cpu().numpy(),previous_rows)],
                    maximum_absolute_integral_change_over_common_Z=float((change/common_Z[:,None,None]).max()))
            name=f'v{view:02d}_order{oi}'
            if a.reference_benchmark is not None and (a.reference_benchmark/(name+'.npz')).exists():
                reference=np.load(a.reference_benchmark/(name+'.npz'))
                nr=len(reference['raw_rows'])
                if not np.array_equal(reference['raw_rows'],ids[:nr]):raise ValueError('Reference identities changed')
                old=np.stack((reference['object_integrals'],reference['reference_integrals']),axis=-1)
                delta=np.abs(actual[:nr]-old)
                error=delta/np.maximum(np.abs(old),1e-10*common_Z[:nr,None,None])
                same=bool(np.all(delta <= 1e-10*np.abs(old)+1e-14*common_Z[:nr,None,None]))
                rec['uncached_reference_regression']=dict(passed=same,maximum_relative_error=float(error.max()),
                    reference_file_sha256=digest(a.reference_benchmark/(name+'.npz')),events=nr)
                if not same:raise ValueError('Factorized cache changed the original complete integrals')
            np.savez_compressed(a.output/(name+'.npz'),object_integrals=actual[:,:,0],
                reference_integrals=actual[:,:,1],mixed_Z=norm.cpu().numpy(),raw_rows=ids,partial_indices=partial)
            rec['integral_file_sha256']=digest(a.output/(name+'.npz'))
            cases.append(rec);previous=actual;previous_rows=rows.cpu().numpy()
            write(a.output/'progress.json',dict(cases=cases,elapsed_seconds=time.monotonic()-start))
    result=dict(status='INTEGRATOR_BENCHMARK_COMPLETE_A_ACCURACY_HOLD',cases=cases,
        geometry_sha256=digest(a.geometry),config_sha256=digest(a.config),
        measure_manifest_sha256=digest(a.measure/'measure_manifest.json'),
        R1_scan_gate_sha256=digest(a.scan/'validation_gate.json'),field_provenance=field.provenance,
        kernel_sha256=digest(Path(__import__('compton_event_response').__file__)),
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('benchmark_compton_overlap_v3.py','compton_overlap_integrator.py','compton_cell_response_field.py','compton_overlap_assembly.py')},
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
        total_gpu_bytes=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else 0,
        provider_xy_cache_bytes=field.cell_cache.bytes,provider_xy_cache_limit_bytes=field.cell_cache.limit,
        elapsed_seconds=time.monotonic()-start,new_transport_photons=0,
        reconstruction_permitted=False,S2_generated=False,
        interpretation='All 6880 partial cells, object and full reference, same frozen R1 identities. '
                       'Local fine A coverage and cell interfaces do not certify global A accuracy; no formal imaging.')
    write(a.output/'benchmark_gate.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_sha256')},indent=2),flush=True)


if __name__=='__main__':main()
