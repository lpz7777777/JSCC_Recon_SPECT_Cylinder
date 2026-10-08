"""Read existing cuboid source, interpolation and window identities; no response run."""
import math
import sys
sys.dont_write_bytecode = True
import numpy as np
from ehe_common import DATA, HERE, REPORT, TRUTH, digest, read, write
from ehe_gpu_pipeline import source_grid


def audit():
    gpu = read(REPORT/'response_repair_freeze.json')
    cpu = read(REPORT/'transport_repair_freeze.json')
    payload = DATA/gpu['payload_dir']
    cpu_payload = DATA/cpu['payload_dir']
    collection = read(DATA/'transport/collection.json')
    gate = read(REPORT/'physical_gate.json')
    assert not gate['passed']
    assert digest(TRUTH) == gate['source_sha256']
    assert digest(HERE/'ehe_gpu_pipeline.py') == gpu['sha256']['ehe_gpu_pipeline.py']
    for n in ['Geant4Code_EHE/src/PrimaryGeneratorAction.cc', 'Geant4Code_EHE/src/EventAction.cc']:
        assert digest(cpu_payload/n) == cpu['sha256'][n]
    for n in ['engine/common/energy_window.h','engine/common/detector_local_scatter.h']:
        assert digest(payload/n) == gpu['sha256'][n]
    truth = np.load(TRUTH)
    coords = [truth[k] for k in ['x_mm','y_mm','z_mm']]
    first = DATA/'transport/source_registry/macros/NEMA_Body_H60_v01.mac'
    source_lines = [s for s in first.read_text().splitlines() if s.startswith('/xcat/add ')]
    fields = [list(map(float,s.split()[1:])) for s in source_lines]
    reconstructed = {e:np.zeros_like(truth[f'activity_{e}_zyx'],dtype=np.float64) for e in [218,440]}
    box_mass = {e:0.0 for e in [218,440]}
    box_number = {e:0 for e in [218,440]}
    for energy,x,y,z,hx,hy,hz,intensity in fields:
        e = int(energy); assert e in reconstructed and min(hx,hy,hz,intensity)>0
        axes = [np.flatnonzero((v >= c-h+1e-7) & (v <= c+h-1e-7)) for v,c,h in zip(coords,[x,y,z],[hx,hy,hz])]
        count = np.prod([len(v) for v in axes])
        assert math.isclose(count*27,8*hx*hy*hz,rel_tol=1e-12,abs_tol=1e-9)
        density = intensity/((.114 if e==218 else .259)*8*hx*hy*hz)
        reconstructed[e][np.ix_(axes[2],axes[1],axes[0])] += density
        box_mass[e] += intensity; box_number[e] += 1
    source_summary = {}
    for e in [218,440]:
        actual = truth[f'activity_{e}_zyx'].astype(np.float64)
        relative_l2 = float(np.linalg.norm(reconstructed[e]-actual)/np.linalg.norm(actual))
        assert relative_l2 < 1e-12, 'Registered continuous source differs from truth'
        mass = float(actual.sum()*27)
        assert math.isclose(box_mass[e],mass*(.114 if e==218 else .259),rel_tol=1e-12)
        source_summary[str(e)] = dict(boxes=box_number[e],truth_integral_gamma=mass,
            source_selection_weight=box_mass[e],reexpanded_density_relative_l2=relative_l2,
            max_density_absolute_error=float(np.max(abs(reconstructed[e]-actual))),
            source_sampling='continuous uniform cuboids, including exact merged constant-density regions',
            point_fold='3mm voxel centers with 6mm XY bilinear response interpolation',
            additional_uniform_3mm_voxel_variance_per_axis_mm2=.75)
    view_source_identity = []
    for v in range(20):
        p = DATA/f'transport/source_registry/macros/NEMA_Body_H60_v{v+1:02}.mac'
        assert digest(p)==collection['files'][p.relative_to(DATA/'transport').as_posix()]
        lines=p.read_text().splitlines()
        assert [s for s in lines if s.startswith('/xcat/add ')]==source_lines
        angle=float(next(s.split()[1] for s in lines if s.startswith('/xcat/angle ')))
        center_y=float(next(s.split()[1] for s in lines if s.startswith('/xcat/centerY ')))
        assert angle==v*18 and center_y==-345
        view_source_identity.append(dict(view=v+1,angle_degrees=angle,source_center_y_mm=center_y,macro_sha256=digest(p)))
    node_coords=np.arange(85)*6-252
    z_nodes=np.arange(40)*3-58.5
    assert np.array_equal(z_nodes,truth['z_mm'])
    interpolation=[]
    for e in [218,440]:
        density=truth[f'activity_{e}_zyx'].astype(np.float64)
        mass=density.sum()
        xy=density.sum(axis=0)
        mx=float(np.sum(xy*coords[0][None,:])/mass)
        my=float(np.sum(xy*coords[1][:,None])/mass)
        mz=float(np.sum(density.sum(axis=(1,2))*coords[2])/mass)
        for v in range(20):
            grid=source_grid(truth,e,v)
            a=math.radians(v*18)
            expected=np.array([mx*math.cos(a)+my*math.sin(a),my*math.cos(a)-mx*math.sin(a),mz])
            actual=np.array([np.sum(grid.sum(axis=(0,1))*node_coords),
                             np.sum(grid.sum(axis=(0,2))*node_coords),
                             np.sum(grid.sum(axis=(1,2))*z_nodes)])
            error=float(np.max(abs(actual-expected)))
            assert error < 1e-5 and abs(float(grid.sum())-1)<1e-12
            interpolation.append(dict(energy_keV=e,view=v+1,normalized_mass=float(grid.sum()),centroid_max_error_mm=error))
    windows={}
    params={}
    for name,e,window in [('A218',218,218),('A440',440,440),('C440to218',440,218)]:
        paths={n:payload/'params'/name/n for n in ['Params_Image.dat','Params_Detector.dat','Params_Physics.dat']}
        for n,p in paths.items():
            assert digest(p)==gpu['sha256'][f'params/{name}/{n}']
            params[name+'/'+n]=digest(p)
        image=np.fromfile(paths['Params_Image.dat'],'<f4')
        assert np.array_equal(image[:6],[85,85,40,6,6,3])
        assert np.array_equal(image[8:12],[0,0,0,323.75])
        d=np.fromfile(paths['Params_Detector.dat'],'<f4')
        assert d[0]==2312 and len(d)==2312*12+1
        detector=d[1:].reshape(2312,12)
        resolution32=detector[:,9]
        resolution=resolution32.astype(float)
        expected_resolution=.13*math.sqrt(511/e)
        assert np.max(abs(resolution-expected_resolution))<1e-7
        physics=np.fromfile(paths['Params_Physics.dat'],'<f4')
        assert physics[7]==e
        cpu_bounds=np.array([window*(1-.13*math.sqrt(511/window)/2),window*(1+.13*math.sqrt(511/window)/2)])
        bounds=np.tile(physics[5:7].astype(float),(2312,1)) if physics[4]>0 else (np.float32(e)*np.stack([
            np.float32(1)-resolution32/np.float32(2),np.float32(1)+resolution32/np.float32(2)],axis=1)).astype(float)
        bound_error=float(np.max(abs(bounds-cpu_bounds)))
        assert bound_error<2e-5
        windows[name]=dict(source_energy_keV=e,window_energy_keV=window,
            response_bounds_keV=bounds[0].tolist(),transport_bounds_keV=cpu_bounds.tolist(),
            max_bound_error_keV=bound_error,resolution_relative_max_error=float(np.max(abs(resolution-expected_resolution))),
            deposited_energy_sigma_equation='0.13*sqrt(511/Edeposit)*Edeposit/2.35482; float32 response versus float64 transport',
            bins_checked=2312)
    result=dict(passed=True,scope='Read-only source density, rotation, interpolation-coordinate and window identity audit; HOLD not cleared',
        scientific_status='HOLD',scientific_gate_passed=False,job=1677211,source=source_summary,
        views=view_source_identity,interpolation=interpolation,windows=windows,parameter_sha256=params,
        frozen_execution_source_sha256={
            'PrimaryGeneratorAction.cc':digest(cpu_payload/'Geant4Code_EHE/src/PrimaryGeneratorAction.cc'),
            'EventAction.cc':digest(cpu_payload/'Geant4Code_EHE/src/EventAction.cc'),
            'ehe_gpu_pipeline.py':digest(HERE/'ehe_gpu_pipeline.py'),
            'energy_window.h':digest(payload/'engine/common/energy_window.h'),
            'detector_local_scatter.h':digest(payload/'engine/common/detector_local_scatter.h')},
        source_truth_sha256=digest(TRUTH),physical_gate_sha256=digest(REPORT/'physical_gate.json'),
        code_sha256=digest(__file__),prediction_refitted=False,production_inputs_modified=False,
        gpu_simulation_or_response_executed=False,
        limitation='Matching source mass/centroids/windows does not bound response-count error. Continuous cuboid averaging, coarse point-response interpolation, finite quadrature and scatter-history coverage remain unquantified.')
    write(REPORT/'physical_source_window_basis_audit.json',result)
    print('SOURCE_WINDOW_IDENTITY_PASS',source_summary,windows)
    print('MAX_INTERPOLATION_CENTROID_ERROR_MM',max(x['centroid_max_error_mm'] for x in interpolation))
    return result


if __name__=='__main__':
    audit()
