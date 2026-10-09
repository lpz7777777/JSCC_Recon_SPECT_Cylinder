"""Read-only geometric audit of the frozen intercrystal visible-face rule.

Integrates only dOmega = h/r**3 dA. No Compton, energy window, attenuation,
absorption, truth folding, simulation, response or reconstruction is evaluated.
"""
import csv
import math
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from ehe_common import DATA, REPORT, digest, read, write


def triangle_angle(a, b, c):
    lengths = [np.linalg.norm(v) for v in (a,b,c)]
    numerator = abs(float(np.dot(a,np.cross(b,c))))
    denominator = (math.prod(lengths) + np.dot(a,b)*lengths[2]
                   + np.dot(b,c)*lengths[0] + np.dot(c,a)*lengths[1])
    return 2*math.atan2(numerator,float(denominator))


def rectangle_angle(h, u0, u1, v0, v1):
    # The mixed derivative of atan(uv/(h*sqrt(h*h+u*u+v*v)))
    # is h/(h*h+u*u+v*v)**(3/2), for h>0. Evaluate its four corners.
    def primitive(u,v):
        return math.atan2(u*v,h*math.sqrt(h*h+u*u+v*v))
    return math.fsum((primitive(u1,v1),-primitive(u0,v1),
                      -primitive(u1,v0),primitive(u0,v0)))


def face_geometry(observer, axis, half):
    first,second = (axis+1)%3,(axis+2)%3
    h = observer[axis]-half[axis]
    assert h>0
    u0,u1 = -half[first]-observer[first],half[first]-observer[first]
    v0,v1 = -half[second]-observer[second],half[second]-observer[second]
    reference = rectangle_angle(h,u0,u1,v0,v1)
    vertices = [np.array([h,u,v]) for u,v in
                ((u0,v0),(u1,v0),(u1,v1),(u0,v1))]
    triangle = (triangle_angle(*vertices[:3])
                + triangle_angle(vertices[0],vertices[2],vertices[3]))
    return h,u0,u1,v0,v1,reference,triangle


def midpoint_angle(h,u0,u1,v0,v1,n):
    u = u0+(np.arange(n)+.5)*(u1-u0)/n
    v = v0+(np.arange(n)+.5)*(v1-v0)/n
    return float(np.sum(h/(h*h+u[:,None]**2+v[None,:]**2)**1.5)
                 *(u1-u0)*(v1-v0)/(n*n))


def gauss_angle(h,u0,u1,v0,v1,rule):
    points,weights = rule
    u = (u0+u1)/2+(u1-u0)*points/2
    v = (v0+v1)/2+(v1-v0)*points/2
    return float(np.sum(weights[:,None]*weights[None,:]
                        *h/(h*h+u[:,None]**2+v[None,:]**2)**1.5)
                 *(u1-u0)*(v1-v0)/4)


def audit():
    began = time.monotonic()
    runtime_path = REPORT/'physical_scatter_runtime_read_only.json'
    runtime = read(runtime_path)
    accepted = read(REPORT/'physical_scatter_runtime_acceptance.json')
    assert runtime['passed'] and accepted['passed']
    assert digest(runtime_path) == accepted['runtime_evidence_sha256']
    assert digest(REPORT/'physical_gate.json') == runtime['original_physical_gate_sha256']
    assert not read(REPORT/'physical_gate.json')['passed']
    freeze = read(REPORT/'response_repair_freeze.json')
    payload = DATA/freeze['payload_dir']
    source_name = 'engine/ScatterGen_RayTracing_CircularHole/scatter.cu'
    assert digest(payload/source_name) == freeze['sha256'][source_name] == runtime['scientific_source_sha256']
    source = (payload/source_name).read_text()
    function = source[source.index('__device__ float integrateIntercrystalTargetSurface('):]
    function = function[:function.index('__device__ int indexFrombitmap_crystal')]
    assert 'const float solid_angle = projected_cosine * cell_area / distance_squared;' in function
    assert 'const float sample_contribution = angular_density * solid_angle' in function
    assert '* window_acceptance * expf(-attenuation) * target_photoelectric;' in function
    assert 'target_half_extent[normal_axis] + 1e-6f' in function
    bindings = []
    geometry = None
    for record in runtime['records']:
        response,slab = record['response'],record['slab']
        name = f'params/{response}/Params_Detector.dat'
        receipt_path = REPORT/'response_preservation_1672966'/response/f'slab_{slab}'/'receipt.json'
        receipt = read(receipt_path)
        assert digest(receipt_path) == record['pre_stop_receipt_sha256']
        assert digest(payload/name) == record['frozen_detector_params_sha256'] == freeze['sha256'][name] == receipt['files']['Params_Detector.dat']
        detector = np.fromfile(payload/name,'<f4')[1:].reshape(2312,12)
        candidate = detector[:,[0,1,2,3,4,5,10]].astype(np.float64)
        if geometry is None: geometry = candidate
        assert np.array_equal(candidate,geometry)
        bindings.append(dict(response=response,slab=slab,params_sha256=digest(payload/name),
                             pre_stop_receipt_sha256=digest(receipt_path)))
    assert len(bindings) == 12
    assert np.all(geometry[:,3:6] == [4,10,4]) and np.all(geometry[:,6] == 0)
    xs,zs = np.unique(geometry[:,0]),np.unique(geometry[:,2])
    assert len(xs) == 68 and len(zs) == 34 and np.all(np.diff(xs)==4) and np.all(np.diff(zs)==4)
    assert len(np.unique(geometry[:,:3],axis=0)) == len(xs)*len(zs) == 2312
    assert np.all(geometry[:,1] == geometry[0,1])
    half = geometry[0,3:6]/2
    rules = [np.polynomial.legendre.leggauss(n) for n in (32,64)]
    rows = []
    face_checks = []
    for ix in range(len(xs)):
        for iz in range(len(zs)):
            if ix == iz == 0: continue
            observer = np.array([4*ix,0.,4*iz])
            multiplicity = ((len(xs)-ix)*(len(zs)-iz)
                            *(2 if ix else 1)*(2 if iz else 1))
            distance = float(np.linalg.norm(observer))
            subdivisions = 8 if distance<=20 else 1
            exact = quadrature = 0.
            face_count = 0
            for axis in (0,2):
                if observer[axis] <= half[axis]+1e-6: continue
                h,u0,u1,v0,v1,reference,triangle = face_geometry(observer,axis,half)
                gauss32,gauss64 = [gauss_angle(h,u0,u1,v0,v1,rule) for rule in rules]
                midpoint = midpoint_angle(h,u0,u1,v0,v1,subdivisions)
                assert reference>0 and np.isfinite([reference,triangle,gauss32,gauss64,midpoint]).all()
                face_checks.append(dict(offset_x_mm=4*ix,offset_z_mm=4*iz,normal_axis=axis,
                    relative_midpoint_bias=midpoint/reference-1,
                    triangle_relative_difference=triangle/reference-1,
                    gauss64_relative_difference=gauss64/reference-1,
                    gauss32_64_relative_difference=gauss32/gauss64-1))
                exact += reference
                quadrature += midpoint
                face_count += 1
            rows.append(dict(offset_x_mm=4*ix,offset_z_mm=4*iz,ordered_pair_multiplicity=multiplicity,
                centre_distance_mm=distance,subdivisions=subdivisions,visible_faces=face_count,
                analytic_solid_angle_sr=exact,midpoint_solid_angle_sr=quadrature,
                relative_midpoint_bias=quadrature/exact-1))
    branch_summary = {}
    for branch,n in (('near',8),('far',1)):
        group = [r for r in rows if r['subdivisions']==n]
        minimum = min(group,key=lambda r:r['relative_midpoint_bias'])
        maximum = max(group,key=lambda r:r['relative_midpoint_bias'])
        exact_sum = math.fsum(r['ordered_pair_multiplicity']*r['analytic_solid_angle_sr'] for r in group)
        midpoint_sum = math.fsum(r['ordered_pair_multiplicity']*r['midpoint_solid_angle_sr'] for r in group)
        branch_summary[branch] = dict(unique_absolute_offset_classes=len(group),
            ordered_pairs=sum(r['ordered_pair_multiplicity'] for r in group),
            minimum_bias_class=minimum,maximum_bias_class=maximum,
            multiplicity_weighted_analytic_angle_sum_sr=exact_sum,
            multiplicity_weighted_midpoint_angle_sum_sr=midpoint_sum,
            multiplicity_weighted_geometry_only_relative_bias=midpoint_sum/exact_sum-1)
        key = f'{branch}_ordered_pairs'
        assert branch_summary[branch]['ordered_pairs'] == runtime['surface_quadrature_geometry'][0][key]
    assert len(rows)==2311 and sum(r['ordered_pair_multiplicity'] for r in rows)==5343032
    reference_checks = dict(visible_face_classes=len(face_checks),
        max_abs_triangle_relative_difference=max(abs(r['triangle_relative_difference']) for r in face_checks),
        max_abs_gauss64_relative_difference=max(abs(r['gauss64_relative_difference']) for r in face_checks),
        max_abs_gauss32_64_relative_difference=max(abs(r['gauss32_64_relative_difference']) for r in face_checks),
        minimum_midpoint_face_bias=min(face_checks,key=lambda r:r['relative_midpoint_bias']),
        maximum_midpoint_face_bias=max(face_checks,key=lambda r:r['relative_midpoint_bias']))
    assert reference_checks['max_abs_triangle_relative_difference']<1e-8
    assert reference_checks['max_abs_gauss64_relative_difference']<1e-8
    assert reference_checks['max_abs_gauss32_64_relative_difference']<1e-8
    output_dir = DATA/'surface_solid_angle_read_only'
    output_dir.mkdir(exist_ok=True)
    table = output_dir/'absolute_offset_geometry.csv'
    assert not table.exists(), 'Preserve existing diagnostics; do not overwrite/recompute'
    with table.open('x',newline='',encoding='utf8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    proof = dict(passed=True,scope='Full detector-pair geometry-only visible-face solid-angle midpoint audit',
        scientific_status='HOLD',physical_gate_passed=False,science_job=1677211,
        code_sha256=digest(__file__),scientific_source_sha256=digest(payload/source_name),
        producer_release_key=freeze['release_key'],original_physical_gate_sha256=runtime['original_physical_gate_sha256'],
        accepted_runtime_evidence_sha256=digest(runtime_path),accepted_runtime_acceptance_sha256=digest(REPORT/'physical_scatter_runtime_acceptance.json'),
        original_detector_params_receipt_bindings=bindings,
        geometry=dict(detector_count=2312,dimensions_mm=[4,10,4],rotation=0,
            lattice_shape=[68,34],lattice_spacing_mm=4,absolute_offset_classes=2311,
            total_ordered_nonself_pairs=5343032),branch_summary=branch_summary,reference_checks=reference_checks,
        table_relative_to_data=table.relative_to(DATA).as_posix(),table_sha256=digest(table),
        class_reduction='Translation and sign-reflection symmetry of the identical axis-aligned boxes with common y. Every absolute x/z offset class and its full ordered-pair multiplicity is included. This is an analytic symmetry reduction, not a production sampling change.',
        method='Float64 re-expression of the frozen projected-cosine*area/r^2 factor only. Four-corner analytic rectangle integral, independently checked by two spherical triangles and 32x32/64x64 Gauss integration on every visible face class. Logged near8/far1 rule retained.',
        aggregate_definition='Pair-multiplicity weighted sums of solid angles only, with all other surface factors held constant. Neither source nor energy nor window nor attenuation nor absorption weights are evaluated.',
        no_response_or_scatter_kernel_or_simulation_or_reconstruction_evaluated=True,
        no_production_input_modified=True,no_gpu_or_slurm_job_submitted=True,no_observed_count_fit=True,
        limitations='Finite geometry-class comparisons and numerical agreement checks are not rigorous floating-point bounds, captured CUDA states or a full integrand error bound. Actual angular density, window, attenuation and target absorption vary across each face and are omitted here. Geometric factor errors cannot prove or exclude a24% source-weighted cross-window response deficit. Root cause remains UNDETERMINED; original22 HOLD and138720 inadequate bin diagnostics persist.',
        elapsed_seconds=time.monotonic()-began)
    write(REPORT/'physical_surface_solid_angle_read_only.json',proof)
    assert digest(REPORT/'physical_gate.json') == proof['original_physical_gate_sha256']
    print('GEOMETRY_ONLY_BRANCHES',branch_summary)
    print('REFERENCE_CHECKS',reference_checks)
    print('HOLD_UNCHANGED',True,'elapsed_seconds',proof['elapsed_seconds'])


if __name__ == '__main__': audit()
