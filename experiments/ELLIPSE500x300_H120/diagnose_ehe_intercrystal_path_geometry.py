"""Read-only geometry of fixed centre-pair paths at frozen surface nodes.

No energy, attenuation coefficient, survival probability, scatter contribution,
response matrix, simulation, source projection or reconstruction is evaluated.
"""
import csv
import math
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from ehe_common import DATA, REPORT, digest, read, write


def clipped_box_lengths(endpoint, centres, half):
    """Independent segment/slab intersections against every detector box."""
    lower = np.zeros(len(centres)); upper = np.ones(len(centres))
    valid = np.ones(len(centres),dtype=bool)
    for axis in range(3):
        first = centres[:,axis]-half[axis]
        second = centres[:,axis]+half[axis]
        if endpoint[axis] == 0:
            valid &= (first<=0)&(second>=0)
        else:
            a,b = first/endpoint[axis],second/endpoint[axis]
            lower = np.maximum(lower,np.minimum(a,b))
            upper = np.minimum(upper,np.maximum(a,b))
    return np.where(valid,np.maximum(upper-lower,0),0)*np.linalg.norm(endpoint)


def source_exit(endpoint,half):
    distance = float(np.linalg.norm(endpoint))
    nonzero = np.abs(endpoint)>0
    return distance*float(np.min(half[nonzero]/np.abs(endpoint[nonzero])))


def audit():
    began = time.monotonic()
    output_dir = DATA/'intercrystal_path_geometry_read_only'
    table = output_dir/'surface_node_paths.csv'
    result_path = REPORT/'physical_intercrystal_path_geometry_read_only.json'
    assert not table.exists() and not result_path.exists(), 'Preserve existing completed diagnostics'
    solid_path = REPORT/'physical_surface_solid_angle_read_only.json'
    solid = read(solid_path)
    solid_acceptance = read(REPORT/'physical_surface_solid_angle_acceptance.json')
    assert solid['passed'] and solid_acceptance['passed']
    assert digest(solid_path) == solid_acceptance['geometric_evidence_sha256']
    class_table = DATA/solid['table_relative_to_data']
    assert digest(class_table) == solid['table_sha256'] == solid_acceptance['geometric_class_table_sha256']
    classes = list(csv.DictReader(class_table.open(encoding='utf8',newline='')))
    assert len(classes) == 2311
    gate_sha = digest(REPORT/'physical_gate.json')
    assert gate_sha == solid['original_physical_gate_sha256'] and not read(REPORT/'physical_gate.json')['passed']
    freeze = read(REPORT/'response_repair_freeze.json')
    payload = DATA/freeze['payload_dir']
    source_name = 'engine/ScatterGen_RayTracing_CircularHole/scatter.cu'
    assert digest(payload/source_name) == freeze['sha256'][source_name] == solid['scientific_source_sha256']
    source = (payload/source_name).read_text()
    function = source[source.index('__device__ float integrateIntercrystalTargetSurface('):]
    function = function[:function.index('__device__ int indexFrombitmap_crystal')]
    assert 'const float source_exit_length = detectorCenterExitDistance(' in function
    assert 'float attenuation = source_exit_length' in function
    assert 'attenuation += pair_path.material_lengths.x' in function
    builder = source[source.index('__global__ void buildStructuredCrystalPairMaterialPaths('):]
    builder = builder[:builder.index('__global__ void reduceCrystalPairMaterialPaths(')]
    assert 'intermediate != scatter && intermediate != target' in builder
    assert 'target_local_x, target_local_y, target_local_z,' in builder
    assert 'material_lengths[material] += length;' in builder
    assert 'buildStructuredCrystalPairMaterialPaths<<<pair_blocks, 256, 0, stream>>>' in source
    runtime = read(REPORT/'physical_scatter_runtime_read_only.json')
    bindings = []
    geometry = None
    for r in runtime['records']:
        response,slab = r['response'],r['slab']
        name = f'params/{response}/Params_Detector.dat'
        receipt_path = REPORT/'response_preservation_1672966'/response/f'slab_{slab}'/'receipt.json'
        log = DATA/'collimator_area_read_only'/f'{response}_slab_{slab}_scatter.log'
        receipt = read(receipt_path)
        assert digest(receipt_path) == r['pre_stop_receipt_sha256']
        assert digest(log) == r['original_log_sha256'] == receipt['files']['scatter.log']
        assert digest(payload/name) == r['frozen_detector_params_sha256'] == freeze['sha256'][name] == receipt['files']['Params_Detector.dat']
        lines = log.read_text().splitlines()
        material_line = 'Detector XCOM materials: NaI=2312 GAGG=0 Pb=0 W=0'
        grid_line = 'Axis-aligned layer-grid traversal: enabled layers=1 cells=2312'
        assert material_line in lines and grid_line in lines
        path_lines = [line for line in lines if line.startswith('Pair material-path generation (layer-grid):')]
        cache_lines = [line for line in lines if line.startswith('Pair material-path cache hit for A=')]
        assert path_lines and not cache_lines
        detector = np.fromfile(payload/name,'<f4')[1:].reshape(2312,12)
        candidate = detector[:,[0,1,2,3,4,5,10]].astype(np.float64)
        if geometry is None: geometry = candidate
        assert np.array_equal(geometry,candidate)
        bindings.append(dict(response=response,slab=slab,params_sha256=digest(payload/name),
            pre_stop_receipt_sha256=digest(receipt_path),original_log_sha256=digest(log),
            material_line=material_line,grid_line=grid_line,
            layer_grid_generation_log_count=len(path_lines),cache_hit_log_count=len(cache_lines)))
    assert len(bindings)==12
    assert np.all(geometry[:,3:6]==[4,10,4]) and np.all(geometry[:,6]==0)
    centres = geometry[:,:3].copy()
    centres -= np.min(centres,axis=0)
    half = geometry[0,3:6]/2
    xs,zs = np.unique(centres[:,0]),np.unique(centres[:,2])
    assert len(xs)==68 and len(zs)==34 and np.all(np.diff(xs)==4) and np.all(np.diff(zs)==4)
    assert np.all(centres[:,1]==0) and len(np.unique(centres,axis=0))==2312
    # Box dimensions equal grid pitch, so their union is a filled convex NaI
    # slab. All segments between a source centre and visible target face stay
    # inside it. This permits an independent total-length closure check.
    index = {(float(c[0]),float(c[2])):i for i,c in enumerate(centres)}
    source_index = index[(0.,0.)]
    rows = []
    centre_max_error = total_max_error = intermediate_max_error = target_max_chord = 0.
    for r in classes:
        dx,dz = float(r['offset_x_mm']),float(r['offset_z_mm'])
        n = int(r['subdivisions'])
        target = np.array([dx,0.,dz])
        target_index = index[(dx,dz)]
        distance = float(np.linalg.norm(target))
        centre_exit = source_exit(target,half)
        centre_intermediate = max(distance-2*centre_exit,0.)
        centre_chords = clipped_box_lengths(target,centres,half)
        centre_chords[[source_index,target_index]]=0
        centre_sum = float(np.sum(centre_chords))
        centre_max_error = max(centre_max_error,abs(centre_sum-centre_intermediate))
        for axis in (0,2):
            if target[axis]<=half[axis]+1e-6: continue
            first,second = (axis+1)%3,(axis+2)%3
            for i in range(n):
                for j in range(n):
                    endpoint = target.copy()
                    endpoint[axis] -= half[axis]
                    endpoint[first] += -half[first]+(i+.5)*2*half[first]/n
                    endpoint[second] += -half[second]+(j+.5)*2*half[second]/n
                    node_distance = float(np.linalg.norm(endpoint))
                    node_exit = source_exit(endpoint,half)
                    exact_intermediate = max(node_distance-node_exit,0.)
                    chords = clipped_box_lengths(endpoint,centres,half)
                    total_max_error = max(total_max_error,abs(float(np.sum(chords))-node_distance))
                    target_max_chord = max(target_max_chord,float(chords[target_index]))
                    source_chord_error = abs(float(chords[source_index])-node_exit)
                    chords[[source_index,target_index]]=0
                    intermediate_sum = float(np.sum(chords))
                    intermediate_max_error = max(intermediate_max_error,source_chord_error,
                                                 abs(intermediate_sum-exact_intermediate))
                    fixed_total = node_exit+centre_intermediate
                    rows.append(dict(offset_x_mm=dx,offset_z_mm=dz,subdivisions=n,
                        ordered_pair_multiplicity=int(r['ordered_pair_multiplicity']),normal_axis=axis,
                        first_midpoint_index=i,second_midpoint_index=j,
                        endpoint_x_mm=float(endpoint[0]),endpoint_y_mm=float(endpoint[1]),endpoint_z_mm=float(endpoint[2]),
                        centre_intermediate_mm=centre_intermediate,node_source_exit_mm=node_exit,
                        node_exact_intermediate_mm=exact_intermediate,node_total_nai_mm=node_distance,
                        fixed_centre_plus_node_exit_nai_mm=fixed_total,
                        fixed_minus_node_exact_nai_mm=fixed_total-node_distance))
    tolerance = 1e-9
    assert max(centre_max_error,total_max_error,intermediate_max_error,target_max_chord)<tolerance
    summaries = {}
    for branch,n in (('near',8),('far',1)):
        group = [r for r in rows if r['subdivisions']==n]
        summaries[branch] = dict(geometry_node_classes=len(group),
            minimum_delta_node=min(group,key=lambda r:r['fixed_minus_node_exact_nai_mm']),
            maximum_delta_node=max(group,key=lambda r:r['fixed_minus_node_exact_nai_mm']),
            positive_delta_node_classes=sum(r['fixed_minus_node_exact_nai_mm']>tolerance for r in group),
            negative_delta_node_classes=sum(r['fixed_minus_node_exact_nai_mm']<-tolerance for r in group),
            near_zero_delta_node_classes=sum(abs(r['fixed_minus_node_exact_nai_mm'])<=tolerance for r in group))
    output_dir.mkdir(exist_ok=True)
    with table.open('x',encoding='utf8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    proof = dict(passed=True,scope='New pure geometric audit of fixed centre-pair intermediary paths at every logged surface midpoint class',
        scientific_status='HOLD',physical_gate_passed=False,science_job=1677211,
        code_sha256=digest(__file__),scientific_source_sha256=digest(payload/source_name),
        producer_release_key=freeze['release_key'],original_physical_gate_sha256=gate_sha,
        previous_surface_geometry_evidence_sha256=digest(solid_path),
        previous_surface_geometry_acceptance_sha256=digest(REPORT/'physical_surface_solid_angle_acceptance.json'),
        previous_absolute_offset_table_sha256=digest(class_table),original_params_receipt_log_bindings=bindings,
        absolute_offset_classes=len(classes),ordered_nonself_pairs=5343032,surface_midpoint_node_classes=len(rows),
        branch_summary=summaries,independent_box_intersections=dict(detector_boxes_per_ray=2312,
            centre_segments_checked=len(classes),surface_segments_checked=len(rows),
            maximum_centre_intermediate_closure_error_mm=centre_max_error,
            maximum_surface_total_closure_error_mm=total_max_error,
            maximum_surface_source_or_intermediate_closure_error_mm=intermediate_max_error,
            maximum_target_chord_before_face_endpoint_mm=target_max_chord,tolerance_mm=tolerance),
        table_relative_to_data=table.relative_to(DATA).as_posix(),table_sha256=digest(table),
        source_observation='Frozen surface helper updates source-crystal exit length for each face direction, but adds the same pair_path.material_lengths. The logged layer-grid builder derives these intermediary lengths from source/target centres and excludes both endpoint boxes.',
        method='Float64 geometry only. Filled NaI slab closure gives centre intermediary d-2*source_exit and midpoint intermediary r-node_source_exit. All2312 box intersections independently check every2311 centre segment and every surface midpoint class. Prior accepted offsets/subdivisions are reused, without redoing the solid-angle integral.',
        sign_definition='Fixed-centre intermediary minus direction-specific intermediary, equivalently fixed-centre intermediary plus node-specific source exit minus total source-centre-to-target-face NaI length. Positive means longer geometric attenuation path under the fixed-centre expression, negative shorter; no attenuation coefficient or exponential was evaluated.',
        class_reduction='Identical axis-aligned boxes fill the68x34 lattice with no gaps. Translation/reflection of each connecting segment stays in the filled slab for every actual pair of that offset, preserving its intermediary length. This is a reference-geometry symmetry reduction, not production downsampling.',
        no_energy_or_attenuation_coefficient_or_survival_or_scatter_contribution_evaluated=True,
        no_response_or_simulation_or_reconstruction_evaluated=True,no_gpu_or_slurm_job_submitted=True,
        no_production_input_modified=True,no_observed_count_fit=True,
        limitations='Reference geometry does not capture actual per-pair CUDA float values or measured histories. No source, angular, energy-window, target absorption or contribution weights are applied. Geometry-node sign counts are not probabilities or response shares, and path-length extrema are not an integrated prediction error or a24% deficit attribution. No correction kernel is implemented. Root cause remains UNDETERMINED; original22 HOLD and138720 inadequate bin diagnostics persist.',
        elapsed_seconds=time.monotonic()-began)
    write(result_path,proof)
    assert digest(REPORT/'physical_gate.json')==gate_sha
    print('PATH_GEOMETRY_BRANCHES',summaries)
    print('INDEPENDENT_BOX_INTERSECTIONS',proof['independent_box_intersections'])
    print('HOLD_UNCHANGED',True,'elapsed_seconds',proof['elapsed_seconds'])


if __name__ == '__main__': audit()
