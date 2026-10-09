"""Extract original model component sums and classify surface quadrature geometry.

No kernel evaluation, response generation, simulation or source-weighted counts.
"""
from decimal import Decimal
import math
import re
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from ehe_common import DATA, REPORT, RESPONSES, digest, read, write


def printed_interval(token):
    value = Decimal(token)
    half_unit = Decimal(5).scaleb(value.as_tuple().exponent-1)
    return float(value-half_unit),float(value+half_unit)


def audit():
    began = time.monotonic()
    freeze = read(REPORT/'response_repair_freeze.json')
    payload = DATA/freeze['payload_dir']
    source_name = 'engine/ScatterGen_RayTracing_CircularHole/scatter.cu'
    assert digest(payload/source_name) == freeze['sha256'][source_name]
    source = (payload/source_name).read_text()
    gate_sha = digest(REPORT/'physical_gate.json')
    assert not read(REPORT/'physical_gate.json')['passed']
    # Bind the meaning of the logged numbers: crystal includes the local kernel
    # before copying, and both complete arrays are summed before their addition.
    local = source.index('if (deviceLocalScatterLookup != NULL && include_global_components)')
    copy = source.index('cudaMemcpyAsync(h_Crystal_SysMat, deviceMatrix_crystal,',local)
    sums = source.index('double crystal_scatter_sum = 0.0;',copy)
    assert local < copy < sums
    body = source[sums:source.index('// Release Sources',sums)]
    for expression in ['crystal_scatter_sum += h_Crystal_SysMat[i];',
                       'collimator_scatter_sum += h_Collimator_SysMat[i];',
                       'dst[i] = h_Collimator_SysMat[i] + h_Crystal_SysMat[i];']:
        assert expression in body
    assert 'pair_path.direction_distance.w\n\t\t\t<= nearTargetDistanceFactor * maximum_dimension' in source
    log_binding = read(DATA/'collimator_area_read_only/original_log_binding.json')
    records = []
    geometries = []
    for record in log_binding['records']:
        response,slab = record['response'],record['slab']
        receipt_path = REPORT/'response_preservation_1672966'/response/f'slab_{slab}'/'receipt.json'
        receipt = read(receipt_path)
        log = DATA/'collimator_area_read_only'/f'{response}_slab_{slab}_scatter.log'
        assert digest(log) == record['original_scatter_log_sha256'] == receipt['files']['scatter.log']
        assert digest(receipt_path) == record['pre_stop_receipt_sha256']
        text = log.read_text()
        name = f'params/{response}/Params_Detector.dat'
        assert digest(payload/name) == freeze['sha256'][name] == receipt['files']['Params_Detector.dat']
        detector = np.fromfile(payload/name,'<f4')[1:].reshape(2312,12)
        geometries.append(np.concatenate((detector[:,:6],detector[:,10:]),axis=1))
        selected_lines = [line for line in text.splitlines() if line.startswith((
            'Scatter crystal source range:', 'Diagnostic component matrices:',
            'Inter-crystal target surface quadrature:', 'Scatter component sums:',
            'Detector local scatter switches:'))]
        assert len(selected_lines) == 5
        assert 'Scatter crystal source range: [0,2312) of 2312; detector-local/collimator components=included' in selected_lines
        assert 'Diagnostic component matrices: disabled' in selected_lines
        assert 'Inter-crystal target surface quadrature: far=1x1 near=8x8 near_distance_factor=2' in selected_lines
        expected_self = 0 if response == 'C440to218' else 1
        assert f'Detector local scatter switches: compton=1 recoil_escape=1 self_compton_photoelectric={expected_self}' in selected_lines
        match = re.search(r'^Scatter component sums: crystal=(\S+) collimator=(\S+) collimator_fraction=(\S+)$',text,re.M)
        assert match is not None
        crystal,collimator,fraction = match.groups()
        ci,pi,fi = [printed_interval(token) for token in (crystal,collimator,fraction)]
        interval = [pi[0]/(ci[1]+pi[0]),pi[1]/(ci[0]+pi[1])]
        assert max(interval[0],fi[0]) <= min(interval[1],fi[1])
        assert 'Kernel collimatorScatterSysMatCuda Launched' in text
        records.append(dict(response=response,slab=slab,original_log_sha256=digest(log),
            pre_stop_receipt_sha256=digest(receipt_path),frozen_detector_params_sha256=digest(payload/name),
            original_lines=selected_lines,crystal_sum=float(crystal),collimator_sum=float(collimator),
            printed_fraction=float(fraction),crystal_printed_interval=ci,collimator_printed_interval=pi,
            fraction_interval_from_printed_sums=interval))
    assert len(records) == 12 and all(np.array_equal(geometries[0],g) for g in geometries)
    aggregates = {}
    for response in RESPONSES:
        group = [r for r in records if r['response'] == response]
        assert sorted(r['slab'] for r in group) == [0,1,2,3]
        c = sum(r['crystal_sum'] for r in group)
        p = sum(r['collimator_sum'] for r in group)
        cmin,cmax = [sum(r['crystal_printed_interval'][i] for r in group) for i in (0,1)]
        pmin,pmax = [sum(r['collimator_printed_interval'][i] for r in group) for i in (0,1)]
        aggregates[response] = dict(crystal_sum=c,collimator_sum=p,
            raw_cartesian_collimator_fraction=p/(c+p),
            fraction_interval_from_rounded_logs=[pmin/(cmax+pmin),pmax/(cmin+pmax)])
    geometry = geometries[0]
    assert np.all(geometry[:,3:6] == [4,10,4]) and np.all(geometry[:,6] == 0)
    classifications = []
    for dtype in (np.float32,np.float64):
        xyz = geometry[:,:3].astype(dtype)
        dimensions = geometry[:,3:6].astype(dtype).max(axis=1)
        near = far = boundary = 0
        for index in range(len(xyz)):
            delta = xyz-xyz[index]
            distance = np.sqrt(np.sum(delta*delta,axis=1,dtype=dtype))
            threshold = dtype(2)*np.maximum(dimensions,dimensions[index])
            off_diagonal = np.arange(len(xyz)) != index
            near += int(((distance <= threshold)&off_diagonal).sum())
            far += int(((distance > threshold)&off_diagonal).sum())
            boundary += int(((distance == threshold)&off_diagonal).sum())
        assert near+far == 5343032
        classifications.append(dict(cpu_dtype=np.dtype(dtype).name,near_ordered_pairs=near,
            far_ordered_pairs=far,total_ordered_pairs=near+far,near_fraction=near/(near+far),
            exact_boundary_pairs=boundary,near_threshold_mm=20.0,
            logged_near_surface_subdivisions=8,logged_far_surface_subdivisions=1))
    assert classifications[0]['near_ordered_pairs'] == classifications[1]['near_ordered_pairs']
    proof = dict(passed=True,scope='New analysis of original scalar component logs and near/far geometry selection only',
        scientific_status='HOLD',physical_gate_passed=False,original_job=1672966,science_job=1677211,
        code_sha256=digest(__file__),scientific_source_sha256=digest(payload/source_name),
        producer_release_key=freeze['release_key'],original_physical_gate_sha256=gate_sha,
        records=records,aggregates=aggregates,surface_quadrature_geometry=classifications,
        logged_crystal_sum_definition='Unweighted sum over all 2312 detector bins and 85x85x10 Cartesian sample points per slab, after adding detector-local scatter to the intercrystal buffer. These are scatter-only buffers, not PE plus scatter Factors or source-folded counts.',
        aggregate_definition='Sums of four equal-size disjoint 10-layer slab summaries; six-significant-digit rounded original log values, with propagated rounding intervals. No Polar/active-volume restriction or truth/view weighting.',
        missing_component_information='Diagnostic component matrices were disabled in all12 original runs. Crystal sum does not separate local recoil/self-PE and intercrystal contributions. Source-weighted component fractions cannot be recovered from these scalar sums.',
        surface_geometry_definition='CPU float32/float64 distance classification using the logged factor2 times the maximum size of both detector boxes. This is not captured per-pair CUDA state or contribution-weighted near/far coverage. Equal20mm pairs enter the near branch.',
        no_response_or_kernel_or_simulation_evaluated=True,no_production_input_modified=True,no_observed_count_fit=True,
        limitations='Small raw model collimator fraction does not imply small actual physical collimator contribution or a bound on its possible missing response. Near/far pair counts are not their weighted contribution fractions or quadrature error estimates. No model component was fitted to observations or isolated by rerunning kernels. Root cause remains UNDETERMINED; original22 HOLD and138720 inadequate bin diagnostics persist.',
        elapsed_seconds=time.monotonic()-began)
    write(REPORT/'physical_scatter_runtime_read_only.json',proof)
    assert digest(REPORT/'physical_gate.json') == gate_sha
    print('RAW_MODEL_COMPONENT_AGGREGATES',aggregates)
    print('SURFACE_GEOMETRY',classifications)
    print('HOLD_UNCHANGED',True,'elapsed_seconds',proof['elapsed_seconds'])


if __name__ == '__main__': audit()
