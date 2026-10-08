"""Conversion-only repair: immutable slabs, RAM interpolation, sequential atomic output.

No Geant4, PE or Scatter executable is launched. The scientific execution release
remains immutable; this separately frozen module changes only output storage.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from ehe_common import allocation, array_write, digest, read, resources, verify_files, write

BINS = 2312
XY = 3301
LAYERS = 40
SLAB_LAYERS = 10
RAW_SHAPE = (BINS, SLAB_LAYERS, 85, 85)
RAW_BYTES = int(np.prod(RAW_SHAPE)) * 4


def fill_slab(cartesian, polar, raw, slab, volume, interpolation, progress=None):
    """Preserve the frozen convert() expression, precision and 64-bin grouping."""
    ix, iy, tx, ty = interpolation
    cartesian[:, slab * SLAB_LAYERS:(slab + 1) * SLAB_LAYERS] = raw
    for k in range(SLAB_LAYERS):
        z = slab * SLAB_LAYERS + k
        for start in range(0, raw.shape[0], 64):
            layer = raw[start:start + 64, k]
            top = layer[:, iy, ix] * (1 - tx) + layer[:, iy, ix + 1] * tx
            bot = layer[:, iy + 1, ix] * (1 - tx) + layer[:, iy + 1, ix + 1] * tx
            polar[z * len(ix):(z + 1) * len(ix), start:start + 64] = (
                (top * (1 - ty) + bot * ty).T * volume[z * len(ix):(z + 1) * len(ix), None])
        if progress:
            progress(z + 1)


def reference_layer(raw, volume, interpolation):
    """Independent full-bin expression for the unchanged bilinear operation."""
    ix, iy, tx, ty = interpolation
    top = raw[:, iy, ix] * (1 - tx) + raw[:, iy, ix + 1] * tx
    bot = raw[:, iy + 1, ix] * (1 - tx) + raw[:, iy + 1, ix + 1] * tx
    return np.asarray((top * (1 - ty) + bot * ty).T * volume[:, None], dtype='<f4')


def memory_digest(array):
    if not array.flags.c_contiguous:
        raise ValueError('Sequential publication requires a contiguous array')
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def publish_array(path, array):
    """One contiguous write, fsync and rename; independently read back its SHA."""
    path = Path(path)
    if path.exists() or path.with_name(path.name + '.writing').exists():
        raise ValueError('Existing output retained; never overwrite: ' + str(path))
    expected = memory_digest(array)
    started = time.monotonic()
    array_write(path, array)
    actual = digest(path)
    if actual != expected or path.stat().st_size != array.nbytes:
        raise ValueError('Sequential output read-back differs')
    return dict(bytes=array.nbytes, sha256=actual, elapsed_seconds=time.monotonic() - started)


def frozen_pipeline(release):
    spec = importlib.util.spec_from_file_location('frozen_ehe_pipeline', release / 'ehe_gpu_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def audit_sources(release, source, output):
    """All 12 receipt member files and both completed full Factors are checked."""
    freeze = read(release / 'release_manifest.json')
    if freeze['release_key'] != '74e129c4460163c5':
        raise ValueError('This repair is bounded to the registered scientific release')
    verify_files(release, freeze['sha256'])
    binary = read(release / 'gpu_binary_manifest.json')
    verify_files(release, binary['files'])
    receipts = {}
    for name in ('A218', 'A440', 'C440to218'):
        for slab in range(4):
            folder = source / name / f'slab_{slab}'
            receipt = read(folder / 'receipt.json')
            if not receipt['passed'] or receipt['pilot'] or receipt['response'] != name or receipt['slab'] != slab or receipt['shape'] != list(RAW_SHAPE):
                raise ValueError('Complete immutable slab identity required')
            for key in ('pe_resource', 'scatter_resource'):
                r = receipt[key]
                if r['host_allocated_bytes'] != 99090432000 or r['rss_fraction'] > .8 or r['gpu_used_fraction'] > .8:
                    raise ValueError('Original computed slab resource certificate failed')
            if (folder / 'response.sysmat').stat().st_size != RAW_BYTES:
                raise ValueError('Full slab length differs')
            verify_files(folder, receipt['files'])
            if name == 'C440to218':
                scatter = [sha for file, sha in receipt['files'].items() if file.startswith('Scatter_SysMat_shift_')]
                if scatter != [receipt['files']['response.sysmat']]:
                    raise ValueError('Cross-window response must remain Scatter-only')
            receipts[f'{name}/{slab}'] = dict(receipt_sha256=digest(folder / 'receipt.json'), receipt=receipt)
            print('IMMUTABLE_SLAB_SHA_PASS', name, slab, flush=True)
    factors = {}
    for name in ('A218', 'A440'):
        folder = source / name
        manifest = read(folder / 'factor_manifest.json')
        if not manifest['passed'] or (manifest['bins'], manifest['full_points'], manifest['active_points']) != (2312, 132040, 78920):
            raise ValueError('Completed Factor identity differs')
        verify_files(folder, manifest['files'])
        verify_files(folder, {n: freeze['sha256'][f'params/{name}/{n}'] for n in ('Params_Collimator.dat', 'Params_Detector.dat', 'Params_Image.dat', 'Params_Physics.dat')})
        if digest(folder / 'whole_geometry.npz') != freeze['sha256']['whole_geometry.npz']:
            raise ValueError('Completed geometry differs')
        s = np.fromfile(folder / 'S_active.float64', '<f8')
        if s.shape != (78920,) or np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError('Completed own sensitivity invalid')
        factors[name] = dict(manifest_sha256=digest(folder / 'factor_manifest.json'), manifest=manifest)
        print('COMPLETE_FACTOR_SHA_PASS', name, flush=True)
    proof = dict(passed=True, producer_release_key=freeze['release_key'], source_root=str(source),
                 original_pipeline_sha256=freeze['sha256']['ehe_gpu_pipeline.py'],
                 original_geometry_sha256=freeze['sha256']['whole_geometry.npz'], binary_manifest=binary,
                 receipts=receipts, factors=factors, no_simulation_or_response_computation=True)
    write(output / 'source_reuse_acceptance.json', proof)
    return proof


def geometry_and_memory(release, alloc):
    g = np.load(release / 'whole_geometry.npz')
    if len(g['coordinates_mm']) != 132040 or len(g['active_indices']) != 78920 or np.any(~np.isin(g['ellipse_fraction'], [0, 1])):
        raise ValueError('Exact full whole geometry required')
    # Arrays plus one slab and independent oracle need under 10 GiB. Refuse an
    # allocation without enough actual headroom before any large allocation.
    if 10 * 1024**3 > .8 * alloc['host_allocated_bytes']:
        raise MemoryError('Insufficient actual allocation for RAM conversion')
    coordinates = g['coordinates_mm']
    interpolation = frozen_pipeline(release).stencil(coordinates[:XY, 0], coordinates[:XY, 1])
    cartesian = np.empty((BINS, LAYERS, 85, 85), dtype='<f4')
    polar = np.empty((XY * LAYERS, BINS), dtype='<f4')
    return g, interpolation, cartesian, polar


def probe(release, source, output, repair):
    output.mkdir(parents=True, exist_ok=False)
    alloc = allocation()
    write(output / 'allocation.json', alloc)
    started = time.monotonic()
    audit_started = time.monotonic()
    audit_sources(release, source, output)
    audit_seconds = time.monotonic() - audit_started
    g, interpolation, cartesian, polar = geometry_and_memory(release, alloc)
    raw = np.fromfile(source / 'C440to218/slab_0/response.sysmat', '<f4').reshape(RAW_SHAPE)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError('Full real source slab invalid')
    t = time.monotonic()
    fill_slab(cartesian, polar, raw, 0, g['cell_volume_mm3'], interpolation)
    conversion_seconds = time.monotonic() - t
    for k in range(10):
        oracle = reference_layer(raw[:, k], g['cell_volume_mm3'][k * XY:(k + 1) * XY], interpolation)
        if not np.array_equal(oracle, polar[k * XY:(k + 1) * XY]):
            raise ValueError('Actual full-bin/full-point numerical equality failed')
    # This is actual converted slab 0, not synthetic I/O; the full stage reuses
    # these accepted bytes and does not repeat this conversion.
    cart = publish_array(output / 'slab0_cartesian.float32', np.ascontiguousarray(cartesian[:, :10]))
    pol = publish_array(output / 'slab0_polar.float32', polar[:10 * XY])
    proof = dict(passed=True, repair_key=repair['repair_key'], producer_release_key='74e129c4460163c5',
                 checked_xy_layers=10, checked_bins=BINS, checked_points=10 * XY,
                 bitwise_equal_to_full_bin_original_expression=True, relative_l2=0.,
                 frozen_pipeline_sha256=digest(release / 'ehe_gpu_pipeline.py'),
                 source_acceptance_sha256=digest(output / 'source_reuse_acceptance.json'),
                 source_audit_seconds=audit_seconds, slab_conversion_seconds=conversion_seconds,
                 sequential_publication={'cartesian': cart, 'polar': pol},
                 resource=resources(alloc), elapsed_seconds=time.monotonic() - started,
                 files={n: digest(output / n) for n in ('slab0_cartesian.float32', 'slab0_polar.float32', 'source_reuse_acceptance.json', 'allocation.json')})
    write(output / 'probe_acceptance.json', proof)
    print('ACTUAL_CONVERSION_PROBE_PASS', proof['elapsed_seconds'], conversion_seconds, cart['elapsed_seconds'], pol['elapsed_seconds'], flush=True)


def complete(release, source, output, probe_root, repair):
    proof = read(probe_root / 'probe_acceptance.json')
    if not proof['passed'] or proof['repair_key'] != repair['repair_key'] or not proof['bitwise_equal_to_full_bin_original_expression']:
        raise ValueError('Actual exact conversion probe required')
    verify_files(probe_root, proof['files'])
    output.mkdir(parents=True, exist_ok=False)
    alloc = allocation()
    write(output / 'allocation.json', alloc)
    started = time.monotonic()
    reuse = audit_sources(release, source, output)
    if digest(output / 'source_reuse_acceptance.json') != proof['source_acceptance_sha256']:
        raise ValueError('Accepted immutable source identity changed')
    for name in ('A218', 'A440'):
        (output / name).symlink_to(source / name, target_is_directory=True)
    target = output / 'C440to218'
    target.mkdir()
    g, interpolation, cartesian, polar = geometry_and_memory(release, alloc)
    cartesian[:, :10] = np.fromfile(probe_root / 'slab0_cartesian.float32', '<f4').reshape(RAW_SHAPE)
    polar[:10 * XY] = np.fromfile(probe_root / 'slab0_polar.float32', '<f4').reshape(10 * XY, BINS)
    for slab in range(1, 4):
        raw = np.fromfile(source / f'C440to218/slab_{slab}/response.sysmat', '<f4').reshape(RAW_SHAPE)
        if np.any(~np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError('Full real source slab invalid')
        def progress(layers):
            write(output / 'progress.json', dict(stage='RAM interpolation', completed_layers=layers, total_layers=40,
                                                elapsed_seconds=time.monotonic() - started))
            print('CONVERSION_LAYER', layers, 40, flush=True)
        fill_slab(cartesian, polar, raw, slab, g['cell_volume_mm3'], interpolation, progress)
        del raw
    if np.any(~np.isfinite(polar)) or np.any(polar < 0):
        raise ValueError('Full density response invalid')
    write(output / 'progress.json', dict(stage='Sequential atomic publication', completed_layers=40, total_layers=40))
    published = {'cartesian': publish_array(target / 'SysMat_cartesian', cartesian),
                 'polar': publish_array(target / 'SysMat_polar', polar)}
    sums = np.asarray(polar.sum(axis=1, dtype=np.float64))
    sens = np.zeros(78920, np.float64)
    for view in range(20):
        sens += sums[g['inverse_rotation'][g['active_indices'], view]] / 20
    if np.any(~np.isfinite(sens)) or np.any(sens <= 0):
        raise ValueError('HOLD: original activity-domain sensitivity not finite positive')
    array_write(target / 'S_active.float64', sens)
    array_write(target / 'S_full.float64', sums)
    del polar, cartesian
    shutil.copy2(release / 'whole_geometry.npz', target / 'whole_geometry.npz')
    for p in (release / 'params/C440to218').glob('Params_*.dat'):
        shutil.copy2(p, target / p.name)
    files = ['SysMat_cartesian', 'SysMat_polar', 'S_active.float64', 'S_full.float64', 'whole_geometry.npz']
    files += [p.name for p in (release / 'params/C440to218').glob('Params_*.dat')]
    write(target / 'factor_manifest.json', dict(passed=True, response='C440to218', bins=BINS,
        full_points=132040, active_points=78920,
        density_equation='B=A diag(volume_mm3), full cell volume exactly once; own S=sum_views(B)/20',
        sensitivity_min=float(sens.min()), sensitivity_max=float(sens.max()),
        files={n: digest(target / n) for n in files}))
    completion = dict(passed=True, repair_key=repair['repair_key'], producer_release_key='74e129c4460163c5',
        source_reuse_acceptance_sha256=digest(output / 'source_reuse_acceptance.json'),
        probe_acceptance_sha256=digest(probe_root / 'probe_acceptance.json'),
        reused_complete_factors=['A218', 'A440'], reused_probe_layers=10, converted_layers=30,
        total_layers=40, density_volume_applied_once=True, scientific_algorithm_unchanged=True,
        publication=published, resource=resources(alloc), elapsed_seconds=time.monotonic() - started,
        factors={n: digest(output / n / 'factor_manifest.json') for n in ('A218', 'A440', 'C440to218')})
    write(output / 'conversion_acceptance.json', completion)
    write(output / 'response_summary.json', dict(passed=True, pilot=False, release_key='74e129c4460163c5',
        stages={n: v['receipt'] for n, v in reuse['receipts'].items()}, conversion_repair=completion))
    write(output / 'progress.json', dict(stage='Complete', completed_layers=40, total_layers=40, passed=True))
    print('CONVERSION_ONLY_COMPLETE', completion['elapsed_seconds'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['probe', 'complete'])
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--source-responses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe', type=Path)
    p.add_argument('--repair-manifest', type=Path, required=True)
    a = p.parse_args()
    repair = read(a.repair_manifest)
    verify_files(a.repair_manifest.parent, repair['files'])
    stop = read(a.repair_manifest.parent / 'source_stop_acceptance.json')
    if not stop['passed'] or stop['job'] != 1672966 or not stop['fully_exited'] or not stop['explicit_human_stop']:
        raise ValueError('Explicitly authorized source job must fully exit')
    if a.output.exists() or a.output.resolve() == a.source_responses.resolve():
        raise ValueError('Preserve old output and refuse partial destination overwrite')
    if a.stage == 'probe':
        probe(a.release, a.source_responses, a.output, repair)
    else:
        if a.probe is None:
            raise ValueError('Accepted actual probe required')
        complete(a.release, a.source_responses, a.output, a.probe, repair)


if __name__ == '__main__':
    main()
