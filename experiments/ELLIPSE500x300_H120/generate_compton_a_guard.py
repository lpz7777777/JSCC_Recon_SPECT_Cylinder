"""Supplement A440 interpolation support using the original production binaries.

The original matrices and calibration remain read-only. An exact common-grid
anchor must pass before any halo is generated. No detector row is omitted:
ScatterGen needs the tungsten rows as first-interaction source bins as well.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

ENGINE_REL = 'Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main'
SOURCE_RUN = 'runs/JSCC_440keV_pe_v4_ELLIPSE500x300_H120'
PE_HASH = '58f1fbc62b7c24f572a0a57818cded6d2278a7c988097ed9832d0bbe57af9c94'
SCATTER_HASH = '717ce146cd56a00717cf0a8d440e464578fa80490a075ab867b3cce832596c2c'
NDET = 11520


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def specs():
    # Existing Cartesian core: x/y=-252:6:252, z=-58.5:3:58.5.
    # Halo covers r<=255 and |z|<=60 without extrapolation.
    return [
        dict(name='anchor', shape=(3, 3, 2), spacing=(6, 6, 3), shift=(0, 0, 0)),
        dict(name='z_minus', shape=(87, 87, 1), spacing=(6, 6, 3), shift=(0, 0, -60)),
        dict(name='z_plus', shape=(87, 87, 1), spacing=(6, 6, 3), shift=(0, 0, 60)),
        dict(name='x_minus', shape=(1, 87, 40), spacing=(6, 6, 3), shift=(-258, 0, 0)),
        dict(name='x_plus', shape=(1, 87, 40), spacing=(6, 6, 3), shift=(258, 0, 0)),
        dict(name='y_minus', shape=(85, 1, 40), spacing=(6, 6, 3), shift=(0, -258, 0)),
        dict(name='y_plus', shape=(85, 1, 40), spacing=(6, 6, 3), shift=(0, 258, 0)),
        # Independent axial midpoints test interpolation, not clamp vs extrapolate.
        dict(name='mid_minus', shape=(3, 3, 1), spacing=(252, 150, 3), shift=(0, 0, -59.25)),
        dict(name='mid_plus', shape=(3, 3, 1), spacing=(252, 150, 3), shift=(0, 0, 59.25)),
    ]


def axes(spec):
    return [((np.arange(n) - (n - 1) / 2) * d + s)
            for n, d, s in zip(spec['shape'], spec['spacing'], spec['shift'])]


def validate_anchor(folder, source, spec):
    x, y, z = axes(spec)
    ix = np.rint((x + 252) / 6).astype(int)
    iy = np.rint((y + 252) / 6).astype(int)
    iz = np.rint((z + 58.5) / 3).astype(int)
    metrics = {}
    for kind, new_name in [('PE_SysMat', 'pe.sysmat'),
                           ('PE_Windowed_SysMat', 'pe_windowed.sysmat'),
                           ('Scatter_SysMat', 'Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat'),
                           ('SysMat_withScatter', 'SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat')]:
        suffix = '_v4' if kind.startswith('PE_') else ''
        parent = source / f'{kind}_shift_0.000000_0.000000_0.000000{suffix}.sysmat'
        old = np.memmap(parent, mode='r', dtype='<f4', shape=(NDET, 40, 85, 85))
        expected = np.asarray(old[:, iz[:, None, None], iy[None, :, None], ix[None, None, :]])
        actual = np.fromfile(folder / new_name, dtype='<f4').reshape(expected.shape)
        delta = actual.astype(float) - expected
        rel_l2 = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30))
        # Near-zero terms use a global numerical floor; a relative L2 alone
        # must not hide a wrong detector or a changed sharp response bin.
        floor = float(expected.max()) * 1e-8
        elementwise = bool(np.all(np.abs(delta) <= 1e-5 * np.abs(expected) + floor))
        metrics[kind] = dict(relative_l2=rel_l2, bitwise_equal=bool(np.array_equal(actual, expected)),
                             max_abs_error=float(np.abs(delta).max()), numerical_floor=floor,
                             elementwise_passed=elementwise, passed=rel_l2 <= 1e-5 and elementwise)
    gate = dict(status='PASSED' if all(v['passed'] for v in metrics.values()) else 'HOLD',
                voxels=int(np.prod(spec['shape'])), detectors=NDET, metrics=metrics,
                original_inputs_read_only=True, calibration_refitted=False)
    write(folder / 'anchor_gate.json', gate)
    if gate['status'] != 'PASSED': raise ValueError('Common-grid anchor failed; no halo may run')
    return gate


def run_one(spec, base, source, pe, scatter, cuda):
    folder = base / spec['name']
    folder.mkdir(exist_ok=False)
    parameters = {}
    for original in sorted(source.glob('Params_*.dat')):
        if original.name == 'Params_Image.dat':
            image = np.fromfile(original, dtype='<f4')
            if not np.array_equal(image[:7], [85, 85, 40, 6, 6, 3, 1]) or image[11] != 270:
                raise ValueError('Wrong source matrix geometry')
            image[:3] = spec['shape']; image[3:6] = spec['spacing']; image[8:11] = spec['shift']
            image.tofile(folder / original.name)
        else: shutil.copy2(original, folder / original.name)
        parameters[original.name] = digest(folder / original.name)
    count = int(np.prod(spec['shape']))
    if NDET * count >= 2**31: raise ValueError('Unsafe legacy ScatterGen index range')
    if len(parameters) != 4: raise ValueError('Expected four frozen parameter files')
    write(folder / 'input_manifest.json', dict(spec=spec, params_sha256=parameters,
        original_params_sha256={p.name: digest(p) for p in source.glob('Params_*.dat')},
        pe_binary_sha256=digest(pe), scatter_binary_sha256=digest(scatter),
        matrix_shape_detector_zyx=[NDET, *reversed(spec['shape'])], new_transport_photons=0))
    environment = os.environ.copy()
    # Original production wrapper supplied no optional ScatterGen model overrides.
    for key in list(environment):
        if key.startswith(('SCATTER_', 'DETECTOR_LOCAL_', 'COLLIMATOR_SCATTER_')): del environment[key]
    start = time.monotonic()
    command = [str(pe), '--cuda', str(cuda), '--face-subdiv', '16', '--rows-per-chunk', '4',
        '--samples-per-launch', '32', '--output-unwindowed', str(folder/'pe.sysmat'),
        '--output-windowed', str(folder/'pe_windowed.sysmat'), '--manifest', str(folder/'pe_manifest.json'),
        '--progress', str(folder/'pe_progress.json'), '--log', str(folder/'pe_progress.tsv')]
    with (folder/'pe_console.log').open('w') as f:
        subprocess.run(command, cwd=folder, env=environment, stdout=f, stderr=subprocess.STDOUT,
                       check=True, timeout=1200)
    for name in ('pe.sysmat', 'pe_windowed.sysmat'):
        if (folder/name).stat().st_size != NDET * count * 4: raise ValueError('PE size mismatch')
    pe_seconds = time.monotonic() - start
    print(json.dumps(dict(part=spec['name'], phase='PE_COMPLETED', seconds=pe_seconds)), flush=True)
    with (folder/'scatter_console.log').open('w') as f:
        subprocess.run([str(scatter), '-PE', str(folder/'pe.sysmat'), '-cuda', str(cuda)],
            cwd=folder, env=environment, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=2400)
    stem = 'shift_' + '_'.join(f'{s:.6f}' for s in spec['shift'])
    matrices = {}
    for name in ('pe.sysmat', 'pe_windowed.sysmat', f'Scatter_SysMat_{stem}.sysmat', f'SysMat_withScatter_{stem}.sysmat'):
        file = folder/name
        if file.stat().st_size != NDET * count * 4: raise ValueError('Guard matrix size mismatch')
        values = np.memmap(file, mode='r', dtype='<f4')
        for offset in range(0, len(values), 1<<20):
            block = values[offset:offset+(1<<20)]
            if not np.isfinite(block).all() or np.any(block < 0): raise ValueError('Invalid guard response')
        matrices[name] = dict(bytes=file.stat().st_size, sha256=digest(file))
    result = dict(spec=spec, status='COMPLETE', matrices=matrices, pe_seconds=pe_seconds,
                  elapsed_seconds=time.monotonic()-start, calibration_refitted=False)
    if spec['name'] == 'anchor': result['anchor'] = validate_anchor(folder, source, spec)
    write(folder/'complete.json', result)
    print(json.dumps(dict(part=spec['name'], phase='COMPLETE', seconds=result['elapsed_seconds'])), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cuda', type=int, default=0); a = p.parse_args()
    engine = a.root/ENGINE_REL; source = engine/SOURCE_RUN
    pe = engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter = engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe) != PE_HASH or digest(scatter) != SCATTER_HASH:
        raise ValueError('Original production binary hash differs; do not regenerate with another model')
    provenance = json.loads((source/'ELLIPSE_inputs.json').read_text())
    for name, value in provenance['parameters'].items():
        if digest(source/name) != value: raise ValueError('Original matrix parameters changed')
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'plan.json', dict(parts=specs(), original_production=provenance,
        extra_voxels=sum(int(np.prod(s['shape'])) for s in specs()),
        note='A440 interpolation halo only; not a new imaging grid or an event selection grid.'))
    results = []
    for spec in specs():
        results.append(run_one(spec, a.output, source, pe, scatter, a.cuda))
        write(a.output/'progress.json', dict(completed_parts=len(results), total_parts=len(specs()),
                                            latest=results[-1]['spec']['name']))
    write(a.output/'guard_ready.json', dict(status='READY_FOR_INTERPOLATION_VALIDATION',
        parts=results, original_params_sha256=provenance['parameters'],
        pe_binary_sha256=PE_HASH, scatter_binary_sha256=SCATTER_HASH,
        original_matrices_read_only=True, original_calibration_unchanged=True,
        new_transport_photons=0, reconstruction_submitted=False))


if __name__ == '__main__': main()
