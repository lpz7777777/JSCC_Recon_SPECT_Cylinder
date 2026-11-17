"""Bounded I/O recovery; old sources, complete A fields and C slab 0 are read-only.

The original C physics flags disable combined output. Its native Scatter_SysMat
is the same input used by the original run_factor_conversion.py. No kernel,
physics parameters, calibration, cell volume or reconstruction source changes.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import psutil


def response_name(channel, physics, zmean):
    if channel not in ('A440', 'A218', 'C440to218'):
        raise ValueError('Unknown response channel')
    cross = channel == 'C440to218'
    if physics[2] != 1 or physics[3] != (0 if cross else 1):
        raise ValueError('Frozen output flags disagree with response channel')
    prefix = 'Scatter_SysMat' if cross else 'SysMat_withScatter'
    return f'{prefix}_shift_0.000000_0.000000_{zmean:.6f}.sysmat'


def compact(raw, selected, scales, matrix, first):
    for offset in range(0, len(selected), 32):
        subset = selected[offset:offset+32]
        values = np.asarray(raw[subset], dtype=np.float64)*scales[offset:offset+32, None, None, None]*64
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Nonfinite/negative Cartesian response')
        matrix[offset:offset+len(subset), first:first+2] = values.astype('<f4')


def validate_reused_slab(part, parameters, spec, size):
    for name in ['Params_Detector.dat', 'Params_Physics.dat', 'Params_Collimator.dat']:
        if (part/name).read_bytes() != (parameters/name).read_bytes():
            raise ValueError('Original slab parameter differs: '+name)
    if (part/'Params_Image.dat').read_bytes() != spec.astype('<f4').tobytes():
        raise ValueError('Original slab grid differs')
    progress = json.loads((part/'pe_progress.json').read_text())
    if progress['status'] != 'complete' or progress['completed_rows'] != 11520:
        raise ValueError('Original PE did not complete')
    tail = (part/'scatter_console.log').read_text()[-6000:]
    if 'Compton Scatter Sysmat written.' not in tail or 'scatterStart=11504' not in tail:
        raise ValueError('Original scatter completion marker missing')
    native = response_name('C440to218', np.fromfile(parameters/'Params_Physics.dat', '<f4'), float(spec[10]))
    files = ['pe.sysmat', 'pe_windowed.sysmat', native]
    for name in files:
        if (part/name).stat().st_size != size:
            raise ValueError('Original slab byte count differs')
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.plan.read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((args.plan.parent/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Repair release changed: '+name)
    plan = manifest['plan']
    root, old, out = [Path(plan[k]) for k in ['original_release', 'original_output', 'output']]
    sys.path.insert(0, str(root))
    from common import read, write, sha, verify_release, now
    from build_matrix import gpu_info, run_bounded, hashlib_slab
    original = verify_release(root)
    if original['key'] != plan['scientific_release']:
        raise ValueError('Wrong scientific release')
    # Full PID identity, then all exact old executable arguments. Do not kill.
    oldreg = plan['original_registration']
    if sha(old.parent/'matrix.stderr.log') != plan['original_stderr_sha256']:
        raise ValueError('Original failure evidence changed')
    for proc in psutil.process_iter(['pid', 'create_time', 'cmdline']):
        if proc.pid == os.getpid():
            continue
        argv = proc.info['cmdline'] or []
        if (proc.pid == oldreg['pid'] and abs(proc.info['create_time']-oldreg['create_time']) < .1) or any(
                x in argv for x in [str(root/'build_matrix.py'), str(root/'bin/pe'), str(root/'bin/scatter')]):
            raise RuntimeError('An original matrix process is still active')
    if out.exists():
        raise FileExistsError('Recovery output already exists; never overwrite or blindly retry')
    cuda = plan['cuda']
    g = gpu_info(cuda)
    if g['used_bytes']+(12<<30) > .8*g['total_bytes']:
        raise MemoryError('Selected GPU lacks measured reserve')
    available = psutil.virtual_memory().available
    host_budget = min(available, 32<<30)
    disk = shutil.disk_usage(old)
    if host_budget < 16<<30 or disk.free < (40<<30)/.8:
        raise MemoryError('Host/disk reserve not met')
    out.mkdir()
    write(out/'started.json', dict(time=now(), pid=os.getpid(), create_time=psutil.Process().create_time(),
        cmdline=psutil.Process().cmdline(), scientific_release=original['key'], execution_repair=manifest['key'],
        gpu=g, host_budget_bytes=host_budget, directory_free_bytes=disk.free))
    products = {}
    for channel in ['A440', 'A218']:
        write(out/'progress.json', dict(time=now(), channel=channel, phase='verify_reused_complete_matrix'), replace=True)
        src = old/channel
        record = read(src/'completed.json')
        source = src/'B_detector_zyx.float32'
        if record['release'] != original['key'] or record['bytes'] != 10496*30*130*130*4 or sha(source) != record['sha256']:
            raise ValueError('Completed original matrix SHA/size/release failed')
        dest = out/channel
        dest.mkdir()
        os.link(source, dest/source.name)  # no duplicate 42.6 GB copy; never modify
        shutil.copyfile(src/'completed.json', dest/'completed.json')
        products[channel] = record
        write(dest/'reuse.json', dict(source=str(src), sha256=record['sha256'], verified=now(), hardlink_read_only=True))
    channel = 'C440to218'
    dest = out/channel
    dest.mkdir()
    parameters = root/'parameters'/channel
    geometry = np.fromfile(parameters/'Params_Image.dat', '<f4')
    assert np.array_equal(geometry[:7], [85, 85, 40, 6, 6, 3, 1]) and geometry[11] == 270
    physics = np.fromfile(parameters/'Params_Physics.dat', '<f4')
    detector = np.fromfile(parameters/'Params_Detector.dat', '<f4')[1:].reshape(11520, 12)
    selected = np.flatnonzero(detector[:, 11] == 1)
    assert len(selected) == 10496
    factor = read(parameters/'factor_manifest.json')
    scales = {m['layer_mm']: m['scale'] for m in factor['calibration']['metrics']}
    scale = np.array([scales[float(abs(y+270))] for y in detector[selected, 1]])
    assert np.isfinite(scale).all() and (scale > 0).all()
    matrix = np.memmap(dest/'B_detector_zyx.float32.partial', mode='w+', dtype='<f4', shape=(10496, 30, 130, 130))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('SCATTER_', 'DETECTOR_LOCAL_', 'COLLIMATOR_SCATTER_')):
            del env[key]
    raw_size = 11520*2*130*130*4
    slabs = []
    for first in range(0, 30, 2):
        z = -58+4*np.arange(first, first+2)
        spec = geometry.copy()
        spec[:3] = [130, 130, 2]
        spec[3:6] = [4, 4, 4]
        spec[8:11] = [0, 0, z.mean()]
        native = response_name(channel, physics, z.mean())
        part = dest/f'slab_{first:02d}'
        part.mkdir()
        if first == 0:
            rawpart = old/channel/'slab_00'
            files = validate_reused_slab(rawpart, parameters, spec, raw_size)
            pe_record = dict(reused=True, manifest=read(rawpart/'pe_manifest.json'))
            scatter_record = dict(reused=True, exit_zero_inferred_from_original_run_bounded_return=True,
                original_exception_after_run_bounded=True, log_sha256=sha(rawpart/'scatter_console.log'),
                original_elapsed_resource_receipt_not_saved=True)
        else:
            rawpart = part
            for name in ['Params_Detector.dat', 'Params_Physics.dat', 'Params_Collimator.dat']:
                shutil.copyfile(parameters/name, part/name)
            spec.tofile(part/'Params_Image.dat')
            write(out/'progress.json', dict(channel=channel, first_z=first, total_z=30, phase='PE', time=now()), replace=True)
            pe_record = run_bounded([str(root/'bin/pe'), '--cuda', str(cuda), '--face-subdiv', '16',
                '--rows-per-chunk', '4', '--samples-per-launch', '32',
                '--output-unwindowed', str(part/'pe.sysmat'), '--output-windowed', str(part/'pe_windowed.sysmat'),
                '--manifest', str(part/'pe_manifest.json'), '--progress', str(part/'pe_progress.json'),
                '--log', str(part/'pe_progress.tsv')], part, env, part/'pe_console.log', cuda, host_budget, 4*3600)
            write(out/'progress.json', dict(channel=channel, first_z=first, total_z=30, phase='detector_scatter', time=now()), replace=True)
            scatter_record = run_bounded([str(root/'bin/scatter'), '-PE', str(part/'pe.sysmat'), '-cuda', str(cuda)],
                part, env, part/'scatter_console.log', cuda, host_budget, 4*3600)
            files = ['pe.sysmat', 'pe_windowed.sysmat', native]
        for name in files:
            if (rawpart/name).stat().st_size != raw_size:
                raise ValueError('Generated raw matrix byte count differs')
        raw = np.memmap(rawpart/native, mode='r', dtype='<f4', shape=(11520, 2, 130, 130))
        compact(raw, selected, scale, matrix, first)
        matrix.flush()
        del raw
        gc.collect()
        receipt = dict(first_z=first, z_mm=z.tolist(), pe=pe_record, scatter=scatter_record,
            raw_source=str(rawpart), selected_native_filename=native,
            raw_sha256={name: sha(rawpart/name) for name in files}, compact_slab_sha256=hashlib_slab(matrix, first),
            factor_manifest_sha256=sha(parameters/'factor_manifest.json'), volume_mm3=64.,
            fraction_applied=False, native_grid_is_not_polar=True, execution_repair=manifest['key'])
        write(part/'complete.json', receipt)
        slabs.append(receipt)
        # Original failed slab 0 stays untouched. Only new intermediate files.
        if first:
            for name in files:
                q = (part/name).resolve()
                if q.parent != part.resolve() or not q.is_relative_to(out.resolve()):
                    raise ValueError('Cleanup escaped own recovery output')
                q.unlink()
        print('CARTESIAN_REPAIR_SLAB', channel, first+2, 30, flush=True)
    del matrix
    final = dest/'B_detector_zyx.float32'
    os.replace(dest/'B_detector_zyx.float32.partial', final)
    complete = dict(channel=channel, shape=[10496, 30, 130, 130], layout='detector_zyx', bytes=final.stat().st_size,
        sha256=sha(final), release=original['key'], execution_repair=manifest['key'], slabs=slabs,
        calibration_refitted=False, volume_mm3=64., overlap_applied=False, time=now())
    write(dest/'completed.json', complete)
    products[channel] = complete
    write(out/'completed.json', dict(release=original['key'], execution_repair=manifest['key'], time=now(),
        channels=products, reused_complete_channels=['A440', 'A218'], reused_C_slabs=[0], newly_computed_C_slabs=list(range(2,30,2)),
        transport_repeated=False, source_freeze_sha256=sha(root/'release_manifest.json')))


if __name__ == '__main__':
    main()
