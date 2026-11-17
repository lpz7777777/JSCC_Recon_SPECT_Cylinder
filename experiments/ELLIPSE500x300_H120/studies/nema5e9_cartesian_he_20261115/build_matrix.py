"""Three fresh 4 mm Cartesian responses, original immutable production binaries.

Run serially on one explicitly selected GPU. Per-slab raw files are temporary
only inside this NEW job directory and are removed after compacted SHA proof.
No original matrix, source, response calibration or scientific result is edited.
"""
import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import psutil
from common import read, write, sha, verify_release, now

NDET, NSELECTED, NZ, NY, NX = 11520, 10496, 30, 130, 130


def gpu_info(cuda):
    cmd = ['nvidia-smi', '-i', str(cuda), '--query-gpu=uuid,name,memory.total,memory.used',
           '--format=csv,noheader,nounits']
    parts = subprocess.check_output(cmd, text=True).strip().split(',')
    if len(parts) != 4:
        raise ValueError('Cannot measure selected GPU')
    return dict(uuid=parts[0].strip(), name=parts[1].strip(),
                total_bytes=int(parts[2])*1048576, used_bytes=int(parts[3])*1048576)


def run_bounded(cmd, folder, env, log, cuda, host_budget, timeout):
    start = time.monotonic()
    peak_rss = peak_gpu = 0
    with log.open('xb') as stream:
        p = subprocess.Popen(cmd, cwd=folder, env=env, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while p.poll() is None:
                children = [psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
                rss = sum(q.memory_info().rss for q in children if q.is_running())
                gpu = gpu_info(cuda)
                peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, gpu['used_bytes'])
                if rss > .8*host_budget or gpu['used_bytes'] > .8*gpu['total_bytes']:
                    raise MemoryError('Matrix actual global GPU/RSS reserve exceeded')
                if time.monotonic()-start > timeout:
                    raise TimeoutError('Bounded matrix slab exceeded time limit')
                time.sleep(2)
            if p.returncode:
                raise RuntimeError('Matrix subprocess failed; partial output preserved: '+str(log))
        except BaseException:
            # Only this newly created process group; never another user's task.
            if p.poll() is None:
                import signal
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            raise
    return dict(elapsed_seconds=time.monotonic()-start, rss_peak_bytes=peak_rss,
                global_gpu_used_peak_bytes=peak_gpu, argv=cmd)


def build(root, out, cuda, disk_budget):
    release = verify_release(root)
    if (out/'completed.json').exists():
        old = read(out/'completed.json')
        if old['release'] != release['key']:
            raise ValueError('Completed response belongs to different release')
        print('ALREADY_COMPLETE_NO_RECOMPUTATION', flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    disk = shutil.disk_usage(out)
    if disk.free < disk_budget/.8:
        raise OSError('Insufficient directory capacity; no matrix started')
    g = gpu_info(cuda)
    if g['used_bytes']+(12<<30) > .8*g['total_bytes']:
        raise MemoryError('Selected GPU lacks measured 20% reserve for matrix generation')
    available = psutil.virtual_memory().available
    host_budget = min(available, 32<<30)
    if host_budget < 16<<30:
        raise MemoryError('Insufficient currently available host memory')
    write(out/'started.json', dict(time=now(), pid=os.getpid(), create_time=psutil.Process().create_time(),
        cmdline=psutil.Process().cmdline(), release=release['key'], gpu=g,
        available_host_bytes=available, host_budget_bytes=host_budget,
        directory_free_bytes=disk.free, maximum_new_disk_bytes=disk_budget,
        source_response='original binaries and params, only Cartesian grid changed'))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('SCATTER_', 'DETECTOR_LOCAL_', 'COLLIMATOR_SCATTER_')):
            del env[key]
    pe, scatter = root/'bin/pe', root/'bin/scatter'
    for binary in [pe, scatter]:
        binary.chmod(0o700)
    products = {}
    for channel in ['A440', 'A218', 'C440to218']:
        parameters = root/'parameters'/channel
        geometry = np.fromfile(parameters/'Params_Image.dat', '<f4')
        assert np.array_equal(geometry[:7], [85, 85, 40, 6, 6, 3, 1]) and geometry[11] == 270
        raw_detector = np.fromfile(parameters/'Params_Detector.dat', '<f4')
        assert raw_detector[0] == NDET
        detector = raw_detector[1:].reshape(NDET, 12)
        selected = np.flatnonzero(detector[:, 11] == 1)
        assert len(selected) == NSELECTED
        factor = read(parameters/'factor_manifest.json')
        scales = {m['layer_mm']: m['scale'] for m in factor['calibration']['metrics']}
        scale = np.array([scales[float(abs(y+270))] for y in detector[selected, 1]])
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Invalid inherited calibration; refitting forbidden')
        dest = out/channel
        dest.mkdir()
        matrix = np.memmap(dest/'B_detector_zyx.float32.partial', mode='w+', dtype='<f4',
                           shape=(NSELECTED, NZ, NY, NX))
        slabs = []
        for first in range(0, NZ, 2):
            z = -58+4*np.arange(first, first+2)
            part = dest/f'slab_{first:02d}'
            part.mkdir()
            for name in ['Params_Detector.dat', 'Params_Physics.dat', 'Params_Collimator.dat']:
                shutil.copyfile(parameters/name, part/name)
            spec = geometry.copy()
            spec[:3] = [NX, NY, 2]
            spec[3:6] = [4, 4, 4]
            spec[8:11] = [0, 0, z.mean()]
            spec.tofile(part/'Params_Image.dat')
            assert NDET*NX*NY*2 < 2**31
            write(out/'progress.json', dict(channel=channel, first_z=first, total_z=30,
                                            phase='PE', time=now()), replace=True)
            p = run_bounded([str(pe), '--cuda', str(cuda), '--face-subdiv', '16',
                '--rows-per-chunk', '4', '--samples-per-launch', '32',
                '--output-unwindowed', str(part/'pe.sysmat'), '--output-windowed', str(part/'pe_windowed.sysmat'),
                '--manifest', str(part/'pe_manifest.json'), '--progress', str(part/'pe_progress.json'),
                '--log', str(part/'pe_progress.tsv')], part, env, part/'pe_console.log', cuda, host_budget, 4*3600)
            write(out/'progress.json', dict(channel=channel, first_z=first, total_z=30,
                                            phase='detector_scatter', time=now()), replace=True)
            s = run_bounded([str(scatter), '-PE', str(part/'pe.sysmat'), '-cuda', str(cuda)],
                part, env, part/'scatter_console.log', cuda, host_budget, 4*3600)
            stem = f'shift_0.000000_0.000000_{z.mean():.6f}'
            combined = part/f'SysMat_withScatter_{stem}.sysmat'
            raw_size = NDET*2*NY*NX*4
            files = ['pe.sysmat', 'pe_windowed.sysmat', f'Scatter_SysMat_{stem}.sysmat', combined.name]
            for n in files:
                if (part/n).stat().st_size != raw_size:
                    raise ValueError('Generated matrix byte count differs')
            raw = np.memmap(combined, mode='r', dtype='<f4', shape=(NDET, 2, NY, NX))
            for offset in range(0, NSELECTED, 32):
                subset = selected[offset:offset+32]
                values = np.asarray(raw[subset], dtype=np.float64)*scale[offset:offset+32, None, None, None]*64
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Nonfinite/negative Cartesian response')
                matrix[offset:offset+len(subset), first:first+2] = values.astype('<f4')
            matrix.flush()
            del raw
            gc.collect()
            receipt = dict(first_z=first, z_mm=z.tolist(), pe=p, scatter=s,
                raw_sha256={n: sha(part/n) for n in files},
                compact_slab_sha256=hashlib_slab(matrix, first),
                factor_manifest_sha256=sha(parameters/'factor_manifest.json'), volume_mm3=64.,
                fraction_applied=False, native_grid_is_not_polar=True)
            write(part/'complete.json', receipt)
            slabs.append(receipt)
            # These are exclusively this new slab's reproducible intermediate
            # arrays. Original/other-job files cannot match this resolved parent.
            for n in files:
                q = (part/n).resolve()
                if q.parent != part.resolve() or not q.is_relative_to(out.resolve()):
                    raise ValueError('Temporary cleanup escaped this new run')
                q.unlink()
            print('CARTESIAN_MATRIX_SLAB', channel, first+2, 30, flush=True)
        del matrix
        final = dest/'B_detector_zyx.float32'
        os.replace(dest/'B_detector_zyx.float32.partial', final)
        complete = dict(channel=channel, shape=[NSELECTED, NZ, NY, NX], layout='detector_zyx',
            bytes=final.stat().st_size, sha256=sha(final), release=release['key'], slabs=slabs,
            calibration_refitted=False, volume_mm3=64., overlap_applied=False, time=now())
        write(dest/'completed.json', complete)
        products[channel] = complete
    write(out/'completed.json', dict(release=release['key'], time=now(), channels=products,
        transport_repeated=False, grid_only_new_physical_response=True))


def hashlib_slab(matrix, first):
    import hashlib
    h = hashlib.sha256()
    for offset in range(0, NSELECTED, 32):
        h.update(np.ascontiguousarray(matrix[offset:offset+32, first:first+2]).tobytes())
    return h.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', type=int, required=True)
    parser.add_argument('--disk-budget', type=int, default=90<<30)
    args = parser.parse_args()
    build(Path(__file__).resolve().parent, args.output.resolve(), args.cuda, args.disk_budget)
