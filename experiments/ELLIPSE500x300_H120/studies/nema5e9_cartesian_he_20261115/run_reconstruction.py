"""Formal three-route native Cartesian MLEM, reusing actual NEMA observations.

Numerical checks run inside this formal allocation. No separate acceptance job,
no image filtering, no width change, no reuse of polar Compton sensitivity.
"""
import argparse
import gc
import json
import os
import resource
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from common import sha, read, write, atomic, now, verify_release
from cartesian import Geometry, ViewResponse, energy_rows, field_coordinates, axes
from detector_csv import load_detector_coordinates
from compton_event_response import ComptonEventSettings, prepare_compton_events, build_detector_position_variance
from compton_energy_probability_v5 import ContinuousTransferLaw
from single_checkpoint_mlem import single_mlem_checkpointed
from jscc_5e10_joint_mlem import joint_mlem
from adapters import OrderedTransferPool, ViewBlocks, current_cuda_budget


def guard(host_bytes):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_reserved()
    if rss > .8*host_bytes or max(total-free, peak) > .8*total:
        raise MemoryError('Actual Cartesian RSS/VRAM exceeds 80% resource limit')
    return dict(rss_peak_bytes=rss, allocated_host_bytes=host_bytes,
                gpu_reserved_peak_bytes=peak, global_gpu_used_bytes=total-free,
                gpu_total_bytes=total)


def sync():
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()


def prepared(rows, det, var, device):
    settings = ComptonEventSettings(.440, .13*np.sqrt(.511/.440),
        2*.440**2/(.511+2*.440)-.001, .05, .350, geometry_mode='stable_float64')
    result, _ = prepare_compton_events(torch.tensor(rows, device=device), settings,
        det, var, var, input_energies_already_smeared=True)
    if result is None or result.count != len(rows):
        raise ValueError('Frozen event would be dropped; Cartesian study must retain it')
    return result


def matrix(root, channel, release_key):
    folder = root/channel
    record = read(folder/'completed.json')
    p = folder/'B_detector_zyx.float32'
    if record['release'] != release_key or p.stat().st_size != record['bytes']:
        raise ValueError('Cartesian response release/size differs')
    return np.memmap(p, dtype='<f4', mode='r', shape=(10496, 507000))


def check_operator(response, sensitivity, rank, device):
    """Deterministic actual-response dot tests and independent SC S."""
    n = response.geometry.active_count
    x = (1+torch.arange(n, device=device, dtype=torch.float64)/n)[:, None]
    rows = response.full_rows.shape[0]
    y = (1+torch.arange(rows, device=device, dtype=torch.float64)/max(1, rows))[:, None]
    errors = []
    for v in [0, 1, 5, 9, 19]:
        A = response.matrix(v)
        lhs = torch.zeros((), dtype=torch.float64, device=device)
        back = torch.zeros_like(x)
        for first in range(0, rows, 32):
            block = A[first:first+32].double()
            lhs += ((block@x)*y[first:first+32]).sum()
            back += block.T@y[first:first+32]
        rhs = (x*back).sum()
        errors.append(float((lhs-rhs).abs()/lhs.abs().clamp_min(1e-300)))
        del A
    # Independent sum in float64, then mapping through the same geometry.
    total = torch.zeros((1, response.full_rows.shape[1]), dtype=torch.float64, device=device)
    for first in range(0, rows, 32):
        total += response.full_rows[first:first+32].double().sum(0)[None, :]
    independent = sum(response.geometry.compact(total, v) for v in range(20)).T/20
    dist.all_reduce(independent)
    rel = float(torch.linalg.vector_norm(sensitivity.double()-independent)/torch.linalg.vector_norm(independent))
    if max(errors+[rel]) > 1e-5:
        raise ValueError('Native Cartesian forward/transpose/S check failed')
    return dict(rank=rank, adjoint_errors=errors, independent_S_relative_L2=rel, threshold=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--matrices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host-bytes', type=int, required=True)
    a = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = read(root/'study.json')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != cfg['reconstruction_nodes'] or int(os.environ['LOCAL_RANK']) != 0:
        raise ValueError('Frozen one GPU per node topology required')
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    if torch.cuda.device_count() != 1 or '5090' not in torch.cuda.get_device_name():
        raise ValueError('Expected one visible ordinary 5090 per rank')
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    release = read(root/'release_manifest.json')
    error = [None]
    if rank == 0:
        try:
            verify_release(root)
            a.output.mkdir(parents=True, exist_ok=False)
            for name, h in cfg['input_sha256'].items():
                if sha(a.input_root/name) != h:
                    raise ValueError('NEMA frozen input differs: '+name)
            for channel in ['A440', 'A218', 'C440to218']:
                record = read(a.matrices/channel/'completed.json')
                if sha(a.matrices/channel/'B_detector_zyx.float32') != record['sha256']:
                    raise ValueError('Fresh Cartesian matrix SHA differs')
            write(a.output/'started.json', dict(time=now(), job=os.environ['SLURM_JOB_ID'],
                release=release['key'], rank_count=world, grids_match_HE=True,
                routes=cfg['channels'], torch=torch.__version__, cuda=torch.version.cuda,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
        except Exception as e:
            error = [repr(e)]
    dist.broadcast_object_list(error, 0)
    if error[0]:
        raise RuntimeError(error[0])
    geo = Geometry(root/'cartesian_geometry.npz', device)
    lo, hi = 10496*rank//world, 10496*(rank+1)//world
    if cfg['response_cache_bytes']/world+(34<<30) > .8*a.host_bytes:
        raise MemoryError('Preflight full Cartesian C plus bounded auxiliary memory exceeds actual allocation')
    checkpoint_times = {}

    def save_callback(route):
        dest = a.output/route
        if rank == 0:
            dest.mkdir()
        dist.barrier()
        times = []
        def callback(iteration, history):
            before = time.monotonic()
            frame = history[-1].numpy().astype('<f4')
            if not np.isfinite(frame).all() or (frame<0).any():
                raise ValueError('Invalid Cartesian checkpoint')
            name = f'checkpoints/iteration_{iteration:05d}.float32'
            atomic(dest/name, frame.tobytes())
            write(dest/f'checkpoints/iteration_{iteration:05d}.json', dict(iteration=iteration,
                release=release['key'], image_sha256=sha(dest/name), active_shape=list(frame.shape),
                native_geometry_sha256=sha(root/'cartesian_geometry.npz')))
            write(a.output/'progress.json', dict(route=route, iteration=iteration,
                                                time=now()), replace=True)
            times.append(time.monotonic()-before)
        checkpoint_times[route] = times
        return callback if rank == 0 else None

    def finish(route, image, history, sens, elapsed, extra=None):
        dest = a.output/route
        resources = guard(a.host_bytes)
        write(dest/f'rank{rank}.json', dict(rank=rank, node=os.environ['SLURMD_NODENAME'],
            solve_seconds=elapsed, per_iteration_seconds=elapsed/10000,
            checkpoint_seconds=sum(checkpoint_times[route]), resources=resources, extra=extra))
        if rank == 0:
            atomic(dest/'Image_active.float32', image.cpu().numpy().astype('<f4').tobytes())
            atomic(dest/'Image_history.float32', history.numpy().astype('<f4').tobytes())
            atomic(dest/'S_active.float32', sens.cpu().numpy().astype('<f4').tobytes())
            import io
            stream = io.BytesIO()
            x, y, z = axes()
            np.savez_compressed(stream, density_xyz=geo.full_image(image.cpu().numpy()), x_mm=x, y_mm=y, z_mm=z)
            atomic(dest/'final.npz', stream.getvalue())
        sync()
        if rank == 0:
            write(dest/'completed.json', dict(route=route, release=release['key'], frames=200,
                iterations=10000, save_step=50, files={str(p.relative_to(dest)): sha(p)
                for p in sorted(dest.rglob('*')) if p.is_file()}, physical_calibration_certified=False))

    def sc_provider(channel):
        raw = matrix(a.matrices, channel, release['key'])
        local = torch.from_numpy(np.array(raw[lo:hi], copy=True)).to(device)
        del raw
        provider = ViewResponse(local, geo)
        sensitivity = provider.sensitivity()
        dist.all_reduce(sensitivity)
        numerical = check_operator(provider, sensitivity, rank, device)
        write(a.output/f'{channel}_numerical_rank{rank}.json', numerical)
        guard(a.host_bytes)
        return provider, sensitivity

    def projection(energy):
        file = a.input_root/f'CntStat/{energy}keV_RotateNum20_Geant4JSCC/CntStat_NEMA_Body_H60_5e9.csv'
        p = np.loadtxt(file, delimiter=',', dtype=np.float32)
        assert p.shape == (20, 10496)
        return torch.tensor(np.ascontiguousarray(p[:, lo:hi].T), device=device)

    # SC440 is solved once and supplies the new Cartesian 218 fixed background.
    response, ss = sc_provider('A440')
    counts440 = projection(440)
    callback = save_callback('440_SC')
    sync()
    start = time.monotonic()
    sc440, history = single_mlem_checkpointed(response, counts440, ss, 10000, 50,
        save_history=rank==0, checkpoint_callback=callback, progress_label='Cartesian440SC',
        phase_limit_seconds=cfg['phase_limit_hours']*3600)
    sync()
    finish('440_SC', sc440, history, ss, time.monotonic()-start)
    response.close()
    del response, history
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare independent matched Compton S, then unchanged fixed NEMA rows.
    raw = matrix(a.matrices, 'A440', release['key'])
    coords = torch.tensor(field_coordinates(), dtype=torch.float32, device=device)
    det = torch.tensor(load_detector_coordinates(root/'Detector.csv', 10496), dtype=torch.float32, device=device)
    var = build_detector_position_variance(det, 0.)
    law = ContinuousTransferLaw.load(root/'transfer_training_summary.json')
    cal = np.load(root/'calibration_rows.npy')
    total = torch.zeros(507000, dtype=torch.float64, device=device)
    start_prepare = time.monotonic()
    audit = None
    for batch, first in enumerate(range(0, len(cal), 32)):
        if batch % world != rank:
            continue
        pp = prepared(cal[first:first+32], det, var, device)
        rows = energy_rows(pp, coords, raw, law, geo.reference_fraction, cfg['source_tile'])
        if audit is None:
            # Tiling is an implementation choice. Test a different tile size on
            # the same complete actual event rows and reference domain.
            other = energy_rows(pp, coords, raw, law, geo.reference_fraction, cfg['source_tile']//2)
            rel = float(torch.linalg.vector_norm((other-rows).double())/torch.linalg.vector_norm(rows.double()))
            if rel > 1e-5:
                raise ValueError('Actual v5 source-tile regression failed')
            audit = dict(tile_relative_L2=rel, reference_points=507000, matched_independent_calibration=True)
            del other
        total += rows.double().sum(0)
        del rows, pp
        if batch%128 == rank:
            guard(a.host_bytes)
            print('CARTESIAN_CALIBRATION', rank, first, len(cal), flush=True)
    dist.all_reduce(total)
    total *= cfg['calibration_volume_mm3']/1e9
    expected = len(cal)*cfg['calibration_volume_mm3']/1e9
    integral = float((total*geo.reference_fraction).sum())
    if not np.isclose(integral, expected, rtol=1e-5):
        raise ValueError('Independent calibration normalization mismatch')
    sc = sum(geo.compact(total[None, :], v) for v in range(20)).T.float()/20
    if rank == 0:
        atomic(a.output/'S_Compton_detector.float64', total.cpu().numpy().astype('<f8').tobytes())
        atomic(a.output/'S_Compton_active.float32', sc.cpu().numpy().astype('<f4').tobytes())
        write(a.output/'S_Compton_provenance.json', dict(events=len(cal), primary_gamma=1000000000,
            calibration_rows_sha256=sha(root/'calibration_rows.npy'), reference_volume_mm3=cfg['calibration_volume_mm3'],
            normalization_error=integral/expected-1, observed_NEMA_used_for_S=False,
            view_transform='same Cartesian stencils and native HE overlap as all responses'))
    del total
    blocks, partitions = [], []
    for view in range(20):
        file = a.input_root/f'List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/{view+1}.csv'
        events = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2, 3), dtype=np.float32, ndmin=2)
        indices = np.load(root/f'selections/{view+1}.npy')
        assert len(indices) == cfg['events_per_view'][view]
        begin, end = len(indices)*rank//world, len(indices)*(rank+1)//world
        part = indices[begin:end]
        view_blocks = []
        for first in range(0, len(part), 32):
            pp = prepared(events[part[first:first+32]], det, var, device)
            rows = energy_rows(pp, coords, raw, law, geo.reference_fraction, cfg['source_tile'])
            compact = geo.compact(rows, view)
            if not bool(torch.isfinite(compact).all()) or bool((compact.sum(1)<=0).any()):
                raise ValueError('Zero/invalid event support; cannot change event selection')
            view_blocks.append(compact.cpu())
            del rows, pp, compact
            if first%512 == 0:
                guard(a.host_bytes)
        blocks.append(view_blocks)
        partitions.append(dict(view=view+1, first=begin, last=end, events=len(part)))
        print('CARTESIAN_EVENT_VIEW', rank, view+1, len(part), flush=True)
    del raw, events, coords, det, var, cal
    gc.collect()
    torch.cuda.empty_cache()
    sync()
    count = torch.tensor(sum(p['events'] for p in partitions), device=device)
    dist.all_reduce(count)
    assert int(count) == cfg['events']
    write(a.output/f'preparation_rank{rank}.json', dict(partitions=partitions,
        prepare_seconds=time.monotonic()-start_prepare, resources=guard(a.host_bytes), numerical=audit))
    response, ss_new = sc_provider('A440')
    if not torch.equal(ss, ss_new):
        raise ValueError('SC cache lifetime changed sensitivity')
    pool = OrderedTransferPool(geo.active_count, device, max_host_staging_bytes=512<<20,
        device_budget=current_cuda_budget(device, 512<<20), asynchronous=False, timing=False)
    transferred = [ViewBlocks(pool, b) for b in blocks]
    callback = save_callback('440_JSCC')
    sync()
    start = time.monotonic()
    image, history = joint_mlem(response, counts440, transferred, ss, sc, 10000, 50,
        rank==0, callback, cfg['phase_limit_hours']*3600)
    sync()
    finish('440_JSCC', image, history, ss+sc, time.monotonic()-start, pool.finish_stats())
    pool.close()
    response.close()
    del pool, response, transferred, blocks, view_blocks, image, history, sc, ss, ss_new
    gc.collect()
    torch.cuda.empty_cache()

    cross, _ = sc_provider('C440to218')
    background = torch.cat([cross.matrix(v)@sc440/20 for v in range(20)], dim=1)
    cross.close()
    del cross
    gc.collect()
    torch.cuda.empty_cache()
    atomic(a.output/f'218_background_rank{rank}.float32', background.cpu().numpy().astype('<f4').tobytes())
    response, ss218 = sc_provider('A218')
    callback = save_callback('218_SC')
    sync()
    start = time.monotonic()
    image, history = single_mlem_checkpointed(response, projection(218), ss218, 10000, 50,
        additive_background=background, save_history=rank==0, checkpoint_callback=callback,
        progress_label='Cartesian218SC', phase_limit_seconds=cfg['phase_limit_hours']*3600)
    sync()
    finish('218_SC', image, history, ss218, time.monotonic()-start,
        dict(background_source='this run final 440_SC through fresh Cartesian C440to218'))
    response.close()
    sync()
    if rank == 0:
        write(a.output/'completed.json', dict(time=now(), release=release['key'],
            channels=cfg['channels'], total_checkpoints=600, iterations=10000,
            native_HE_grid_match=True, strict_fetch_passed=False, visual_QA_passed=False))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
