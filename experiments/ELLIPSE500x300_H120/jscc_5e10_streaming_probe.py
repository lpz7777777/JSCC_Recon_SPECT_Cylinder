"""Actual 24-GPU node-local storage and unchanged-response preflight."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
from jscc_5e10_common import digest, read, write, verify_files
from jscc_5e10_streaming_runtime import setup, resource_record, gather_resources
from jscc_5e10_streaming_contract import load_contract, verify_topology
from jscc_5e10_streaming_cache import CacheWriter, DiskBlocks, POLICY
from run_energy_preflight_v5 import (partition_indices, rows_for, load_matrix, full_rows,
    ActiveGeometry, ContinuousTransferLaw, load_detector_coordinates,
    build_detector_position_variance, NAMES)


def local_storage(root):
    root = Path(root).resolve(strict=True)
    filesystem = subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'FSTYPE'], text=True).strip()
    free = shutil.disk_usage(root).free
    if filesystem not in ('ext4', 'xfs', 'btrfs'):
        raise ValueError('Cache must use a measured local disk, not RAM/network storage: ' + filesystem)
    return dict(root=str(root), filesystem=filesystem, available_bytes=free,
                device_id=root.stat().st_dev)


def main():
    p = argparse.ArgumentParser()
    for name in ('contract', 'factors', 'output', 'allocation'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    started = time.monotonic()
    cfg = load_contract(a.contract, 'validation', 10, 10)
    rank, world, local, device = setup()
    root = a.contract.parent
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    storage = local_storage('/tmp')
    path = Path(storage['root']) / 'jscc_geant4_5e10_streaming' / digest(a.contract)[:16] / ('probe_' + os.environ['SLURM_JOB_ID']) / f'rank{rank:02d}'
    path.mkdir(parents=True, exist_ok=False)
    geometry = ActiveGeometry.from_npz(root / 'whole_geometry.npz', device)
    matrix = load_matrix(a.factors, NAMES['A440'], 132040, 10496)
    B = full_rows(matrix, device)
    g = np.load(root / 'whole_geometry.npz')
    coordinates = torch.tensor(g['coordinates_mm'], dtype=torch.float32, device=device)
    detector = torch.tensor(load_detector_coordinates(a.factors / NAMES['A440'] / 'Detector.csv', 10496), device=device, dtype=torch.float32)
    variance = build_detector_position_variance(detector, 0)
    law = ContinuousTransferLaw.load(root / 'transfer_training_summary.json')
    selected = np.load(root / 'selected_rows/1.npy', mmap_mode='r')
    lo = len(selected) * rank // world
    hi = len(selected) * (rank + 1) // world
    sample = np.array(selected[lo:min(lo + 32, hi)], copy=True)
    response = geometry.compact(rows_for(sample, detector, variance, coordinates, B, law, 'continuous_energy'), 0)
    writer = CacheWriter(path / 'sample.cache', 78920)
    writer.append(response)
    receipt = writer.finish()
    original = response.cpu()
    cached = DiskBlocks(receipt)
    for _ in range(4):
        actual = next(iter(cached))
        if not torch.equal(original, actual):
            raise ValueError('Actual full-grid response cache differs after roundtrip')
    del response, B, coordinates, detector, variance
    torch.cuda.empty_cache()
    local_events = sum(count * (rank + 1) // world - count * rank // world for count in cfg['events_per_view'])
    time.sleep(max(0., 6. - (time.monotonic() - started)))
    record = resource_record(rank, local, device, started)
    record.update(storage=storage, local_events=local_events, sample_cache=receipt,
                  actual_lossless_response_roundtrip=True, probe_only_not_full_input_validation=True)
    resources = gather_resources(record)
    verify_topology(resources, a.allocation)
    if rank == 0:
        node_budget = []
        for node in sorted({r['node'] for r in resources}):
            group = [r for r in resources if r['node'] == node]
            dense = sum(r['local_events'] for r in group) * 78920 * 4
            available = min(r['storage']['available_bytes'] for r in group)
            ratio = max(r['sample_cache']['encoded_bytes'] / r['sample_cache']['raw_bytes'] for r in group)
            reserve = 16 << 30
            conservative_sample_bytes = dense * ratio * 4 + reserve
            node_budget.append(dict(node=node, dense_bytes=dense, available_bytes=available,
                observed_sample_storage_ratio=ratio, worst_case_capacity_passed=available >= dense * 1.02 + reserve,
                sample_budget_with_fourfold_margin=conservative_sample_bytes,
                guarded_full_cache_attempt_permitted=(available >= dense * 1.02 + reserve or available >= conservative_sample_bytes),
                sample_is_not_complete_cache_capacity_certificate=True))
        passed = all(x['guarded_full_cache_attempt_permitted'] for x in node_budget)
        write(a.output / 'probe.json', dict(passed=passed, resources=resources, node_storage_budget=node_budget,
            contract_sha256=digest(a.contract), policy=POLICY, world_size=24, nodes=8, gpus_per_node=3,
            complete_input_validation_still_required=True, full_response_memory_certificate=False,
            no_selected_events_removed=True, original_5090_job_untouched=True))
        print('STREAMING_PROBE_COMPLETE', json.dumps(node_budget), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
