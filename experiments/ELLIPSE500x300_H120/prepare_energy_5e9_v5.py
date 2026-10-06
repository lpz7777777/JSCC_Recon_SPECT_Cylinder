"""Freeze existing legacy 5e9 and calibration events with the v5 stable q rule.

No truth, source labels, image values or learned energy law enter selection.
Inputs remain read-only; per-file receipts allow bounded interrupted scans.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import time
import numpy as np
import torch
from compton_event_response import (prepare_compton_events, min_standardized_compton_arm,
    build_compton_cone_weights, build_detector_position_variance)
from detector_csv import load_detector_coordinates
from process_list_global_audit_v4 import digest, write, settings
from compton_event_response import PreparedComptonEvents

STUDY = 'compton_energy_probability_v5_5e9'


def subset(prepared, index):
    return PreparedComptonEvents(**{f.name: (None if getattr(prepared, f.name) is None
        else getattr(prepared, f.name)[index]) for f in fields(prepared)})


def validate_transport(collection):
    c = collection
    for key in ('primary_counts', 'views', 'worker_indices', 'seeds'):
        if any(type(n) is not int for n in c.get(key, [])):
            raise ValueError('Transport counters must be integers')
    if (c.get('event_policy', 'legacy') != 'legacy'
        or c.get('dataset') != 'NEMA_Body_H60' or c.get('level') != '5e9'
        or c.get('views') != list(range(1, 21))
        or c.get('primary_counts') != [1469053733, 3530946267, 0]
        or c.get('worker_indices') != list(range(200))
        or c.get('seeds') != list(range(30100101, 30100301))):
        raise ValueError('Actual legacy 5e9 transport identity differs')
    return dict(actual_primary_gamma=sum(c['primary_counts']), primary_counts=c['primary_counts'],
        workers=200, seeds=200, views=20, event_policy='legacy', photons_per_worker=25000000)


def metadata(folder, view, wanted=None):
    """Explicit legacy row association; ideal-only rows never substitute for legacy."""
    with (folder / f'events_v{view:02d}.csv').open() as stream:
        for r in csv.DictReader(stream):
            index = int(r['global_legacy_row'])
            if index >= 0 and (wanted is None or index in wanted):
                yield r


def selected_rows(a, dataset, view):
    return np.load(a.analysis / f'{dataset}_legacy_v{view:02d}_kept_rows.npy')


def scan_one(path, name, view, expected, output, detector, variance, coords, B, batch):
    key = f'{name}_legacy_v{view:02d}'
    receipt = output / (key + '.json')
    selection = output / (key + '_kept_rows.npy')
    sha = digest(path)
    if sha != expected:
        raise ValueError('Original input changed: ' + str(path))
    if receipt.exists():
        r = json.loads(receipt.read_text())
        if r['input_sha256'] != sha or digest(selection) != r['selection_sha256']:
            raise ValueError('Existing selection identity differs')
        return r
    if selection.exists():
        raise ValueError('Unregistered selection must be diagnosed before resuming')
    raw = np.loadtxt(path, delimiter=',', usecols=(0, 1, 2, 3), dtype=np.float32, ndmin=2)
    p, diag = prepare_compton_events(torch.tensor(raw, device=coords.device), settings(),
        detector, variance, variance, input_energies_already_smeared=True)
    kept = []; removed = []; accepted = 0; chunk_error = 0.; started = time.monotonic()
    if p is not None:
        for start in range(0, p.count, batch):
            e = subset(p, slice(start, start + batch))
            weight = build_compton_cone_weights(e, coords, settings()) * B[e.cpnum1 - 1]
            sums = weight.sum(1)
            valid = torch.isfinite(weight).all(1) & torch.isfinite(sums) & (sums > 0)
            diag.invalid_kernel_events += int((~valid).sum())
            e = subset(e, valid)
            if not e.count:
                continue
            # Original min effective support 1 accepts every positive normalized row.
            accepted += e.count
            q = min_standardized_compton_arm(e, coords, settings())
            if not bool(torch.isfinite(q).all()):
                raise ValueError('Nonfinite mismatch score')
            if start == 0:
                separate = torch.cat([min_standardized_compton_arm(subset(e, slice(i, i+1)),
                    coords, settings()) for i in range(e.count)])
                chunk_error = float((q - separate).abs().max())
                if chunk_error > 1e-10 or not torch.equal(q <= 3, separate <= 3):
                    raise ValueError('q selection changes with event partition')
            keep = q <= 3
            kept.extend(e.source_row_indices[keep].cpu().tolist())
            removed.extend(zip(e.source_row_indices[~keep].cpu().tolist(), q[~keep].cpu().tolist()))
            del e, weight, sums, q
            if torch.cuda.memory_reserved() > .60 * torch.cuda.get_device_properties(coords.device).total_memory:
                torch.cuda.empty_cache()
            if torch.cuda.max_memory_reserved() > .75 * torch.cuda.get_device_properties(coords.device).total_memory:
                raise RuntimeError('Proactive GPU memory limit exceeded')
    np.save(selection, np.asarray(kept, dtype=np.int64))
    with (output / (key + '_removed.csv')).open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['raw_row_0based', 'q'])
        writer.writerows(removed)
    r = dict(dataset=name, view=view, event_policy='legacy', raw_rows=len(raw),
        original_accepted=accepted, kept=len(kept), removed=len(removed),
        input_sha256=sha, selection_sha256=digest(selection),
        selection='complete-circle stable_float64 q <= 3; equal 3 retained',
        grid_points=132040, diagnostics=diag.to_dict(),
        partition_q_max_absolute_difference=chunk_error, elapsed_seconds=time.monotonic()-started)
    if r['kept'] + r['removed'] != r['original_accepted']:
        raise ValueError('Event closure failed')
    write(receipt, r)
    print('FROZEN_LEGACY_SELECTION', key, accepted, len(kept), len(removed), flush=True)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('base', 'inputs', 'factors', 'geometry', 'output'):
        parser.add_argument('--' + n, type=Path, required=True)
    parser.add_argument('--batch', type=int, default=32)
    a = parser.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); torch.set_grad_enabled(False); torch.cuda.set_device(0)
    geo = np.load(a.geometry); coords = torch.tensor(geo['coordinates_mm'], dtype=torch.float32, device='cuda:0')
    if coords.shape != (132040, 3): raise ValueError('Incomplete circle grid')
    factor = a.factors / '440keV_RotateNum20'
    detector = torch.tensor(load_detector_coordinates(factor/'Detector.csv', 10496), device='cuda:0')
    variance = build_detector_position_variance(detector, 0)
    matrix = np.memmap(factor/'SysMat_polar', dtype='<f4', mode='r', shape=(132040,10496))
    B = torch.tensor(np.array(matrix.T, copy=True), device='cuda:0'); del matrix
    original = json.loads((a.base/'generated/nema_h60_imaging_5e9_files.json').read_text())
    collection = a.base/'generated/collections/NEMA_Body_H60_5e9.json'
    if digest(collection) != original['collections/NEMA_Body_H60_5e9.json']['sha256']:
        raise ValueError('Transport collection changed')
    transport = validate_transport(json.loads(collection.read_text()))
    records = []
    for v in range(1, 21):
        rel = f'List/218-440keV_RotateNum20_Geant4JSCC/List_NEMA_Body_H60_5e9/{v}.csv'
        records.append(scan_one(a.base/'generated'/rel, 'NEMA', v, original[rel]['sha256'],
            a.output, detector, variance, coords, B, a.batch))
    nema = [r for r in records if r['dataset'] == 'NEMA']
    if sum(r['original_accepted'] for r in nema) != 484936:
        raise ValueError('Original 484936 accepted events not reproduced')
    manifest = json.loads((a.inputs/'input_manifest.json').read_text())['files']
    seed_sets = []
    for name in ('circle_train', 'circle_validation', 'ellipse_validation', *[f'point_{i}' for i in range(7)]):
        cpath = a.inputs/name/'collection.json'
        if digest(cpath) != manifest[name+'/collection.json']: raise ValueError('Calibration collection changed')
        c = json.loads(cpath.read_text()); seeds = set(c['seeds'])
        if len(seeds) != len(c['seeds']) or any(seeds & previous for previous in seed_sets):
            raise ValueError('Calibration and validation seeds are not independent')
        seed_sets.append(seeds)
        expected_photons = 1000000000 if name.startswith('circle_') else (100000000 if name == 'ellipse_validation' else 10000000)
        if c['primary_counts'] != [0, expected_photons, 0]: raise ValueError('Calibration photons differ')
        for v in c['views']:
            rel = f'{name}/legacy_v{v:02d}.csv'
            records.append(scan_one(a.inputs/rel, name, v, manifest[rel], a.output,
                detector, variance, coords, B, a.batch))
    write(a.output/'selection_gate.json', dict(status='PASSED', study=STUDY,
        event_policy='legacy', geometry_mode='stable_float64', grid_points=132040,
        original_nema_accepted=484936, nema_kept=sum(r['kept'] for r in nema),
        nema_removed=sum(r['removed'] for r in nema), records=records, transport=transport,
        input_manifest_sha256=digest(a.inputs/'input_manifest.json'), geometry_sha256=digest(a.geometry),
        original_input_manifest_sha256=digest(a.base/'generated/nema_h60_imaging_5e9_files.json'),
        kernel_sha256=digest(__import__('compton_event_response').__file__),
        new_photons=0, new_training=False, source_or_truth_used_in_selection=False))


if __name__ == '__main__': main()
