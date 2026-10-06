"""Explicit six-output 5e9 continuous-energy 10000/save50 contract.

The delivered v5 paired entry retains its 2000-iteration protection. This new
entry revalidates its immutable data/response payload and has its own authority.
"""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from run_reconstruction import collect_response, digest
from energy_5e9_v5_contract import load_contract as load_source_contract, sync_file

STUDY = 'compton_energy_probability_v5_5e9_full10000'
PHASE_CHANNELS = {
    '440_single': ('440_SinglePhoton',),
    '218_corrected': ('218_SinglePhoton_CrossTalkCorrected', '440SinglePlus218Single'),
    'compton_jscc': ('440_ComptonOnly', '440_SinglePlusCompton', '440SingleComptonPlus218Single'),
}
CHANNELS = tuple(x for channels in PHASE_CHANNELS.values() for x in channels)


def execution_policy(mode):
    if mode not in ('validation', 'formal'):
        raise ValueError('Explicit validation or formal mode required')
    return (10, 10) if mode == 'validation' else (10000, 50)


def load_contract(path, mode, iterations, save_step):
    path = Path(path); root = path.parent
    cfg = json.loads(path.read_text())
    if (iterations, save_step) != execution_policy(mode):
        raise ValueError('Only validation10/save10 or formal10000/save50')
    if (cfg['study'] != STUDY or cfg['iterations'] != 10000 or cfg['save_step'] != 50
        or cfg['model'] != 'continuous_energy' or cfg['channels'] != list(CHANNELS)
        or cfg['nodes'] != 8 or cfg['accepted_events'] != 483743
        or cfg['event_policy'] != 'legacy' or cfg['new_photons'] != 0 or cfg['new_training']
        or cfg['cross_prediction_source'] != '440_SinglePhoton_final'
        or cfg['composites_are_gamma_density_sums'] is not True):
        raise ValueError('Frozen full six-channel scope differs')
    source = root / 'source_v5_contract.json'
    if digest(source) != cfg['source_v5_contract_sha256']:
        raise ValueError('Delivered v5 source contract changed')
    old = load_source_contract(source, 'validation', 10, 10)
    for key in ('input_sha256', 'factor_manifest_sha256', 'factor_payload_sha256',
                'whole_geometry_sha256', 'events_per_view', 'accepted_events', 'calibration_release'):
        if cfg[key] != old[key]:
            raise ValueError('Delivered data/operator identity differs: ' + key)
    for name, sha in cfg['files'].items():
        if digest(root / name) != sha:
            raise ValueError('Frozen full entry changed: ' + name)
    return cfg


def write_checkpoint(output, phase, iteration, frames, geometry, contract_sha, mode):
    limit, step = execution_policy(mode)
    if phase not in PHASE_CHANNELS or not 0 < iteration <= limit or iteration % step:
        raise ValueError('Checkpoint execution policy differs')
    if set(frames) != set(PHASE_CHANNELS[phase]):
        raise ValueError('Incomplete checkpoint channels')
    parent = Path(output) / ('checkpoints_' + phase)
    parent.mkdir(exist_ok=True)
    target = parent / f'checkpoint_{iteration:06d}'
    temp = parent / f'.checkpoint_{iteration:06d}.partial'
    if target.exists() or temp.exists():
        raise ValueError('Never overwrite an existing checkpoint')
    temp.mkdir()
    try:
        outputs = {}
        for name, frame in frames.items():
            if (frame.numel() != geometry.active_count or not bool(torch.isfinite(frame).all())
                or bool((frame < 0).any())):
                raise ValueError('Invalid persistent image')
            collect_response(temp, name, frame, None, geometry, 0)
            outputs[name] = {}
            for kind in ('active', 'full'):
                p = temp / f'Image_{name}_{kind}.float32'
                sync_file(p); outputs[name][kind] = digest(p)
        record = dict(study=STUDY, mode=mode, phase=phase, iteration=iteration,
                      contract_sha256=contract_sha, outputs=outputs)
        p = temp / 'checkpoint_manifest.json'
        p.write_bytes((json.dumps(record, indent=2, allow_nan=False) + '\n').encode())
        sync_file(p); temp.rename(target)
        if os.name == 'posix':
            fd = os.open(str(parent), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        return record
    except Exception:
        if temp.resolve().parent != parent.resolve():
            raise ValueError('Unsafe generated snapshot path')
        shutil.rmtree(temp)
        raise


def verify_checkpoints(result, histories, active, contract_sha, mode, full_count=132040):
    iterations, step = execution_policy(mode)
    records = []; inactive = np.ones(full_count, bool); inactive[active] = False
    for phase, channels in PHASE_CHANNELS.items():
        parent = Path(result) / ('checkpoints_' + phase)
        expected = {f'checkpoint_{i:06d}' for i in range(step, iterations + 1, step)}
        if {p.name for p in parent.glob('checkpoint_*') if p.is_dir()} != expected:
            raise ValueError('Missing/extra persistent checkpoints: ' + phase)
        if list(parent.glob('*.partial')):
            raise ValueError('Unpublished checkpoint remains')
        for frame, iteration in enumerate(range(step, iterations + 1, step)):
            folder = parent / f'checkpoint_{iteration:06d}'; p = folder / 'checkpoint_manifest.json'
            r = json.loads(p.read_text())
            if (r['study'], r['mode'], r['phase'], r['iteration'], r['contract_sha256']) != (
                STUDY, mode, phase, iteration, contract_sha) or set(r['outputs']) != set(channels):
                raise ValueError('Checkpoint identity differs')
            for channel in channels:
                values = {}
                for kind, count in (('active', len(active)), ('full', full_count)):
                    f = folder / f'Image_{channel}_{kind}.float32'
                    if f.stat().st_size != count * 4 or digest(f) != r['outputs'][channel][kind]:
                        raise ValueError('Persistent file bytes/SHA differ')
                    values[kind] = np.fromfile(f, '<f4')
                if (not np.array_equal(values['active'], histories[channel][frame])
                    or not np.array_equal(values['full'][active], values['active'])
                    or np.any(values['full'][inactive] != 0)
                    or not np.isfinite(values['full']).all() or np.any(values['full'] < 0)):
                    raise ValueError('Persistent history/support differs')
            records.append(dict(phase=phase, iteration=iteration, manifest_sha256=digest(p)))
    return records
