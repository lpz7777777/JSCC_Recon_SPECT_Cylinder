"""Independent execution-only repair: private temporary files and measured 96h cap.

Scientific payloads, validation evidence and the original 48h gate stay immutable.
This module has no numerical dependencies and is also checked on the GPU host.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

MAX_MINUTES = 96 * 60


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(root, files):
    if not files:
        raise ValueError('Missing strict evidence')
    for name, sha in files.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or digest(Path(root) / p) != sha:
            raise ValueError('Evidence identity differs: ' + name)


def limits(phase, preparation, estimated_total):
    values = list(phase.values()) + [preparation, estimated_total]
    if not phase or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Positive finite measured times required')
    phase_s = max(1800, math.ceil(max(phase.values()) * 1.5 + 300))
    prep_s = max(1800, math.ceil(preparation * 1.5 + 300))
    minutes = max(120, math.ceil((estimated_total * 1.5 + 1800) / 60))
    if minutes > MAX_MINUTES or max(phase_s, prep_s) >= minutes * 60 - 180:
        raise ValueError('Measured throughput exceeds independent bounded 96h policy')
    return dict(phase=phase_s, prepare=prep_s, minutes=minutes, total=minutes * 60 - 180)


def measured(kind, proof):
    if not proof.get('passed') or proof['mode'] != 'validation':
        raise ValueError('Successful full-input validation required')
    channels = (['440_single', '218_corrected', '440_compton'] if kind == 'nema'
                else ['440_SinglePhoton', '440_SinglePlusCompton'])
    records = proof['resources']
    phase = {c: max(r['phase_solve_seconds'][c] for r in records) for c in channels}
    preparation = max(r['prepare_seconds'] for r in records)
    once = max(r['elapsed_seconds'] for r in records) - sum(phase.values())
    scaled = {c: seconds * 1000 for c, seconds in phase.items()}
    return limits(scaled, preparation, sum(scaled.values()) + max(once, preparation, 0))


def slurm_seconds(value):
    days, _, clock = value.rpartition('-')
    h, m, s = map(int, clock.split(':'))
    return (int(days or 0) * 24 + h) * 3600 + m * 60 + s


def allocation(snapshot, job, nodes, minutes):
    f = dict(x.split('=', 1) for x in snapshot.split() if '=' in x)
    if (f.get('JobId') != str(job) or f.get('NumNodes') not in (str(nodes), f'{nodes}-{nodes}')
            or f.get('TresPerNode') != 'gres/gpu:4'
            or slurm_seconds(f['TimeLimit']) != minutes * 60):
        raise ValueError('Actual formal Slurm allocation/time differs')


def check(gate_path, job, snapshot=None):
    g = read(gate_path)
    if (g.get('passed') is not True or g.get('strict_fetch_passed') is not True
            or g['formal_job'] != int(job) or g['policy_max_minutes'] != MAX_MINUTES):
        raise ValueError('Strict execution authority for this exact job required')
    for name in ('authority', 'science', 'validation_proof', 'strict_acceptance'):
        if digest(g[name + '_file']) != g[name + '_sha256']:
            raise ValueError('Bound gate file differs: ' + name)
    a = read(g['authority_file']); p = read(g['validation_proof_file'])
    accepted = read(g['strict_acceptance_file'])
    if (accepted.get('passed') is not True or accepted.get('strict_fetch_passed') is not True
            or p.get('passed') is not True or a.get('passed') is not True
            or a['validation_job'] != g['validation_job']
            or a['strict_acceptance_sha256'] != g['strict_acceptance_sha256']):
        raise ValueError('Strict validation linkage differs')
    if g['kind'] == 'nema':
        if (a['actual_primary_photons'] != 50000000000
                or (a['nodes'], a['gpus_per_node'], a['world_size']) != (8, 4, 32)
                or a['contract_sha256'] != g['science_sha256']
                or p['contract_sha256'] != g['science_sha256']
                or accepted['job'] != a['validation_job']
                or a['verification_sha256'] != g['validation_proof_sha256']):
            raise ValueError('NEMA science/authority identity differs')
        verify(a['result'], a['evidence_sha256']); nodes = 8
    elif g['kind'] == 'derenzo':
        plan = read(g['science_file'])
        if (a.get('strict_fetch_passed') is not True or a['formal_job'] != int(job)
                or a['dataset'] != 'JSCC_440' or p['dataset'] != 'JSCC_440'
                or a['plan_sha256'] != g['science_sha256']
                or a['collection_sha256'] != plan['collection_sha256']
                or a['release_sha256'] != plan['release_sha256']
                or accepted['source_job'] != a['validation_job']
                or plan['channels'] != ['440_SinglePhoton', '440_SinglePlusCompton']
                or (plan['iterations'], plan['save_step']) != (10000, 50)):
            raise ValueError('Derenzo science/authority identity differs')
        verify(a['validation_result'], a['validation_files_sha256']); nodes = plan['nodes']
    else:
        raise ValueError('Unsupported repair kind')
    if g['limits'] != measured(g['kind'], p):
        raise ValueError('Limits differ from actual validated throughput')
    if snapshot is None:
        snapshot = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True)
    allocation(snapshot, job, nodes, g['limits']['minutes'])
    return g


def temporary_probe(root):
    root = Path(root).resolve(strict=True)
    job = os.environ['SLURM_JOB_ID']; node = os.environ.get('SLURMD_NODENAME', os.uname().nodename)
    if not re.fullmatch(r'\d+', job) or not re.fullmatch(r'[A-Za-z0-9_.-]+', node):
        raise ValueError('Invalid actual job/node identity')
    target = root / 'tmp' / ('job_' + job) / ('node_' + node)
    target.mkdir(parents=True, mode=0o700, exist_ok=True)
    resolved = target.resolve(strict=True)
    if root not in resolved.parents or resolved.stat().st_uid != os.getuid():
        raise ValueError('Private temporary directory ownership/path differs')
    if any(os.environ.get(k) != str(target) for k in ('TMPDIR', 'TMP', 'TEMP')):
        raise ValueError('All temporary environment variables must be bound')
    if Path(tempfile.gettempdir()).resolve() != resolved:
        raise ValueError('Python temporary directory differs')
    stat = os.statvfs(resolved); free = stat.f_bavail * stat.f_frsize
    if free < max(512 << 20, stat.f_blocks * stat.f_frsize * .2):
        raise OSError('Actual project directory lacks 20 percent disk margin')
    data = bytes(range(256)) * 128
    with tempfile.TemporaryFile(dir=resolved) as f:
        f.write(data); f.flush(); os.fsync(f.fileno()); f.seek(0)
        if f.read() != data:
            raise OSError('Temporary file byte round trip differs')
    print('REPAIR_TMP_READY ' + json.dumps(dict(job=job, node=node, uid=os.getuid(),
          path=str(resolved), actual_free_bytes=free, roundtrip_bytes=len(data))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp-root', type=Path)
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    if args.tmp_root:
        temporary_probe(args.tmp_root)
    else:
        g = check(args.gate, os.environ['SLURM_JOB_ID'])
        prefix = 'JSCC' if g['kind'] == 'nema' else 'DERENZO'
        for name in ('phase', 'prepare', 'total'):
            print('export ' + prefix + '_' + name.upper() + '_SECONDS=' + str(g['limits'][name]))
        print('export ' + prefix + '_AUTHORITY_SHA256=' + g['authority_sha256'])
