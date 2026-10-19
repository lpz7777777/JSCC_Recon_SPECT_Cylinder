"""Fail closed before allocating reconstruction processes for a queued formal job."""
import argparse
import math
import os
from pathlib import Path

from jscc_5e10_common import TOTAL, digest, read, verify_files


def measured_limits(authority):
    values = [authority['estimated_max_phase_seconds'], authority['measured_prepare_seconds'],
              authority['estimated_total_seconds']]
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Positive finite measured validation times required')
    phase = max(1800, math.ceil(values[0] * 1.5 + 300))
    prepare = max(1800, math.ceil(values[1] * 1.5 + 300))
    minutes = max(120, math.ceil((values[2] * 1.5 + 1800) / 60))
    if phase > 86400 or minutes > 2880:
        raise ValueError('Measured reconstruction exceeds the bounded 48-hour budget')
    return dict(phase=phase, prepare=prepare, minutes=minutes, total=minutes * 60 - 180)


def check_gate(gate_path, contract, job):
    gate = read(gate_path)
    if (gate.get('passed') is not True or gate.get('strict_fetch_passed') is not True
            or gate['formal_job'] != int(job) or gate['contract_sha256'] != digest(contract)):
        raise ValueError('This formal job needs strict current-contract acceptance')
    authority_path = Path(gate['authority_file'])
    if digest(authority_path) != gate['authority_sha256']:
        raise ValueError('Queued authority SHA differs')
    authority = read(authority_path)
    if (authority.get('passed') is not True or authority['contract_sha256'] != digest(contract)
            or authority['actual_primary_photons'] != TOTAL
            or (authority['nodes'], authority['gpus_per_node'], authority['world_size']) != (8, 4, 32)
            or authority['validation_job'] != gate['validation_job']
            or authority['strict_acceptance_sha256'] != gate['strict_acceptance_sha256']):
        raise ValueError('Full-input validation authority identity differs')
    if gate['limits'] != measured_limits(authority):
        raise ValueError('Execution limits must come from actual validation throughput')
    verify_files(Path(authority['result']), authority['evidence_sha256'])
    if len(gate['authority_sha256']) != 64 or any(x not in '0123456789abcdef' for x in gate['authority_sha256']):
        raise ValueError('Invalid authority hash')
    return gate


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    a = p.parse_args()
    g = check_gate(a.gate, a.contract, os.environ['SLURM_JOB_ID'])
    print('export JSCC_PHASE_SECONDS=' + str(g['limits']['phase']))
    print('export JSCC_PREPARE_SECONDS=' + str(g['limits']['prepare']))
    print('export JSCC_TOTAL_SECONDS=' + str(g['limits']['total']))
    print('export JSCC_AUTHORITY_SHA256=' + g['authority_sha256'])
