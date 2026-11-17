"""Independent measured execution bounds for the fragmented ordinary 5090 queue."""
import argparse
import math
import os
import re
import subprocess
from pathlib import Path
from jscc_5e10_common import read, digest, verify_files
from reconstruction_execution_gate_20261020 import temporary_probe

MAX_MINUTES = 10 * 24 * 60
TOPOLOGY = {'nema': (8, 2, 16), 'derenzo': (4, 1, 4)}


def measured(kind, proof):
    if not proof.get('passed') or proof.get('mode') != 'validation' or proof['iterations'] != 10:
        raise ValueError('Successful complete new-topology validation10 required')
    records = proof['resources']; nodes, local, world = TOPOLOGY[kind]
    if (len(records) != world or len({r['gpu_uuid'] for r in records}) != world
            or sorted(r['rank'] for r in records) != list(range(world))
            or len({r['node'] for r in records}) != nodes):
        raise ValueError('Actual new allocation identities required')
    for node in {r['node'] for r in records}:
        group = [r for r in records if r['node'] == node]
        if sorted(r['local_rank'] for r in group) != list(range(local)):
            raise ValueError('Actual local ranks differ')
        if sum(r['host_peak_rss_bytes'] for r in group) > .8 * min(r['host_allocated_bytes_node'] for r in group):
            raise MemoryError('Actual node RSS lacks 20 percent headroom')
    for r in records:
        if (r['gpu_memory_samples'] < 2 or r['peak_reserved_bytes'] > .8*r['total_device_bytes']
                or r['measured_gpu_used_peak_bytes'] > .8*r['total_device_bytes']):
            raise MemoryError('Actual GPU lacks measured 20 percent headroom')
    channels = ('440_single', '218_corrected', '440_compton') if kind == 'nema' else ('440_SinglePhoton', '440_SinglePlusCompton')
    ten = {c: max(r['phase_solve_seconds'][c] for r in records) for c in channels}
    prep = max(r['prepare_seconds'] for r in records)
    if any(not math.isfinite(v) or v <= 0 for v in list(ten.values())+[prep]):
        raise ValueError('Positive finite measured timing required')
    phases = {c: v*1000 for c, v in ten.items()}
    once = max(max(r['elapsed_seconds'] for r in records)-sum(ten.values()), prep, 0)
    estimate = sum(phases.values())+once
    minutes = max(120, math.ceil((estimate*1.5+1800)/60))
    if minutes > MAX_MINUTES:
        raise ValueError('Measured full solve exceeds independent ten-day bound')
    return dict(phase=max(1800, math.ceil(max(phases.values())*1.5+300)),
                prepare=max(1800, math.ceil(prep*1.5+300)), minutes=minutes, total=minutes*60-180,
                estimated_total_seconds=estimate, preparation_count=1, margin=1.5)


def check(path, job, snapshot=None):
    a = read(path); kind = a['kind']; nodes, gpus, world = TOPOLOGY[kind]
    if (a.get('passed') is not True or a.get('strict_fetch_passed') is not True
            or a['formal_job'] != int(job) or a['policy_max_minutes'] != MAX_MINUTES):
        raise ValueError('Strict authority for this exact formal job required')
    for label in ('strict_acceptance', 'validation_proof', 'science'):
        if digest(a[label+'_file']) != a[label+'_sha256']:
            raise ValueError('Authority evidence changed: '+label)
    accepted = read(a['strict_acceptance_file']); proof = read(a['validation_proof_file'])
    source = accepted.get('job', accepted.get('source_job'))
    if (not accepted.get('passed') or not accepted.get('strict_fetch_passed')
            or source != a['validation_job'] or accepted['verification_sha256'] != a['validation_proof_sha256']
            or a['limits'] != measured(kind, proof)):
        raise ValueError('New strict validation linkage/timing differs')
    science = read(a['science_file'])
    if (science['nodes'], science['gpus_per_node'], science['world_size']) != (nodes, gpus, world):
        raise ValueError('Frozen topology differs')
    expected = ('contract_sha256', 'evidence_sha256', 'result') if kind == 'nema' else ('plan_sha256', 'validation_files_sha256', 'validation_result')
    if (a[expected[0]] != a['science_sha256'] or proof[expected[0]] != a['science_sha256']
            or a[expected[1]] != proof['result_files_sha256']):
        raise ValueError('Complete scientific evidence identity differs')
    verify_files(Path(a[expected[2]]), a[expected[1]])
    if snapshot is None:
        snapshot = subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True)
    fields = dict(t.split('=',1) for t in snapshot.split() if '=' in t)
    if (fields.get('JobId') != str(job) or fields.get('NumNodes') not in (str(nodes),f'{nodes}-{nodes}')
            or fields.get('TresPerNode') != 'gres/gpu:'+str(gpus)):
        raise ValueError('Actual requested formal allocation differs')
    from reconstruction_execution_gate_20261020 import slurm_seconds
    if slurm_seconds(fields['TimeLimit']) != a['limits']['minutes']*60:
        raise ValueError('Actual formal Slurm time differs')
    return a


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--tmp-root',type=Path);p.add_argument('--authority',type=Path);args=p.parse_args()
    if args.tmp_root:
        temporary_probe(args.tmp_root)
    else:
        authority=check(args.authority,os.environ['SLURM_JOB_ID'])
        prefix='JSCC' if authority['kind']=='nema' else 'DERENZO'
        for name in ('phase','prepare','total'):
            print('export '+prefix+'_'+name.upper()+'_SECONDS='+str(authority['limits'][name]))
        print('export '+prefix+'_AUTHORITY_SHA256='+digest(args.authority))
