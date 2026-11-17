"""Strict new mono acquisition authority; existing HE calibration not rerun."""
from pathlib import Path
import argparse
import json
import re
import opengate_ehe_archive_20261113 as base


def seal(root, kind):
    root = Path(root)
    target = root / 'control/transport_delivery_authority.json'
    if target.exists():
        return base.read(target)
    cfg = base.read(root / 'inputs/config.json')
    complete = base.read(root / 'collected/complete.json')
    jobs = base.read(root / 'inputs/imaging_jobs.json')
    release = (root / 'inputs').resolve().parent
    frozen = base.read(release / 'release.json')
    for name, r in frozen['files'].items():
        p = release / name
        if p.stat().st_size != r['bytes'] or base.sha(p) != r['sha256']:
            raise ValueError('Changed frozen execution member: ' + name)
    assert complete['passed'] and complete['actual_imaging_primaries'] == cfg['imaging_primary_gamma_total']
    acct = dict(imaging=base.accounting(root, 'imaging', len(jobs)), collect=base.accounting(root, 'collect'))
    resources = {}; peak = 0.
    for i in range(len(jobs)):
        p = root / 'resources' / f'imaging_{i:04d}.json'
        r = base.read(p)
        allocation = (root / 'resources' / f'imaging_{i:04d}.allocation.txt').read_text()
        if r['status'] != 'COMPLETED' or r['rss_fraction'] > .8:
            raise ValueError('Failed actual resource/exit gate')
        if f"ArrayJobId={acct['imaging']['job']}" not in allocation or not re.search(r'\bArrayTaskId=' + str(i) + r'\b', allocation):
            raise ValueError('Allocation identity mismatch')
        resources[str(i)] = dict(sha256=base.sha(p), job=r['job'])
        peak = max(peak, r['rss_fraction'])
    geometry = base.read(root / 'workers/imaging/0000/actual_hexagon_geometry.json')
    source = base.read(root / 'workers/imaging/0000/source_check/complete.json')
    assert geometry['passed'] and source['passed'] and source['actual_primary_vertex_rotation_verified']
    assert base.sha(root / 'collected/response_kernels.npz') == cfg['response_kernel_sha256']
    result = dict(passed=True, actual_primary_gamma_total=cfg['imaging_primary_gamma_total'],
        energy_keV=cfg['mono_energy_keV'], actual_primary_vertex_rotation_verified=True,
        inherited_imaging_primaries=0, new_response_primaries=0, response_kernel_sha256=cfg['response_kernel_sha256'],
        response_provenance_sha256=base.sha(root / 'inputs/response_provenance.json'),
        accounting=acct, resources=resources, max_worker_rss_fraction=peak,
        executed_release=frozen['release'], executed_release_files=frozen['files'],
        actual_geometry=geometry, source_check=source, physical_calibration_passed=False,
        collected_sha256=base.sha(root / 'collected/complete.json'))
    base.save(target, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--kind', choices=['bridge', 'transport'], required=True)
    a = p.parse_args(); base.seal = seal
    print('CAMPAIGN_JSON ' + json.dumps(base.bundle(a.root, a.kind)))
