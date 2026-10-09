"""Immutable dispatcher; reuse accepted workers and run only missing indices."""
import argparse, sys
from pathlib import Path
from ehe_common import read, digest, verify_files


def dispatch(release, simulation, output, index, limit):
    freeze = read(release/'release_manifest.json')
    verify_files(release, freeze['sha256'])
    config = read(release/'config.json')
    recovery = config['transport_recovery']
    source = Path(recovery['original_worker_release'])
    original = read(source/'release_manifest.json')
    verify_files(source, original['sha256'])
    if digest(source/'ehe_5e10_transport.py') != recovery['original_worker_source_sha256']:
        raise ValueError('Original executed worker wrapper changed')
    sys.path.insert(0, str(source))
    from ehe_5e10_transport import worker, registry_identity
    registry_identity(simulation, config)
    if str(index) in recovery['reused_workers']:
        entry = recovery['reused_workers'][str(index)]
        folder = Path(recovery['original_transport_root'])/f'worker_{index:04d}'
        if digest(folder/'receipt.json') != entry['receipt_sha256']:
            raise ValueError('Original completed worker receipt changed')
        verify_files(folder, entry['files'])
        print('EHE_5E10_REUSE_VERIFIED', index, flush=True)
    else:
        worker(source, simulation, output/f'worker_{index:04d}', index, limit)


if __name__ == '__main__':
    import os
    p=argparse.ArgumentParser()
    for name in ('release','simulation','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,required=True)
    a=p.parse_args()
    dispatch(a.release,a.simulation,a.output,int(os.environ['SLURM_PROCID']),a.limit)
