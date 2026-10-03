"""Fetch only verified cut3 outputs; recheck transfer and active/full geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'FOV120'))
from reconstruction_ssh import connect
from verify_response_mismatch import digest

REMOTE = '/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120'
REPORT = HERE / 'reports/NEMA_Body_H60/response_mismatch_cut3_v1'


def read_json(sftp, path):
    with sftp.open(path, 'r') as stream:
        return json.loads(stream.read().decode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--phase', choices=('regression', 'pilot', 'interim', 'formal'), default='formal')
    args = parser.parse_args()
    registered = json.loads((REPORT / 'job.json').read_text())
    if registered['job_id'] != str(args.job):
        raise ValueError('Job differs from registered study; record any repaired job first')
    phase = 'formal' if args.phase == 'interim' else args.phase
    remote = f'{REMOTE}/generated/response_mismatch_cut3_v1/{phase}_{args.job}'
    if args.phase == 'interim':
        remote += '/checkpoint_2000'
    dest = HERE / 'generated/RemoteResults' / f'NEMA_Body_H60_5e9_cut3_{args.phase}_{args.job}'
    with connect() as ssh, ssh.open_sftp() as sftp:
        try:
            marker = 'checkpoint_manifest.json' if args.phase == 'interim' else 'verification.json'
            record = read_json(sftp, remote + '/' + marker)
        except FileNotFoundError:
            _, out, err = ssh.exec_command(f'sacct -j {args.job} --format=JobID,State,Elapsed,ExitCode -P', timeout=30)
            state = out.read().decode(errors='replace')
            if out.channel.recv_exit_status():
                raise RuntimeError(err.read().decode(errors='replace'))
            print('No passed verification yet. No unverified images fetched.\n' + state)
            return
        if args.phase == 'interim':
            if (record['iteration'] != 2000 or record['accepted_events'] != 483768
                    or record['sensi_d_sha256'] != json.loads((HERE / 'response_mismatch_cut3_v1.json').read_text())['sensi_d_sha256']):
                raise ValueError('Interim frozen contract differs')
            with sftp.open(remote + '/' + marker, 'rb') as stream:
                marker_sha = hashlib.sha256(stream.read()).hexdigest()
            verification = dict(record, passed=True, snapshot_manifest_sha256=marker_sha,
                                warning='Snapshot transfer verified only; formal reconstruction not yet accepted')
        else:
            verification = record
        if not verification['passed'] or verification['mode'] != args.phase:
            raise ValueError('Passed verification for requested phase required')
        if verification['config_sha256'] != digest(HERE / 'response_mismatch_cut3_v1.json'):
            raise ValueError('Frozen experiment configuration differs')
        dest.mkdir(parents=True, exist_ok=True)
        expected = {f"Image_{row['channel']}_{kind}.float32": row['sha256'][kind]
                    for row in verification['outputs'] for kind in ('active', 'full', 'history')}
        if args.phase == 'interim':
            expected['checkpoint_manifest.json'] = verification['snapshot_manifest_sha256']
        else:
            expected['run_manifest.json'] = verification['run_manifest_sha256']
        for name, sha in expected.items():
            path = dest / name
            if not path.exists() or digest(path) != sha:
                temporary = dest / (name + '.downloading')
                sftp.get(remote + '/' + name, str(temporary))
                if digest(temporary) != sha:
                    raise ValueError('Transfer checksum mismatch: ' + name)
                temporary.replace(path)
            print('VERIFIED', name, flush=True)
        # Verification is read again so a concurrent replacement cannot be hidden.
        if read_json(sftp, remote + '/' + marker) != record:
            raise ValueError('Remote verification changed while fetching')
    (dest / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    geometry_path = HERE / 'generated/Geometry/geometry.npz'
    cfg = json.loads((HERE / 'response_mismatch_cut3_v1.json').read_text())
    if digest(geometry_path) != cfg['geometry_sha256']:
        raise ValueError('Local frozen geometry differs')
    active = np.load(geometry_path)['active_indices']
    inactive = np.ones(132040, bool)
    inactive[active] = False
    for channel in cfg['channels']:
        a = np.fromfile(dest / f'Image_{channel}_active.float32', '<f4')
        full = np.fromfile(dest / f'Image_{channel}_full.float32', '<f4')
        if not np.array_equal(full[active], a) or np.any(full[inactive] != 0):
            raise ValueError('Full image/active-column mapping differs')
    run = json.loads((dest / ('checkpoint_manifest.json' if args.phase == 'interim' else 'run_manifest.json')).read_text())
    if args.phase != 'interim':
        if sorted(r['rank'] for r in run['resources']) != list(range(8)):
            raise ValueError('Eight unique ranks required')
        if len({r['node'] for r in run['resources']}) != 8:
            raise ValueError('Eight distinct nodes required')
    if run['input_sha256'] != cfg['baseline_input_sha256']:
        raise ValueError('Baseline transport inputs differ')
    evidence = dict(verification, transfer_sha256=expected, local_result=str(dest),
                    active_full_consistency=True, unique_ranks_and_nodes=args.phase != 'interim')
    (REPORT / f'{args.phase}_{args.job}_verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (dest / 'transfer_manifest.json').write_text(json.dumps(expected, indent=2) + '\n')
    print('LOCAL_RESULT', dest)
    if args.phase in ('formal', 'interim'):
        print('NEXT: python experiments/ELLIPSE500x300_H120/compare_response_mismatch.py --result '
              + shlex.quote(str(dest)) + ' --output '
              + shlex.quote(str(REPORT / f'comparison_{args.phase}_{args.job}'))
              + (' --through-iteration 2000' if args.phase == 'interim' else ''))


if __name__ == '__main__':
    main()
