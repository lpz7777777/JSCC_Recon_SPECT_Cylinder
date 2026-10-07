"""Read-only verification of the accepted 2026-10-07 dual-energy baseline.

No SSH, simulation, event selection, training, submission or reconstruction.
Optional data/result checks read complete bytes; missing data is a failure.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'docs/baselines/dual_energy_20261007/manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def local(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path escapes repository: ' + relative)
    return path


def read(relative):
    return json.loads(local(relative).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_map(mapping, label, normalize_lf=()):
    for relative, sha in mapping.items():
        path = local(relative)
        actual = (hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                  if relative in normalize_lf else digest(path))
        require(actual == sha, 'SHA mismatch: ' + relative)
    print(label, len(mapping))


def verify(m, payload=False, data=False, results=False):
    require(m['schema_version'] == 1 and m['baseline_id'] == 'dual_energy_20261007',
            'Unsupported baseline identity')
    normalized = m.get('source_lf_normalized_paths', [])
    require(not set(normalized).intersection(m['frozen_live_source_sha256']),
            'Frozen executed source must retain exact bytes')
    verify_map(m['code_sha256'], 'CODE_SHA_VERIFIED', normalized)
    verify_map(m['evidence_sha256'], 'EVIDENCE_SHA_VERIFIED',
               m.get('metadata_lf_normalized_paths', []))
    cleanup = read(m['cleanup_manifest'])
    for item in cleanup['removed']:
        require(not local(item['path']).exists(), 'Retired launcher restored: ' + item['path'])
    for gallery in m['artifact_manifests']:
        metadata = read(gallery)
        require(metadata['job'] == m['formal_job'], 'Gallery job changed')
        verify_map({(Path(gallery).parent / name).as_posix(): item['sha256']
                    for name, item in metadata['files'].items()}, 'ARTIFACT_SHA_VERIFIED')

    s = read(m['formal_summary'])
    v = read(m['formal_verification'])
    a = read(m['execution_acceptance'])
    short = read(m['validation_summary'])
    require(short['passed'] and short['job'] == 1669189, 'Actual six-channel pilot missing')
    require(s['passed'] and s['full_six_imaging_completed'] and s['job'] == 1669255,
            'Actual formal completion missing')
    require(v['passed'] and v['mode'] == 'formal' and v['model'] == 'continuous_energy',
            'Formal verification/model changed')
    require(v['iterations'] == 10000 and v['save_step'] == 50 and v['accepted_events'] == 483743,
            'Iteration/event contract changed')
    for record in (s, v, a):
        require(record['contract_sha256'] == m['contract_sha256'], 'Contract identity differs')
    require(v['authority_sha256'] == a['authority_sha256'] == m['authority_sha256'],
            'Actual authority differs')
    require(a['passed'] and a['scientific_QA_completed'] and a['active_cells'] == 78920 and
            a['full_cells'] == 132040 and a['actual_primary_gamma'] == 5000000000 and
            a['workers'] == 200 and a['views'] == 20 and a['event_policy'] == 'legacy',
            'Transport/geometry/acceptance differs')
    require(len(v['outputs']) == 6 and {x['channel'] for x in v['outputs']} == set(m['channels'])
            and all(x['frames'] == 200 for x in v['outputs']), 'Six complete histories required')
    require(len(v['checkpoints']) == 600, '600 checkpoints required')
    for phase in ('440_single', '218_corrected', 'compton_jscc'):
        require(sorted(x['iteration'] for x in v['checkpoints'] if x['phase'] == phase)
                == list(range(50, 10001, 50)), 'Checkpoint phase coverage changed')
    rr = v['resources']
    require(len(rr) == 8 and len({x['rank'] for x in rr}) == 8 and
            len({x['node'] for x in rr}) == 8 and
            sum(x['accepted_events'] for x in rr) == 483743, 'Rank/event identity differs')
    require(all(x == 0 for x in a['regression_2000_relative_L2'].values()) and
            len(a['regression_2000_relative_L2']) == 2, 'Accepted 2000-prefix regression differs')
    for name in ('peak_GPU_reserved_fraction', 'peak_process_RSS_fraction', 'slurm_peak_RSS_fraction'):
        require(0 < a[name] <= .8, 'Actual resource margin failed: ' + name)
    require(a['actual_allocated_host_bytes_per_node'] == 60000 * 1024**2,
            'Actual memory allocation differs')
    cal = read(m['calibration_acceptance'])
    require(cal['passed'] and cal['event_policy'] == 'legacy' and cal['new_photons'] == 0
            and not cal['new_training'], 'Accepted legacy calibration differs')
    require(read(m['source_inventory'])['baseline_id'] == m['baseline_id'], 'Inventory differs')
    print('ACTUAL_SMALL_PROOFS_VERIFIED', 'formal=1669255', 'pilot=1669189', 'ranks=8',
          'events=483743', 'channels=6', 'checkpoints=600')
    if payload:
        verify_map(m['frozen_payload_sha256'], 'FROZEN_PAYLOAD_SHA_VERIFIED')
    if data:
        verify_map(m['original_input_sha256'], 'ORIGINAL_INPUT_SHA_VERIFIED')
        verify_map(m['factor_payload_sha256'], 'COMPLETE_FACTOR_SHA_VERIFIED')
        verify_map(m['truth_sha256'], 'ACTUAL_TRUTH_SHA_VERIFIED')
    if results:
        folder = local(m['local_result'])
        for item in v['outputs']:
            for kind, sha in item['sha256'].items():
                p = folder / ('Image_' + item['channel'] + '_' + kind + '.float32')
                require(digest(p) == sha, 'Result SHA differs: ' + str(p))
        require(digest(folder / 'PredictedCntStat_218_From440.float32') == v['prediction_sha256'],
                'Fixed cross-window prediction differs')
        for snapshot in v['checkpoints']:
            relative = f"checkpoints_{snapshot['phase']}/checkpoint_{snapshot['iteration']:06d}"
            sub = folder / relative
            p = sub / 'checkpoint_manifest.json'
            require(digest(p) == snapshot['manifest_sha256'], 'Checkpoint manifest differs')
            record = json.loads(p.read_text(encoding='utf-8'))
            for channel, kinds in record['outputs'].items():
                for kind, sha in kinds.items():
                    require(digest(sub / ('Image_' + channel + '_' + kind + '.float32')) == sha,
                            'Checkpoint output differs: ' + relative + '/' + channel + '/' + kind)
        print('RESULT_AND_CHECKPOINT_SHA_VERIFIED', 'channels=6', 'checkpoints=600')
    omitted = [name for name, enabled in [('frozen_payload', payload), ('original_inputs_and_Factors', data),
                                          ('raw_results_and_checkpoints', results)] if not enabled]
    print('OPTIONAL_FULL_BYTE_CHECKS_NOT_RUN', ','.join(omitted) or 'none')
    print('BASELINE_READ_ONLY_CHECK_PASSED', m['baseline_id'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DEFAULT)
    p.add_argument('--verify-payload', action='store_true')
    p.add_argument('--verify-data', action='store_true')
    p.add_argument('--verify-results', action='store_true')
    args = p.parse_args()
    verify(json.loads(args.manifest.read_text(encoding='utf-8')),
           args.verify_payload, args.verify_data, args.verify_results)
