"""Original MLEM continuation with a separately frozen human HOLD authorization."""
import hashlib
import json
import math
from pathlib import Path
import shutil

from ehe_common import DATA, HERE, REPORT, RESPONSES, digest, read, verify_files, write


def binding():
    path = REPORT / 'reconstruction_execution_freeze.json'
    if not path.exists():
        return None
    value = read(path)
    payload = DATA / value['payload_dir']
    verify_files(payload, value['sha256'])
    if digest(payload / 'release_manifest.json') != value['manifest_sha256']:
        raise ValueError('Immutable reconstruction release manifest changed')
    if digest(REPORT / 'physical_continuation_policy.json') != value['policy_sha256']:
        raise ValueError('Explicit continuation policy changed')
    if value['producer_release_key'] != read(REPORT / 'response_repair_freeze.json')['release_key']:
        raise ValueError('Scientific response release changed')
    from ehe_common import GPU_BASE
    if value['root'] != GPU_BASE + '/reconstruction_releases/' + value['release_key']:
        raise ValueError('Reconstruction release path differs')
    return value


def execution_release():
    value = binding()
    if value is not None:
        return value['root']
    from ehe_5e9_workflow import release
    return release('gpu')


def policy_argument():
    from ehe_5e9_workflow import q
    value = binding()
    return '' if value is None else ' --physical-policy ' + q(value['root'] + '/physical_continuation_policy.json')


def setup():
    existing = binding()
    if existing is not None:
        return existing
    from ehe_5e9_workflow import base, connection, command, put_tree, q, GPU_PYTHON
    scientific = read(REPORT / 'response_repair_freeze.json')
    policy = REPORT / 'physical_continuation_policy.json'
    original = DATA / scientific['payload_dir']
    baseline_names = ('ehe_common.py', 'torch_active_operator.py', 'single_checkpoint_mlem.py',
                      'whole_geometry.npz', 'truth_3mm.npz', 'config.json')
    for name in baseline_names:
        if digest(original / name) != scientific['sha256'][name]:
            raise ValueError('Original reconstruction input changed: ' + name)
    # The solve, saved-history loop and fixed-background construction stay exact.
    def core(path):
        text = Path(path).read_text(encoding='utf-8')
        return text.split('    out.mkdir', 1)[1].split('    record=dict', 1)[0]
    if core(original / 'run_ehe_reconstruction.py') != core(HERE / 'run_ehe_reconstruction.py'):
        raise ValueError('Original scientific reconstruction body changed')
    if digest(original / 'run_ehe_reconstruction.py') != scientific['sha256']['run_ehe_reconstruction.py']:
        raise ValueError('Original producer reconstruction source changed')
    sources = {n: scientific['sha256'][n] for n in baseline_names}
    sources.update({n: digest(HERE / n) for n in ('run_ehe_reconstruction.py', 'ehe_execution_policy.py')})
    sources['physical_continuation_policy.json'] = digest(policy)
    identity = dict(sha256=sources, producer_release_key=scientific['release_key'],
                    producer_manifest_sha256=digest(REPORT / 'response_repair_freeze.json'),
                    scientific_reconstruction_body_sha256=hashlib.sha256(core(original / 'run_ehe_reconstruction.py').encode()).hexdigest(),
                    scope='Human continuation policy only; original MLEM, geometry, matrices and fixed-background method')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    payload = DATA / 'reconstruction_releases' / key
    payload.mkdir(parents=True, exist_ok=False)
    for name in sources:
        source = policy if name == 'physical_continuation_policy.json' else (
            original / name if name in baseline_names else HERE / name)
        shutil.copy2(source, payload / name)
    manifest = dict(release_key=key, **identity)
    write(payload / 'release_manifest.json', manifest)
    root = base('gpu') + '/reconstruction_releases/' + key
    with connection('gpu') as c:
        # There must be no unfinished reconstruction to overwrite or duplicate.
        code = 'from pathlib import Path;p=Path(' + repr(base('gpu') + '/results') + ');assert not (p/"validation").exists();assert not (p/"formal").exists();print("NEW_RECONSTRUCTION_OUTPUTS")'
        print(command(c, q(GPU_PYTHON) + ' -c ' + q(code)))
        command(c, 'mkdir -p ' + q(base('gpu') + '/reconstruction_releases'))
        with c.open_sftp() as s:
            put_tree(s, payload, root)
        code = 'from pathlib import Path;from ehe_common import *;p=Path(' + repr(root) + ');verify_files(p,read(p/"release_manifest.json")["sha256"]);print("IMMUTABLE_RECONSTRUCTION_RELEASE_SHA_PASS")'
        print(command(c, 'cd ' + q(root) + ' && ' + q(GPU_PYTHON) + ' -c ' + q(code)))
        from ehe_conversion_workflow import response_root
        code = 'from pathlib import Path;from ehe_common import read;from ehe_execution_policy import physical_permission;p=Path(' + repr(root) + ');v=physical_permission(Path(' + repr(base('gpu') + '/physical') + '),Path(' + repr(base('gpu') + '/counts') + '),Path(' + repr(response_root()) + '),read(p/"release_manifest.json"),p/"physical_continuation_policy.json");assert not v["physical_calibration_passed"];print("BOUND_HUMAN_CONTINUATION_IDENTITY_PASS")'
        print(command(c, 'cd ' + q(root) + ' && ' + q(GPU_PYTHON) + ' -c ' + q(code)))
    value = dict(release_key=key, producer_release_key=scientific['release_key'],
                 payload_dir=str(payload.relative_to(DATA)), root=root, sha256=sources,
                 manifest_sha256=digest(payload / 'release_manifest.json'),
                 policy_sha256=digest(policy), scientific_reconstruction_body_sha256=identity['scientific_reconstruction_body_sha256'],
                 original_producer_source_sha256=scientific['sha256']['run_ehe_reconstruction.py'])
    write(REPORT / 'reconstruction_execution_freeze.json', value)
    return value


def advance_reconstruction():
    from ehe_5e9_workflow import connection, completed, fetch, submit_reconstruction
    value = binding()
    if value is None:
        raise ValueError('Explicit reconstruction freeze required')
    acceptance = read(REPORT / 'response_conversion_identity_acceptance.json')
    authorization = read(REPORT / 'physical_continuation_policy.json')
    if not acceptance['passed'] or acceptance['factor_manifest_sha256'] != authorization['factor_sha256']:
        raise ValueError('Original complete responses required')
    if not (REPORT / 'validation_job.json').exists():
        submit_reconstruction('validation', 1800, 90)
        return
    with connection('gpu') as c:
        done, _ = completed(c, read(REPORT / 'validation_job.json')['job'])
    if not done:
        return
    if not (REPORT / 'validation_summary.json').exists():
        fetch('validation')
    proof = read(REPORT / 'validation_summary.json')
    if (not proof['passed'] or proof['release_key'] != value['release_key']
            or proof['physical_policy_sha256'] != value['policy_sha256']
            or proof['physical_calibration_passed'] is not False):
        raise ValueError('Complete actual numerical validation and policy identity required')
    if not (REPORT / 'formal_job.json').exists():
        limit = math.ceil(max(proof['phase_seconds'].values()) * 20 * 1.8 + 300)
        submit_reconstruction('formal', limit, math.ceil(limit * 2 / 60) + 15)
        return
    with connection('gpu') as c:
        done, _ = completed(c, read(REPORT / 'formal_job.json')['job'])
    if done and not (REPORT / 'formal_summary.json').exists():
        fetch('formal')
