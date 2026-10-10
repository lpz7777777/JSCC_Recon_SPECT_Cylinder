"""Independent expected-dose5e10 matrix-Poisson run, reusing accepted science bytes."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from ehe_common import HERE, GPU_BASE, digest, hashes, read, write, verify_files
import ehe_forward_poisson_workflow as original
from ehe_forward_poisson_5e10_data import STUDY, dose_budget, verify_counts
from ehe_5e10_workflow import pid_alive

DATA = HERE / 'generated' / STUDY
REPORT = HERE / 'reports/NEMA_Body_H60' / STUDY
BASE = GPU_BASE.rsplit('/', 1)[0] + '/' + STUDY
PREVIOUS_REPORT = HERE / 'reports/NEMA_Body_H60/ehe_forward_poisson_5e9_200'
PREVIOUS_DATA = HERE / 'generated/ehe_forward_poisson_5e9_200'
SEEDS = {'A218': 34100101, 'A440': 34100102, 'C440to218': 34100103}
DOSE = 50_000_000_000


def adapted_data_source(source):
    changes = {
        b"STUDY = 'ehe_forward_poisson_5e9_200'":
            b"STUDY = 'ehe_forward_poisson_5e10_200'",
        b"cfg['expected_emitted_photons']!=5_000_000_000":
            b"cfg['expected_emitted_photons']!=50_000_000_000",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError('Accepted data-source adaptation is not exactly bounded')
        source = source.replace(before, after)
    return source


def freeze():
    path = REPORT / 'freeze.json'
    if path.exists():
        frozen = read(path)
        verify_files(DATA / frozen['payload_dir'], frozen['sha256'])
        return frozen
    previous = read(PREVIOUS_REPORT / 'freeze.json')
    source = PREVIOUS_DATA / previous['payload_dir']
    verify_files(source, previous['sha256'])
    accepted = read(PREVIOUS_REPORT / 'formal_summary.json')
    if not accepted['passed'] or accepted['iterations'] != 200:
        raise ValueError('Original accepted matrix-Poisson200 baseline required')
    config = read(source / 'config.json')
    if accepted['factor_sha256'] != config['factor_sha256']:
        raise ValueError('Original complete factor identity differs')
    if set(SEEDS.values()) & set(config['noise_seeds'].values()):
        raise ValueError('New independent noise seeds required')
    config.update(study=STUDY, expected_emitted_photons=DOSE,
                  noise_seeds=SEEDS, physical_calibration_claim=False,
                  transport_performed=False,
                  human_instruction='New EHE5e10 matrix-forward plus independent Poisson noise; original MLEM200')
    payload = DATA / 'payload'
    payload.mkdir(parents=True, exist_ok=False)
    for name in previous['sha256']:
        if name == 'config.json':
            continue
        shutil.copy2(source / name, payload / name)
    data_source = adapted_data_source((source / 'ehe_forward_poisson_data.py').read_bytes())
    if data_source != (HERE / 'ehe_forward_poisson_5e10_data.py').read_bytes():
        raise ValueError('Local5e10 adapter differs from the bounded accepted source')
    (payload / 'ehe_forward_poisson_data.py').write_bytes(data_source)
    write(payload / 'config.json', config)
    budget = dose_budget(np.load(payload / 'truth_3mm.npz'), config)
    if not math.isclose(budget['expected_emitted_photons'], DOSE, rel_tol=1e-13):
        raise ValueError('Expected emitted5e10 source budget differs')
    if not math.isclose(budget['expected_primary_photons']['218'] / DOSE,
                        .29380779868182727, rel_tol=1e-13):
        raise ValueError('Actual3D source integral/gamma-yield fraction differs')
    preserved = {n: digest(payload / n) for n in previous['sha256']
                 if n not in ('config.json', 'ehe_forward_poisson_data.py')}
    if any(s != previous['sha256'][n] for n, s in preserved.items()):
        raise ValueError('Original scientific runner/operator/verifier/source changed')
    files = hashes(payload)
    key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    frozen = dict(study=STUDY, release_key=key, payload_dir='payload',
                  root=BASE + '/releases/' + key, sha256=files,
                  response_root=config['response_root'], budget=budget,
                  original_response_release_key=previous['original_response_release_key'],
                  original_reconstruction_release_key=previous['original_reconstruction_release_key'],
                  preregistered_noise_seeds=SEEDS,
                  original_matrix_poisson_release_key=previous['release_key'])
    write(payload / 'release_manifest.json', frozen)
    write(path, frozen)
    write(REPORT / 'science_reuse_acceptance.json', dict(
        passed=True, original_freeze_sha256=digest(PREVIOUS_REPORT / 'freeze.json'),
        original_formal_summary_sha256=digest(PREVIOUS_REPORT / 'formal_summary.json'),
        unchanged_scientific_files_sha256=preserved,
        original_data_source_sha256=digest(source / 'ehe_forward_poisson_data.py'),
        adapted_data_source_sha256=digest(payload / 'ehe_forward_poisson_data.py'),
        bounded_source_changes=['Study identity5e9->5e10', 'Strict expected emitted dose5e9->5e10'],
        config_changes=['Study identity', 'Expected emitted dose', 'New independent PCG64 seeds', 'Human instruction'],
        expected_emitted_photons=DOSE, noise_seeds=SEEDS,
        source_integral_normalization='Expected emitted gamma only; no detected-count matching',
        physical_calibration_claim=False, transport_performed=False,
        workflow_sha256=digest(__file__),
        reused_workflow_sha256=digest(original.__file__)))
    return frozen


def configure_original_workflow():
    # Reuse the existing submit/verify/fetch logic only inside this Python process.
    # Its previously delivered files, payloads and results remain unchanged.
    original.STUDY, original.DATA, original.REPORT, original.BASE = STUDY, DATA, REPORT, BASE
    original.freeze = freeze
    original.verify_counts = verify_counts


@contextlib.contextmanager
def controller(hours=None):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / 'controller.json'
    if path.exists():
        previous = read(path)
        if previous['status'] == 'running' and pid_alive(previous['pid']):
            raise RuntimeError('Registered study controller is alive; no concurrent advance/fetch/submit')
    value = dict(pid=os.getpid(), status='running', started_epoch=time.time(),
                 workflow_sha256=digest(__file__), bounded_hours=hours,
                 recurring_automation=False)
    write(path, value)
    try:
        yield
    except BaseException as exc:
        value.update(status='failed', error=str(exc), finished_epoch=time.time())
        write(path, value)
        raise
    else:
        value.update(status='complete', exit_code=0, finished_epoch=time.time())
        write(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'advance', 'watch', 'status'])
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    configure_original_workflow()
    if args.action == 'freeze':
        print(freeze()['release_key'], flush=True)
    elif args.action == 'status':
        with original.connection('gpu') as remote:
            for stage in ('generate_validation', 'validation_acceptance', 'formal', 'formal_acceptance'):
                registered = original.registered(stage)
                if registered:
                    print(stage, original.accounting(remote, registered['job']), flush=True)
    else:
        with controller(args.hours if args.action == 'watch' else None):
            began = time.monotonic()
            while True:
                original.advance()
                if args.action == 'advance' or (REPORT / 'formal_summary.json').exists():
                    break
                if time.monotonic() - began >= args.hours * 3600:
                    raise TimeoutError('Bounded local watch elapsed; preserve jobs and all outputs')
                time.sleep(30)


if __name__ == '__main__':
    main()
