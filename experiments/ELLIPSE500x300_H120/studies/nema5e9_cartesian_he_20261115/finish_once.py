"""One finite Cartesian study handoff, at most seven days; not a scheduled task.

Only this registered matrix process and this new three-route formal allocation.
Never starts another matrix process, cancels jobs, retries uncertain submissions,
or resubmits a scientific failure. Inputs and controller code are SHA-bound.
"""
import datetime
import os
import sys
import time
import traceback
from pathlib import Path
import psutil
from common import read, write, sha, now
from prepare import CODE, DATA, REPORT
from workflow import matrix_status, gpu_preflight
from handoff import transfer_and_submit, formal_status, fetch

REG = DATA/'completion_once_registration.json'
BOUND = REPORT/'completion_source_manifest.json'


def verify():
    binding = read(BOUND)
    for name, expected in binding['files'].items():
        if sha(name) != expected:
            raise ValueError('Bound Cartesian completion source changed: '+name)
    return binding


def main():
    verify()
    if REG.exists():
        raise RuntimeError('Completion registration exists; inspect identity, never duplicate')
    lock = DATA/'completion_once.lock'
    fd = os.open(str(lock), os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    p = psutil.Process()
    deadline = time.time()+7*86400
    registration = dict(pid=p.pid, create_time=p.create_time(), cmdline=p.cmdline(),
        status='ACTIVE', started=now(), deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
        study=DATA.name, matrix_registration_sha256=sha(REPORT/'matrix_registration.json'),
        source_manifest_sha256=sha(BOUND), periodic_chat_automation=False,
        can_cancel=False, scientific_resubmission=False,
        authorized_formal_routes=['440_SC','440_JSCC','218_SC'])
    write(REG, registration)
    network_failures = 0
    try:
        while time.time() < deadline:
            verify()
            if not (REPORT/'formal_registration.json').exists():
                try:
                    state = matrix_status()
                except Exception as e:
                    network_failures += 1
                    registration.update(last_check=now(), stage='MATRIX_STATUS_NETWORK_WAIT', error=repr(e))
                    write(REG, registration, replace=True)
                    if network_failures > 144:
                        raise RuntimeError('Bounded matrix status network wait exhausted') from e
                    time.sleep(300)
                    continue
                network_failures = 0
                if not state['alive'] and 'completed.json' not in state:
                    raise RuntimeError('Matrix process exited without completion; preserve partials, no automatic retry')
                if state['alive']:
                    registration.update(last_check=now(), stage='MATRIX_RUNNING', progress=state.get('progress.json'))
                    write(REG, registration, replace=True)
                    time.sleep(300)
                    continue
                availability = gpu_preflight()
                if not availability['passed']:
                    registration.update(last_check=now(), stage='GPU_CONNECTION_OR_QUOTA_WAIT',
                                        reason=availability.get('error', 'Actual quota/partition not ready'))
                    write(REG, registration, replace=True)
                    time.sleep(300)
                    continue
                registration.update(last_check=now(), stage='STRICT_MATRIX_TRANSFER_AND_FORMAL_SUBMIT')
                write(REG, registration, replace=True)
                job = transfer_and_submit()  # any failure/ambiguous write stops
                registration['job'] = job['job']
            try:
                state = formal_status()
            except Exception as e:
                network_failures += 1
                registration.update(last_check=now(), stage='FORMAL_STATUS_NETWORK_WAIT', error=repr(e))
                write(REG, registration, replace=True)
                if network_failures > 144:
                    raise RuntimeError('Bounded formal status network wait exhausted') from e
                time.sleep(300)
                continue
            network_failures = 0
            write(REPORT/'formal_latest_status.json', state, replace=True)
            registration.update(last_check=now(), stage='FORMAL_'+state['decision'],
                                progress=state['details'].get('progress.json'))
            write(REG, registration, replace=True)
            if state['decision'] == 'FAILED':
                raise RuntimeError('Formal reconstruction failed; successful outputs retained, no resubmission')
            if state['decision'] == 'SUCCESS':
                verify()
                delivery = fetch()
                registration.update(status='STRICTLY_DELIVERED_VISUAL_QA_PENDING', completed=now(), delivery=delivery)
                write(REG, registration, replace=True)
                return
            time.sleep(300)
        raise TimeoutError('Finite seven-day handoff deadline reached; remote jobs unchanged')
    except Exception as e:
        registration.update(status='STOPPED_FAILURE', stopped=now(), error=repr(e), traceback=traceback.format_exc())
        write(REG, registration, replace=True)
        raise
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    main()
