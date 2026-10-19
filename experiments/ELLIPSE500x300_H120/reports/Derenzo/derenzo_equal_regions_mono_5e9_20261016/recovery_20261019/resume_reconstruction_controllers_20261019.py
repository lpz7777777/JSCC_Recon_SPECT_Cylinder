"""Resume existing authorized jobs; preserve frozen scientific code and outputs.

Only execution control is repaired: distinguish reused Windows PIDs and map the
observed SSH channel timeout during read-only polls to the existing wait path.
Submission, transfer, authority and release failures retain their original
fail-closed behavior. Original controller deadlines are retained.
"""
import argparse
import ctypes
import functools
import json
import os
import time
from ctypes import wintypes
from pathlib import Path

from paramiko import SSHException

HERE = Path(__file__).resolve().parent
RECOVERY = HERE / 'reports/Derenzo/derenzo_equal_regions_mono_5e9_20261016/recovery_20261019'


def process_identity(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(ctypes.c_void_p(handle), ctypes.byref(created),
                                      ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
            raise OSError('Cannot verify process creation time')
        size = wintypes.DWORD(32768)
        name = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(ctypes.c_void_p(handle), 0, name, ctypes.byref(size)):
            raise OSError('Cannot verify process executable')
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return dict(created_epoch=ticks / 10000000 - 11644473600,
                    executable=Path(name.value).name)
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def live_registration(path):
    path = Path(path)
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('status') in ('complete', 'failed', 'stopped') or int(record.get('pid', 0)) <= 0:
        return False
    identity = process_identity(record['pid'])
    if identity is None:
        return False
    # A process created after the registration cannot be its original owner.
    if identity['created_epoch'] > float(record['started_epoch']) + 2:
        return False
    return identity['executable'].lower().startswith('python')


def read_only_timeout(function, error_type):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except SSHException as error:
            if str(error) != 'Timeout opening channel.':
                raise
            raise error_type('Read-only SSH channel timeout; preserve all registered jobs') from error
    return wrapped


def run(mode):
    if mode == 'nema':
        import jscc_5e10_5090_production_queue as workflow
        previous = json.loads((RECOVERY / 'nema_production_5090_8x4_queue_registration.json').read_text())
        deadline = previous['started_epoch'] + 14 * 86400
        workflow.live_registration = live_registration
        workflow.job_state = read_only_timeout(workflow.job_state, TimeoutError)
        original_command = workflow.command

        def command(client, text, *args, **kwargs):
            call = read_only_timeout(original_command, TimeoutError) if text.startswith('scontrol show job') else original_command
            return call(client, text, *args, **kwargs)

        workflow.command = command
        advance = workflow.advance_release
    else:
        import derenzo_campaign_workflow as workflow
        previous = json.loads((RECOVERY / 'derenzo_controller_registration.json').read_text())
        deadline = previous['started_epoch'] + 21 * 86400
        workflow.live_registration = live_registration
        workflow.source_complete = read_only_timeout(workflow.source_complete, ConnectionError)
        advance = workflow.advance
    with workflow.controlled():
        while time.time() < deadline:
            if advance():
                print('RECOVERY_CURRENT_WORKFLOW_COMPLETE', mode, flush=True)
                return
            print('RECOVERY_WAITING_REGISTERED_WORKFLOW', mode, time.time(), flush=True)
            time.sleep(min(300, max(0, deadline - time.time())))
        raise TimeoutError('Original controller deadline reached; preserve all jobs and outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['nema', 'derenzo'])
    run(parser.parse_args().mode)
