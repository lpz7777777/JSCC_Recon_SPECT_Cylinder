"""Read-only recovery of a registered job after a local pending-range false alarm."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

H = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(H))
import jscc_5e10_5090_8x4_monitor_repair as repair
from ehe_5e9_workflow import connection

record_path = repair.REPORT / (repair.STAGE + '_job.json')
record = repair.read(record_path)
freeze_path = repair.REPORT / (repair.STAGE + '_freeze.json')
freeze = repair.read(freeze_path)
repair.verify_files(repair.PAYLOAD, freeze['sha256'])
repair.verify_files(repair.REPORT, freeze['preserved_records'])
registrations = {}
for stage in repair.GUARDS:
    path = repair.DATA / (stage + '_registration.json')
    registrations[stage] = dict(record=repair.read(path) if path.exists() else None,
                               pid_alive=repair.live_registration(path))
assert not any(row['pid_alive'] for row in registrations.values()), registrations
tests = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest',
                        'test_jscc_5e10_5090_8x4_monitor_repair', '-v'], cwd=H,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
test_path = repair.REPORT / (repair.STAGE + '_controller_repair_tests.txt')
test_path.write_bytes(tests.stdout)
assert tests.returncode == 0 and b'\nOK' in tests.stdout
with connection('gpu') as c:
    snapshot = repair.remote(c, 'scontrol show job -o ' + str(record['job']), 60)
    assert repair.submitted_topology_matches(snapshot), snapshot
    accounting = repair.remote(c, 'sacct -j ' + str(record['job'])
                               + ' -n -P --format=JobID,State,ExitCode,Elapsed,MaxRSS,Start,End', 60)
    queue = repair.remote(c, "squeue -h -u scxi717 -o '%i|%j|%T|%M|%R'", 60)
    timestamp = repair.remote(c, "date -u '+%Y-%m-%dT%H:%M:%SZ'", 60).strip()
    log_paths = [repair.GPU_BASE + '/logs/' + repair.STAGE + '.' + str(record['job']) + ext
                 for ext in ('.out', '.err')]
    logs = {}
    with c.open_sftp() as s:
        for path in log_paths:
            try:
                stat = s.stat(path)
                with s.open(path, 'rb') as f:
                    content = f.read(min(stat.st_size, 1 << 20))
                local = repair.REPORT / (repair.STAGE + '_observed_' + Path(path).name)
                local.write_bytes(content)
                logs[path] = dict(exists=True, remote_size=stat.st_size,
                                  captured_bytes=len(content), captured_sha256=repair.digest(local),
                                  partial_live_observation=True)
            except FileNotFoundError:
                logs[path] = dict(exists=False)
snapshot_path = repair.REPORT / (repair.STAGE + '_recovered_scontrol.txt')
snapshot_path.write_bytes(snapshot.encode())
parsed = dict(t.split('=', 1) for t in snapshot.split() if '=' in t)
result = dict(passed=True, job=record['job'], checked_epoch=time.time(), remote_utc=timestamp,
              actual_state=parsed['JobState'], accounting=accounting, queue=queue,
              local_false_alarm='Pending scontrol NumNodes=8-8 was compared with exact string 8',
              repaired_pending_range_parser=True, additional_submissions=0,
              original_controller_exit_code=registrations[repair.STAGE]['record']['exit_code'],
              original_registration_preserved=True,
              job_record_sha256=repair.digest(record_path),
              freeze_sha256=repair.digest(freeze_path),
              actual_scontrol_sha256=repair.digest(snapshot_path),
              controller_before_sha256=freeze['sha256']['jscc_5e10_5090_8x4_monitor_repair.py'],
              controller_after_sha256=repair.digest(H / 'jscc_5e10_5090_8x4_monitor_repair.py'),
              controller_test_sha256=repair.digest(test_path),
              preserved_records=freeze['preserved_records'], registrations=registrations,
              submitted_topology_verified=True, gpu_execution_acceptance_pending=True,
              observed_logs=logs, automations_remain_paused=True,
              validation_or_formal_submitted=False, read_only=True)
repair.write(repair.REPORT / (repair.STAGE + '_submission_recovery_acceptance.json'), result)
print(json.dumps(dict(job=record['job'], state=parsed['JobState'], queue=queue,
                     recovered_topology=True, logs=logs)), flush=True)
