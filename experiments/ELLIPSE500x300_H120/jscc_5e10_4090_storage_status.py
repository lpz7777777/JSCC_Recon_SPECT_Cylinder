"""Read-only storage gate after actual 24-GPU probe 1686694; never submits jobs."""
import hashlib
import json
import sys
from jscc_5e10_common import DATA, REPORT, TOTAL, digest, read, write
from jscc_5e10_workflow import connection, command, q
from jscc_5e10_5090_8x2_trial import live_registration

GUARDS = ('advance', 'selection_4090_trial', 'selection_5090_8x2_trial',
          'selection_5090_8x2_monitor_repair', 'selection_5090_8x2_monitor_repair_verification',
          'streaming_4090_8x3')
SSD_ROOT = '/ssd/scxi717'
INSPECTION = '''import json,os
from pathlib import Path
p=Path('/ssd/scxi717');s=p.stat();v=os.statvfs(p)
print(json.dumps(dict(path=str(p),owner_uid=s.st_uid,mode=oct(s.st_mode&0o777),
 account_uid=os.getuid(),readable=os.access(p,os.R_OK),writable=os.access(p,os.W_OK),
 quota_exposed_total_bytes=v.f_blocks*v.f_frsize,
 quota_exposed_available_bytes=v.f_bavail*v.f_frsize)))
'''


def status():
    busy = [n for n in GUARDS if live_registration(DATA / (n + '_registration.json'))]
    if busy:
        print('CONTROLLER_ACTIVE', ','.join(busy))
        return
    old = REPORT / 'selection_job.json'
    if digest(old) != read(REPORT / 'streaming_4090_8x3_freeze.json')['original_selection_registry_sha256']:
        raise ValueError('Original 5090 registry changed; inspect without modifying it')
    selected = read(REPORT / 'selection_5090_8x2_monitor_repair_acceptance.json')
    if not selected['passed'] or not selected['strict_fetch_passed']:
        raise ValueError('Previously delivered selection acceptance missing')
    dense = 4849087 * 78920 * 4
    minimum_quota = 2 << 40
    required = (dense * 5 + 3) // 4 + (16 << 30)
    with connection('gpu') as c:
        storage = json.loads(command(c, '/data/home/scxi717/.conda/envs/torch/bin/python -c ' + q(INSPECTION), timeout=60))
        original = command(c, 'sacct -j 1685272 -n -P --format=JobID,State,ExitCode', timeout=60)
    available = (storage['writable'] and storage['quota_exposed_total_bytes'] >= minimum_quota
                 and storage['quota_exposed_available_bytes'] >= required)
    evidence = dict(resource_available=bool(available), storage=storage,
        minimum_quota_bytes=minimum_quota, required_available_bytes=required,
        full_uncompressed_response_bytes=dense,
        selected_events=4849087, actual_primary_photons=TOTAL,
        source_selection_acceptance_sha256=digest(REPORT / 'selection_5090_8x2_monitor_repair_acceptance.json'),
        initial_probe_job=1686694, initial_probe_freeze='9b5938545ad3d6c7',
        initial_probe_storage_passed=False, initial_probe_resources_passed=True,
        inspection_source_sha256=hashlib.sha256(INSPECTION.encode()).hexdigest(),
        monitor_source_sha256=digest(__file__), original_5090_registry_sha256=digest(old),
        original_5090_accounting=original, original_5090_untouched=True,
        read_only=True, validation_or_formal_submit_permitted=False,
        next_step='New bounded persistent-storage release and actual I/O check required' if available
                  else 'Await writable persistent SSD quota; preserve all completed stages')
    path = REPORT / 'streaming_4090_8x3_storage_latest_status.json'
    if not path.exists() or read(path) != evidence:
        write(path, evidence)
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    status()
