"""Verify retrieved HOLD statistics against original workers, without simulation or fits."""
import csv
import math
import re
from collections import Counter

import numpy as np
from ehe_common import DATA, REPORT, RESPONSES, allocated_bytes, digest, read, write


def verify():
    job = read(REPORT / 'physical_job.json')['job']
    proof = read(REPORT / 'physical_hold_fetch_acceptance.json')
    gate = read(REPORT / 'physical_gate.json')
    audit = DATA / f'physical_{job}/physical_audit.csv'
    workers_path = DATA / 'transport/worker_counts.npz'
    assert proof['passed'] and proof['job'] == job and not gate['passed']
    assert digest(audit) == gate['files']['physical_audit.csv'] == proof['data_csv_sha256']
    assert digest(workers_path) == proof['remote_before']['worker_counts.npz']['sha256']
    with audit.open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * (20 * (2312 + 1) + 1)
    workers = np.load(workers_path)
    seen = set()
    summaries = {}
    totals = []
    max_se_error = 0.0
    hold_count = 0
    undetermined = 0
    for name, energy, window in [('A218', 218, 218), ('A440', 440, 440), ('C440to218', 440, 218)]:
        data = workers[f'CntStat_{window}_from{energy}'].reshape(20, 10, 2312)
        view_counts = data.sum(axis=1)
        bin_se = data.std(axis=1, ddof=1) * math.sqrt(10)
        view_se = data.sum(axis=2).std(axis=1, ddof=1) * math.sqrt(10)
        global_se = math.sqrt(sum(float(data[v].sum(axis=1).var(ddof=1)) * 10 for v in range(20)))
        channel_rows = [r for r in rows if r['response'] == name]
        assert len(channel_rows) == 20 * 2313 + 1
        for r in channel_rows:
            scope = r['scope']
            view = int(r['view']) - 1 if r['view'] else None
            key = (name, view, scope)
            assert key not in seen
            seen.add(key)
            p, o, se = (float(r[k]) for k in ('predicted', 'observed', 'standard_error'))
            assert all(math.isfinite(x) and x >= 0 for x in (p, o, se))
            if scope == 'global':
                expected_o, expected_se = float(data.sum()), global_se
            elif scope == 'view_total':
                assert 0 <= view < 20
                expected_o, expected_se = float(view_counts[view].sum()), float(view_se[view])
            else:
                assert scope.startswith('bin_') and 0 <= view < 20
                b = int(scope[4:]); assert 0 <= b < 2312
                expected_o, expected_se = float(view_counts[view, b]), float(bin_se[view, b])
            assert o == expected_o
            max_se_error = max(max_se_error, abs(se - expected_se))
            assert math.isclose(se, expected_se, rel_tol=1e-12, abs_tol=1e-12)
            relative = abs(p - o) / max(o, 1)
            adequate = o >= 100
            hold = adequate and relative > 0.10 and abs(p - o) > 3 * se
            status = 'HOLD' if hold else 'PASSED' if adequate else 'UNDETERMINED'
            assert r['adequate'] == str(adequate) and r['status'] == status
            assert math.isclose(relative, float(r['relative_bias']), rel_tol=1e-12, abs_tol=1e-12)
            hold_count += int(hold)
            undetermined += int(not adequate)
            if scope in ('view_total', 'global'):
                totals.append(r)
        global_row = next(r for r in channel_rows if r['scope'] == 'global')
        views = [r for r in channel_rows if r['scope'] == 'view_total']
        bins = [r for r in channel_rows if r['scope'].startswith('bin_')]
        p, o, se = (float(global_row[k]) for k in ('predicted', 'observed', 'standard_error'))
        assert math.isclose(p, sum(float(r['predicted']) for r in views), rel_tol=1e-5, abs_tol=1e-4)
        for r in views:
            bp = sum(float(b['predicted']) for b in bins if b['view'] == r['view'])
            assert math.isclose(bp, float(r['predicted']), rel_tol=1e-5, abs_tol=1e-4)
        summaries[name] = dict(predicted=p, observed=o, worker_standard_error=se,
            signed_relative_bias=(p-o)/o, predicted_over_observed=p/o,
            absolute_difference_worker_se=abs(p-o)/se, global_status=global_row['status'],
            view_hold=[int(r['view']) for r in views if r['status'] == 'HOLD'],
            status_counts=dict(Counter(r['status'] for r in channel_rows)),
            bins_adequate=sum(r['adequate'] == 'True' for r in bins))
    assert hold_count == gate['hold_count'] == 22 and undetermined == gate['undetermined_bins'] == 138720
    accounting_path = REPORT / f'physical_{job}/accounting.txt'
    assert digest(accounting_path) == proof['accounting_sha256']
    records = [line.split('|') for line in accounting_path.read_text().splitlines() if line.strip()]
    root = next(r for r in records if r[0] == str(job))
    assert root[1:3] == ['FAILED', '1:0']
    assert not any((REPORT / n).exists() for n in ['validation_job.json', 'formal_job.json'])
    allocated = allocated_bytes('AllocTRES=' + root[5])
    peak = 0
    for r in records:
        m = re.fullmatch(r'([0-9.]+)([KMGT]?)', r[3])
        if m:
            peak = max(peak, int(float(m[1]) * 1024 ** (' KMGT'.index(m[2]) if m[2] else 0)))
    assert 0 < peak <= .8 * allocated
    assert allocated == gate['resource']['host_allocated_bytes']
    assert gate['resource']['rss_fraction'] <= .8 and gate['resource']['gpu_reserved_fraction'] <= .8
    csv_path = REPORT / f'physical_{job}/view_global_audit.csv'
    with csv_path.open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(totals)
    result = dict(passed=True, scope='Read-only independent statistical and resource verification of scientific HOLD; no physical PASS',
        job=job, scientific_status='HOLD', scientific_gate_passed=False, rows_verified=len(rows),
        hold_count=hold_count, undetermined_bins=undetermined, max_worker_se_absolute_error=max_se_error,
        channels=summaries, resource=dict(actual_allocated_bytes=allocated, slurm_maxrss_bytes=peak,
            slurm_maxrss_fraction=peak/allocated, runtime=gate['resource'], margin_passed=True,
            exit_state='FAILED', exit_code='1:0', reason='Physical response HOLD'),
        files={'physical_gate.json':digest(REPORT/'physical_gate.json'),
            'physical_hold_fetch_acceptance.json':digest(REPORT/'physical_hold_fetch_acceptance.json'),
            f'physical_{job}/view_global_audit.csv':digest(csv_path),
            f'physical_{job}/accounting.txt':digest(accounting_path)},
        complete_audit_data_sha256=digest(audit), verifier_sha256=digest(__file__),
        prediction_refitted=False, simulations_repeated=False, validation_submitted=False, formal_submitted=False,
        root_cause='UNDETERMINED; systematic cross-window shortfall identified, no causal attribution from these totals alone')
    write(REPORT / 'physical_hold_diagnosis.json', result)
    print('READ_ONLY_HOLD_STATISTICS_PASS', result['channels'])
    print('RESOURCES', result['resource'])
    return result


if __name__ == '__main__':
    verify()
