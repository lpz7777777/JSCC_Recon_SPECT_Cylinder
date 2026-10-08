"""Slurm completion guard with explicitly evidenced read-only diagnostic steps."""
import re


FAILURES = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')


def stage_completed(text, job, diagnostic_steps=None):
    job = str(job)
    diagnostic_steps = diagnostic_steps or {}
    for step in diagnostic_steps:
        if not re.fullmatch(re.escape(job) + r'\.\d+', step):
            raise ValueError('Only a named auxiliary numeric step can be excluded')
    rows = [line.split('|') for line in text.splitlines() if line.strip()]
    for row in rows:
        if len(row) < 2:
            continue
        failed = any(state in row[1] for state in FAILURES)
        if not failed:
            continue
        # No wildcard, job/batch/extern exemption or mismatched exit code.
        if len(row) >= 3 and diagnostic_steps.get(row[0]) == row[1:3]:
            continue
        raise RuntimeError('Failed stage retained; diagnose before repair:\n' + text)
    if any(row[:3] == [job, 'COMPLETED', '0:0'] for row in rows):
        return True
    array = {int(m[1]): row[1:3] for row in rows
             if (m := re.fullmatch(re.escape(job) + r'_(\d+)', row[0]))}
    return (set(array) == set(range(200))
            and all(value == ['COMPLETED', '0:0'] for value in array.values()))
