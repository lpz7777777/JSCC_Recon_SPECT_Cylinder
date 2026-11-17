"""Collect only new monoenergetic HE imaging, reuse accepted HE response bytes."""
from pathlib import Path
import argparse
import shutil
import numpy as np
from opengate_ehe_common_20261113 import read, save, save_array, sha, now


def collect(root):
    root = Path(root)
    inputs = root / 'inputs'
    cfg = read(inputs / 'config.json')
    dest = root / 'collected'
    if (dest / 'complete.json').exists():
        raise FileExistsError('Do not repeat successful collection')
    jobs = read(inputs / 'imaging_jobs.json')
    observations = np.zeros((2, 20, 68, 34), dtype=np.int64)
    tagged = np.zeros((2, 2, 20, 68, 34), dtype=np.int64)
    emitted = np.zeros((20, 2), dtype=np.int64)
    records = {}
    for job in jobs:
        out = root / 'workers/imaging' / f"{job['index']:04d}"
        meta = read(out / 'complete.json')
        if not meta['passed'] or meta['job'] != job or meta['actual_primary_events'] != job['primaries']:
            raise ValueError('Incomplete/changed worker ' + str(out))
        for name, r in meta['files'].items():
            p = out / name
            if p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
                raise ValueError('Worker member identity mismatch')
        with np.load(out / 'counts.npz') as f:
            c = f['counts']
        if c.shape != (2, 68, 34) or c.dtype != np.int64 or np.any(c < 0):
            raise ValueError('Bad counts')
        ei = (218, 440).index(job['energy_keV'])
        observations[:, job['view']] += c
        tagged[:, ei, job['view']] += c
        emitted[job['view'], ei] += job['primaries']
        records[str(job['index'])] = dict(sha256=sha(out / 'complete.json'), primaries=job['primaries'])
    np.testing.assert_array_equal(observations, tagged.sum(axis=1))
    np.testing.assert_array_equal(emitted, cfg['emitted_primaries_by_view_energy'])
    assert int(emitted.sum()) == cfg['imaging_primary_gamma_total']
    check = read(root / 'workers/imaging/0000/source_check/complete.json')
    assert check['passed'] and check['formal_imaging_primaries'] == 0
    response = inputs / 'response_kernels.npz'
    assert sha(response) == cfg['response_kernel_sha256']
    save_array(dest / 'observations.npz', counts=observations, tagged=tagged, emitted=emitted)
    shutil.copy2(response, dest / 'response_kernels.npz')
    save(dest / 'complete.json', dict(passed=True, created_utc=now(), workers=records,
        actual_imaging_primaries=int(emitted.sum()), counts_by_window=observations.sum(axis=(1, 2, 3)).tolist(),
        actual_primary_by_view_energy=emitted.tolist(), config_sha256=sha(inputs / 'config.json'),
        response_reused_from='accepted HE 46d438089710a6ce; 88 workers; 4.4e9 histories',
        response_new_primaries=0, physical_calibration_passed=False,
        files={n: dict(bytes=(dest / n).stat().st_size, sha256=sha(dest / n))
               for n in ('observations.npz', 'response_kernels.npz')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    collect(p.parse_args().root)
