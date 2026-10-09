"""Strict mixed-allocation acceptance for launcher-only recovery of interrupted workers.
The original worker executable/wrapper are reused without modification.
"""
import argparse, hashlib, json, math, os, re, shutil, tarfile, time
from pathlib import Path
import numpy as np
from ehe_common import digest, read, write, verify_files, hashes, allocation, bounded_process

STUDY = 'ehe_spect_5e10_200'
TOTAL = 50_000_000_000
WORKERS = 1000
PER_VIEW = 50
PER_WORKER = 50_000_000
SEED_BASE = 33100101
FILES = [f'CntStat_{e}.csv' for e in (218, 440)] + [
    f'CntStat_{e}_from{p}.csv' for e in (218, 440) for p in (218, 440)]
GEOMETRY_FILES = ['EHE_CollimatorHoles.csv', 'EHE_DetectorGeometry.csv',
                  'EHE_GeometrySummary.txt']


def save_npy(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('wb') as stream:
        np.save(stream, array)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def registry_identity(simulation, config):
    registry = read(simulation / 'jobs.json')
    if digest(simulation / 'jobs.json') != config['simulation_manifest_sha256']:
        raise ValueError('New source registry identity differs')
    jobs = registry['jobs']
    if len(jobs) != WORKERS or registry['total_primary_photons'] != TOTAL:
        raise ValueError('Exact new worker/dose partition required')
    if registry['workers_per_view'] != PER_VIEW:
        raise ValueError('Every view requires 50 independent workers')
    for i, job in enumerate(jobs):
        if (job['index'], job['view'], job['worker'], job['seed'], job['photons']) != (
                i, i // PER_VIEW + 1, i % PER_VIEW, SEED_BASE + i, PER_WORKER):
            raise ValueError('New index/view/seed/dose registry differs')
    return registry


def worker(release, simulation, output, index, limit):
    freeze = read(release / 'release_manifest.json')
    verify_files(release, freeze['sha256'])
    config = read(release / 'config.json')
    registry = registry_identity(simulation, config)
    job = registry['jobs'][index]
    binary = Path(config['binary_path'])
    if digest(binary) != config['binary_sha256']:
        raise ValueError('Accepted Geant4 executable bytes changed')
    macro = simulation / job['macro']
    if digest(macro) != job['macro_sha256']:
        raise ValueError('Registered macro changed')
    if int(os.environ['SLURM_PROCID']) != index:
        raise ValueError('Scheduler task and independent worker index differ')
    output.mkdir(parents=True, exist_ok=False)
    # Preserve registered Windows bytes separately; only newline conversion here.
    (output / 'source.mac').write_bytes(macro.read_bytes().replace(b'\r\n', b'\n'))
    alloc = allocation(unlimited_transport=True)
    # cnmix explicitly has UNLIMITED memory and placeholder partition TRES.
    # Preserve actual scontrol; this is a physical runtime guard, not a certificate.
    meminfo = Path('/proc/meminfo').read_text()
    alloc.update(host_allocated_bytes=int(re.search(r'MemAvailable:\s+(\d+)',meminfo)[1])*1024,
        denominator_kind='CPU transport physical MemAvailable at start; cnmix UNLIMITED',
        imaging_allocation_certificate=False, proc_meminfo_at_start=meminfo)
    write(output / 'allocation.json', alloc)
    env = os.environ.copy()
    env['EHE_RANDOM_SEED'] = str(job['seed'])
    began = time.time()
    try:
        usage = bounded_process([str(binary), str(output / 'source.mac')], output,
                                limit, output / 'transport.log', alloc, env)
        summary = read(output / 'TransportSummary.json')
        if (summary['primary_events'] != PER_WORKER or
            sum(summary['primary_counts']) != PER_WORKER or
            summary['primary_counts'][2] != 0 or summary['detector_bins'] != 2312):
            raise ValueError('Actual primary events do not close')
        vectors = {}
        for name in FILES:
            value = np.loadtxt(output / name, delimiter=',', dtype=np.int64, ndmin=2)
            if value.shape != (1, 2312) or np.any(value < 0):
                raise ValueError('Complete nonnegative 2312-bin observation required')
            vectors[name] = value[0]
        for e in (218, 440):
            if not np.array_equal(vectors[f'CntStat_{e}.csv'],
                    vectors[f'CntStat_{e}_from218.csv'] + vectors[f'CntStat_{e}_from440.csv']):
                raise ValueError('Primary-tagged window closure failed')
        audit = read(output / 'EHE_MultiUnionAudit.json')
        if not audit['passed'] or audit['points'] != 11252 or audit['holes'] != 1250:
            raise ValueError('Actual geometry classification failed')
        verify_files(output, config['geometry_evidence_sha256'])
        members = FILES + GEOMETRY_FILES + ['TransportSummary.json',
                    'TransportTiming.json', 'EHE_MultiUnionAudit.json', 'source.mac',
                    'allocation.json', 'transport.log']
        receipt = dict(passed=True, study=STUDY, index=index, view=job['view'],
            worker=job['worker'], seed=job['seed'], photons=PER_WORKER,
            release_key=freeze['release_key'], binary_sha256=digest(binary),
            primary_counts=summary['primary_counts'],
            counts={k: int(v.sum()) for k, v in vectors.items()},
            resource=usage, phase_seconds=read(output / 'TransportTiming.json'),
            registered_macro_sha256=digest(macro),
            actual_macro_sha256=digest(output / 'source.mac'),
            allocation_job=alloc['job'], started_epoch=began, finished_epoch=time.time(),
            files={name: digest(output / name) for name in members})
        write(output / 'receipt.json', receipt)
        print('EHE_5E10_WORKER_COMPLETE', index, flush=True)
    except BaseException as exc:
        write(output / 'failure.json', dict(passed=False, index=index, error=str(exc)))
        raise


def expected_worker_job(config, collection, index, folder):
    recovery = config['transport_recovery']
    reused = recovery['reused_workers']
    indices = [int(i) for i in reused]
    if (len(indices) != recovery['reused_count'] or len(set(indices)) != len(indices)
        or any(i < 0 or i >= WORKERS for i in indices)
        or recovery['missing_count'] + len(indices) != WORKERS
        or collection['job'] == recovery['original_failed_job']
        or recovery['original_state'] != 'FAILED'):
        raise ValueError('Recovery must preserve failed source and disjoint missing partition')
    if str(index) in reused:
        original = reused[str(index)]
        if (digest(folder / 'receipt.json') != original['receipt_sha256']
            or original['allocation_job'] != str(recovery['original_failed_job'])
            or read(folder / 'receipt.json')['files'] != original['files']):
            raise ValueError('Completed original worker receipt identity changed')
        return original['allocation_job']
    return str(collection['job'])


def verify_counts(folder, release):
    config = read(release / 'config.json')
    collection = read(folder / 'collection.json')
    if (not collection['passed'] or collection['study'] != STUDY or
        collection['data_kind'] != 'Geant4_transport' or
        collection['total_primary_photons'] != TOTAL or
        collection['workers'] != WORKERS or collection['views'] != 20):
        raise ValueError('Actual independent 5e10 acquisition required')
    if collection.get('recovery') != config.get('transport_recovery'):
        raise ValueError('Frozen recovery partition differs')
    verify_files(folder, collection['files'])
    registry = registry_identity(folder / 'source_registry', config)
    verify_files(folder / 'source_registry', config['source_registry_sha256'])
    arrays = {name[:-4]: np.zeros((WORKERS, 2312), np.int64) for name in FILES}
    primary = np.zeros((WORKERS, 3), np.int64)
    for i, job in enumerate(registry['jobs']):
        sub = folder / f'worker_{i:04d}'
        receipt = read(sub / 'receipt.json')
        if any(receipt[k] != job[k] for k in ('index', 'view', 'worker', 'seed', 'photons')):
            raise ValueError('Actual receipt differs from new registry')
        if (not receipt['passed'] or receipt['study'] != STUDY or
            receipt['release_key'] != config['transport_release_key'] or
            receipt['binary_sha256'] != config['binary_sha256'] or
            receipt['allocation_job'] != expected_worker_job(config, collection, i, sub)):
            raise ValueError('Actual CPU execution identity differs')
        verify_files(sub, receipt['files'])
        verify_files(sub, config['geometry_evidence_sha256'])
        if (sub / 'source.mac').read_bytes() != (folder / 'source_registry' / job['macro']).read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Source commands changed beyond CRLF to LF')
        if receipt['registered_macro_sha256'] != job['macro_sha256']:
            raise ValueError('Registered macro SHA differs')
        a = read(sub / 'EHE_MultiUnionAudit.json')
        if not a['passed'] or (a['points'], a['holes']) != (11252, 1250):
            raise ValueError('Actual geometry audit differs')
        s = read(sub / 'TransportSummary.json')
        if s['primary_events'] != PER_WORKER or s['primary_counts'] != receipt['primary_counts']:
            raise ValueError('Actual primary summary differs')
        primary[i] = receipt['primary_counts']
        if primary[i].sum() != PER_WORKER or primary[i, 2] != 0:
            raise ValueError('Actual dose/primary labels do not close')
        for name in FILES:
            v = np.loadtxt(sub / name, delimiter=',', dtype=np.int64, ndmin=2)
            if v.shape != (1, 2312) or np.any(v < 0) or int(v.sum()) != receipt['counts'][name]:
                raise ValueError('All native bin observations must match receipts')
            arrays[name[:-4]][i] = v[0]
        alloc=read(sub/'allocation.json')
        available=int(re.search(r'MemAvailable:\s+(\d+)',alloc['proc_meminfo_at_start'])[1])*1024
        usage=receipt['resource']
        if (alloc['imaging_allocation_certificate'] or alloc['host_allocated_bytes']!=available or
            usage['host_allocated_bytes']!=available or not 0<usage['rss_peak_bytes']<=.8*available or
            not math.isclose(usage['rss_fraction'],usage['rss_peak_bytes']/available,rel_tol=1e-12)):
            raise ValueError('CPU operational memory guard failed')
    for e in (218, 440):
        if not np.array_equal(arrays[f'CntStat_{e}'],
                arrays[f'CntStat_{e}_from218'] + arrays[f'CntStat_{e}_from440']):
            raise ValueError('Complete primary/window identity differs')
        expected = arrays[f'CntStat_{e}'].reshape(20, PER_VIEW, 2312).sum(axis=1).T
        if not np.array_equal(np.load(folder / f'projection_{e}.npy'), expected):
            raise ValueError('Projection is not this acquisition all-worker sum')
    if primary.sum() != TOTAL or collection['primary_counts'] != primary.sum(axis=0).tolist():
        raise ValueError('Complete actual 5e10 dose differs')
    expected_fraction = registry['expected_primary_energy_fraction']['218']
    observed = primary[:, 0].sum() / TOTAL
    if abs(observed - expected_fraction) > 5 * math.sqrt(expected_fraction * (1-expected_fraction) / TOTAL):
        raise ValueError('Actual source mixture exceeds original 5SE identity guard')
    saved = np.load(folder / 'worker_counts.npz')
    for name, values in dict(arrays, primary_counts=primary).items():
        if not np.array_equal(saved[name], values):
            raise ValueError('All independent worker diagnostic arrays differ')
    return collection


def collect(release, simulation, transport, output, job, accounting):
    from ehe_slurm_status import stage_completed
    if not stage_completed(accounting, job):
        raise ValueError('Complete CPU allocation and srun step must exit successfully')
    freeze = read(release / 'release_manifest.json')
    config = read(release / 'config.json')
    registry = registry_identity(simulation, config)
    verify_files(release, freeze['sha256'])
    verify_files(Path(config['original_transport_root']),
                 config['original_transport_source_sha256'])
    if digest(config['binary_path']) != config['binary_sha256']:
        raise ValueError('Accepted binary changed after transport')
    if output.exists():
        return verify_counts(output, release)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(simulation, output / 'source_registry')
    arrays = {name[:-4]: np.zeros((WORKERS, 2312), np.int64) for name in FILES}
    primary = np.zeros((WORKERS, 3), np.int64)
    for i in range(WORKERS):
        reuse = config.get('transport_recovery', {}).get('reused_workers', {})
        source_root = Path(config['transport_recovery']['original_transport_root']) if str(i) in reuse else transport
        source = source_root / f'worker_{i:04d}'
        receipt = read(source / 'receipt.json')
        verify_files(source, receipt['files'])
        dest = output / source.name
        dest.mkdir()
        for name in list(receipt['files']) + ['receipt.json']:
            shutil.copy2(source / name, dest / name)
        for name in FILES:
            arrays[name[:-4]][i] = np.loadtxt(dest / name, delimiter=',', dtype=np.int64)
        primary[i] = receipt['primary_counts']
    np.savez_compressed(output / 'worker_counts.npz', **arrays, primary_counts=primary)
    for e in (218, 440):
        save_npy(output / f'projection_{e}.npy',
                 arrays[f'CntStat_{e}'].reshape(20, PER_VIEW, 2312).sum(axis=1).T)
    write(output / 'transport_accounting.json', dict(job=job, accounting=accounting,
          passed=True, cpu_operational_guard_only=True, imaging_resource_certificate=False))
    proof = dict(passed=True, study=STUDY, job=job, data_kind='Geant4_transport',
        total_primary_photons=int(primary.sum()), primary_counts=primary.sum(axis=0).tolist(),
        workers=WORKERS, views=20, seed_first=SEED_BASE, seed_last=SEED_BASE+WORKERS-1,
        window_counts={str(e): int(arrays[f'CntStat_{e}'].sum()) for e in (218,440)},
        tagged_counts={name: int(a.sum()) for name,a in arrays.items()},
        physical_calibration_claim=False, transport_performed=True,
        recovery=config['transport_recovery'], files=hashes(output))
    write(output / 'collection.json', proof)
    verify_counts(output, release)
    return proof


def archive(folder, destination):
    proof = read(folder / 'collection.json')
    with tarfile.open(destination.with_suffix('.writing'), 'w:gz') as tar:
        for name in sorted(proof['files']):
            tar.add(folder / name, arcname=name, recursive=False)
        tar.add(folder / 'collection.json', arcname='collection.json', recursive=False)
    os.replace(destination.with_suffix('.writing'), destination)
    write(destination.with_suffix('.json'), dict(passed=True,
          archive_sha256=digest(destination), collection_sha256=digest(folder/'collection.json'),
          files=len(proof['files'])+1, bytes=destination.stat().st_size))


def extract(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or '\\' in member.name:
                raise ValueError('Archive member is not an approved relative regular file')
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Archive member escapes new experiment')
        for member in tar.getmembers():
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['worker', 'collect'])
    for key in ('release', 'simulation', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_PROCID', '0')))
    p.add_argument('--limit', type=int)
    p.add_argument('--transport', type=Path)
    p.add_argument('--job', type=int)
    p.add_argument('--accounting', type=Path)
    a = p.parse_args()
    if a.action == 'worker':
        worker(a.release, a.simulation, a.output / f'worker_{a.index:04d}', a.index, a.limit)
    else:
        collect(a.release, a.simulation, a.transport, a.output, a.job, a.accounting.read_text())
        archive(a.output, a.output.parent / 'transport_counts.tar.gz')
