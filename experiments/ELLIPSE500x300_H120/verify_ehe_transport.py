"""Read-only full-transport identity audit; writes only small local evidence."""
import datetime, json, math, re, sys, textwrap
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ehe_5e9_workflow import DATA, REPORT, connection, query, command, base, release, frozen, q
from ehe_common import digest, read, write, verify_files
from run_ehe_worker import FILES, validate
from verify_ehe import source_macro_matches


def checked_remote_code(source):
    code = textwrap.dedent(source)
    compile(code, '<readonly-remote-sha-audit>', 'exec')
    return q(code)


def main():
    folder = DATA / 'transport'
    acceptance = read(REPORT / 'transport_acceptance.json')
    assert acceptance['passed']
    collection = read(folder / 'collection.json')
    verify_files(folder, collection['files'])
    registry = folder / 'source_registry'
    jobs_document = read(registry / 'jobs.json')
    jobs = jobs_document['jobs']
    pilot = read(REPORT / 'transport_pilot.json')
    registration = read(REPORT / 'transport_job.json')
    receipts = collection['receipts']
    assert len(jobs) == len(receipts) == 200
    assert collection['seeds'] == list(range(31100101, 31100301))
    remote_receipt_shas = {}
    for index, (job, receipt) in enumerate(zip(jobs, receipts)):
        worker = folder / f'worker_{index:03d}'
        original = registry / job['macro']
        assert receipt == read(worker / 'receipt.json')
        assert receipt['passed'] and not receipt['pilot']
        assert receipt['index'] == job['index'] == index
        assert receipt['view'] == job['view'] == index // 10 + 1
        assert receipt['worker'] == job['worker'] == index % 10
        assert receipt['seed'] == job['seed'] == 31100101 + index
        assert receipt['photons'] == job['photons'] == 25_000_000
        assert receipt['release_key'] == registration['release_key'] == frozen('maty')['release_key']
        assert receipt['binary_sha256'] == pilot['binary_sha256']
        assert receipt['release_manifest_sha256'] == pilot['release_manifest_sha256']
        assert digest(original) == job['macro_sha256'] == receipt['registered_macro_sha256']
        assert source_macro_matches(original, worker / 'source.mac')
        assert digest(worker / 'source.mac') == receipt['actual_macro_sha256']
        verify_files(worker, receipt['files'])
        summary, counts = validate(worker, 25_000_000)
        assert summary['primary_counts'] == receipt['primary_counts']
        assert counts == receipt['counts']
        audit = read(worker / 'EHE_MultiUnionAudit.json')
        assert audit['passed'] and audit['points'] == 11252 and audit['holes'] == 1250
        allocation = read(worker / 'allocation.json')
        assert allocation['denominator_kind'] == receipt['allocation_denominator_kind']
        remote_receipt_shas[worker.name] = digest(worker / 'receipt.json')

    with connection('maty') as c:
        accounting = query(c, registration['job'])
        rows = [line.split('|') for line in accounting.splitlines()
                if re.fullmatch(str(registration['job']) + r'_\d+', line.split('|')[0])
                and len(line.split('|')) >= 6]
        assert len(rows) == 200
        assert {int(row[0].split('_')[1]) for row in rows} == set(range(200))
        assert all(row[1:3] == ['COMPLETED', '0:0'] for row in rows)
        assert not any(len(line.split('|')) == 4 for line in accounting.splitlines())
        code = f'''
    import hashlib,json
    from pathlib import Path
    def sha(p):
     h=hashlib.sha256()
     with p.open('rb') as stream:
      for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
     return h.hexdigest()
    r=Path({release('maty')!r}); root=Path({(base('maty')+'/transport')!r})
    manifest=json.loads((r/'release_manifest.json').read_text())
    expected={frozen('maty')['sha256']!r}
    assert manifest['sha256']==expected
    for name,value in expected.items():assert sha(r/name)==value
    assert sha(r/'release_manifest.json')=={pilot['release_manifest_sha256']!r}
    assert sha(r/'build/ehe_spect')=={pilot['binary_sha256']!r}
    receipts={remote_receipt_shas!r}; count=0
    for name,value in receipts.items():
     p=root/name;assert sha(p/'receipt.json')==value
     receipt=json.loads((p/'receipt.json').read_text())
     for filename,checksum in receipt['files'].items():
      assert sha(p/filename)==checksum;count+=1
    print(json.dumps(dict(passed=True,workers=len(receipts),receipt_files_verified=count,release_files_verified=len(expected))))
    '''
        remote = json.loads(command(c, 'python3 -c ' + checked_remote_code(code), 120))
        assert remote['passed'] and remote['workers'] == 200

    with connection('gpu') as c:
        code = f'''
    import hashlib,json
    from pathlib import Path
    def sha(p):
     h=hashlib.sha256()
     with p.open('rb') as stream:
      for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
     return h.hexdigest()
    r=Path({(base('gpu')+'/counts')!r})
    assert sha(r/'collection.json')=={digest(folder/'collection.json')!r}
    collection=json.loads((r/'collection.json').read_text())
    for name,value in collection['files'].items():assert sha(r/name)==value
    print(json.dumps(dict(passed=True,files_verified=len(collection['files']),collection_sha256=sha(r/'collection.json'))))
    '''
        remote_sync = json.loads(command(c, q('/data/home/scxi717/.conda/envs/torch/bin/python')+' -c '+checked_remote_code(code), 300))
        assert remote_sync['passed'] and remote_sync['files_verified'] == len(collection['files'])

    arrays = np.load(folder / 'worker_counts.npz')
    primary = arrays['primary_counts'].sum(axis=0).astype(np.int64)
    assert int(primary.sum()) == collection['total_primary_photons'] == 5_000_000_000
    assert primary.tolist() == collection['primary_counts'] == acceptance['primary_counts']
    totals = {name[:-4]: int(arrays[name[:-4]].sum()) for name in FILES}
    per_view = {name[:-4]: arrays[name[:-4]].reshape(20, 10, 2312).sum(axis=(1, 2)).tolist() for name in FILES}
    for energy in (218, 440):
        direct = arrays[f'CntStat_{energy}']
        assert np.array_equal(direct, arrays[f'CntStat_{energy}_from218'] + arrays[f'CntStat_{energy}_from440'])
        assert np.array_equal(np.load(folder / f'projection_{energy}.npy'), direct.reshape(20, 10, 2312).sum(axis=1).T)
    expected = jobs_document['expected_primary_energy_fraction']['218']
    observed = float(primary[0] / primary.sum())
    standard_error = math.sqrt(expected * (1 - expected) / primary.sum())
    assert abs(observed - expected) <= 5 * standard_error
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    identity = dict(passed=True,scope='Completed EHE transport only; response physics and reconstruction not accepted',
        job=registration['job'],checked_utc=stamp,workers=200,views=20,photons_per_worker=25_000_000,
        actual_primary_photons=5_000_000_000,primary_counts=primary.tolist(),seed_first=31100101,seed_last=31100300,
        release_key=registration['release_key'],binary_sha256=pilot['binary_sha256'],
        release_manifest_sha256=pilot['release_manifest_sha256'],collection_sha256=digest(folder/'collection.json'),
        source_registry_sha256=digest(registry/'jobs.json'),fetched_files_verified=len(collection['files']),
        remote_after_exit_sha_audit=remote,remote_after_sync_sha_audit=remote_sync,all_workers_union_points=11252,
        source_macros='All 200 original and actual macros compared; only CRLF to LF conversion allowed',
        expected_218_primary_fraction=expected,actual_218_primary_fraction=observed,
        mixture_deviation_standard_errors=(observed-expected)/standard_error,
        cpu_memory_basis='Start MemAvailable operational guard; no Slurm memory TRES; not an imaging resource certificate',
        imaging_resource_certificate=False,audit_script_sha256=digest(__file__),accounting=accounting)
    write(REPORT/'transport_identity_acceptance.json',identity)
    measurement = dict(passed=True,scope='Actual primary-tagged EHE window measurements; no response calibration or image normalization',
        job=registration['job'],checked_utc=stamp,actual_primary_photons=5_000_000_000,
        primary_counts=primary.tolist(),window_and_tagged_counts=totals,window_and_tagged_counts_by_view=per_view,
        measured_440_to_218_fraction_of_218_window=totals['CntStat_218_from440']/totals['CntStat_218'],
        measured_218_to_440_count=totals['CntStat_440_from218'],
        source_registry_sha256=digest(registry/'jobs.json'),worker_counts_sha256=digest(folder/'worker_counts.npz'),
        collection_sha256=digest(folder/'collection.json'),transport_identity_sha256=digest(REPORT/'transport_identity_acceptance.json'),
        phase_seconds={phase:dict(minimum=min(r['phase_seconds'][phase] for r in receipts),
            median=float(np.median([r['phase_seconds'][phase] for r in receipts])),
            maximum=max(r['phase_seconds'][phase] for r in receipts)) for phase in ('initialization_seconds','beam_seconds')},
        physical_gate_status='Not evaluated: full responses still required',audit_script_sha256=digest(__file__))
    write(REPORT/'transport_measurement.json',measurement)
    print('FULL_TRANSPORT_IDENTITY_PASS',identity['fetched_files_verified'],remote)
    print(json.dumps(dict(primary_counts=primary.tolist(),counts=totals,cross_fraction=measurement['measured_440_to_218_fraction_of_218_window'],mixture_z=identity['mixture_deviation_standard_errors'])))


if __name__ == '__main__':
    main()
