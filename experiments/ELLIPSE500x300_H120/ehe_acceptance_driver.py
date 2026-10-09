"""Run unchanged read-only acceptance with persistent Slurm phase/exit evidence."""
import argparse
from pathlib import Path
import resource
import time
import verify_ehe as verifier
from ehe_common import allocation, digest, read, verify_files, write


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('result', 'release', 'responses', 'counts', 'physical', 'accounting',
                 'output', 'verification-manifest', 'evidence'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--physical-policy', type=Path)
    a = p.parse_args()
    started = time.monotonic()
    a.evidence.mkdir(parents=True, exist_ok=False)
    alloc = allocation()
    write(a.evidence/'allocation.json', alloc)
    original_verify = verifier.verify_files
    def traced(root, files):
        print('READ_ONLY_SHA_BEGIN', root, len(files), flush=True)
        original_verify(root, files)
        print('READ_ONLY_SHA_PASS', root, flush=True)
    verifier.verify_files = traced
    original_closure = verifier.operator_closure
    def closure(*args):
        print('ALL_VIEW_CPU_OPERATOR_BEGIN', flush=True)
        result = original_closure(*args)
        print('ALL_VIEW_CPU_OPERATOR_PASS', result['elapsed_seconds'], flush=True)
        return result
    verifier.operator_closure = closure
    try:
        verify_files(Path(__file__).parent, read(a.verification_manifest)['files'])
        proof = verifier.verify(a.result, a.release, a.responses, a.counts,
            a.physical, a.accounting.read_text(), a.physical_policy)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if rss > .8*alloc['host_allocated_bytes']:
            raise MemoryError('Read-only acceptance allocation reserve failed')
        proof['verification_manifest_sha256'] = digest(a.verification_manifest)
        proof['acceptance_driver_sha256'] = digest(__file__)
        write(a.output, proof)
        write(a.evidence/'receipt.json', dict(passed=True, elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=rss, host_allocated_bytes=alloc['host_allocated_bytes'],
            rss_fraction=rss/alloc['host_allocated_bytes'], allocation_job=alloc['job'],
            authority_sha256=digest(a.output), driver_sha256=digest(__file__),
            imaging_resource_certificate=False, gpu_calculation=False))
        print('EHE_VERIFIED', proof['mode'], flush=True)
    except BaseException as e:
        write(a.evidence/'failure.json', dict(passed=False, error=str(e), elapsed_seconds=time.monotonic()-started))
        raise
