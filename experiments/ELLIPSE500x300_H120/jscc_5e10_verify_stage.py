"""Independent frozen read-only acceptance; never writes the scientific result."""
import argparse, os, tarfile, time
from pathlib import Path
from jscc_5e10_common import *
from jscc_5e10_contract import verify_topology
from verify_jscc_5e10 import verify,verify_selection

def main():
    p=argparse.ArgumentParser()
    for n in ('release','input','factors','result','allocation','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('selection','validation','formal'),required=True)
    a=p.parse_args();began=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
    release=read(a.release/'verification_release.json');verify_files(a.release,release['sha256'])
    if a.mode=='selection':
        manifest=read(a.result/'selection_manifest.json')
        verify_files(a.result/'selections',manifest['selection_files'])
        verify_files(a.result/'selected_rows',manifest['selected_rows_files'])
        verify_topology(manifest['resources'],a.allocation)
        coll=validate_collection(read(a.input/'collection.json'));verify_files(a.input,coll['files'])
        if manifest['input_collection_sha256']!=digest(a.input/'collection.json') or manifest['actual_primary_photons']!=TOTAL:
            raise ValueError('Selection does not belong to this fresh transport')
        verify_selection(a.input,a.result,manifest)
        proof=dict(passed=True,study=STUDY,mode=a.mode,selection_manifest_sha256=digest(a.result/'selection_manifest.json'),
                   actual_primary_photons=TOTAL,all_selected_rows_exact=True,resources=manifest['resources'])
        members={'selection_manifest.json':digest(a.result/'selection_manifest.json')}
        members.update({'selections/'+n:s for n,s in manifest['selection_files'].items()})
        members.update({'selected_rows/'+n:s for n,s in manifest['selected_rows_files'].items()})
    else:
        proof=verify(a.result,a.release/'contract.json',a.allocation,a.mode,a.input,a.factors)
        members=hashes(a.result)
    proof.update(result_files_sha256=members,allocation_sha256=digest(a.allocation),
                 verification_release_sha256=digest(a.release/'verification_release.json'),
                 verification_sources_sha256=release['sha256'],elapsed_seconds=time.monotonic()-began,
                 cpu_read_only_verification_is_not_gpu_resource_certificate=True)
    write(a.output/'verification.json',proof)
    archive=a.output/'accepted_result.tar.gz'
    with tarfile.open(str(archive)+'.writing','w:gz',compresslevel=1) as t:
        for n in sorted(members):t.add(a.result/n,arcname=n,recursive=False)
        t.add(a.output/'verification.json',arcname='verification.json',recursive=False)
        t.add(a.allocation,arcname='allocation.txt',recursive=False)
    os.replace(str(archive)+'.writing',archive)
    verify_files(a.result,members)
    if digest(a.allocation)!=proof['allocation_sha256']:raise ValueError('Actual allocation changed during acceptance')
    files=dict(members,**{'verification.json':digest(a.output/'verification.json'),'allocation.txt':digest(a.allocation)})
    write(a.output/'package.json',dict(passed=True,study=STUDY,mode=a.mode,archive_sha256=digest(archive),
            archive_bytes=archive.stat().st_size,files=files,result=str(a.result),
            verification_sha256=files['verification.json'],verified_result_unchanged_after_packaging=True))
    print('JSCC5E10_READ_ONLY_ACCEPTANCE_COMPLETE',a.mode,flush=True)

if __name__=='__main__':main()
