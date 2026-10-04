"""Preserve exact historical execution sources, without copying large inputs."""
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'experiments/FOV120')]
from compton_geometry_v3_workflow import DATA,REPORT,digest,write,SERVER
from reconstruction_ssh import connect
from first_scatter_imaging import command


def main():
    records=('guard_integral','guard_patch','guard_patch_refined','guard_patch_ultrafine','R1_preflight')
    rows=[]
    for tag in records:
        record=json.loads((REPORT/(tag+'_deployment.json')).read_text());release=record['release']
        for name,expected in record['code_sha256'].items():
            candidates=[HERE/name,ROOT/name,DATA/'R1_preflight'/name]
            if name=='geometry.npz':candidates.insert(0,HERE/'generated/Geometry/geometry.npz')
            source=next((p for p in candidates if p.is_file() and digest(p)==expected),None)
            if source is None:
                # Only execution source is copied to Git. Geometry/S/input
                # receipts must already match an immutable generated input.
                if not name.endswith('.py'):raise ValueError('Unresolved immutable input '+name)
                target=REPORT/'frozen_execution'/release.rsplit('/',1)[-1]/name
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    if tag=='R1_preflight':
                        with connect() as client:
                            with client.open_sftp() as sftp:sftp.get(release+'/'+name,str(target))
                    else:
                        subprocess.run(['scp','-o','BatchMode=yes',SERVER+':'+release+'/'+name,str(target)],check=True)
                if digest(target)!=expected:raise ValueError('Historical source differs '+name)
                source=target
            rows.append(dict(deployment=tag,name=name,sha256=expected,
                             local_path=source.relative_to(ROOT).as_posix()))
    write(REPORT/'integral_execution_source_audit.json',dict(status='PASSED',entries=len(rows),sources=rows,
        note='Current or exact frozen historical source bytes; large immutable inputs remain generated and are excluded from Git'))
    print('INTEGRAL_EXECUTION_SOURCES_PASSED',len(rows))


if __name__=='__main__':main()
