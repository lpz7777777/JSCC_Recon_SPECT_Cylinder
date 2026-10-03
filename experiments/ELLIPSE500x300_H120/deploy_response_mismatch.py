"""Freeze and deploy only response-cut code and independently validated Sensi_d."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/FOV120'))
from reconstruction_ssh import connect

REMOTE_ROOT='/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor'
REMOTE=REMOTE_ROOT+'/experiments/ELLIPSE500x300_H120'
REPORT=HERE/'reports/NEMA_Body_H60/response_mismatch_cut3_v1'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(client,text):
    _,out,err=client.exec_command(text,timeout=120)
    data=out.read().decode(errors='replace')
    if out.channel.recv_exit_status(): raise RuntimeError(err.read().decode(errors='replace') or data)
    return data.strip()


def main():
    names=['run_reconstruction.py','torch_active_operator.py','validate_factors.py','geometry.py','config.json',
           'response_mismatch_cut3_v1.json','verify_response_mismatch.py','reconstruct_response_mismatch.sh']
    paths={name:HERE/name for name in names}
    paths.update({name:ROOT/name for name in ('compton_event_response.py','process_list_plane_sparse.py',
                                             'compton_sparse_ops.py','detector_csv.py')})
    hashes={name:digest(path) for name,path in paths.items()}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    release=REMOTE+'/code_releases/response_mismatch_cut3_v1_'+key
    scan=HERE/'generated/response_mismatch_cut3_v1/scan'
    payload={name:scan/name for name in ('Sensi_d','Sensi_d_provenance.json','scan_manifest.json','progress.json','rejected_events.csv')}
    config=json.loads(paths['response_mismatch_cut3_v1.json'].read_text())
    if digest(payload['Sensi_d'])!=config['sensi_d_sha256'] or digest(payload['scan_manifest.json'])!=config['scan_manifest_sha256']:
        raise ValueError('Validated scan payload differs')
    with connect() as client:
        command(client,'mkdir -p -- '+shlex.quote(release)+' '+shlex.quote(REMOTE+'/generated/response_mismatch_cut3_v1/scan'))
        with client.open_sftp() as sftp:
            for name,path in paths.items():
                dst=release+'/'+name
                sftp.put(str(path),dst)
                if command(client,'sha256sum -- '+shlex.quote(dst)).split()[0]!=hashes[name]:
                    raise ValueError('Code upload hash differs')
            for name,path in payload.items():
                dst=REMOTE+'/generated/response_mismatch_cut3_v1/scan/'+name
                sftp.put(str(path),dst)
                if command(client,'sha256sum -- '+shlex.quote(dst)).split()[0]!=digest(path):
                    raise ValueError('Independent sensitivity upload hash differs')
        command(client,'bash -n '+shlex.quote(release+'/reconstruct_response_mismatch.sh'))
        record={'study':config['study'],'release':release,'code_sha256':hashes,
                'scan_payload_sha256':{name:digest(path) for name,path in payload.items()},
                'geometry_sha256':config['geometry_sha256'],'submitted':False}
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'deployment.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))


if __name__=='__main__': main()
