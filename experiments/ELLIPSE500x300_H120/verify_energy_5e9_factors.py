"""Cross-host complete matrix/geometry SHA identity for calibration and imaging."""
import hashlib
import json
from pathlib import Path
import shlex
from first_scatter_pipeline import ssh,SERVER,SERVER_ROOT,SERVER_BASE
from energy_preflight_v5_workflow import connect,command,BASE,PYTHON
from prepare_energy_5e9_v5 import STUDY


PROGRAM='''import hashlib,json,pathlib,sys
base=pathlib.Path(sys.argv[1]);files={}
for folder in ('218keV_RotateNum20','440keV_RotateNum20','440keV_to218win_RotateNum20'):
 for name in ('SysMat_polar','Detector.csv','coor_polar_full.csv','RotMat_full.csv','RotMatInv_full.csv',
              'polar_cell_volume_mm3.float64','ellipse_fraction.float64','ellipse_active_indices.int32','factor_manifest.json'):
  path=base/folder/name;h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  files[folder+'/'+name]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
  print('FACTOR_HASH_CHECKED',folder+'/'+name,flush=True)
print(json.dumps(files))
'''


def main():
    from concurrent.futures import ThreadPoolExecutor
    def calibration():return json.loads(ssh(SERVER,'timeout --signal=TERM --kill-after=10s 600s '+SERVER_ROOT+'/.venv/bin/python -c '+shlex.quote(PROGRAM)+' '+shlex.quote(SERVER_BASE+'/generated/FactorsCalibrated')).splitlines()[-1])
    def imaging():
        with connect() as c:
            _,stdout,stderr=c.exec_command('timeout --signal=TERM --kill-after=10s 600s '+PYTHON+' -c '+shlex.quote(PROGRAM)+' '+shlex.quote(BASE+'/generated/FactorsCalibrated'),timeout=660)
            out=stdout.read().decode();err=stderr.read().decode()
            if stdout.channel.recv_exit_status():raise RuntimeError(err or out)
            return json.loads(out.splitlines()[-1])
    with ThreadPoolExecutor(2) as pool:
        ca=pool.submit(calibration);im=pool.submit(imaging)
        first,second=ca.result(),im.result()
    if first!=second:raise ValueError('Calibration and imaging matrix/geometry SHA differ')
    p=Path(__file__).parent/'reports/NEMA_Body_H60'/STUDY/'factor_identity.json'
    p.write_text(json.dumps(dict(passed=True,files=first,calibration_host=SERVER,imaging_host='scxi717'),indent=2)+'\n')
    print('COMPLETE_CROSS_HOST_FACTOR_IDENTITY_PASSED',len(first))


if __name__=='__main__':main()
