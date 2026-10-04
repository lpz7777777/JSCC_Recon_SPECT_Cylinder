"""Independent radial midpoint samples after the registered halo releases GPU.

No device is held while waiting. Occupied devices are left untouched; an
explicit bounded deadline prevents an indefinite idle continuation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from generate_compton_a_guard import run_one,digest,ENGINE_REL,SOURCE_RUN,PE_HASH,SCATTER_HASH
from validate_compton_guard_field import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','field','guard','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--producer-pid',type=int,required=True)
    a=p.parse_args();start=time.monotonic()
    while True:
        if time.monotonic()-start>3000:raise TimeoutError('Radial validation dependency/device deadline')
        field_ready=(a.field/'field_manifest.json').exists()
        guard_ready=(a.guard/'guard_ready.json').exists()
        if field_ready and guard_ready:
            free,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                '--format=csv,noheader,nounits','-i',str(a.gpu)],text=True).strip().split(','))
            if free>=40000 and util<=5:break
        elif not guard_ready:
            try:os.kill(a.producer_pid,0)
            except ProcessLookupError:raise ValueError('Halo producer exited without readiness')
        time.sleep(10)
    engine=a.root/ENGINE_REL;pe=engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter=engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe)!=PE_HASH or digest(scatter)!=SCATTER_HASH:raise ValueError('Production binaries differ')
    a.output.mkdir(parents=True,exist_ok=False)
    spec=dict(name='radial_mid',shape=(3,3,2),spacing=(255,255,3),shift=(0,0,0))
    # CUDA_VISIBLE_DEVICES is intentionally unset here: --cuda uses physical
    # identity after the free-device check, and no GPU is allocated while waiting.
    os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    run_one(spec,a.output,engine/SOURCE_RUN,pe,scatter,a.gpu)
    validate(a.field,a.output,a.output/'radial_midpoint_gate.json',parts=('radial_mid',))


if __name__=='__main__':main()
