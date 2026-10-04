"""Bounded CPU continuation after one registered guard producer finishes."""
import argparse
import json
import os
from pathlib import Path
import time
from build_compton_a_guard_field import build
from prepare_compton_overlap_measure import prepare
from validate_compton_guard_field import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','guard','factors','geometry','config','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--producer-pid',type=int,required=True);p.add_argument('--wait-seconds',type=int,default=2400)
    a=p.parse_args();start=time.monotonic()
    while not (a.guard/'guard_ready.json').exists():
        gate=a.guard/'anchor/anchor_gate.json'
        if gate.exists() and json.loads(gate.read_text())['status']!='PASSED':raise ValueError('Anchor HOLD')
        if time.monotonic()-start>a.wait_seconds:raise TimeoutError('Guard producer deadline reached')
        try:os.kill(a.producer_pid,0)
        except ProcessLookupError:raise ValueError('Guard producer exited before publishing readiness')
        stat=Path(f'/proc/{a.producer_pid}/stat')
        if stat.exists() and stat.read_text().split(') ',1)[1].startswith('Z'):
            raise ValueError('Guard producer terminated before publishing readiness')
        time.sleep(10)
    print('GUARD_READY: building sampled A with exact original interior nodes',flush=True)
    build(a.root,a.guard,a.factors,a.geometry,a.config,a.output)
    measure=prepare(a.geometry,a.config,a.output/'precise_measure')
    result=validate(a.output,a.guard,a.output/'midpoint_gate.json')
    (a.output/'stage_complete.json').write_text(json.dumps(dict(
        status='DIAGNOSTICS_COMPLETE',midpoint_gate=result['status'],measure_gate=measure['status'],
        interpolation_gate_still_required=True,S2_generated=False,reconstruction_submitted=False),indent=2)+'\n')


if __name__=='__main__':main()
