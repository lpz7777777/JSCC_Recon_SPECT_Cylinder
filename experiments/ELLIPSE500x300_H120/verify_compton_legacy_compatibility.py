"""Compare the unchanged legacy path to a frozen historical kernel on real events."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from compton_event_response import (ComptonEventSettings,prepare_compton_events,
    build_detector_position_variance,build_compton_cone_weights,min_standardized_compton_arm)
from detector_csv import load_detector_coordinates

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('old-kernel','inputs','factors','geometry','config','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if hashlib.sha256(a.old_kernel.read_bytes()).hexdigest()!=cfg['baseline_kernel_sha256']:
        raise ValueError('Historical kernel SHA differs')
    spec=importlib.util.spec_from_file_location('frozen_legacy_kernel',a.old_kernel)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    device='cuda:0';torch.set_grad_enabled(False)
    factor=a.factors/'440keV_RotateNum20'
    d=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    v=build_detector_position_variance(d,0)
    s=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
    c=torch.tensor(np.load(a.geometry)['coordinates_mm'],dtype=torch.float32,device=device)
    checks=[]
    for view in (1,4,10,20):
        raw=np.loadtxt(a.inputs/'NEMA'/f'ideal_v{view:02d}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        selected=raw[np.linspace(0,len(raw)-1,32,dtype=int)]
        events,_=prepare_compton_events(torch.tensor(selected,device=device),s,d,v,v,input_energies_already_smeared=True)
        if events is None:raise ValueError('No eligible compatibility events')
        for name,new,previous in (
            ('K',build_compton_cone_weights(events,c,s),old.build_compton_cone_weights(events,c,s)),
            ('q',min_standardized_compton_arm(events,c,s),old.min_standardized_compton_arm(events,c,s))):
            if not torch.equal(new,previous):raise ValueError('Legacy differs: '+name)
            checks.append(dict(view=view,quantity=name,events=events.count,bitwise_equal=True))
    result=dict(status='PASSED',device=device,full_points=len(c),checks=checks,
        historical_kernel_sha256=cfg['baseline_kernel_sha256'],note='Response compatibility; does not replace 50-iteration image regression')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
