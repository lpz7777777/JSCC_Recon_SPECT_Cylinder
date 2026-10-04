"""Bounded .75/.375 physical A samples in one frozen axial control group."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from generate_compton_a_guard import digest,write,axes,run_one,ENGINE_REL,SOURCE_RUN,NDET,PE_HASH,SCATTER_HASH
from generate_compton_a_column_v3 import comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','plan','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--group',type=int,choices=(0,1,2),required=True);p.add_argument('--cuda',type=int,default=0)
    a=p.parse_args();plan=json.loads(a.plan.read_text());cases=[c for c in plan['cases'] if c['group']==a.group]
    if plan['status']!='FROZEN_REGIONAL_GEOMETRY_DIAGNOSTIC_ONLY' or len(cases)!=4:
        raise ValueError('Frozen control budget differs')
    engine=a.root/ENGINE_REL;source=engine/SOURCE_RUN
    pe=engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter=engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe)!=PE_HASH or digest(scatter)!=SCATTER_HASH:raise ValueError('Physical models differ')
    original=json.loads((source/'ELLIPSE_inputs.json').read_text())
    if any(digest(source/n)!=v for n,v in original['parameters'].items()):raise ValueError('Physical parameters differ')
    required=sum(np.prod(s['shape'])*NDET*4*4 for c in cases for s in c['parts'])
    if shutil.disk_usage(a.output.parent).free<required*1.2+(10<<30):raise ValueError('Bounded regional disk budget insufficient')
    a.output.mkdir(exist_ok=False,parents=True);results=[]
    kinds=('pe.sysmat','pe_windowed.sysmat','Scatter_SysMat','SysMat_withScatter')
    for case in cases:
        receipts=[run_one(s,a.output,source,pe,scatter,a.cuda) for s in case['parts']]
        coarse,fine=receipts;checks={}
        for kind in kinds:
            matrices=[]
            for part in receipts:
                name=next(n for n in part['matrices'] if n.startswith(kind))
                matrices.append(np.memmap(a.output/part['spec']['name']/name,mode='r',dtype='<f4',
                    shape=(NDET,*reversed(part['spec']['shape']))))
            actual=np.array(matrices[1][:,::2,::2,::2],copy=True);expected=np.array(matrices[0],copy=True)
            if actual.shape!=expected.shape:raise ValueError('Nested physical grids are not aligned')
            check=comparison(actual,expected);checks[kind]=check
            if not check['passed'] or (kind.startswith('pe') and not check['bitwise_equal']):
                write(a.output/'common_point_failure.json',dict(case=case,checks=checks))
                raise ValueError('Independent fine-grid common-point regression failed')
        results.append(dict(case=case,receipts=receipts,common_points=checks))
        write(a.output/'progress.json',dict(completed_cases=len(results),total_cases=4,latest=case['name']))
    value=dict(status='REGIONAL_PHYSICAL_COMMON_POINTS_PASSED_ACCURACY_PENDING',group=a.group,
        plan_sha256=digest(a.plan),source_sha256=digest(Path(__file__)),results=results,
        pe_binary_sha256=PE_HASH,scatter_binary_sha256=SCATTER_HASH,original_params_sha256=original['parameters'],
        new_transport_photons=0,original_data_read_only=True,calibration_refitted=False,
        reconstruction_permitted=False,S2_generated=False,
        interpretation='Independent physical grid production/public-node consistency only. '
            'K-weighted full-reference/object-intersection accuracy and independent regional error gates remain required.')
    write(a.output/'regional_ready.json',value);print(value['status'],flush=True)


if __name__=='__main__':main()
