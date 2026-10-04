"""Held-out point-source consistency at the true position; never selects events."""
import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
from compton_event_response import (ComptonEventSettings,PreparedComptonEvents,
    prepare_compton_events,build_detector_position_variance,min_standardized_compton_arm)
import compton_event_response as kernel
from detector_csv import load_detector_coordinates

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(inputs,analysis,factors,output):
    gate=json.loads((analysis/'validation_gate.json').read_text())
    if digest(kernel.__file__)!=gate['kernel_sha256']:raise ValueError('Diagnostic kernel differs')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    detector=torch.tensor(load_detector_coordinates(factors/'440keV_RotateNum20/Detector.csv',10496))
    variance=build_detector_position_variance(detector,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
    rows=[];summaries=[]
    for folder in sorted(inputs.glob('point_*')):
        for group in ('legacy','ideal'):
            part=[]
            for view in range(1,21):
                path=folder/f'{group}_v{view:02d}.csv'
                raw=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
                prepared,_=prepare_compton_events(torch.tensor(raw),settings,detector,variance,variance,
                    input_energies_already_smeared=True)
                kept=np.load(analysis/f'{folder.name}_{group}_v{view:02d}_kept_rows.npy')
                if not len(kept):continue
                if prepared is None:raise ValueError('Selected point events no longer prepare')
                mask=torch.tensor(np.isin(prepared.source_row_indices.numpy(),kept))
                p=PreparedComptonEvents(**{f.name:None if getattr(prepared,f.name) is None else
                    getattr(prepared,f.name)[mask] for f in fields(prepared)})
                wanted=set(map(int,kept));metadata={}
                with (folder/f'events_v{view:02d}.csv').open() as stream:
                    for r in csv.DictReader(stream):
                        row=int(r[f'global_{group}_row'])
                        if row in wanted:metadata[row]=r
                if p.count!=len(kept) or len(metadata)!=len(kept):raise ValueError('Point identities differ')
                source=np.array([[float(r['source_'+k]) for k in 'xyz'] for r in metadata.values()])
                if np.max(abs(source-source[0]))>1e-6:raise ValueError('Point file has multiple locations')
                source=source[0]+np.array([0.,345.,0.])
                q=min_standardized_compton_arm(p,torch.tensor(source[None],dtype=torch.float32),settings).numpy()
                v1=p.pos1.numpy()-source;v2=(p.pos2-p.pos1).numpy()
                cosine=np.clip(np.sum(v1*v2,axis=1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)),-1,1)
                prediction=.440-.440/(1+(.440/.511)*(1-cosine))
                sigma=.13/2.35482*np.sqrt(.511*np.maximum(prediction,1e-12))
                pulls=(p.e1.numpy()-prediction)/sigma
                file_sha=digest(path)
                for index,row in enumerate(p.source_row_indices.tolist()):
                    r=metadata[row]
                    item=dict(dataset=folder.name,group=group,view=view,seed=int(r['seed']),
                        worker=int(r['worker']),event_id=int(r['event_id']),input_row=row,input_sha256=file_sha,
                        standardized_arm_at_true_source=float(q[index]),center_energy_pull=float(pulls[index]),
                        physical_first_pair_matches_list=(int(r['c1'])==int(raw[row,0]) and int(r['c2'])==int(raw[row,2])))
                    part.append(item);rows.append(item)
            if len(part)!=gate['scans'][folder.name+'_'+group]['kept']:raise ValueError('Point event closure differs')
            values=np.array([r['standardized_arm_at_true_source'] for r in part])
            summaries.append(dict(dataset=folder.name,group=group,events=len(part),
                true_source_arm_coverage={str(k):float(np.mean(values<=k)) for k in (1,2,3)},
                true_source_arm_p95=float(np.quantile(values,.95)),
                first_pair_mismatch_fraction=float(np.mean([not r['physical_first_pair_matches_list'] for r in part]))))
    with (output/'point_consistency.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(study='compton_first_scatter_v2',diagnostic_only=True,true_positions_used_for_selection=False,
        input_manifest_sha256=digest(inputs/'input_manifest.json'),kernel_sha256=gate['kernel_sha256'],
        validation_gate_sha256=digest(analysis/'validation_gate.json'),summaries=summaries)
    (output/'point_consistency.json').write_text(json.dumps(result,indent=2)+'\n')
    print('POINT_CONSISTENCY_DIAGNOSTIC_OK',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','analysis','factors','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with torch.no_grad():run(a.inputs,a.analysis,a.factors,a.output)
