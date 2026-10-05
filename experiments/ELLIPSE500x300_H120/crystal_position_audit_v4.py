"""Compare recorded physical hit offsets with the kernel's uniform crystal model."""
import argparse
from pathlib import Path
import json
import numpy as np
from process_list_global_audit_v4 import metadata,vector,digest,write,table
from detector_csv import load_detector_coordinates
from compton_event_response import FRONT_LAYER_CRYSTAL_MM,REAR_LAYER_CRYSTAL_MM

def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    detector=load_detector_coordinates(a.detector,10496)
    layer=np.rint((abs(detector[:,1])-300)/30).astype(int)
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())['files']
    with a.points.open() as stream:
        import csv
        prepared={(r['dataset'],int(r['view']),int(r['input_row'])):r['reco_energy_pass']=='True' for r in csv.DictReader(stream)}
    groups={};outside=0
    for folder in sorted(a.inputs.glob('point_*')):
        for view in range(1,21):
            assert digest(folder/f'events_v{view:02d}.csv')==manifest[folder.name+f'/events_v{view:02d}.csv']
            kept=set(map(int,np.load(a.analysis/f'{folder.name}_ideal_v{view:02d}_kept_rows.npy')))
            for r in metadata(folder,view):
                row=int(r['global_ideal_row'])
                if row<0:continue
                for hit in (1,2):
                    index=int(r['c'+str(hit)])-1;position=vector(r,'p'+str(hit)+'_')+np.array([0,345,0])
                    offset=position-detector[index];size=np.array(FRONT_LAYER_CRYSTAL_MM if layer[index]<3 else REAR_LAYER_CRYSTAL_MM)
                    if np.any(abs(offset)>size/2+1e-5):outside+=1
                    for cohort,selected in [('before_reco',True),('energy_selected',prepared[folder.name,view,row]),('stable_q3',row in kept)]:
                        if selected:groups.setdefault((folder.name,cohort,hit,int(layer[index])),[]).append(offset)
    rows=[]
    for key,offsets in sorted(groups.items()):
        offsets=np.array(offsets);size=np.array(FRONT_LAYER_CRYSTAL_MM if key[3]<3 else REAR_LAYER_CRYSTAL_MM)
        for axis,label in enumerate('xyz'):
            rows.append(dict(dataset=key[0],cohort=key[1],hit=key[2],layer=key[3],axis=label,events=len(offsets),
                mean_offset_mm=float(offsets[:,axis].mean()),standard_deviation_mm=float(offsets[:,axis].std(ddof=1)),
                assumed_uniform_sd_mm=float(size[axis]/np.sqrt(12)),
                variance_ratio_to_uniform=float(offsets[:,axis].var(ddof=1)/(size[axis]**2/12))))
    table(a.output/'crystal_offset_moments.csv',rows)
    write(a.output/'summary.json',dict(source_sha256=digest(__file__),point_residual_sha256=digest(a.points),
        input_manifest_sha256=digest(a.inputs/'input_manifest.json'),
        hits_outside_declared_crystal=outside,crystal_sizes_mm=dict(front=FRONT_LAYER_CRYSTAL_MM,rear=REAR_LAYER_CRYSTAL_MM),
        comparison='Actual interaction offsets for recorded ideal events, not an unconditional uniform-in-crystal sample',
        new_transport_photons=0,new_reconstruction=False))
    print('CRYSTAL_OFFSET_AUDIT_COMPLETE',len(rows),outside,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','analysis','points','detector','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
