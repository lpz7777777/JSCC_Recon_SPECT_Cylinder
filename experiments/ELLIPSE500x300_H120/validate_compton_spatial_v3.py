"""Held-out boundary/interior/outer-support gates, with worker uncertainty.

Direct efficiencies use the known uniform source measure and actual total
primary count. Their denominators are expected bin emissions, not falsely
labelled as recorded per-bin emissions. Training rotations remain correlated
within each worker estimate.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from geometry import grid
from analyze_first_scatter import digest,subset
from compton_event_response import ComptonEventSettings,prepare_compton_events,build_detector_position_variance,build_compton_cone_weights
from detector_csv import load_detector_coordinates

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','analysis','factors','geometry','grid-config','output'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    gate=json.loads((a.analysis/'validation_gate.json').read_text())
    if gate['geometry_mode']!='stable_float64' or gate['geometry_sha256']!=digest(a.geometry):raise ValueError('R1 manifest differs')
    config=json.loads(a.grid_config.read_text());coords,cells,slices=grid(config)
    geo=np.load(a.geometry);f=geo['ellipse_fraction'];vol=geo['cell_volume_mm3'];V=float(vol.sum())
    label=np.where(f<=1e-12,2,np.where(f>=1-1e-12,0,1))*3+np.where(np.abs(coords[:,2])<=30,0,np.where(np.abs(coords[:,2])<=45,1,2))
    mask=np.eye(9)[label];binvol=np.bincount(label,weights=vol,minlength=9)
    averaged=sum(mask[geo['inverse_rotation'][:,view]] for view in range(20))/20
    start=np.array([x[0] for x in slices]);counts=np.array([x[1] for x in slices])
    def true_contributions(rows):
        position=np.array([[float(r['source_x']),float(r['source_y'])+345,float(r['source_z'])] for r in rows])
        radius=np.hypot(position[:,0],position[:,1]);z=position[:,2]
        if np.any(radius>255+1e-6) or np.any(np.abs(z)>60+1e-6):raise ValueError('Source outside support')
        ring=np.floor((radius+3)/6).astype(int);noncenter=ring>0
        layer=np.minimum(39,np.floor((z+60)/3).astype(int))
        result=np.zeros((len(rows),9));theta=np.arctan2(position[:,1],position[:,0])
        for view in range(20):
            # Circle transport is pooled at detector view 1. True source labels
            # must receive the SAME rotation average as response-derived S.
            cell=np.zeros(len(rows),dtype=int);angular=theta+view*np.pi/10
            n=counts[ring[noncenter]-1]
            cell[noncenter]=start[ring[noncenter]-1]+np.floor(angular[noncenter]*n/(2*np.pi)+.5).astype(int)%n
            result[np.arange(len(rows)),label[layer*3301+cell]]+=1/20
        return result
    torch.set_num_threads(8);torch.set_grad_enabled(False);device=torch.device(a.device)
    torch.cuda.set_device(device)
    factor=a.factors/'440keV_RotateNum20';det=torch.tensor(load_detector_coordinates(factor/'Detector.csv',10496),device=device)
    variance=build_detector_position_variance(det,0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35,geometry_mode='stable_float64')
    coords_t=torch.tensor(coords,dtype=torch.float32,device=device);mask_t=torch.tensor(averaged,dtype=torch.float64,device=device)
    rawmat=np.memmap(factor/'SysMat_polar',dtype='<f4',mode='r',shape=(132040,10496))
    B=torch.tensor(np.array(rawmat.T,copy=True),device=device);del rawmat
    worker=np.zeros((200,9));validation_worker=np.zeros((200,9));direct_counts=np.zeros(9)
    for dataset in ('circle_train','circle_validation'):
        collection=json.loads((a.inputs/dataset/'collection.json').read_text())
        if collection['views']!=[1]:raise ValueError('Expected invariant-circle pooled single-view collection')
        kept=np.load(a.analysis/f'{dataset}_ideal_v01_kept_rows.npy');wanted=set(kept.tolist());metadata={}
        with (a.inputs/dataset/'events_v01.csv').open() as f:
            for row in csv.DictReader(f):
                index=int(row['global_ideal_row'])
                if index in wanted:metadata[index]=row
        if len(metadata)!=len(kept):raise ValueError('Accepted identities incomplete')
        if dataset=='circle_validation':
            meta=[metadata[int(index)] for index in kept]
            contribution=true_contributions(meta);direct_counts=contribution.sum(0)
            np.add.at(validation_worker,np.array([int(r['worker']) for r in meta]),contribution)
            Nval=sum(collection['primary_counts']);continue
        Ntrain=sum(collection['primary_counts'])
        raw=np.loadtxt(a.inputs/dataset/'ideal_v01.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        for offset in range(0,len(kept),32):
            rows=kept[offset:offset+32]
            prepared,_=prepare_compton_events(torch.tensor(raw[rows],device=device),settings,det,variance,variance,input_energies_already_smeared=True)
            if prepared is None or prepared.count!=len(rows):raise ValueError('Frozen accepted event fails original energy selection')
            response=build_compton_cone_weights(prepared,coords_t,settings)*B[prepared.cpnum1-1]
            normal=response/response.sum(1,keepdim=True)
            contribution=(normal.double()@mask_t).cpu().numpy()
            for index,values in zip(rows,contribution):worker[int(metadata[int(index)]['worker'])]+=values
            if offset%32000==0:print('SPATIAL_TRAIN_EVENTS',offset,len(kept),flush=True)
    sensitivity=np.fromfile(a.analysis/'ideal/Sensi_d',dtype='<f4')
    predicted=np.bincount(label,weights=sensitivity,minlength=9)/binvol
    independently_predicted=worker.sum(0)*V/Ntrain/binvol
    if not np.allclose(predicted,independently_predicted,rtol=1e-5,atol=1e-12):raise ValueError('Rotation/worker sensitivity disagrees')
    w_eff=worker*V/(Ntrain/200*binvol)
    train_se=w_eff.std(0,ddof=1)/math.sqrt(200)
    emission_fraction=binvol/V;observed=direct_counts/Nval/emission_fraction
    validation_eff=validation_worker/(Nval/200*emission_fraction)
    observed_se=validation_eff.std(0,ddof=1)/math.sqrt(200)
    rows=[]
    for i in range(9):
        adequate=direct_counts[i]>=400
        error=float(predicted[i]/observed[i]-1)
        relse=float(math.hypot(train_se[i],observed_se[i])/observed[i])
        passed=adequate and not(abs(error)>.2 and abs(error)>3*relse)
        rows.append(dict(domain=('interior','partial_boundary','outside_ellipse')[i//3],axial_bin=i%3,
            rotation_averaged_accepted_mass=float(direct_counts[i]),expected_emissions=float(Nval*emission_fraction[i]),
            observed=float(observed[i]),predicted=float(predicted[i]),relative_error=error,
            relative_standard_error=relse,statistically_adequate=bool(adequate),passed=bool(passed)))
    result=dict(status='PASSED' if all(r['passed'] for r in rows) else 'HOLD',gates=rows,
        sensitivity_sha256=digest(a.analysis/'ideal/Sensi_d'),R1_gate_sha256=digest(a.analysis/'validation_gate.json'),
        geometry_sha256=digest(a.geometry),source_normalization='actual primary totals times analytical uniform bin volume fraction',
        training_uncertainty='200 independent workers; response and true-source rotations combined within each worker',
        validation_uncertainty='200 independent workers; correlated source labels across 20 views are not separate samples',
        true_source_rotation_average=True,new_transport_photons=0)
    (a.output/'spatial_gate.json').write_text(json.dumps(result,indent=2)+'\n');print('SPATIAL_GATE',result['status'])

if __name__=='__main__':main()
