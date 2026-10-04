"""Read-only response-model investigation after paired first-scatter imaging.

Uses frozen accepted identities, not a new event selection or reconstruction.
Ellipse scores are discrete probes, not certified continuous-domain minima.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from geometry import grid
from first_scatter_offline import quadrature
from detector_csv import load_detector_coordinates
from compton_event_response import (ComptonEventSettings, build_detector_position_variance,
    prepare_compton_events, compton_theta_from_e1, _energy_angle_sigma,
    _position_angle_sigma, min_standardized_compton_arm)

DATA = HERE/'generated/compton_first_scatter_v2'
OUT = HERE/'reports/NEMA_Body_H60/process_list_followup_20261004'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def arm_matrix(p, coords, settings):
    from diagnose_compton_geometry_stability import stable_geometry
    a=p.pos1.double()[:,None]-coords.double()[None]
    b=(p.pos2.double()-p.pos1.double())[:,None]
    beta,sp,_=stable_geometry(a,b,p.sigma_pos1_sq,p.sigma_pos2_sq)
    theta=compton_theta_from_e1(p.e1.double(),settings.energy_mev)
    se=_energy_angle_sigma(p.e1.double(),settings,beta,theta)
    return (beta-theta[:,None]).abs()/(se.square()+sp.square()).clamp_min(1e-12).sqrt()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    gp=HERE/'generated/Geometry/geometry.npz';g=np.load(gp)
    coords=g['coordinates_mm'];active=g['active_indices'];frac=g['ellipse_fraction']
    volumes=g['cell_volume_mm3'];inside=(coords[:,0]/250)**2+(coords[:,1]/150)**2<=1+1e-12
    peaks=[]
    for group in ('legacy','ideal'):
        result=DATA/'RemoteResults'/group
        run=json.loads((result/'run_manifest.json').read_text())
        assert digest(gp)==run['geometry_sha256']
        sens=np.fromfile(DATA/f'analysis/{group}/Sensi_d','<f4')
        assert digest(DATA/f'analysis/{group}/Sensi_d')==run['sensi_d_sha256']
        for channel in ('440_ComptonOnly','440_SinglePlusCompton'):
            image=np.memmap(result/f'Image_{channel}_history.float32','<f4',mode='r',shape=(40,82040))[-1]
            j=int(active[int(image.argmax())]);rho=float(image.max())
            peaks.append(dict(group=group,channel=channel,index=j,coordinates_mm=coords[j].tolist(),
                fraction=float(frac[j]),volume_mm3=float(volumes[j]*frac[j]),density=rho,
                source_integral=rho*float(volumes[j]*frac[j]),representative_inside=bool(inside[j]),
                compton_rho_times_sensitivity=rho*float(sens[j]*frac[j]),
                note='rho*S is expected detector count term, not measured event responsibility; JSCC also has single sensitivity'))
    # Algebraic row-scaling invariance: fixed physical sensitivity and no active clamp.
    rng=np.random.default_rng(20261004)
    t=rng.lognormal(size=(64,200));x=rng.lognormal(size=200);s=rng.uniform(.1,1,200)
    scales=10**rng.uniform(-6,6,64)
    update=lambda a:x*(a.T@(1/(a@x)))/s
    row_error=float(np.linalg.norm(update(t)-update(t*scales[:,None]))/np.linalg.norm(update(t)))
    assert row_error<1e-12
    # Width comparison only, on identical measured energy and resolution.
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
    widths=[]
    for energy in (.060,.100,.150,.200,.250,.270,.277):
        e=torch.tensor([energy],dtype=torch.float64);theta=compton_theta_from_e1(e,.440)
        sigma=e*settings.energy_resolution/2.355*(.440/e).sqrt()
        d1=.511/((.440-e)**2*theta.sin().abs())
        d2=.511/(theta.sin()*(.440-e)**3)*(2-theta.cos()/theta.sin()**2*.511/(.440-e))
        widths.append(dict(measured_keV=energy*1000,theta_deg=float(theta[0]*180/np.pi),
            current_minus_deg=float(_energy_angle_sigma(e,settings,theta[:,None]-.01,theta)[0,0]*180/np.pi),
            current_plus_deg=float(_energy_angle_sigma(e,settings,theta[:,None]+.01,theta)[0,0]*180/np.pi),
            old_minus_deg=float((d1*sigma-.5*d2*sigma**2).abs()[0]*180/np.pi),
            old_plus_deg=float((d1*sigma+.5*d2*sigma**2).abs()[0]*180/np.pi)))
    detector_path=HERE/'generated/Diagnostics/process_list_audit/Detector.csv'
    detector=torch.tensor(load_detector_coordinates(detector_path,10496),dtype=torch.float32)
    assert np.array_equal(np.unique(abs(detector.numpy()[:,1])),[300,330,360,390])
    var=build_detector_position_variance(detector,0)
    _,cells,_=grid(json.loads((HERE/'config.json').read_text()))
    partial=np.flatnonzero((frac[:3301]>1e-13)&(frac[:3301]<1-1e-12))
    centroid=[]
    for cell in partial:
        p,w=quadrature(cells[cell],0,128,8,1);centroid.append((p*w[:,None]).sum(0)/w.sum())
    centroid=np.asarray(centroid)
    centroid=np.concatenate([centroid+np.array([0,0,z]) for z in np.unique(coords[:,2])])
    assert np.all((centroid[:,0]/250)**2+(centroid[:,1]/150)**2<=1+1e-12)
    input_manifest=json.loads((DATA/'analysis_inputs/input_manifest.json').read_text())['files']
    rows=[];sources={};coord_tensor=torch.tensor(coords,dtype=torch.float32)
    for view in range(20):
        path=DATA/f'analysis_inputs/NEMA/ideal_v{view+1:02d}.csv'
        sha=digest(path);assert sha==input_manifest[path.relative_to(DATA/'analysis_inputs').as_posix()]
        kept=np.load(DATA/f'analysis/NEMA_ideal_v{view+1:02d}_kept_rows.npy')
        selected=np.sort(rng.choice(kept,min(64,len(kept)),replace=False))
        raw=np.loadtxt(path,delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)[selected]
        sources[path.name]=sha
        angle=view*np.pi/10;c,sn=np.cos(angle),np.sin(angle)
        cc=centroid.copy();cc[:,0]=c*centroid[:,0]+sn*centroid[:,1];cc[:,1]=c*centroid[:,1]-sn*centroid[:,0]
        cc=torch.tensor(cc,dtype=torch.float32)
        circle_to_inside=torch.tensor(g['inverse_rotation'][np.flatnonzero(inside),view],dtype=torch.long)
        for offset in range(0,len(raw),8):
            p,_=prepare_compton_events(torch.tensor(raw[offset:offset+8]),settings,detector,var,var,input_energies_already_smeared=True)
            assert p.count==len(raw[offset:offset+8])
            q=arm_matrix(p,coord_tensor,settings);qfull=q.amin(1)
            qi=q[:,circle_to_inside].amin(1);qc=arm_matrix(p,cc,settings).amin(1)
            best=q.argmin(1).numpy();objbest=g['rotation'][best,view]
            for k in range(p.count):
                rows.append(dict(view=view+1,input_row=int(selected[offset+k]),full_circle_q=float(qfull[k]),
                    inside_centres_q=float(qi[k]),inside_plus_centroids_q=float(torch.minimum(qi,qc)[k]),
                    full_best_in_ellipse=bool(inside[objbest[k]]),full_best_object_index=int(objbest[k]),
                    measured_e1_keV=float(p.e1[k]*1000),measured_sum_keV=float((p.e1[k]+p.e2[k])*1000)))
        print('RESPONSE_DOMAIN_AUDIT_VIEW',view+1,flush=True)
    with (OUT/'ellipse_domain_probe.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(diagnostic_only=True,reconstruction_changed=False,score_mode='stable_float64_cross_product_prototype',geometry_sha256=digest(gp),
        production_kernel_sha256=digest(HERE.parents[1]/'compton_event_response.py'),
        detector_sha256=digest(detector_path),input_sha256=sources,
        active_cells=82040,active_representatives_outside_ellipse=int(np.sum(~inside[active])),
        tiny_fraction_cells=int(np.sum(frac[active]<.1)),peaks=peaks,
        row_rescaling_relative_l2=row_error,energy_angle_width_comparison=widths,
        domain_probe=dict(events=len(rows),stable_full_circle_q_over_3=sum(r['full_circle_q']>3 for r in rows),events_with_full_best_outside=sum(not r['full_best_in_ellipse'] for r in rows),
            inside_centres_q_over_3=sum(r['inside_centres_q']>3 for r in rows),
            inside_plus_centroids_q_over_3=sum(r['inside_plus_centroids_q']>3 for r in rows),
            interpretation='Discrete probes only. A score >3 here is an upper bound on continuous-domain best score; not proof for deletion. No B or image enters these scores.'))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['domain_probe']),flush=True)

if __name__=='__main__':run()
