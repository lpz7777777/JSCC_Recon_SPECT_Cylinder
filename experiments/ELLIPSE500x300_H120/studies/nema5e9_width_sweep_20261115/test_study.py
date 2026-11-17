"""CPU scientific-definition and engineering checks; same tests inside formal job."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
import compton_energy_probability_v5 as old
from width_kernel import scaled_energy_log_density,scaled_response
from detector_csv import load_detector_coordinates
from compton_event_response import ComptonEventSettings,prepare_compton_events,build_detector_position_variance
from jscc_5e10_joint_mlem import joint_mlem
import torch_active_operator as op
from adapters import BoundedViewResponse,MemoryBudget,OrderedTransferPool,ViewBlocks

def main():
    root=Path(__file__).resolve().parent;torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.manual_seed(35100101)
    law=old.ContinuousTransferLaw.load(root/'transfer_training_summary.json')
    det=torch.tensor(load_detector_coordinates(root/'Detector.csv',10496),dtype=torch.float32)
    var=build_detector_position_variance(det,0.)
    raw=np.load(root/'calibration_rows.npy')[[0,231,17000,80000,170246]]
    settings=ComptonEventSettings(.440,.13*np.sqrt(.511/.440),2*.440**2/(.511+2*.440)-.001,.05,.350,geometry_mode='stable_float64')
    rng=torch.random.get_rng_state().clone()
    p,_=prepare_compton_events(torch.tensor(raw),settings,det,var,var,input_energies_already_smeared=True)
    assert p.count==5 and torch.equal(rng,torch.random.get_rng_state())
    geo=np.load(root/'whole_geometry.npz');co=torch.tensor(geo['coordinates_mm'][[0,100,66121,66124,67001,68001]],dtype=torch.float32)
    B=torch.rand((10496,len(co)))+.01;checks={};diagnostics=[]
    ref=old.normalized_proxy_response(p,co,B,law,node_chunk=16,backend='tail16_mid2',gaussian_float32=True)
    checks['alpha100_original_response_bitwise']=torch.equal(ref,scaled_response(p,co,B,law,1.))
    beta,sp=old.geometry_arrays(p,co)
    for scale in [1.,.8,.6]:
        logp,_=scaled_energy_log_density(p,co,law,node_chunk=16,backend='tail16_mid2',gaussian_float32=True,width_scale=scale)
        errors=[];means=[];ratios=[]
        for i in range(len(raw)):
            layer=int(round((abs(float(p.pos1[i,1]))-300)/30))
            for j in range(len(co)):
                angle=float(beta[i,j]);pos=float(sp[i,j]);m,s,w,_=law.components(angle,layer,pos,order='tail16_mid2')
                mu=float(np.sum(m*w));mean=mu+scale*(m-mu);sd=scale*s
                reference=logsumexp(-.5*((float(p.e1[i])-mean)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)+np.log(w))
                errors.append(abs(float(logp[i,j])-reference)/max(1.,abs(reference)))
                mean_new=np.sum(w*mean);v0=np.sum(w*((m-mu)**2+s*s));v1=np.sum(w*((mean-mean_new)**2+sd*sd))
                means.append(abs(mean_new-mu));ratios.append(v1/v0)
        # FP32 Gaussian evaluation is inherited; independent scalar reference double.
        assert max(errors)<1e-5,(scale,max(errors))
        assert max(means)<1e-12 and max(abs(np.array(ratios)-scale**2))<1e-12
        response=scaled_response(p,co,B,law,scale)
        assert torch.allclose(response.sum(1),torch.ones(len(raw)),rtol=0,atol=1e-5)
        checks[f'alpha{scale}_definition_and_normalization']=True
        diagnostics.append(dict(scale=scale,max_scaled_logpdf_error=max(errors),max_mean_change_MeV=max(means),variance_ratio_min=min(ratios),variance_ratio_max=max(ratios)))
    # Existing joint-only helper versus original joint branch, then cache/packing.
    active=np.arange(6);fraction=np.ones(6);rotation=np.tile(np.arange(6)[:,None],(1,20))
    g=op.ActiveGeometry(active,fraction,rotation);A=torch.rand((4,6))+.1;r=op.ViewResponse(A,g,'none');y=torch.rand((4,20))+.1;ss=r.sensitivity()
    events=[[ref[:0].clone(),ref[:3].clone(),ref[3:].clone()] for _ in range(20)]
    sc=torch.tensor(np.fromfile(root/'continuous_energy_Sensi_full','<f4')[:6].copy())[:,None]
    original=op.compton_and_joint_mlem(r,y,events,ss,sc,100,50)[1]
    direct=joint_mlem(r,y,events,ss,sc,100,50)
    checks['joint_branch_and_save_history_bitwise']=all(torch.equal(a,b) for a,b in zip(original,direct))
    with BoundedViewResponse(op,A,g,budget=MemoryBudget(1<<28,0),phase='joint_solve_after_compton_prepare') as cached:
        with OrderedTransferPool(6,'cpu',max_host_staging_bytes=1<<20) as pool:
            result=joint_mlem(cached,y,[ViewBlocks(pool,b) for b in events],ss,sc,100,50)
            checks['cache_transfer_joint_history_bitwise']=all(torch.equal(a,b) for a,b in zip(direct,result))
    # RNG changes above are ONLY manufactured fixture generation, not event prep.
    checks['fixed_readout_rng_and_input_energies']=np.array_equal(p.e1.numpy(),raw[:,1]) and np.array_equal(p.e2.numpy(),raw[:,3])
    assert all(checks.values()),checks
    print('WIDTH_SWEEP_LOCAL_PASS',json.dumps(dict(checks=checks,scalar_reference=diagnostics,passed=True)))

if __name__=='__main__':main()
