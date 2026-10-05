"""Independent energy-domain probability probe, never a production kernel.

Learn true-transfer residuals only from circle_train. Predict the seven held-out
point sources. Condition both probability models on the same measured E1 window
and observed E2 energy-sum bound. No q-based selection in probability scores.
The empirical transfer law is conditional on recorded ideal multihit events;
this is explicitly not an unconditional material scattering distribution.
"""
import argparse
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from scipy.special import log_ndtr, logsumexp
from process_list_global_audit_v4 import metadata,vector,transfer,digest,write,table
from detector_csv import load_detector_coordinates

EDGES=np.array([0,45,60,75,90,120,150,180.000001])

def conditioned_pdf_pit(energy,lo,hi,means,sigma):
    means=np.asarray(means);sigma=np.asarray(sigma)
    logpdf=-.5*((energy-means)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)
    def log_interval(lower,upper):
        lower=(lower-means)/sigma;upper=(upper-means)/sigma
        # Positive tails use survival probabilities to avoid subtracting 1-1.
        high=np.where(lower>=0,log_ndtr(-lower),log_ndtr(upper))
        low=np.where(lower>=0,log_ndtr(-upper),log_ndtr(lower))
        with np.errstate(divide='ignore'):
            return high+np.log(-np.expm1(low-high))
    logmass=logsumexp(log_interval(lo,hi))-np.log(len(means))
    if not np.isfinite(logmass):raise ValueError('Conditional probability normalization is nonfinite')
    pit=np.exp(logsumexp(log_interval(lo,energy))-np.log(len(means))-logmass)
    return float(logsumexp(logpdf)-np.log(len(means))-logmass),float(pit)

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    manifest=json.loads((a.inputs/'input_manifest.json').read_text())['files']
    for name in ('circle_train/events_v01.csv','circle_train/collection.json'):
        assert digest(a.inputs/name)==manifest[name]
    detector=load_detector_coordinates(a.detector,10496)
    layer=np.rint((abs(detector[:,1])-300)/30).astype(int)
    training={};count=0
    for r in metadata(a.inputs/'circle_train',1):
        if int(r['global_ideal_row'])<0 or float(r['measured_e2'])<=.05:continue
        c1,c2=int(r['c1'])-1,int(r['c2'])-1
        if layer[c1]==layer[c2]:continue
        source=vector(r,'source_')+np.array([0,345,0])
        first=vector(r,'p1_')+np.array([0,345,0]);second=vector(r,'p2_')+np.array([0,345,0])
        free,beta=transfer(first,second,source)
        key=(int(layer[c1]),int(np.searchsorted(EDGES,np.degrees(beta),side='right')-1))
        training.setdefault(key,[]).append(float(r['transfer_mev'])-float(free));count+=1
    laws={};description=[]
    for key,values in sorted(training.items()):
        if len(values)<200:continue
        # Fixed quadrature of the empirical distribution; no NEMA/image fitting.
        nodes=np.quantile(values,(np.arange(128)+.5)/128)
        laws[key]=nodes
        description.append(dict(layer=key[0],angle_bin=key[1],count=len(values),
            mean_residual_keV=float(np.mean(values)*1000),rms_residual_keV=float(np.sqrt(np.mean(np.array(values)**2))*1000),
            quantile_nodes_MeV=nodes.tolist()))
    with a.points.open() as stream:points=list(csv.DictReader(stream))
    upper=2*.440**2/(.511+2*.440)-.001
    records=[]
    for values in points:
        if values['reco_energy_pass']!='True':continue
        row=SimpleNamespace(**{k:(v if k=='dataset' else float(v)) for k,v in values.items()
            if k in ['dataset','view','worker','seed','event_id','input_row','measured_e1_keV','measured_e2_keV',
                     'beta_true_deg','free_transfer_keV','beta_center_deg','center_transfer_keV','layer1']})
        energy=row.measured_e1_keV/1000;lo=max(.05,.35-row.measured_e2_keV/1000)
        if not lo<=energy<=upper:raise ValueError('Scoring cohort does not meet frozen energy gate')
        models={}
        for position,beta,free in [('true',row.beta_true_deg,row.free_transfer_keV/1000),
                                   ('centre',row.beta_center_deg,row.center_transfer_keV/1000)]:
            key=(int(row.layer1),int(np.searchsorted(EDGES,beta,side='right')-1))
            if key not in laws:continue
            for name,delta in [('free',np.zeros(1)),('empirical_atomic',laws[key])]:
                means=free+delta;means=means[(means>0)&(means<.440)]
                if not len(means):continue
                sigma=.13/2.355*np.sqrt(.511*means)
                ll,pit=conditioned_pdf_pit(energy,lo,upper,means,sigma)
                models[name+'_'+position]=(ll,pit)
        if len(models)!=4:continue
        for name,(ll,pit) in models.items():
            records.append(dict(dataset=row.dataset,view=row.view,worker=row.worker,seed=row.seed,event_id=row.event_id,
                model=name,log_conditional_density_per_MeV=ll,pit=pit,input_row=row.input_row))
    table(a.output/'independent_point_probability.csv',records)
    results=[]
    for dataset in sorted({r['dataset'] for r in points}):
        subset=[r for r in records if r['dataset']==dataset]
        for position in ('true','centre'):
            old=sorted([r for r in subset if r['model']=='free_'+position],key=lambda r:(r['view'],r['input_row']))
            new=sorted([r for r in subset if r['model']=='empirical_atomic_'+position],key=lambda r:(r['view'],r['input_row']))
            gain=np.array([b['log_conditional_density_per_MeV']-aa['log_conditional_density_per_MeV'] for aa,b in zip(old,new)])
            workers=np.array([r['seed'] for r in old]);worker=np.array([gain[workers==i].mean() for i in sorted(set(workers))])
            assert len(worker)==20,'Independent seeds must span all 20 point-source views'
            pit_old=np.array([r['pit'] for r in old]);pit_new=np.array([r['pit'] for r in new])
            results.append(dict(dataset=dataset,position=position,events=len(gain),
                mean_log_density_gain_nats=float(gain.mean()),worker_mean_gain_nats=float(worker.mean()),
                median_log_density_gain_nats=float(np.median(gain)),
                gain_05_95_nats=np.quantile(gain,[.05,.95]).tolist(),
                worker_standard_error=float(worker.std(ddof=1)/np.sqrt(len(worker))),
                free_pit_tail_fraction=float(np.mean((pit_old<.01)|(pit_old>.99))),
                empirical_pit_tail_fraction=float(np.mean((pit_new<.01)|(pit_new>.99)))))
    write(a.output/'energy_probability_summary.json',dict(training_events=count,training_laws=description,
        validation=results,source_sha256=digest(__file__),point_residuals_sha256=digest(a.points),
        training_metadata_sha256=manifest['circle_train/events_v01.csv'],elapsed_seconds=time.monotonic()-start,
        probability_measure='Measured E1 in MeV, normalized conditional on lower=max(50keV,350keV-E2), upper=ComptonEdge-1keV',
        q_selection_used_in_scores=False,held_out_sources='All seven point locations; none trained or tuned this surrogate',
        worker_identity='Independent seed; local worker index resets in each view',
        production_eligible=False,
        limitations=['Empirical transfer law is conditioned on ideal multihit recording and E2>50keV; not unconditional material physics',
                    'No crystal-position integral, joint C1/C2 measurement probability or q-selection normalizer certified',
                    'Stepwise angle-bin empirical law is a diagnostic prototype, not a production response'],
        new_photons=0,new_fine_A_matrices=0))
    print('ENERGY_PROBE_FINISHED',len(records),round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','points','detector','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
