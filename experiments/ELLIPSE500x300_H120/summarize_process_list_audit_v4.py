"""Build a five-track evidence ledger and figures from checked diagnostic outputs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPORT=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4'
DATA=HERE/'generated/process_list_global_audit_v4/diagnostics'

def load(name):return json.loads((DATA/name).read_text())

def run():
    output=REPORT/'figures';output.mkdir(exist_ok=True)
    points=load('points/point_residual_summary.json');spatial=load('spatial/fine_spatial_gate.json')
    sampling=load('sampling/whole_cell_K_sampling_summary.json');energy=load('energy/energy_probability_summary.json')
    cohorts=[r for r in points['cohorts'] if r['cohort']=='energy_selected']
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    x=np.arange(7)
    labels=('atomic_residual_keV','position_residual_keV','measurement_residual_keV','total_residual_keV')
    for i,key in enumerate(labels):ax.bar(x+(i-1.5)*.19,[r['residuals'][key]['rms'] for r in cohorts],width=.19,label=key.replace('_residual_keV',''))
    ax.set(xticks=x,xticklabels=['centre','x+225','x-225','y+135','y-135','z+57','z-57'],ylabel='signed residual RMS (keV)',title='Independent point sources: selected energies, before global q3 cut')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.savefig(output/'point_residual_components.png',dpi=160);plt.close(fig)
    adequate=[r for r in spatial['gates'] if r['statistically_adequate']]
    bias=np.array([r['relative_error'] for r in spatial['gates']]).reshape(4,8,6).transpose(2,0,1).reshape(6,32)*100
    mask=np.array([r['statistically_adequate'] for r in spatial['gates']]).reshape(4,8,6).transpose(2,0,1).reshape(6,32)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    palette=plt.get_cmap('RdBu_r').copy();palette.set_bad('#bdbdbd')
    color=ax.imshow(np.ma.array(bias,mask=~mask),aspect='auto',cmap=palette,vmin=-20,vmax=20)
    ax.set(xlabel='radial block (4) x azimuth sector (8)',ylabel='axial bin',title='Matched S versus direct source counts: gray = insufficient statistics')
    for i in (7.5,15.5,23.5):ax.axvline(i,color='black',lw=.5)
    fig.colorbar(color,ax=ax,label='relative efficiency bias (%)');fig.savefig(output/'fine_spatial_efficiency.png',dpi=170);plt.close(fig)
    values=[r for r in energy['validation'] if r['position']=='centre']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].errorbar(x,[r['worker_mean_gain_nats'] for r in values],yerr=[1.96*r['worker_standard_error'] for r in values],fmt='o',capsize=4)
    axes[0].axhline(0,color='black',lw=.5);axes[0].set(ylabel='conditional log-density gain (nats/event)',title='Circle-trained atomic surrogate; 95% worker-SE bars')
    for name,color in [('free','gray'),('empirical','tab:blue')]:
        axes[1].plot(x,[r[name+'_pit_tail_fraction']*100 for r in values],'o-',label=name,color=color)
    axes[1].axhline(2,color='black',ls='--',label='2% PIT reference');axes[1].set(ylabel='PIT outside [0.01,0.99] (%)',title='Same energy/sum selection, no q3 selection in scores');axes[1].legend()
    for ax in axes:ax.set(xticks=x,xticklabels=['centre','x+','x-','y+','y-','z+','z-']);ax.grid(alpha=.2)
    fig.savefig(output/'independent_energy_probability.png',dpi=165);plt.close(fig)
    with (DATA/'responsibility/native_peak_responsibility.csv').open() as stream:responsibility=list(csv.DictReader(stream))
    interior=[r for r in responsibility if int(r['full_cell_index'])==57231]
    with (DATA/'sampling/whole_cell_K_sampling.csv').open() as stream:sample=list(csv.DictReader(stream))
    weighted_errors=[abs(float(r['center_K'])-float(r['average_K_8'])) for r in sample]
    peakchecks=json.loads((REPORT/'whole_cell_implementation_checks.json').read_text())
    with (DATA/'crystal/crystal_offset_moments.csv').open() as stream:
        crystal=[r for r in csv.DictReader(stream) if r['cohort']=='energy_selected' and int(r['events'])>=200]
    summary=dict(study='process_list_global_audit_v4',status='OFFLINE_INVESTIGATION_COMPLETED_PRODUCTION_HOLD',
        stages=dict(offline_evidence='completed with explicit limits',whole_cell_implementation='isolated and tested',
            low_cost_energy_prototype='independent diagnostic positive',production_candidate='not certified',
            new_physical_S_and_paired_reconstruction='not launched because production probability contract is incomplete'),
        point_energy_selected_events=sum(r['events'] for r in cohorts),
        point_arm_source_outside_3_range=[min(r['arm_outside_3_fraction'] for r in cohorts),max(r['arm_outside_3_fraction'] for r in cohorts)],
        point_atomic_rms_keV_range=[min(r['residuals']['atomic_residual_keV']['rms'] for r in cohorts),max(r['residuals']['atomic_residual_keV']['rms'] for r in cohorts)],
        spatial=dict(status=spatial['status'],adequate_bins=len(adequate),unresolved_bins=192-len(adequate),
            sufficient_bin_error_range=[min(r['relative_error'] for r in adequate),max(r['relative_error'] for r in adequate)],
            caveat='Rotationally averaged 192-bin validation, not every voxel or every individual view'),
        K_sampling=dict(events=sampling['events'],control_cells=sampling['control_cells'],cases=sampling['cases'],
            adequate_cases=sampling['adequate_cases'],converged_cases=sampling['converged_cases'],
            centre_error_rms=sampling['center_relative_error']['rms'],
            mean_absolute_K_difference=float(np.mean(weighted_errors))),
        energy_diagnostic=dict(training_events=energy['training_events'],scored_point_events=values and sum(r['events'] for r in values),
            unsupported_point_events=sum(r['events'] for r in cohorts)-sum(r['events'] for r in values),
            independent_locations=7,positive_mean_gain_locations=sum(r['mean_log_density_gain_nats']>0 for r in values),
            centre_log_density_gain_range=[min(r['mean_log_density_gain_nats'] for r in values),max(r['mean_log_density_gain_nats'] for r in values)],
            free_PIT_tail_range=[min(r['free_pit_tail_fraction'] for r in values),max(r['free_pit_tail_fraction'] for r in values)],
            empirical_PIT_tail_range=[min(r['empirical_pit_tail_fraction'] for r in values),max(r['empirical_pit_tail_fraction'] for r in values)],
            production_eligible=False,remaining_requirements=energy['limitations']),
        interior_peak=interior,whole_cell_geometry=peakchecks,
        crystal_positions=dict(hits_outside_declared_crystal=load('crystal/summary.json')['hits_outside_declared_crystal'],
            statistically_adequate_moment_rows=len(crystal),
            variance_ratio_to_uniform_range=[min(float(r['variance_ratio_to_uniform']) for r in crystal),max(float(r['variance_ratio_to_uniform']) for r in crystal)],
            depth_bias_mm_range=[min(float(r['mean_offset_mm']) for r in crystal if r['axis']=='y'),max(float(r['mean_offset_mm']) for r in crystal if r['axis']=='y')],
            interpretation='Selected-hit depth distribution is not centred uniform; cannot use these conditional moments as unconditional correction'),
        new_transport_photons=0,new_fine_A_matrices=0,new_project_reconstruction=False,
        toy_is_not_project_reconstruction=True,
        priority='A normalized energy-domain response with material transfer tails, same fixed events, and independently validated matching S',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/'scientific_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('interior_peak','whole_cell_geometry')},indent=2))

if __name__=='__main__':run()
