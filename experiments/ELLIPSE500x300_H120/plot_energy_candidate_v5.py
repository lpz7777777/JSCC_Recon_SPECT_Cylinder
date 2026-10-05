"""Independent probability diagnostics; never presents these as reconstruction."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).parent
DATA=HERE/'generated/compton_energy_probability_v5/validation'
REPORT=HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_figures(figures):
    """Show all predeclared categories, and mark limited statistical power."""
    spatial=DATA/'spatial/spatial_efficiency.csv'
    joint=DATA/'spatial/independent_joint_categories.csv'
    if not spatial.exists() or not joint.exists():return {}
    rows=list(csv.DictReader(spatial.open()))
    records=list(csv.DictReader(joint.open()))
    fig,axes=plt.subplots(1,2,figsize=(13,4.4),layout='constrained')
    colors={'legacy':'tab:blue','candidate':'tab:orange'}
    for model in colors:
        adequate=[r for r in rows if r['model']==model and r['adequate']=='True']
        low=[r for r in rows if r['model']==model and r['adequate']!='True']
        for r in low:axes[0].axvspan(int(r['bin'])-.45,int(r['bin'])+.45,color='.8',alpha=.12)
        axes[0].plot([int(r['bin']) for r in adequate],
            [100*float(r['relative_bias']) for r in adequate],'.',color=colors[model],label=model)
        meaningful=[r for r in records if r['model']==model and r['adequate']=='True']
        position=[int(r['point'])*72+int(r['category']) for r in meaningful]
        axes[1].errorbar(position,[100*float(r['relative_bias']) for r in meaningful],
            yerr=[100*float(r['relative_se']) for r in meaningful],fmt='.',
            alpha=.65,color=colors[model],elinewidth=.7,label=model)
    for ax in axes:
        ax.axhline(0,color='.3',lw=.8);ax.set_ylabel('Prediction / observed efficiency - 1 (%)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set(xlabel='Frozen r / azimuth / z bin index',
        title='Spatial validation: 144 / 192 bins adequate\n48 shaded bins unresolved; no threshold failures')
    axes[1].set(xlabel='Point index * 72 + layer / E1 / E2 category',
        title='Joint validation: 57 / 504 categories adequate\n447 unresolved; error bars = 1 combined SE')
    fig.suptitle('Independent validation of matched sensitivity; fixed event sets and actual primary counts\nDiagnostics only: passing coarse tests does not establish voxel accuracy or spike reduction',fontsize=10)
    fig.savefig(figures/'independent_spatial_joint_validation.png',dpi=180);plt.close(fig)
    return {name:digest(DATA/'spatial'/name) for name in
        ('spatial_efficiency.csv','independent_joint_categories.csv','spatial_gate.json','joint_gate.json','execution.json')}


def main():
    figures=REPORT/'figures';figures.mkdir(exist_ok=True)
    gate=json.loads((DATA/'points/point_gate.json').read_text())
    rows=[r for r in gate['reports'] if r['position']=='centre']
    labels=[r['dataset'].replace('point_','P') for r in rows]
    x=np.arange(len(rows));fig,axes=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    axes[0].errorbar(x,[r['seed_mean_gain_nats'] for r in rows],
        yerr=[r['seed_standard_error'] for r in rows],fmt='o',capsize=4,label='20-seed mean +/- 1 SE')
    axes[0].plot(x,[r['median_gain_nats'] for r in rows],'x',label='event median')
    axes[0].axhline(0,color='.4',lw=1);axes[0].set_ylabel('Conditional log-density gain (nats)')
    axes[0].legend(fontsize=8);axes[0].set_title('Improvement is concentrated in tails')
    axes[1].bar(x-.17,[100*r['free_tail_fraction'] for r in rows],.34,label='free energy reference')
    axes[1].bar(x+.17,[100*r['material_tail_fraction'] for r in rows],.34,label='material candidate')
    axes[1].axhline(2,color='.4',lw=1,ls='--',label='uniform PIT: 2%')
    axes[1].set_ylabel('PIT <0.01 or >0.99 (%)');axes[1].legend(fontsize=8)
    axes[1].set_title('Same electronics and crystal variance')
    axes[2].bar(x,[100*r['endpoint_extrapolated_events']/r['events'] for r in rows],color='.5')
    axes[2].set_ylabel('Angle endpoint extrapolation (%)');axes[2].set_title('Extrapolation remains a limitation')
    for ax in axes:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Independent point-source energy diagnostics: 10,270 events; no reconstructed images\nEnergy-window conditional score; complete q domain checked separately',fontsize=11)
    fig.savefig(figures/'independent_point_probability.png',dpi=180);plt.close(fig)

    records=list(csv.DictReader((DATA/'points/point_probability.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,model in zip(axes,('free_centre','material_centre')):
        values=[float(r['pit']) for r in records if r['model']==model]
        ax.hist(values,bins=np.linspace(0,1,21),density=True,color='.45',edgecolor='white')
        ax.axhline(1,color='tab:red',ls='--');ax.set(xlim=(0,1),ylim=(0,3),xlabel='Conditional PIT',ylabel='Density',title=model.replace('_',' '))
    fig.suptitle('Held-out point sources pooled; no smoothing; not an unconditional joint-model certificate',fontsize=10)
    fig.savefig(figures/'conditional_pit.png',dpi=180);plt.close(fig)
    extra=validation_figures(figures)
    manifest=dict(kind='response probability diagnostics, not reconstruction',input_sha256={
        'point_gate.json':digest(DATA/'points/point_gate.json'),
        'point_probability.csv':digest(DATA/'points/point_probability.csv'),**extra},
        figures={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in figures.glob('*.png')},
        no_smoothing=True,new_reconstruction=False)
    (REPORT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('ENERGY_PROBABILITY_DIAGNOSTIC_FIGURES_READY')


if __name__=='__main__':main()
