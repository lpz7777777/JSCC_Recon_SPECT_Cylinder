"""Plot response evidence, never reconstructed-image smoothing or clipping."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'reports/NEMA_Body_H60/process_list_audit'
DATA = HERE/'generated/Diagnostics/process_list_audit'


def main():
    report = json.loads((OUT/'audit.json').read_text())
    example = json.loads((OUT/'example_incompatible_event.json').read_text())
    attribution = json.loads((OUT/'peak_attribution.json').read_text())
    with (DATA/'events.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['variant']=='current' and float(r['raw_sum'])>0]
    support = np.array([float(r['effective_support_full']) for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), layout='constrained')
    ax = axes[0, 0]
    ax.hist(support, bins=np.geomspace(10, 100000, 45), color='#456a89')
    ax.axvline(50, color='#b33d33', ls='--', label='Historical threshold = 50')
    ax.set(xscale='log', yscale='log', xlabel='Effective support (full circle)',
           ylabel='Sampled events', title='A. Rare concentrated responses remain')
    ax.legend(fontsize=9)
    ax.text(.04, .92, f'{len(rows):,} valid sampled rows; {(support<50).sum()} below 50',
            transform=ax.transAxes, va='top', fontsize=10)
    ax = axes[0, 1]
    intervals = [example['geometric_angle_full_circle_deg'],
                 example['geometric_angle_actual_source_cells_deg']]
    for y, bounds, label, color in zip([1, 0], intervals,
                                     ['Any full-grid source', 'Actual NEMA source'],
                                     ['#4d7b97', '#24826c']):
        ax.plot(bounds, [y,y], lw=13, color=color, solid_capstyle='butt')
        ax.text(np.mean(bounds), y+.2, f'{bounds[0]:.1f} - {bounds[1]:.1f} deg', ha='center', fontsize=10)
    ax.axvline(example['inferred_theta_deg'], color='#b33d33', lw=2,
               label=f"Energy angle {example['inferred_theta_deg']:.1f} deg")
    ax.set(xlim=(30,180), ylim=(-.5,1.7), yticks=[0,1],
           yticklabels=['NEMA', 'Full circle'], xlabel='Scattering angle (degrees)',
           title='B. Example passes energy-sum gate: 445 keV')
    ax.legend(loc='upper left', fontsize=9)
    ax = axes[1, 0]
    labels = ['All sampled\nvalid events','Remove support\n< 50','Remove full-grid\nARM > 3 sigma']
    keys = ['all','exclude_neff_lt50','exclude_arm_full_gt3']
    values = [attribution[k]['peak_truth_score_with_floor'] for k in keys]
    ax.bar(range(3), values, color=['#b33d33','#b77a41','#456a89'])
    ax.axhline(1, color='black', ls='--')
    for i,v in enumerate(values):
        ax.text(i,v*1.17,f'{v:.1f}',ha='center',fontsize=10)
    ax.set(xticks=range(3),xticklabels=labels,yscale='log',ylim=(.6,6e4),
           ylabel='Estimated truth score / sensitivity',
           title='C. Source-outside ascent remains after diagnostic cuts')
    ax.text(.44,.94,'Current sensitivity held fixed.\nStratified estimate only;\nnot a new reconstruction.',
            transform=ax.transAxes, fontsize=9,va='top')
    ax = axes[1, 1]
    keys = ['current','legacy_taylor_2mm_same_energy','no_first_source_leg']
    values = [report['summary'][k]['final_tiny_responsibility_weighted_mean']*100 for k in keys]
    ax.bar(range(3), values, color=['#456a89','#8c6a99','#42866b'])
    for i,v in enumerate(values):
        ax.text(i,v+.15,f'{v:.2f}%',ha='center')
    ax.set(xticks=range(3),xticklabels=['Current','Legacy Taylor / 2 mm\nsame measured energy','Current without\nfirst source leg'],
           ylim=(0,11),ylabel='Responsibility in f < 0.1 cells (%)',
           title='D. Kernel swaps at the unchanged final baseline image')
    fig.suptitle('NEMA 5e9: process_list audit | 20 views x 512 candidate events | no reconstruction',fontsize=13)
    fig.savefig(OUT/'response_audit.png',dpi=160)
    plt.close(fig)
    rows = json.loads((OUT/'angle_widths.json').read_text())['energy_sigma_comparison_same_current_resolution']
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax, side in zip(axes,['minus','plus']):
        for method,color in [('old','#9b6546'),('current','#3d758e')]:
            ax.plot([r['e1_keV'] for r in rows],[r[f'{method}_{side}_deg'] for r in rows],
                    'o-',color=color,label=method)
        ax.set(xlabel='Measured E1 (keV)',ylabel='Angular width (degrees)',
               title=f'Energy uncertainty: {side} side')
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Same 13% FWHM at 511 keV: historical Taylor is not uniformly broader')
    fig.savefig(OUT/'angle_widths.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
