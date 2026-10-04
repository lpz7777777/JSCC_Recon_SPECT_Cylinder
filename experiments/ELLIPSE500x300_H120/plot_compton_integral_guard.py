"""Whole-cell response diagnostics, explicitly not reconstructed images."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_compton_a_guard import digest


def main():
    here=Path(__file__).resolve().parent;data=here/'generated/compton_response_geometry_v3'
    report=here/'reports/NEMA_Body_H60/compton_response_geometry_v3'
    import csv
    def load(tag):return list(csv.DictReader((data/tag/'integral_cases.csv').open()))
    whole=load('guard_integral');rows=load('guard_patch_refined')
    ultra=data/'guard_patch_ultrafine/integral_cases.csv'
    if ultra.exists():rows=load('guard_patch_ultrafine')
    reference_spacing = 0.375 if ultra.exists() else 0.75
    reference_column = 'patch_0p375mm' if ultra.exists() else 'patch_0p75mm'
    def reference_error(row):
        value = float(row[reference_column])
        delta = float(row['fine']) - value
        norm = float(row['reference_norm'])
        return delta / max(abs(value), 1e-10 * norm), delta / norm
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    groups=[np.array([float(r['patch_refinement_relative_change'])*100 for r in rows]),
            np.array([float(r['refined_patch_relative_change'])*100 for r in rows])]
    labels=['3 -> 1.5 mm','1.5 -> 0.75 mm']
    if ultra.exists():
        groups.append(np.array([float(r['ultrafine_patch_relative_change'])*100 for r in rows]));labels.append('0.75 -> 0.375 mm')
    axs[0,0].boxplot([np.maximum(v,1e-7) for v in groups],tick_labels=labels,showfliers=True)
    axs[0,0].axhline(1,color='red',ls='--',label='1% response tolerance')
    axs[0,0].set_yscale('log');axs[0,0].set_ylabel('Absolute relative integral change (%)')
    axs[0,0].set_title('A sampling refinement: same K and quadrature');axs[0,0].legend(fontsize=8)
    for domain,color,label in [('object_intersection','#176b9a','ellipse intersection'),('full_reference_cell','#db7c24','full reference cell')]:
        selected=[r for r in rows if r['domain']==domain]
        x=np.arange(len(selected))
        axs[0,1].scatter(x,[reference_error(r)[0]*100 for r in selected],s=14,color=color,label=label)
    axs[0,1].axhline(0,color='black',lw=.7);axs[0,1].set_xlabel('Predetermined event / axial-layer case')
    axs[0,1].set_ylabel('Signed relative integral error (%)')
    axs[0,1].text(.02,.025,'Denominator: max(|reference|, 1e-10 Z)',
                  transform=axs[0,1].transAxes,fontsize=8,
                  bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
    axs[0,1].set_title(f'Whole-cell error; physical A reference {reference_spacing:g} mm');axs[0,1].legend(fontsize=8)
    values=np.array([float(r['quadrature_relative_change'])*100 for r in whole])
    axs[1,0].hist(values,bins=40,color='#387d5a');axs[1,0].set_xlabel('16/8/8 -> 32/12/12 quadrature change (%)')
    axs[1,0].set_ylabel('Cases');axs[1,0].set_title(f'Guarded A: {len(whole)} quadrature cases, max {values.max():.4f}%')
    rel=np.array([abs(reference_error(r)[0])*100 for r in rows])
    absolute=np.array([abs(reference_error(r)[1]) for r in rows])
    axs[1,1].scatter(rel,np.maximum(absolute,1e-15),s=15,color='#624891')
    axs[1,1].set_yscale('log');axs[1,1].set_xlabel('Absolute local integral error (%)')
    axs[1,1].set_ylabel('Absolute cell error / common baseline full-circle Z')
    axs[1,1].set_title('Keep local and row-normalization effects separate')
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle('Compton K x A boundary-response diagnostics; no NEMA images, no smoothing',fontsize=12)
    path=report/'whole_cell_interpolation_diagnostics.png';fig.savefig(path,dpi=160);plt.close(fig)
    evidence=[data/'guard_integral/integral_cases.csv',data/'guard_patch_refined/integral_cases.csv']
    if ultra.exists():evidence.append(ultra)
    (report/'whole_cell_figure_manifest.json').write_text(json.dumps(dict(
        figure_sha256=digest(path),plot_script_sha256=digest(Path(__file__)),
        physical_A_reference_spacing_mm=reference_spacing,
        relative_error_denominator_floor_over_common_Z=1e-10,
        sources={str(p.relative_to(here)):digest(p) for p in evidence},
        diagnostic_only=True,reconstructed_image=False,new_transport_photons=0),indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
