"""Summarize complete-cell operator diagnostics and physical A coverage.

Scientific response figures only. No reconstruction, smoothing or unknown
phantom truth is created. Physical-grid counts remain frozen.
"""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import grid
from generate_compton_a_guard import axes,digest
from compton_overlap_integrator import reference_cell_bounds
from compton_boundary_quadrature import rotate_to_detector

HERE=Path(__file__).resolve().parent
DATA=HERE/'generated/compton_response_geometry_v3'
REPORT=HERE/'reports/NEMA_Body_H60/compton_response_geometry_v3'


def coverage():
    config=HERE/'config.json';geometry=HERE/'generated/Geometry/geometry.npz'
    cfg=json.loads(config.read_text());geo=np.load(geometry);coords,cells,_=grid(cfg)
    measure=np.load(DATA/'precise_measure/measure.npz');partial=measure['partial_indices'];n=cfg['points_per_layer']
    if len(partial)!=6880 or len(geo['active_indices'])!=82040:raise ValueError('Frozen grid differs')
    ready_path=DATA/'A440_near_patch_ultrafine_ready.json';ready=json.loads(ready_path.read_text())
    boxes=[(part['spec']['name'],axes(part['spec'])) for part in ready['parts']]
    patch_names={name for name,_ in boxes};column_record=REPORT/'guard_column_summary.json'
    column_hash=None
    if column_record.exists():
        column=json.loads(column_record.read_text())
        if column['status']!='COLUMN_COMMON_POINTS_PASSED_GLOBAL_ACCURACY_HOLD':
            raise ValueError('Column common-point validation failed')
        boxes.append((column['part']['spec']['name'],axes(column['part']['spec'])))
        column_hash=column['evidence_sha256']
    views=[];required=[]
    for view in range(20):
        mapped=geo['inverse_rotation'][partial,view]
        np.testing.assert_allclose(rotate_to_detector(coords[partial],view),coords[mapped],rtol=0,atol=1e-10)
        required.append(mapped);counts={'guarded':0,**{name:0 for name,_ in boxes}}
        for index in partial:
            lo,hi=reference_cell_bounds(cells[index%n],coords[index,2],view);choice='guarded'
            for name,xyz in boxes:
                if all(axis[0]-1e-10<=l and h<=axis[-1]+1e-10 for axis,l,h in zip(xyz,lo,hi)):
                    choice=name;break
            counts[choice]+=1
        views.append(dict(view=view+1,partial_cells=len(partial),fine_cells=len(partial)-counts['guarded'],fields=counts))
    required=np.unique(np.concatenate(required));volume=float(geo['cell_volume_mm3'][required].sum())
    annulus=np.pi*(255**2-cells[partial%n,0].min()**2)*cfg['height_mm']
    estimates=[]
    for spacing in ((.375,.375,.375),(.75,.75,.375),(.375,.375,.75),(.75,.75,.75)):
        points=volume/np.prod(spacing);annulus_points=annulus/np.prod(spacing)
        estimates.append(dict(spacing_mm=list(spacing),
            unique_sector_volume_without_padding_sample_estimate=points,
            full_annulus_volume_without_padding_sample_estimate=annulus_points,
            four_raw_11520_row_files_volume_estimate_bytes=points*11520*4*4,
            one_combined_10496_row_file_volume_estimate_bytes=points*10496*4,
            full_annulus_four_raw_volume_estimate_bytes=annulus_points*11520*4*4,
            note='Volume/spacing-product estimates, excluding Cartesian box padding, halo and shared-face overhead; not production reservations.'))
    result=dict(status='COVERAGE_PLAN_ONLY_GLOBAL_A_ACCURACY_HOLD',views=views,
        object_partial_cells=6880,unique_detector_frame_reference_cells=len(required),
        unique_transverse_reference_cells=len(np.unique(required%n)),required_reference_volume_mm3=volume,
        covered_cell_view_pairs=sum(r['fine_cells'] for r in views),total_cell_view_pairs=6880*20,
        original_patch_only_cell_view_pairs=sum(sum(r['fields'][name] for name in patch_names) for r in views),
        column_gate_sha256=column_hash,
        geometry_sha256=digest(geometry),patch_ready_sha256=digest(ready_path),storage_estimates=estimates,
        production_started=False,reconstruction_permitted=False,
        interpretation='Complete reference-cell coverage, not representative-point or object-only coverage. '
                       'Full reference outside the physical ellipse remains necessary for normalization. '
                       'Includes the .75 mm column when present; coverage is not an accuracy certificate.')
    (REPORT/'A_coverage_plan.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    cov=coverage();benchmark=json.loads((REPORT/'overlap_benchmark_cached_summary.json').read_text())
    components=json.loads((REPORT/'guard_components_summary.json').read_text())
    sampling=json.loads((REPORT/'guard_sampling_summary.json').read_text())
    if components['component_sums_passed']!=84:raise ValueError('Component additivity failed')
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axs[0,0]
    fine=[c for c in benchmark['cases'] if c['order']==[32,12,12]]
    ax.bar([str(c['view']) for c in fine],[c['maximum_integral_relative_change']*100 for c in fine],color='#245f91')
    ax.axhline(1,color='red',ls='--',label='1% criterion')
    ax.set(xlabel='View (8 fixed events per view)',ylabel='Maximum integral change (%)',
           title='All 6880 partial cells: medium to fine quadrature',ylim=(0,1.1));ax.legend()
    ax=axs[0,1];share=components['PE_share_of_component_absolute_changes']*100
    bars=ax.bar(['PE','Scatter'],[share,100-share],color=['#245f91','#b75d30'])
    ax.bar_label(bars,fmt='%.1f%%');ax.set(ylim=(0,100),ylabel='Share of absolute component differences (%)',
        title='Coarse A vs 0.375 mm A: 84 local point cases')
    ax=axs[1,0];entries=list(sampling['candidates'].items())
    labels=['/'.join(f'{x:g}' for x in r['spacing_mm']) for _,r in entries]
    values=[r['maximum_relative_change']*100 for _,r in entries]
    bars=ax.barh(np.arange(len(entries)),values,color=['#245f91' if r['passed']==84 else '#b75d30' for _,r in entries])
    ax.set_yticks(np.arange(len(entries)),labels);ax.invert_yaxis();ax.bar_label(bars,fmt='%.2f',padding=2)
    ax.axvline(1,color='red',ls='--');ax.set(xlabel='Maximum integral change (%)',ylabel='x/y/z A sampling (mm)',
        title='Nested subsampling of existing fine matrix; local evidence',xlim=(0,max(values)*1.2))
    ax=axs[1,1];ax.bar([r['view'] for r in cov['views']],[r['fine_cells'] for r in cov['views']],color='#245f91')
    ax.set(xlabel='View',ylabel='Complete cells covered by fine A',xticks=[1,6,11,16,20],
        title=f"Local .375/.75 mm A: {cov['covered_cell_view_pairs']}/{cov['total_cell_view_pairs']} pairs")
    fig.suptitle('Compton R2 response diagnostics — no reconstruction or smoothing',fontsize=15)
    destination=REPORT/'full_operator_diagnostics.png';fig.savefig(destination,dpi=160);plt.close(fig)
    (REPORT/'full_operator_figure_manifest.json').write_text(json.dumps(dict(
        artifact=destination.name,sha256=digest(destination),type='response_diagnostic_not_reconstruction',
        inputs_sha256={n:digest(REPORT/n) for n in ('overlap_benchmark_cached_summary.json',
            'guard_components_summary.json','guard_sampling_summary.json','A_coverage_plan.json','guard_column_summary.json') if (REPORT/n).exists()},
        source_sha256=digest(Path(__file__))),indent=2)+'\n',encoding='utf-8')
    backend=REPORT/'overlap_backend_validation.json'
    if backend.exists():
        check=json.loads(backend.read_text())
        if check['status']!='PASSED_NUMERICAL_EQUIVALENCE_ONLY':raise ValueError('Device response differs from CPU reference')
        cpu=json.loads((REPORT/'overlap_benchmark_batched_summary.json').read_text())
        gpu=json.loads((REPORT/'overlap_benchmark_cuda_summary.json').read_text())
        timings=[]
        for order in ((8,4,4),(16,8,8),(32,12,12)):
            a=np.mean([r['statistics']['elapsed_seconds'] for r in cpu['cases'] if tuple(r['order'])==order])
            b=np.mean([r['statistics']['elapsed_seconds'] for r in gpu['cases'] if tuple(r['order'])==order])
            timings.append(dict(order=list(order),events_per_batch=64,cpu_seconds=float(a),device_seconds=float(b),speedup=float(a/b)))
        fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');x=np.arange(3)
        axs[0].bar(x-.18,[r['cpu_seconds'] for r in timings],.36,label='CPU A and accumulation')
        axs[0].bar(x+.18,[r['device_seconds'] for r in timings],.36,label='Device A and accumulation')
        axs[0].set(xticks=x,xticklabels=['8/4/4','16/8/8','32/12/12'],xlabel='Quadrature order',
            ylabel='Seconds / 64 events / all 6880 cells',title='Actual full-cell benchmark (views 1 and 6)');axs[0].legend()
        bars=axs[1].bar(x,[r['speedup'] for r in timings],color='#245f91');axs[1].bar_label(bars,fmt='%.2fx')
        axs[1].set(xticks=x,xticklabels=['8/4/4','16/8/8','32/12/12'],ylabel='Speedup',xlabel='Quadrature order',ylim=(0,11),
            title='Same K, A, weights, events and normalization')
        fig.suptitle(f"{check['passed_integral_entries']:,} / {check['total_integral_entries']:,} integrals agree; global A accuracy remains HOLD")
        target=REPORT/'operator_backend_benchmark.png';fig.savefig(target,dpi=160);plt.close(fig)
        (REPORT/'operator_backend_benchmark.json').write_text(json.dumps(dict(
            timings=timings,figure_sha256=digest(target),backend_gate_sha256=digest(backend),
            cpu_gate_sha256=cpu['evidence_sha256'],device_gate_sha256=gpu['evidence_sha256'],
            peak_device_reserved_percent=gpu['peak_gpu_reserved_bytes']/gpu['total_gpu_bytes']*100,
            measurements_include_parallel_diagnostics=True,
            interpretation='Measured bounded diagnostics, not an S2 or reconstruction ETA.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:cov[k] for k in ('unique_detector_frame_reference_cells','unique_transverse_reference_cells',
        'covered_cell_view_pairs','total_cell_view_pairs','required_reference_volume_mm3')}))


if __name__=='__main__':main()
