"""Small, reproducible evidence and scientific plots from frozen response audits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPORT=HERE/'reports/NEMA_Body_H60/process_list_followup_20261004'
DATA=HERE/'generated/compton_first_scatter_v2'


def main():
    events_path=DATA/'geometry_stability_full/events.csv'
    expected=json.loads((REPORT/'download_manifest.json').read_text())['events.csv']['sha256']
    assert hashlib.sha256(events_path.read_bytes()).hexdigest()==expected
    rows=list(csv.DictReader(events_path.open()))
    assert len(rows)==91231
    def array(key):
        a=np.array([float(r[key]) for r in rows])
        assert np.isfinite(a).all()
        return a
    oldq=array('q_current_gpu');newq=array('q_stable')
    tv=array('ellipse_conditional_total_variation')
    cut=newq>3
    witnesses=json.loads((REPORT/'witness_cpu_cuda.json').read_text())['results']
    stats={}
    responsibilities={}
    for channel in ('compton','jscc'):
        a=array(channel+'_peak_responsibility_old');ranked=np.sort(a)[::-1]
        cumulative=ranked.cumsum()/ranked.sum()
        responsibilities[channel]=cumulative
        stats[channel]=dict(total=float(a.sum()),max_single_event=float(a.max()),
            events_to_50_percent=int(np.searchsorted(cumulative,.5)+1),
            events_to_90_percent=int(np.searchsorted(cumulative,.9)+1),
            effective_event_count=float(a.sum()**2/(a@a)),
            fraction_from_new_q_rejects=float(a[cut].sum()/a.sum()),
            fraction_from_TV_over_01=float(a[tv>.1].sum()/a.sum()))
    selected=[r for r in rows if float(r['q_stable'])>3 or float(r['ellipse_conditional_total_variation'])>.1]
    with (REPORT/'changed_events.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
    boundary=json.loads((DATA/'offline/boundary_integration.json').read_text())
    valid=[r for r in boundary if list(r['quadrature'].values())[-1]>1e-15]
    ratios=np.array([r['representative']/list(r['quadrature'].values())[-1] for r in valid])
    centroid=np.array([r['centroid']/list(r['quadrature'].values())[-1] for r in valid])
    result=dict(diagnostic_only=True,production_changed=False,
        population='Frozen ideal-first-scatter B accepted NEMA 1e9 events; excludes previously rejected candidates',
        number_of_events=len(rows),new_q_rejects=int(cut.sum()),
        ellipse_TV_gt_01=int((tv>.1).sum()),peak_responsibilities=stats,
        boundary=dict(cases=len(boundary),valid_cases=len(valid),converged=sum(r['converged'] for r in boundary),
            representative_over_10x=int((ratios>10).sum()),
            representative_ratio_quantiles=np.quantile(ratios,[0,.05,.5,.95,1]).tolist(),
            centroid_ratio_quantiles=np.quantile(centroid,[0,.05,.5,.95,1]).tolist(),
            note='Previously acquired independent point-source cases, not the two image peak response integrals'),
        interpretation='Responsibility is R_ej*x_j/(R_e*x) at the frozen B iteration-2000 image. '
            'JSCC values cover its Compton numerator only. This is not a new reconstruction, '
            'new matched sensitivity, or a bound on eventual image changes.')
    (REPORT/'impact_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axes[0,0];x=np.arange(len(witnesses))
    ax.bar(x-.16,[r['old_q_min'] for r in witnesses],width=.32,label='Current formula',color='#ce563e')
    ax.bar(x+.16,[r['stable_q_min'] for r in witnesses],width=.32,label='Stable prototype',color='#286b9a')
    ax.axhline(3,color='black',ls='--',lw=1,label='q = 3')
    ax.set_xticks(x,['CPU f32','CPU f64','CUDA f32','CUDA f64']);ax.set_ylabel('Best full-circle ARM score q')
    ax.set_title('Same frozen event: numerical acceptance differs');ax.legend(fontsize=8)
    ax=axes[0,1];ax.scatter(oldq,newq,s=2,alpha=.2,color='#286b9a',rasterized=True)
    ax.scatter(oldq[cut],newq[cut],s=35,color='#ce563e',label='6 newly over q=3')
    ax.plot([0,11],[0,11],color='gray',lw=.7);ax.axhline(3,color='black',ls='--',lw=.7)
    ax.set(xlim=(-.2,11),ylim=(-.2,11),xlabel='Current CUDA float32 q',ylabel='Stable float64 q',
        title='All 91,231 frozen accepted events');ax.legend(fontsize=9)
    ax=axes[1,0];ax.hist(np.log10(np.maximum(tv,1e-12)),bins=60,color='#286b9a')
    ax.set_yscale('log');ax.axvline(-1,color='#ce563e',ls='--')
    ax.set(xlabel='log10(TV distance of ellipse response row)',ylabel='Events (log scale)',
        title='Only 4 events have TV > 0.1; most rows barely change')
    ax=axes[1,1]
    for channel,label in [('compton','Compton'),('jscc','JSCC: Compton contribution')]:
        a=responsibilities[channel];ax.plot(np.arange(1,len(a)+1),100*a,label=label)
    ax.axhline(50,color='gray',ls='--',lw=.7);ax.set_xscale('log')
    ax.set(xlabel='Events ranked by contribution to current peak',ylabel='Cumulative peak responsibility (%)',
        title='Current peaks are supported by many events');ax.legend(fontsize=9)
    fig.suptitle('process_list response audit | frozen B at iteration 2000 | no image updates',fontsize=14)
    fig.savefig(REPORT/'response_numerical_audit.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
