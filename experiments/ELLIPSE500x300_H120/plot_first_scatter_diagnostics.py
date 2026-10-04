"""Render transport/model diagnostics, explicitly separate from image reconstruction."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent

def plot(study,output):
    output.mkdir(parents=True,exist_ok=False)
    gate=json.loads((study/"analysis/validation_gate.json").read_text())
    groups=("legacy","ideal");fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for row,gp in enumerate(groups):
        bins=[r for r in gate["gates"] if r["group"]==gp and r["test"]=="direct_source_bin_efficiency"]
        x=np.arange(9)
        direct=np.array([r["observed"] for r in bins]);pred=np.array([r["predicted"] for r in bins])
        axes[row,0].plot(x,direct,"o-",label="direct counts at true emission positions")
        axes[row,0].plot(x,pred,"s--",label="normalized K*B sensitivity prediction")
        axes[row,0].set(title=gp+": independent circle source efficiency",ylabel="accepted / emitted")
        errors=100*np.array([r["relative_error"] for r in bins]);se=100*np.array([r["relative_standard_error"] for r in bins])
        axes[row,1].errorbar(x,errors,yerr=3*se,fmt="o",label="relative error; bars = 3 SE")
        for i,r in enumerate(bins):
            if not r["passed"]:axes[row,1].plot(i,errors[i],"rx",ms=12)
        for limit in (-20,20):axes[row,1].axhline(limit,ls="--",color="grey")
        axes[row,1].set(title=gp+": spatial validation",ylabel="prediction / direct - 1 (%)")
        for col in (0,1):
            axes[row,col].set_xticks(x,[f"r{r} z{z}" for r in range(3) for z in range(3)],rotation=45)
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    fig.suptitle("First-scatter diagnostics; sensitivity gate = "+gate["status"]+"; these are not reconstruction images")
    fig.savefig(output/"independent_spatial_efficiency.png",dpi=160);plt.close(fig)
    figure,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    for gp,style in (("legacy","--"),("ideal","-")):
        pulls=[]
        with (study/"analysis/energy_domain_diagnostics.csv").open() as f:
            for r in csv.DictReader(f):
                if r["group"]==gp:pulls.append(float(r["pull"]))
        p=np.array(pulls);p=p[np.isfinite(p)]
        axes[0].hist(p,bins=np.linspace(-8,8,81),density=True,histtype="step",label=gp,linestyle=style)
        q=np.linspace(0,5,101);axes[1].plot(q,[np.mean(np.abs(p)<=k) for k in q],style,label=gp)
    from scipy.special import erf
    u=np.linspace(-8,8,201);axes[0].plot(u,np.exp(-u*u/2)/np.sqrt(2*np.pi),"k:",label="unit Gaussian")
    axes[1].plot(q,erf(q/np.sqrt(2)),"k:",label="unit Gaussian coverage")
    axes[0].set(title="held-out point-source energy-domain pulls",xlabel="(measured E1 - predicted E1) / predicted sigma")
    axes[1].set(title="diagnostic prototype coverage",xlabel="absolute pull threshold",ylabel="fraction inside")
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    figure.suptitle("Offline likelihood prototype; Doppler/atomic effects retained; no kernel change")
    figure.savefig(output/"energy_domain_prototype.png",dpi=160);plt.close(figure)
    layers=[]
    with (study/"offline/measurement_layers.csv").open() as f:layers=list(csv.DictReader(f))
    summary=[];fig,axes=plt.subplots(1,4,figsize=(17,5),layout="constrained")
    keys=("doppler_and_atomic_residual","crystal_center_residual","accumulated_energy_residual","measurement_residual")
    for i,key in enumerate(keys):
        for gp,flag,style in (("legacy","legacy","--"),("ideal","ideal","-")):
            a=1000*np.array([float(r[key]) for r in layers if int(r[flag]) and np.isfinite(float(r[key]))])
            axes[i].hist(a,bins=np.linspace(-100,100,81),density=True,histtype="step",label=gp,linestyle=style)
            summary.append(dict(group=gp,scope="raw_candidate_sample_including_same_layer",residual=key,samples=len(a),mean_keV=float(a.mean()),
                rms_keV=float(np.sqrt(np.mean(a*a))),p95_abs_keV=float(np.quantile(np.abs(a),.95))))
        axes[i].set(title=key.replace("_"," "),xlabel="residual (keV)");axes[i].grid(alpha=.2)
    axes[0].legend();fig.suptitle("Raw point-source candidates, including same-layer pairs; display +/-100 keV; not all used by reconstruction")
    fig.savefig(output/"measurement_error_layers.png",dpi=150);plt.close(fig)
    selected={}
    with (study/"point_diagnostics/point_consistency.csv").open() as f:
        for r in csv.DictReader(f):
            if r["physical_first_pair_matches_list"]=="True":
                selected.setdefault(r["group"],set()).add((r["dataset"],r["view"],r["seed"],r["event_id"]))
    fig,axes=plt.subplots(1,4,figsize=(17,5),layout="constrained")
    for i,key in enumerate(keys):
        for gp,style in (("legacy","--"),("ideal","-")):
            a=1000*np.array([float(r[key]) for r in layers if
                (r["dataset"],r["view"],r["seed"],r["event_id"]) in selected[gp] and np.isfinite(float(r[key]))])
            axes[i].hist(a,bins=np.linspace(-50,50,81),density=True,histtype="step",label=gp,linestyle=style)
            summary.append(dict(group=gp,scope="accepted_point_sample_matching_physical_first_pair",residual=key,
                samples=len(a),mean_keV=float(a.mean()),rms_keV=float(np.sqrt(np.mean(a*a))),
                p95_abs_keV=float(np.quantile(np.abs(a),.95))))
        axes[i].set(title=key.replace("_"," "),xlabel="residual (keV)");axes[i].grid(alpha=.2)
    axes[0].legend();fig.suptitle("Accepted point events with matching physical first pair; deterministic sample; display +/-50 keV")
    fig.savefig(output/"accepted_measurement_error_layers.png",dpi=150);plt.close(fig)
    points=json.loads((study/"point_diagnostics/point_consistency.json").read_text())["summaries"]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    for gp,style in (("legacy","o--"),("ideal","s-")):
        records=[r for r in points if r["group"]==gp];x=np.arange(len(records))
        fraction=np.array([1-r["true_source_arm_coverage"]["3"] for r in records])
        n=np.array([r["events"] for r in records]);se=np.sqrt(fraction*(1-fraction)/n)
        axes[0].errorbar(x,100*fraction,yerr=300*se,fmt=style,label=gp+" (3 SE bars)")
        axes[1].plot(x,100*np.array([r["first_pair_mismatch_fraction"] for r in records]),style,label=gp)
    names=("center","x+225","x-225","y+135","y-135","z+57","z-57")
    for ax in axes:ax.set_xticks(np.arange(7),names,rotation=45);ax.grid(alpha=.2);ax.legend()
    axes[0].set(ylabel="ARM at true source >3 sigma (%)",title="Retained q(full circle)<=3 events")
    axes[1].set(ylabel="List pair differs from physical first pair (%)",title="Truth-based topology audit")
    fig.suptitle("Held-out point sources; truth only used for diagnostics, never event selection")
    fig.savefig(output/"point_source_consistency.png",dpi=150);plt.close(fig)
    cases=json.loads((study/"offline/boundary_integration.json").read_text())
    meaningful=[r for r in cases if list(r["quadrature"].values())[-1]>1e-30]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    fractions=np.array([r["fraction"] for r in meaningful])
    final=np.array([list(r["quadrature"].values())[-1] for r in meaningful])
    for key,label in (("representative","representative point"),("centroid","overlap centroid")):
        ratio=np.array([r[key] for r in meaningful])/final
        axes[0].scatter(fractions,ratio,s=8,alpha=.45,label=label)
    last_change=np.array([abs(list(r["quadrature"].values())[-1]-list(r["quadrature"].values())[-2])/
                          list(r["quadrature"].values())[-1] for r in meaningful])
    axes[0].set(xscale="log",yscale="log",xlabel="ellipse overlap fraction",ylabel="approximation / refined integral")
    axes[0].axhline(1,color="black",ls=":");axes[0].legend()
    axes[1].scatter(fractions,100*last_change,s=8,alpha=.45)
    axes[1].axhline(1,color="black",ls="--",label="1% convergence target")
    axes[1].set(xscale="log",yscale="log",xlabel="ellipse overlap fraction",ylabel="last refinement change (%)")
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Offline actual ellipse-intersection integration; zero/tiny numerical integrals excluded")
    fig.savefig(output/"boundary_integration.png",dpi=150);plt.close(fig)
    report=dict(study="compton_first_scatter_v2",gate_status=gate["status"],measurement_layers=summary,
        boundary_cases=len(cases),boundary_converged=sum(r["converged"] for r in cases),
        boundary_nonzero_cases=len(meaningful),boundary_nonzero_converged=sum(r["converged"] for r in meaningful),
        production_kernel_changed=False,production_boundary_changed=False)
    (output/"diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,default=HERE/"generated/compton_first_scatter_v2")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();plot(a.study,a.output)
if __name__=="__main__":main()
