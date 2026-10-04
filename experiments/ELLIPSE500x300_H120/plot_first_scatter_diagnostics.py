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
            summary.append(dict(group=gp,residual=key,samples=len(a),mean_keV=float(a.mean()),
                rms_keV=float(np.sqrt(np.mean(a*a))),p95_abs_keV=float(np.quantile(np.abs(a),.95))))
        axes[i].set(title=key.replace("_"," "),xlabel="residual (keV)");axes[i].grid(alpha=.2)
    axes[0].legend();fig.suptitle("Point-source measurement layers; histogram display clipped at +/-100 keV; statistics use all samples")
    fig.savefig(output/"measurement_error_layers.png",dpi=150);plt.close(fig)
    cases=json.loads((study/"offline/boundary_integration.json").read_text())
    report=dict(study="compton_first_scatter_v2",gate_status=gate["status"],measurement_layers=summary,
        boundary_cases=len(cases),boundary_converged=sum(r["converged"] for r in cases),
        production_kernel_changed=False,production_boundary_changed=False)
    (output/"diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,default=HERE/"generated/compton_first_scatter_v2")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();plot(a.study,a.output)
if __name__=="__main__":main()
