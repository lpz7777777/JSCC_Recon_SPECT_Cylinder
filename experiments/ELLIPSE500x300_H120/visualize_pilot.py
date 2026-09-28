"""Plot measured 1e8-photon monoenergetic pilot efficiencies by detector layer."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(report_path,output):
    data=json.loads(report_path.read_text())
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
    for name,color in (("A218","tab:blue"),("A440","tab:orange"),
                       ("C440to218","tab:green")):
        layer=data["responses"][name]["layers"]
        x=np.array([r["normal_distance_mm"] for r in layer])
        y=np.array([r["measured_efficiency"] for r in layer])
        err=np.array([r["poisson_relative_se"] for r in layer])*y
        ax.errorbar(x,y,yerr=err,fmt="o-",capsize=2,label=name,color=color)
    ax.set(xlabel="Detector crystal normal distance from FOV centre (mm)",
           ylabel="Measured window counts per emitted photon",
           title="Independent 1e8-photon monoenergetic pilot: four layers")
    ax.grid(alpha=.25)
    ax.set_xticks([300,330,360,390])
    ax.legend()
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    plt.close(fig)
    print(output)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,default=Path(__file__).with_name("reports")/"pilot_layer_counts.json")
    p.add_argument("--output",type=Path,default=Path(__file__).with_name("generated")/"pilot_counts.png")
    a=p.parse_args()
    render(a.report,a.output)
