"""Plot volume-weighted uniform-source CV from a verified formal result."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    args=p.parse_args()
    data=json.loads(args.report.read_text())
    labels={"440_SinglePhoton":"440 single photon",
            "440_ComptonOnly":"440 Compton",
            "440_SinglePlusCompton":"440 joint MLEM",
            "218_SinglePhoton_CrossTalkCorrected":"218 corrected"}
    fig,ax=plt.subplots(figsize=(8,5),layout="constrained")
    for name,label in labels.items():
        curve=data["channels"][name]["center"]["cv_history"]
        ax.plot([r["iteration"] for r in curve],[r["cv"] for r in curve],
                marker="o",linewidth=2,label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MLEM iterations")
    ax.set_ylabel("Volume-weighted CV in r ≤ 135 mm, |z| ≤ 30 mm")
    ax.set_title("Uniform cylinder, new detector distance, 10⁹ primaries")
    ax.grid(True,which="both",alpha=.25)
    ax.legend(frameon=False)
    out=args.report.with_name(args.report.stem+".png")
    fig.savefig(out,dpi=180)
    print(out)

if __name__=="__main__": main()
