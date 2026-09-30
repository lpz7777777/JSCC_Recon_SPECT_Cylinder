"""Scientific QA figure for the full elliptical uniform-source reconstruction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNELS=(("440_SinglePhoton","440 single photon"),
          ("440_ComptonOnly","440 Compton"),
          ("440_SinglePlusCompton","440 joint MLEM"),
          ("218_SinglePhoton_CrossTalkCorrected","218 corrected"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    args=p.parse_args()
    report=json.loads(args.report.read_text())
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for channel,label in CHANNELS:
        curve=report["channels"][channel]["z_center"]["cv_history"]
        ax.plot([x["iteration"] for x in curve],[x["cv"] for x in curve],
                marker="o",linewidth=2,label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MLEM iterations")
    ax.set_ylabel("Volume-weighted CV, ρ ≤ 0.9, |z| ≤ 30 mm")
    ax.grid(True,which="both",alpha=.25)
    ax.legend(frameon=False,fontsize=8)
    ratios=[report["channels"][name]["z_edge"]["mean_over_z_center"]
            for name,_ in CHANNELS]
    bx.bar(range(len(CHANNELS)),ratios,color=["C0","C1","C2","C3"])
    bx.axhline(1,color="black",linewidth=1,linestyle="--")
    bx.set_xticks(range(len(CHANNELS)),["440 single","440 Compton","440 joint","218 corrected"],rotation=25)
    bx.set_ylabel("Mean density at 45 < |z| ≤ 60 / center")
    bx.set_ylim(0,max(1.3,max(ratios)*1.1))
    bx.grid(axis="y",alpha=.25)
    fig.suptitle("Uniform ellipse 500 × 300 × 120 mm, 10⁹ primaries")
    out=args.report.with_suffix(".png")
    fig.savefig(out,dpi=180)
    print(out)

if __name__=="__main__": main()
