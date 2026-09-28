"""Render geometry and XCAT source QA panels (not reconstructed images)."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import generate


def render(xcat, output):
    cfg = json.loads((HERE / "config.json").read_text())
    coords, volumes, fractions, active, *_ = generate(cfg)
    truth = np.load(xcat / "truth_3mm.npz")
    x, y = truth["x_mm"], truth["y_mm"]
    fr, bi = truth["fr_zyx"], truth["bi_zyx"]
    kidney = truth["kidney_fraction_zyx"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    t = np.linspace(0, 2*np.pi, 1000)
    ax[0,0].scatter(coords[:3301,0], coords[:3301,1], c=fractions[:3301],
                    s=2, cmap="viridis", vmin=0, vmax=1)
    ax[0,0].plot(250*np.cos(t),150*np.sin(t),color="red",lw=1)
    ax[0,0].set(title="Circular grid and ellipse overlap",xlabel="x (mm)",ylabel="y (mm)",
                aspect="equal",xlim=(-260,260),ylim=(-260,260))
    for target, data, title in ((ax[0,1],fr[20],"218-keV axial truth, z=1.5 mm"),
                                (ax[0,2],bi[20],"440-keV axial truth, z=1.5 mm"),
                                (ax[1,0],kidney[20],"Kidney fraction, z=1.5 mm"),
                                (ax[1,1],fr.sum(axis=1),"218-keV coronal integral"),
                                (ax[1,2],bi.sum(axis=1),"440-keV coronal integral")):
        extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5) if data.shape[0]==100 else (x[0]-1.5,x[-1]+1.5,-60,60)
        target.imshow(data,origin="lower",extent=extent,aspect="auto",cmap="magma")
        target.set(title=title,xlabel="x (mm)",ylabel="y/z (mm)")
    fig.suptitle("ELLIPSE500x300_H120 setup QA (source truth; no reconstruction)")
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--xcat",type=Path,default=HERE/"generated/XCAT_1e9_fullx")
    p.add_argument("--output",type=Path,default=HERE/"generated/setup_qa.png")
    a=p.parse_args()
    render(a.xcat,a.output)
