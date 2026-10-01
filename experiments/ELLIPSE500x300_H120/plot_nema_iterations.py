"""Six-channel NEMA H60 galleries and 200-frame spatial metrics.

Uses the experiment's explicit dual-energy 3D truth, not the skill's legacy
all-hot cylindrical NEMA catalog. No XY crop or smoothing; gray_r. MIP
excludes three axial layers per end by default. Gallery
normalization is fixed across iterations using each channel's final background.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_nema_result import digest, xy_interpolator, weighted_mean
from mip_projection import axial_mip, axial_selection, DEFAULT_TRIM_LAYERS

HERE = Path(__file__).resolve().parent
CHANNELS = (
    ("440_SinglePhoton", "440 single", 440),
    ("440_ComptonOnly", "440 Compton", 440),
    ("440_SinglePlusCompton", "440 JSCC", 440),
    ("218_SinglePhoton_CrossTalkCorrected", "218 corrected", 218),
    ("440SinglePlus218Single", "440 single + 218", "sum"),
    ("440SingleComptonPlus218Single", "440 JSCC + 218", "sum"),
)
SELECTED = (100, 500, 1000, 2000, 5000, 10000)
DIAMETERS = (10, 13, 17, 22, 28, 37)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("result")
    p.add_argument("--mip-trim-layers",type=int,default=DEFAULT_TRIM_LAYERS,
                   help="Exclude this many z layers per end from MIP only (default 3)")
    args = p.parse_args()
    if "/" in args.result or "\\" in args.result:
        p.error("Result must be one directory name")
    data = HERE / "generated/RemoteResults" / args.result
    out = HERE / "reports/NEMA_Body_H60" / args.result
    out.mkdir(parents=True, exist_ok=True)
    integrity_path = HERE / "reports" / f"{args.result}_integrity.json"
    integrity = json.loads(integrity_path.read_text())
    run = json.loads((data / "run_manifest.json").read_text())
    if (run["dataset"] != "NEMA_Body_H60" or run["iterations"] != 10000
            or run["save_step"] != 50 or integrity["accepted_compton_events"] <= 0
            or integrity["accepted_compton_events"] != run["accepted_compton_events"]):
        raise ValueError("Unexpected NEMA run provenance")
    level = run["count_level"]
    collection_path = HERE / "generated/collections" / f"NEMA_Body_H60_{level}.json"
    if digest(collection_path) != integrity["collection_sha256"]:
        raise ValueError("Collection hash mismatch")
    collection = json.loads(collection_path.read_text())
    expected_total = {"1e9":10**9,"5e9":5*10**9,"1e10":10**10}[level]
    if sum(collection["primary_counts"]) != expected_total or collection["level"] != level:
        raise ValueError("Actual emitted photon total differs from dose label")
    truth_path = HERE / "generated/NEMA_Body_H60/truth_3mm.npz"
    manifest = json.loads((HERE / "reports/NEMA_Body_H60/manifest.json").read_text())
    if digest(truth_path) != manifest["truth_sha256"]:
        raise ValueError("Source truth hash mismatch")
    with np.load(truth_path) as t:
        x, y, z = (t[f"{a}_mm"].copy() for a in "xyz")
        truth = {e: t[f"activity_{e}_zyx"].copy() for e in (218, 440)}
        spheres = {d: t[f"sphere_{d}_fraction_zyx"].copy() for d in DIAMETERS}
        bg_fraction = np.maximum(t["body_fraction_zyx"]-t["lung_fraction_zyx"]-sum(spheres.values()), 0)
    # Gamma-density composite: source concentrations times the frozen yields.
    truth["sum"] = (truth[218]*.114 + truth[440]*.259)/(.114+.259)
    _, mip_policy = axial_selection(z,args.mip_trim_layers)
    with np.load(HERE / "generated/Geometry/geometry.npz") as g:
        coords = g["coordinates_mm"].copy()
        active = g["active_indices"].copy()
        volumes = (g["cell_volume_mm3"]*g["ellipse_fraction"])[active]
    if not np.allclose(coords[::3301, 2], z):
        raise ValueError("Truth/reconstruction z centers differ")
    sampler = xy_interpolator(coords[:3301, :2], x, y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ellipse = (xx/250)**2+(yy/150)**2 <= 1
    background = (bg_fraction >= .99) & (np.abs(z[:, None, None]) <= 25.5)
    masks = {}
    for d in DIAMETERS:
        cx, cy, cz = next(s["center_mm"] for s in manifest["spheres"] if s["diameter_mm"] == d)
        masks[d] = ((xx-cx)**2+(yy-cy)**2 <= (d/2+25)**2)[None] & (np.abs(z[:, None, None]-cz) <= d/2+3) & (bg_fraction >= .99)
    extent = (x[0]-1.5, x[-1]+1.5, y[0]-1.5, y[-1]+1.5)
    slices = (20, 10, 29, 0, 39)
    views = {s: [] for s in slices}
    final_views = []
    selected_volumes = {}
    rows, summary = [], {}
    primaries = {218: collection["primary_counts"][0],
                 440: collection["primary_counts"][1], "sum": expected_total}
    expected_bg = {e: primaries[e]/float(truth[e].sum(dtype=np.float64)*27) for e in (218,440)}
    expected_bg["sum"] = expected_bg[218]+expected_bg[440]
    outside_z = np.abs(coords[active,2]) > 30
    for channel, title, energy in CHANNELS:
        info = next(r for r in integrity["outputs"] if r["channel"] == channel)
        path = data / f"Image_{channel}_history.float32"
        if path.stat().st_size != 200*82040*4 or digest(path) != info["sha256"]["history"]:
            raise ValueError(f"History checksum/size mismatch: {channel}")
        hist = np.memmap(path, mode="r", dtype="<f4", shape=(200, 82040))
        full = np.zeros(132040, dtype=np.float32)
        full[active] = hist[-1]
        final = np.maximum(sampler(full.reshape(40, 3301)), 0)
        final[:, ~ellipse] = 0
        normalizer = float(final[background].mean())
        if normalizer <= 0:
            raise ValueError("Nonpositive final background")
        selected = {}
        for frame, values in enumerate(hist):
            iteration = (frame+1)*50
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid history values")
            full.fill(0)
            full[active] = values
            image = np.maximum(sampler(full.reshape(40, 3301)), 0)
            image[:, ~ellipse] = 0
            bg = image[background]
            mean, std = float(bg.mean()), float(bg.std(ddof=1))
            fit = float(np.sum(image*truth[energy], dtype=np.float64)/np.sum(image*image, dtype=np.float64))
            error = float(np.linalg.norm((fit*image-truth[energy]).ravel())/np.linalg.norm(truth[energy].ravel()))
            recovery = float(np.dot(values.astype(np.float64), volumes)/primaries[energy])
            for d in DIAMETERS:
                if energy != "sum" and d not in ((10, 17, 28) if energy == 218 else (13, 22, 37)):
                    continue
                local = image[masks[d]]
                local_mean, local_std = float(local.mean()), float(local.std(ddof=1))
                hot = weighted_mean(image, spheres[d])
                expected_ratio = 10 if energy != "sum" else 10*(.114 if d in (10,17,28) else .259)/(.114+.259)
                sampled_crc = (weighted_mean(truth[energy], spheres[d])-1)/(expected_ratio-1)
                rows.append(dict(channel=channel, iteration=iteration, diameter_mm=d,
                    background_mean=mean, background_cv=std/mean,
                    background_bias=mean/expected_bg[energy]-1,
                    local_background_mean=local_mean, local_background_cv=local_std/local_mean,
                    hot_mean=hot, crc=(hot/local_mean-1)/(expected_ratio-1),
                    cnr=(hot-local_mean)/local_std, expected_hot_ratio=expected_ratio,
                    sampled_truth_crc=sampled_crc, truth_fit_scalar=fit, truth_fit_nrmse=error,
                    emitted_photon_integral_recovery=recovery))
            if iteration in SELECTED:
                selected[iteration] = image/normalizer
        selected_volumes[channel] = selected
        summary[channel] = {"fixed_display_background": normalizer,
            "final_background_cv": std/mean, "final_truth_fit_nrmse": error,
            "expected_background_emitted_density": expected_bg[energy],
            "final_background_bias": mean/expected_bg[energy]-1,
            "final_mass_fraction_outside_source_z": float(np.dot(hist[-1][outside_z].astype(np.float64),volumes[outside_z])/np.dot(hist[-1].astype(np.float64),volumes)),
            "final_emitted_photon_integral_recovery": recovery,
            "history_sha256": info["sha256"]["history"]}
        for s in slices:
            views[s].append((title, energy, [selected[i][s] for i in SELECTED]))
        final_views.append((title, energy, final/normalizer))
        print("ANALYZED", channel, flush=True)
    with (out / "iteration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Fixed color range 0..10 across every channel, every displayed iteration.
    for s, channel_rows in views.items():
        fig, axes = plt.subplots(6, 7, figsize=(22, 15), layout="constrained")
        for r, (title, energy, images) in enumerate(channel_rows):
            for c, image in enumerate([truth[energy][s]]+images):
                ax=axes[r,c]
                im=ax.imshow(image, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=10, interpolation="nearest")
                ax.set(xlim=(-252,252), ylim=(-150,150), aspect="equal")
                ax.set_title("truth" if c==0 else f"iter {SELECTED[c-1]}", fontsize=10)
                if c==0: ax.set_ylabel(title)
                ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes, shrink=.6, label="relative gamma density; fixed final-background normalization")
        fig.suptitle(f"NEMA H60 {level}: six channels, z={z[s]:+.1f} mm; no smoothing, full ellipse FOV")
        fig.savefig(out/f"iterations_z{s:02d}.png", dpi=115); plt.close(fig)
    fig,axes=plt.subplots(2,7,figsize=(21,7),layout="constrained")
    for r,(channel,title,energy) in enumerate((CHANNELS[3],CHANNELS[2])):
        panels=[truth[energy][20]]+[selected_volumes[channel][i][20] for i in SELECTED]
        for c,panel in enumerate(panels):
            ax=axes[r,c]
            im=ax.imshow(panel,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
            ax.set(xlim=(-160,160),ylim=(-120,120),aspect="equal",title="truth" if c==0 else f"iter {SELECTED[c-1]}")
            if c==0: ax.set_ylabel(title)
            ax.tick_params(labelsize=7)
    fig.colorbar(im,ax=axes,shrink=.65,label="relative gamma density")
    fig.suptitle("NEMA central detail: display window x=±160, y=±120 mm; full arrays retained; sigma=0")
    fig.savefig(out/"central_detail.png",dpi=135);plt.close(fig)
    # All six final channels: truth and reconstruction in four spatial views.
    ix, iy, iz = int(np.argmin(abs(x))), int(np.argmin(abs(y))), 20
    fig, axes=plt.subplots(6,8,figsize=(23,17),layout="constrained")
    for r,(title,energy,image) in enumerate(final_views):
        for kind in range(4):
            for col,volume in enumerate((truth[energy],image)):
                if kind==0: panel,ex=volume[iz],extent
                elif kind==1: panel,ex=volume[:,iy,:],(x[0]-1.5,x[-1]+1.5,-60,60)
                elif kind==2: panel,ex=volume[:,:,ix],(y[0]-1.5,y[-1]+1.5,-60,60)
                else: panel,ex=axial_mip(volume,z,args.mip_trim_layers),extent
                ax=axes[r,2*kind+col]
                im=ax.imshow(panel,origin="lower",extent=ex,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
                ax.set_title(("axial","coronal","sagittal",f"MIP trim {args.mip_trim_layers}/end")[kind]+(" truth" if col==0 else " recon"),fontsize=9)
                if kind==0 and col==0: ax.set_ylabel(title)
                ax.tick_params(labelsize=6)
    fig.colorbar(im,ax=axes,shrink=.6,label="relative gamma density")
    fig.suptitle(f"NEMA H60: 10000 iterations, full XY ellipse; MIP z slab {mip_policy['retained_slab_bounds_mm']} mm; sigma=0")
    fig.savefig(out/"final_multiplanar.png",dpi=115);plt.close(fig)
    fig,axes=plt.subplots(6,3,figsize=(15,19),layout="constrained")
    for r,(channel,title,energy) in enumerate(CHANNELS):
        ds=sorted({row["diameter_mm"] for row in rows if row["channel"]==channel})
        for d in ds:
            rr=[row for row in rows if row["channel"]==channel and row["diameter_mm"]==d]
            for c,key in enumerate(("crc","cnr")):
                axes[r,c].plot([a["iteration"] for a in rr],[a[key] for a in rr],label=f"{d} mm")
        rr=[row for row in rows if row["channel"]==channel and row["diameter_mm"]==ds[0]]
        axes[r,2].plot([a["iteration"] for a in rr],[a["background_cv"] for a in rr])
        for c in range(3):
            axes[r,c].set(xlabel="MLEM iteration",ylabel=("CRC","CNR","background CV")[c],title=title)
            axes[r,c].grid(alpha=.25)
        axes[r,0].axhline(1,color="black",ls=":",lw=.8)
        axes[r,0].legend(fontsize=7,ncol=3)
    fig.savefig(out/"crc_cnr_cv_iterations.png",dpi=130);plt.close(fig)
    report={"result":args.result,"count_level":level,"primary_counts_218_440_other":collection["primary_counts"],
        "truth_sha256":digest(truth_path),"integrity_sha256":digest(integrity_path),
        "selected_iterations":SELECTED,"all_metric_iterations":list(range(50,10001,50)),"mip_policy":mip_policy,
        "method":"3-mm XY barycentric interpolation; no smoothing/XY crop; MIP-only axial trim per mip_policy; ellipse mask; gray_r; fixed 0..10 color range; each channel uses one final-background scalar for all gallery frames; least-squares truth-fit scalar used only for NRMSE metrics, not gallery rendering",
        "composite_truth":"(.114*activity218+.259*activity440)/.373; gamma density, not parent Ac activity",
        "roi":"volume-fraction sphere means; local pure background within sphere radius+25mm in XY and radius+3mm in z; global pure background |z|<=25.5mm; spatial std ddof=1; CRC denominator=expected hot/background ratio-1",
        "summary":summary,"final_spheres":[row for row in rows if row["iteration"]==10000]}
    (out/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(data/"nema_gallery_truth.npz",**{str(k):v for k,v in truth.items()})
    (out/"README.md").write_text(f"# NEMA H60: verified six-channel {level} result\n\n"
        "See analysis.json and iteration_metrics.csv for reproducible methods and 200-frame metrics. "
        "All figures retain the ellipse FOV, use gray_r (white low, black high), sigma=0 and no edge crop. "
        "Color limits are fixed at 0..10. Background normalization is fixed per channel across iterations; "
        "values above 10 saturate visually but remain unchanged in metrics. Composite truth includes gamma yields, "
        "and is not parent 225Ac activity.\n\n"
        f"Only MIP excludes {args.mip_trim_layers} z layers per end; retained slab {mip_policy['retained_slab_bounds_mm']} mm. Quantitative metrics and other views use the complete reconstruction.\n\n"
        "![Six-channel iteration gallery](iterations_z20.png)\n\n"
        "![Final multiplanar truth comparison](final_multiplanar.png)\n\n"
        "![Central two-energy detail](central_detail.png)\n\n"
        "![CRC CNR CV curves](crc_cnr_cv_iterations.png)\n")
    print(out)

if __name__ == "__main__": main()
