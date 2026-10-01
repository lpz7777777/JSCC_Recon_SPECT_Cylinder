"""Matched 1e9/5e9 H60 spherical NEMA comparison, preserving raw data.

Require completed integrity reports and the existing 200-frame analysis. Use
explicit dual-energy truth, identical ROIs, sigma=0, full elliptical FOV.
"""
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_nema_result import digest, xy_interpolator
from plot_nema_iterations import CHANNELS, SELECTED

HERE = Path(__file__).resolve().parent
RESULTS = ("NEMA_Body_H60_1e9_1643142", "NEMA_Body_H60_5e9_1644876")


def main():
    out = HERE / "reports/NEMA_Body_H60/5e9/comparison"
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((out.parent / "comparison_plan.json").read_text())
    reports, runs, tables, provenance = [], [], [], []
    for name in RESULTS:
        folder = HERE / "reports/NEMA_Body_H60" / name
        path = folder / "analysis.json"
        a = json.loads(path.read_text())
        data = HERE / "generated/RemoteResults" / name
        run = json.loads((data / "run_manifest.json").read_text())
        ip = HERE / "reports" / f"{name}_integrity.json"
        integrity = json.loads(ip.read_text())
        if a["truth_sha256"] != plan["truth_sha256"] or a["roi"] != plan["matched_roi_definition"]:
            raise ValueError("Truth/ROI mismatch")
        if a["integrity_sha256"] != digest(ip) or run["iterations"] != 10000 or run["save_step"] != 50:
            raise ValueError("Analysis provenance mismatch")
        if name == RESULTS[0] and digest(path) != plan["baseline_analysis_sha256"]:
            raise ValueError("Frozen baseline changed")
        with (folder / "iteration_metrics.csv").open() as f:
            table = list(csv.DictReader(f))
        reports.append(a); runs.append(run); tables.append(table)
        provenance.append(dict(result=name, analysis_sha256=digest(path),
            metrics_sha256=digest(folder / "iteration_metrics.csv"), integrity_sha256=digest(ip),
            run_manifest_sha256=digest(data / "run_manifest.json"),
            actual_primary_photons=integrity["primaries"],
            accepted_compton_events=run["accepted_compton_events"]))
    for key in ("geometry_sha256", "sensi_d_sha256"):
        if runs[0][key] != runs[1][key]:
            raise ValueError(f"Unmatched {key}")
    truth_path = HERE / "generated/NEMA_Body_H60/truth_3mm.npz"
    if digest(truth_path) != plan["truth_sha256"]:
        raise ValueError("Truth hash mismatch")
    with np.load(truth_path) as t:
        x, y, z = (t[f"{axis}_mm"].copy() for axis in "xyz")
        truth = {e: t[f"activity_{e}_zyx"].copy() for e in (218, 440)}
    truth["sum"] = (.114 * truth[218] + .259 * truth[440]) / .373
    with np.load(HERE / "generated/Geometry/geometry.npz") as g:
        coords = g["coordinates_mm"].copy()
        active = g["active_indices"].copy()
        volumes = (g["cell_volume_mm3"] * g["ellipse_fraction"])[active]
    if digest(HERE / "generated/Geometry/geometry.npz") != runs[0]["geometry_sha256"]:
        raise ValueError("Local geometry hash mismatch")
    sampler = xy_interpolator(coords[:3301, :2], x, y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ellipse = (xx / 250)**2 + (yy / 150)**2 <= 1
    outside = abs(coords[active, 2]) > 30
    extent = (x[0]-1.5, x[-1]+1.5, y[0]-1.5, y[-1]+1.5)
    regional_rows, final_comparison, sphere_comparison = [], [], []
    final_panels = []
    for channel, title, energy in CHANNELS:
        panels = []
        for dose, name in enumerate(RESULTS):
            path = HERE / "generated/RemoteResults" / name / f"Image_{channel}_history.float32"
            if path.stat().st_size != 200*82040*4 or digest(path) != reports[dose]["summary"][channel]["history_sha256"]:
                raise ValueError("History size/hash mismatch")
            hist = np.memmap(path, mode="r", dtype="<f4", shape=(200,82040))
            selected = []
            full = np.zeros(132040, dtype=np.float32)
            for frame, values in enumerate(hist):
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("Invalid history")
                total = float(np.dot(values.astype(np.float64), volumes))
                leakage = float(np.dot(values[outside].astype(np.float64), volumes[outside]) / total)
                matches = [r for r in tables[dose] if r["channel"] == channel and int(r["iteration"]) == (frame+1)*50]
                if not matches:
                    raise ValueError("Missing frame metrics")
                first = matches[0]
                regional_rows.append(dict(count_level=("1e9","5e9")[dose], channel=channel,
                    iteration=(frame+1)*50, background_cv=float(first["background_cv"]),
                    background_bias=float(first["background_bias"]),
                    integral_recovery=float(first["emitted_photon_integral_recovery"]),
                    mass_fraction_outside_source_z=leakage))
                if (frame+1)*50 in SELECTED:
                    full.fill(0); full[active] = values
                    image = np.maximum(sampler(full.reshape(40,3301)), 0)
                    image[:,~ellipse] = 0
                    selected.append(image[20])
            panels.append(selected)
        # Two defensible scales: expected emitted density preserves absolute bias;
        # one final BG scalar per dose/channel shows contrast at all iterations.
        for scale, field in (("expected_density","expected_background_emitted_density"),
                             ("fixed_final_background","fixed_display_background")):
            fig, axes = plt.subplots(2,7,figsize=(21,6),layout="constrained")
            for dose in range(2):
                scalar = reports[dose]["summary"][channel][field]
                for col, panel in enumerate([truth[energy][20]]+[p/scalar for p in panels[dose]]):
                    ax = axes[dose,col]
                    im = ax.imshow(panel,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
                    ax.set(xlim=(-252,252),ylim=(-150,150),aspect="equal",
                           title="truth" if col == 0 else f"iter {SELECTED[col-1]}")
                    if col == 0: ax.set_ylabel(("1e9","5e9")[dose])
                    ax.tick_params(labelsize=7)
            fig.colorbar(im,ax=axes,shrink=.7,label="relative gamma density; 0..10")
            fig.suptitle(f"{title}: 1e9 vs 5e9, z=+1.5 mm, {scale}; full FOV, sigma=0")
            fig.savefig(out/f"{channel}_{scale}.png",dpi=110); plt.close(fig)
        final_panels.append((title, energy, [panels[d][-1]/reports[d]["summary"][channel]["expected_background_emitted_density"] for d in range(2)]))
        metrics = ("final_background_cv", "final_background_bias", "final_emitted_photon_integral_recovery", "final_mass_fraction_outside_source_z")
        for metric in metrics:
            a, b = (reports[d]["summary"][channel][metric] for d in range(2))
            final_comparison.append(dict(channel=channel,metric=metric,baseline_1e9=a,candidate_5e9=b,change=b-a))
        for a in reports[0]["final_spheres"]:
            if a["channel"] != channel: continue
            b = next(r for r in reports[1]["final_spheres"] if r["channel"] == channel and r["diameter_mm"] == a["diameter_mm"])
            sphere_comparison.append(dict(channel=channel,diameter_mm=a["diameter_mm"],
                crc_1e9=a["crc"],crc_5e9=b["crc"],cnr_1e9=a["cnr"],cnr_5e9=b["cnr"],
                sampled_truth_crc=a["sampled_truth_crc"]))
    fig, axes = plt.subplots(6,3,figsize=(12,15),layout="constrained")
    for row,(title,energy,images) in enumerate(final_panels):
        for col,panel in enumerate([truth[energy][20]]+images):
            im=axes[row,col].imshow(panel,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
            axes[row,col].set(title=("truth","1e9","5e9")[col],xlim=(-252,252),ylim=(-150,150))
            if col == 0: axes[row,col].set_ylabel(title)
    fig.colorbar(im,ax=axes,shrink=.65,label="density / expected emitted background density")
    fig.suptitle("10000 iterations; same source-normalized scale, full ellipse, sigma=0")
    fig.savefig(out/"final_expected_density.png",dpi=130); plt.close(fig)
    # Complete 200-frame curves: matched ROIs, colors per sphere, dashed 1e9.
    fig, axes = plt.subplots(6,3,figsize=(15,20),layout="constrained")
    colors = dict(zip((10,13,17,22,28,37),plt.get_cmap("tab10").colors))
    for row,(channel,title,energy) in enumerate(CHANNELS):
        for dose,table in enumerate(tables):
            ds = sorted({int(r["diameter_mm"]) for r in table if r["channel"]==channel})
            for diameter in ds:
                rr=[r for r in table if r["channel"]==channel and int(r["diameter_mm"])==diameter]
                for col,key in enumerate(("crc","cnr")):
                    axes[row,col].plot([int(r["iteration"]) for r in rr],[float(r[key]) for r in rr],
                        color=colors[diameter],ls=("--","-")[dose],label=f'{diameter} mm {("1e9","5e9")[dose]}')
            rr=[r for r in regional_rows if r["channel"]==channel and r["count_level"]==("1e9","5e9")[dose]]
            axes[row,2].plot([r["iteration"] for r in rr],[r["background_cv"] for r in rr],label=("1e9","5e9")[dose])
        for col in range(3):
            axes[row,col].set(title=title,xlabel="iteration",ylabel=("CRC","CNR","background CV")[col]); axes[row,col].grid(alpha=.25)
        axes[row,0].legend(fontsize=6,ncol=2); axes[row,2].legend(fontsize=8)
    fig.savefig(out/"matched_crc_cnr_cv.png",dpi=120); plt.close(fig)
    fig,axes=plt.subplots(6,3,figsize=(15,19),layout="constrained")
    for row,(channel,title,energy) in enumerate(CHANNELS):
        for dose in ("1e9","5e9"):
            rr=[r for r in regional_rows if r["channel"]==channel and r["count_level"]==dose]
            for col,key in enumerate(("background_bias","integral_recovery","mass_fraction_outside_source_z")):
                axes[row,col].plot([r["iteration"] for r in rr],[r[key] for r in rr],label=dose)
        for col in range(3):
            axes[row,col].set(title=title,xlabel="iteration",ylabel=("BG relative bias","photon integral recovery","mass fraction |z|>30 mm")[col])
            axes[row,col].grid(alpha=.25); axes[row,col].legend(fontsize=8)
    fig.savefig(out/"bias_integral_axial_leakage.png",dpi=120); plt.close(fig)
    for filename, rows in (("regional_metrics.csv",regional_rows),("final_channel_comparison.csv",final_comparison),("final_sphere_comparison.csv",sphere_comparison)):
        with (out/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest=dict(status="matched_comparison_complete",truth_sha256=digest(truth_path),
        geometry_sha256=runs[0]["geometry_sha256"],sensi_d_sha256=runs[0]["sensi_d_sha256"],
        provenance=provenance,roi=plan["matched_roi_definition"],selected_iterations=SELECTED,
        sigma_pixels=0,crop_pixels=0,fov_mm=[500,300,120],display_limits=[0,10],
        normalizations=["expected emitted BG density: preserves absolute bias","one final BG scalar per channel/dose fixed across iterations"],
        composite="gamma density, not parent Ac activity",final_channels=final_comparison,final_spheres=sphere_comparison,
        provenance_limit="1e9 historical run/integrity lacks Factor manifest hashes; geometry and Sensi_d hashes match, but Factor byte identity between runs cannot be independently proven from these historical manifests",
        candidate_factor_manifest_sha256=runs[1]["factor_manifest_sha256"])
    (out/"comparison.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"scripts").mkdir(exist_ok=True)
    for script in ("compare_nema_counts.py","plot_nema_iterations.py","analyze_nema_result.py"):
        shutil.copy2(HERE/script,out/"scripts"/script)
    print(out)


if __name__ == "__main__":
    main()
