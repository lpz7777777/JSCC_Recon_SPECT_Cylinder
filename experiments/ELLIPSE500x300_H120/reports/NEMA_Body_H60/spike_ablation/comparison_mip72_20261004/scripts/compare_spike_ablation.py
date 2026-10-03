"""Compare completed ablation groups with the frozen 5e9 NEMA baseline.

Read-only native-grid spike metrics and matched H60 sphere metrics. Require
formal gates and complete histories; never infer missing results or smooth data.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_nema_result import digest, xy_interpolator
from mip_projection import axial_mip, axial_selection, DEFAULT_TRIM_LAYERS
from plot_nema_iterations import CHANNELS

HERE = Path(__file__).resolve().parent
REPORT = HERE / "reports/NEMA_Body_H60/spike_ablation"
BASELINE = "NEMA_Body_H60_5e9_1644876"
LABELS = {"baseline": "MLEM baseline", "bind_f010": "Boundary binding",
          "huber_weak": "Huber weak", "huber_medium": "Huber medium",
          "huber_strong": "Huber strong", "tv_medium": "Graph TV"}
COLORS = ("#343434", "#df8b22", "#2679b8", "#278b62", "#9a3f8c", "#b64339")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def volume_quantile(values, volume, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(volume[order], dtype=np.float64)
    index = np.searchsorted(cumulative, quantile*cumulative[-1], side="left")
    return float(values[order[min(index, len(order)-1)]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", required=True, choices=list(LABELS)[1:])
    parser.add_argument("--output-name", required=True, help="New report subdirectory; no slashes")
    parser.add_argument("--mip-trim-layers", type=int, default=DEFAULT_TRIM_LAYERS,
                        help="Display only: remove this many z layers per end (default: central 72 mm)")
    args = parser.parse_args()
    if "/" in args.output_name or "\\" in args.output_name or args.output_name in (".", ".."):
        parser.error("One report directory name is required")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants")
    out = REPORT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    study_path = HERE / "generated/SpikeAblation/NEMA_5e9_SPIKE_ABLATION_V1/study.json"
    study = json.loads(study_path.read_text())
    study_sha = digest(study_path)
    # Local release.json is a report wrapper with extra location/ID fields.
    # Its manifest_sha256 identifies the immutable remote release bytes.
    release_sha = json.loads((REPORT / "release.json").read_text())["manifest_sha256"]
    selected = [("baseline", BASELINE)]
    for variant in args.variants:
        gate = json.loads((REPORT / f"gates/{variant}.formal.json").read_text())
        if (not gate["passed"] or gate["study_sha256"] != study_sha or gate["variant"] != variant
                or gate["release_sha256"] != release_sha):
            raise ValueError("Unverified formal variant")
        selected.append((variant, Path(gate["result"]).name))
    if (out / "comparison.json").exists():
        old = json.loads((out / "comparison.json").read_text())
        if (old["variants"] != [v for v, _ in selected] or old["study_sha256"] != study_sha
                or old["mip_policy"]["trim_layers_each_end"] != args.mip_trim_layers):
            raise ValueError("Preserve the previous comparison; choose a new output name")
    geometry_path = HERE / "generated/Geometry/geometry.npz"
    with np.load(geometry_path) as g:
        coords = g["coordinates_mm"].copy()
        active = g["active_indices"].copy()
        fraction = g["ellipse_fraction"][active].copy()
        volumes = (g["cell_volume_mm3"]*g["ellipse_fraction"])[active]
    native_z = coords[active, 2]
    tiny = fraction < .1
    outside = abs(native_z) > 30
    inner = abs(native_z) <= 49.5
    truth_path = HERE / "generated/NEMA_Body_H60/truth_3mm.npz"
    manifest = json.loads((HERE / "reports/NEMA_Body_H60/manifest.json").read_text())
    if digest(truth_path) != manifest["truth_sha256"]:
        raise ValueError("Source truth mismatch")
    with np.load(truth_path) as t:
        x, y, z = (t[f"{axis}_mm"].copy() for axis in "xyz")
        truth = {e: t[f"activity_{e}_zyx"].copy() for e in (218, 440)}
    truth["sum"] = (.114*truth[218]+.259*truth[440])/.373
    if not np.allclose(coords[::3301, 2], z):
        raise ValueError("Truth z centers mismatch")
    _, mip_policy = axial_selection(z, args.mip_trim_layers)
    sampler = xy_interpolator(coords[:3301, :2], x, y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ellipse = (xx/250)**2+(yy/150)**2 <= 1
    provenance, channel_rows, sphere_rows, sphere_curves, curves, panels = [], [], [], [], [], {}
    reference = None
    for variant, result in selected:
        data = HERE / "generated/RemoteResults" / result
        folder = HERE / "reports/NEMA_Body_H60" / result
        ip = HERE / "reports" / f"{result}_integrity.json"
        integrity = json.loads(ip.read_text())
        run = json.loads((data / "run_manifest.json").read_text())
        analysis = json.loads((folder / "analysis.json").read_text())
        with (folder / "iteration_metrics.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        sphere_curves.extend({"variant": variant, **row} for row in metrics)
        if (integrity["primaries"] != 5*10**9 or integrity["views"] != 20
                or integrity["worker_count"] != 200 or run["accepted_compton_events"] != 484936
                or run["iterations"] != 10000 or run["save_step"] != 50
                or analysis["truth_sha256"] != manifest["truth_sha256"]
                or analysis["integrity_sha256"] != digest(ip)
                or run["geometry_sha256"] != digest(geometry_path)
                or run["input_sha256"] != study["baseline_input_sha256"]
                or run["factor_manifest_sha256"] != study["baseline_factor_manifest_sha256"]
                or run["sensi_d_sha256"] != study["baseline_sensi_d_sha256"]):
            raise ValueError("Result provenance mismatch")
        if variant == "baseline" and digest(data / "run_manifest.json") != study["baseline_run_manifest_sha256"]:
            raise ValueError("Frozen baseline run manifest changed")
        signature = {k: run[k] for k in ("geometry_sha256", "sensi_d_sha256", "input_sha256", "factor_manifest_sha256")}
        signature["roi"] = analysis["roi"]
        signature["selected_iterations"] = analysis["selected_iterations"]
        if reference is not None and reference != signature:
            raise ValueError("Input, Factors, geometry, sensitivity, ROI or iterations differ")
        reference = signature
        provenance.append({"variant": variant, "result": result, "run_manifest_sha256": digest(data / "run_manifest.json"),
                           "integrity_sha256": digest(ip), "analysis_sha256": digest(folder / "analysis.json"),
                           "metrics_sha256": digest(folder / "iteration_metrics.csv")})
        panels[variant] = {}
        for channel, _, _ in CHANNELS:
            path = data / f"Image_{channel}_history.float32"
            full_path = data / f"Image_{channel}_full.float32"
            expected = next(row["sha256"] for row in integrity["outputs"] if row["channel"] == channel)
            if (path.stat().st_size != 200*len(active)*4 or digest(path) != expected["history"]
                    or digest(full_path) != expected["full"] or analysis["summary"][channel]["history_sha256"] != expected["history"]):
                raise ValueError("Image/history checksum mismatch")
            hist = np.memmap(path, mode="r", dtype="<f4", shape=(200, len(active)))
            final = np.fromfile(full_path, dtype="<f4")
            if not np.isfinite(hist).all() or np.min(hist) < 0 or not np.array_equal(final[active], hist[-1]):
                raise ValueError("Invalid history or inconsistent final frame")
            summary = analysis["summary"][channel]
            bg = summary["fixed_display_background"]
            native = hist[-1].astype(np.float64)
            total = float(native @ volumes)
            peak = int(native.argmax())
            row = {"variant": variant, "result": result, "channel": channel,
                   "peak_over_final_background": float(native[peak]/bg),
                   "peak_over_expected_background": float(native[peak]/summary["expected_background_emitted_density"]),
                   "inner_z_peak_over_final_background": float(native[inner].max()/bg),
                   "volume_weighted_p999_over_final_background": volume_quantile(native, volumes, .999)/bg,
                   "tiny_cells_integral_fraction": float(native[tiny] @ volumes[tiny]/total),
                   "outside_source_z_integral_fraction": float(native[outside] @ volumes[outside]/total),
                   "background_cv": summary["final_background_cv"],
                   "background_bias": summary["final_background_bias"],
                   "emitted_photon_integral_recovery": summary["final_emitted_photon_integral_recovery"],
                   "peak_x_mm": coords[active[peak], 0], "peak_y_mm": coords[active[peak], 1],
                   "peak_z_mm": coords[active[peak], 2], "peak_ellipse_fraction": fraction[peak]}
            channel_rows.append(row)
            for frame, values in enumerate(hist):
                values = values.astype(np.float64)
                mass = float(values @ volumes)
                if mass <= 0:
                    raise ValueError("Nonpositive reconstructed integral")
                iteration = (frame+1)*50
                match = next(r for r in metrics if r["channel"] == channel and int(r["iteration"]) == iteration)
                curves.append({"variant": variant, "channel": channel, "iteration": iteration,
                               "peak_over_final_background": float(values.max()/bg),
                               "tiny_cells_integral_fraction": float(values[tiny] @ volumes[tiny]/mass),
                               "outside_source_z_integral_fraction": float(values[outside] @ volumes[outside]/mass),
                               "background_cv": float(match["background_cv"])})
            image = np.maximum(sampler(final.reshape(40, 3301)), 0)
            image[:, ~ellipse] = 0
            panels[variant][channel] = image
        sphere_rows.extend({"variant": variant, "result": result, **r} for r in analysis["final_spheres"])
        print("VERIFIED_COMPARE", variant, flush=True)
    write_csv(out / "final_channels.csv", channel_rows)
    write_csv(out / "final_spheres.csv", sphere_rows)
    write_csv(out / "native_iteration_metrics.csv", curves)
    extent = (x[0]-1.5, x[-1]+1.5, y[0]-1.5, y[-1]+1.5)
    for kind in ("center", "mip"):
        for scale in ("final_bg", "emitted_density"):
            fig, axes = plt.subplots(6, len(selected)+1, figsize=(4*(len(selected)+1), 14), layout="constrained")
            for r, (channel, title, energy) in enumerate(CHANNELS):
                volumes_to_show = [truth[energy]]
                for variant, _ in selected:
                    # Expected/final background relationship comes from the matched analysis.
                    index = next(i for i, p in enumerate(provenance) if p["variant"] == variant)
                    a = json.loads((HERE / "reports/NEMA_Body_H60" / provenance[index]["result"] / "analysis.json").read_text())
                    s = a["summary"][channel]
                    normalizer = s["fixed_display_background"] if scale == "final_bg" else s["expected_background_emitted_density"]
                    volumes_to_show.append(panels[variant][channel]/normalizer)
                for c, volume in enumerate(volumes_to_show):
                    panel = volume[20] if kind == "center" else axial_mip(volume, z, args.mip_trim_layers)
                    im = axes[r, c].imshow(panel, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=10, interpolation="nearest")
                    axes[r, c].set(xlim=(-252, 252), ylim=(-150, 150), aspect="equal")
                    if r == 0:
                        axes[r, c].set_title("3D truth" if c == 0 else LABELS[selected[c-1][0]])
                    if c == 0:
                        axes[r, c].set_ylabel(title)
                    axes[r, c].tick_params(labelsize=7)
            fig.colorbar(im, ax=axes, shrink=.6, label="Relative gamma density; fixed 0..10")
            lo, hi = mip_policy["retained_slab_bounds_mm"]
            height = mip_policy["retained_slab_height_mm"]
            location = ("z=+1.5 mm single slice" if kind == "center" else
                        f"Axial MIP, central {height:g} mm, z slab [{lo:g},{hi:g}] mm")
            scaling = "one final background scalar per method/channel" if scale == "final_bg" else "same expected emitted background per channel"
            fig.suptitle(f"NEMA 5e9, iteration 10000: {location}\n{scaling}; full ellipse; no smoothing")
            fig.savefig(out / f"{kind}_{scale}.png", dpi=130)
            plt.close(fig)
    key_channels = (CHANNELS[1], CHANNELS[2], CHANNELS[3])
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), layout="constrained")
    for r, (channel, title, _) in enumerate(key_channels):
        for i, (variant, _) in enumerate(selected):
            rows = [row for row in curves if row["channel"] == channel and row["variant"] == variant]
            for c, key in enumerate(("peak_over_final_background", "tiny_cells_integral_fraction", "background_cv", "outside_source_z_integral_fraction")):
                values = [row[key]*(100 if c in (1, 3) else 1) for row in rows]
                axes[r, c].plot([row["iteration"] for row in rows], values, label=LABELS[variant], color=COLORS[i])
        for c, ylabel in enumerate(("Native peak / final BG", "Tiny-cell integral (%)", "Pure-background CV", "Integral at |z|>30 mm (%)")):
            axes[r, c].set(xlabel="Iteration", ylabel=ylabel, title=title)
            axes[r, c].grid(alpha=.2)
            axes[r, c].legend(fontsize=8)
        axes[r, 0].set_yscale("log")
    fig.suptitle("Completed ablations: raw full-40-layer spike, noise and leakage metrics")
    fig.savefig(out / "spike_noise_leakage.png", dpi=140)
    plt.close(fig)
    for key in ("crc", "cnr"):
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained")
        for r, (channel, title, energy) in enumerate(key_channels):
            diameters = (10, 17, 28) if energy == 218 else (13, 22, 37)
            for c, diameter in enumerate(diameters):
                for i, (variant, _) in enumerate(selected):
                    rows = [row for row in sphere_curves if row["variant"] == variant and row["channel"] == channel
                            and int(row["diameter_mm"]) == diameter]
                    axes[r, c].plot([int(row["iteration"]) for row in rows], [float(row[key]) for row in rows],
                                    color=COLORS[i], label=LABELS[variant])
                axes[r, c].set(xlabel="Iteration", ylabel=key.upper(), title=f"{title}: {diameter} mm")
                axes[r, c].grid(alpha=.2)
                axes[r, c].legend(fontsize=8)
                if key == "crc":
                    axes[r, c].axhline(1, color="gray", lw=.8, ls=":")
        fig.suptitle(f"Completed ablations: matched 3D hot-sphere {key.upper()}, sigma=0")
        fig.savefig(out / f"matched_{key}.png", dpi=140)
        plt.close(fig)
    metadata = {"status": "partial_completed_groups_only", "variants": [v for v, _ in selected],
                "study_sha256": study_sha, "truth_sha256": digest(truth_path),
                "geometry_sha256": digest(geometry_path), "script_sha256": digest(Path(__file__)),
                "provenance": provenance, "mip_policy": mip_policy,
                "roi": reference["roi"], "tiny_cell_volume_fraction": float(volumes[tiny].sum()/volumes.sum()),
                "display": f"gray_r, full ellipse XY, sigma=0, fixed 0..10. MIP trims {args.mip_trim_layers} layers/end; all quantitative metrics use all 40 layers.",
                "inner_z_metric_center_range_mm": [-49.5, 49.5],
                "scales": "final_bg: each method/channel's fixed final pure-background scalar. emitted_density: identical source-primary-based expected background per channel. Neither rescales truth.",
                "native_metrics": "Native active polar density and cell_volume*ellipse_fraction integrals; P99.9 weighted by physical cell volume. CRC/CNR/CV use the existing matched H60 3D spherical ROIs, no smoothing."}
    (out / "comparison.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    scripts = out / "scripts"
    scripts.mkdir(exist_ok=True)
    shutil.copy2(__file__, scripts / Path(__file__).name)
    for name in ("plot_nema_iterations.py", "analyze_nema_result.py", "mip_projection.py"):
        shutil.copy2(HERE / name, scripts / name)
    print(out, flush=True)


if __name__ == "__main__":
    main()
