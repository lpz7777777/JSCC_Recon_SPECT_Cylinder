"""Read-only, volume-aware spike diagnostics for verified NEMA reconstructions.

Uses native polar density arrays, not interpolated/rendered pixels. A separate
verified Sensi_d copy is expected under generated/Diagnostics. No reconstruction,
clipping, smoothing, or changes to the full ellipse support are performed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_nema_result import digest

HERE = Path(__file__).resolve().parent
OUT = HERE / "reports/NEMA_Body_H60/spike_research"
RESULTS = ("NEMA_Body_H60_1e9_1643142", "NEMA_Body_H60_5e9_1644876")
CHANNELS = ("440_SinglePhoton", "440_ComptonOnly", "440_SinglePlusCompton")
LABELS = ("440 single", "440 Compton", "440 JSCC")
COLORS = ("#2465a8", "#c44032", "#258851")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geometry_path = HERE / "generated/Geometry/geometry.npz"
    sensitivity_path = HERE / "generated/Diagnostics/Sensi_d_440.float32"
    with np.load(geometry_path) as g:
        active = g["active_indices"].copy()
        coordinates = g["coordinates_mm"][active]
        fraction = g["ellipse_fraction"][active]
        volume = g["cell_volume_mm3"][active] * fraction
        full_count = len(g["coordinates_mm"])
    full_sd = np.fromfile(sensitivity_path, dtype="<f4")
    assert len(full_sd) == full_count and np.isfinite(full_sd).all()
    sd = full_sd[active] * fraction
    efficiency = sd / volume
    z = coordinates[:, 2]
    groups = {
        "fraction_lt_0p01": fraction < .01,
        "fraction_lt_0p1": fraction < .1,
        "partial_cell": fraction < 1-1e-9,
        "full_cell": fraction >= 1-1e-9,
        "outer_three_z_layers": np.abs(z) > 49.5,
        "inner_z": np.abs(z) <= 49.5,
        "full_cell_inner_z": (fraction >= 1-1e-9) & (np.abs(z) <= 49.5),
        "outside_source_z": np.abs(z) > 30,
    }
    report = {
        "geometry_sha256": digest(geometry_path),
        "sensi_d_sha256": digest(sensitivity_path),
        "script_sha256": digest(Path(__file__)),
        "method": "Unfiltered native polar densities; integrals use cell_volume * ellipse_fraction; each curve uses its channel's fixed final Cartesian background from the verified gallery. All 40 z layers remain in raw metrics.",
        "sensitivity_note": "Compton sensitivity only: sd = full_Sensi_d[active] * fraction. Unit-volume sensitivity = sd / effective_volume. JSCC total sensitivity is not analyzed here.",
        "support_volume_mm3": float(volume.sum()),
        "active_cells": len(active),
        "sensi_d_active_min_max": [float(sd.min()), float(sd.max())],
        "compton_unit_volume_sensitivity_min_max": [float(efficiency.min()), float(efficiency.max())],
        "groups": {name: {"cells": int(mask.sum()), "volume_fraction": float(volume[mask].sum()/volume.sum())} for name, mask in groups.items()},
        "results": {},
    }
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
    for dose_index, result in enumerate(RESULTS):
        data = HERE / "generated/RemoteResults" / result
        run = json.loads((data / "run_manifest.json").read_text())
        integrity_path = HERE / "reports" / f"{result}_integrity.json"
        integrity = json.loads(integrity_path.read_text())
        analysis_path = HERE / "reports/NEMA_Body_H60" / result / "analysis.json"
        analysis = json.loads(analysis_path.read_text())
        assert run["geometry_sha256"] == report["geometry_sha256"]
        assert run["sensi_d_sha256"] == report["sensi_d_sha256"]
        assert run["pixels_active"] == len(active)
        count = run["iterations"] // run["save_step"]
        iterations = np.arange(1, count+1)*run["save_step"]
        result_report = {"integrity_sha256": digest(integrity_path),
                         "analysis_sha256": digest(analysis_path), "channels": {}}
        for channel, label, color in zip(CHANNELS, LABELS, COLORS):
            history_path = data / f"Image_{channel}_history.float32"
            full_path = data / f"Image_{channel}_full.float32"
            expected = next(x["sha256"] for x in integrity["outputs"] if x["channel"] == channel)
            assert digest(history_path) == expected["history"]
            assert digest(full_path) == expected["full"]
            assert history_path.stat().st_size == count*len(active)*4
            history = np.memmap(history_path, mode="r", dtype="<f4", shape=(count, len(active)))
            assert np.isfinite(history).all() and np.min(history) >= 0
            final = np.asarray(history[-1], dtype=np.float64)
            assert np.array_equal(np.fromfile(full_path, dtype="<f4")[active], final)
            bg = analysis["summary"][channel]["fixed_display_background"]
            peak = int(final.argmax())
            integral = float(final @ volume)
            channel_report = {
                "history_sha256": expected["history"],
                "fixed_final_background": bg,
                "peak": {"active_index": peak, "full_index": int(active[peak]),
                         "coordinate_mm": coordinates[peak].tolist(),
                         "ellipse_fraction": float(fraction[peak]),
                         "effective_volume_mm3": float(volume[peak]),
                         "density": float(final[peak]), "density_over_background": float(final[peak]/bg),
                         "integral_fraction": float(final[peak]*volume[peak]/integral),
                         "compton_sensitivity": float(sd[peak]),
                         "compton_sensitivity_per_mm3": float(efficiency[peak]),
                         "compton_unit_volume_sensitivity_volume_percentile": float(100*volume[efficiency <= efficiency[peak]].sum()/volume.sum())},
                "groups": {name: {"integral_fraction": float(final[mask] @ volume[mask]/integral),
                                  "max_density_over_background": float(final[mask].max()/bg)} for name, mask in groups.items()},
            }
            for index, iteration in enumerate(iterations):
                x = np.asarray(history[index], dtype=np.float64)
                mass = float(x @ volume)
                rows.append({"result": result, "channel": channel, "iteration": int(iteration),
                             "peak_over_fixed_final_background": float(x.max()/bg),
                             "peak_inner_z_over_fixed_final_background": float(x[groups['inner_z']].max()/bg),
                             "tiny_cells_integral_fraction": float(x[groups['fraction_lt_0p1']] @ volume[groups['fraction_lt_0p1']]/mass),
                             "outside_source_z_integral_fraction": float(x[groups['outside_source_z']] @ volume[groups['outside_source_z']]/mass)})
            channel_rows = rows[-count:]
            channel_report["selected_iterations"] = [r for r in channel_rows if r["iteration"] in (50, 100, 500, 1000, 2000, 5000, 10000)]
            result_report["channels"][channel] = channel_report
            ax = axes[dose_index, 0]
            ax.semilogy(iterations, [r["peak_over_fixed_final_background"] for r in channel_rows], color=color, label=label)
            ax.semilogy(iterations, [r["peak_inner_z_over_fixed_final_background"] for r in channel_rows], color=color, ls="--", alpha=.85)
            axes[dose_index, 1].plot(iterations, [100*r["tiny_cells_integral_fraction"] for r in channel_rows], color=color, label=label)
            unique_z = np.unique(z)
            z_mass = [float(final[z == zi] @ volume[z == zi])/integral*100 for zi in unique_z]
            axes[dose_index, 2].plot(unique_z, z_mass, color=color, label=label)
        report["results"][result] = result_report
        axes[dose_index, 0].set(title=f"{run['count_level']}: maximum density / fixed final BG", xlabel="Iteration", ylabel="Ratio (log scale)")
        axes[dose_index, 0].text(.03, .03, "Solid: all layers; dashed: |z| <= 49.5 mm", transform=axes[dose_index, 0].transAxes, fontsize=9)
        axes[dose_index, 1].axhline(report["groups"]["fraction_lt_0p1"]["volume_fraction"]*100, color="black", ls=":", label="Their FOV volume fraction")
        axes[dose_index, 1].set(title="Cells with ellipse fraction < 0.1", xlabel="Iteration", ylabel="Reconstructed integral in these cells (%)")
        axes[dose_index, 2].axvspan(-30, 30, color="gray", alpha=.12, label="True source z support")
        axes[dose_index, 2].set(title="Final axial integral profile", xlabel="z (mm)", ylabel="Fraction of reconstructed integral / layer (%)")
        for ax in axes[dose_index]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("NEMA H60: native-grid spike diagnostics; no smoothing or clipping", fontsize=14)
    fig.savefig(OUT / "spike_diagnostics.png", dpi=170)
    plt.close(fig)
    with (OUT / "iteration_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "rows": len(rows), "verified_results": list(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
