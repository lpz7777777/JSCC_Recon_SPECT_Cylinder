"""Stream Geant4 List CSV files and render comparable E1/E2 histograms."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def accumulate(paths, edges, chunk_rows):
    hist = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.int64)
    sums = []
    total = 0
    for path in paths:
        for chunk in pd.read_csv(
            path,
            header=None,
            usecols=[1, 3],
            names=["e1", "e2"],
            chunksize=chunk_rows,
        ):
            values = chunk.to_numpy(dtype=np.float64, copy=False)
            values = values[np.isfinite(values).all(axis=1)]
            if values.size == 0:
                continue
            hist += np.histogram2d(values[:, 0], values[:, 1], bins=(edges, edges))[0].astype(np.int64)
            sums.append(values[:, 0] + values[:, 1])
            total += values.shape[0]
    sum_values = np.concatenate(sums) if sums else np.empty(0, dtype=np.float64)
    return hist, sum_values, total


def draw_histograms(results, edges, output):
    positive = np.concatenate([hist[hist > 0] for hist, _, _ in results.values()])
    vmin = max(1.0, float(positive.min()))
    vmax = float(positive.max())
    fig, axes = plt.subplots(1, len(results), figsize=(14, 5.5), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    mesh = None
    for ax, (name, (hist, _, total)) in zip(axes, results.items()):
        display = np.ma.masked_where(hist.T <= 0, hist.T)
        mesh = ax.pcolormesh(
            edges * 1000.0,
            edges * 1000.0,
            display,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="magma",
        )
        ax.set_title(f"{name}\nN={total:,}")
        ax.set_xlabel("E1 (keV)")
        ax.set_ylabel("E2 (keV)")
        ax.set_xlim(edges[0] * 1000, edges[-1] * 1000)
        ax.set_ylim(edges[0] * 1000, edges[-1] * 1000)
        ax.set_aspect("equal")
    fig.colorbar(mesh, ax=axes, label="event count per bin (log scale)")
    fig.suptitle("Geant4 Compton List: E1-E2 joint histogram")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def draw_normalized(results, edges, output):
    fig, axes = plt.subplots(1, len(results), figsize=(14, 5.5), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    vmax = max(float((hist / max(total, 1)).max()) for hist, _, total in results.values())
    for ax, (name, (hist, _, total)) in zip(axes, results.items()):
        display = np.ma.masked_where(hist.T <= 0, hist.T / max(total, 1))
        mesh = ax.pcolormesh(
            edges * 1000.0, edges * 1000.0, display, shading="auto",
            vmin=0, vmax=vmax, cmap="viridis",
        )
        ax.set_title(name)
        ax.set_xlabel("E1 (keV)"); ax.set_ylabel("E2 (keV)")
        ax.set_xlim(edges[0] * 1000, edges[-1] * 1000)
        ax.set_ylim(edges[0] * 1000, edges[-1] * 1000)
        ax.set_aspect("equal")
    fig.colorbar(mesh, ax=axes, label="fraction of List rows per bin")
    fig.suptitle("Geant4 Compton List: normalized E1-E2 distribution")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensitivity-list", type=Path,
        default=Path("List/440keV_RotateNum20_Geant4JSCC/ComptonSensitivity_UniformFullFOV_5e10/List_UniformFullFOV_440keV.csv"),
    )
    parser.add_argument(
        "--contrast-list", type=Path,
        default=Path("List/218-440keV_RotateNum20_Geant4JSCC/List_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e9"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Results/ListDiagnostics/E1_E2_1e9_vs_SensiD"))
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--max-energy-mev", type=float, default=0.45)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sensitivity_paths = sorted(args.sensitivity_list.resolve().glob("*.csv")) if args.sensitivity_list.is_dir() else [args.sensitivity_list.resolve()]
    contrast_paths = sorted(args.contrast_list.resolve().glob("*.csv")) if args.contrast_list.is_dir() else [args.contrast_list.resolve()]
    edges = np.linspace(0.0, args.max_energy_mev, args.bins + 1)
    results = {}
    summary = {}
    for name, paths in (("Uniform 440 List for Sensi_d", sensitivity_paths), ("Contrast phantom 1e9 mixed List", contrast_paths)):
        hist, sums, total = accumulate(paths, edges, args.chunk_rows)
        results[name] = (hist, sums, total)
        summary[name] = {
            "files": [str(path) for path in paths],
            "rows": total,
            "e1_mev": {"min": float(np.min(hist * 0 + edges[0])) if total == 0 else None},
            "sum_e1_e2_mev": {
                "min": float(sums.min()) if total else None,
                "max": float(sums.max()) if total else None,
                "mean": float(sums.mean()) if total else None,
                "fraction_sum_lt_0_329": float(np.mean(sums < 0.329)) if total else None,
                "fraction_sum_ge_0_329": float(np.mean(sums >= 0.329)) if total else None,
            },
        }
    draw_histograms(results, edges, output / "E1_E2_joint_histogram_log_counts.png")
    draw_normalized(results, edges, output / "E1_E2_joint_histogram_normalized.png")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
