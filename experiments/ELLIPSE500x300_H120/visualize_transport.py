"""Summarize independent 1e9-photon rule-phantom projections by view."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ("CircleNewDist", "EllipseUniform", "EllipseContrast")
LABELS = ("Circle, 270 mm", "Ellipse uniform, 270 mm", "Ellipse contrast, 270 mm")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summarize(folder: Path) -> dict:
    result = {}
    for energy in (218, 440):
        entries = {}
        for name in DATASETS:
            path = folder / str(energy) / f"CntStat_{name}_1e9.csv"
            data = np.loadtxt(path, delimiter=",", dtype=np.int64)
            if data.shape != (20, 10496) or np.any(data < 0):
                raise ValueError(f"Unexpected CntStat shape or value: {path}")
            count = data.sum(axis=1)
            entries[name] = {
                "input_sha256": digest(path),
                "total_detected_counts": int(count.sum()),
                "counts_by_view": count.tolist(),
                "max_over_min_view_counts": float(count.max() / count.min()),
            }
        result[str(energy)] = entries
    return result


def render(result: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    views = np.arange(20)
    for axis, energy in zip(axes, (218, 440)):
        for name, label in zip(DATASETS, LABELS):
            counts = result[str(energy)][name]["counts_by_view"]
            axis.plot(views, counts, marker="o", markersize=3, label=label)
        axis.set_ylabel(f"{energy} keV window counts / view")
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
    axes[0].set_title("1e9-primary Geant4 rule phantoms: measured projection counts")
    axes[-1].set_xlabel("Acquisition view index (0–19)")
    axes[-1].set_xticks(views)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--counts", type=Path, default=HERE / "generated/TransportCounts")
    p.add_argument("--report", type=Path, default=HERE / "reports/rule_1e9_view_counts.json")
    p.add_argument("--output", type=Path, default=HERE / "generated/rule_1e9_view_counts.png")
    a = p.parse_args()
    report = {"experiment": "ELLIPSE500x300_H120", "status": "transport counts, not reconstructed image",
              "data": summarize(a.counts)}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    render(report["data"], a.output)
    print(a.report)
    print(a.output)


if __name__ == "__main__":
    main()
