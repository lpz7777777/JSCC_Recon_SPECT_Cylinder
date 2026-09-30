"""Plot detector-pattern agreement for independent Geant4 point sources."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / "reports/selected_point_factor_alignment.json"
    report = json.loads(path.read_text())
    rows = report["rows"]
    if len(rows) != 10:
        raise ValueError("Expected ten independent points")
    labels = ("center", "long −z", "long +z", "short −z", "short +z")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, energy in zip(axes, (218, 440)):
        selected = [row for row in rows if row["energy_keV"] == energy]
        if len(selected) != 5:
            raise ValueError(f"Missing {energy} keV points")
        x = np.arange(5)
        neg = [row["orientation"]["physical_negative_angle"]["cosine"] for row in selected]
        pos = [row["orientation"]["opposite_positive_angle"]["cosine"] for row in selected]
        ax.bar(x - .19, neg, .38, label="Geant4 rotation (−angle)")
        ax.bar(x + .19, pos, .38, label="opposite (+angle)")
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), title=f"{energy} keV",
               ylabel="20-view detector-pattern cosine")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=7)
    fig.suptitle("Independent point sources vs calibrated response factors")
    output = path.with_suffix(".png")
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
