"""Draw a categorical combined 225Ac Contrast Phantom source schematic."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

from plot_dual_energy_contrast_phantom import parse_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = parse_sources(args.macro.resolve())
    background = next(source for source in sources if source["energy_keV"] == 218 and source["radius_mm"] > 50)
    center_x, center_y = background["x_mm"], background["y_mm"]
    rods = [source for source in sources if source["radius_mm"] <= 50]

    colors = {218: "#0072B2", 440: "#D55E00"}  # Fr-218 blue; Bi-440 red.
    figure, axis = plt.subplots(figsize=(8.2, 8.0), constrained_layout=True)
    axis.set_facecolor("white")  # Activity 0 outside the source cylinder.
    axis.add_patch(Circle((0, 0), background["radius_mm"], facecolor="#B8B8B8", edgecolor="black", linewidth=1.5))
    for rod in rods:
        x = rod["x_mm"] - center_x
        y = rod["y_mm"] - center_y
        axis.add_patch(Circle((x, y), rod["radius_mm"], facecolor=colors[int(rod["energy_keV"])], edgecolor="black", linewidth=1.2))

    legend = [
        Patch(facecolor="white", edgecolor="black", label="0: outside source cylinder"),
        Patch(facecolor="#B8B8B8", edgecolor="black", label="1: uniform background"),
        Patch(facecolor=colors[218], edgecolor="black", label="Fr-218 hot rods (1, 3, 5)"),
        Patch(facecolor=colors[440], edgecolor="black", label="Bi-440 hot rods (2, 4, 6)"),
    ]
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    axis.set_xlim(-140, 140)
    axis.set_ylim(-140, 140)
    axis.set_aspect("equal")
    axis.set_xlabel("x relative to phantom center (mm)")
    axis.set_ylabel("y relative to phantom center (mm)")
    axis.set_title("225Ac dual-energy Contrast Phantom: combined source distribution")
    axis.grid(False)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
