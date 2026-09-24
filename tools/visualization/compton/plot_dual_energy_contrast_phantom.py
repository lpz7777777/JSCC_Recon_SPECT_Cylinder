"""Draw source-distribution schematics from a dual-energy Contrast Phantom GPS macro."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def parse_sources(path: Path):
    sources, current = [], None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("/gps/source/add"):
            if current is not None:
                sources.append(current)
            current = {"weight": float(line.split()[-1])}
        elif line.startswith("/gps/particle"):
            if current is None:
                current = {"weight": 1.0}
        elif line.startswith("/gps/energy"):
            current["energy_keV"] = float(line.split()[-2])
        elif line.startswith("/gps/pos/centre"):
            parts = line.split()
            current["x_mm"] = float(parts[-4])
            current["y_mm"] = float(parts[-3])
        elif line.startswith("/gps/pos/radius"):
            current["radius_mm"] = float(line.split()[-2])
        elif line.startswith("/run/beamOn") and current is not None:
            sources.append(current)
            current = None
    if current is not None:
        sources.append(current)
    required = {"weight", "energy_keV", "x_mm", "y_mm", "radius_mm"}
    if not sources or any(required - source.keys() for source in sources):
        raise ValueError(f"Could not parse complete GPS sources from {path}")
    return sources


def density_map(sources, energy_keV, x, y, normalizer):
    density = np.zeros_like(x, dtype=np.float64)
    for source in sources:
        if source["energy_keV"] != energy_keV:
            continue
        mask = (x - source["x_mm"]) ** 2 + (y - source["y_mm"]) ** 2 <= source["radius_mm"] ** 2
        density[mask] += source["weight"] / (np.pi * source["radius_mm"] ** 2)
    return density / normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sources = parse_sources(args.macro.resolve())
    background_218 = next(s for s in sources if s["energy_keV"] == 218 and s["radius_mm"] > 50)
    background_440 = next(s for s in sources if s["energy_keV"] == 440 and s["radius_mm"] > 50)
    center_x, center_y = background_218["x_mm"], background_218["y_mm"]
    if (background_440["x_mm"], background_440["y_mm"], background_440["radius_mm"]) != (
        center_x, center_y, background_218["radius_mm"]
    ):
        raise ValueError("218 and 440 background cylinders do not share the same geometry")

    local_axis = np.linspace(-130.0, 130.0, 801)
    local_x, local_y = np.meshgrid(local_axis, local_axis)
    x, y = local_x + center_x, local_y + center_y
    density_218_raw = density_map(sources, 218, x, y, normalizer=1.0)
    density_440_raw = density_map(sources, 440, x, y, normalizer=1.0)
    density_218 = density_218_raw / (background_218["weight"] / (np.pi * background_218["radius_mm"] ** 2))
    density_440 = density_440_raw / (background_440["weight"] / (np.pi * background_440["radius_mm"] ** 2))
    combined = (density_218_raw + density_440_raw) / (
        background_218["weight"] / (np.pi * background_218["radius_mm"] ** 2)
        + background_440["weight"] / (np.pi * background_440["radius_mm"] ** 2)
    )

    panels = [
        (density_218, "218 keV (Fr): distribution / 218-keV background", "YlOrBr"),
        (density_440, "440 keV (Bi): distribution / 440-keV background", "Blues"),
        (combined, "Combined 218 + 440: distribution / combined background", "magma"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.7), constrained_layout=True)
    rods = [source for source in sources if source["radius_mm"] <= 50]
    for axis, (image, title, colormap) in zip(axes, panels):
        shown = axis.imshow(
            image, origin="lower", extent=(-130, 130, -130, 130), cmap=colormap,
            vmin=0, vmax=float(np.max(image)), interpolation="nearest",
        )
        axis.add_patch(Circle((0, 0), background_218["radius_mm"], fill=False, color="black", linewidth=1.2))
        for index, rod in enumerate(rods, start=1):
            color = "#e08b00" if rod["energy_keV"] == 218 else "#1261a0"
            x_local, y_local = rod["x_mm"] - center_x, rod["y_mm"] - center_y
            axis.add_patch(Circle((x_local, y_local), rod["radius_mm"], fill=False, color=color, linewidth=1.25))
            axis.text(x_local, y_local, str(index), ha="center", va="center", fontsize=8, color="white", weight="bold")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x relative to phantom center (mm)")
        axis.set_ylabel("y relative to phantom center (mm)")
        axis.set_aspect("equal")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04, label="relative primary-emission density")
    figure.suptitle("225Ac dual-energy Contrast Phantom source distributions (30-mm axial cylinder height)")
    figure.text(0.5, 0.01, "Orange outlines: Fr-218 rods (1, 3, 5); blue outlines: Bi-440 rods (2, 4, 6).", ha="center")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)

    report = {
        "source_macro": str(args.macro.resolve()),
        "output": str(output),
        "geometry": {"background_diameter_mm": 240, "height_mm": 30, "hot_to_background_within_energy": 6},
        "source_energy_yields": {"218_keV": 0.114, "440_keV": 0.261},
        "rods": [
            {"number": index, "energy_keV": int(rod["energy_keV"]), "diameter_mm": 2 * rod["radius_mm"],
             "x_relative_mm": rod["x_mm"] - center_x, "y_relative_mm": rod["y_mm"] - center_y}
            for index, rod in enumerate(rods, start=1)
        ],
        "normalization": {
            "218_panel": "218-keV background density = 1",
            "440_panel": "440-keV background density = 1",
            "combined_panel": "combined 218+440 background density = 1",
        },
        "max_relative_density": {"218": float(density_218.max()), "440": float(density_440.max()), "combined": float(combined.max())},
    }
    args.report.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
