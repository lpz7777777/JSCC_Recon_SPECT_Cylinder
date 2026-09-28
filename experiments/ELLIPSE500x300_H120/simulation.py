"""Prepare isolated ellipse, distance-control, and calibration Geant4 jobs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.FOV120.workflow import collect, digest, gps_source, run_job, write_json

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"


def rods():
    groups = (("center", 0, 0, 0), ("long_plus", 180, 0, 45),
              ("long_minus", -180, 0, -45), ("short_plus", 0, 95, 45),
              ("short_minus", 0, -95, -45))
    result = []
    for group, cx, cy, cz in groups:
        for index, diameter in enumerate(range(10, 31, 4)):
            theta = index * math.pi / 3
            x, y = cx + 32 * math.cos(theta), cy + 32 * math.sin(theta)
            radius = diameter / 2
            # A conservative corner check keeps the whole hot column inside.
            if ((abs(x) + radius) / 250)**2 + ((abs(y) + radius) / 150)**2 >= 1:
                raise ValueError(f"Rod touches ellipse boundary: {group}/{index}")
            result.append({"group": group, "energy": 218 if index % 2 == 0 else 440,
                           "center_mm": [x, y, cz], "radius_mm": radius,
                           "height_mm": 30, "excess_activity": 5})
    return result


def ellipse_macro(config, angle, include_rods=False):
    ax, by = config["semi_axes_mm"]
    height = config["height_mm"]
    volume = math.pi * ax * by * height
    lines = ["# Independent ellipse FOV; no object attenuation", "/ellipse/clear",
             f"/ellipse/centerY {config['geant4_center_mm'][1]}",
             f"/ellipse/angle {angle}"]
    for energy in (218, 440):
        weight = config["gamma_yields"][str(energy)] * volume
        lines.append(f"/ellipse/add {energy} {ax} {by} {height / 2} {weight:.15g}")
    if include_rods:
        for rod in rods():
            x, y, z = rod["center_mm"]
            radius = rod["radius_mm"]
            halfz = rod["height_mm"] / 2
            energy = rod["energy"]
            weight = (rod["excess_activity"] * config["gamma_yields"][str(energy)] *
                      math.pi * radius**2 * rod["height_mm"])
            lines.append(f"/ellipse/addRod {energy} {x:.12g} {y:.12g} {z:.12g} "
                         f"{radius:.12g} {halfz:.12g} {weight:.15g}")
    return "\n".join(lines) + "\n"


def circle_macro(config, radius, energy=None):
    center = config["geant4_center_mm"]
    body = "# Distance-control cylinder; no object attenuation\n/gps/source/multiplevertex false\n"
    energies = (energy,) if energy else (218, 440)
    for index, item in enumerate(energies):
        weight = config["gamma_yields"][str(item)] * math.pi * radius**2 * config["height_mm"]
        body += gps_source(item, center, radius, config["height_mm"],
                           intensity=weight, first=index == 0)
    return body


def prepare(config, output):
    if output.exists():
        raise FileExistsError(output)
    macros = output / "macros"
    macros.mkdir(parents=True)
    jobs = []

    def add(dataset, level, view, body, total, workers, mono=None, role="imaging"):
        if total % workers or total // workers > 2147483647:
            raise ValueError("Invalid photon split")
        macro = macros / f"{dataset}_{level}_v{view:02d}.mac"
        macro.write_text(body + f"/run/beamOn {total // workers}\n", encoding="ascii")
        for worker in range(workers):
            index = len(jobs)
            jobs.append({"index": index, "dataset": dataset, "level": level,
                         "view": view, "worker": worker,
                         "photons": total // workers, "seed": 28092801 + index,
                         "mono_keV": mono, "role": role,
                         "macro": macro.relative_to(output).as_posix(),
                         "macro_sha256": digest(macro)})

    for role in ("calibration", "sensitivity", "sensitivity_validation"):
        for energy in ((218, 440) if role == "calibration" else (440,)):
            body = circle_macro(config, config["support_radius_mm"], energy)
            for level, photons, workers in (("pilot", 100_000_000, 10),
                                            ("extension", 900_000_000, 90)):
                add(f"{role}_{energy}", level, 1, body, photons, workers,
                    energy, role)
    truth = {"shape": "ellipse_cylinder", "semi_axes_mm": config["semi_axes_mm"],
             "height_mm": config["height_mm"], "rods": rods(),
             "gamma_yields": config["gamma_yields"],
             "normalization": "common activity scale; yield times physical volume"}
    write_json(output / "Contrast_truth.json", truth)
    for dataset in ("CircleNewDist", "EllipseUniform", "EllipseContrast"):
        for photons in config["count_levels"]:
            level = f"1e{int(math.log10(photons))}"
            for view in range(1, config["rotate_num"] + 1):
                if dataset == "CircleNewDist":
                    body = circle_macro(config, 150)
                else:
                    body = ellipse_macro(config, (view - 1) * 360 / config["rotate_num"],
                                         dataset == "EllipseContrast")
                add(dataset, level, view, body, photons // config["rotate_num"], 10)
    write_json(output / "jobs.json", {"format_version": 1,
               "experiment_id": config["experiment_id"],
               "config_sha256": digest(HERE / "config.json"), "jobs": jobs})
    print(f"Prepared {len(jobs)} immutable jobs in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, default=GENERATED / "Simulation")
    p = sub.add_parser("run")
    p.add_argument("--manifest", type=Path, default=GENERATED / "Simulation/jobs.json")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--crystal", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("collect")
    p.add_argument("--manifest", type=Path, default=GENERATED / "Simulation/jobs.json")
    p.add_argument("--dataset", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--destination", type=Path, default=GENERATED)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(json.loads((HERE / "config.json").read_text()), args.output.resolve())
    elif args.command == "run":
        run_job(args.manifest.resolve(), args.index, args.executable, args.crystal, args.smoke)
    else:
        collect(args.manifest.resolve(), args.dataset, args.level, args.destination)


if __name__ == "__main__":
    main()
