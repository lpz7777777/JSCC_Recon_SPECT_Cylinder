"""Freeze two-energy 20-view point-source scans across the ellipse and z range."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.FOV120.workflow import digest, gps_source, write_json


def prepare(out, photons_per_point):
    cfg = json.loads((HERE / "config.json").read_text())
    if out.exists() or photons_per_point % 20:
        raise ValueError("Output exists or photon count cannot divide across views")
    out.mkdir(parents=True)
    (out / "macros").mkdir()
    sites = [("center", 0.0, 0.0, 0.0)]
    for radius in (.5, .9, .98):
        for direction in range(8):
            angle = direction * math.pi / 4
            sites.append((f"rho{round(radius*100):02d}_a{direction:02d}",
                          radius * cfg["semi_axes_mm"][0] * math.cos(angle),
                          radius * cfg["semi_axes_mm"][1] * math.sin(angle), radius))
    jobs = []
    truth = []
    for site, x, y, radius in sites:
        for z in (0, -45, 45, -57, 57):
            for energy in (218, 440):
                dataset = f"Point_{site}_z{z:+03d}_{energy}keV"
                truth.append({"dataset": dataset, "object_mm": [x, y, z],
                              "normalized_ellipse_radius": radius, "energy_keV": energy})
                for view in range(1, 21):
                    theta = (view - 1) * 2 * math.pi / 20
                    world = (x * math.cos(theta) + y * math.sin(theta),
                             cfg["geant4_center_mm"][1] + y * math.cos(theta) - x * math.sin(theta), z)
                    macro = out / "macros" / f"{dataset}_v{view:02d}.mac"
                    macro.write_text("# Point source in object coordinates; no attenuation\n" +
                                     gps_source(energy, world, first=True) +
                                     f"/run/beamOn {photons_per_point//20}\n", encoding="ascii")
                    index = len(jobs)
                    jobs.append({"index": index, "dataset": dataset,
                                 "level": f"{photons_per_point:g}", "view": view,
                                 "worker": 0, "photons": photons_per_point//20,
                                 "seed": 30092801 + index, "mono_keV": energy,
                                 "role": "point_imaging", "macro": macro.relative_to(out).as_posix(),
                                 "macro_sha256": digest(macro)})
    write_json(out / "jobs.json", {"format_version": 1, "experiment_id": cfg["experiment_id"],
                                   "photons_per_point_all_views": photons_per_point,
                                   "jobs": jobs})
    write_json(out / "truth.json", {"sites": truth,
                  "note": "250 independent position-energy datasets, each with 20 views"})
    print(f"Prepared {len(jobs)} workers over {len(truth)} point datasets in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=HERE / "generated/PointScan")
    p.add_argument("--photons-per-point", type=int, default=10_000_000)
    a = p.parse_args()
    prepare(a.output, a.photons_per_point)
