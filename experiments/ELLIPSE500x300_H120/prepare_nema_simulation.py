"""Freeze 20-view, two-energy NEMA Geant4 macros from the approved 3-mm truth.

This uses the existing /xcat/add weighted-cuboid source interface. Constant
adjacent truth cells are merged exactly; each cuboid's weight is concentration
times volume times the photon yield. Boundary activity is sampled uniformly
within a 3-mm truth cell, so the transport source matches the saved voxel truth
but approximates the analytic sphere/body boundaries at that resolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
PHANTOM = HERE / "nema_body_h60_config.json"
TRUTH = HERE / "generated/NEMA_Body_H60/truth_3mm.npz"
TRUTH_MANIFEST = HERE / "reports/NEMA_Body_H60/manifest.json"
DEFAULT_OUTPUT = HERE / "generated/NEMA_Body_H60/Simulation_1e9"
DATASET = "NEMA_Body_H60"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def boxes(source: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
          energy: int, gamma_yield: float):
    """Greedily partition nonzero float32 cells into equal-value cuboids."""
    if source.dtype != np.float32 or source.shape != (len(z), len(y), len(x)):
        raise ValueError("Unexpected NEMA truth shape or dtype")
    if not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("Nonfinite or negative activity")
    used = np.zeros(source.shape, dtype=bool)
    result = []
    spacing = 3.0
    for k, j, i in np.argwhere(source > 0):
        if used[k, j, i]:
            continue
        value = source[k, j, i]
        i1 = i + 1
        while i1 < len(x) and not used[k, j, i1] and source[k, j, i1] == value:
            i1 += 1
        j1 = j + 1
        while (j1 < len(y) and
               np.all(~used[k, j1, i:i1] & (source[k, j1, i:i1] == value))):
            j1 += 1
        k1 = k + 1
        while (k1 < len(z) and
               np.all(~used[k1, j:j1, i:i1] & (source[k1, j:j1, i:i1] == value))):
            k1 += 1
        used[k:k1, j:j1, i:i1] = True
        half_sizes = ((i1-i)*spacing/2, (j1-j)*spacing/2, (k1-k)*spacing/2)
        center = ((x[i]+x[i1-1])/2, (y[j]+y[j1-1])/2,
                  (z[k]+z[k1-1])/2)
        volume = (i1-i)*(j1-j)*(k1-k)*spacing**3
        weight = float(value) * volume * gamma_yield
        result.append((energy, *center, *half_sizes, weight))
    if not np.array_equal(used, source > 0):
        raise AssertionError("Merged boxes do not cover nonzero NEMA truth")
    expected = float(source.sum(dtype=np.float64) * spacing**3 * gamma_yield)
    actual = math.fsum(box[-1] for box in result)
    if not math.isclose(actual, expected, rel_tol=1e-12):
        raise AssertionError(f"{energy} keV source integral mismatch: {actual} vs {expected}")
    return result, expected


def prepare(output: Path, total_photons: int, workers_per_view: int):
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads(CONFIG.read_text())
    phantom = json.loads(PHANTOM.read_text())
    truth_manifest = json.loads(TRUTH_MANIFEST.read_text())
    if (truth_manifest["truth_sha256"] != digest(TRUTH) or
        truth_manifest["configuration_sha256"] != digest(PHANTOM) or
        truth_manifest["status"] !=
            "geometry_and_truth_preview_only_no_transport_no_reconstruction"):
        raise ValueError("NEMA truth provenance mismatch")
    if (cfg["experiment_id"] != "ELLIPSE500x300_H120" or
        cfg["physical_shape"] != "ellipse_cylinder" or
        cfg["semi_axes_mm"] != [250.0, 150.0] or
        cfg["height_mm"] != 120.0 or
        phantom["relative_activity_concentration"]["hot_sphere_to_own_background_ratio"] != 10):
        raise ValueError("Unapproved geometry or source prescription")
    views = int(cfg["rotate_num"])
    if (views != 20 or workers_per_view <= 0 or
        total_photons % (views * workers_per_view) or
        total_photons // (views * workers_per_view) > 2_147_483_647):
        raise ValueError("Invalid photon partition")
    with np.load(TRUTH) as payload:
        x, y, z = (payload[f"{axis}_mm"].astype(np.float64)
                   for axis in "xyz")
        arrays = {energy: payload[f"activity_{energy}_zyx"].copy()
                  for energy in (218, 440)}
    if (len(x), len(y), len(z)) != (168, 100, 40):
        raise ValueError("Wrong NEMA truth canvas")
    all_boxes = []
    integrals = {}
    counts = {}
    for energy in (218, 440):
        current, weighted_integral = boxes(arrays[energy], x, y, z,
                                            energy, cfg["gamma_yields"][str(energy)])
        all_boxes.extend(current)
        integrals[str(energy)] = weighted_integral
        counts[str(energy)] = len(current)
    if not all_boxes:
        raise ValueError("Empty NEMA source")
    for _, cx, cy, cz, hx, hy, hz, _ in all_boxes:
        if (((abs(cx)+hx)/250)**2 + ((abs(cy)+hy)/150)**2 >= 1 or
            abs(cz)+hz > 60):
            raise ValueError("A source cuboid protrudes outside physical ellipse FOV")
    photons_per_worker = total_photons // (views * workers_per_view)
    output.mkdir(parents=True)
    macros_dir = output / "macros"
    macros_dir.mkdir()
    jobs = []
    macro_rows = []
    for view in range(1, views+1):
        macro = macros_dir / f"{DATASET}_v{view:02d}.mac"
        with macro.open("w", encoding="ascii") as stream:
            stream.write("# NEMA Body H60 3-mm truth; no object attenuation\n")
            stream.write("/xcat/clear\n")
            stream.write(f"/xcat/centerY {cfg['geant4_center_mm'][1]:.12g}\n")
            stream.write(f"/xcat/angle {(view-1)*360/views:.12g}\n")
            for energy, cx, cy, cz, hx, hy, hz, weight in all_boxes:
                stream.write(f"/xcat/add {energy} {cx:.9f} {cy:.9f} {cz:.9f} "
                             f"{hx:.9f} {hy:.9f} {hz:.9f} {weight:.15g}\n")
            stream.write(f"/run/beamOn {photons_per_worker}\n")
        macro_hash = digest(macro)
        macro_rows.append({"view": view, "path": macro.relative_to(output).as_posix(),
                           "bytes": macro.stat().st_size, "sha256": macro_hash})
        for worker in range(workers_per_view):
            index = len(jobs)
            jobs.append({"index": index, "dataset": DATASET, "level": "1e9",
                         "view": view, "worker": worker,
                         "photons": photons_per_worker,
                         "seed": 30093001+index, "mono_keV": None,
                         "role": "imaging",
                         "macro": macro.relative_to(output).as_posix(),
                         "macro_sha256": macro_hash})
    if sum(job["photons"] for job in jobs) != total_photons:
        raise AssertionError("Photon total did not close")
    manifest = {"format_version": 1, "experiment_id": cfg["experiment_id"],
                "dataset": DATASET, "level": "1e9", "jobs": jobs,
                "nema_manifest_sha256": digest(TRUTH_MANIFEST),
                "nema_truth_sha256": digest(TRUTH),
                "config_sha256": digest(CONFIG),
                "phantom_config_sha256": digest(PHANTOM),
                "total_primary_photons": total_photons,
                "workers_per_view": workers_per_view,
                "source_boxes_by_energy": counts,
                "yield_weighted_activity_integral": integrals,
                "expected_primary_energy_fraction": {
                    key: value/math.fsum(integrals.values())
                    for key, value in integrals.items()},
                "source_model": "weighted cuboids from 3-mm fractional-volume truth; isotropic gamma; no object attenuation",
                "macros": macro_rows}
    (output / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in
          ("total_primary_photons", "source_boxes_by_energy",
           "yield_weighted_activity_integral", "expected_primary_energy_fraction")},
          indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total", type=int, default=1_000_000_000)
    parser.add_argument("--workers-per-view", type=int, default=10)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.total, args.workers_per_view)


if __name__ == "__main__":
    main()
