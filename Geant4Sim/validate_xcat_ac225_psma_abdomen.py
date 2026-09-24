"""Independently validate generated XCAT Ac-225 macros and 3-mm truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_DIR = Path(__file__).resolve().parent / "Macro" / "XCAT_Ac225_PSMA_Abdomen_300x300x60"


def parse_macro(path: Path) -> tuple[list[dict], int, float]:
    sources: list[dict] = []
    beam_on = None
    angle = None
    with path.open("r", encoding="ascii") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("/xcat/add "):
                fields = line.split()
                if len(fields) != 9:
                    raise AssertionError(f"Malformed source in {path}")
                sources.append({"energy": int(fields[1]),
                                "center": tuple(map(float, fields[2:5])),
                                "halfx": float(fields[5]), "halfy": float(fields[6]),
                                "halfz": float(fields[7]), "intensity": float(fields[8])})
            elif line.startswith("/xcat/angle "):
                angle = float(line.split()[1])
            elif line.startswith("/run/beamOn "):
                beam_on = int(line.split()[1])
    if beam_on is None or angle is None or not sources:
        raise AssertionError(f"Missing angle, source, or beamOn in {path}")
    return sources, beam_on, angle


def backproject(sources: list[dict], yield_by_energy: dict[int, float], shape=(20, 100, 100)) -> dict[int, np.ndarray]:
    halfheight = shape[0] * 3 / 2
    output = {energy: np.zeros(shape, dtype=np.float64) for energy in yield_by_energy}
    for source in sources:
        energy = source["energy"]
        x, global_y, z = source["center"]
        y = global_y
        hx, hy, hz = source["halfx"], source["halfy"], source["halfz"]
        activity = source["intensity"] / (yield_by_energy[energy] * 8.0 * hx * hy * hz)
        if hx == hy == hz == 0.75:
            i, j, k = (int(math.floor((coord + limit) / 3.0))
                       for coord, limit in ((x, 150), (y, 150), (z, halfheight)))
            if not (0 <= k < shape[0] and 0 <= j < shape[1] and 0 <= i < shape[2]):
                raise AssertionError("Native source outside target grid")
            output[energy][k, j, i] += activity / 8.0
        else:
            i0, i1 = round((x - hx + 150) / 3), round((x + hx + 150) / 3)
            j0, j1 = round((y - hy + 150) / 3), round((y + hy + 150) / 3)
            k0, k1 = round((z - hz + halfheight) / 3), round((z + hz + halfheight) / 3)
            if not (0 <= i0 < i1 <= 100 and 0 <= j0 < j1 <= 100 and 0 <= k0 < k1 <= shape[0]):
                raise AssertionError("Source outside the target grid")
            output[energy][k0:k1, j0:j1, i0:i1] += activity
    return output


def validate(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    halfheight = manifest["target_shape_zyx"][0] * 1.5
    z0 = manifest["crop_half_open"]["z"][0]
    with np.load(directory / "truth_3mm.npz", allow_pickle=False) as truth:
        expected_maps = {218: truth["fr_zyx"].astype(np.float64),
                         440: truth["bi_zyx"].astype(np.float64)}
        names = ("body", "kidney", "liver", "bowel", "spine_bone", "lesion")
        organ_fractions = {name: truth[name + "_fraction_zyx"].astype(np.float64)
                           for name in names}
        for name, fraction in organ_fractions.items():
            if not (0 <= fraction).all() or not (fraction <= 1).all():
                raise AssertionError(f"Invalid {name} fractional mask")
        bi_fr_difference = expected_maps[440] - expected_maps[218]
        if np.any(bi_fr_difference[organ_fractions["kidney"] == 0] != 0) or np.any(
                bi_fr_difference[organ_fractions["lesion"] > 0] != 0):
            raise AssertionError("Fr/Bi activity may differ only in kidneys, not the lesion")
    with np.load(directory / "masks_1p5mm.npz", allow_pickle=False) as masks:
        kidneys = masks["kidney_zyx"]
        liver = masks["liver_zyx"]
        lesion = masks["lesion_zyx"]
        codes = masks["xcat_codes_zyx"]
        if (kidneys[:, :, :100].sum() == 0 or kidneys[:, :, 100:].sum() == 0
                or liver.sum() == 0):
            raise AssertionError("Crop must contain both kidneys and liver")
        if codes[795 - z0, 153 - 28, 132 - 28] != 2 or not lesion[795 - z0, 153 - 28, 132 - 28]:
            raise AssertionError("Lesion centre must be inside the vertebral body")
    yields = {int(k): float(v) for k, v in manifest["gamma_yields"].items()}
    records = {item["file"]: item for item in manifest["macros"]}
    base, beam, base_angle = parse_macro(directory / "view_01_mixed.mac")
    if len(base) != sum(manifest["source_counts"].values()) or beam != manifest["total_primary_photons_all_workers_views"] // manifest["views"] // manifest["workers_per_view"] or base_angle != 0:
        raise AssertionError("Incorrect source or per-worker event count")
    reconstructed = backproject(base, yields, tuple(manifest["target_shape_zyx"]))
    max_error = {}
    organ_integral_error = {}
    for energy, expected in expected_maps.items():
        error = float(np.max(np.abs(reconstructed[energy] - expected)))
        if error > 1e-6:
            raise AssertionError(f"{energy} keV macro-to-truth error: {error}")
        max_error[str(energy)] = error
        organ_integral_error[str(energy)] = {}
        for name, fraction in organ_fractions.items():
            before = float(np.sum(expected * fraction, dtype=np.float64) * 27.0)
            after = float(np.sum(reconstructed[energy] * fraction, dtype=np.float64) * 27.0)
            difference = abs(after - before)
            if difference > max(1e-5, before * 1e-6):
                raise AssertionError(f"{energy} keV {name} source integral mismatch")
            organ_integral_error[str(energy)][name] = difference
    for source in base:
        x, y, z = source["center"]
        hx, hy, hz = source["halfx"], source["halfy"], source["halfz"]
        if math.hypot(abs(x) + hx, abs(y) + hy) > 150 + 1e-8 or abs(z) + hz > halfheight + 1e-8:
            raise AssertionError("Source extends beyond physical FOV")
    per_energy = {energy: float(math.fsum(source["intensity"] for source in base if source["energy"] == energy))
                  for energy in yields}
    for energy, expected in expected_maps.items():
        target_intensity = float(expected.sum() * 27.0 * yields[energy])
        if not math.isclose(per_energy[energy], target_intensity, rel_tol=1e-7):
            raise AssertionError(f"{energy} keV integrated photon weight mismatch")
    total_view_photons = 0
    # Rotate actual cuboid corners about the source centre, then shift y by -245 mm.
    # All four corners of every box must stay inside the physical cylindrical FOV.
    centres = np.asarray([source["center"][:2] for source in base], dtype=np.float64)
    halfwidths = np.asarray([[source["halfx"], source["halfy"]] for source in base], dtype=np.float64)
    max_rotated_radius = 0.0
    for view in range(1, 21):
        name = f"view_{view:02d}_mixed.mac"
        path = directory / name
        sources, beam_on, angle = (base, beam, base_angle) if view == 1 else parse_macro(path)
        record = records[name]
        if len(sources) != len(base) or beam_on != record["beam_on_per_worker"] or angle != (view - 1) * 18:
            raise AssertionError(f"Wrong source/event count: {name}")
        if sources != base:
            raise AssertionError(f"Source table mismatch: {name}")
        theta = math.radians(angle)
        cosine, sine = math.cos(theta), math.sin(theta)
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            corners = centres + halfwidths * np.array([sx, sy])
            world_x = corners[:, 0] * cosine + corners[:, 1] * sine
            world_y = -245 + corners[:, 1] * cosine - corners[:, 0] * sine
            radius = float(np.hypot(world_x, world_y + 245).max())
            max_rotated_radius = max(max_rotated_radius, radius)
            if radius > 150 + 1e-8:
                raise AssertionError(f"Rotated source outside physical FOV: {name}")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != record["sha256"]:
            raise AssertionError(f"Checksum mismatch: {name}")
        total_view_photons += beam_on * manifest["workers_per_view"]
    if total_view_photons != manifest["total_primary_photons_all_workers_views"]:
        raise AssertionError("Whole-study beamOn sum mismatch")
    for energy in (218, 440):
        name = f"view_01_{energy}keV_check.mac"
        sources, beam_on, angle = parse_macro(directory / name)
        reference = [source for source in base if source["energy"] == energy]
        if beam_on != 10000 or angle != 0 or sources != reference:
            raise AssertionError(f"Monoenergetic check macro mismatch: {name}")
    return {"status": "passed", "mixed_views": 20, "mono_check_macros": 2,
            "sources_per_mixed_view": len(base), "max_abs_activity_error": max_error,
            "organ_integral_abs_error_activity_mm3": organ_integral_error,
            "max_rotated_corner_radius_mm": max_rotated_radius,
            "integrated_photon_weight": per_energy,
            "total_primary_photons_all_workers_views": total_view_photons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
