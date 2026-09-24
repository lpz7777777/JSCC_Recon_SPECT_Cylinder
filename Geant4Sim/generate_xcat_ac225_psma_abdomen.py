"""Generate a reproducible 3-D 221Fr/213Bi source phantom from XCAT_TB.xif.

The source uses calibrated 1.5 mm XCAT voxels. The XIF AMIDE header says
1 mm, which is inconsistent with the adult body extent (1178 occupied slices).
At 1.5 mm the occupied height is 1767 mm and the 256-pixel width is 384 mm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XIF = Path(r"D:\JSCC_Recon\20250307_JSCCGC_32x32x4_DiffEne\Geant4Sim\XCAT_TB.xif")
DEFAULT_OUT = ROOT / "Geant4Sim" / "Macro" / "XCAT_Ac225_PSMA_Abdomen_300x300x60"
Z0, Z1 = 775, 815
XY0, XY1 = 28, 228
NATIVE_MM = 1.5
GRID_MM = 3.0
FOV_RADIUS_MM = 150.0
FOV_CENTER_MM = (0.0, -245.0, 0.0)
YIELDS = {218: 0.114, 440: 0.259}
ACTIVITY = {"soft": 0.3, "bowel": 0.4, "liver": 1.0,
            "kidney_fr": 3.2, "kidney_bi": 3.4, "lesion": 5.0}
VIEWS = 20
WORKERS_PER_VIEW = 10
TOTAL_PHOTONS = 1_000_000_000
LESION_CENTER_XYZ = (132.0, 153.0, 795.0)
LESION_FWHM_MM = 18.0


@dataclass(frozen=True)
class Box:
    energy: int
    x: float
    y: float
    z: float
    hx: float
    hy: float
    hz: float
    activity: float

    @property
    def volume(self) -> float:
        return 8.0 * self.hx * self.hy * self.hz

    @property
    def intensity(self) -> float:
        return self.activity * YIELDS[self.energy] * self.volume


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _location(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s*", value)
    if not match:
        raise ValueError(f"Invalid XIF location: {value!r}")
    return int(match.group(1), 16), int(match.group(2), 16)


def _xml_at(stream, location: tuple[int, int]) -> ET.Element:
    stream.seek(location[0])
    data = stream.read(location[1])
    if len(data) != location[1]:
        raise ValueError("Truncated XIF XML block")
    return ET.fromstring(data)


def parse_xif(path: Path) -> tuple[tuple[int, int, int], tuple[float, ...], int]:
    with path.open("rb") as stream:
        if stream.read(48).rstrip(b"\0") != b"AMIDE XML Image Format Flat File Version 2.0":
            raise ValueError("Unexpected XIF signature")
        stream.seek(0x40)
        study = _xml_at(stream, struct.unpack("<QQ", stream.read(16)))
        dataset = _xml_at(stream, _location(study.findtext("./study/children/object_location_and_size", "")))
        data = dataset.find("./data-set")
        if data is None:
            raise ValueError("Missing XIF dataset")
        raw = _xml_at(stream, _location(data.findtext("raw_data_location_and_size", "")))
        dims = tuple(map(int, raw.findtext("dim", "").split()))
        if dims != (256, 256, 1220, 1, 1) or raw.findtext("raw_format") != "float-32-le":
            raise ValueError(f"Unexpected XCAT data layout: {dims}")
        offset, size = _location(raw.findtext("raw_data_location_and_size", ""))
        if size != math.prod(dims[:3]) * 4:
            raise ValueError("XIF raw byte count does not match dimensions")
        header_mm = tuple(map(float, data.findtext("voxel_size", "").split()))
    return dims[:3], header_mm, offset


def local_centers(indices: np.ndarray, origin: int, extent: int) -> np.ndarray:
    return (indices.astype(np.float64) + 0.5 - origin - extent / 2.0) * NATIVE_MM


def native_phantom(codes: np.ndarray, z0: int = Z0, z1: int = Z1) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if codes.shape != (z1 - z0, 200, 200):
        raise ValueError(f"Unexpected crop shape {codes.shape}")
    known = {0, 2, 4, 5, 6, 8, 16, 30, 50, 60, 75, 97, 98, 99, 100, 101, 102}
    unknown = set(np.unique(codes).tolist()) - known
    if unknown:
        raise ValueError(f"Unrecognized XCAT codes: {sorted(unknown)}")

    zz = np.arange(z0, z1)[:, None, None]
    yy = np.arange(XY0, XY1)[None, :, None]
    xx = np.arange(XY0, XY1)[None, None, :]
    x_mm = local_centers(xx, XY0, 200)
    y_mm = local_centers(yy, XY0, 200)
    # Every emitted native cube is fully contained in the r <= 150 mm support.
    radial_corner = np.hypot(np.abs(x_mm) + NATIVE_MM / 2,
                             np.abs(y_mm) + NATIVE_MM / 2)
    support = np.broadcast_to(radial_corner <= FOV_RADIUS_MM + 1e-10, codes.shape).copy()
    body = (codes != 0) & support
    masks = {
        "body": body,
        "kidney": (codes == 75) & support,
        "liver": (codes == 101) & support,
        "bowel": (codes == 102) & support,
        "spine_bone": (codes == 6) & support,
    }
    # The vertebral body interior has XCAT label 2 and is enclosed by label 6.
    # This small ellipsoid sits inside the cortical ring around (132,153,795).
    dx = (xx - LESION_CENTER_XYZ[0]) * NATIVE_MM
    dy = (yy - LESION_CENTER_XYZ[1]) * NATIVE_MM
    dz = (zz - LESION_CENTER_XYZ[2]) * NATIVE_MM
    vertebral_roi = (dx / 15.0) ** 2 + (dy / 12.0) ** 2 + (dz / 18.0) ** 2 <= 1.0
    lesion = vertebral_roi & np.isin(codes, (2, 6)) & support
    if codes[795 - z0, 153 - XY0, 132 - XY0] != 2 or np.count_nonzero(lesion) < 1000:
        raise ValueError("Vertebral lesion is not contained in the expected XCAT anatomy")
    masks["lesion"] = lesion
    sigma = LESION_FWHM_MM / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    profile = np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma))

    common = np.zeros(codes.shape, dtype=np.float64)
    common[body] = ACTIVITY["soft"]
    common[masks["bowel"]] = ACTIVITY["bowel"]
    common[masks["liver"]] = ACTIVITY["liver"]
    fr = common.copy()
    bi = common.copy()
    fr[masks["kidney"]] = ACTIVITY["kidney_fr"]
    bi[masks["kidney"]] = ACTIVITY["kidney_bi"]
    for array in (fr, bi):
        array[lesion] = ACTIVITY["soft"] + (ACTIVITY["lesion"] - ACTIVITY["soft"]) * profile[lesion]
    return fr, bi, masks, support


def block_mean(native: np.ndarray) -> np.ndarray:
    return native.reshape(native.shape[0] // 2, 2, 100, 2, 100, 2).mean(axis=(1, 3, 5))


def whole_target_cell_mask(nz: int = 20) -> np.ndarray:
    y, x = np.ogrid[:100, :100]
    cx = (x + 0.5 - 50) * GRID_MM
    cy = (y + 0.5 - 50) * GRID_MM
    inside = np.hypot(np.abs(cx) + GRID_MM / 2, np.abs(cy) + GRID_MM / 2) <= FOV_RADIUS_MM + 1e-10
    return np.broadcast_to(inside, (nz, 100, 100)).copy()


def boxify(energy: int, target: np.ndarray, native: np.ndarray, support: np.ndarray) -> list[Box]:
    boxes: list[Box] = []
    nz = target.shape[0]
    whole = whole_target_cell_mask(nz)
    used = np.zeros(target.shape, dtype=bool)
    # Exact-value greedy 3-D cuboid merging; deterministic z/y/x order.
    for k in range(nz):
        for j in range(100):
            for i in range(100):
                if used[k, j, i] or not whole[k, j, i] or target[k, j, i] <= 0:
                    continue
                value = target[k, j, i]
                i1 = i + 1
                while i1 < 100 and whole[k, j, i1] and not used[k, j, i1] and target[k, j, i1] == value:
                    i1 += 1
                j1 = j + 1
                while j1 < 100 and np.all(whole[k, j1, i:i1] & ~used[k, j1, i:i1] & (target[k, j1, i:i1] == value)):
                    j1 += 1
                k1 = k + 1
                while k1 < nz and np.all(whole[k1, j:j1, i:i1] & ~used[k1, j:j1, i:i1] & (target[k1, j:j1, i:i1] == value)):
                    k1 += 1
                used[k:k1, j:j1, i:i1] = True
                boxes.append(Box(energy, (i + i1 - 100) * GRID_MM / 2,
                                 (j + j1 - 100) * GRID_MM / 2,
                                 (k + k1 - nz) * GRID_MM / 2,
                                 (i1 - i) * GRID_MM / 2, (j1 - j) * GRID_MM / 2,
                                 (k1 - k) * GRID_MM / 2, float(value)))
    # At the circular boundary, retain the constituent native 1.5 mm cubes.
    for k, j, i in np.argwhere(~whole & (target > 0)):
        for dk in range(2):
            for dj in range(2):
                for di in range(2):
                    nk, nj, ni = 2 * k + dk, 2 * j + dj, 2 * i + di
                    value = float(native[nk, nj, ni])
                    if value <= 0 or not support[nk, nj, ni]:
                        continue
                    boxes.append(Box(energy, (ni + 0.5 - 100) * NATIVE_MM,
                                     (nj + 0.5 - 100) * NATIVE_MM,
                                     (nk + 0.5 - nz) * NATIVE_MM,
                                     NATIVE_MM / 2, NATIVE_MM / 2, NATIVE_MM / 2, value))
    expected = float(target.sum(dtype=np.float64) * GRID_MM ** 3 * YIELDS[energy])
    actual = math.fsum(box.intensity for box in boxes)
    if not math.isclose(actual, expected, rel_tol=2e-12):
        raise AssertionError(f"{energy} keV source integral mismatch: {actual} vs {expected}")
    return boxes


def write_macro(path: Path, boxes: list[Box], angle_deg: float, beam_on: int, z0: int = Z0, z1: int = Z1, workers: int = WORKERS_PER_VIEW) -> dict:
    energy_intensities = {218: 0.0, 440: 0.0}
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"# XCAT Ac-225 PSMA-I&T, 24 h, 3-D 218+440 keV photon proxy\n"
                     f"# angle_deg={angle_deg:.0f}; FOV=300x300x{(z1-z0)*NATIVE_MM:g} mm; r<=150 mm\n"
                     f"# XCAT z=[{z0},{z1}), x/y=[28,228), calibrated voxel=1.5 mm\n"
                     f"# 1 beamOn event = 1 emitted primary gamma, not 1 Ac-225 decay\n"
                     f"# gamma yields: Fr-221/218={YIELDS[218]}, Bi-213/440={YIELDS[440]}\n"
                     f"# one worker: {beam_on} photons; {workers} workers/view\n"
                     f"/xcat/clear\n/xcat/angle {angle_deg:.0f}\n\n")
        for box in boxes:
            intensity = box.intensity
            if intensity <= 0:
                continue
            energy_intensities[box.energy] += intensity
            stream.write(f"/xcat/add {box.energy} {box.x:.9f} {box.y:.9f} {box.z:.9f} "
                         f"{box.hx:.9f} {box.hy:.9f} {box.hz:.9f} {intensity:.15g}\n")
        stream.write(f"/run/beamOn {beam_on}\n")
    return {"file": path.name, "angle_deg": angle_deg, "sources": len(boxes),
            "beam_on_per_worker": beam_on, "energy_intensities": energy_intensities,
            "sha256": sha256_file(path)}


def preview(path: Path, fr: np.ndarray, bi: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    lesion_3mm = block_mean(masks["lesion"].astype(np.float64))
    mid = fr.shape[0] // 2
    halfheight = fr.shape[0] * GRID_MM / 2
    panels = [(fr[mid], "Fr-221 axial, centre z", (-150, 150, -150, 150), "y (mm)"),
              (bi[mid], "Bi-213 axial, centre z", (-150, 150, -150, 150), "y (mm)"),
              (bi[mid] - fr[mid], "Bi minus Fr, axial", (-150, 150, -150, 150), "y (mm)"),
              (fr.sum(axis=1), "Fr-221 coronal projection", (-150, 150, -halfheight, halfheight), "z (mm)"),
              (bi.sum(axis=1), "Bi-213 coronal projection", (-150, 150, -halfheight, halfheight), "z (mm)"),
              (lesion_3mm.sum(axis=1), "Bone lesion, coronal projection", (-150, 150, -halfheight, halfheight), "z (mm)")]
    for ax, (array, title, extent, ylabel) in zip(axes.flat, panels):
        image = ax.imshow(array, origin="lower", extent=extent, cmap="magma", interpolation="nearest", aspect="auto")
        ax.set(title=title, xlabel="x (mm)", ylabel=ylabel)
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle(f"XCAT abdominal crop: {2*halfheight:g} mm; boundary contact reported in manifest")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build(xif: Path, out: Path, z0: int = Z0, z1: int = Z1, total_photons: int = TOTAL_PHOTONS, workers: int = WORKERS_PER_VIEW) -> None:
    if z0 < 0 or z1 <= z0 or (z1-z0) % 2 or total_photons <= 0 or workers <= 0 or total_photons % (VIEWS*workers):
        raise ValueError("Invalid crop or photon/worker allocation")
    if not 1 <= total_photons // (VIEWS*workers) <= 2147483647:
        raise ValueError("beamOn must fit a positive signed 32-bit integer")
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Refusing to replace an existing experiment: {out}")
    dims, header_mm, offset = parse_xif(xif)
    if header_mm != (1.0, 1.0, 1.0):
        raise ValueError(f"Unexpected XIF header voxel size: {header_mm}")
    volume = np.memmap(xif, dtype="<f4", mode="r", offset=offset,
                       shape=(dims[2], dims[1], dims[0]))
    codes = np.asarray(volume[z0:z1, XY0:XY1, XY0:XY1], dtype=np.int16)
    fr_native, bi_native, masks, support = native_phantom(codes, z0, z1)
    fr, bi = block_mean(fr_native), block_mean(bi_native)
    fr_boxes = boxify(218, fr, fr_native, support)
    bi_boxes = boxify(440, bi, bi_native, support)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "truth_3mm.npz", fr_zyx=fr.astype(np.float32),
                        bi_zyx=bi.astype(np.float32),
                        fr_xyz=np.transpose(fr, (2, 1, 0)).astype(np.float32),
                        bi_xyz=np.transpose(bi, (2, 1, 0)).astype(np.float32),
                        x_mm=(np.arange(100) + 0.5 - 50) * GRID_MM,
                        y_mm=(np.arange(100) + 0.5 - 50) * GRID_MM,
                        z_mm=(np.arange(fr.shape[0]) + 0.5 - fr.shape[0]/2) * GRID_MM,
                        **{name + "_fraction_zyx": block_mean(array.astype(np.float64)).astype(np.float32)
                           for name, array in masks.items()})
    np.savez_compressed(out / "masks_1p5mm.npz", xcat_codes_zyx=codes,
                        **{name + "_zyx": array for name, array in masks.items()})
    preview(out / "preview.png", fr, bi, masks)
    macros = []
    for view in range(VIEWS):
        macros.append(write_macro(out / f"view_{view + 1:02d}_mixed.mac",
                                  fr_boxes + bi_boxes, view * 360.0 / VIEWS,
                                  total_photons // VIEWS // workers, z0, z1, workers))
    for energy, boxes in ((218, fr_boxes), (440, bi_boxes)):
        macros.append(write_macro(out / f"view_01_{energy}keV_check.mac", boxes, 0.0, 10000, z0, z1, workers))
    kidney_full = int(np.count_nonzero(volume == 75))
    kidney_kept = int(masks["kidney"].sum())
    coverage = {name: {"touches_z_min": bool(mask[0].any()),
                       "touches_z_max": bool(mask[-1].any()),
                       "retained_voxels": int(mask.sum())}
                for name, mask in masks.items()}
    coverage["kidney"]["fraction_of_full_xcat_kidneys"] = kidney_kept / kidney_full if kidney_full else 0.0
    # Lesion ROI support is an analytic ellipsoid, evaluated beyond the crop.
    lesion_z_min = LESION_CENTER_XYZ[2] - 18.0 / NATIVE_MM
    lesion_z_max = LESION_CENTER_XYZ[2] + 18.0 / NATIVE_MM
    coverage["lesion"]["axial_roi_fully_inside"] = bool(z0 <= lesion_z_min and z1 > lesion_z_max)
    manifest = {"organ_coverage": coverage, "height_mm": (z1-z0)*NATIVE_MM,
        "xif": str(xif.resolve()), "xif_sha256": sha256_file(xif),
        "xif_dimensions_xyz": dims, "xif_header_voxel_mm": header_mm,
        "calibrated_voxel_mm": NATIVE_MM,
        "calibration_basis": "1178 occupied axial slices imply 1767 mm adult body height at 1.5 mm; transverse width 384 mm",
        "crop_half_open": {"z": [z0, z1], "x": [XY0, XY1], "y": [XY0, XY1]},
        "target_shape_zyx": list(fr.shape), "target_spacing_mm": GRID_MM,
        "fov_center_geant4_mm": FOV_CENTER_MM, "physical_radius_mm": FOV_RADIUS_MM,
        "lesion_center_xcat_xyz": LESION_CENTER_XYZ, "lesion_fwhm_mm": LESION_FWHM_MM,
        "relative_activity_parameters": ACTIVITY, "gamma_yields": YIELDS,
        "activity_units": "illustrative SUV-like relative emission density, not patient-specific Bq/mm3",
        "source_interface": "Geant4Code /xcat/add weighted cuboids; one gamma per event; 4pi isotropic",
        "native_organ_voxels": {key: int(mask.sum()) for key, mask in masks.items()},
        "integrated_activity_mm3": {"fr": float(fr.sum() * 27), "bi": float(bi.sum() * 27)},
        "source_counts": {"fr": len(fr_boxes), "bi": len(bi_boxes)},
        "views": VIEWS, "workers_per_view": workers,
        "total_primary_photons_all_workers_views": total_photons,
        "macros": macros,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("native_organ_voxels", "integrated_activity_mm3", "source_counts")}, indent=2))
    print(f"Wrote {len(macros)} macros to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xif", type=Path, default=DEFAULT_XIF)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--total-photons", type=int, default=TOTAL_PHOTONS)
    parser.add_argument("--workers-per-view", type=int, default=WORKERS_PER_VIEW)
    args = parser.parse_args()
    z0, z1 = Z0, Z1
    out = args.output or DEFAULT_OUT
    if args.config:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        z0, z1 = cfg["xcat_crop_z"]
        if cfg["xcat_crop_xy"] != [XY0, XY1] or cfg["native_spacing_mm"] != NATIVE_MM or cfg["truth_spacing_mm"] != GRID_MM:
            raise ValueError("This XCAT importer requires the established transverse crop and spacing")
        if (z1-z0)*NATIVE_MM != cfg["height_mm"] or cfg["gamma_yields"] != {str(k): v for k,v in YIELDS.items()}:
            raise ValueError("XCAT crop/yields do not match experiment")
        out = args.output or ROOT / "experiments" / cfg["experiment_id"] / "generated" / "XCAT"
    build(args.xif, out, z0, z1, args.total_photons, args.workers_per_view)


if __name__ == "__main__":
    main()
