"""Stream a complete Cartesian response into ellipse-experiment raw Factors.

The source .sysmat is detector-major with x varying fastest. The output keeps
the existing density-basis, detector-fast on-disk layout; ellipse fractions are
stored separately and are applied only by the reconstruction operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from geometry import generate, ring_counts

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def stencil(coordinates, axis):
    n = len(axis)
    spacing = axis[1] - axis[0]
    xy = (coordinates[:, :2] - axis[0]) / spacing
    if np.any(xy < -1e-9) or np.any(xy > n - 1 + 1e-9):
        raise ValueError("Polar coordinates exceed Cartesian matrix grid")
    xy = np.clip(xy, 0, n - 1)
    ix = np.minimum(np.floor(xy[:, 0]).astype(np.int32), n - 2)
    iy = np.minimum(np.floor(xy[:, 1]).astype(np.int32), n - 2)
    tx = (xy[:, 0] - ix).astype(np.float32)
    ty = (xy[:, 1] - iy).astype(np.float32)
    return ix, iy, tx, ty


def bilinear(layer, indices):
    ix, iy, tx, ty = indices
    a = layer[:, iy, ix] * (1 - tx) + layer[:, iy, ix + 1] * tx
    b = layer[:, iy + 1, ix] * (1 - tx) + layer[:, iy + 1, ix + 1] * tx
    return a * (1 - ty) + b * ty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysmat", type=Path, required=True)
    parser.add_argument("--params-image", type=Path, required=True)
    parser.add_argument("--params-detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response", choices=("A218", "A440", "C440to218"), required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--detector-block", type=int, default=64)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    params = np.fromfile(args.params_image, dtype="<f4")
    nx, ny, nz = map(int, params[:3])
    if (nx, ny, nz) != (85, 85, 40) or params[11] != 270:
        raise ValueError("Cartesian grid or source-to-collimator distance mismatch")
    raw_det = np.fromfile(args.params_detector, dtype="<f4")
    ndet = int(raw_det[0])
    detector = raw_det[1:].reshape(ndet, 12)
    selected = np.flatnonzero(detector[:, 11] == 1)
    if ndet != 11520 or len(selected) != config["detector_count"]:
        raise ValueError("Detector count or crystal filtering mismatch")
    if args.sysmat.stat().st_size != ndet * nx * ny * nz * 4:
        raise ValueError("Raw Cartesian matrix size mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    axis = np.arange(nx) * float(params[3]) - (nx - 1) * float(params[3]) / 2 + float(params[8])
    if not np.array_equal(axis, np.arange(-252, 253, 6)):
        raise ValueError("Cartesian axis mismatch")
    coords, volumes, fractions, active, rotation, inverse = generate(config)
    per_layer = config["points_per_layer"]
    indices = stencil(coords[:per_layer], axis)
    raw = np.memmap(args.sysmat, mode="r", dtype="<f4", shape=(ndet, nz, ny, nx))
    staging = args.output.with_name(args.output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        output = np.memmap(staging / "SysMat_polar", mode="w+", dtype="<f4",
                           shape=(len(coords), len(selected)))
        for z in range(nz):
            destination = output[z * per_layer:(z + 1) * per_layer]
            for offset in range(0, len(selected), args.detector_block):
                subset = selected[offset:offset + args.detector_block]
                values = bilinear(raw[subset, z], indices)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError(f"Invalid response at z={z}, detector={offset}")
                destination[:, offset:offset + len(subset)] = (
                    values.T * volumes[z * per_layer:(z + 1) * per_layer, None])
            output.flush()
            print(f"Completed axial layer {z + 1}/{nz}", flush=True)
        del output
        np.savetxt(staging / "coor_polar_full.csv", coords, delimiter=",", fmt="%.12g")
        np.savetxt(staging / "RotMat_full.csv", rotation + 1, delimiter=",", fmt="%d")
        np.savetxt(staging / "RotMatInv_full.csv", inverse + 1, delimiter=",", fmt="%d")
        volumes.astype("<f8").tofile(staging / "polar_cell_volume_mm3.float64")
        np.savetxt(staging / "polar_cell_volume_mm3.csv", volumes, delimiter=",", fmt="%.12g")
        fractions.astype("<f8").tofile(staging / "ellipse_fraction.float64")
        active.astype("<i4").tofile(staging / "ellipse_active_indices.int32")
        with (staging / "Detector.csv").open("w", newline="") as stream:
            stream.write("index,x,y,z\n")
            xyz = detector[selected, :3].astype(np.float64)
            xyz[:, 1] += params[11]
            for index, position in enumerate(xyz, 1):
                stream.write(f"{index},{position[0]:.12g},{position[1]:.12g},{position[2]:.12g}\n")
        if not np.array_equal(np.unique(np.abs(xyz[:, 1])),
                              config["detector_y_mm"]):
            raise ValueError("Unexpected detector normal distances")
        volume_record = {"support_radius_mm": 255, "height_mm": 120,
                         "sum_mm3": float(volumes.sum()),
                         "ellipse_effective_volume_mm3": float(np.dot(volumes, fractions))}
        (staging / "polar_cell_volume_manifest.json").write_text(
            json.dumps(volume_record, indent=2) + "\n")
        manifest = {"experiment": config["experiment_id"], "response": args.response,
                    "pixel_num": len(coords), "detector_num": len(selected),
                    "rotate_num": config["rotate_num"], "maps_activity_density": True,
                    "calibration": {"enabled": False},
                    "underlying_point_response_normalization": "per emitted monoenergetic source photon",
                    "polar_volume_weighting": {"enabled": True,
                        "forward_model": "y = A * diag(DeltaV_mm3) * rho"},
                    "ellipse_fraction_applied": False,
                    "input_sha256": {"matrix": digest(args.sysmat),
                                     "params_image": digest(args.params_image),
                                     "params_detector": digest(args.params_detector)},
                    "ring_counts": ring_counts(config)}
        (staging / "factor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staging, args.output)
    except Exception:
        print(f"Incomplete staging retained for inspection: {staging}")
        raise


if __name__ == "__main__":
    main()
