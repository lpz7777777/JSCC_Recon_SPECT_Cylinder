"""Generate the circular response grid and object-frame ellipse overlap.

All arrays are independent of FOV120. Rotation maps are 0-based internally and
1-based in the on-disk CSV format expected by the existing Factors pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def ring_counts(config):
    result = []
    for radius in np.arange(6, 253, 6):
        if radius <= 150:
            # Preserve all 25 FOV120 rings byte for byte.
            count = 20 * int(np.ceil((radius / 6) / (25 / 4)))
        else:
            count = next((int(n) for lo, hi, n in config["outer_ring_counts"]
                          if lo <= radius <= hi), None)
        if count is None or count % config["rotate_num"]:
            raise ValueError(f"Invalid angular count at radius {radius}")
        result.append((int(radius), count))
    return result


def grid(config):
    rings = ring_counts(config)
    xy = [(0.0, 0.0)]
    cells = [(0.0, 3.0, 0.0, 2 * np.pi)]
    slices = []
    offset = 1
    for radius, n in rings:
        angle = 2 * np.pi * np.arange(n) / n
        xy.extend(zip(radius * np.cos(angle), radius * np.sin(angle)))
        width = 2 * np.pi / n
        cells.extend((radius - 3, radius + 3, t - width / 2, t + width / 2)
                     for t in angle)
        slices.append((offset, n))
        offset += n
    if len(xy) != config["points_per_layer"]:
        raise ValueError(f"Grid has {len(xy)} points instead of configured count")
    xy = np.asarray(xy, dtype=np.float64)
    z = (np.arange(int(config["height_mm"] / config["z_spacing_mm"])) + .5) * config["z_spacing_mm"] - config["height_mm"] / 2
    coordinates = np.column_stack((np.tile(xy, (len(z), 1)),
                                   np.repeat(z, len(xy))))
    return coordinates, np.asarray(cells), slices


def overlap(config, cells, order):
    """Integrate the ellipse intersection over each polar sector in r²,theta."""
    a, b = config["semi_axes_mm"]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    lower, upper, start, end = cells.T
    angles = (start[:, None] + end[:, None]) / 2 + (end - start)[:, None] * nodes / 2
    radial_sq = 1 / ((np.cos(angles) / a)**2 + (np.sin(angles) / b)**2)
    admitted = np.clip(radial_sq - lower[:, None]**2,
                       0, upper[:, None]**2 - lower[:, None]**2)
    integrated = (admitted @ weights) / 2
    area = (upper**2 - lower**2) * (end - start) / 2
    fraction = integrated / (upper**2 - lower**2)
    return area, np.clip(fraction, 0, 1)


def rotations(config, slices):
    count = config["points_per_layer"]
    views = config["rotate_num"]
    one = np.arange(count, dtype=np.int32)
    rot = np.empty((count, views), dtype=np.int32)
    for view in range(views):
        rot[:, view] = one
        for start, n in slices:
            rot[start:start+n, view] = start + (np.arange(n) + view * n // views) % n
    inv = np.argsort(rot, axis=0).astype(np.int32)
    return rot, inv


def generate(config, order=64):
    coords, cells, slices = grid(config)
    area, fraction = overlap(config, cells, order)
    volume_one = area * config["z_spacing_mm"]
    fractions = np.tile(fraction, int(config["height_mm"] / config["z_spacing_mm"]))
    volumes = np.tile(volume_one, int(config["height_mm"] / config["z_spacing_mm"]))
    rot, inv = rotations(config, slices)
    nz = int(config["height_mm"] / config["z_spacing_mm"])
    offsets = np.arange(nz, dtype=np.int32)[:, None, None] * config["points_per_layer"]
    rot_full = (rot[None, :, :] + offsets).reshape(-1, config["rotate_num"])
    inv_full = (inv[None, :, :] + offsets).reshape(-1, config["rotate_num"])
    active = np.flatnonzero(fractions > 1e-13).astype(np.int32)
    return coords, volumes, fractions, active, rot_full, inv_full


def digest(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, default=HERE / "generated/Geometry")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    coords, volumes, fractions, active, rot, inv = generate(config)
    if not np.all(rot[inv, np.arange(config["rotate_num"])] ==
                  np.arange(len(coords))[:, None]):
        raise ValueError("Rotation/inverse mismatch")
    if not np.allclose(volumes[rot], volumes[:, None], rtol=0, atol=1e-10):
        raise ValueError("Rotations change cell volumes")
    expected = np.pi * np.prod(config["semi_axes_mm"]) * config["height_mm"]
    effective = np.dot(volumes, fractions)
    if abs(effective / expected - 1) > .001:
        raise ValueError(f"Ellipse volume error: {effective / expected - 1:.3%}")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "geometry.npz", coordinates_mm=coords,
                        cell_volume_mm3=volumes, ellipse_fraction=fractions,
                        active_indices=active, rotation=rot, inverse_rotation=inv)
    manifest = {"experiment_id": config["experiment_id"],
                "points_per_layer": config["points_per_layer"],
                "points_full": len(coords), "points_active": len(active),
                "z_layers": int(config["height_mm"] / config["z_spacing_mm"]),
                "views": config["rotate_num"], "ellipse_volume_mm3": effective,
                "ellipse_volume_relative_error": effective / expected - 1,
                "arrays_sha256": {name: digest(array) for name, array in
                                  (("coordinates_mm", coords), ("cell_volume_mm3", volumes),
                                   ("ellipse_fraction", fractions), ("active_indices", active),
                                   ("rotation", rot), ("inverse_rotation", inv))}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
