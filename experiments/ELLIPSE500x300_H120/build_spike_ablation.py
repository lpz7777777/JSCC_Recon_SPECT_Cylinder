"""Freeze physical-face graph and density ties for the NEMA spike ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from geometry import grid

HERE = Path(__file__).resolve().parent
STUDY = "NEMA_5e9_SPIKE_ABLATION_V1"
VARIANTS = [
    {"id": "bind_f010", "method": "binding", "fraction_threshold": .1},
    {"id": "huber_weak", "method": "huber", "strength": .001, "huber_delta": 1.},
    {"id": "huber_medium", "method": "huber", "strength": .01, "huber_delta": 1.},
    {"id": "huber_strong", "method": "huber", "strength": .1, "huber_delta": 1.},
    {"id": "tv_medium", "method": "tv", "strength": .01},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admitted_angle(lo, hi, radius, axes):
    a, b = axes
    if radius <= b:
        return max(0., hi-lo)
    if radius >= a:
        return 0.
    c = (1/radius**2-1/b**2)/(1/a**2-1/b**2)
    limit = np.arccos(np.sqrt(np.clip(c, 0, 1)))
    return sum(max(0., min(hi, center+limit)-max(lo, center-limit))
               for center in (-2*np.pi, -np.pi, 0, np.pi, 2*np.pi))


def make_graph(config, geometry):
    coordinates, cells, rings = grid(config)
    if not np.allclose(coordinates, geometry["coordinates_mm"], atol=1e-12):
        raise ValueError("Configured cells differ from frozen geometry")
    active = geometry["active_indices"]
    f = geometry["ellipse_fraction"]
    volume = geometry["cell_volume_mm3"]*f
    nxy = config["points_per_layer"]
    nz = len(coordinates)//nxy
    dz = config["z_spacing_mm"]
    axes = config["semi_axes_mm"]
    active_one = active[active < nxy]
    index = {int(full): i for i, full in enumerate(active_one)}
    edges = {}

    def add(i, j, area):
        if i == j or area <= 1e-12 or i not in index or j not in index:
            return
        key = tuple(sorted((index[i], index[j])))
        edges[key] = edges.get(key, 0.)+area

    for start, count in rings:
        width = 2*np.pi/count
        rin, rout = cells[start, :2]
        for i in range(count):
            theta = (i+.5)*width
            rmax = 1/np.sqrt((np.cos(theta)/axes[0])**2+(np.sin(theta)/axes[1])**2)
            add(start+i, start+(i+1)%count, max(0., min(rout, rmax)-rin)*dz)
    first, count = rings[0]
    for j in range(count):
        add(0, first+j, 3*2*np.pi/count*dz)
    for (left, nleft), (right, nright) in zip(rings[:-1], rings[1:]):
        radius = cells[left, 1]
        for i in range(nleft):
            if left+i not in index:
                continue
            lo, hi = cells[left+i, 2:]
            for j in range(nright):
                if right+j not in index:
                    continue
                angle = 0.
                for shift in (-2*np.pi, 0, 2*np.pi):
                    lo2, hi2 = cells[right+j, 2:]+shift
                    low, high = max(lo, lo2), min(hi, hi2)
                    if high > low:
                        angle += admitted_angle(low, high, radius, axes)
                add(left+i, right+j, radius*angle*dz)
    n = len(active_one)
    pairs = np.asarray(list(edges), dtype=np.int64)
    area = np.asarray(list(edges.values()), dtype=np.float64)
    all_pairs = [pairs+k*n for k in range(nz)]
    all_areas = [area for _ in range(nz)]
    for k in range(nz-1):
        all_pairs.append(np.column_stack((np.arange(n)+k*n, np.arange(n)+(k+1)*n)))
        all_areas.append(volume[active_one]/dz)
    pairs = np.concatenate(all_pairs)
    area = np.concatenate(all_areas)
    xyz = coordinates[active]
    distance = np.linalg.norm(xyz[pairs[:, 0]]-xyz[pairs[:, 1]], axis=1)
    if np.any(distance <= 0):
        raise ValueError("Zero edge distance")
    # R(u) = sum (face area * distance / V_FOV) rho(3mm * delta_u / distance).
    weight = area*distance/volume[active].sum()
    gradient = 3./distance
    anchor = np.arange(len(active), dtype=np.int64)
    small = f[active] < .1
    full = f[active] >= 1-1e-9
    for z in np.unique(xyz[:, 2]):
        candidates = np.flatnonzero(full & (xyz[:, 2] == z))
        targets = np.flatnonzero(small & (xyz[:, 2] == z))
        _, nearest = cKDTree(xyz[candidates, :2]).query(xyz[targets, :2])
        anchor[targets] = candidates[nearest]
    _, groups = np.unique(anchor, return_inverse=True)
    return dict(edge_i=pairs[:, 0], edge_j=pairs[:, 1], face_area_mm2=area,
                distance_mm=distance, graph_weight=weight, gradient_scale=gradient,
                binding_group=groups.astype(np.int64), binding_anchor=anchor,
                effective_volume_mm3=volume[active])


def main():
    output = HERE/"generated/SpikeAblation"/STUDY
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((HERE/"config.json").read_text())
    geometry_path = HERE/"generated/Geometry/geometry.npz"
    with np.load(geometry_path) as geometry:
        arrays = make_graph(config, geometry)
        xyz = geometry["coordinates_mm"][geometry["active_indices"]]
    output.mkdir(parents=True)
    np.savez_compressed(output/"spatial_model.npz", **arrays)
    baseline_path=HERE/"generated/RemoteResults/NEMA_Body_H60_5e9_1644876/run_manifest.json"
    baseline=json.loads(baseline_path.read_text())
    record = {"study": STUDY, "dataset": "NEMA_Body_H60", "level": "5e9",
              "baseline": "NEMA_Body_H60_5e9_1644876", "variants": VARIANTS,
              "geometry_sha256": sha(geometry_path),
              "spatial_model_sha256": sha(output/"spatial_model.npz"),
              "builder_sha256": sha(__file__), "active_cells": len(xyz),
              "baseline_input_sha256":baseline["input_sha256"],
              "baseline_factor_manifest_sha256":baseline["factor_manifest_sha256"],
              "baseline_accepted_compton_events":baseline["accepted_compton_events"],
              "baseline_sensi_d_sha256":baseline["sensi_d_sha256"],
              "baseline_run_manifest_sha256":sha(baseline_path),
              "edges": len(arrays["edge_i"]),
              "bound_cells": int(np.count_nonzero(arrays["binding_anchor"] != np.arange(len(xyz)))),
              "binding_parameters": int(arrays["binding_group"].max()+1),
              "max_binding_distance_mm": float(np.linalg.norm(xyz-xyz[arrays['binding_anchor']], axis=1).max()),
              "reference_length_mm": 3., "iterations": 10000, "save_step": 50,
              "inner_max": 80, "inner_gap_tolerance": 1e-6,
              "penalty": "finite-volume physical-face graph; anisotropic 3D graph TV or Huber; no exterior zero neighbours",
              "normalization": "Per channel: C=observed count total, alpha=C/sum(sensitivity); u=x/alpha; objective=L(x)/C+strength*R(u). The same frozen spatial model applies to all four reconstructed component images.",
              "gate": "10 iterations full events/grid -> 200 iterations -> 10000; six arrays, identical events/hashes, surrogate/objective descent, GPU and host <=80%"}
    (output/"study.json").write_text(json.dumps(record, indent=2)+"\n")
    report = HERE/"reports/NEMA_Body_H60/spike_ablation"
    report.mkdir(exist_ok=True)
    (report/"study.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
