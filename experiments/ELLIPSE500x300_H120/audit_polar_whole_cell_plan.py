"""Read-only geometry/volume audit for the proposed whole polar-cell support.

Writes only a small planning receipt. Does not change production geometry,
Factors, sensitivity, reconstruction images or event selection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from geometry import digest, generate


def main():
    root = Path(__file__).resolve().parent
    config_file = root / "config.json"
    config = json.loads(config_file.read_text())
    coordinates, volumes, fractions, old_active, rotation, inverse = generate(config)
    a, b = config["semi_axes_mm"]
    radius_sq = (coordinates[:, 0] / a)**2 + (coordinates[:, 1] / b)**2
    tolerance = 1e-12
    selected = radius_sq <= 1 + tolerance
    active = np.flatnonzero(selected).astype(np.int32)
    nominal_volume = float(np.pi * a * b * config["height_mm"])
    actual_volume = float(volumes[selected].sum())
    identity = np.arange(len(coordinates))[:, None]
    inverse_ok = bool(np.all(rotation[inverse, np.arange(config["rotate_num"])] == identity))
    volume_ok = bool(np.allclose(volumes[rotation], volumes[:, None], rtol=0, atol=1e-10))
    if not inverse_ok or not volume_ok:
        raise ValueError("Existing full-circle rotation contract failed")
    report = {
        "status": "PLANNING_AUDIT_ONLY_NOT_DEPLOYED",
        "rule": "Keep full polar sector iff its representative centre has normalized ellipse radius squared <= 1 + 1e-12",
        "tolerance": tolerance,
        "points_full": len(coordinates), "old_active_cells": len(old_active),
        "proposed_active_cells": len(active), "z_layers": 40, "views": config["rotate_num"],
        "nominal_ellipse_volume_mm3": nominal_volume,
        "proposed_union_volume_mm3": actual_volume,
        "volume_relative_error": actual_volume / nominal_volume - 1,
        "old_min_effective_volume_mm3": float((volumes * fractions)[old_active].min()),
        "proposed_min_cell_volume_mm3": float(volumes[selected].min()),
        "proposed_max_cell_volume_mm3": float(volumes[selected].max()),
        "old_positive_fraction_below_0_1": int(((fractions > 1e-13) & (fractions < .1)).sum()),
        "selected_cells_crossing_nominal_ellipse": int((selected & (fractions < 1-1e-13)).sum()),
        "rotation_inverse_passed": inverse_ok, "rotation_volume_conservation_passed": volume_ok,
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256((root / "geometry.py").read_bytes()).hexdigest(),
        "arrays_sha256": {"coordinates": digest(coordinates), "full_cell_volume": digest(volumes),
                          "proposed_active_indices": digest(active), "rotation": digest(rotation)},
        "warning": "Small total volume error is not a local boundary accuracy or imaging performance certificate. Polar cell volumes remain unequal.",
    }
    destination = root / "reports/NEMA_Body_H60/process_list_global_audit_v4"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "whole_cell_planning_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
