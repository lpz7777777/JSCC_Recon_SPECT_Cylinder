#!/usr/bin/env python3
"""Validate retained PE v4 GPU detector rows against CPU reference details."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VOXEL_COUNT = 51 * 51 * 20


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-details",
        type=Path,
        default=repo
        / "Results"
        / "Analysis"
        / "PEV4ProductionSampling_20260718_JSCC218"
        / "pe_v4_reference_details.csv",
    )
    parser.add_argument(
        "--gpu-root",
        type=Path,
        default=repo
        / "Results"
        / "Analysis"
        / "PEV4GPUValidation_20260718_JSCC218",
    )
    parser.add_argument("--production-face-level", type=int, default=16)
    parser.add_argument("--reference-face-level", type=int, default=32)
    parser.add_argument("--maximum-gpu-cpu-relative-error", type=float, default=1e-4)
    parser.add_argument("--maximum-production-convergence", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    details = pd.read_csv(args.reference_details)
    production = details[
        details.face_subdivisions == args.production_face_level
    ].copy()
    reference = details[
        details.face_subdivisions == args.reference_face_level
    ].copy()
    key = ["layer_y_mm", "source_y_mm", "detector_index", "voxel_index"]
    merged = production.merge(reference, on=key, suffixes=("_production", "_reference"))
    if len(merged) != 8:
        raise ValueError(f"Expected eight representative pairs; found {len(merged)}")

    rows = []
    for item in merged.itertuples(index=False):
        detector_index = int(item.detector_index)
        path = (
            args.gpu_root
            / f"detector_{detector_index}_face{args.production_face_level}"
            / "PE_v4_rows.sysmat"
        )
        values = np.fromfile(path, dtype="<f4")
        if values.shape != (VOXEL_COUNT,):
            raise ValueError(f"{path} has shape {values.shape}")
        gpu_value = float(values[int(item.voxel_index)])
        cpu_production = float(item.photoelectric_probability_production)
        cpu_reference = float(item.photoelectric_probability_reference)
        gpu_cpu_error = abs(gpu_value / cpu_production - 1.0)
        convergence = abs(gpu_value / cpu_reference - 1.0)
        rows.append(
            {
                "layer_y_mm": float(item.layer_y_mm),
                "source_y_mm": float(item.source_y_mm),
                "detector_index": detector_index,
                "voxel_index": int(item.voxel_index),
                "gpu_production": gpu_value,
                "cpu_same_sampling": cpu_production,
                "cpu_fine_reference": cpu_reference,
                "gpu_cpu_relative_error": gpu_cpu_error,
                "production_convergence": convergence,
                "passed": gpu_cpu_error <= args.maximum_gpu_cpu_relative_error
                and convergence <= args.maximum_production_convergence,
            }
        )

    table = pd.DataFrame(rows)
    passed = bool(table.passed.all())
    output = args.output or args.gpu_root / "gpu_cpu_validation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "production_face_level": args.production_face_level,
        "reference_face_level": args.reference_face_level,
        "maximum_gpu_cpu_relative_error_allowed": args.maximum_gpu_cpu_relative_error,
        "maximum_production_convergence_allowed": args.maximum_production_convergence,
        "observed_maximum_gpu_cpu_relative_error": float(
            table.gpu_cpu_relative_error.max()
        ),
        "observed_maximum_production_convergence": float(
            table.production_convergence.max()
        ),
        "pairs": rows,
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table.to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
