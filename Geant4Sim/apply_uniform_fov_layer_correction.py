#!/usr/bin/env python3
"""Apply absolute Uniform-FOV Geant4/Matrix layer scales to JSCC Factors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PIXEL_COUNT = 25620
DETECTOR_COUNT = 10496
RESPONSES = {
    "A218": "218keV_RotateNum20",
    "A440": "440keV_RotateNum20",
    "C440to218": "440keV_to218win_RotateNum20",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors-dir", type=Path, default=repo / "Factors")
    parser.add_argument(
        "--layer-metrics",
        type=Path,
        default=repo
        / "Results"
        / "Analysis"
        / "UniformFov_PEv3_vs_PEv4_SymmetricHalton"
        / "layer_metrics.csv",
    )
    parser.add_argument("--source-suffix", default="CenterPoint_PEv4")
    parser.add_argument(
        "--target-suffix", default="CenterPoint_PEv4_UniformFOVLayer"
    )
    parser.add_argument("--factor-set", default="candidate")
    parser.add_argument("--chunk-rows", type=int, default=128)
    return parser.parse_args()


def factor_name(prefix: str, suffix: str) -> str:
    suffix = suffix.strip("_")
    return f"{prefix}_{suffix}" if suffix else prefix


def load_layer_factors(
    metrics: pd.DataFrame, response: str, factor_set: str
) -> tuple[np.ndarray, pd.DataFrame]:
    table = metrics[
        (metrics.response == response) & (metrics.factor_set == factor_set)
    ].copy()
    table = table.sort_values("layer_y_mm")
    expected_layers = np.array([30, 60, 90, 120])
    if len(table) != 4 or not np.array_equal(
        table.layer_y_mm.to_numpy(dtype=int), expected_layers
    ):
        raise ValueError(f"Expected layers {expected_layers.tolist()} for {response}")
    factors = table.geant4_over_matrix.to_numpy(dtype=np.float64)
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError(f"Invalid layer factors for {response}: {factors}")
    return factors, table


def copy_metadata(source: Path, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=False)
    excluded = {"SysMat_polar", "SysMat_tmp", "factor_manifest.json"}
    for item in source.iterdir():
        if item.is_file() and item.name not in excluded:
            shutil.copy2(item, staging / item.name)


def correct_matrix(
    source: Path,
    target: Path,
    detector_correction: np.ndarray,
    chunk_rows: int,
) -> tuple[float, float]:
    expected_bytes = PIXEL_COUNT * DETECTOR_COUNT * np.dtype("<f4").itemsize
    if source.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected matrix size for {source}")
    src = np.memmap(
        source, dtype="<f4", mode="r", shape=(PIXEL_COUNT, DETECTOR_COUNT)
    )
    dst = np.memmap(
        target, dtype="<f4", mode="w+", shape=(PIXEL_COUNT, DETECTOR_COUNT)
    )
    source_sum = 0.0
    corrected_sum = 0.0
    for start in range(0, PIXEL_COUNT, chunk_rows):
        stop = min(PIXEL_COUNT, start + chunk_rows)
        chunk = np.asarray(src[start:stop], dtype=np.float32)
        corrected = chunk * detector_correction[np.newaxis, :]
        dst[start:stop] = corrected
        source_sum += float(chunk.sum(dtype=np.float64))
        corrected_sum += float(corrected.sum(dtype=np.float64))
    dst.flush()
    del dst, src
    return source_sum, corrected_sum


def build_one(
    factors_dir: Path,
    metrics: pd.DataFrame,
    response: str,
    prefix: str,
    args: argparse.Namespace,
) -> None:
    source = factors_dir / factor_name(prefix, args.source_suffix)
    target = factors_dir / factor_name(prefix, args.target_suffix)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)

    detector = pd.read_csv(source / "Detector.csv")
    if len(detector) != DETECTOR_COUNT:
        raise ValueError(f"Unexpected detector count in {source}")
    layer_factors, layer_table = load_layer_factors(
        metrics, response, args.factor_set
    )
    factor_by_layer = dict(zip((30, 60, 90, 120), layer_factors))
    detector_correction = detector.y.map(factor_by_layer).to_numpy(dtype=np.float32)
    if not np.isfinite(detector_correction).all():
        raise ValueError(f"Detector layer mapping failed for {response}")

    staging = target.with_name(f".build_{target.name}_{os.getpid()}")
    try:
        copy_metadata(source, staging)
        pd.DataFrame(
            {
                "detector_index": detector["index"],
                "x": detector.x,
                "y": detector.y,
                "z": detector.z,
                "correction_factor": detector_correction,
            }
        ).to_csv(staging / "correction_vector.csv", index=False)
        source_sum, corrected_sum = correct_matrix(
            source / "SysMat_polar",
            staging / "SysMat_polar",
            detector_correction,
            args.chunk_rows,
        )
        parent = json.loads(
            (source / "factor_manifest.json").read_text(encoding="utf-8")
        )
        manifest = dict(parent)
        manifest["parent_factor_dir"] = str(source.resolve())
        manifest["parent_calibration"] = parent.get("calibration")
        manifest["grid"] = dict(parent["grid"])
        manifest["grid"]["output_suffix"] = args.target_suffix
        manifest["calibration"] = {
            "enabled": True,
            "name": f"JSCC_{response}_PEv4_UniformFOVAbsoluteLayer_20260718",
            "method": "per-detector-row multiplication by layer Geant4/Matrix ratio",
            "source": str(args.layer_metrics.resolve()),
            "factor_set": args.factor_set,
            "total_efficiency_preserved": False,
            "layer_y_mm": layer_table.layer_y_mm.astype(int).tolist(),
            "incremental_layer_factor": layer_factors.tolist(),
            "source_matrix_sum": source_sum,
            "corrected_matrix_sum": corrected_sum,
            "relative_total_change": corrected_sum / source_sum - 1.0,
            "correction_csv": "correction_vector.csv",
        }
        manifest["cartesian_matrix_included"] = False
        manifest["generated_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        (staging / "factor_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"Created {target.name}: response={response}, "
        f"layers={layer_factors.tolist()}, "
        f"total_change={corrected_sum / source_sum - 1.0:+.6%}"
    )


def main() -> int:
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    factors_dir = args.factors_dir.resolve()
    metrics = pd.read_csv(args.layer_metrics.resolve())
    for response, prefix in RESPONSES.items():
        build_one(factors_dir, metrics, response, prefix, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
