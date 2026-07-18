#!/usr/bin/env python3
"""Create total-preserving layer and detector-row JSCC Factor variants."""

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
    "A218": "218keV_RotateNum20_CenterPoint",
    "A440": "440keV_RotateNum20_CenterPoint",
    "C440to218": "440keV_to218win_RotateNum20_CenterPoint",
}
VARIANTS = {
    "layer": "FOVLayerTP",
    "detector": "FOVDetectorTP",
}


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo = script.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors-dir", type=Path, default=repo / "Factors")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=repo / "Geant4Sim" / "run" / "merged_UniformFovCntStat"
        / "uniform_fov_analysis",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--chunk-rows", type=int, default=128)
    return parser.parse_args()


def target_name(source_name: str, variant_tag: str) -> str:
    return source_name.replace("_CenterPoint", f"_CenterPoint_{variant_tag}")


def load_correction(
    response: str,
    variant: str,
    detector: pd.DataFrame,
    analysis_dir: Path,
) -> tuple[np.ndarray, dict]:
    if variant == "layer":
        table = pd.read_csv(analysis_dir / "recommended_total_preserving_layer_corrections.csv")
        table = table[table.response == response].copy()
        if len(table) != 4:
            raise ValueError(f"Expected four layer factors for {response}")
        factor_by_layer = dict(
            zip(table.layer_y_mm.astype(int), table.total_preserving_layer_factor)
        )
        correction = np.array(
            [factor_by_layer[int(y)] for y in detector.y], dtype=np.float64
        )
        details = {
            "scope": "detector_layers",
            "layer_y_mm": table.layer_y_mm.astype(int).tolist(),
            "incremental_layer_factor": table.total_preserving_layer_factor.tolist(),
            "proposed_composed_layer_scale": table.proposed_factor_layer_scale.tolist(),
        }
    else:
        table = pd.read_csv(analysis_dir / f"detector_row_correction_{response}.csv")
        table = table.sort_values("detector_index")
        if len(table) != DETECTOR_COUNT or not table.valid_for_correction.all():
            raise ValueError(f"Invalid detector-row correction table for {response}")
        expected_indices = np.arange(1, DETECTOR_COUNT + 1)
        if not np.array_equal(table.detector_index.to_numpy(), expected_indices):
            raise ValueError(f"Detector order mismatch for {response}")
        correction = table.total_preserving_factor.to_numpy(dtype=np.float64)
        details = {
            "scope": "individual_detector_rows",
            "correction_min": float(correction.min()),
            "correction_median": float(np.median(correction)),
            "correction_max": float(correction.max()),
            "split_half_warning": (
                "Raw per-detector factors include Monte Carlo noise; this variant is "
                "created for the requested controlled reconstruction comparison."
            ),
        }
    if correction.shape != (DETECTOR_COUNT,) or not np.isfinite(correction).all():
        raise ValueError(f"Invalid correction vector for {response}/{variant}")
    if np.any(correction <= 0):
        raise ValueError(f"Nonpositive correction for {response}/{variant}")
    return correction.astype(np.float32), details


def copy_metadata(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=False)
    excluded = {"SysMat_polar", "SysMat_tmp", "factor_manifest.json"}
    for source in source_dir.iterdir():
        if source.name in excluded or not source.is_file():
            continue
        shutil.copy2(source, target_dir / source.name)


def write_corrected_matrix(
    source_path: Path,
    target_path: Path,
    correction: np.ndarray,
    chunk_rows: int,
) -> tuple[float, float]:
    expected_bytes = PIXEL_COUNT * DETECTOR_COUNT * np.dtype(np.float32).itemsize
    if source_path.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected matrix size: {source_path}")
    temporary = target_path.with_name(f".tmp_{target_path.name}_{os.getpid()}")
    source = np.memmap(
        source_path, dtype=np.float32, mode="r", shape=(PIXEL_COUNT, DETECTOR_COUNT)
    )
    target = np.memmap(
        temporary, dtype=np.float32, mode="w+", shape=(PIXEL_COUNT, DETECTOR_COUNT)
    )
    source_sum = 0.0
    target_sum = 0.0
    for start in range(0, PIXEL_COUNT, chunk_rows):
        stop = min(start + chunk_rows, PIXEL_COUNT)
        source_chunk = np.asarray(source[start:stop], dtype=np.float32)
        target_chunk = source_chunk * correction[np.newaxis, :]
        target[start:stop] = target_chunk
        source_sum += float(source_chunk.sum(dtype=np.float64))
        target_sum += float(target_chunk.sum(dtype=np.float64))
    target.flush()
    del target, source
    temporary.replace(target_path)
    return source_sum, target_sum


def build_manifest(
    source_dir: Path,
    target_dir: Path,
    variant: str,
    details: dict,
    source_sum: float,
    target_sum: float,
) -> dict:
    with (source_dir / "factor_manifest.json").open(encoding="utf-8") as handle:
        parent = json.load(handle)
    manifest = dict(parent)
    manifest["parent_factor_dir"] = str(source_dir.resolve())
    manifest["parent_calibration"] = parent.get("calibration")
    manifest["grid"] = dict(parent["grid"])
    manifest["grid"]["output_suffix"] = target_dir.name.split("RotateNum20_", 1)[-1]
    manifest["calibration"] = {
        "enabled": True,
        "name": f"JSCC_{parent['response']}_UniformFOV_{VARIANTS[variant]}_20260718",
        "method": "detector-row multiplication after the existing center calibration",
        "source": "102.48e9-primary pure-energy uniform-FOV Geant4 CntStat per energy",
        "total_efficiency_preserved": True,
        "source_matrix_sum": source_sum,
        "corrected_matrix_sum": target_sum,
        "relative_total_change": target_sum / source_sum - 1.0,
        "correction_csv": "correction_vector.csv",
        **details,
    }
    manifest["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return manifest


def main() -> None:
    args = parse_args()
    factors_dir = args.factors_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    for response, source_name in RESPONSES.items():
        source_dir = factors_dir / source_name
        detector = pd.read_csv(source_dir / "Detector.csv")
        if len(detector) != DETECTOR_COUNT:
            raise ValueError(f"Unexpected detector count: {source_dir}")
        for variant in args.variants:
            tag = VARIANTS[variant]
            target_dir = factors_dir / target_name(source_name, tag)
            if target_dir.exists():
                raise FileExistsError(target_dir)
            correction, details = load_correction(
                response, variant, detector, analysis_dir
            )
            copy_metadata(source_dir, target_dir)
            pd.DataFrame(
                {
                    "detector_index": detector["index"].to_numpy(),
                    "x": detector.x.to_numpy(),
                    "y": detector.y.to_numpy(),
                    "z": detector.z.to_numpy(),
                    "correction_factor": correction,
                }
            ).to_csv(target_dir / "correction_vector.csv", index=False)
            source_sum, corrected_sum = write_corrected_matrix(
                source_dir / "SysMat_polar",
                target_dir / "SysMat_polar",
                correction,
                args.chunk_rows,
            )
            manifest = build_manifest(
                source_dir, target_dir, variant, details, source_sum, corrected_sum
            )
            with (target_dir / "factor_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
            print(
                f"Created {target_dir.name}: response={response}, "
                f"relative total change={corrected_sum / source_sum - 1.0:+.3e}"
            )


if __name__ == "__main__":
    main()
