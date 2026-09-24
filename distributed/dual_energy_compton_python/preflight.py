"""Fast input validation for the distributed 218/440 reconstruction."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detector_csv import load_detector_coordinates
from fov_config import load_config, validate_factor_geometry, validate_sensitivity_provenance


DATASET = "ContrastPhantom_DualEnergy_10_30_240_30_225Ac"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-level", default="1e10")
    parser.add_argument("--data-file-name", default=DATASET)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--factors-dir", type=Path, default=Path("Factors"))
    parser.add_argument("--cntstat-dir", type=Path, default=Path("CntStat"))
    parser.add_argument("--list-dir", type=Path, default=Path("List"))
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--scan-matrix", action="store_true")
    parser.add_argument("--estimated-accepted-events", type=int)
    parser.add_argument("--gpu-memory-gib", type=float)
    args = parser.parse_args()

    factor_dirs = {
        "218": REPO_ROOT / args.factors_dir / "218keV_RotateNum20",
        "440": REPO_ROOT / args.factors_dir / "440keV_RotateNum20",
        "cross": REPO_ROOT / args.factors_dir / "440keV_to218win_RotateNum20",
    }
    geometry = validate_factor_geometry(factor_dirs,
        load_config(args.experiment_config) if args.experiment_config else None, args.scan_matrix)
    if args.experiment_config:
        validate_sensitivity_provenance(factor_dirs["440"], verify_matrix=True)
    coordinates = np.loadtxt(
        factor_dirs["440"] / "coor_polar_full.csv", delimiter=",", ndmin=2
    )
    pixel_num = coordinates.shape[0]
    detector = load_detector_coordinates(factor_dirs["440"] / "Detector.csv")
    detector_count = detector.shape[0]
    if args.world_size <= 0 or args.world_size > detector_count:
        raise ValueError("world-size must be within the detector-bin count")

    expected_matrix_bytes = pixel_num * detector_count * np.dtype(np.float32).itemsize
    for label, directory in factor_dirs.items():
        matrix = directory / "SysMat_polar"
        if not matrix.is_file() or matrix.stat().st_size != expected_matrix_bytes:
            raise ValueError(
                f"{label} matrix size mismatch: {matrix}; "
                f"expected {expected_matrix_bytes} bytes"
            )
    sensi_d = factor_dirs["440"] / "Sensi_d"
    if sensi_d.stat().st_size != pixel_num * np.dtype(np.float32).itemsize:
        raise ValueError(f"Sensi_d size mismatch: {sensi_d}")
    sensi_values = np.fromfile(sensi_d, dtype=np.float32)
    if not np.isfinite(sensi_values).all() or np.any(sensi_values <= 0):
        raise ValueError("Sensi_d contains non-positive or non-finite values")

    cntstat = []
    for energy in (218, 440):
        path = (
            REPO_ROOT / args.cntstat_dir / f"{energy}keV_RotateNum20_Geant4JSCC" /
            f"CntStat_{args.data_file_name}_{args.count_level}.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if values.size != detector_count * 20:
            raise ValueError(f"CntStat size mismatch: {path}: {values.size}")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"CntStat contains invalid counts: {path}")
        cntstat.append({"energy_keV": energy, "path": str(path), "counts": float(values.sum())})

    list_dir = (
        REPO_ROOT / args.list_dir / "218-440keV_RotateNum20_Geant4JSCC" /
        f"List_{args.data_file_name}_{args.count_level}"
    )
    list_files = [list_dir / f"{view}.csv" for view in range(1, 21)]
    missing = [str(path) for path in list_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing List views: {missing}")

    report = {
        "geometry": geometry,
        "status": "ok",
        "pixel_num": pixel_num,
        "detector_count": detector_count,
        "world_size": args.world_size,
        "detector_bins_per_rank_min": detector_count // args.world_size,
        "matrix_bytes_each": expected_matrix_bytes,
        "sensi_d": str(sensi_d),
        "cntstat": cntstat,
        "list_dir": str(list_dir),
        "list_bytes_total": sum(path.stat().st_size for path in list_files),
    }
    if args.estimated_accepted_events is not None:
        if args.estimated_accepted_events < 0:
            raise ValueError("Event estimate must be nonnegative")
        # Event rows + complete 440 matrix + three matrix shards + workspace.
        estimate = (args.estimated_accepted_events * pixel_num * 4 / args.world_size
                    + expected_matrix_bytes * (1 + 3 / args.world_size)) / 2**30 + 4
        report["estimated_peak_gpu_gib"] = estimate
        report["estimated_host_matrix_gib_per_rank"] = 3 * expected_matrix_bytes / 2**30
        if args.gpu_memory_gib and estimate > .8 * args.gpu_memory_gib:
            raise ValueError(f"Estimated {estimate:.2f} GiB exceeds 80% GPU budget; increase ranks")
    elif args.gpu_memory_gib:
        raise ValueError("Provide --estimated-accepted-events for a memory budget check")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
