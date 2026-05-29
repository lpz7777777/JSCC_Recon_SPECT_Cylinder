from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import warnings

import numpy as np


NUM_VOXELS = 52020
NUM_DETECTORS = 2312
DET_ROWS = 34
DET_COLS = 68
DET_PIXEL_SIZE_CM = 0.4
BLUR_TO_INPUT_RATIO = 1.9965
DETECTOR_PARAM_UNIT = "mm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a system matrix A, apply detector-domain Gaussian mixing to obtain B, "
            "scale B so that the total-mass ratio A:B equals 1:1.9965 by default, then "
            "write A + B. The Gaussian sigma is read from an energy-weighted free-path "
            "Monte Carlo summary unless explicitly overridden."
        )
    )
    parser.add_argument(
        "--input-sysmat",
        type=Path,
        default=Path(__file__).resolve().parent
        / "Work_SysMat"
        / "511keV_RotateNum20_SPECTEHENaI"
        / "SysMat.sysmat",
        help="Input system matrix path. Binary float32, shape = (52020, 2312).",
    )
    parser.add_argument(
        "--output-sysmat",
        type=Path,
        default=None,
        help="Output system matrix path. Defaults to <input>_gaussmix.sysmat.",
    )
    parser.add_argument(
        "--num-voxels",
        type=int,
        default=NUM_VOXELS,
        help=f"Number of voxels. Default: {NUM_VOXELS}.",
    )
    parser.add_argument(
        "--num-detectors",
        type=int,
        default=NUM_DETECTORS,
        help=f"Number of detectors. Default: {NUM_DETECTORS}.",
    )
    parser.add_argument(
        "--det-rows",
        type=int,
        default=DET_ROWS,
        help=f"Detector grid row count. Default: {DET_ROWS}.",
    )
    parser.add_argument(
        "--det-cols",
        type=int,
        default=DET_COLS,
        help=f"Detector grid column count. Default: {DET_COLS}.",
    )
    parser.add_argument(
        "--det-pixel-cm",
        type=float,
        default=DET_PIXEL_SIZE_CM,
        help=f"Detector pixel size in cm. Default: {DET_PIXEL_SIZE_CM}.",
    )
    parser.add_argument(
        "--energy-weighted-summary",
        type=Path,
        default=Path(__file__).resolve().parent / "simulation_output_energy_weighted_1e7" / "summary.json",
        help=(
            "Path to the energy-weighted Monte Carlo summary.json used to read the "
            "Gaussian sigma."
        ),
    )
    parser.add_argument(
        "--sigma-summary-key",
        type=str,
        default="energy_weighted_perpendicular_free_path_cm",
        help=(
            "Summary entry used to source the Gaussian sigma. "
            "Default: energy_weighted_perpendicular_free_path_cm."
        ),
    )
    parser.add_argument(
        "--sigma-summary-stat",
        choices=["mean", "std", "median", "p05", "p95", "min", "max"],
        default="std",
        help=(
            "Statistic read from the Monte Carlo summary entry to use as the Gaussian sigma. "
            "Default: std."
        ),
    )
    parser.add_argument(
        "--gauss-sigma-cm",
        type=float,
        default=None,
        help=(
            "Gaussian sigma in cm. If omitted, it is read from the energy-weighted "
            "Monte Carlo summary."
        ),
    )
    parser.add_argument(
        "--gauss-sigma-row-cm",
        type=float,
        default=None,
        help=(
            "Deprecated. This script now uses an isotropic Gaussian based on absolute "
            "detector-center distance. If provided, it overrides --gauss-sigma-cm."
        ),
    )
    parser.add_argument(
        "--gauss-sigma-col-cm",
        type=float,
        default=None,
        help=(
            "Deprecated. This script now uses an isotropic Gaussian based on absolute "
            "detector-center distance. If provided together with --gauss-sigma-row-cm, "
            "the two values must match."
        ),
    )
    parser.add_argument(
        "--blur-to-input-ratio",
        type=float,
        default=BLUR_TO_INPUT_RATIO,
        help=(
            "Target total-mass ratio between the blurred matrix B and the input matrix A, "
            f"i.e. A:B = 1:{BLUR_TO_INPUT_RATIO} by default."
        ),
    )
    parser.add_argument(
        "--params-detector",
        type=Path,
        default=None,
        help=(
            "Path to Params_Detector.dat. Defaults to the same directory as the input sysmat."
        ),
    )
    parser.add_argument(
        "--detector-param-unit",
        choices=["mm", "cm"],
        default=DETECTOR_PARAM_UNIT,
        help=(
            "Length unit used in Params_Detector.dat for detector positions and sizes. "
            "Default: mm."
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=512,
        help="Number of voxel rows processed per chunk. Default: 512.",
    )
    parser.add_argument(
        "--detector-flat-order",
        choices=["34_first", "68_first"],
        default="34_first",
        help=(
            "How the 2312 detector bins are flattened. "
            "'34_first' means the 34-dimension is the fast index in storage. "
            "This matches the user's stated detector layout priority."
        ),
    )
    parser.add_argument(
        "--boundary-mode",
        choices=["constant"],
        default="constant",
        help=(
            "Boundary handling for detector mixing. Only 'constant' is supported in the "
            "column-mixing implementation."
        ),
    )
    parser.add_argument(
        "--roughness-check-samples",
        type=int,
        default=16,
        help=(
            "Number of sampled voxels used to report detector-grid roughness for "
            "the two candidate flattening orders. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--col-block-size",
        type=int,
        default=0,
        help=(
            "Optional detector-column block size. Gaussian blur is applied independently "
            "within each column block to avoid mixing disconnected detector modules. "
            "Default: 0, i.e. no column blocking."
        ),
    )
    return parser.parse_args()


def validate_shape(num_voxels: int, num_detectors: int, det_rows: int, det_cols: int) -> None:
    if det_rows * det_cols != num_detectors:
        raise ValueError(
            f"Detector grid mismatch: {det_rows} * {det_cols} != {num_detectors}."
        )


def flat_to_grid(
    flat_values: np.ndarray,
    det_rows: int,
    det_cols: int,
    detector_flat_order: str,
) -> np.ndarray:
    if detector_flat_order == "34_first":
        return flat_values.reshape(det_rows, det_cols, order="F")
    if detector_flat_order == "68_first":
        return flat_values.reshape(det_rows, det_cols, order="C")
    raise ValueError(f"Unsupported detector_flat_order: {detector_flat_order}")


def grid_to_flat(
    grid_values: np.ndarray,
    detector_flat_order: str,
) -> np.ndarray:
    if detector_flat_order == "34_first":
        return np.asarray(grid_values, dtype=np.float32).reshape(-1, order="F")
    if detector_flat_order == "68_first":
        return np.asarray(grid_values, dtype=np.float32).reshape(-1, order="C")
    raise ValueError(f"Unsupported detector_flat_order: {detector_flat_order}")


def load_detector_params(detector_param_path: Path, num_detectors: int) -> np.ndarray:
    raw = np.fromfile(detector_param_path, dtype=np.float32)
    if raw.size < 1:
        raise RuntimeError(f"Detector parameter file is empty: {detector_param_path}")
    detector_count = int(round(float(raw[0])))
    if detector_count != num_detectors:
        raise ValueError(
            f"Detector count mismatch in Params_Detector.dat: "
            f"expected {num_detectors}, got {detector_count}."
        )
    expected_payload = num_detectors * 12
    if raw.size != 1 + expected_payload:
        raise ValueError(
            f"Unexpected detector parameter size: expected {1 + expected_payload} float32 values, "
            f"got {raw.size}."
        )
    return raw[1:].reshape(num_detectors, 12)


def detector_unit_scale_to_cm(detector_param_unit: str) -> float:
    if detector_param_unit == "mm":
        return 0.1
    if detector_param_unit == "cm":
        return 1.0
    raise ValueError(f"Unsupported detector_param_unit: {detector_param_unit}")


def load_sigma_from_summary(
    summary_path: Path,
    summary_key: str,
    summary_stat: str,
) -> float:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Energy-weighted summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise KeyError(f"Invalid summary payload in: {summary_path}")

    entry = summary.get(summary_key)
    if not isinstance(entry, dict):
        raise KeyError(
            f"Summary key '{summary_key}' was not found in {summary_path}."
        )

    value = entry.get(summary_stat)
    if value is None:
        raise KeyError(
            f"Statistic '{summary_stat}' was not found under summary['{summary_key}']."
        )

    sigma_cm = float(value)
    if sigma_cm <= 0.0:
        raise ValueError(
            f"Sigma loaded from summary must be positive, got {sigma_cm} cm."
        )
    return sigma_cm


def build_detector_mixing_matrix(
    detector_params: np.ndarray,
    sigma_cm: float,
    detector_param_unit: str,
) -> np.ndarray:
    unit_scale_to_cm = detector_unit_scale_to_cm(detector_param_unit)
    x = detector_params[:, 0].astype(np.float64) * unit_scale_to_cm
    y = detector_params[:, 1].astype(np.float64) * unit_scale_to_cm
    z = detector_params[:, 2].astype(np.float64) * unit_scale_to_cm

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dz = z[:, None] - z[None, :]
    distance_cm = np.sqrt(dx * dx + dy * dy + dz * dz)

    if sigma_cm > 0.0:
        weights = np.exp(-0.5 * (distance_cm / sigma_cm) ** 2)
    else:
        weights = np.isclose(distance_cm, 0.0, atol=1.0e-6).astype(np.float64)
    return weights.astype(np.float32)


def roughness_score(grid: np.ndarray) -> tuple[float, float, float]:
    row_diff = float(np.mean(np.abs(np.diff(grid, axis=0))))
    col_diff = float(np.mean(np.abs(np.diff(grid, axis=1))))
    return row_diff + col_diff, row_diff, col_diff


def estimate_layout_roughness(
    sysmat_memmap: np.memmap,
    num_voxels: int,
    det_rows: int,
    det_cols: int,
    sample_count: int,
) -> dict[str, dict[str, float]]:
    if sample_count <= 0:
        return {}

    sample_indices = np.linspace(0, num_voxels - 1, min(sample_count, num_voxels), dtype=int)
    total_scores = {
        "34_first": [],
        "68_first": [],
    }

    for idx in sample_indices:
        row = np.asarray(sysmat_memmap[idx], dtype=np.float32)
        for order_name in total_scores:
            grid = flat_to_grid(row, det_rows, det_cols, order_name)
            total_scores[order_name].append(roughness_score(grid))

    summary: dict[str, dict[str, float]] = {}
    for order_name, scores in total_scores.items():
        arr = np.asarray(scores, dtype=np.float64)
        summary[order_name] = {
            "roughness_total_mean": float(np.mean(arr[:, 0])),
            "roughness_row_mean": float(np.mean(arr[:, 1])),
            "roughness_col_mean": float(np.mean(arr[:, 2])),
        }
    return summary


def mix_compton_chunk(
    compton_chunk: np.ndarray,
    mixing_matrix: np.ndarray,
) -> np.ndarray:
    return (compton_chunk.astype(np.float32) @ mixing_matrix.astype(np.float32)).astype(np.float32)


def main() -> None:
    args = parse_args()
    validate_shape(args.num_voxels, args.num_detectors, args.det_rows, args.det_cols)

    input_path = args.input_sysmat.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input system matrix not found: {input_path}")

    if args.output_sysmat is None:
        output_path = input_path.with_name(input_path.stem + "_gaussmix" + input_path.suffix)
    else:
        output_path = args.output_sysmat.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.energy_weighted_summary.resolve()
    if args.params_detector is None:
        detector_param_path = input_path.with_name("Params_Detector.dat")
    else:
        detector_param_path = args.params_detector.resolve()
    if not detector_param_path.is_file():
        raise FileNotFoundError(f"Params_Detector.dat not found: {detector_param_path}")

    expected_bytes = args.num_voxels * args.num_detectors * np.dtype(np.float32).itemsize
    actual_bytes = input_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Input file size mismatch: expected {expected_bytes} bytes, got {actual_bytes} bytes."
        )

    sigma_source_value_cm = (
        load_sigma_from_summary(
            summary_path=summary_path,
            summary_key=args.sigma_summary_key,
            summary_stat=args.sigma_summary_stat,
        )
        if args.gauss_sigma_cm is None
        else float(args.gauss_sigma_cm)
    )
    if args.gauss_sigma_row_cm is not None and args.gauss_sigma_col_cm is not None:
        if not np.isclose(args.gauss_sigma_row_cm, args.gauss_sigma_col_cm, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "This version uses a single isotropic Gaussian sigma. "
                "--gauss-sigma-row-cm and --gauss-sigma-col-cm must match if both are provided."
            )
    sigma_cm = sigma_source_value_cm
    if args.gauss_sigma_row_cm is not None:
        sigma_cm = float(args.gauss_sigma_row_cm)
    if args.gauss_sigma_col_cm is not None:
        sigma_cm = float(args.gauss_sigma_col_cm)
    sigma_pixel = sigma_cm / args.det_pixel_cm
    if args.blur_to_input_ratio < 0.0:
        raise ValueError("blur_to_input_ratio must be non-negative.")
    blur_target_ratio = float(args.blur_to_input_ratio)

    print("=" * 72)
    print("System Matrix Gaussian Mixing")
    print("=" * 72)
    print(f"Input sysmat            : {input_path}")
    print(f"Output sysmat           : {output_path}")
    print(f"Params_Detector         : {detector_param_path}")
    print(f"Params_Detector unit    : {args.detector_param_unit}")
    print(f"Energy-weighted summary : {summary_path}")
    print(f"Matrix shape            : ({args.num_voxels}, {args.num_detectors})")
    print(f"Detector grid           : {args.det_rows} x {args.det_cols}")
    print(f"Detector flat order     : {args.detector_flat_order}")
    print("Detector axis meaning   : rows = z (34 contiguous), cols = x (68 slow-varying)")
    print(f"Detector pixel size     : {args.det_pixel_cm:.6f} cm")
    print(f"Target A:B ratio        : 1:{args.blur_to_input_ratio:.8f}")
    print("Blur scaling factor     : computed after building raw B")
    print(f"Gaussian sigma          : {sigma_cm:.6f} cm = {sigma_pixel:.6f} pixel")
    print(f"Boundary mode           : {args.boundary_mode}")
    print(f"Column block size       : {args.col_block_size}")
    print(f"Chunk rows              : {args.chunk_rows}")
    print()

    detector_params = load_detector_params(detector_param_path, args.num_detectors)
    mixing_matrix = build_detector_mixing_matrix(
        detector_params=detector_params,
        sigma_cm=sigma_cm,
        detector_param_unit=args.detector_param_unit,
    )

    sysmat_memmap = np.memmap(
        input_path,
        dtype=np.float32,
        mode="r",
        shape=(args.num_voxels, args.num_detectors),
    )

    roughness_info = estimate_layout_roughness(
        sysmat_memmap=sysmat_memmap,
        num_voxels=args.num_voxels,
        det_rows=args.det_rows,
        det_cols=args.det_cols,
        sample_count=args.roughness_check_samples,
    )
    if roughness_info:
        print("Detector layout roughness check on sampled voxels:")
        for order_name, stats in roughness_info.items():
            print(
                f"  {order_name:8s} total={stats['roughness_total_mean']:.6e}  "
                f"row={stats['roughness_row_mean']:.6e}  "
                f"col={stats['roughness_col_mean']:.6e}"
            )
        print()

    output_memmap = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(args.num_voxels, args.num_detectors),
    )

    original_sum = 0.0
    blurred_raw_sum = 0.0
    t0 = time.time()

    blurred_raw_path = output_path.with_name(output_path.stem + "_rawB" + output_path.suffix)
    blurred_raw_memmap = np.memmap(
        blurred_raw_path,
        dtype=np.float32,
        mode="w+",
        shape=(args.num_voxels, args.num_detectors),
    )

    for start in range(0, args.num_voxels, args.chunk_rows):
        end = min(start + args.chunk_rows, args.num_voxels)

        sys_chunk = np.asarray(sysmat_memmap[start:end], dtype=np.float32)
        blurred_raw_chunk = mix_compton_chunk(
            compton_chunk=sys_chunk,
            mixing_matrix=mixing_matrix,
        )
        blurred_raw_memmap[start:end] = blurred_raw_chunk

        original_sum += float(np.sum(sys_chunk, dtype=np.float64))
        blurred_raw_sum += float(np.sum(blurred_raw_chunk, dtype=np.float64))

        elapsed = time.time() - t0
        processed = end
        eta = elapsed / processed * (args.num_voxels - processed)
        print(
            f"Processed voxels {processed:6d}/{args.num_voxels} "
            f"({100.0 * processed / args.num_voxels:5.1f}%)  "
            f"elapsed {elapsed:7.1f}s  ETA {eta:7.1f}s"
        )

    blurred_raw_memmap.flush()

    if original_sum <= 0.0:
        raise RuntimeError("Input sysmat sum must be positive.")
    if blurred_raw_sum <= 0.0:
        raise RuntimeError("Blurred raw B sum must be positive.")

    blur_scale = (blur_target_ratio * original_sum) / blurred_raw_sum
    print()
    print(f"Raw B / A ratio         : {blurred_raw_sum / original_sum:.8f}")
    print(f"Applied blur scale      : {blur_scale:.8f}")

    blurred_scaled_sum = 0.0
    output_sum = 0.0
    output_min = np.inf
    output_max = -np.inf

    for start in range(0, args.num_voxels, args.chunk_rows):
        end = min(start + args.chunk_rows, args.num_voxels)
        sys_chunk = np.asarray(sysmat_memmap[start:end], dtype=np.float32)
        blurred_raw_chunk = np.asarray(blurred_raw_memmap[start:end], dtype=np.float32)
        blurred_scaled_chunk = blurred_raw_chunk * np.float32(blur_scale)
        out_chunk = sys_chunk + blurred_scaled_chunk
        output_memmap[start:end] = out_chunk

        blurred_scaled_sum += float(np.sum(blurred_scaled_chunk, dtype=np.float64))
        output_sum += float(np.sum(out_chunk, dtype=np.float64))
        output_min = min(output_min, float(np.min(out_chunk)))
        output_max = max(output_max, float(np.max(out_chunk)))

    output_memmap.flush()
    blurred_raw_memmap.flush()
    del blurred_raw_memmap
    del output_memmap
    try:
        blurred_raw_path.unlink(missing_ok=True)
    except PermissionError:
        warnings.warn(
            f"Temporary raw-B file is still in use and was not removed: {blurred_raw_path}"
        )

    meta = {
        "description": (
            "Read input sysmat as A, apply detector-domain Gaussian mixing to obtain B_raw, "
            "scale B_raw to B so that the total-mass ratio A:B matches the requested target, "
            "then save A + B."
        ),
        "input_sysmat": str(input_path),
        "output_sysmat": str(output_path),
        "shape": [args.num_voxels, args.num_detectors],
        "dtype": "float32",
        "params_detector": str(detector_param_path),
        "params_detector_unit": args.detector_param_unit,
        "energy_weighted_summary": str(summary_path),
        "detector_grid": {
            "rows": args.det_rows,
            "cols": args.det_cols,
            "flat_order": args.detector_flat_order,
            "pixel_size_cm": args.det_pixel_cm,
            "row_axis": "detector_z",
            "col_axis": "detector_x",
        },
        "mixing_ratio": {
            "input_A_to_blurred_B": [
                1.0,
                float(args.blur_to_input_ratio),
            ],
            "blur_scale_applied_to_B_raw": float(blur_scale),
            "raw_B_over_A_before_scaling": float(blurred_raw_sum / original_sum),
        },
        "gaussian_blur": {
            "sigma_loaded_from_summary_cm": float(sigma_source_value_cm),
            "sigma_cm": float(sigma_cm),
            "sigma_pixel": float(sigma_pixel),
            "boundary_mode": args.boundary_mode,
            "sigma_source_summary_key": args.sigma_summary_key,
            "sigma_source_summary_stat": args.sigma_summary_stat,
            "col_block_size": args.col_block_size,
            "implementation": "fixed detector-to-detector isotropic Gaussian mixing matrix",
            "distance_metric": "euclidean detector-center distance in 3D",
            "mixing_matrix_row_normalized": False,
            "self_weight_included": True,
            "self_weight_note": (
                "A detector contributes to itself with Gaussian weight exp(0)=1 before "
                "any global scaling, so self-response also follows the same Gaussian rule."
            ),
        },
        "roughness_check": roughness_info,
        "statistics": {
            "original_sysmat_sum": original_sum,
            "blurred_raw_B_sum": blurred_raw_sum,
            "blurred_scaled_B_sum": blurred_scaled_sum,
            "output_sysmat_sum": output_sum,
            "output_sysmat_min": output_min,
            "output_sysmat_max": output_max,
            "achieved_B_over_A_ratio": (
                float(blurred_scaled_sum / original_sum) if original_sum > 0.0 else None
            ),
        },
        "notes": [
            "This version no longer splits the input sysmat by interaction cross sections.",
            "The input matrix is treated directly as A, and the Gaussian-mixed copy is treated as B.",
            "This version defaults to detector_flat_order = 34_first, i.e. reshape(..., order='F').",
            "Params_Detector.dat confirms the detector order is: for each x position, z runs across 34 contiguous bins.",
            "The default Gaussian sigma is read from the energy-weighted Monte Carlo summary.",
            "The Gaussian blur is implemented from detector-center Euclidean distance, so each source detector 52020x1 column is redistributed to all detector columns according to absolute detector separation.",
            "The detector-to-detector Gaussian weights are not row-normalized; this follows the MATLAB reference test.m.",
            "After raw B is built, a single global scale factor is applied so that the final A:B sum ratio matches the requested target.",
            "Using the wrong detector flattening order can smear non-neighboring crystals together and create periodic artifacts in image space.",
        ],
    }

    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print()
    print(f"Finished in {elapsed:.1f}s")
    print(f"Output sysmat written to: {output_path}")
    print(f"Metadata written to     : {meta_path}")


if __name__ == "__main__":
    main()
