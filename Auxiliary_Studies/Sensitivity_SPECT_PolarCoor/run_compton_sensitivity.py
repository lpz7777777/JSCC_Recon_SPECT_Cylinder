from __future__ import annotations

import argparse
import re
from pathlib import Path

from spect_sensitivity import (
    ComptonPhysicsConfig,
    SensitivityRunConfig,
    run_sensitivity_calculation,
)


TOOL_DIR = Path(__file__).resolve().parent


def _infer_energy_mev(factor_dir: Path) -> float:
    match = re.search(r"(?P<kev>\d+(?:\.\d+)?)keV", factor_dir.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Cannot infer energy from factor directory name {factor_dir.name!r}; "
            "provide --energy-mev."
        )
    return float(match.group("kev")) / 1000.0


def _infer_rotate_num(factor_dir: Path) -> int:
    match = re.search(r"RotateNum(?P<count>\d+)", factor_dir.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Cannot infer rotation count from factor directory name {factor_dir.name!r}; "
            "provide --rotate-num."
        )
    return int(match.group("count"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the normalized Compton-list sensitivity Sensi_d using a current "
            "Factors/<energy>keV_RotateNum<N> directory."
        )
    )
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument(
        "--compton-list",
        type=Path,
        nargs="+",
        required=True,
        help="One or more CSV files or directories. Directory CSV files are processed in name order.",
    )
    parser.add_argument(
        "--source-photons",
        type=float,
        required=True,
        help="Number of emitted photons represented by all supplied Compton list files.",
    )
    parser.add_argument("--energy-mev", type=float, default=None)
    parser.add_argument("--rotate-num", type=int, default=None)
    parser.add_argument("--event-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--expected-detector-count", type=int, default=10496)

    parser.add_argument("--energy-resolution-662kev", type=float, default=0.1)
    parser.add_argument("--energy-threshold-min-mev", type=float, default=0.050)
    parser.add_argument(
        "--energy-threshold-sum-mev",
        type=float,
        default=None,
        help=(
            "Defaults to the project values 0.18/0.40/0.46/0.60 MeV for "
            "218/440/511/662 keV; otherwise 0.9 * incident energy."
        ),
    )
    parser.add_argument("--delta-r1-mm", type=float, default=0.0)
    parser.add_argument("--delta-r2-mm", type=float, default=0.0)
    parser.add_argument("--min-event-effective-support", type=float, default=50.0)
    parser.add_argument(
        "--include-first-hit-source-leg-uncertainty",
        action="store_true",
        help="Legacy option. Normally disabled because SysMat_polar already models this extent.",
    )

    parser.add_argument("--system-matrix", type=Path, default=None)
    parser.add_argument("--detector-csv", type=Path, default=None)
    parser.add_argument("--coordinate-csv", type=Path, default=None)
    parser.add_argument("--rotation-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-rotation-average", action="store_true")
    parser.add_argument("--no-save-raw", action="store_true")
    parser.add_argument("--checkpoint-every-batches", type=int, default=100)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument(
        "--install-to-factor-dir",
        action="store_true",
        help="After validation, install the result as <factor-dir>/Sensi_d.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    factor_dir = args.factor_dir.resolve()
    energy_mev = args.energy_mev if args.energy_mev is not None else _infer_energy_mev(factor_dir)
    rotate_num = args.rotate_num if args.rotate_num is not None else _infer_rotate_num(factor_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else TOOL_DIR / "Result" / factor_dir.name
    )
    physics = ComptonPhysicsConfig(
        energy_mev=energy_mev,
        energy_resolution_662kev=args.energy_resolution_662kev,
        energy_threshold_min_mev=args.energy_threshold_min_mev,
        energy_threshold_sum_mev=args.energy_threshold_sum_mev,
        delta_r1_mm=args.delta_r1_mm,
        delta_r2_mm=args.delta_r2_mm,
        min_event_effective_support=args.min_event_effective_support,
        include_first_hit_source_leg_uncertainty=args.include_first_hit_source_leg_uncertainty,
    )
    config = SensitivityRunConfig(
        factor_dir=factor_dir,
        compton_paths=tuple(path.resolve() for path in args.compton_list),
        output_dir=output_dir,
        source_photons=args.source_photons,
        physics=physics,
        system_matrix_path=(args.system_matrix or factor_dir / "SysMat_polar").resolve(),
        detector_path=(args.detector_csv or factor_dir / "Detector.csv").resolve(),
        coordinate_path=(args.coordinate_csv or factor_dir / "coor_polar_full.csv").resolve(),
        rotation_path=(args.rotation_csv or factor_dir / "RotMat_full.csv").resolve(),
        rotate_num=rotate_num,
        event_fraction=args.event_fraction,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        expected_detector_count=args.expected_detector_count,
        apply_rotation_average=not args.no_rotation_average,
        save_raw=not args.no_save_raw,
        checkpoint_every_batches=args.checkpoint_every_batches,
        progress_every_batches=args.progress_every_batches,
        resume=args.resume,
        overwrite=args.overwrite,
        keep_checkpoint=args.keep_checkpoint,
        install_to_factor_dir=args.install_to_factor_dir,
    )
    run_sensitivity_calculation(config)


if __name__ == "__main__":
    main()
