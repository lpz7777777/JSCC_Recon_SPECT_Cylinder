from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entry point for center_geometry and supervised event-order inference experiments."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["center", "supervised"],
        help="Experiment mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Result output directory.")
    parser.add_argument("--detector-csv", type=Path, required=True, help="Detector.csv path.")
    parser.add_argument("--energy-mev", type=float, default=0.511)
    parser.add_argument("--front-prior-ratio", type=float, default=1.2)
    parser.add_argument("--geometry-sigma-deg", type=float, default=20.0)
    parser.add_argument("--geometry-power", type=float, default=1.0)
    parser.add_argument("--ene-resolution-662keV", type=float, default=0.1)
    parser.add_argument("--ene-threshold-min", type=float, default=0.05)
    parser.add_argument("--ene-threshold-sum", type=float, default=0.46)
    parser.add_argument("--disable-energy-smear", action="store_true")
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument(
        "--reject-frac-list",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
    )

    parser.add_argument("--list-dir", type=Path, help="List directory used by center mode.")
    parser.add_argument("--smear-repeats", type=int, default=5, help="Repeat count used by center mode.")

    parser.add_argument("--train-list-dir", type=Path, help="Training list directory used by supervised mode.")
    parser.add_argument("--test-list-dir", type=Path, help="Test list directory used by supervised mode.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-smear-repeats", type=int, default=3)
    parser.add_argument("--test-smear-repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--supervised-conda-env",
        type=str,
        default="pytorch",
        help="Conda env used for supervised training and export.",
    )
    return parser.parse_args()


def run_command(command: list[str], workdir: Path) -> None:
    print("Running:")
    print("  " + " ".join(command))
    subprocess.run(command, cwd=str(workdir), check=True)


def export_visualization(result_dir: Path, python_command_prefix: list[str]) -> None:
    command = python_command_prefix + [
        str(THIS_DIR / "export_visualization_data.py"),
        "--result-dir",
        str(result_dir),
    ]
    run_command(command, REPO_ROOT)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    detector_csv = args.detector_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_python_prefix = [sys.executable]
    supervised_python_prefix = ["conda", "run", "-n", args.supervised_conda_env, "python"]

    if args.mode == "center":
        if args.list_dir is None:
            raise ValueError("--list-dir is required when --mode center.")

        command = current_python_prefix + [
            str(THIS_DIR / "infer_event_order.py"),
            "--list-dir",
            str(args.list_dir.resolve()),
            "--detector-csv",
            str(detector_csv),
            "--energy-mev",
            str(args.energy_mev),
            "--front-prior-ratio",
            str(args.front_prior_ratio),
            "--geometry-sigma-deg",
            str(args.geometry_sigma_deg),
            "--geometry-power",
            str(args.geometry_power),
            "--ene-resolution-662keV",
            str(args.ene_resolution_662keV),
            "--ene-threshold-min",
            str(args.ene_threshold_min),
            "--ene-threshold-sum",
            str(args.ene_threshold_sum),
            "--smear-repeats",
            str(args.smear_repeats),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(output_dir),
            "--reject-frac-list",
            *[str(x) for x in args.reject_frac_list],
        ]
        if args.disable_energy_smear:
            command.append("--disable-energy-smear")

        run_command(command, REPO_ROOT)
        export_visualization(output_dir, current_python_prefix)

    elif args.mode == "supervised":
        if args.train_list_dir is None or args.test_list_dir is None:
            raise ValueError("--train-list-dir and --test-list-dir are required when --mode supervised.")

        command = supervised_python_prefix + [
            str(THIS_DIR / "train_supervised_order_classifier.py"),
            "--train-list-dir",
            str(args.train_list_dir.resolve()),
            "--test-list-dir",
            str(args.test_list_dir.resolve()),
            "--detector-csv",
            str(detector_csv),
            "--energy-mev",
            str(args.energy_mev),
            "--front-prior-ratio",
            str(args.front_prior_ratio),
            "--geometry-sigma-deg",
            str(args.geometry_sigma_deg),
            "--geometry-power",
            str(args.geometry_power),
            "--ene-resolution-662keV",
            str(args.ene_resolution_662keV),
            "--ene-threshold-min",
            str(args.ene_threshold_min),
            "--ene-threshold-sum",
            str(args.ene_threshold_sum),
            "--val-fraction",
            str(args.val_fraction),
            "--train-smear-repeats",
            str(args.train_smear_repeats),
            "--test-smear-repeats",
            str(args.test_smear_repeats),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--hidden-dims",
            str(args.hidden_dims),
            "--dropout",
            str(args.dropout),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--output-dir",
            str(output_dir),
            "--reject-frac-list",
            *[str(x) for x in args.reject_frac_list],
        ]
        if args.disable_energy_smear:
            command.append("--disable-energy-smear")

        run_command(command, REPO_ROOT)
        export_visualization(output_dir, supervised_python_prefix)

    print(f"\nCompleted. Result folder: {output_dir}")


if __name__ == "__main__":
    main()
