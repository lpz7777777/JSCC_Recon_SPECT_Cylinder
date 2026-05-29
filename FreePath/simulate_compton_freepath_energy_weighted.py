from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulate_compton_freepath import (
    NAI_DENSITY_G_CM3,
    build_hist_axes,
    load_cross_section_table,
    save_summary_json,
    simulate_scattered_photons,
    summarize_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo simulation for energy-weighted Compton free path in NaI. "
            "The first interaction is Compton scatter with deposited energy E1 = E0 - E', "
            "and the second interaction is assumed to fully absorb the scattered photon with "
            "deposited energy E2 = E'. The effective interaction position is the energy-weighted "
            "centroid of the two deposits, and the energy-weighted free path is measured from the "
            "first interaction point to that centroid."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parent / "data.txt",
        help="Path to the NaI attenuation coefficient table.",
    )
    parser.add_argument(
        "--incident-energy-mev",
        type=float,
        default=0.511,
        help="Incident photon energy in MeV.",
    )
    parser.add_argument(
        "--num-photons",
        type=int,
        default=200000,
        help="Number of Compton-scattered photons to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260420,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=120,
        help="Histogram bin count used in the output figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simulation_output_energy_weighted",
        help="Output directory for plots and summary files.",
    )
    parser.add_argument(
        "--skip-samples-csv",
        action="store_true",
        help="Skip writing the full per-photon samples.csv file.",
    )
    return parser.parse_args()


def simulate_energy_weighted_photons(
    incident_energy_mev: float,
    sample_count: int,
    cross_section_table: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    result = simulate_scattered_photons(
        incident_energy_mev=incident_energy_mev,
        sample_count=sample_count,
        cross_section_table=cross_section_table,
        rng=rng,
    )

    second_deposit_energy_mev = result["scattered_energy_mev"]
    first_deposit_energy_mev = incident_energy_mev - second_deposit_energy_mev
    total_deposit_energy_mev = first_deposit_energy_mev + second_deposit_energy_mev
    second_position_weight = second_deposit_energy_mev / np.maximum(total_deposit_energy_mev, 1.0e-15)
    first_position_weight = first_deposit_energy_mev / np.maximum(total_deposit_energy_mev, 1.0e-15)

    energy_weighted_free_path_cm = result["free_path_cm"] * second_position_weight
    energy_weighted_perpendicular_free_path_cm = (
        result["perpendicular_free_path_cm"] * second_position_weight
    )
    energy_weighted_parallel_free_path_cm = result["parallel_free_path_cm"] * second_position_weight

    result.update(
        {
            "first_deposit_energy_mev": first_deposit_energy_mev,
            "second_deposit_energy_mev": second_deposit_energy_mev,
            "total_deposit_energy_mev": total_deposit_energy_mev,
            "first_position_weight": first_position_weight,
            "second_position_weight": second_position_weight,
            "energy_weighted_free_path_cm": energy_weighted_free_path_cm,
            "energy_weighted_perpendicular_free_path_cm": energy_weighted_perpendicular_free_path_cm,
            "energy_weighted_parallel_free_path_cm": energy_weighted_parallel_free_path_cm,
        }
    )
    return result


def build_summary(
    incident_energy_mev: float,
    sample_count: int,
    data_path: Path,
    result: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "config": {
            "incident_energy_mev": incident_energy_mev,
            "num_photons": sample_count,
            "density_g_cm3": NAI_DENSITY_G_CM3,
            "data_path": str(data_path.resolve()),
            "energy_weighting_definition": (
                "r_weighted = (E1 * r1 + E2 * r2) / (E1 + E2), "
                "with E1 = E0 - E_scattered and E2 = E_scattered."
            ),
            "assumption": "The second interaction fully absorbs the scattered photon.",
        },
        "summary": {
            "theta_deg": summarize_array(result["theta_deg"]),
            "first_deposit_energy_mev": summarize_array(result["first_deposit_energy_mev"]),
            "second_deposit_energy_mev": summarize_array(result["second_deposit_energy_mev"]),
            "first_position_weight": summarize_array(result["first_position_weight"]),
            "second_position_weight": summarize_array(result["second_position_weight"]),
            "true_free_path_cm": summarize_array(result["free_path_cm"]),
            "true_perpendicular_free_path_cm": summarize_array(result["perpendicular_free_path_cm"]),
            "true_parallel_free_path_cm": summarize_array(result["parallel_free_path_cm"]),
            "energy_weighted_free_path_cm": summarize_array(result["energy_weighted_free_path_cm"]),
            "energy_weighted_perpendicular_free_path_cm": summarize_array(
                result["energy_weighted_perpendicular_free_path_cm"]
            ),
            "energy_weighted_parallel_free_path_cm": summarize_array(
                result["energy_weighted_parallel_free_path_cm"]
            ),
            "abs_energy_weighted_parallel_free_path_cm": summarize_array(
                np.abs(result["energy_weighted_parallel_free_path_cm"])
            ),
            "forward_scatter_fraction": float(np.mean(result["cos_theta"] >= 0.0)),
            "backscatter_fraction": float(np.mean(result["cos_theta"] < 0.0)),
        },
    }


def save_energy_weighted_samples_csv(output_path: Path, result: dict[str, np.ndarray]) -> None:
    header = (
        "theta_deg,cos_theta,first_deposit_energy_mev,second_deposit_energy_mev,"
        "first_position_weight,second_position_weight,free_path_cm,"
        "perpendicular_free_path_cm,parallel_free_path_cm,energy_weighted_free_path_cm,"
        "energy_weighted_perpendicular_free_path_cm,energy_weighted_parallel_free_path_cm"
    )
    matrix = np.column_stack(
        [
            result["theta_deg"],
            result["cos_theta"],
            result["first_deposit_energy_mev"],
            result["second_deposit_energy_mev"],
            result["first_position_weight"],
            result["second_position_weight"],
            result["free_path_cm"],
            result["perpendicular_free_path_cm"],
            result["parallel_free_path_cm"],
            result["energy_weighted_free_path_cm"],
            result["energy_weighted_perpendicular_free_path_cm"],
            result["energy_weighted_parallel_free_path_cm"],
        ]
    )
    np.savetxt(output_path, matrix, delimiter=",", header=header, comments="")


def save_overview_figure(output_path: Path, incident_energy_mev: float, result: dict[str, np.ndarray], bins: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        f"Energy-Weighted Compton Free Path in NaI, E0 = {incident_energy_mev:.3f} MeV"
    )

    build_hist_axes(
        axes[0, 0],
        result["first_deposit_energy_mev"],
        bins,
        "First Deposit Energy",
        "E1 (MeV)",
    )
    build_hist_axes(
        axes[0, 1],
        result["second_deposit_energy_mev"],
        bins,
        "Second Deposit Energy",
        "E2 (MeV)",
    )
    build_hist_axes(
        axes[0, 2],
        result["second_position_weight"],
        bins,
        "Second-Point Weight",
        "E2 / (E1 + E2)",
    )
    build_hist_axes(
        axes[1, 0],
        result["energy_weighted_free_path_cm"],
        bins,
        "Energy-Weighted Free Path",
        "Weighted free path (cm)",
        mean_in_title=True,
    )
    build_hist_axes(
        axes[1, 1],
        result["energy_weighted_perpendicular_free_path_cm"],
        bins,
        "Energy-Weighted Perpendicular Free Path",
        "Weighted perpendicular component (cm)",
        mean_in_title=True,
    )
    build_hist_axes(
        axes[1, 2],
        result["energy_weighted_parallel_free_path_cm"],
        bins,
        "Energy-Weighted Parallel Free Path",
        "Weighted parallel component (cm)",
        mean_in_title=True,
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_comparison_figure(output_path: Path, result: dict[str, np.ndarray], bins: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    comparisons = [
        (
            axes[0],
            result["free_path_cm"],
            result["energy_weighted_free_path_cm"],
            "True vs Energy-Weighted Free Path",
            "Path (cm)",
        ),
        (
            axes[1],
            result["perpendicular_free_path_cm"],
            result["energy_weighted_perpendicular_free_path_cm"],
            "Perpendicular Component",
            "Perpendicular component (cm)",
        ),
        (
            axes[2],
            result["parallel_free_path_cm"],
            result["energy_weighted_parallel_free_path_cm"],
            "Parallel Component",
            "Parallel component (cm)",
        ),
    ]

    for ax, true_values, weighted_values, title, xlabel in comparisons:
        ax.hist(true_values, bins=bins, density=True, alpha=0.55, color="#4472C4", label="True")
        ax.hist(weighted_values, bins=bins, density=True, alpha=0.55, color="#ED7D31", label="Energy-Weighted")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cross_section_table = load_cross_section_table(args.data_path.resolve())
    result = simulate_energy_weighted_photons(
        incident_energy_mev=args.incident_energy_mev,
        sample_count=args.num_photons,
        cross_section_table=cross_section_table,
        rng=rng,
    )

    summary = build_summary(
        incident_energy_mev=args.incident_energy_mev,
        sample_count=args.num_photons,
        data_path=args.data_path.resolve(),
        result=result,
    )

    summary_path = output_dir / "summary.json"
    samples_path = output_dir / "samples.csv"
    overview_fig_path = output_dir / "overview.png"
    comparison_fig_path = output_dir / "comparison.png"

    save_summary_json(summary_path, summary)
    if not args.skip_samples_csv:
        save_energy_weighted_samples_csv(samples_path, result)
    save_overview_figure(overview_fig_path, args.incident_energy_mev, result, args.num_bins)
    save_comparison_figure(comparison_fig_path, result, args.num_bins)

    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
