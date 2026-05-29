from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ELECTRON_REST_MEV = 0.511
NAI_DENSITY_G_CM3 = 3.67


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo simulation for Compton-scattered photons in NaI. "
            "The script samples the scattering angle and scattered energy using "
            "the Klein-Nishina distribution, then samples the scattered-photon free path "
            "using the tabulated NaI attenuation coefficients."
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
        default=Path(__file__).resolve().parent / "simulation_output",
        help="Output directory for plots and summary files.",
    )
    parser.add_argument(
        "--skip-samples-csv",
        action="store_true",
        help="Skip writing the full per-photon samples.csv file. Useful for very large simulations.",
    )
    return parser.parse_args()


def load_cross_section_table(data_path: Path) -> dict[str, np.ndarray]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Cross-section table not found: {data_path}")

    energies = []
    incoherent = []
    photoelectric = []
    total_wo_coherent = []

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                e_val = float(parts[0])
                incoh_val = float(parts[1])
                photo_val = float(parts[2])
                total_val = float(parts[3])
            except ValueError:
                continue

            energies.append(e_val)
            incoherent.append(incoh_val)
            photoelectric.append(photo_val)
            total_wo_coherent.append(total_val)

    if not energies:
        raise RuntimeError(f"No numeric rows were parsed from: {data_path}")

    energy_arr = np.asarray(energies, dtype=np.float64)
    order = np.argsort(energy_arr, kind="stable")

    energy_arr = energy_arr[order]
    incoherent_arr = np.asarray(incoherent, dtype=np.float64)[order]
    photoelectric_arr = np.asarray(photoelectric, dtype=np.float64)[order]
    total_arr = np.asarray(total_wo_coherent, dtype=np.float64)[order]

    # Keep the last entry for duplicate energies. This is pragmatic and stable for
    # interpolation in the presence of absorption edges.
    unique_energy = []
    unique_incoh = []
    unique_photo = []
    unique_total = []
    for idx in range(energy_arr.size):
        if unique_energy and np.isclose(energy_arr[idx], unique_energy[-1], atol=0.0, rtol=0.0):
            unique_incoh[-1] = incoherent_arr[idx]
            unique_photo[-1] = photoelectric_arr[idx]
            unique_total[-1] = total_arr[idx]
        else:
            unique_energy.append(float(energy_arr[idx]))
            unique_incoh.append(float(incoherent_arr[idx]))
            unique_photo.append(float(photoelectric_arr[idx]))
            unique_total.append(float(total_arr[idx]))

    return {
        "energy_mev": np.asarray(unique_energy, dtype=np.float64),
        "incoherent_cm2_g": np.asarray(unique_incoh, dtype=np.float64),
        "photoelectric_cm2_g": np.asarray(unique_photo, dtype=np.float64),
        "total_wo_coherent_cm2_g": np.asarray(unique_total, dtype=np.float64),
    }


def interpolate_cross_section(energy_mev: np.ndarray, table_energy: np.ndarray, table_value: np.ndarray) -> np.ndarray:
    energy_clip = np.clip(energy_mev, table_energy[0], table_energy[-1])
    return np.interp(energy_clip, table_energy, table_value)


def klein_nishina_relative_pdf_mu(mu: np.ndarray, incident_energy_mev: float) -> np.ndarray:
    alpha = incident_energy_mev / ELECTRON_REST_MEV
    k = 1.0 / (1.0 + alpha * (1.0 - mu))
    return k**2 * (k + 1.0 / k - (1.0 - mu**2))


def sample_klein_nishina_mu(incident_energy_mev: float, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    accepted = np.empty(sample_count, dtype=np.float64)
    filled = 0

    # The Klein-Nishina density in mu = cos(theta) is maximal near forward scatter.
    # Use a conservative bound evaluated at mu=1.
    max_pdf = float(klein_nishina_relative_pdf_mu(np.array([1.0]), incident_energy_mev)[0])

    while filled < sample_count:
        batch_size = max(4096, 2 * (sample_count - filled))
        mu_prop = rng.uniform(-1.0, 1.0, size=batch_size)
        accept_u = rng.uniform(0.0, 1.0, size=batch_size)
        pdf_prop = klein_nishina_relative_pdf_mu(mu_prop, incident_energy_mev)
        keep = accept_u < (pdf_prop / max_pdf)
        keep_count = int(np.count_nonzero(keep))
        if keep_count == 0:
            continue
        take = min(keep_count, sample_count - filled)
        accepted[filled : filled + take] = mu_prop[keep][:take]
        filled += take

    return accepted


def simulate_scattered_photons(
    incident_energy_mev: float,
    sample_count: int,
    cross_section_table: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    mu = sample_klein_nishina_mu(incident_energy_mev, sample_count, rng)
    theta_rad = np.arccos(np.clip(mu, -1.0, 1.0))
    scattered_energy_mev = incident_energy_mev / (1.0 + (incident_energy_mev / ELECTRON_REST_MEV) * (1.0 - mu))

    total_cm2_g = interpolate_cross_section(
        scattered_energy_mev,
        cross_section_table["energy_mev"],
        cross_section_table["total_wo_coherent_cm2_g"],
    )
    incoherent_cm2_g = interpolate_cross_section(
        scattered_energy_mev,
        cross_section_table["energy_mev"],
        cross_section_table["incoherent_cm2_g"],
    )
    photoelectric_cm2_g = interpolate_cross_section(
        scattered_energy_mev,
        cross_section_table["energy_mev"],
        cross_section_table["photoelectric_cm2_g"],
    )

    mu_total_cm_inv = total_cm2_g * NAI_DENSITY_G_CM3
    free_path_cm = -np.log(np.maximum(rng.uniform(0.0, 1.0, size=sample_count), 1.0e-15)) / np.maximum(mu_total_cm_inv, 1.0e-15)
    perpendicular_free_path_cm = free_path_cm * np.sin(theta_rad)
    parallel_free_path_cm = free_path_cm * mu

    scatter_probability = incoherent_cm2_g / np.maximum(total_cm2_g, 1.0e-15)
    photoelectric_probability = photoelectric_cm2_g / np.maximum(total_cm2_g, 1.0e-15)

    return {
        "cos_theta": mu,
        "theta_deg": np.degrees(theta_rad),
        "scattered_energy_mev": scattered_energy_mev,
        "total_cm2_g": total_cm2_g,
        "incoherent_cm2_g": incoherent_cm2_g,
        "photoelectric_cm2_g": photoelectric_cm2_g,
        "mu_total_cm_inv": mu_total_cm_inv,
        "free_path_cm": free_path_cm,
        "perpendicular_free_path_cm": perpendicular_free_path_cm,
        "parallel_free_path_cm": parallel_free_path_cm,
        "scatter_probability": scatter_probability,
        "photoelectric_probability": photoelectric_probability,
    }


def summarize_array(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


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
        },
        "summary": {
            "scattered_energy_mev": summarize_array(result["scattered_energy_mev"]),
            "theta_deg": summarize_array(result["theta_deg"]),
            "free_path_cm": summarize_array(result["free_path_cm"]),
            "perpendicular_free_path_cm": summarize_array(result["perpendicular_free_path_cm"]),
            "parallel_free_path_cm": summarize_array(result["parallel_free_path_cm"]),
            "abs_parallel_free_path_cm": summarize_array(np.abs(result["parallel_free_path_cm"])),
            "mu_total_cm_inv": summarize_array(result["mu_total_cm_inv"]),
            "scatter_probability_next_interaction": summarize_array(result["scatter_probability"]),
            "photoelectric_probability_next_interaction": summarize_array(result["photoelectric_probability"]),
            "forward_scatter_fraction": float(np.mean(result["cos_theta"] >= 0.0)),
            "backscatter_fraction": float(np.mean(result["cos_theta"] < 0.0)),
        },
    }


def save_summary_json(output_path: Path, summary: dict[str, object]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_samples_csv(output_path: Path, result: dict[str, np.ndarray]) -> None:
    header = (
        "theta_deg,cos_theta,scattered_energy_mev,free_path_cm,"
        "perpendicular_free_path_cm,parallel_free_path_cm,mu_total_cm_inv,"
        "scatter_probability,photoelectric_probability"
    )
    matrix = np.column_stack(
        [
            result["theta_deg"],
            result["cos_theta"],
            result["scattered_energy_mev"],
            result["free_path_cm"],
            result["perpendicular_free_path_cm"],
            result["parallel_free_path_cm"],
            result["mu_total_cm_inv"],
            result["scatter_probability"],
            result["photoelectric_probability"],
        ]
    )
    np.savetxt(output_path, matrix, delimiter=",", header=header, comments="")


def build_hist_axes(
    ax: plt.Axes,
    data: np.ndarray,
    bins: int,
    title: str,
    xlabel: str,
    mean_in_title: bool = False,
) -> None:
    ax.hist(data, bins=bins, density=True, color="#4472C4", alpha=0.85, edgecolor="none")
    if mean_in_title:
        ax.set_title(f"{title}\nmean = {np.mean(data):.6f} cm")
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.grid(True, alpha=0.25)


def save_overview_figure(output_path: Path, incident_energy_mev: float, result: dict[str, np.ndarray], bins: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(f"Compton Scatter and Free-Path Monte Carlo in NaI, E0 = {incident_energy_mev:.3f} MeV")

    build_hist_axes(axes[0, 0], result["theta_deg"], bins, "Scattering Angle", "Theta (deg)")
    build_hist_axes(axes[0, 1], result["scattered_energy_mev"], bins, "Scattered Photon Energy", "Energy (MeV)")
    build_hist_axes(axes[0, 2], result["mu_total_cm_inv"], bins, "Attenuation Coefficient", "Mu_total (cm^-1)")
    build_hist_axes(
        axes[1, 0],
        result["free_path_cm"],
        bins,
        "Free Path",
        "Free path (cm)",
        mean_in_title=True,
    )
    build_hist_axes(
        axes[1, 1],
        result["perpendicular_free_path_cm"],
        bins,
        "Perpendicular Free Path",
        "Perpendicular component (cm)",
        mean_in_title=True,
    )
    build_hist_axes(
        axes[1, 2],
        result["parallel_free_path_cm"],
        bins,
        "Parallel Free Path",
        "Parallel component (cm)",
        mean_in_title=True,
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_component_figure(output_path: Path, result: dict[str, np.ndarray], bins: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    build_hist_axes(
        axes[0],
        result["parallel_free_path_cm"],
        bins,
        "Parallel Component (Signed)",
        "Parallel free path (cm)",
        mean_in_title=True,
    )
    build_hist_axes(
        axes[1],
        np.abs(result["parallel_free_path_cm"]),
        bins,
        "Parallel Component (Absolute)",
        "|Parallel free path| (cm)",
        mean_in_title=True,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cross_section_table = load_cross_section_table(args.data_path.resolve())
    result = simulate_scattered_photons(
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
    component_fig_path = output_dir / "parallel_component.png"

    save_summary_json(summary_path, summary)
    if not args.skip_samples_csv:
        save_samples_csv(samples_path, result)
    save_overview_figure(overview_fig_path, args.incident_energy_mev, result, args.num_bins)
    save_component_figure(component_fig_path, result, args.num_bins)

    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
