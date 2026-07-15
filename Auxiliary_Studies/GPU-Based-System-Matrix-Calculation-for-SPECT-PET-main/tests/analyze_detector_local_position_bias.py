#!/usr/bin/env python3
"""Quantify the crystal-center bias in the detector-local second-PE model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CASES = (
    (218.0, (3.0, 3.0, 3.0)),
    (218.0, (2.0, 6.0, 2.0)),
    (440.0, (3.0, 3.0, 3.0)),
    (440.0, (2.0, 6.0, 2.0)),
)


def sample_klein_nishina_cosine(
    rng: np.random.Generator,
    count: int,
    energy_kev: float,
) -> np.ndarray:
    output = np.empty(count, dtype=np.float64)
    alpha = energy_kev / 511.0
    filled = 0
    while filled < count:
        proposal_count = int((count - filled) * 2.2) + 100
        cosine = rng.uniform(-1.0, 1.0, proposal_count)
        factor1 = alpha * (1.0 - cosine)
        factor2 = 1.0 + cosine * cosine
        weight = factor2 / (1.0 + factor1) ** 2 * (
            1.0 + factor1 * factor1 / (factor2 * (1.0 + factor1))
        )
        accepted = cosine[rng.random(proposal_count) < weight / 2.0]
        take = min(len(accepted), count - filled)
        output[filled : filled + take] = accepted[:take]
        filled += take
    return output


def evaluate_case(
    rng: np.random.Generator,
    sample_count: int,
    energy_kev: float,
    dimensions_mm: tuple[float, float, float],
    energy_grid: np.ndarray,
    mu_pe_grid: np.ndarray,
    mu_compton_grid: np.ndarray,
) -> dict[str, object]:
    width, thickness, height = dimensions_mm
    half_width = width / 2.0
    half_thickness = thickness / 2.0
    half_height = height / 2.0
    mu_pe_source = float(np.interp(energy_kev, energy_grid, mu_pe_grid))
    mu_compton_source = float(
        np.interp(energy_kev, energy_grid, mu_compton_grid)
    )
    mu_total_source = mu_pe_source + mu_compton_source

    cosine = sample_klein_nishina_cosine(rng, sample_count, energy_kev)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, sample_count)
    sine = np.sqrt(1.0 - cosine * cosine)
    direction_x = sine * np.cos(azimuth)
    direction_y = cosine
    direction_z = sine * np.sin(azimuth)
    alpha = energy_kev / 511.0
    scattered_energy = energy_kev / (1.0 + alpha * (1.0 - cosine))
    mu_pe = np.interp(scattered_energy, energy_grid, mu_pe_grid)
    mu_compton = np.interp(scattered_energy, energy_grid, mu_compton_grid)
    mu_total = mu_pe + mu_compton

    center_path = np.minimum.reduce(
        (
            half_width / np.maximum(np.abs(direction_x), 1e-30),
            half_thickness / np.maximum(np.abs(direction_y), 1e-30),
            half_height / np.maximum(np.abs(direction_z), 1e-30),
        )
    )
    center_second_pe = float(
        np.mean((1.0 - np.exp(-mu_total * center_path)) * mu_pe / mu_total)
    )

    first_x = rng.uniform(-half_width, half_width, sample_count)
    first_z = rng.uniform(-half_height, half_height, sample_count)
    uniform = rng.random(sample_count)
    first_depth = -np.log(
        1.0 - uniform * (1.0 - np.exp(-mu_total_source * thickness))
    ) / mu_total_source
    first_y = -half_thickness + first_depth
    path_x = np.where(
        direction_x > 0,
        (half_width - first_x) / direction_x,
        (-half_width - first_x) / direction_x,
    )
    path_y = np.where(
        direction_y > 0,
        (half_thickness - first_y) / direction_y,
        (-half_thickness - first_y) / direction_y,
    )
    path_z = np.where(
        direction_z > 0,
        (half_height - first_z) / direction_z,
        (-half_height - first_z) / direction_z,
    )
    distributed_path = np.minimum.reduce((path_x, path_y, path_z))
    distributed_second_pe = float(
        np.mean(
            (1.0 - np.exp(-mu_total * distributed_path)) * mu_pe / mu_total
        )
    )

    return {
        "energy_keV": energy_kev,
        "dimensions_mm": list(dimensions_mm),
        "source_mu_total_per_mm": mu_total_source,
        "center_second_photoelectric_probability": center_second_pe,
        "distributed_second_photoelectric_probability": distributed_second_pe,
        "center_to_distributed_ratio": center_second_pe / distributed_second_pe,
        "center_mean_exit_path_mm": float(np.mean(center_path)),
        "distributed_mean_exit_path_mm": float(np.mean(distributed_path)),
    }


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    table = np.loadtxt(
        repository / "physics_data" / "nist_xcom_materials_1_1000keV.csv",
        delimiter=",",
        skiprows=4,
    )
    energy_grid = table[:, 0]
    mu_pe_grid = table[:, 7]
    mu_compton_grid = table[:, 8]
    rng = np.random.default_rng(args.seed)
    results = [
        evaluate_case(
            rng,
            args.samples,
            energy,
            dimensions,
            energy_grid,
            mu_pe_grid,
            mu_compton_grid,
        )
        for energy, dimensions in CASES
    ]
    report = {
        "method": (
            "Normal incidence; uniform entrance-face position; conditional "
            "exponential first-interaction depth; Klein-Nishina outgoing angle"
        ),
        "sample_count_per_case": args.samples,
        "seed": args.seed,
        "material": "GAGG",
        "results": results,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
