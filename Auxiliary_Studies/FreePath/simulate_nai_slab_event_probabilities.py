from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulate_compton_freepath import (
    NAI_DENSITY_G_CM3,
    interpolate_cross_section,
    load_cross_section_table,
    sample_klein_nishina_mu,
)


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values64 = np.asarray(values, dtype=np.float64)
        self.count += int(values64.size)
        self.sum += float(np.sum(values64))
        self.sum_sq += float(np.sum(values64 * values64))
        self.min = min(self.min, float(np.min(values64)))
        self.max = max(self.max, float(np.max(values64)))

    def to_dict(self) -> dict[str, float] | None:
        if self.count == 0:
            return None
        mean = self.sum / self.count
        variance = max(self.sum_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": self.min,
            "max": self.max,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo simulation for 511 keV photons incident normally on a NaI slab. "
            "The script estimates the probabilities of four event classes: "
            "direct photoelectric absorption, one Compton scatter followed by a second "
            "interaction inside the slab, one Compton scatter followed by escape through "
            "the entrance/exit surfaces, and direct transmission without interaction."
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
        "--thickness-mm",
        type=float,
        default=10.0,
        help="NaI slab thickness in mm.",
    )
    parser.add_argument(
        "--num-photons",
        type=int,
        default=1000000,
        help="Number of incident photons to simulate.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000000,
        help="Photon count processed per chunk to limit memory usage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260420,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "slab_event_output",
        help="Output directory for the summary JSON.",
    )
    return parser.parse_args()


def simulate_chunk(
    photon_count: int,
    incident_energy_mev: float,
    thickness_cm: float,
    cross_section_table: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, object]:
    energy_axis = cross_section_table["energy_mev"]
    total_axis = cross_section_table["total_wo_coherent_cm2_g"]
    photo_axis = cross_section_table["photoelectric_cm2_g"]

    total_cs = float(interpolate_cross_section(np.array([incident_energy_mev]), energy_axis, total_axis)[0])
    photo_cs = float(interpolate_cross_section(np.array([incident_energy_mev]), energy_axis, photo_axis)[0])
    mu_total = total_cs * NAI_DENSITY_G_CM3
    if mu_total <= 0.0:
        raise RuntimeError("Total attenuation coefficient must be positive.")

    photo_fraction = np.clip(photo_cs / total_cs, 0.0, 1.0)

    first_free_path_cm = -np.log(np.maximum(rng.uniform(0.0, 1.0, size=photon_count), 1.0e-15)) / mu_total
    interacted = first_free_path_cm < thickness_cm
    transmit_mask = ~interacted

    direct_photo_mask = np.zeros(photon_count, dtype=bool)
    compton_mask = np.zeros(photon_count, dtype=bool)

    interacted_count = int(np.count_nonzero(interacted))
    if interacted_count > 0:
        first_is_photo = rng.uniform(0.0, 1.0, size=interacted_count) < photo_fraction
        interacted_indices = np.flatnonzero(interacted)
        direct_photo_mask[interacted_indices] = first_is_photo
        compton_mask[interacted_indices] = ~first_is_photo

    compton_count = int(np.count_nonzero(compton_mask))

    compton_then_absorb_count = 0
    compton_escape_top_count = 0
    compton_escape_bottom_count = 0
    energy_stats = {
        "compton_first_deposit_mev": RunningStats(),
        "compton_escape_scattered_energy_mev": RunningStats(),
        "compton_absorbed_scattered_energy_mev": RunningStats(),
    }

    if compton_count > 0:
        compton_depth_cm = first_free_path_cm[compton_mask]
        mu_scatter = sample_klein_nishina_mu(incident_energy_mev, compton_count, rng)
        scattered_energy_mev = incident_energy_mev / (
            1.0 + (incident_energy_mev / 0.511) * (1.0 - mu_scatter)
        )
        first_deposit_mev = incident_energy_mev - scattered_energy_mev

        total_cs_scattered = interpolate_cross_section(scattered_energy_mev, energy_axis, total_axis)
        mu_total_scattered = np.maximum(total_cs_scattered * NAI_DENSITY_G_CM3, 1.0e-15)
        second_free_path_cm = -np.log(np.maximum(rng.uniform(0.0, 1.0, size=compton_count), 1.0e-15)) / mu_total_scattered

        boundary_distance_cm = np.full(compton_count, np.inf, dtype=np.float64)
        moving_down = mu_scatter > 0.0
        moving_up = mu_scatter < 0.0
        boundary_distance_cm[moving_down] = (thickness_cm - compton_depth_cm[moving_down]) / mu_scatter[moving_down]
        boundary_distance_cm[moving_up] = compton_depth_cm[moving_up] / (-mu_scatter[moving_up])

        absorbed_inside = second_free_path_cm <= boundary_distance_cm
        escaped = ~absorbed_inside
        escaped_top = escaped & moving_up
        escaped_bottom = escaped & moving_down

        compton_then_absorb_count = int(np.count_nonzero(absorbed_inside))
        compton_escape_top_count = int(np.count_nonzero(escaped_top))
        compton_escape_bottom_count = int(np.count_nonzero(escaped_bottom))

        energy_stats["compton_first_deposit_mev"].update(first_deposit_mev)
        energy_stats["compton_escape_scattered_energy_mev"].update(scattered_energy_mev[escaped])
        energy_stats["compton_absorbed_scattered_energy_mev"].update(scattered_energy_mev[absorbed_inside])

    return {
        "counts": {
            "direct_photoelectric_absorption": int(np.count_nonzero(direct_photo_mask)),
            "compton_then_absorbed_inside": compton_then_absorb_count,
            "compton_then_escape_top": compton_escape_top_count,
            "compton_then_escape_bottom": compton_escape_bottom_count,
            "compton_then_escape_any": compton_escape_top_count + compton_escape_bottom_count,
            "direct_transmission_without_interaction": int(np.count_nonzero(transmit_mask)),
            "first_interaction_compton_total": compton_count,
            "first_interaction_any_total": interacted_count,
        },
        "energy_stats": energy_stats,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    thickness_cm = args.thickness_mm / 10.0
    rng = np.random.default_rng(args.seed)
    cross_section_table = load_cross_section_table(args.data_path.resolve())

    total_counts = {
        "direct_photoelectric_absorption": 0,
        "compton_then_absorbed_inside": 0,
        "compton_then_escape_top": 0,
        "compton_then_escape_bottom": 0,
        "compton_then_escape_any": 0,
        "direct_transmission_without_interaction": 0,
        "first_interaction_compton_total": 0,
        "first_interaction_any_total": 0,
    }
    aggregate_energy_stats = {
        "compton_first_deposit_mev": RunningStats(),
        "compton_escape_scattered_energy_mev": RunningStats(),
        "compton_absorbed_scattered_energy_mev": RunningStats(),
    }

    remaining = args.num_photons
    while remaining > 0:
        chunk = min(remaining, args.chunk_size)
        chunk_result = simulate_chunk(
            photon_count=chunk,
            incident_energy_mev=args.incident_energy_mev,
            thickness_cm=thickness_cm,
            cross_section_table=cross_section_table,
            rng=rng,
        )
        for key, value in chunk_result["counts"].items():
            total_counts[key] += value
        for key, chunk_stats in chunk_result["energy_stats"].items():
            aggregate_energy_stats[key].count += chunk_stats.count
            aggregate_energy_stats[key].sum += chunk_stats.sum
            aggregate_energy_stats[key].sum_sq += chunk_stats.sum_sq
            aggregate_energy_stats[key].min = min(aggregate_energy_stats[key].min, chunk_stats.min)
            aggregate_energy_stats[key].max = max(aggregate_energy_stats[key].max, chunk_stats.max)
        remaining -= chunk

    total_photons = float(args.num_photons)
    summary = {
        "config": {
            "incident_energy_mev": args.incident_energy_mev,
            "thickness_mm": args.thickness_mm,
            "thickness_cm": thickness_cm,
            "num_photons": args.num_photons,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "density_g_cm3": NAI_DENSITY_G_CM3,
            "data_path": str(args.data_path.resolve()),
            "assumptions": [
                "Normal incidence on a laterally infinite NaI slab.",
                "Only escape through the entrance surface or exit surface is considered.",
                "Category 2 counts photons whose first interaction is Compton and whose scattered photon undergoes any next interaction inside the slab.",
            ],
        },
        "counts": total_counts,
        "probabilities": {
            "direct_photoelectric_absorption": total_counts["direct_photoelectric_absorption"] / total_photons,
            "compton_then_absorbed_inside": total_counts["compton_then_absorbed_inside"] / total_photons,
            "compton_then_escape_any": total_counts["compton_then_escape_any"] / total_photons,
            "compton_then_escape_top": total_counts["compton_then_escape_top"] / total_photons,
            "compton_then_escape_bottom": total_counts["compton_then_escape_bottom"] / total_photons,
            "direct_transmission_without_interaction": total_counts["direct_transmission_without_interaction"] / total_photons,
            "first_interaction_any": total_counts["first_interaction_any_total"] / total_photons,
            "first_interaction_compton": total_counts["first_interaction_compton_total"] / total_photons,
        },
        "energy_summaries": {
            key: stats.to_dict() for key, stats in aggregate_energy_stats.items()
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
