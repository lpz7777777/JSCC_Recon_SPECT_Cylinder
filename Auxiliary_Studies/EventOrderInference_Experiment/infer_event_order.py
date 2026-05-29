from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ELECTRON_REST_MEV = 0.511
SCORE_EPS = 1.0e-12


@dataclass
class FileMetrics:
    file_name: str
    filtered_cross_layer_event_count: int
    truth_front_first_count: int
    pred_front_first_count: int
    correct_count: int
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-hit Compton event order inference using center_geometry only."
    )
    parser.add_argument("--list-dir", type=Path, required=True, help="Directory containing List CSV files.")
    parser.add_argument("--detector-csv", type=Path, required=True, help="Detector.csv path.")
    parser.add_argument("--energy-mev", type=float, default=0.511, help="Incident photon energy in MeV.")
    parser.add_argument(
        "--front-prior-ratio",
        type=float,
        default=1.2,
        help="Prior multiplier for the hypothesis that the front-layer hit occurred first.",
    )
    parser.add_argument(
        "--geometry-sigma-deg",
        type=float,
        default=20.0,
        help="Angular width in degrees for the center-geometry cone-consistency term.",
    )
    parser.add_argument(
        "--geometry-power",
        type=float,
        default=1.0,
        help="Exponent applied to the geometry term before combining with the KN score.",
    )
    parser.add_argument(
        "--ene-resolution-662keV",
        type=float,
        default=0.1,
        help="Reference energy resolution, consistent with the reconstruction scripts.",
    )
    parser.add_argument(
        "--ene-threshold-min",
        type=float,
        default=0.05,
        help="Lower energy threshold applied to both interactions after smearing.",
    )
    parser.add_argument(
        "--ene-threshold-sum",
        type=float,
        default=0.46,
        help="Lower threshold for e1 + e2 after smearing.",
    )
    parser.add_argument(
        "--disable-energy-smear",
        action="store_true",
        help="Disable energy smearing and evaluate on the original simulated energies.",
    )
    parser.add_argument(
        "--smear-repeats",
        type=int,
        default=5,
        help="Number of random-smearing repeats used to estimate accuracy under energy resolution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260331,
        help="Base random seed used by the energy smearing evaluation.",
    )
    parser.add_argument(
        "--reject-frac-list",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
        help="Fractions of lowest-confidence events to reject before computing retained accuracy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to EventOrderInference_Experiment/<list_dir_name>_results",
    )
    return parser.parse_args()


def load_detector(detector_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    detector = np.genfromtxt(detector_csv, delimiter=",", dtype=np.float64)
    if detector.ndim != 2 or detector.shape[1] < 4:
        raise ValueError(f"Detector CSV must have at least 4 columns: {detector_csv}")

    detector_pos = detector[:, 1:4]
    detector_y_abs = np.abs(detector_pos[:, 1])
    layer_values = np.sort(np.unique(np.round(detector_y_abs, decimals=6)))
    if layer_values.size < 2:
        raise ValueError("Failed to resolve detector layers from Detector.csv")

    layer_by_det = np.zeros(detector.shape[0], dtype=np.int32)
    for idx, value in enumerate(layer_values, start=1):
        layer_by_det[np.isclose(detector_y_abs, value, atol=1e-6)] = idx

    if np.any(layer_by_det == 0):
        raise ValueError("Some detector elements were not assigned to a layer.")

    return detector_pos, layer_by_det


def numeric_csv_files(list_dir: Path) -> list[Path]:
    csv_files = [p for p in list_dir.glob("*.csv") if p.is_file()]

    def sort_key(path: Path) -> tuple[int, int | str]:
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    return sorted(csv_files, key=sort_key)


def compton_theta_from_first_deposit(first_deposit_mev: np.ndarray, incident_energy_mev: float) -> np.ndarray:
    e0 = incident_energy_mev
    e1 = first_deposit_mev

    valid_energy = (e1 > 0.0) & (e1 < e0)
    denom = (e0 - e1) * e0
    cos_theta = np.full_like(e1, fill_value=np.nan, dtype=np.float64)
    cos_theta[valid_energy] = 1.0 - (ELECTRON_REST_MEV * e1[valid_energy]) / denom[valid_energy]

    valid = valid_energy & (cos_theta > -1.0) & (cos_theta < 1.0)
    theta = np.full_like(e1, fill_value=np.nan, dtype=np.float64)
    theta[valid] = np.arccos(np.clip(cos_theta[valid], -1.0 + 1e-9, 1.0 - 1.0e-9))
    return theta


def klein_nishina_weight_from_first_deposit(first_deposit_mev: np.ndarray, incident_energy_mev: float) -> np.ndarray:
    theta = compton_theta_from_first_deposit(first_deposit_mev, incident_energy_mev)
    weight = np.zeros_like(first_deposit_mev, dtype=np.float64)
    valid = np.isfinite(theta)
    if np.any(valid):
        scattered_energy = incident_energy_mev - first_deposit_mev[valid]
        weight[valid] = (
            incident_energy_mev / scattered_energy
            + scattered_energy / incident_energy_mev
            - np.sin(theta[valid]) ** 2
        )
    return weight


def center_geometry_weight(
    detector_pos: np.ndarray,
    first_det: np.ndarray,
    second_det: np.ndarray,
    theta: np.ndarray,
    sigma_deg: float,
) -> np.ndarray:
    sigma_rad = math.radians(sigma_deg)
    first_pos = detector_pos[first_det]
    second_pos = detector_pos[second_det]

    vector01 = first_pos
    vector12 = second_pos - first_pos
    norm01 = np.linalg.norm(vector01, axis=1)
    norm12 = np.linalg.norm(vector12, axis=1)
    valid = np.isfinite(theta) & (norm01 > 0.0) & (norm12 > 0.0)

    beta = np.full(theta.shape, fill_value=np.nan, dtype=np.float64)
    if np.any(valid):
        cos_beta = np.sum(vector01[valid] * vector12[valid], axis=1) / (norm01[valid] * norm12[valid])
        beta[valid] = np.arccos(np.clip(cos_beta, -1.0 + 1e-9, 1.0 - 1.0e-9))

    weight = np.zeros(theta.shape, dtype=np.float64)
    valid = valid & np.isfinite(beta)
    if np.any(valid):
        weight[valid] = np.exp(-0.5 * ((beta[valid] - theta[valid]) / sigma_rad) ** 2)
    return weight


def compute_energy_resolution(e0: float, ene_resolution_662keV: float) -> float:
    return ene_resolution_662keV * (0.662 / e0) ** 0.5


def compute_energy_threshold_max(e0: float) -> float:
    return 2.0 * e0 ** 2 / (0.511 + 2.0 * e0) - 0.001


def smear_energies_like_recon(
    e1: np.ndarray,
    e2: np.ndarray,
    e0: float,
    ene_resolution: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    e1_safe = np.maximum(e1, 1.0e-12)
    e2_safe = np.maximum(e2, 1.0e-12)
    sigma_1 = e1_safe * ene_resolution / 2.355 * np.sqrt(e0 / e1_safe)
    sigma_2 = e2_safe * ene_resolution / 2.355 * np.sqrt(e0 / e2_safe)
    return e1 + sigma_1 * rng.standard_normal(e1.shape), e2 + sigma_2 * rng.standard_normal(e2.shape)


def filter_events_like_recon(
    cpnum1: np.ndarray,
    cpnum2: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    e0: float,
    ene_threshold_max: float,
    ene_threshold_min: float,
    ene_threshold_sum: float,
) -> np.ndarray:
    flag_max_1 = e1 < ene_threshold_max
    flag_min_1 = e1 > ene_threshold_min
    flag_min_2 = e2 > ene_threshold_min
    flag_sum = (e1 + e2) > ene_threshold_sum
    flag = flag_max_1 & flag_min_1 & flag_min_2 & flag_sum

    cos_theta_raw = np.full(e1.shape, fill_value=np.nan, dtype=np.float64)
    valid_energy = flag & (e1 > 0.0) & (e1 < e0)
    denom = (e0 - e1) * e0
    cos_theta_raw[valid_energy] = 1.0 - (ELECTRON_REST_MEV * e1[valid_energy]) / denom[valid_energy]
    valid_kinematics = (cos_theta_raw > -1.0 + 1e-6) & (cos_theta_raw < 1.0 - 1e-6)

    valid_det = cpnum1 != cpnum2
    return flag & valid_kinematics & valid_det


def summarize_rejection_sweep(
    confidence: np.ndarray,
    correct: np.ndarray,
    reject_fractions: list[float],
) -> list[dict]:
    total_count = int(confidence.size)
    if total_count == 0:
        return []

    order = np.argsort(confidence, kind="stable")
    sweep = []
    for reject_fraction in reject_fractions:
        reject_count = int(math.floor(reject_fraction * total_count))
        reject_count = min(max(reject_count, 0), total_count)

        retain_mask = np.ones(total_count, dtype=bool)
        if reject_count > 0:
            retain_mask[order[:reject_count]] = False

        retained_count = int(np.count_nonzero(retain_mask))
        retained_correct_count = int(np.count_nonzero(correct[retain_mask]))
        confidence_threshold = (
            float(confidence[order[reject_count - 1]])
            if reject_count > 0
            else 0.0
        )
        retained_accuracy = (
            retained_correct_count / retained_count
            if retained_count > 0
            else float("nan")
        )

        sweep.append(
            {
                "reject_fraction_target": float(reject_fraction),
                "reject_fraction_actual": reject_count / total_count,
                "reject_count": reject_count,
                "retained_count": retained_count,
                "retained_fraction": retained_count / total_count,
                "retained_correct_count": retained_correct_count,
                "retained_accuracy": retained_accuracy,
                "confidence_threshold": confidence_threshold,
            }
        )

    return sweep


def load_raw_cross_layer_events(
    list_dir: Path,
    detector_pos: np.ndarray,
    layer_by_det: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    front_mask = layer_by_det <= (layer_by_det.max() - 1)
    rear_mask = layer_by_det == layer_by_det.max()

    records: list[dict] = []
    cpnum1_parts = []
    cpnum2_parts = []
    e1_parts = []
    e2_parts = []
    truth_front_first_parts = []
    total_cross_layer = 0

    for csv_path in numeric_csv_files(list_dir):
        if csv_path.name.startswith("compton_"):
            continue

        data = np.genfromtxt(csv_path, delimiter=",", dtype=np.float64)
        if data.size == 0:
            records.append({"file_name": csv_path.name, "raw_cross_layer_count": 0, "start": 0, "end": 0})
            continue

        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 4:
            raise ValueError(f"List CSV must have at least 4 columns: {csv_path}")

        cpnum1 = data[:, 0].astype(np.int64) - 1
        cpnum2 = data[:, 2].astype(np.int64) - 1
        e1 = data[:, 1]
        e2 = data[:, 3]

        valid = (
            (cpnum1 >= 0)
            & (cpnum1 < layer_by_det.size)
            & (cpnum2 >= 0)
            & (cpnum2 < layer_by_det.size)
            & np.isfinite(e1)
            & np.isfinite(e2)
        )
        cpnum1 = cpnum1[valid]
        cpnum2 = cpnum2[valid]
        e1 = e1[valid]
        e2 = e2[valid]

        first_is_front = front_mask[cpnum1]
        first_is_rear = rear_mask[cpnum1]
        second_is_front = front_mask[cpnum2]
        second_is_rear = rear_mask[cpnum2]
        cross_layer = (first_is_front & second_is_rear) | (first_is_rear & second_is_front)

        cpnum1 = cpnum1[cross_layer]
        cpnum2 = cpnum2[cross_layer]
        e1 = e1[cross_layer]
        e2 = e2[cross_layer]
        truth_front_first = (front_mask[cpnum1] & rear_mask[cpnum2])

        start = total_cross_layer
        end = start + cpnum1.size
        records.append(
            {
                "file_name": csv_path.name,
                "raw_cross_layer_count": int(cpnum1.size),
                "start": start,
                "end": end,
            }
        )

        total_cross_layer = end
        cpnum1_parts.append(cpnum1)
        cpnum2_parts.append(cpnum2)
        e1_parts.append(e1)
        e2_parts.append(e2)
        truth_front_first_parts.append(truth_front_first)

    if total_cross_layer == 0:
        raise RuntimeError("No cross-layer front/rear events were found in the selected list directory.")

    return (
        records,
        np.concatenate(cpnum1_parts),
        np.concatenate(cpnum2_parts),
        np.concatenate(e1_parts),
        np.concatenate(e2_parts),
        np.concatenate(truth_front_first_parts),
    )


def evaluate_single_seed(
    detector_pos: np.ndarray,
    layer_by_det: np.ndarray,
    records: list[dict],
    cpnum1_raw: np.ndarray,
    cpnum2_raw: np.ndarray,
    e1_raw: np.ndarray,
    e2_raw: np.ndarray,
    truth_front_first_raw: np.ndarray,
    e0: float,
    front_prior_ratio: float,
    geometry_sigma_deg: float,
    geometry_power: float,
    ene_resolution: float,
    ene_threshold_max: float,
    ene_threshold_min: float,
    ene_threshold_sum: float,
    apply_energy_smear: bool,
    seed: int,
    reject_fractions: list[float],
) -> tuple[dict, list[FileMetrics]]:
    rng = np.random.default_rng(seed)
    if apply_energy_smear:
        e1_obs, e2_obs = smear_energies_like_recon(e1_raw, e2_raw, e0, ene_resolution, rng)
    else:
        e1_obs = e1_raw.copy()
        e2_obs = e2_raw.copy()

    valid = filter_events_like_recon(
        cpnum1=cpnum1_raw,
        cpnum2=cpnum2_raw,
        e1=e1_obs,
        e2=e2_obs,
        e0=e0,
        ene_threshold_max=ene_threshold_max,
        ene_threshold_min=ene_threshold_min,
        ene_threshold_sum=ene_threshold_sum,
    )

    cpnum1 = cpnum1_raw[valid]
    cpnum2 = cpnum2_raw[valid]
    e1 = e1_obs[valid]
    e2 = e2_obs[valid]
    truth_front_first = truth_front_first_raw[valid]

    front_mask = layer_by_det <= (layer_by_det.max() - 1)
    rear_mask = layer_by_det == layer_by_det.max()
    front_first_det = np.where(front_mask[cpnum1], cpnum1, cpnum2)
    rear_first_det = np.where(rear_mask[cpnum1], cpnum1, cpnum2)
    front_second_det = np.where(front_mask[cpnum1], cpnum2, cpnum1)
    rear_second_det = np.where(rear_mask[cpnum1], cpnum2, cpnum1)
    front_energy = np.where(front_mask[cpnum1], e1, e2)
    rear_energy = np.where(rear_mask[cpnum1], e1, e2)

    front_theta = compton_theta_from_first_deposit(front_energy, e0)
    rear_theta = compton_theta_from_first_deposit(rear_energy, e0)
    front_kn = front_prior_ratio * klein_nishina_weight_from_first_deposit(front_energy, e0)
    rear_kn = klein_nishina_weight_from_first_deposit(rear_energy, e0)
    front_center = center_geometry_weight(detector_pos, front_first_det, front_second_det, front_theta, geometry_sigma_deg)
    rear_center = center_geometry_weight(detector_pos, rear_first_det, rear_second_det, rear_theta, geometry_sigma_deg)
    front_score = front_kn * np.power(front_center, geometry_power)
    rear_score = rear_kn * np.power(rear_center, geometry_power)
    pred_front_first = front_score >= rear_score
    correct = pred_front_first == truth_front_first
    confidence = np.abs(np.log(front_score + SCORE_EPS) - np.log(rear_score + SCORE_EPS))
    rejection_sweep = summarize_rejection_sweep(confidence, correct, reject_fractions)

    total_filtered_cross_layer = int(cpnum1.size)
    total_truth_front_first = int(np.count_nonzero(truth_front_first))
    total_pred_front_first = int(np.count_nonzero(pred_front_first))
    total_correct = int(np.count_nonzero(correct))

    summary = {
        "seed": seed,
        "apply_energy_smear": apply_energy_smear,
        "filtered_cross_layer_event_count": total_filtered_cross_layer,
        "truth_front_first_count": total_truth_front_first,
        "truth_front_first_ratio": total_truth_front_first / total_filtered_cross_layer if total_filtered_cross_layer > 0 else float("nan"),
        "pred_front_first_count": total_pred_front_first,
        "pred_front_first_ratio": total_pred_front_first / total_filtered_cross_layer if total_filtered_cross_layer > 0 else float("nan"),
        "correct_count": total_correct,
        "accuracy": total_correct / total_filtered_cross_layer if total_filtered_cross_layer > 0 else float("nan"),
        "rejection_sweep": rejection_sweep,
    }

    per_file_metrics: list[FileMetrics] = []
    raw_valid = valid.astype(np.int8)
    valid_prefix = np.cumsum(raw_valid)
    valid_prefix = np.concatenate(([0], valid_prefix))

    for record in records:
        raw_count = record["raw_cross_layer_count"]
        raw_start = record["start"]
        raw_end = record["end"]
        if raw_count == 0:
            per_file_metrics.append(
                FileMetrics(
                    file_name=record["file_name"],
                    filtered_cross_layer_event_count=0,
                    truth_front_first_count=0,
                    pred_front_first_count=0,
                    correct_count=0,
                    accuracy=float("nan"),
                )
            )
            continue

        filtered_start = int(valid_prefix[raw_start])
        filtered_end = int(valid_prefix[raw_end])
        filtered_count = filtered_end - filtered_start
        if filtered_count == 0:
            per_file_metrics.append(
                FileMetrics(
                    file_name=record["file_name"],
                    filtered_cross_layer_event_count=0,
                    truth_front_first_count=0,
                    pred_front_first_count=0,
                    correct_count=0,
                    accuracy=float("nan"),
                )
            )
            continue

        truth_part = truth_front_first[filtered_start:filtered_end]
        pred_part = pred_front_first[filtered_start:filtered_end]
        correct_part = correct[filtered_start:filtered_end]
        per_file_metrics.append(
            FileMetrics(
                file_name=record["file_name"],
                filtered_cross_layer_event_count=filtered_count,
                truth_front_first_count=int(np.count_nonzero(truth_part)),
                pred_front_first_count=int(np.count_nonzero(pred_part)),
                correct_count=int(np.count_nonzero(correct_part)),
                accuracy=int(np.count_nonzero(correct_part)) / filtered_count,
            )
        )

    return summary, per_file_metrics


def aggregate_summaries(per_seed_summaries: list[dict]) -> dict:
    keys = [
        "filtered_cross_layer_event_count",
        "truth_front_first_ratio",
        "pred_front_first_ratio",
        "accuracy",
    ]
    aggregate = {
        "repeat_count": len(per_seed_summaries),
        "seed_list": [item["seed"] for item in per_seed_summaries],
    }
    for key in keys:
        values = np.array([item[key] for item in per_seed_summaries], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    rejection_by_fraction: dict[float, list[dict]] = {}
    for item in per_seed_summaries:
        for reject_item in item.get("rejection_sweep", []):
            fraction = float(reject_item["reject_fraction_target"])
            rejection_by_fraction.setdefault(fraction, []).append(reject_item)

    aggregate["rejection_sweep"] = []
    for fraction in sorted(rejection_by_fraction):
        entries = rejection_by_fraction[fraction]
        retained_accuracy = np.array([x["retained_accuracy"] for x in entries], dtype=np.float64)
        retained_count = np.array([x["retained_count"] for x in entries], dtype=np.float64)
        rejected_count = np.array([x["reject_count"] for x in entries], dtype=np.float64)
        actual_fraction = np.array([x["reject_fraction_actual"] for x in entries], dtype=np.float64)
        aggregate["rejection_sweep"].append(
            {
                "reject_fraction_target": fraction,
                "reject_fraction_actual": {
                    "mean": float(np.mean(actual_fraction)),
                    "std": float(np.std(actual_fraction)),
                    "min": float(np.min(actual_fraction)),
                    "max": float(np.max(actual_fraction)),
                },
                "reject_count": {
                    "mean": float(np.mean(rejected_count)),
                    "std": float(np.std(rejected_count)),
                    "min": float(np.min(rejected_count)),
                    "max": float(np.max(rejected_count)),
                },
                "retained_count": {
                    "mean": float(np.mean(retained_count)),
                    "std": float(np.std(retained_count)),
                    "min": float(np.min(retained_count)),
                    "max": float(np.max(retained_count)),
                },
                "retained_accuracy": {
                    "mean": float(np.mean(retained_accuracy)),
                    "std": float(np.std(retained_accuracy)),
                    "min": float(np.min(retained_accuracy)),
                    "max": float(np.max(retained_accuracy)),
                },
            }
        )
    return aggregate


def write_per_file_csv(output_path: Path, per_file_metrics: list[FileMetrics]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_name",
                "filtered_cross_layer_event_count",
                "truth_front_first_count",
                "pred_front_first_count",
                "correct_count",
                "accuracy",
            ]
        )
        for item in per_file_metrics:
            writer.writerow(
                [
                    item.file_name,
                    item.filtered_cross_layer_event_count,
                    item.truth_front_first_count,
                    item.pred_front_first_count,
                    item.correct_count,
                    item.accuracy,
                ]
            )


def write_report(output_path: Path, config: dict, aggregate: dict, per_seed_summaries: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Center-geometry event order inference report\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\nAggregate over smearing repeats\n")
        for key, value in aggregate.items():
            if isinstance(value, dict):
                f.write(
                    f"  {key}: mean={value['mean']:.6f}, std={value['std']:.6f}, "
                    f"min={value['min']:.6f}, max={value['max']:.6f}\n"
                )
            else:
                f.write(f"  {key}: {value}\n")

        f.write("\nPer-seed results\n")
        for item in per_seed_summaries:
            f.write(
                f"  seed={item['seed']}, "
                f"filtered_cross_layer_event_count={item['filtered_cross_layer_event_count']}, "
                f"accuracy={item['accuracy']:.6f}, "
                f"truth_front_first_ratio={item['truth_front_first_ratio']:.6f}, "
                f"pred_front_first_ratio={item['pred_front_first_ratio']:.6f}\n"
            )
            for reject_item in item.get("rejection_sweep", []):
                f.write(
                    f"    reject_fraction_target={reject_item['reject_fraction_target']:.3f}, "
                    f"reject_fraction_actual={reject_item['reject_fraction_actual']:.6f}, "
                    f"retained_count={reject_item['retained_count']}, "
                    f"retained_accuracy={reject_item['retained_accuracy']:.6f}\n"
                )


def main() -> None:
    args = parse_args()
    list_dir = args.list_dir.resolve()
    detector_csv = args.detector_csv.resolve()
    if not list_dir.is_dir():
        raise FileNotFoundError(f"List directory not found: {list_dir}")
    if not detector_csv.is_file():
        raise FileNotFoundError(f"Detector CSV not found: {detector_csv}")

    if args.smear_repeats <= 0:
        raise ValueError("--smear-repeats must be positive.")
    if any((x < 0.0) or (x >= 1.0) for x in args.reject_frac_list):
        raise ValueError("Each value in --reject-frac-list must satisfy 0 <= x < 1.")

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / f"{list_dir.name}_results"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_pos, layer_by_det = load_detector(detector_csv)
    records, cpnum1_raw, cpnum2_raw, e1_raw, e2_raw, truth_front_first_raw = load_raw_cross_layer_events(
        list_dir,
        detector_pos,
        layer_by_det,
    )

    ene_resolution = compute_energy_resolution(args.energy_mev, args.ene_resolution_662keV)
    ene_threshold_max = compute_energy_threshold_max(args.energy_mev)
    apply_energy_smear = not args.disable_energy_smear

    per_seed_summaries = []
    per_file_metrics_first_seed: list[FileMetrics] | None = None
    for repeat_idx in range(args.smear_repeats):
        seed = args.seed + repeat_idx
        summary, per_file_metrics = evaluate_single_seed(
            detector_pos=detector_pos,
            layer_by_det=layer_by_det,
            records=records,
            cpnum1_raw=cpnum1_raw,
            cpnum2_raw=cpnum2_raw,
            e1_raw=e1_raw,
            e2_raw=e2_raw,
            truth_front_first_raw=truth_front_first_raw,
            e0=args.energy_mev,
            front_prior_ratio=args.front_prior_ratio,
            geometry_sigma_deg=args.geometry_sigma_deg,
            geometry_power=args.geometry_power,
            ene_resolution=ene_resolution,
            ene_threshold_max=ene_threshold_max,
            ene_threshold_min=args.ene_threshold_min,
            ene_threshold_sum=args.ene_threshold_sum,
            apply_energy_smear=apply_energy_smear,
            seed=seed,
            reject_fractions=args.reject_frac_list,
        )
        per_seed_summaries.append(summary)
        if per_file_metrics_first_seed is None:
            per_file_metrics_first_seed = per_file_metrics

    aggregate = aggregate_summaries(per_seed_summaries)
    config = {
        "list_dir": str(list_dir),
        "detector_csv": str(detector_csv),
        "energy_mev": args.energy_mev,
        "front_prior_ratio": args.front_prior_ratio,
        "geometry_sigma_deg": args.geometry_sigma_deg,
        "geometry_power": args.geometry_power,
        "apply_energy_smear": apply_energy_smear,
        "smear_repeats": args.smear_repeats,
        "base_seed": args.seed,
        "ene_resolution_662keV": args.ene_resolution_662keV,
        "ene_resolution": ene_resolution,
        "ene_threshold_max": ene_threshold_max,
        "ene_threshold_min": args.ene_threshold_min,
        "ene_threshold_sum": args.ene_threshold_sum,
        "reject_frac_list": args.reject_frac_list,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "aggregate": aggregate,
                "per_seed": per_seed_summaries,
            },
            f,
            indent=2,
        )

    write_report(output_dir / "report.txt", config, aggregate, per_seed_summaries)
    if per_file_metrics_first_seed is not None:
        write_per_file_csv(output_dir / "per_file_metrics_seed0.csv", per_file_metrics_first_seed)

    print(
        json.dumps(
            {
                "config": config,
                "aggregate": aggregate,
            },
            indent=2,
        )
    )
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
