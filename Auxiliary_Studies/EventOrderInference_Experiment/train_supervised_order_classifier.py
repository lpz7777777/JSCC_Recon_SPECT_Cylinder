from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EventOrderInference_Experiment.infer_event_order import (
    center_geometry_weight,
    compton_theta_from_first_deposit,
    compute_energy_resolution,
    compute_energy_threshold_max,
    filter_events_like_recon,
    klein_nishina_weight_from_first_deposit,
    load_detector,
    load_raw_cross_layer_events,
    smear_energies_like_recon,
)


EPS = 1.0e-12


@dataclass
class RawEvents:
    records: list[dict]
    cpnum1: np.ndarray
    cpnum2: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    truth_front_first: np.ndarray


@dataclass
class PreparedDataset:
    features: np.ndarray
    labels: np.ndarray
    baseline_pred: np.ndarray
    baseline_confidence: np.ndarray
    filtered_count: int
    truth_front_first_ratio: float
    baseline_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised classifier for two-hit event-order inference."
    )
    parser.add_argument("--train-list-dir", type=Path, required=True, help="Training list directory.")
    parser.add_argument("--test-list-dir", type=Path, required=True, help="Independent test list directory.")
    parser.add_argument("--detector-csv", type=Path, required=True, help="Detector.csv path.")
    parser.add_argument("--energy-mev", type=float, default=0.511, help="Incident photon energy in MeV.")
    parser.add_argument(
        "--front-prior-ratio",
        type=float,
        default=1.2,
        help="Same center_geometry prior ratio used for the baseline score.",
    )
    parser.add_argument(
        "--geometry-sigma-deg",
        type=float,
        default=20.0,
        help="Same center_geometry angular sigma used for the baseline score.",
    )
    parser.add_argument(
        "--geometry-power",
        type=float,
        default=1.0,
        help="Same center_geometry exponent used for the baseline score.",
    )
    parser.add_argument(
        "--ene-resolution-662keV",
        type=float,
        default=0.1,
        help="Reference energy resolution, matched to the reconstruction scripts.",
    )
    parser.add_argument(
        "--ene-threshold-min",
        type=float,
        default=0.05,
        help="Lower threshold for each interaction after smearing.",
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
        help="Disable energy smearing. Mainly useful for debugging.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation file fraction split from the training directory.",
    )
    parser.add_argument(
        "--train-smear-repeats",
        type=int,
        default=3,
        help="Number of independently smeared copies used to build the training set.",
    )
    parser.add_argument(
        "--test-smear-repeats",
        type=int,
        default=5,
        help="Number of independently smeared test repeats used for reporting.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer widths.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used by the MLP.")
    parser.add_argument("--seed", type=int, default=20260418, help="Base random seed.")
    parser.add_argument(
        "--reject-frac-list",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
        help="Fractions of lowest-confidence events to reject when reporting retained accuracy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to EventOrderInference_Experiment/supervised_results",
    )
    return parser.parse_args()


def parse_hidden_dims(text: str) -> list[int]:
    dims = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not dims:
        raise ValueError("--hidden-dims must specify at least one positive integer.")
    if any(dim <= 0 for dim in dims):
        raise ValueError("--hidden-dims entries must be positive integers.")
    return dims


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    labels64 = labels.astype(np.float64, copy=False)
    probs64 = probs.astype(np.float64, copy=False)
    probs_clip = np.clip(probs64, 1.0e-7, 1.0 - 1.0e-7)
    return float(
        -np.mean(labels64 * np.log(probs_clip) + (1.0 - labels64) * np.log(1.0 - probs_clip))
    )


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def average_true_class_probability(labels_bool: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean(np.where(labels_bool, probs, 1.0 - probs)))


def rejection_sweep(
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
        retained_accuracy = (
            retained_correct_count / retained_count if retained_count > 0 else float("nan")
        )
        confidence_threshold = float(confidence[order[reject_count - 1]]) if reject_count > 0 else 0.0
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


def aggregate_rejection_sweeps(per_repeat_metrics: list[dict], key: str) -> list[dict]:
    grouped: dict[float, list[dict]] = {}
    for item in per_repeat_metrics:
        for reject_item in item.get(key, []):
            fraction = float(reject_item["reject_fraction_target"])
            grouped.setdefault(fraction, []).append(reject_item)

    aggregate = []
    for fraction in sorted(grouped):
        entries = grouped[fraction]
        reject_fraction_actual = np.array([x["reject_fraction_actual"] for x in entries], dtype=np.float64)
        reject_count = np.array([x["reject_count"] for x in entries], dtype=np.float64)
        retained_count = np.array([x["retained_count"] for x in entries], dtype=np.float64)
        retained_accuracy = np.array([x["retained_accuracy"] for x in entries], dtype=np.float64)
        aggregate.append(
            {
                "reject_fraction_target": fraction,
                "reject_fraction_actual": {
                    "mean": float(np.mean(reject_fraction_actual)),
                    "std": float(np.std(reject_fraction_actual)),
                },
                "reject_count": {
                    "mean": float(np.mean(reject_count)),
                    "std": float(np.std(reject_count)),
                },
                "retained_count": {
                    "mean": float(np.mean(retained_count)),
                    "std": float(np.std(retained_count)),
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


def choose_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_file_event_mask(records: list[dict], selected_record_indices: list[int]) -> np.ndarray:
    total_count = max((record["end"] for record in records), default=0)
    mask = np.zeros(total_count, dtype=bool)
    for idx in selected_record_indices:
        start = records[idx]["start"]
        end = records[idx]["end"]
        if end > start:
            mask[start:end] = True
    return mask


def split_train_val_masks(records: list[dict], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    valid_indices = [idx for idx, record in enumerate(records) if record["raw_cross_layer_count"] > 0]
    if not valid_indices:
        raise RuntimeError("No non-empty files available for train/validation split.")

    rng = np.random.default_rng(seed)
    shuffled = valid_indices.copy()
    rng.shuffle(shuffled)

    val_file_count = int(round(len(shuffled) * val_fraction))
    val_file_count = min(max(val_file_count, 1), len(shuffled) - 1)
    val_indices = sorted(shuffled[:val_file_count])
    train_indices = sorted(shuffled[val_file_count:])

    train_mask = build_file_event_mask(records, train_indices)
    val_mask = build_file_event_mask(records, val_indices)
    train_names = [records[idx]["file_name"] for idx in train_indices]
    val_names = [records[idx]["file_name"] for idx in val_indices]
    return train_mask, val_mask, train_names, val_names


def slice_raw_events(raw: RawEvents, event_mask: np.ndarray) -> RawEvents:
    return RawEvents(
        records=[],
        cpnum1=raw.cpnum1[event_mask],
        cpnum2=raw.cpnum2[event_mask],
        e1=raw.e1[event_mask],
        e2=raw.e2[event_mask],
        truth_front_first=raw.truth_front_first[event_mask],
    )


def build_feature_matrix(
    detector_pos: np.ndarray,
    layer_by_det: np.ndarray,
    cpnum1: np.ndarray,
    cpnum2: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    truth_front_first: np.ndarray,
    e0: float,
    front_prior_ratio: float,
    geometry_sigma_deg: float,
    geometry_power: float,
) -> PreparedDataset:
    front_layer_mask = layer_by_det <= (layer_by_det.max() - 1)
    rear_layer_mask = layer_by_det == layer_by_det.max()

    front_det = np.where(front_layer_mask[cpnum1], cpnum1, cpnum2)
    rear_det = np.where(rear_layer_mask[cpnum1], cpnum1, cpnum2)
    front_energy = np.where(front_layer_mask[cpnum1], e1, e2)
    rear_energy = np.where(rear_layer_mask[cpnum1], e1, e2)

    front_theta = compton_theta_from_first_deposit(front_energy, e0)
    rear_theta = compton_theta_from_first_deposit(rear_energy, e0)
    front_theta_valid = np.isfinite(front_theta).astype(np.float64)
    rear_theta_valid = np.isfinite(rear_theta).astype(np.float64)
    front_theta = np.nan_to_num(front_theta, nan=0.0, posinf=0.0, neginf=0.0)
    rear_theta = np.nan_to_num(rear_theta, nan=0.0, posinf=0.0, neginf=0.0)

    front_kn = front_prior_ratio * klein_nishina_weight_from_first_deposit(front_energy, e0)
    rear_kn = klein_nishina_weight_from_first_deposit(rear_energy, e0)
    front_center = center_geometry_weight(detector_pos, front_det, rear_det, front_theta, geometry_sigma_deg)
    rear_center = center_geometry_weight(detector_pos, rear_det, front_det, rear_theta, geometry_sigma_deg)
    front_score = front_kn * np.power(front_center, geometry_power)
    rear_score = rear_kn * np.power(rear_center, geometry_power)
    baseline_pred = front_score >= rear_score
    baseline_logit = np.log(front_score + EPS) - np.log(rear_score + EPS)
    baseline_confidence = np.abs(baseline_logit)

    front_pos = detector_pos[front_det]
    rear_pos = detector_pos[rear_det]
    delta = rear_pos - front_pos
    line_length = np.linalg.norm(delta, axis=1)
    front_radius_xz = np.linalg.norm(front_pos[:, [0, 2]], axis=1)
    rear_radius_xz = np.linalg.norm(rear_pos[:, [0, 2]], axis=1)
    energy_sum = front_energy + rear_energy
    energy_diff = front_energy - rear_energy
    front_frac = front_energy / np.maximum(energy_sum, EPS)
    rear_frac = rear_energy / np.maximum(energy_sum, EPS)

    features = np.column_stack(
        [
            front_energy,
            rear_energy,
            energy_sum,
            energy_diff,
            np.abs(energy_diff),
            front_frac,
            rear_frac,
            front_theta,
            rear_theta,
            front_theta_valid,
            rear_theta_valid,
            front_kn,
            rear_kn,
            front_center,
            rear_center,
            np.log(front_kn + EPS) - np.log(rear_kn + EPS),
            np.log(front_center + EPS) - np.log(rear_center + EPS),
            np.log(front_score + EPS) - np.log(rear_score + EPS),
            front_pos[:, 0],
            front_pos[:, 2],
            rear_pos[:, 0],
            rear_pos[:, 2],
            delta[:, 0],
            delta[:, 1],
            delta[:, 2],
            line_length,
            front_radius_xz,
            rear_radius_xz,
        ]
    ).astype(np.float32, copy=False)

    labels = truth_front_first.astype(np.float32, copy=False)
    baseline_accuracy = float(np.mean(baseline_pred == truth_front_first)) if labels.size > 0 else float("nan")
    truth_front_first_ratio = float(np.mean(truth_front_first)) if labels.size > 0 else float("nan")
    return PreparedDataset(
        features=features,
        labels=labels,
        baseline_pred=baseline_pred.astype(bool, copy=False),
        baseline_confidence=baseline_confidence.astype(np.float32, copy=False),
        filtered_count=int(labels.size),
        truth_front_first_ratio=truth_front_first_ratio,
        baseline_accuracy=baseline_accuracy,
    )


def prepare_dataset(
    raw: RawEvents,
    detector_pos: np.ndarray,
    layer_by_det: np.ndarray,
    e0: float,
    ene_resolution: float,
    ene_threshold_max: float,
    ene_threshold_min: float,
    ene_threshold_sum: float,
    front_prior_ratio: float,
    geometry_sigma_deg: float,
    geometry_power: float,
    apply_energy_smear: bool,
    seed: int,
) -> PreparedDataset:
    if apply_energy_smear:
        rng = np.random.default_rng(seed)
        e1_obs, e2_obs = smear_energies_like_recon(raw.e1, raw.e2, e0, ene_resolution, rng)
    else:
        e1_obs = raw.e1.copy()
        e2_obs = raw.e2.copy()

    valid = filter_events_like_recon(
        cpnum1=raw.cpnum1,
        cpnum2=raw.cpnum2,
        e1=e1_obs,
        e2=e2_obs,
        e0=e0,
        ene_threshold_max=ene_threshold_max,
        ene_threshold_min=ene_threshold_min,
        ene_threshold_sum=ene_threshold_sum,
    )
    return build_feature_matrix(
        detector_pos=detector_pos,
        layer_by_det=layer_by_det,
        cpnum1=raw.cpnum1[valid],
        cpnum2=raw.cpnum2[valid],
        e1=e1_obs[valid],
        e2=e2_obs[valid],
        truth_front_first=raw.truth_front_first[valid],
        e0=e0,
        front_prior_ratio=front_prior_ratio,
        geometry_sigma_deg=geometry_sigma_deg,
        geometry_power=geometry_power,
    )


def concat_prepared(datasets: list[PreparedDataset]) -> PreparedDataset:
    return PreparedDataset(
        features=np.concatenate([item.features for item in datasets], axis=0),
        labels=np.concatenate([item.labels for item in datasets], axis=0),
        baseline_pred=np.concatenate([item.baseline_pred for item in datasets], axis=0),
        baseline_confidence=np.concatenate([item.baseline_confidence for item in datasets], axis=0),
        filtered_count=int(sum(item.filtered_count for item in datasets)),
        truth_front_first_ratio=float(
            np.mean(np.concatenate([item.labels for item in datasets], axis=0))
        ),
        baseline_accuracy=float(
            np.mean(
                np.concatenate([item.baseline_pred for item in datasets], axis=0)
                == np.concatenate([item.labels for item in datasets], axis=0).astype(bool)
            )
        ),
    )


def standardize_train_val_test(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = train_x.std(axis=0, dtype=np.float64)
    std[std < 1.0e-6] = 1.0

    train_x_std = ((train_x - mean) / std).astype(np.float32, copy=False)
    val_x_std = ((val_x - mean) / std).astype(np.float32, copy=False)
    test_x_std = [((x - mean) / std).astype(np.float32, copy=False) for x in test_x_list]
    return train_x_std, val_x_std, test_x_std, mean.astype(np.float32), std.astype(np.float32)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def evaluate_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            pred = logits >= 0.0
            total_loss += loss.item() * labels.size(0)
            total_correct += int((pred == (labels >= 0.5)).sum().item())
            total_count += labels.size(0)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(features)
    tensor_y = torch.from_numpy(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    if not args.train_list_dir.is_dir():
        raise FileNotFoundError(f"Training list directory not found: {args.train_list_dir}")
    if not args.test_list_dir.is_dir():
        raise FileNotFoundError(f"Test list directory not found: {args.test_list_dir}")
    if not args.detector_csv.is_file():
        raise FileNotFoundError(f"Detector CSV not found: {args.detector_csv}")
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must satisfy 0 < val_fraction < 1.")
    if args.train_smear_repeats <= 0 or args.test_smear_repeats <= 0:
        raise ValueError("--train-smear-repeats and --test-smear-repeats must be positive.")
    if any((x < 0.0) or (x >= 1.0) for x in args.reject_frac_list):
        raise ValueError("Each value in --reject-frac-list must satisfy 0 <= x < 1.")

    output_dir = args.output_dir.resolve() if args.output_dir else (
        Path(__file__).resolve().parent / "supervised_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    detector_pos, layer_by_det = load_detector(args.detector_csv.resolve())
    train_records, train_cpnum1, train_cpnum2, train_e1, train_e2, train_truth = load_raw_cross_layer_events(
        args.train_list_dir.resolve(),
        detector_pos,
        layer_by_det,
    )
    test_records, test_cpnum1, test_cpnum2, test_e1, test_e2, test_truth = load_raw_cross_layer_events(
        args.test_list_dir.resolve(),
        detector_pos,
        layer_by_det,
    )

    raw_train = RawEvents(train_records, train_cpnum1, train_cpnum2, train_e1, train_e2, train_truth)
    raw_test = RawEvents(test_records, test_cpnum1, test_cpnum2, test_e1, test_e2, test_truth)

    train_mask, val_mask, train_files, val_files = split_train_val_masks(train_records, args.val_fraction, args.seed)
    raw_train_split = slice_raw_events(raw_train, train_mask)
    raw_val_split = slice_raw_events(raw_train, val_mask)

    ene_resolution = compute_energy_resolution(args.energy_mev, args.ene_resolution_662keV)
    ene_threshold_max = compute_energy_threshold_max(args.energy_mev)
    apply_energy_smear = not args.disable_energy_smear

    train_prepared_parts = []
    for repeat_idx in range(args.train_smear_repeats):
        train_prepared_parts.append(
            prepare_dataset(
                raw=raw_train_split,
                detector_pos=detector_pos,
                layer_by_det=layer_by_det,
                e0=args.energy_mev,
                ene_resolution=ene_resolution,
                ene_threshold_max=ene_threshold_max,
                ene_threshold_min=args.ene_threshold_min,
                ene_threshold_sum=args.ene_threshold_sum,
                front_prior_ratio=args.front_prior_ratio,
                geometry_sigma_deg=args.geometry_sigma_deg,
                geometry_power=args.geometry_power,
                apply_energy_smear=apply_energy_smear,
                seed=args.seed + 100 * repeat_idx,
            )
        )
    train_prepared = concat_prepared(train_prepared_parts)

    val_prepared = prepare_dataset(
        raw=raw_val_split,
        detector_pos=detector_pos,
        layer_by_det=layer_by_det,
        e0=args.energy_mev,
        ene_resolution=ene_resolution,
        ene_threshold_max=ene_threshold_max,
        ene_threshold_min=args.ene_threshold_min,
        ene_threshold_sum=args.ene_threshold_sum,
        front_prior_ratio=args.front_prior_ratio,
        geometry_sigma_deg=args.geometry_sigma_deg,
        geometry_power=args.geometry_power,
        apply_energy_smear=apply_energy_smear,
        seed=args.seed + 5000,
    )

    test_prepared_list = []
    for repeat_idx in range(args.test_smear_repeats):
        test_prepared_list.append(
            prepare_dataset(
                raw=raw_test,
                detector_pos=detector_pos,
                layer_by_det=layer_by_det,
                e0=args.energy_mev,
                ene_resolution=ene_resolution,
                ene_threshold_max=ene_threshold_max,
                ene_threshold_min=args.ene_threshold_min,
                ene_threshold_sum=args.ene_threshold_sum,
                front_prior_ratio=args.front_prior_ratio,
                geometry_sigma_deg=args.geometry_sigma_deg,
                geometry_power=args.geometry_power,
                apply_energy_smear=apply_energy_smear,
                seed=args.seed + 10000 + repeat_idx,
            )
        )

    train_x, val_x, test_x_list, feature_mean, feature_std = standardize_train_val_test(
        train_prepared.features,
        val_prepared.features,
        [item.features for item in test_prepared_list],
    )
    train_y = train_prepared.labels.astype(np.float32, copy=False)
    val_y = val_prepared.labels.astype(np.float32, copy=False)
    test_y_list = [item.labels.astype(np.float32, copy=False) for item in test_prepared_list]

    train_loader = build_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = build_loader(val_x, val_y, args.batch_size, shuffle=False)
    test_loaders = [
        build_loader(test_x, test_y, args.batch_size, shuffle=False)
        for test_x, test_y in zip(test_x_list, test_y_list)
    ]

    model = MLPClassifier(train_x.shape[1], hidden_dims, args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_val_accuracy = -math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        total_correct = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            pred = logits >= 0.0
            total_loss += loss.item() * labels.size(0)
            total_correct += int((pred == (labels >= 0.5)).sum().item())
            total_count += labels.size(0)

        train_loss = total_loss / max(total_count, 1)
        train_accuracy = total_correct / max(total_count, 1)
        val_loss, val_accuracy = evaluate_logits(model, val_loader, device, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)

    val_logits = predict_logits(model, val_loader, device)
    val_probs = sigmoid_np(val_logits)
    val_labels = val_prepared.labels
    val_labels_bool = val_labels >= 0.5
    val_pred = val_logits >= 0.0

    test_repeat_metrics = []
    for repeat_idx, (prepared, loader) in enumerate(zip(test_prepared_list, test_loaders)):
        logits = predict_logits(model, loader, device)
        probs = sigmoid_np(logits)
        pred = logits >= 0.0
        labels_bool = prepared.labels >= 0.5
        classifier_accuracy = float(np.mean(pred == labels_bool))
        baseline_accuracy = float(prepared.baseline_accuracy)
        classifier_correct = pred == labels_bool
        baseline_correct = prepared.baseline_pred == labels_bool
        classifier_rejection = rejection_sweep(np.abs(logits), classifier_correct, args.reject_frac_list)
        baseline_rejection = rejection_sweep(
            prepared.baseline_confidence.astype(np.float64),
            baseline_correct,
            args.reject_frac_list,
        )
        test_repeat_metrics.append(
            {
                "repeat_index": repeat_idx,
                "filtered_count": prepared.filtered_count,
                "truth_front_first_ratio": prepared.truth_front_first_ratio,
                "baseline_center_geometry_accuracy": baseline_accuracy,
                "baseline_rejection_sweep": baseline_rejection,
                "classifier_accuracy": classifier_accuracy,
                "classifier_brier_score": brier_score(prepared.labels, probs),
                "classifier_log_loss": binary_log_loss(prepared.labels, probs),
                "classifier_avg_true_class_prob": average_true_class_probability(labels_bool, probs),
                "classifier_rejection_sweep": classifier_rejection,
                "accuracy_gain": classifier_accuracy - baseline_accuracy,
            }
        )

    baseline_values = np.array([item["baseline_center_geometry_accuracy"] for item in test_repeat_metrics], dtype=np.float64)
    classifier_values = np.array([item["classifier_accuracy"] for item in test_repeat_metrics], dtype=np.float64)
    classifier_brier_values = np.array([item["classifier_brier_score"] for item in test_repeat_metrics], dtype=np.float64)
    classifier_log_loss_values = np.array([item["classifier_log_loss"] for item in test_repeat_metrics], dtype=np.float64)
    classifier_true_prob_values = np.array([item["classifier_avg_true_class_prob"] for item in test_repeat_metrics], dtype=np.float64)
    gain_values = classifier_values - baseline_values
    filtered_values = np.array([item["filtered_count"] for item in test_repeat_metrics], dtype=np.float64)
    baseline_rejection_aggregate = aggregate_rejection_sweeps(test_repeat_metrics, "baseline_rejection_sweep")
    classifier_rejection_aggregate = aggregate_rejection_sweeps(test_repeat_metrics, "classifier_rejection_sweep")

    summary = {
        "config": {
            "train_list_dir": str(args.train_list_dir.resolve()),
            "test_list_dir": str(args.test_list_dir.resolve()),
            "detector_csv": str(args.detector_csv.resolve()),
            "energy_mev": args.energy_mev,
            "front_prior_ratio": args.front_prior_ratio,
            "geometry_sigma_deg": args.geometry_sigma_deg,
            "geometry_power": args.geometry_power,
            "ene_resolution_662keV": args.ene_resolution_662keV,
            "ene_resolution": ene_resolution,
            "ene_threshold_max": ene_threshold_max,
            "ene_threshold_min": args.ene_threshold_min,
            "ene_threshold_sum": args.ene_threshold_sum,
            "apply_energy_smear": apply_energy_smear,
            "train_smear_repeats": args.train_smear_repeats,
            "test_smear_repeats": args.test_smear_repeats,
            "reject_frac_list": args.reject_frac_list,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "device": str(device),
            "seed": args.seed,
        },
        "split": {
            "train_files": train_files,
            "val_files": val_files,
            "train_raw_cross_layer_count": int(raw_train_split.cpnum1.size),
            "val_raw_cross_layer_count": int(raw_val_split.cpnum1.size),
            "test_raw_cross_layer_count": int(raw_test.cpnum1.size),
            "train_filtered_count_total": int(train_prepared.filtered_count),
            "val_filtered_count": int(val_prepared.filtered_count),
        },
        "training_history": history,
        "validation": {
            "baseline_center_geometry_accuracy": val_prepared.baseline_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "classifier_accuracy": float(np.mean(val_pred == val_labels_bool)),
            "classifier_brier_score": brier_score(val_labels, val_probs),
            "classifier_log_loss": binary_log_loss(val_labels, val_probs),
            "classifier_avg_true_class_prob": average_true_class_probability(val_labels_bool, val_probs),
            "baseline_rejection_sweep": rejection_sweep(
                val_prepared.baseline_confidence.astype(np.float64),
                val_prepared.baseline_pred == val_labels_bool,
                args.reject_frac_list,
            ),
            "classifier_rejection_sweep": rejection_sweep(
                np.abs(val_logits),
                val_pred == val_labels_bool,
                args.reject_frac_list,
            ),
        },
        "test_aggregate": {
            "filtered_count": {
                "mean": float(np.mean(filtered_values)),
                "std": float(np.std(filtered_values)),
            },
            "baseline_center_geometry_accuracy": {
                "mean": float(np.mean(baseline_values)),
                "std": float(np.std(baseline_values)),
            },
            "classifier_accuracy": {
                "mean": float(np.mean(classifier_values)),
                "std": float(np.std(classifier_values)),
            },
            "classifier_brier_score": {
                "mean": float(np.mean(classifier_brier_values)),
                "std": float(np.std(classifier_brier_values)),
            },
            "classifier_log_loss": {
                "mean": float(np.mean(classifier_log_loss_values)),
                "std": float(np.std(classifier_log_loss_values)),
            },
            "classifier_avg_true_class_prob": {
                "mean": float(np.mean(classifier_true_prob_values)),
                "std": float(np.std(classifier_true_prob_values)),
            },
            "accuracy_gain": {
                "mean": float(np.mean(gain_values)),
                "std": float(np.std(gain_values)),
                "min": float(np.min(gain_values)),
                "max": float(np.max(gain_values)),
            },
            "baseline_rejection_sweep": baseline_rejection_aggregate,
            "classifier_rejection_sweep": classifier_rejection_aggregate,
        },
        "test_repeats": test_repeat_metrics,
        "feature_stats": {
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "config": summary["config"],
        },
        output_dir / "model.pt",
    )

    print(
        json.dumps(
            {
                "validation": summary["validation"],
                "test_aggregate": summary["test_aggregate"],
            },
            indent=2,
        )
    )
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
