from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EventOrderInference_Experiment.infer_event_order import (
    center_geometry_weight,
    compton_theta_from_first_deposit,
    filter_events_like_recon,
    klein_nishina_weight_from_first_deposit,
    load_detector,
    load_raw_cross_layer_events,
    smear_energies_like_recon,
)


EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detailed event-order inference metrics for downstream visualization."
    )
    parser.add_argument("--result-dir", type=Path, required=True, help="Experiment result directory containing summary.json.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <result-dir>/visualization_metrics.json",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins used for classifier probability calibration curves.",
    )
    return parser.parse_args()


def load_summary(result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def compute_binary_metrics(truth_front_first: np.ndarray, pred_front_first: np.ndarray) -> dict[str, float]:
    truth = truth_front_first.astype(bool, copy=False)
    pred = pred_front_first.astype(bool, copy=False)

    truth_front_count = int(np.count_nonzero(truth))
    truth_rear_count = int(truth.size - truth_front_count)
    pred_front_count = int(np.count_nonzero(pred))
    pred_rear_count = int(pred.size - pred_front_count)

    tp = int(np.count_nonzero(truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))
    fp = int(np.count_nonzero(~truth & pred))
    tn = int(np.count_nonzero(~truth & ~pred))
    correct_count = tp + tn

    front_acc = tp / truth_front_count if truth_front_count > 0 else float("nan")
    rear_acc = tn / truth_rear_count if truth_rear_count > 0 else float("nan")
    balanced_acc = 0.5 * (front_acc + rear_acc) if np.isfinite(front_acc) and np.isfinite(rear_acc) else float("nan")

    return {
        "event_count": int(truth.size),
        "truth_front_first_count": truth_front_count,
        "truth_rear_first_count": truth_rear_count,
        "pred_front_first_count": pred_front_count,
        "pred_rear_first_count": pred_rear_count,
        "correct_count": correct_count,
        "correct_front_first_count": tp,
        "correct_rear_first_count": tn,
        "front_as_front_count": tp,
        "front_as_rear_count": fn,
        "rear_as_front_count": fp,
        "rear_as_rear_count": tn,
        "accuracy": correct_count / truth.size if truth.size > 0 else float("nan"),
        "front_first_accuracy": front_acc,
        "rear_first_accuracy": rear_acc,
        "balanced_accuracy": balanced_acc,
        "truth_front_first_ratio": truth_front_count / truth.size if truth.size > 0 else float("nan"),
        "pred_front_first_ratio": pred_front_count / truth.size if truth.size > 0 else float("nan"),
    }


def rejection_sweep_with_class_metrics(
    confidence: np.ndarray,
    truth_front_first: np.ndarray,
    pred_front_first: np.ndarray,
    reject_fractions: list[float],
) -> list[dict[str, float]]:
    total_count = int(confidence.size)
    if total_count == 0:
        return []

    order = np.argsort(confidence, kind="stable")
    sweeps = []
    for reject_fraction in reject_fractions:
        reject_count = int(math.floor(reject_fraction * total_count))
        reject_count = min(max(reject_count, 0), total_count)

        retain_mask = np.ones(total_count, dtype=bool)
        if reject_count > 0:
            retain_mask[order[:reject_count]] = False

        retained_truth = truth_front_first[retain_mask]
        retained_pred = pred_front_first[retain_mask]
        metrics = compute_binary_metrics(retained_truth, retained_pred)
        metrics.update(
            {
                "reject_fraction_target": float(reject_fraction),
                "reject_fraction_actual": reject_count / total_count,
                "reject_count": reject_count,
                "retained_count": metrics.pop("event_count"),
                "retained_fraction": np.count_nonzero(retain_mask) / total_count,
                "confidence_threshold": float(confidence[order[reject_count - 1]]) if reject_count > 0 else 0.0,
            }
        )
        sweeps.append(metrics)
    return sweeps


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = metric_dicts[0].keys()
    return {key: summarize_values([item[key] for item in metric_dicts]) for key in keys}


def aggregate_rejection_sweeps(per_repeat_sweeps: list[list[dict[str, float]]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, float]]] = {}
    for sweep in per_repeat_sweeps:
        for item in sweep:
            grouped.setdefault(float(item["reject_fraction_target"]), []).append(item)

    aggregate = []
    for reject_fraction in sorted(grouped):
        entries = grouped[reject_fraction]
        aggregate_entry: dict[str, Any] = {"reject_fraction_target": reject_fraction}
        for key in entries[0].keys():
            if key == "reject_fraction_target":
                continue
            aggregate_entry[key] = summarize_values([item[key] for item in entries])
        aggregate.append(aggregate_entry)
    return aggregate


def empty_soft_metrics() -> dict[str, dict[str, float]]:
    nan_stat = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "classifier_brier_score": dict(nan_stat),
        "classifier_log_loss": dict(nan_stat),
        "classifier_avg_true_class_prob": dict(nan_stat),
    }


def empty_soft_repeat_metrics() -> dict[str, float]:
    return {
        "classifier_brier_score": float("nan"),
        "classifier_log_loss": float("nan"),
        "classifier_avg_true_class_prob": float("nan"),
    }


def compute_calibration_bins(labels: np.ndarray, probs: np.ndarray, num_bins: int) -> list[dict[str, float]]:
    labels_arr = labels.astype(np.float64, copy=False)
    probs_arr = probs.astype(np.float64, copy=False)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    for bin_idx in range(num_bins):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        if bin_idx == num_bins - 1:
            mask = (probs_arr >= left) & (probs_arr <= right)
        else:
            mask = (probs_arr >= left) & (probs_arr < right)
        count = int(np.count_nonzero(mask))
        if count > 0:
            mean_prob = float(np.mean(probs_arr[mask]))
            empirical_front_ratio = float(np.mean(labels_arr[mask]))
        else:
            mean_prob = float("nan")
            empirical_front_ratio = float("nan")
        bins.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "sample_count": count,
                "mean_pred_prob": mean_prob,
                "empirical_front_first_ratio": empirical_front_ratio,
            }
        )
    return bins


def calibration_bins_from_repeats(labels_list: list[np.ndarray], probs_list: list[np.ndarray], num_bins: int) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    sum_count = np.zeros(num_bins, dtype=np.int64)
    sum_prob = np.zeros(num_bins, dtype=np.float64)
    sum_truth = np.zeros(num_bins, dtype=np.float64)

    for labels, probs in zip(labels_list, probs_list):
        labels_arr = labels.astype(np.float64, copy=False)
        probs_arr = probs.astype(np.float64, copy=False)
        for bin_idx in range(num_bins):
            left = edges[bin_idx]
            right = edges[bin_idx + 1]
            if bin_idx == num_bins - 1:
                mask = (probs_arr >= left) & (probs_arr <= right)
            else:
                mask = (probs_arr >= left) & (probs_arr < right)
            count = int(np.count_nonzero(mask))
            if count > 0:
                sum_count[bin_idx] += count
                sum_prob[bin_idx] += float(np.sum(probs_arr[mask]))
                sum_truth[bin_idx] += float(np.sum(labels_arr[mask]))

    bins = []
    for bin_idx in range(num_bins):
        count = int(sum_count[bin_idx])
        bins.append(
            {
                "bin_left": float(edges[bin_idx]),
                "bin_right": float(edges[bin_idx + 1]),
                "sample_count": count,
                "mean_pred_prob": float(sum_prob[bin_idx] / count) if count > 0 else float("nan"),
                "empirical_front_first_ratio": float(sum_truth[bin_idx] / count) if count > 0 else float("nan"),
            }
        )
    return bins


def binary_log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    labels64 = labels.astype(np.float64, copy=False)
    probs64 = probs.astype(np.float64, copy=False)
    probs_clip = np.clip(probs64, 1.0e-7, 1.0 - 1.0e-7)
    return float(-np.mean(labels64 * np.log(probs_clip) + (1.0 - labels64) * np.log(1.0 - probs_clip)))


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    labels64 = labels.astype(np.float64, copy=False)
    probs64 = probs.astype(np.float64, copy=False)
    return float(np.mean((probs64 - labels64) ** 2))


def average_true_class_probability(labels: np.ndarray, probs: np.ndarray) -> float:
    labels_bool = labels.astype(bool, copy=False)
    probs64 = probs.astype(np.float64, copy=False)
    return float(np.mean(np.where(labels_bool, probs64, 1.0 - probs64)))


def analyze_center_geometry(summary: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    config = summary["config"]
    detector_pos, layer_by_det = load_detector(Path(config["detector_csv"]))
    records, cpnum1_raw, cpnum2_raw, e1_raw, e2_raw, truth_front_first_raw = load_raw_cross_layer_events(
        Path(config["list_dir"]),
        detector_pos,
        layer_by_det,
    )
    del records

    repeat_count = int(summary["aggregate"]["repeat_count"])
    base_seed = int(config["base_seed"])
    apply_energy_smear = bool(config["apply_energy_smear"])
    reject_fractions = [float(x) for x in config.get("reject_frac_list", [0.0])]

    per_repeat_metrics = []
    per_repeat_sweeps = []
    for repeat_idx in range(repeat_count):
        seed = base_seed + repeat_idx
        if apply_energy_smear:
            rng = np.random.default_rng(seed)
            e1_obs, e2_obs = smear_energies_like_recon(
                e1_raw,
                e2_raw,
                float(config["energy_mev"]),
                float(config["ene_resolution"]),
                rng,
            )
        else:
            e1_obs = e1_raw.copy()
            e2_obs = e2_raw.copy()

        valid = filter_events_like_recon(
            cpnum1=cpnum1_raw,
            cpnum2=cpnum2_raw,
            e1=e1_obs,
            e2=e2_obs,
            e0=float(config["energy_mev"]),
            ene_threshold_max=float(config["ene_threshold_max"]),
            ene_threshold_min=float(config["ene_threshold_min"]),
            ene_threshold_sum=float(config["ene_threshold_sum"]),
        )

        cpnum1 = cpnum1_raw[valid]
        cpnum2 = cpnum2_raw[valid]
        e1 = e1_obs[valid]
        e2 = e2_obs[valid]
        truth_front_first = truth_front_first_raw[valid].astype(bool, copy=False)

        front_mask = layer_by_det <= (layer_by_det.max() - 1)
        rear_mask = layer_by_det == layer_by_det.max()
        front_first_det = np.where(front_mask[cpnum1], cpnum1, cpnum2)
        rear_first_det = np.where(rear_mask[cpnum1], cpnum1, cpnum2)
        front_second_det = np.where(front_mask[cpnum1], cpnum2, cpnum1)
        rear_second_det = np.where(rear_mask[cpnum1], cpnum2, cpnum1)
        front_energy = np.where(front_mask[cpnum1], e1, e2)
        rear_energy = np.where(rear_mask[cpnum1], e1, e2)

        front_theta = compton_theta_from_first_deposit(front_energy, float(config["energy_mev"]))
        rear_theta = compton_theta_from_first_deposit(rear_energy, float(config["energy_mev"]))
        front_kn = float(config["front_prior_ratio"]) * klein_nishina_weight_from_first_deposit(front_energy, float(config["energy_mev"]))
        rear_kn = klein_nishina_weight_from_first_deposit(rear_energy, float(config["energy_mev"]))
        front_center = center_geometry_weight(detector_pos, front_first_det, front_second_det, front_theta, float(config["geometry_sigma_deg"]))
        rear_center = center_geometry_weight(detector_pos, rear_first_det, rear_second_det, rear_theta, float(config["geometry_sigma_deg"]))
        front_score = front_kn * np.power(front_center, float(config["geometry_power"]))
        rear_score = rear_kn * np.power(rear_center, float(config["geometry_power"]))
        pred_front_first = front_score >= rear_score
        confidence = np.abs(np.log(front_score + EPS) - np.log(rear_score + EPS))

        metrics = compute_binary_metrics(truth_front_first, pred_front_first)
        metrics["repeat_index"] = repeat_idx
        metrics["seed"] = seed
        sweep = rejection_sweep_with_class_metrics(confidence, truth_front_first, pred_front_first, reject_fractions)
        per_repeat_metrics.append(metrics)
        per_repeat_sweeps.append(sweep)

    method = {
        "name": "center_geometry",
        "display_name": f"{result_dir.name} / center_geometry",
        "overall": aggregate_metric_dicts(per_repeat_metrics),
        "rejection_sweep": aggregate_rejection_sweeps(per_repeat_sweeps),
        "soft_metrics": empty_soft_metrics(),
        "calibration_bins": [],
        "per_repeat": [
            {
                **metric,
                **empty_soft_repeat_metrics(),
                "calibration_bins": [],
                "rejection_sweep": sweep,
            }
            for metric, sweep in zip(per_repeat_metrics, per_repeat_sweeps)
        ],
    }

    return {
        "analysis_type": "center_geometry",
        "analysis_label": result_dir.name,
        "source_summary": str((result_dir / "summary.json").resolve()),
        "config": config,
        "methods": [method],
    }


def analyze_supervised(summary: dict[str, Any], result_dir: Path, calibration_bins: int) -> dict[str, Any]:
    import torch

    from EventOrderInference_Experiment.train_supervised_order_classifier import (
        MLPClassifier,
        PreparedDataset,
        RawEvents,
        build_loader,
        choose_device,
        load_detector as train_load_detector,  # same implementation
        load_raw_cross_layer_events as train_load_raw_cross_layer_events,  # same implementation
        predict_logits,
        prepare_dataset,
    )

    config = summary["config"]
    checkpoint = torch.load(result_dir / "model.pt", map_location="cpu")

    detector_pos, layer_by_det = train_load_detector(Path(config["detector_csv"]))
    test_records, test_cpnum1, test_cpnum2, test_e1, test_e2, test_truth = train_load_raw_cross_layer_events(
        Path(config["test_list_dir"]),
        detector_pos,
        layer_by_det,
    )
    del test_records

    raw_test = RawEvents(records=[], cpnum1=test_cpnum1, cpnum2=test_cpnum2, e1=test_e1, e2=test_e2, truth_front_first=test_truth)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    device = choose_device(str(config.get("device", "auto")))
    hidden_dims = [int(x) for x in config["hidden_dims"]]
    model = MLPClassifier(int(feature_mean.size), hidden_dims, float(config["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    reject_fractions = [float(x) for x in config.get("reject_frac_list", [0.0])]
    repeat_count = int(config["test_smear_repeats"])
    seed_base = int(config["seed"]) + 10000

    classifier_repeat_metrics = []
    classifier_repeat_sweeps = []
    baseline_repeat_metrics = []
    baseline_repeat_sweeps = []
    probs_list = []
    labels_list = []

    for repeat_idx in range(repeat_count):
        prepared: PreparedDataset = prepare_dataset(
            raw=raw_test,
            detector_pos=detector_pos,
            layer_by_det=layer_by_det,
            e0=float(config["energy_mev"]),
            ene_resolution=float(config["ene_resolution"]),
            ene_threshold_max=float(config["ene_threshold_max"]),
            ene_threshold_min=float(config["ene_threshold_min"]),
            ene_threshold_sum=float(config["ene_threshold_sum"]),
            front_prior_ratio=float(config["front_prior_ratio"]),
            geometry_sigma_deg=float(config["geometry_sigma_deg"]),
            geometry_power=float(config["geometry_power"]),
            apply_energy_smear=bool(config["apply_energy_smear"]),
            seed=seed_base + repeat_idx,
        )

        test_x = ((prepared.features - feature_mean) / feature_std).astype(np.float32, copy=False)
        loader = build_loader(test_x, prepared.labels.astype(np.float32, copy=False), int(config["batch_size"]), shuffle=False)
        logits = predict_logits(model, loader, device)
        probs = sigmoid_np(logits)
        truth_front_first = prepared.labels.astype(bool, copy=False)
        classifier_pred = logits >= 0.0
        classifier_confidence = np.abs(logits)
        baseline_pred = prepared.baseline_pred.astype(bool, copy=False)
        baseline_confidence = prepared.baseline_confidence.astype(np.float64, copy=False)

        classifier_metrics = compute_binary_metrics(truth_front_first, classifier_pred)
        classifier_metrics.update(
            {
                "repeat_index": repeat_idx,
                "classifier_brier_score": brier_score(prepared.labels, probs),
                "classifier_log_loss": binary_log_loss(prepared.labels, probs),
                "classifier_avg_true_class_prob": average_true_class_probability(prepared.labels, probs),
            }
        )
        baseline_metrics = compute_binary_metrics(truth_front_first, baseline_pred)
        baseline_metrics["repeat_index"] = repeat_idx

        classifier_sweep = rejection_sweep_with_class_metrics(classifier_confidence, truth_front_first, classifier_pred, reject_fractions)
        baseline_sweep = rejection_sweep_with_class_metrics(baseline_confidence, truth_front_first, baseline_pred, reject_fractions)

        classifier_repeat_metrics.append(classifier_metrics)
        baseline_repeat_metrics.append(baseline_metrics)
        classifier_repeat_sweeps.append(classifier_sweep)
        baseline_repeat_sweeps.append(baseline_sweep)
        probs_list.append(probs)
        labels_list.append(prepared.labels.astype(np.float32, copy=False))

    baseline_method = {
        "name": "center_geometry",
        "display_name": f"{result_dir.name} / center_geometry",
        "overall": aggregate_metric_dicts(baseline_repeat_metrics),
        "rejection_sweep": aggregate_rejection_sweeps(baseline_repeat_sweeps),
        "soft_metrics": empty_soft_metrics(),
        "calibration_bins": [],
        "per_repeat": [
            {
                **metric,
                **empty_soft_repeat_metrics(),
                "calibration_bins": [],
                "rejection_sweep": sweep,
            }
            for metric, sweep in zip(baseline_repeat_metrics, baseline_repeat_sweeps)
        ],
    }

    classifier_overall = aggregate_metric_dicts(classifier_repeat_metrics)
    classifier_method = {
        "name": "classifier",
        "display_name": f"{result_dir.name} / classifier",
        "overall": classifier_overall,
        "rejection_sweep": aggregate_rejection_sweeps(classifier_repeat_sweeps),
        "soft_metrics": {
            "classifier_brier_score": classifier_overall["classifier_brier_score"],
            "classifier_log_loss": classifier_overall["classifier_log_loss"],
            "classifier_avg_true_class_prob": classifier_overall["classifier_avg_true_class_prob"],
        },
        "calibration_bins": calibration_bins_from_repeats(labels_list, probs_list, calibration_bins),
        "per_repeat": [
            {
                **metric,
                "calibration_bins": compute_calibration_bins(labels, probs, calibration_bins),
                "rejection_sweep": sweep,
            }
            for metric, labels, probs, sweep in zip(classifier_repeat_metrics, labels_list, probs_list, classifier_repeat_sweeps)
        ],
    }

    return {
        "analysis_type": "supervised",
        "analysis_label": result_dir.name,
        "source_summary": str((result_dir / "summary.json").resolve()),
        "config": config,
        "methods": [baseline_method, classifier_method],
    }


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    summary = load_summary(result_dir)
    if "test_aggregate" in summary:
        analysis = analyze_supervised(summary, result_dir, int(args.calibration_bins))
    else:
        analysis = analyze_center_geometry(summary, result_dir)

    output_json = args.output_json.resolve() if args.output_json else (result_dir / "visualization_metrics.json")
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(json.dumps({"analysis_type": analysis["analysis_type"], "output_json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
