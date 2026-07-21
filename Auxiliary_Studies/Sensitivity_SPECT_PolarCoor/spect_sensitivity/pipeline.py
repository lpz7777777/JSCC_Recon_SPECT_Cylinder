from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import SensitivityRunConfig
from .io import (
    ResolvedDataset,
    count_event_rows,
    iter_event_batches,
    load_system_matrix,
    resolve_dataset,
)
from .kernel import (
    BatchDiagnostics,
    accumulate_event_batch,
    build_detector_position_variance,
)


CHECKPOINT_VERSION = 2


def _resolve_normalization(
    config: SensitivityRunConfig,
    dataset: ResolvedDataset,
) -> dict[str, Any]:
    manifest_path = dataset.factor_dir / "factor_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

    maps_activity_density = bool(manifest.get("maps_activity_density", False))
    if not maps_activity_density:
        if config.source_volume_mm3 is not None:
            raise ValueError(
                "--source-volume-mm3 is only valid for density-basis Factors."
            )
        return {
            "mode": "legacy_integrated_activity_per_polar_cell",
            "maps_activity_density": False,
            "source_volume_mm3": None,
            "factor_support_volume_mm3": None,
            "volume_file": None,
            "equation": "Sensi_d = accumulator * pixel_count / represented_source_photons",
        }

    volume_path = dataset.factor_dir / "polar_cell_volume_mm3.float64"
    if not volume_path.is_file():
        raise FileNotFoundError(
            f"Density-basis Factors require a polar-cell volume file: {volume_path}"
        )
    volumes = np.fromfile(volume_path, dtype="<f8")
    if volumes.size != dataset.pixel_count:
        raise ValueError(
            f"Polar-cell volume count is {volumes.size}, expected {dataset.pixel_count}."
        )
    if not np.isfinite(volumes).all() or np.any(volumes <= 0.0):
        raise ValueError("Polar-cell volumes must all be finite and positive.")

    support_volume_mm3 = float(np.sum(volumes, dtype=np.float64))
    source_volume_mm3 = (
        support_volume_mm3
        if config.source_volume_mm3 is None
        else float(config.source_volume_mm3)
    )
    relative_volume_error = abs(source_volume_mm3 / support_volume_mm3 - 1.0)
    if relative_volume_error > 1e-6:
        raise ValueError(
            "The current density-basis estimator requires a uniform source covering every "
            "polar cell completely. source_volume_mm3 must equal the Factors support volume: "
            f"source={source_volume_mm3:.12g}, support={support_volume_mm3:.12g}, "
            f"relative error={relative_volume_error:.3e}."
        )

    return {
        "mode": "uniform_full_support_activity_density",
        "maps_activity_density": True,
        "source_volume_mm3": source_volume_mm3,
        "factor_support_volume_mm3": support_volume_mm3,
        "volume_file": str(volume_path),
        "volume_file_signature": _path_signature(volume_path),
        "equation": "Sensi_d = accumulator * source_volume_mm3 / represented_source_photons",
    }


def _select_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def _configuration_fingerprint(
    config: SensitivityRunConfig,
    dataset: ResolvedDataset,
    selected_event_count: int,
    resolved_device: torch.device,
    normalization: dict[str, Any],
) -> str:
    payload = {
        "factor_dir": str(dataset.factor_dir),
        "compton_files": [_path_signature(path) for path in dataset.compton_paths],
        "system_matrix": _path_signature(dataset.system_matrix_path),
        "detector": _path_signature(dataset.detector_path),
        "coordinates": _path_signature(dataset.coordinate_path),
        "rotation": _path_signature(dataset.rotation_path) if dataset.rotation_path else None,
        "source_photons": config.source_photons,
        "normalization": normalization,
        "event_fraction": config.event_fraction,
        "selected_event_count": selected_event_count,
        "batch_size": config.batch_size,
        "resolved_device": str(resolved_device),
        "seed": config.seed,
        "rotate_num": config.rotate_num,
        "apply_rotation_average": config.apply_rotation_average,
        "input_energies_already_smeared": config.input_energies_already_smeared,
        "physics": config.physics.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_binary(path: Path, values: np.ndarray) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as file:
        np.asarray(values, dtype=np.float32).tofile(file)
    os.replace(temporary_path, path)


def _atomic_write_json(path: Path, values: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_path, path)


def _save_checkpoint(
    path: Path,
    fingerprint: str,
    accumulator: torch.Tensor,
    generator: torch.Generator,
    diagnostics: BatchDiagnostics,
    processed_events: int,
    completed_batches: int,
    file_index: int,
    file_offset: int,
) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            version=np.asarray(CHECKPOINT_VERSION, dtype=np.int64),
            fingerprint=np.asarray(fingerprint),
            accumulator=accumulator.detach().cpu().numpy().astype(np.float32, copy=False),
            generator_state=generator.get_state().cpu().numpy(),
            diagnostics=np.asarray(json.dumps(diagnostics.to_dict(), sort_keys=True)),
            processed_events=np.asarray(processed_events, dtype=np.int64),
            completed_batches=np.asarray(completed_batches, dtype=np.int64),
            file_index=np.asarray(file_index, dtype=np.int64),
            file_offset=np.asarray(file_offset, dtype=np.int64),
        )
    os.replace(temporary_path, path)


def _load_checkpoint(
    path: Path,
    expected_fingerprint: str,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, BatchDiagnostics, int, int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume requested, but checkpoint was not found: {path}")
    with np.load(path, allow_pickle=False) as checkpoint:
        version = int(checkpoint["version"].item())
        fingerprint = str(checkpoint["fingerprint"].item())
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}.")
        if fingerprint != expected_fingerprint:
            raise ValueError(
                "Checkpoint inputs or numerical settings differ from this run. "
                "Use the original settings, or start a new output directory."
            )
        accumulator = torch.from_numpy(checkpoint["accumulator"].copy()).to(device)
        generator_state = torch.from_numpy(checkpoint["generator_state"].copy())
        generator.set_state(generator_state)
        diagnostics = BatchDiagnostics.from_dict(json.loads(str(checkpoint["diagnostics"].item())))
        processed_events = int(checkpoint["processed_events"].item())
        completed_batches = int(checkpoint["completed_batches"].item())
        file_index = int(checkpoint["file_index"].item())
        file_offset = int(checkpoint["file_offset"].item())
    return accumulator, diagnostics, processed_events, completed_batches, file_index, file_offset


def _apply_rotation_average(
    sensitivity: np.ndarray,
    rotation_matrix: np.ndarray,
    rotate_num: int,
) -> np.ndarray:
    source = torch.from_numpy(np.asarray(sensitivity, dtype=np.float32))
    result = torch.zeros_like(source)
    indices = torch.from_numpy(rotation_matrix[:, :rotate_num]).long() - 1
    for rotate_index in range(rotate_num):
        result += torch.index_select(source, 0, indices[:, rotate_index])
    return (result / rotate_num).numpy()


def _format_eta(elapsed_seconds: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= completed:
        return "00:00:00"
    remaining = elapsed_seconds * (total - completed) / completed
    hours, remainder = divmod(int(remaining), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _install_result(config: SensitivityRunConfig, source_path: Path) -> Path | None:
    if not config.install_to_factor_dir:
        return None
    destination = config.factor_dir.resolve() / "Sensi_d"
    if destination.exists() and not config.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing factor sensitivity: {destination}. "
            "Inspect the Result output, then rerun with --overwrite to install it."
        )
    temporary_path = destination.with_name(".Sensi_d.tmp")
    shutil.copyfile(source_path, temporary_path)
    os.replace(temporary_path, destination)
    return destination


def run_sensitivity_calculation(config: SensitivityRunConfig) -> dict[str, Any]:
    config.validate()
    device = _select_device(config.device)
    dataset = resolve_dataset(
        factor_dir=config.factor_dir,
        compton_paths=config.compton_paths,
        system_matrix_path=config.system_matrix_path,
        detector_path=config.detector_path,
        coordinate_path=config.coordinate_path,
        rotation_path=config.rotation_path,
        rotate_num=config.rotate_num,
        expected_detector_count=config.expected_detector_count,
        apply_rotation_average=config.apply_rotation_average,
    )
    normalization = _resolve_normalization(config, dataset)

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Sensi_d"
    raw_output_path = output_dir / "Sensi_d_raw"
    metadata_path = output_dir / "run_metadata.json"
    checkpoint_path = output_dir / "checkpoint.npz"
    existing_outputs = [path for path in (output_path, metadata_path) if path.exists()]
    if existing_outputs and not config.overwrite and not config.resume:
        raise FileExistsError(
            f"Output already exists in {output_dir}. Use --overwrite or choose another directory."
        )
    if checkpoint_path.exists() and not config.resume and not config.overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. Use --resume or --overwrite."
        )
    install_path = config.factor_dir.resolve() / "Sensi_d"
    if config.install_to_factor_dir and install_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing factor sensitivity: {install_path}. "
            "Inspect a Result run first, then use --overwrite to replace it."
        )

    print("Counting Compton input rows...")
    event_rows_per_file = count_event_rows(dataset.compton_paths)
    total_event_rows = int(sum(event_rows_per_file))
    if total_event_rows <= 0:
        raise ValueError("All supplied Compton list files are empty.")
    selected_event_count = int(total_event_rows * config.event_fraction)
    if selected_event_count <= 0:
        raise ValueError(
            f"event_fraction={config.event_fraction} selects zero rows from {total_event_rows} events."
        )
    represented_fraction = selected_event_count / total_event_rows
    represented_source_photons = config.source_photons * represented_fraction
    fingerprint = _configuration_fingerprint(
        config, dataset, selected_event_count, device, normalization
    )

    print(
        f"Dataset: detectors={dataset.detector_count}, pixels={dataset.pixel_count}, "
        f"rotations={dataset.rotate_num}"
    )
    print(
        f"Events: total={total_event_rows}, selected={selected_event_count} "
        f"({represented_fraction:.8f}), represented photons={represented_source_photons:.6e}"
    )
    print(f"Device: {device}; batch size: {config.batch_size}")

    load_start = time.perf_counter()
    system_matrix = load_system_matrix(dataset, device)
    detector_coordinates = torch.from_numpy(dataset.detector_coordinates).to(device)
    voxel_coordinates = torch.from_numpy(dataset.voxel_coordinates).to(device)
    detector_sigma_r1_sq = build_detector_position_variance(
        detector_coordinates, config.physics.delta_r1_mm
    )
    detector_sigma_r2_sq = build_detector_position_variance(
        detector_coordinates, config.physics.delta_r2_mm
    )
    print(f"Static inputs loaded in {time.perf_counter() - load_start:.2f} s")

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    accumulator = torch.zeros(dataset.pixel_count, dtype=torch.float32, device=device)
    diagnostics = BatchDiagnostics()
    processed_events = 0
    completed_batches = 0
    start_file_index = 0
    start_file_offset = 0
    if config.resume:
        (
            accumulator,
            diagnostics,
            processed_events,
            completed_batches,
            start_file_index,
            start_file_offset,
        ) = _load_checkpoint(checkpoint_path, fingerprint, device, generator)
        if accumulator.numel() != dataset.pixel_count:
            raise ValueError("Checkpoint accumulator size does not match the current factor grid.")
        print(
            f"Resumed checkpoint after {processed_events}/{selected_event_count} input events "
            f"and {completed_batches} batches."
        )

    calculation_start = time.perf_counter()
    session_start_processed_events = processed_events
    last_file_index = start_file_index
    last_file_offset = start_file_offset
    for event_batch in iter_event_batches(
        dataset.compton_paths,
        config.batch_size,
        selected_event_count,
        start_file_index=start_file_index,
        start_file_offset=start_file_offset,
        already_processed=processed_events,
    ):
        events = event_batch.values.to(device, non_blocking=True)
        batch_sum, batch_diagnostics = accumulate_event_batch(
            events=events,
            physics=config.physics,
            detector_coordinates=detector_coordinates,
            detector_sigma_r1_sq=detector_sigma_r1_sq,
            detector_sigma_r2_sq=detector_sigma_r2_sq,
            voxel_coordinates=voxel_coordinates,
            system_matrix=system_matrix,
            generator=generator,
            input_energies_already_smeared=config.input_energies_already_smeared,
        )
        accumulator += batch_sum
        diagnostics.add_(batch_diagnostics)
        processed_events += int(events.shape[0])
        completed_batches += 1
        last_file_index = event_batch.file_index
        last_file_offset = event_batch.next_file_offset

        if completed_batches % config.progress_every_batches == 0 or processed_events == selected_event_count:
            elapsed = time.perf_counter() - calculation_start
            completed_this_session = processed_events - session_start_processed_events
            remaining_this_session = selected_event_count - session_start_processed_events
            print(
                f"Batch {completed_batches}: input {processed_events}/{selected_event_count} "
                f"({100.0 * processed_events / selected_event_count:.2f}%), "
                f"kept {diagnostics.kept_events}, ETA "
                f"{_format_eta(elapsed, completed_this_session, remaining_this_session)}"
            )
        if (
            config.checkpoint_every_batches > 0
            and completed_batches % config.checkpoint_every_batches == 0
        ):
            _save_checkpoint(
                checkpoint_path,
                fingerprint,
                accumulator,
                generator,
                diagnostics,
                processed_events,
                completed_batches,
                last_file_index,
                last_file_offset,
            )

    if processed_events != selected_event_count:
        raise RuntimeError(
            f"Processed {processed_events} events, expected {selected_event_count}."
        )
    if diagnostics.kept_events <= 0:
        raise RuntimeError("No Compton events remained after filtering; no Sensi_d was written.")

    accumulator_cpu = accumulator.detach().cpu().numpy().astype(np.float64)
    average_before_scaling = float(np.mean(accumulator_cpu))
    if not np.isfinite(average_before_scaling) or average_before_scaling <= 0:
        raise RuntimeError("Accumulated sensitivity is non-finite or non-positive.")
    accepted_events_per_photon = diagnostics.kept_events / represented_source_photons
    if normalization["maps_activity_density"]:
        source_volume_mm3 = float(normalization["source_volume_mm3"])
        raw_sensitivity = (
            accumulator_cpu * source_volume_mm3 / represented_source_photons
        ).astype(np.float32)
        target_average = (
            accepted_events_per_photon * source_volume_mm3 / dataset.pixel_count
        )
    else:
        raw_sensitivity = (
            accumulator_cpu * dataset.pixel_count / represented_source_photons
        ).astype(np.float32)
        target_average = accepted_events_per_photon
    sensitivity = raw_sensitivity
    if config.apply_rotation_average:
        if dataset.rotation_matrix is None:
            raise RuntimeError("Rotation averaging was requested but no matrix was loaded.")
        sensitivity = _apply_rotation_average(
            raw_sensitivity, dataset.rotation_matrix, config.rotate_num
        ).astype(np.float32)

    for name, values in (("Sensi_d_raw", raw_sensitivity), ("Sensi_d", sensitivity)):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError(f"{name} contains non-finite or negative values.")
    final_average = float(np.mean(sensitivity, dtype=np.float64))
    relative_mean_error = abs(final_average - target_average) / target_average
    if relative_mean_error > 5e-5:
        raise RuntimeError(
            f"Final sensitivity mean failed normalization validation: relative error {relative_mean_error:.3e}."
        )

    _atomic_write_binary(output_path, sensitivity)
    if config.save_raw:
        _atomic_write_binary(raw_output_path, raw_sensitivity)
    elif config.overwrite and raw_output_path.exists():
        raw_output_path.unlink()
    installed_path = _install_result(config, output_path)

    elapsed_seconds = time.perf_counter() - calculation_start
    metadata = {
        "tool": "Auxiliary_Studies/Sensitivity_SPECT_PolarCoor",
        "checkpoint_version": CHECKPOINT_VERSION,
        "configuration_fingerprint": fingerprint,
        "device": str(device),
        "seed": config.seed,
        "factor_dir": str(dataset.factor_dir),
        "files": {
            "compton": [str(path) for path in dataset.compton_paths],
            "system_matrix": str(dataset.system_matrix_path),
            "detector": str(dataset.detector_path),
            "coordinates": str(dataset.coordinate_path),
            "rotation": str(dataset.rotation_path) if dataset.rotation_path else None,
        },
        "dimensions": {
            "detector_count": dataset.detector_count,
            "pixel_count": dataset.pixel_count,
            "rotate_num": dataset.rotate_num,
        },
        "events": {
            "rows_per_file": list(event_rows_per_file),
            "total_rows": total_event_rows,
            "selected_rows": selected_event_count,
            "represented_fraction": represented_fraction,
            **diagnostics.to_dict(),
        },
        "normalization": {
            **normalization,
            "full_input_source_photons": config.source_photons,
            "represented_source_photons": represented_source_photons,
            "average_before_scaling": average_before_scaling,
            "accepted_events_per_photon": accepted_events_per_photon,
            "target_average_sensitivity": target_average,
            "final_average": final_average,
            "final_integral_over_source_volume": (
                float(np.sum(sensitivity, dtype=np.float64))
                / float(normalization["source_volume_mm3"])
                if normalization["maps_activity_density"]
                else None
            ),
            "relative_mean_error": relative_mean_error,
        },
        "physics": config.physics.to_dict(),
        "input_energies_already_smeared": config.input_energies_already_smeared,
        "execution": {
            "batch_size": config.batch_size,
            "completed_batches": completed_batches,
            "elapsed_seconds_this_session": elapsed_seconds,
            "rotation_average_applied": config.apply_rotation_average,
            "raw_output_saved": config.save_raw,
        },
        "outputs": {
            "sensitivity": str(output_path),
            "raw_sensitivity": str(raw_output_path) if config.save_raw else None,
            "installed_sensitivity": str(installed_path) if installed_path else None,
        },
    }
    _atomic_write_json(metadata_path, metadata)
    if checkpoint_path.exists() and not config.keep_checkpoint:
        checkpoint_path.unlink()

    print(f"Kept events: {diagnostics.kept_events}/{selected_event_count}")
    print(f"Target/final mean: {target_average:.6e} / {metadata['normalization']['final_average']:.6e}")
    print(f"Sensi_d saved to: {output_path}")
    if installed_path is not None:
        print(f"Installed to factor directory: {installed_path}")
    # Release the NumPy-backed matrix tensor before callers delete a temporary
    # Factor directory on Windows.
    del system_matrix, detector_coordinates, voxel_coordinates, accumulator
    gc.collect()
    return metadata
