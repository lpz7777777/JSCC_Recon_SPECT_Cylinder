from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class ResolvedDataset:
    factor_dir: Path
    compton_paths: tuple[Path, ...]
    system_matrix_path: Path
    detector_path: Path
    coordinate_path: Path
    rotation_path: Path | None
    detector_coordinates: np.ndarray
    voxel_coordinates: np.ndarray
    rotation_matrix: np.ndarray | None
    detector_count: int
    pixel_count: int
    rotate_num: int


@dataclass(frozen=True)
class EventBatch:
    values: torch.Tensor
    file_index: int
    next_file_offset: int


def expand_compton_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_dir():
            expanded.extend(sorted(item.resolve() for item in resolved.glob("*.csv") if item.is_file()))
        elif resolved.is_file():
            expanded.append(resolved)
        else:
            raise FileNotFoundError(f"Compton list path not found: {resolved}")
    if not expanded:
        raise FileNotFoundError("No Compton CSV files were found in the supplied paths.")
    return tuple(expanded)


def _load_csv(path: Path, dtype: np.dtype, usecols: tuple[int, ...] | None = None) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig") as handle:
        first_fields = handle.readline().strip().split(",")
    selected_fields = (
        first_fields
        if usecols is None
        else [first_fields[index] for index in usecols if index < len(first_fields)]
    )
    try:
        for field in selected_fields:
            float(field)
        skiprows = 0
    except ValueError:
        skiprows = 1
    try:
        values = np.loadtxt(
            path,
            delimiter=",",
            dtype=dtype,
            usecols=usecols,
            ndmin=2,
            skiprows=skiprows,
        )
    except ValueError as exc:
        raise ValueError(f"Failed to parse CSV file {path}: {exc}") from exc
    if values.size == 0:
        raise ValueError(f"CSV file is empty: {path}")
    return values


def _load_detector_coordinates(path: Path) -> np.ndarray:
    detector = _load_csv(path, np.float32)
    if detector.shape[1] == 4:
        detector_ids = detector[:, 0]
        rounded_ids = np.rint(detector_ids).astype(np.int64)
        expected_ids = np.arange(1, detector.shape[0] + 1, dtype=np.int64)
        if not np.allclose(detector_ids, rounded_ids) or not np.array_equal(rounded_ids, expected_ids):
            raise ValueError(
                f"Detector IDs in {path} must be consecutive, one-based, and match row order."
            )
        detector = detector[:, 1:4]
    elif detector.shape[1] != 3:
        raise ValueError(f"Detector CSV must contain [id,x,y,z] or [x,y,z], got {detector.shape}.")
    if not np.isfinite(detector).all():
        raise ValueError(f"Detector CSV contains non-finite coordinates: {path}")
    return np.ascontiguousarray(detector, dtype=np.float32)


def _load_voxel_coordinates(path: Path) -> np.ndarray:
    coordinates = _load_csv(path, np.float32)
    if coordinates.shape[1] < 3:
        raise ValueError(f"Coordinate CSV must contain at least three columns, got {coordinates.shape}.")
    coordinates = coordinates[:, :3]
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Coordinate CSV contains non-finite values: {path}")
    return np.ascontiguousarray(coordinates, dtype=np.float32)


def _load_rotation_matrix(path: Path, pixel_count: int, rotate_num: int) -> np.ndarray:
    rotation = _load_csv(path, np.int64)
    if rotation.shape[0] != pixel_count or rotation.shape[1] < rotate_num:
        raise ValueError(
            f"Rotation matrix shape {rotation.shape} is incompatible with "
            f"pixel_count={pixel_count}, rotate_num={rotate_num}."
        )
    rotation = rotation[:, :rotate_num]
    for column_index in range(rotate_num):
        column = rotation[:, column_index]
        if column.min() != 1 or column.max() != pixel_count or np.unique(column).size != pixel_count:
            raise ValueError(
                f"Rotation column {column_index + 1} in {path} is not a one-based permutation."
            )
    return np.ascontiguousarray(rotation, dtype=np.int64)


def resolve_dataset(
    factor_dir: Path,
    compton_paths: tuple[Path, ...],
    system_matrix_path: Path,
    detector_path: Path,
    coordinate_path: Path,
    rotation_path: Path | None,
    rotate_num: int,
    expected_detector_count: int,
    apply_rotation_average: bool,
) -> ResolvedDataset:
    factor_dir = factor_dir.resolve()
    if not factor_dir.is_dir():
        raise FileNotFoundError(f"Factor directory not found: {factor_dir}")

    resolved_compton_paths = expand_compton_paths(compton_paths)
    resolved_system_matrix_path = system_matrix_path.resolve()
    resolved_detector_path = detector_path.resolve()
    resolved_coordinate_path = coordinate_path.resolve()
    resolved_rotation_path = (
        rotation_path.resolve()
        if apply_rotation_average and rotation_path is not None
        else None
    )

    required_paths = [resolved_system_matrix_path, resolved_detector_path, resolved_coordinate_path]
    if apply_rotation_average:
        if resolved_rotation_path is None:
            raise ValueError("A rotation matrix is required when rotation averaging is enabled.")
        required_paths.append(resolved_rotation_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    detector_coordinates = _load_detector_coordinates(resolved_detector_path)
    voxel_coordinates = _load_voxel_coordinates(resolved_coordinate_path)
    detector_count = int(detector_coordinates.shape[0])
    pixel_count = int(voxel_coordinates.shape[0])

    if expected_detector_count and detector_count != expected_detector_count:
        raise ValueError(
            f"Detector count is {detector_count}, expected {expected_detector_count}. "
            "For the current JSCC geometry this should be 10496 (tungsten blocks excluded)."
        )

    expected_bytes = detector_count * pixel_count * np.dtype(np.float32).itemsize
    actual_bytes = resolved_system_matrix_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"System matrix size mismatch: {resolved_system_matrix_path} has {actual_bytes} bytes; "
            f"expected {expected_bytes} for [{detector_count}, {pixel_count}] float32."
        )

    rotation_matrix = None
    if apply_rotation_average and resolved_rotation_path is not None:
        rotation_matrix = _load_rotation_matrix(resolved_rotation_path, pixel_count, rotate_num)

    return ResolvedDataset(
        factor_dir=factor_dir,
        compton_paths=resolved_compton_paths,
        system_matrix_path=resolved_system_matrix_path,
        detector_path=resolved_detector_path,
        coordinate_path=resolved_coordinate_path,
        rotation_path=resolved_rotation_path,
        detector_coordinates=detector_coordinates,
        voxel_coordinates=voxel_coordinates,
        rotation_matrix=rotation_matrix,
        detector_count=detector_count,
        pixel_count=pixel_count,
        rotate_num=rotate_num,
    )


def load_system_matrix(dataset: ResolvedDataset, device: torch.device) -> torch.Tensor:
    # MATLAB writes the detector axis first. Reshape as [pixel, detector], then transpose.
    mapped = np.memmap(dataset.system_matrix_path, dtype=np.float32, mode="c")
    matrix = mapped.reshape(dataset.pixel_count, dataset.detector_count).T
    tensor = torch.from_numpy(matrix)
    if device.type == "cuda":
        tensor = tensor.to(device, non_blocking=False)
    return tensor


def count_event_rows(paths: tuple[Path, ...]) -> tuple[int, ...]:
    row_counts: list[int] = []
    for path in paths:
        line_count = 0
        last_byte = b""
        with path.open("rb") as file:
            while True:
                block = file.read(8 * 1024 * 1024)
                if not block:
                    break
                line_count += block.count(b"\n")
                last_byte = block[-1:]
        if last_byte and last_byte != b"\n":
            line_count += 1
        row_counts.append(line_count)
    return tuple(row_counts)


def iter_event_batches(
    paths: tuple[Path, ...],
    batch_size: int,
    selected_event_count: int,
    start_file_index: int = 0,
    start_file_offset: int = 0,
    already_processed: int = 0,
) -> Iterator[EventBatch]:
    remaining = selected_event_count - already_processed
    if remaining <= 0:
        return

    for file_index in range(start_file_index, len(paths)):
        path = paths[file_index]
        with path.open("rb") as file:
            if file_index == start_file_index and start_file_offset:
                file.seek(start_file_offset)
            while remaining > 0:
                rows_to_read = min(batch_size, remaining)
                try:
                    values = np.loadtxt(
                        file,
                        delimiter=",",
                        dtype=np.float32,
                        usecols=(0, 1, 2, 3),
                        max_rows=rows_to_read,
                        ndmin=2,
                    )
                except ValueError as exc:
                    raise ValueError(f"Failed to parse Compton list {path}: {exc}") from exc
                if values.size == 0:
                    break
                values = np.ascontiguousarray(values, dtype=np.float32)
                remaining -= int(values.shape[0])
                yield EventBatch(
                    values=torch.from_numpy(values),
                    file_index=file_index,
                    next_file_offset=file.tell(),
                )
                if values.shape[0] < rows_to_read:
                    break
            if remaining <= 0:
                return

    if remaining > 0:
        raise RuntimeError(
            f"Compton input ended {remaining} rows before the selected event count was reached."
        )
