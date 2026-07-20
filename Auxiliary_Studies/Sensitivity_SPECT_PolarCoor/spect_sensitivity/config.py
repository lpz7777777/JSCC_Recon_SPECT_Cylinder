from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComptonPhysicsConfig:
    energy_mev: float
    energy_resolution_662kev: float = 0.1
    energy_threshold_min_mev: float = 0.050
    energy_threshold_sum_mev: float | None = None
    delta_r1_mm: float = 0.0
    delta_r2_mm: float = 0.0
    min_event_effective_support: float = 50.0
    include_first_hit_source_leg_uncertainty: bool = False

    @property
    def energy_resolution(self) -> float:
        return self.energy_resolution_662kev * (0.662 / self.energy_mev) ** 0.5

    @property
    def energy_threshold_max_mev(self) -> float:
        electron_rest_energy_mev = 0.511
        return (
            2.0 * self.energy_mev**2
            / (electron_rest_energy_mev + 2.0 * self.energy_mev)
            - 0.001
        )

    @property
    def resolved_energy_threshold_sum_mev(self) -> float:
        if self.energy_threshold_sum_mev is not None:
            return self.energy_threshold_sum_mev
        project_defaults = {
            0.218: 0.18,
            0.440: 0.40,
            0.511: 0.46,
            0.662: 0.60,
        }
        for energy_mev, threshold_mev in project_defaults.items():
            if abs(self.energy_mev - energy_mev) < 5e-4:
                return threshold_mev
        return 0.9 * self.energy_mev

    def validate(self) -> None:
        if self.energy_mev <= 0:
            raise ValueError("energy_mev must be positive.")
        if self.energy_resolution_662kev < 0:
            raise ValueError("energy_resolution_662kev cannot be negative.")
        if self.energy_threshold_min_mev < 0:
            raise ValueError("energy_threshold_min_mev cannot be negative.")
        if not 0 < self.resolved_energy_threshold_sum_mev <= self.energy_mev * 1.25:
            raise ValueError(
                "energy_threshold_sum_mev must be positive and physically "
                "consistent with the incident energy."
            )
        if self.delta_r1_mm < 0 or self.delta_r2_mm < 0:
            raise ValueError("Position uncertainty values cannot be negative.")
        if self.min_event_effective_support <= 0:
            raise ValueError("min_event_effective_support must be positive.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["energy_resolution"] = self.energy_resolution
        result["energy_threshold_max_mev"] = self.energy_threshold_max_mev
        result["resolved_energy_threshold_sum_mev"] = self.resolved_energy_threshold_sum_mev
        return result


@dataclass(frozen=True)
class SensitivityRunConfig:
    factor_dir: Path
    compton_paths: tuple[Path, ...]
    output_dir: Path
    source_photons: float
    physics: ComptonPhysicsConfig
    system_matrix_path: Path
    detector_path: Path
    coordinate_path: Path
    rotation_path: Path | None
    rotate_num: int
    source_volume_mm3: float | None = None
    event_fraction: float = 1.0
    batch_size: int = 256
    device: str = "auto"
    seed: int = 20260710
    expected_detector_count: int = 10496
    apply_rotation_average: bool = True
    save_raw: bool = True
    checkpoint_every_batches: int = 100
    progress_every_batches: int = 10
    resume: bool = False
    overwrite: bool = False
    keep_checkpoint: bool = False
    install_to_factor_dir: bool = False

    def validate(self) -> None:
        self.physics.validate()
        if self.source_photons <= 0:
            raise ValueError("source_photons must be positive.")
        if self.source_volume_mm3 is not None and self.source_volume_mm3 <= 0:
            raise ValueError("source_volume_mm3 must be positive when provided.")
        if not 0 < self.event_fraction <= 1:
            raise ValueError("event_fraction must be in (0, 1].")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.rotate_num <= 0:
            raise ValueError("rotate_num must be positive.")
        if self.expected_detector_count < 0:
            raise ValueError("expected_detector_count cannot be negative.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        if self.checkpoint_every_batches < 0 or self.progress_every_batches <= 0:
            raise ValueError("Checkpoint interval cannot be negative; progress interval must be positive.")
        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite cannot be enabled together.")
        if not self.compton_paths:
            raise ValueError("At least one Compton list file is required.")
