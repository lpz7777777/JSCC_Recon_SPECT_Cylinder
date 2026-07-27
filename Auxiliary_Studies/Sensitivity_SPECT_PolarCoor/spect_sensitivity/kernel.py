"""Sensi_d adapter around the shared local Compton event response."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .config import ComptonPhysicsConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from compton_event_response import (  # noqa: E402
    BatchDiagnostics,
    ComptonEventSettings,
    build_compton_cone_weights,
    build_detector_position_variance,
    normalize_event_response,
    prepare_compton_events,
)


def _settings_from_physics(physics: ComptonPhysicsConfig) -> ComptonEventSettings:
    return ComptonEventSettings(
        energy_mev=physics.energy_mev,
        energy_resolution=physics.energy_resolution,
        energy_threshold_max_mev=physics.energy_threshold_max_mev,
        energy_threshold_min_mev=physics.energy_threshold_min_mev,
        energy_threshold_sum_mev=physics.resolved_energy_threshold_sum_mev,
        delta_r1_mm=physics.delta_r1_mm,
        delta_r2_mm=physics.delta_r2_mm,
        min_event_effective_support=physics.min_event_effective_support,
        include_first_hit_source_leg_uncertainty=physics.include_first_hit_source_leg_uncertainty,
    )


@torch.inference_mode()
def accumulate_event_batch(
    events: torch.Tensor,
    physics: ComptonPhysicsConfig,
    detector_coordinates: torch.Tensor,
    detector_sigma_r1_sq: torch.Tensor,
    detector_sigma_r2_sq: torch.Tensor,
    voxel_coordinates: torch.Tensor,
    system_matrix: torch.Tensor,
    generator: torch.Generator,
    input_energies_already_smeared: bool = False,
) -> tuple[torch.Tensor, BatchDiagnostics]:
    """Accumulate posterior rows of the same density-basis operator used by MLEM."""
    diagnostics = BatchDiagnostics(input_events=int(events.shape[0]))
    pixel_count = int(voxel_coordinates.shape[0])
    empty_sum = torch.zeros(pixel_count, dtype=torch.float32, device=events.device)
    settings = _settings_from_physics(physics)
    prepared, prepared_diagnostics = prepare_compton_events(
        events,
        settings,
        detector_coordinates,
        detector_sigma_r1_sq,
        detector_sigma_r2_sq,
        input_energies_already_smeared=input_energies_already_smeared,
        generator=generator,
    )
    diagnostics = prepared_diagnostics
    if prepared is None:
        return empty_sum, diagnostics

    cone_weights = build_compton_cone_weights(prepared, voxel_coordinates, settings)
    normalized, _, _, invalid_count, low_support_count = normalize_event_response(
        cone_weights,
        prepared.cpnum1,
        system_matrix,
        settings.min_event_effective_support,
    )
    diagnostics.invalid_kernel_events = invalid_count
    diagnostics.low_support_rejected_events = low_support_count
    diagnostics.kept_events = int(normalized.shape[0])
    if normalized.numel() == 0:
        return empty_sum, diagnostics
    return torch.sum(normalized, dim=0), diagnostics
