from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .config import ComptonPhysicsConfig


FRONT_LAYER_CRYSTAL_MM = (3.0, 3.0, 3.0)
REAR_LAYER_CRYSTAL_MM = (2.0, 6.0, 2.0)
ELECTRON_REST_ENERGY_MEV = 0.511


@dataclass
class BatchDiagnostics:
    input_events: int = 0
    invalid_raw_energy_events: int = 0
    energy_rejected_events: int = 0
    kinematic_rejected_events: int = 0
    same_layer_rejected_events: int = 0
    invalid_kernel_events: int = 0
    low_support_rejected_events: int = 0
    kept_events: int = 0

    def add_(self, other: "BatchDiagnostics") -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int]) -> "BatchDiagnostics":
        return cls(**{name: int(values.get(name, 0)) for name in cls.__dataclass_fields__})


def _uniform_sigma_sq_from_size(size_mm: float) -> float:
    return size_mm**2 / 12.0


def build_detector_position_variance(
    detector_coordinates: torch.Tensor,
    extra_sigma_mm: float,
) -> torch.Tensor:
    detector_y_abs = torch.abs(detector_coordinates[:, 1])
    layer_y_abs = torch.sort(torch.unique(detector_y_abs))[0]
    if layer_y_abs.numel() != 4:
        raise ValueError(
            "The current detector uncertainty model requires exactly four absolute-y layers; "
            f"found {layer_y_abs.numel()}. Check Detector.csv and detector ordering."
        )

    front_variance = torch.tensor(
        [_uniform_sigma_sq_from_size(value) for value in FRONT_LAYER_CRYSTAL_MM],
        dtype=detector_coordinates.dtype,
        device=detector_coordinates.device,
    )
    rear_variance = torch.tensor(
        [_uniform_sigma_sq_from_size(value) for value in REAR_LAYER_CRYSTAL_MM],
        dtype=detector_coordinates.dtype,
        device=detector_coordinates.device,
    )
    variance = torch.empty_like(detector_coordinates)
    for layer_index, layer_y in enumerate(layer_y_abs):
        variance[detector_y_abs == layer_y] = front_variance if layer_index < 3 else rear_variance
    if extra_sigma_mm:
        variance = variance + extra_sigma_mm**2
    return variance


def _compton_theta_from_e1(e1: torch.Tensor, energy_mev: float) -> torch.Tensor:
    cos_theta = 1.0 - (
        ELECTRON_REST_ENERGY_MEV * e1 / ((energy_mev - e1) * energy_mev)
    )
    return torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))


def _energy_angle_sigma(
    e1: torch.Tensor,
    physics: ComptonPhysicsConfig,
    beta: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    sigma_e = (
        e1
        * physics.energy_resolution
        / 2.355
        * (physics.energy_mev / e1) ** 0.5
    )
    e1_low = torch.clamp(e1 - sigma_e, 1e-7, physics.energy_mev - 1e-7)
    e1_high = torch.clamp(e1 + sigma_e, 1e-7, physics.energy_mev - 1e-7)
    theta_low = _compton_theta_from_e1(e1_low, physics.energy_mev)
    theta_high = _compton_theta_from_e1(e1_high, physics.energy_mev)
    sigma_minus = torch.clamp(theta - theta_low, min=1e-7)
    sigma_plus = torch.clamp(theta_high - theta, min=1e-7)
    delta_theta = beta - theta.unsqueeze(1)
    return torch.where(
        delta_theta >= 0,
        sigma_plus.unsqueeze(1),
        sigma_minus.unsqueeze(1),
    )


def _position_angle_sigma(
    vector01: torch.Tensor,
    vector12: torch.Tensor,
    sigma_pos1_sq: torch.Tensor,
    sigma_pos2_sq: torch.Tensor,
    include_first_hit_source_leg_uncertainty: bool,
) -> torch.Tensor:
    distance01 = torch.norm(vector01, dim=2, keepdim=True)
    distance12 = torch.norm(vector12, dim=2, keepdim=True)
    unit01 = vector01 / torch.clamp(distance01, min=1e-7)
    unit12 = vector12 / torch.clamp(distance12, min=1e-7)
    cos_beta = torch.sum(unit01 * unit12, dim=2, keepdim=True)
    sin_beta = torch.sqrt(torch.clamp(1.0 - cos_beta**2, min=1e-12))

    jac_source_leg = (unit12 - cos_beta * unit01) / torch.clamp(distance01, min=1e-7)
    jac_inter_crystal = (unit01 - cos_beta * unit12) / torch.clamp(distance12, min=1e-7)

    # SysMat_polar already integrates first-crystal extent on the source-to-hit leg.
    grad_pos1 = jac_inter_crystal / sin_beta
    if include_first_hit_source_leg_uncertainty:
        grad_pos1 = grad_pos1 - jac_source_leg / sin_beta
    grad_pos2 = -jac_inter_crystal / sin_beta

    variance = torch.sum(grad_pos1**2 * sigma_pos1_sq.unsqueeze(1), dim=2)
    variance += torch.sum(grad_pos2**2 * sigma_pos2_sq.unsqueeze(1), dim=2)
    return torch.sqrt(torch.clamp(variance, min=1e-12))


def _validate_detector_ids(events: torch.Tensor, detector_count: int) -> None:
    detector_ids = events[:, (0, 2)]
    rounded = torch.round(detector_ids)
    valid = (
        torch.isfinite(detector_ids).all(dim=1)
        & torch.isclose(detector_ids, rounded, atol=1e-4, rtol=0).all(dim=1)
        & (rounded >= 1).all(dim=1)
        & (rounded <= detector_count).all(dim=1)
    )
    if not bool(torch.all(valid)):
        invalid_count = int((~valid).sum().item())
        raise ValueError(
            f"Found {invalid_count} events with invalid detector IDs. Expected one-based IDs "
            f"in [1, {detector_count}]; the List and Factors likely use different geometries."
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
    system_matrix_column_scale: torch.Tensor | None,
    generator: torch.Generator,
    input_energies_already_smeared: bool = False,
) -> tuple[torch.Tensor, BatchDiagnostics]:
    diagnostics = BatchDiagnostics(input_events=int(events.shape[0]))
    pixel_count = int(voxel_coordinates.shape[0])
    empty_sum = torch.zeros(pixel_count, dtype=torch.float32, device=events.device)
    if events.numel() == 0:
        return empty_sum, diagnostics

    _validate_detector_ids(events, int(detector_coordinates.shape[0]))
    cpnum1 = torch.round(events[:, 0]).long()
    cpnum2 = torch.round(events[:, 2]).long()
    e1 = events[:, 1]
    e2 = events[:, 3]

    valid_raw_energy = torch.isfinite(e1) & torch.isfinite(e2) & (e1 > 0) & (e2 > 0)
    diagnostics.invalid_raw_energy_events = int((~valid_raw_energy).sum().item())
    cpnum1, cpnum2, e1, e2 = (
        value[valid_raw_energy] for value in (cpnum1, cpnum2, e1, e2)
    )
    if e1.numel() == 0:
        return empty_sum, diagnostics

    if not input_energies_already_smeared:
        sigma_1 = e1 * physics.energy_resolution / 2.355 * (physics.energy_mev / e1) ** 0.5
        sigma_2 = e2 * physics.energy_resolution / 2.355 * (physics.energy_mev / e2) ** 0.5
        e1 = e1 + sigma_1 * torch.randn(e1.shape, device=e1.device, generator=generator)
        e2 = e2 + sigma_2 * torch.randn(e2.shape, device=e2.device, generator=generator)

    valid_energy = (
        (e1 < physics.energy_threshold_max_mev)
        & (e1 > physics.energy_threshold_min_mev)
        & (e2 > physics.energy_threshold_min_mev)
        & ((e1 + e2) > physics.resolved_energy_threshold_sum_mev)
    )
    diagnostics.energy_rejected_events = int((~valid_energy).sum().item())
    cpnum1, cpnum2, e1, e2 = (
        value[valid_energy] for value in (cpnum1, cpnum2, e1, e2)
    )
    if e1.numel() == 0:
        return empty_sum, diagnostics

    cos_theta_raw = 1.0 - (
        ELECTRON_REST_ENERGY_MEV * e1
        / ((physics.energy_mev - e1) * physics.energy_mev)
    )
    valid_kinematics = (
        torch.isfinite(cos_theta_raw)
        & (cos_theta_raw > -1.0 + 1e-6)
        & (cos_theta_raw < 1.0 - 1e-6)
    )
    diagnostics.kinematic_rejected_events = int((~valid_kinematics).sum().item())
    cpnum1, cpnum2, e1, e2 = (
        value[valid_kinematics] for value in (cpnum1, cpnum2, e1, e2)
    )
    if e1.numel() == 0:
        return empty_sum, diagnostics

    pos1 = detector_coordinates[cpnum1 - 1]
    pos2 = detector_coordinates[cpnum2 - 1]
    sigma_pos1_sq = detector_sigma_r1_sq[cpnum1 - 1]
    sigma_pos2_sq = detector_sigma_r2_sq[cpnum2 - 1]
    valid_layer = torch.abs(pos1[:, 1] - pos2[:, 1]) > 0.1
    diagnostics.same_layer_rejected_events = int((~valid_layer).sum().item())
    cpnum1, e1, pos1, pos2, sigma_pos1_sq, sigma_pos2_sq = (
        value[valid_layer]
        for value in (cpnum1, e1, pos1, pos2, sigma_pos1_sq, sigma_pos2_sq)
    )
    if e1.numel() == 0:
        return empty_sum, diagnostics

    vector01 = pos1.unsqueeze(1) - voxel_coordinates.unsqueeze(0)
    vector12 = (pos2 - pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)
    theta = _compton_theta_from_e1(e1, physics.energy_mev)
    klein_nishina = physics.energy_mev / (physics.energy_mev - e1)
    klein_nishina += (physics.energy_mev - e1) / physics.energy_mev
    beta_cos = torch.sum(vector01 * vector12, dim=2) / torch.clamp(
        distance01 * distance12, min=1e-7
    )
    beta = torch.acos(torch.clamp(beta_cos, -1.0 + 1e-7, 1.0 - 1e-7))

    sigma_energy = _energy_angle_sigma(e1, physics, beta, theta)
    sigma_position = _position_angle_sigma(
        vector01,
        vector12,
        sigma_pos1_sq,
        sigma_pos2_sq,
        physics.include_first_hit_source_leg_uncertainty,
    )
    sigma_angle = torch.sqrt(torch.clamp(sigma_energy**2 + sigma_position**2, min=1e-12))
    weights = torch.exp(-((beta - theta.unsqueeze(1)) ** 2) / (2.0 * sigma_angle**2))
    weights *= klein_nishina.unsqueeze(1) - torch.sin(beta) ** 2
    point_response = torch.index_select(system_matrix, 0, cpnum1 - 1)
    if system_matrix_column_scale is not None:
        point_response = point_response * system_matrix_column_scale.unsqueeze(0)
    weights *= point_response

    row_sums = torch.sum(weights, dim=1)
    valid_kernel = torch.isfinite(weights).all(dim=1) & torch.isfinite(row_sums) & (row_sums > 0)
    diagnostics.invalid_kernel_events = int((~valid_kernel).sum().item())
    weights = weights[valid_kernel]
    row_sums = row_sums[valid_kernel]
    if weights.numel() == 0:
        return empty_sum, diagnostics

    normalized = weights / row_sums.unsqueeze(1)
    effective_support = 1.0 / torch.sum(normalized**2, dim=1)
    stable = effective_support >= physics.min_event_effective_support
    diagnostics.low_support_rejected_events = int((~stable).sum().item())
    normalized = normalized[stable]
    diagnostics.kept_events = int(normalized.shape[0])
    if normalized.numel() == 0:
        return empty_sum, diagnostics

    # Reduce on the active device. Only one pixel vector is returned per batch.
    return torch.sum(normalized, dim=0), diagnostics
