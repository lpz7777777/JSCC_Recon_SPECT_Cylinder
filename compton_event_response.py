"""Shared event filtering and geometric Compton response kernels.

The local reconstruction and Sensi_d calibration must use this module rather
than independently maintained copies of the same event model.  The current
model deliberately preserves the existing cone-times-first-detector-response
approximation; its role is to make the numerical operator identical before
the physical model is extended further.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


FRONT_LAYER_CRYSTAL_MM = (3.0, 3.0, 3.0)
REAR_LAYER_CRYSTAL_MM = (2.0, 6.0, 2.0)
ELECTRON_REST_ENERGY_MEV = 0.511


@dataclass(frozen=True)
class ComptonEventSettings:
    energy_mev: float
    energy_resolution: float
    energy_threshold_max_mev: float
    energy_threshold_min_mev: float
    energy_threshold_sum_mev: float
    delta_r1_mm: float = 0.0
    delta_r2_mm: float = 0.0
    min_event_effective_support: float = 1.0
    include_first_hit_source_leg_uncertainty: bool = True
    require_different_layers: bool = True
    max_min_standardized_arm: float | None = None
    geometry_mode: str = "legacy"

    def __post_init__(self):
        if self.geometry_mode not in ("legacy", "stable_float64"):
            raise ValueError("Unknown Compton geometry mode")


@dataclass
class BatchDiagnostics:
    input_events: int = 0
    invalid_raw_energy_events: int = 0
    energy_rejected_events: int = 0
    kinematic_rejected_events: int = 0
    same_layer_rejected_events: int = 0
    invalid_kernel_events: int = 0
    low_support_rejected_events: int = 0
    mismatch_rejected_events: int = 0
    kept_events: int = 0

    def add_(self, other: "BatchDiagnostics") -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int]) -> "BatchDiagnostics":
        return cls(**{name: int(values.get(name, 0)) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PreparedComptonEvents:
    cpnum1: torch.Tensor
    cpnum2: torch.Tensor
    e1: torch.Tensor
    e2: torch.Tensor
    pos1: torch.Tensor
    pos2: torch.Tensor
    sigma_pos1_sq: torch.Tensor
    sigma_pos2_sq: torch.Tensor
    source_row_indices: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return int(self.e1.numel())


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


def compton_theta_from_e1(e1: torch.Tensor, energy_mev: float) -> torch.Tensor:
    cos_theta = 1.0 - (
        ELECTRON_REST_ENERGY_MEV * e1 / ((energy_mev - e1) * energy_mev)
    )
    return torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))


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


def prepare_compton_events(
    events: torch.Tensor,
    settings: ComptonEventSettings,
    detector_coordinates: torch.Tensor,
    detector_sigma_r1_sq: torch.Tensor,
    detector_sigma_r2_sq: torch.Tensor,
    *,
    input_energies_already_smeared: bool,
    generator: torch.Generator | None = None,
) -> tuple[PreparedComptonEvents | None, BatchDiagnostics]:
    diagnostics = BatchDiagnostics(input_events=int(events.shape[0]))
    if events.numel() == 0:
        return None, diagnostics
    if events.ndim != 2 or events.size(1) < 4:
        raise ValueError("Compton events must have shape [event_count, >=4].")

    _validate_detector_ids(events, int(detector_coordinates.shape[0]))
    cpnum1 = torch.round(events[:, 0]).long()
    cpnum2 = torch.round(events[:, 2]).long()
    e1 = events[:, 1]
    e2 = events[:, 3]
    source_rows = torch.arange(events.shape[0], device=events.device)

    valid_raw_energy = torch.isfinite(e1) & torch.isfinite(e2) & (e1 > 0) & (e2 > 0)
    diagnostics.invalid_raw_energy_events = int((~valid_raw_energy).sum().item())
    source_rows = source_rows[valid_raw_energy]
    cpnum1, cpnum2, e1, e2 = (
        value[valid_raw_energy] for value in (cpnum1, cpnum2, e1, e2)
    )
    if e1.numel() == 0:
        return None, diagnostics

    if not input_energies_already_smeared:
        sigma_1 = e1 * settings.energy_resolution / 2.355 * (settings.energy_mev / e1) ** 0.5
        sigma_2 = e2 * settings.energy_resolution / 2.355 * (settings.energy_mev / e2) ** 0.5
        e1 = e1 + sigma_1 * torch.randn(e1.shape, device=e1.device, generator=generator)
        e2 = e2 + sigma_2 * torch.randn(e2.shape, device=e2.device, generator=generator)

    valid_energy = (
        (e1 < settings.energy_threshold_max_mev)
        & (e1 > settings.energy_threshold_min_mev)
        & (e2 > settings.energy_threshold_min_mev)
        & ((e1 + e2) > settings.energy_threshold_sum_mev)
    )
    diagnostics.energy_rejected_events = int((~valid_energy).sum().item())
    source_rows = source_rows[valid_energy]
    cpnum1, cpnum2, e1, e2 = (
        value[valid_energy] for value in (cpnum1, cpnum2, e1, e2)
    )
    if e1.numel() == 0:
        return None, diagnostics

    cos_theta_raw = 1.0 - (
        ELECTRON_REST_ENERGY_MEV * e1
        / ((settings.energy_mev - e1) * settings.energy_mev)
    )
    valid_kinematics = (
        torch.isfinite(cos_theta_raw)
        & (cos_theta_raw > -1.0 + 1e-6)
        & (cos_theta_raw < 1.0 - 1e-6)
    )
    diagnostics.kinematic_rejected_events = int((~valid_kinematics).sum().item())
    source_rows = source_rows[valid_kinematics]
    cpnum1, cpnum2, e1, e2 = (
        value[valid_kinematics] for value in (cpnum1, cpnum2, e1, e2)
    )
    if e1.numel() == 0:
        return None, diagnostics

    pos1 = detector_coordinates[cpnum1 - 1]
    pos2 = detector_coordinates[cpnum2 - 1]
    sigma_pos1_sq = detector_sigma_r1_sq[cpnum1 - 1]
    sigma_pos2_sq = detector_sigma_r2_sq[cpnum2 - 1]
    if settings.require_different_layers:
        valid_layer = torch.abs(pos1[:, 1] - pos2[:, 1]) > 0.1
        diagnostics.same_layer_rejected_events = int((~valid_layer).sum().item())
        source_rows = source_rows[valid_layer]
        cpnum1, cpnum2, e1, e2, pos1, pos2, sigma_pos1_sq, sigma_pos2_sq = (
            value[valid_layer]
            for value in (cpnum1, cpnum2, e1, e2, pos1, pos2, sigma_pos1_sq, sigma_pos2_sq)
        )
    if e1.numel() == 0:
        return None, diagnostics

    return PreparedComptonEvents(
        cpnum1=cpnum1,
        cpnum2=cpnum2,
        e1=e1,
        e2=e2,
        pos1=pos1,
        pos2=pos2,
        sigma_pos1_sq=sigma_pos1_sq,
        sigma_pos2_sq=sigma_pos2_sq,
        source_row_indices=source_rows,
    ), diagnostics


def _energy_angle_sigma(
    e1: torch.Tensor,
    settings: ComptonEventSettings,
    beta: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    sigma_e = e1 * settings.energy_resolution / 2.355 * (settings.energy_mev / e1) ** 0.5
    e1_low = torch.clamp(e1 - sigma_e, 1e-7, settings.energy_mev - 1e-7)
    e1_high = torch.clamp(e1 + sigma_e, 1e-7, settings.energy_mev - 1e-7)
    theta_low = compton_theta_from_e1(e1_low, settings.energy_mev)
    theta_high = compton_theta_from_e1(e1_high, settings.energy_mev)
    sigma_minus = torch.clamp(theta - theta_low, min=1e-7)
    sigma_plus = torch.clamp(theta_high - theta, min=1e-7)
    delta_theta = beta - theta.unsqueeze(1)
    return torch.where(delta_theta >= 0, sigma_plus.unsqueeze(1), sigma_minus.unsqueeze(1))


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
    grad_pos1 = jac_inter_crystal / sin_beta
    if include_first_hit_source_leg_uncertainty:
        grad_pos1 = grad_pos1 - jac_source_leg / sin_beta
    grad_pos2 = -jac_inter_crystal / sin_beta

    variance = torch.sum(grad_pos1**2 * sigma_pos1_sq.unsqueeze(1), dim=2)
    variance += torch.sum(grad_pos2**2 * sigma_pos2_sq.unsqueeze(1), dim=2)
    return torch.sqrt(torch.clamp(variance, min=1e-12))


def stable_compton_geometry(vector01, vector12, variance1, variance2,
                            include_first_hit_source_leg_uncertainty=True):
    """Stable double-precision angle and position covariance propagation.

    Callers must cast positions before subtraction. Exactly collinear scalar
    angle derivatives have no unique direction; the explicit conservative
    finite bound is used only for sine <= 1e-12.
    """
    a, b, v1, v2 = (x.double() for x in
                    (vector01, vector12, variance1, variance2))
    da = a.norm(dim=2, keepdim=True)
    db = b.norm(dim=2, keepdim=True)
    if bool((da <= 0).any() or (db <= 0).any()):
        raise ValueError("Coincident Compton geometry positions")
    u, v = a / da, b / db
    cross = torch.linalg.cross(u, v.expand_as(u), dim=2)
    sine = cross.norm(dim=2, keepdim=True)
    cosine = (u * v).sum(2, keepdim=True)
    beta = torch.atan2(sine, cosine).squeeze(2)
    normal = cross / sine.clamp_min(1e-12)
    source_tangent = torch.linalg.cross(normal, u, dim=2) / da
    inter_tangent = torch.linalg.cross(v.expand_as(u), normal, dim=2) / db
    grad1 = inter_tangent
    if include_first_hit_source_leg_uncertainty:
        grad1 = grad1 - source_tangent
    grad2 = -inter_tangent
    variance = (grad1.square() * v1[:, None]).sum(2)
    variance += (grad2.square() * v2[:, None]).sum(2)
    first_bound = 1 / db.squeeze(2)
    if include_first_hit_source_leg_uncertainty:
        first_bound = first_bound + 1 / da.squeeze(2)
    upper = v1.amax(1)[:, None] * first_bound.square()
    upper += v2.amax(1)[:, None] / db.squeeze(2).square()
    variance = torch.where(sine.squeeze(2) <= 1e-12, upper, variance)
    if not bool(torch.isfinite(beta).all() and torch.isfinite(variance).all()):
        raise ValueError("Nonfinite stable Compton geometry")
    if bool((variance > upper * (1 + 1e-10)).any()):
        raise ValueError("Compton position derivative bound exceeded")
    return beta, variance.clamp_min(1e-12).sqrt(), upper.sqrt()


def _stable_event_angles(prepared, coordinates, settings):
    vector01 = prepared.pos1.double()[:, None] - coordinates.double()[None]
    vector12 = (prepared.pos2.double() - prepared.pos1.double())[:, None]
    beta, sigma_position, _ = stable_compton_geometry(
        vector01, vector12, prepared.sigma_pos1_sq, prepared.sigma_pos2_sq,
        settings.include_first_hit_source_leg_uncertainty)
    theta = compton_theta_from_e1(prepared.e1.double(), settings.energy_mev)
    sigma_energy = _energy_angle_sigma(prepared.e1.double(), settings, beta, theta)
    sigma = (sigma_energy.square() + sigma_position.square()).clamp_min(1e-12).sqrt()
    return beta, theta, sigma


def build_compton_cone_weights(
    prepared: PreparedComptonEvents,
    voxel_coordinates: torch.Tensor,
    settings: ComptonEventSettings,
) -> torch.Tensor:
    if settings.geometry_mode == "stable_float64":
        beta, theta, sigma = _stable_event_angles(prepared, voxel_coordinates, settings)
        e1 = prepared.e1.double()
        kn = settings.energy_mev / (settings.energy_mev - e1)
        kn += (settings.energy_mev - e1) / settings.energy_mev
        weights = torch.exp(-0.5 * ((beta - theta[:, None]) / sigma).square())
        weights *= kn[:, None] - beta.sin().square()
        if not bool(torch.isfinite(weights).all() and (weights >= 0).all()):
            raise ValueError("Invalid stable Compton cone")
        return weights.to(prepared.e1.dtype)
    vector01 = prepared.pos1.unsqueeze(1) - voxel_coordinates.unsqueeze(0)
    vector12 = (prepared.pos2 - prepared.pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)
    theta = compton_theta_from_e1(prepared.e1, settings.energy_mev)
    klein_nishina = settings.energy_mev / (settings.energy_mev - prepared.e1)
    klein_nishina += (settings.energy_mev - prepared.e1) / settings.energy_mev
    beta_cos = torch.sum(vector01 * vector12, dim=2) / torch.clamp(distance01 * distance12, min=1e-7)
    beta = torch.acos(torch.clamp(beta_cos, -1.0 + 1e-7, 1.0 - 1e-7))

    sigma_energy = _energy_angle_sigma(prepared.e1, settings, beta, theta)
    sigma_position = _position_angle_sigma(
        vector01,
        vector12,
        prepared.sigma_pos1_sq,
        prepared.sigma_pos2_sq,
        settings.include_first_hit_source_leg_uncertainty,
    )
    sigma_angle = torch.sqrt(torch.clamp(sigma_energy**2 + sigma_position**2, min=1e-12))
    weights = torch.exp(-((beta - theta.unsqueeze(1)) ** 2) / (2.0 * sigma_angle**2))
    weights *= klein_nishina.unsqueeze(1) - torch.sin(beta) ** 2
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_event_response(
    cone_weights: torch.Tensor,
    cpnum1: torch.Tensor,
    system_matrix_density: torch.Tensor,
    min_event_effective_support: float,
    *,
    min_standardized_arm: torch.Tensor | None = None,
    max_min_standardized_arm: float | None = None,
    diagnostics: BatchDiagnostics | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return normalized rows for the density-basis response K_i * B[c1_i,:]."""
    point_response = torch.index_select(system_matrix_density, 0, cpnum1 - 1)
    weights = cone_weights * point_response
    row_sums = torch.sum(weights, dim=1)
    valid = torch.isfinite(weights).all(dim=1) & torch.isfinite(row_sums) & (row_sums > 0)
    invalid_count = int((~valid).sum().item())
    weights = weights[valid]
    cpnum1 = cpnum1[valid]
    if weights.numel() == 0:
        return weights, cpnum1, valid, invalid_count, 0

    normalized = weights / torch.sum(weights, dim=1, keepdim=True)
    stable, low_support_count, mismatch_count = select_normalized_response_rows(
        normalized, min_event_effective_support,
        None if min_standardized_arm is None else min_standardized_arm[valid],
        max_min_standardized_arm)
    if diagnostics is not None:
        diagnostics.mismatch_rejected_events = mismatch_count
    return normalized[stable], cpnum1[stable], valid, invalid_count, low_support_count


def min_standardized_compton_arm(prepared, voxel_coordinates, settings):
    """Best angular compatibility over the complete calculation grid, before B.

    Uses exactly the production angle/uncertainty expressions. No phantom,
    activity, sensitivity or reconstruction enters this quality criterion.
    """
    if settings.geometry_mode == "stable_float64":
        beta, theta, sigma = _stable_event_angles(prepared, voxel_coordinates, settings)
        score = ((beta - theta[:, None]).abs() / sigma).amin(dim=1)
        if not bool(torch.isfinite(score).all()):
            raise ValueError("Nonfinite stable complete-grid Compton ARM score")
        return score
    vector01 = prepared.pos1.unsqueeze(1) - voxel_coordinates.unsqueeze(0)
    vector12 = (prepared.pos2 - prepared.pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)
    theta = compton_theta_from_e1(prepared.e1, settings.energy_mev)
    beta_cos = torch.sum(vector01 * vector12, dim=2) / torch.clamp(distance01 * distance12, min=1e-7)
    beta = torch.acos(torch.clamp(beta_cos, -1.0 + 1e-7, 1.0 - 1e-7))
    sigma_energy = _energy_angle_sigma(prepared.e1, settings, beta, theta)
    sigma_position = _position_angle_sigma(vector01, vector12,
        prepared.sigma_pos1_sq, prepared.sigma_pos2_sq,
        settings.include_first_hit_source_leg_uncertainty)
    sigma = torch.sqrt(torch.clamp(sigma_energy**2 + sigma_position**2, min=1e-12))
    score = ((beta-theta.unsqueeze(1)).abs()/sigma).amin(dim=1)
    if not bool(torch.isfinite(score).all()):
        raise ValueError("Nonfinite complete-grid Compton ARM score")
    return score


def select_normalized_response_rows(normalized, min_support, min_arm=None, max_min_arm=None):
    """Apply original support selection first, then the optional mismatch cut."""
    effective_support = 1.0 / torch.sum(normalized**2, dim=1)
    baseline = effective_support >= min_support
    low_count = int((~baseline).sum().item())
    if max_min_arm is None:
        return baseline, low_count, 0
    if max_min_arm <= 0 or min_arm is None or min_arm.shape != baseline.shape:
        raise ValueError("Mismatch selection requires positive threshold and aligned scores")
    if not bool(torch.isfinite(min_arm).all()):
        raise ValueError("Nonfinite Compton mismatch score")
    mismatch = min_arm > max_min_arm
    return baseline & ~mismatch, low_count, int((baseline & mismatch).sum().item())
