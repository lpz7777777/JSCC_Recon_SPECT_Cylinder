import torch


FRONT_LAYER_CRYSTAL_MM = (3.0, 3.0, 3.0)
REAR_LAYER_CRYSTAL_MM = (2.0, 6.0, 2.0)
ELECTRON_REST_MEV = 0.511
MIN_EVENT_EFFECTIVE_SUPPORT = 50.0


def _compton_theta_from_e1(e1, e0, ee):
    cos_theta = 1 - ((ee * e1) / ((e0 - e1) * e0))
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_theta)


def _filter_kinematically_valid_events(cpnum1, cpnum2, e1, e2, e0, ee):
    cos_theta_raw = 1 - ((ee * e1) / ((e0 - e1) * e0))
    # Energy smearing can push a small fraction of events outside the physical
    # Compton domain. If they are kept, theta gets clamped to pi and the strict
    # sigma_ene collapses, which produces isolated hot voxels after normalization.
    valid = (cos_theta_raw > -1.0 + 1e-6) & (cos_theta_raw < 1.0 - 1e-6)
    return cpnum1[valid], cpnum2[valid], e1[valid], e2[valid]


def _compute_angle_sigma_ene_strict(e1, e0, ene_resolution, ee, beta, theta):
    sigma_e = e1 * ene_resolution / 2.355 * (e0 / e1) ** 0.5
    e_eps = 1e-7
    e1_low = torch.clamp(e1 - sigma_e, e_eps, e0 - e_eps)
    e1_high = torch.clamp(e1 + sigma_e, e_eps, e0 - e_eps)

    theta_low = _compton_theta_from_e1(e1_low, e0, ee)
    theta_high = _compton_theta_from_e1(e1_high, e0, ee)

    sigma_minus = torch.clamp(theta - theta_low, min=1e-7)
    sigma_plus = torch.clamp(theta_high - theta, min=1e-7)

    delta_theta = beta - theta.unsqueeze(-1)
    return sigma_plus.unsqueeze(-1) * (delta_theta >= 0).float() + sigma_minus.unsqueeze(-1) * (delta_theta < 0).float()


def _uniform_sigma_sq_from_size(size_mm):
    return (size_mm ** 2) / 12.0


def _build_detector_pos_sigma_sq(detector, extra_sigma):
    detector_pos = detector[:, :3]
    detector_y_abs = torch.abs(detector_pos[:, 1])
    layer_y_abs = torch.sort(torch.unique(detector_y_abs))[0]

    if layer_y_abs.numel() != 4:
        fallback_sigma_sq = torch.full(
            (detector_pos.size(0), 3),
            fill_value=max(extra_sigma, 0.0) ** 2,
            dtype=detector_pos.dtype,
            device=detector_pos.device,
        )
        return fallback_sigma_sq

    sigma_sq = torch.zeros((detector_pos.size(0), 3), dtype=detector_pos.dtype, device=detector_pos.device)
    front_sigma_sq = torch.tensor(
        [
            _uniform_sigma_sq_from_size(FRONT_LAYER_CRYSTAL_MM[0]),
            _uniform_sigma_sq_from_size(FRONT_LAYER_CRYSTAL_MM[1]),
            _uniform_sigma_sq_from_size(FRONT_LAYER_CRYSTAL_MM[2]),
        ],
        dtype=detector_pos.dtype,
        device=detector_pos.device,
    )
    rear_sigma_sq = torch.tensor(
        [
            _uniform_sigma_sq_from_size(REAR_LAYER_CRYSTAL_MM[0]),
            _uniform_sigma_sq_from_size(REAR_LAYER_CRYSTAL_MM[1]),
            _uniform_sigma_sq_from_size(REAR_LAYER_CRYSTAL_MM[2]),
        ],
        dtype=detector_pos.dtype,
        device=detector_pos.device,
    )

    for layer_idx in range(4):
        layer_mask = detector_y_abs == layer_y_abs[layer_idx]
        sigma_sq[layer_mask, :] = front_sigma_sq if layer_idx < 3 else rear_sigma_sq

    if extra_sigma > 0:
        sigma_sq = sigma_sq + (extra_sigma ** 2)

    return sigma_sq


def _compute_angle_sigma_pos_strict(vector01, vector12, sigma_pos1_sq, sigma_pos2_sq):
    distance01 = torch.norm(vector01, dim=2, keepdim=True)
    distance12 = torch.norm(vector12, dim=2, keepdim=True)

    unit01 = vector01 / torch.clamp(distance01, min=1e-7)
    unit12 = vector12 / torch.clamp(distance12, min=1e-7)

    cos_beta = torch.sum(unit01 * unit12, dim=2, keepdim=True)
    sin_beta = torch.sqrt(torch.clamp(1.0 - cos_beta ** 2, min=1e-12))

    jac_unit01_unit12 = (unit12 - cos_beta * unit01) / torch.clamp(distance01, min=1e-7)
    jac_unit12_unit01 = (unit01 - cos_beta * unit12) / torch.clamp(distance12, min=1e-7)

    grad_pos1 = -(jac_unit01_unit12 - jac_unit12_unit01) / sin_beta
    grad_pos2 = -jac_unit12_unit01 / sin_beta

    sigma_beta_pos_sq = torch.sum(grad_pos1 ** 2 * sigma_pos1_sq.unsqueeze(1), dim=2)
    sigma_beta_pos_sq = sigma_beta_pos_sq + torch.sum(grad_pos2 ** 2 * sigma_pos2_sq.unsqueeze(1), dim=2)

    return torch.sqrt(torch.clamp(sigma_beta_pos_sq, min=1e-12))


def _filter_unstable_event_kernels(t):
    t_norm = t / t.sum(dim=1, keepdim=True)

    # Rare events can still collapse onto a handful of voxels after combining
    # the Compton kernel with the system matrix. They contribute little to the
    # statistics but leave visible point/short-line artifacts in the image.
    effective_support = 1.0 / torch.sum(t_norm ** 2, dim=1)
    stable = effective_support >= MIN_EVENT_EFFECTIVE_SUPPORT

    return t_norm[stable]


def get_compton_backproj_list(list_origin, compton_arg, detector, coor_polar, sysmat, device):
    cpnum1 = list_origin[:, 0].int()
    cpnum2 = list_origin[:, 2].int()
    e1 = list_origin[:, 1]
    e2 = list_origin[:, 3]

    sigma_1 = e1 * compton_arg.ene_resolution / 2.355 * (compton_arg.e0 / e1) ** 0.5
    sigma_2 = e2 * compton_arg.ene_resolution / 2.355 * (compton_arg.e0 / e2) ** 0.5
    e1 += sigma_1 * torch.randn(e1.shape[0]).to(device)
    e2 += sigma_2 * torch.randn(e2.shape[0]).to(device)

    ene_threshold_sum = getattr(compton_arg, "ene_threshold_sum", 0.40)

    flag_max_1 = e1 < compton_arg.ene_threshold_max
    flag_min_1 = e1 > compton_arg.ene_threshold_min
    flag_min_2 = e2 > compton_arg.ene_threshold_min
    flag_sum = (e1 + e2) > ene_threshold_sum

    flag = flag_max_1 * flag_min_1 * flag_min_2 * flag_sum
    cpnum1 = cpnum1[flag]
    cpnum2 = cpnum2[flag]
    e1 = e1[flag]
    e2 = e2[flag]
    cpnum1, cpnum2, e1, e2 = _filter_kinematically_valid_events(
        cpnum1, cpnum2, e1, e2, compton_arg.e0, ELECTRON_REST_MEV
    )

    detector_pos = detector[:, :3]
    detector_sigma_r1_sq = _build_detector_pos_sigma_sq(detector, getattr(compton_arg, "delta_r1", 0.0))
    detector_sigma_r2_sq = _build_detector_pos_sigma_sq(detector, getattr(compton_arg, "delta_r2", 0.0))

    pos1 = detector_pos[cpnum1 - 1, :]
    pos2 = detector_pos[cpnum2 - 1, :]
    sigma_pos1_sq = detector_sigma_r1_sq[cpnum1 - 1, :]
    sigma_pos2_sq = detector_sigma_r2_sq[cpnum2 - 1, :]
    flag = abs(pos1[:, 1] - pos2[:, 1]) > 0.1

    cpnum1 = cpnum1[flag]
    cpnum2 = cpnum2[flag]
    e1 = e1[flag]
    e2 = e2[flag]
    pos1 = pos1[flag]
    pos2 = pos2[flag]
    sigma_pos1_sq = sigma_pos1_sq[flag]
    sigma_pos2_sq = sigma_pos2_sq[flag]
    ee = ELECTRON_REST_MEV

    vector01 = pos1.unsqueeze(1) - coor_polar.unsqueeze(0)
    vector12 = (pos2 - pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)

    theta = _compton_theta_from_e1(e1, compton_arg.e0, ee)
    klein_nishina = compton_arg.e0 / (compton_arg.e0 - e1) + (compton_arg.e0 - e1) / compton_arg.e0

    beta_cos = (vector01 * vector12).sum(2) / torch.clamp(distance01 * distance12, min=1e-7)
    beta = torch.acos(torch.clamp(beta_cos, -1.0 + 1e-7, 1.0 - 1e-7))

    angle_sigma_ene = _compute_angle_sigma_ene_strict(e1, compton_arg.e0, compton_arg.ene_resolution, ee, beta, theta)
    angle_sigma_pos = _compute_angle_sigma_pos_strict(vector01, vector12, sigma_pos1_sq, sigma_pos2_sq)
    angle_sigma = torch.sqrt(torch.clamp(angle_sigma_pos ** 2 + angle_sigma_ene ** 2, min=1e-12))

    t = torch.exp(- (beta - theta.unsqueeze(-1)) ** 2 / (2 * angle_sigma ** 2)) * (
        klein_nishina.unsqueeze(-1) - torch.sin(beta) ** 2
    )

    t = t * sysmat[cpnum1 - 1, :]
    flag_nan = torch.isnan(t).sum(dim=1)
    flag_zero = (t.sum(dim=1) == 0)
    t = t[(flag_nan + flag_zero) == 0, :]
    t = _filter_unstable_event_kernels(t).cpu()

    return t
