import torch

from compton_sparse_ops import pack_sparse_event_rows, reduce_fine_rows_to_coarse
from process_list_plane_strict import (
    ELECTRON_REST_MEV,
    MIN_EVENT_EFFECTIVE_SUPPORT,
    _build_detector_pos_sigma_sq,
    _compton_theta_from_e1,
    _compute_angle_sigma_ene_strict,
    _compute_angle_sigma_pos_strict,
    _filter_kinematically_valid_events,
)


def _filter_unstable_event_kernels_sparse(t, t_compton, t_single):
    t_norm = t / t.sum(dim=1, keepdim=True)
    t_compton_norm = t_compton / t_compton.sum(dim=1, keepdim=True)
    t_single_norm = t_single / t_single.sum(dim=1, keepdim=True)

    effective_support = 1.0 / torch.sum(t_norm ** 2, dim=1)
    stable = effective_support >= MIN_EVENT_EFFECTIVE_SUPPORT
    return t_norm[stable], t_compton_norm[stable], t_single_norm[stable], stable


def get_compton_backproj_list_single_sparse(
    sysmat,
    detector,
    sparse_projector,
    list_origin,
    delta_r1,
    delta_r2,
    e0,
    ene_resolution,
    ene_threshold_max,
    ene_threshold_min,
    ene_threshold_sum,
    device,
    model_compton_generator=None,
):
    cpnum1 = list_origin[:, 0].int()
    cpnum2 = list_origin[:, 2].int()
    e1 = list_origin[:, 1]
    e2 = list_origin[:, 3]

    sigma_1 = e1 * ene_resolution / 2.355 * (e0 / e1) ** 0.5
    sigma_2 = e2 * ene_resolution / 2.355 * (e0 / e2) ** 0.5
    e1 = e1 + sigma_1 * torch.randn(e1.shape[0], device=device)
    e2 = e2 + sigma_2 * torch.randn(e2.shape[0], device=device)

    flag = (e1 < ene_threshold_max) & (e1 > ene_threshold_min) & (e2 > ene_threshold_min) & ((e1 + e2) > ene_threshold_sum)
    cpnum1 = cpnum1[flag]
    cpnum2 = cpnum2[flag]
    e1 = e1[flag]
    e2 = e2[flag]
    cpnum1, cpnum2, e1, e2 = _filter_kinematically_valid_events(cpnum1, cpnum2, e1, e2, e0, ELECTRON_REST_MEV)

    detector_pos = detector[:, :3]
    detector_sigma_r1_sq = _build_detector_pos_sigma_sq(detector, delta_r1)
    detector_sigma_r2_sq = _build_detector_pos_sigma_sq(detector, delta_r2)

    pos1 = detector_pos[cpnum1 - 1, :]
    pos2 = detector_pos[cpnum2 - 1, :]
    sigma_pos1_sq = detector_sigma_r1_sq[cpnum1 - 1, :]
    sigma_pos2_sq = detector_sigma_r2_sq[cpnum2 - 1, :]
    flag = torch.abs(pos1[:, 1] - pos2[:, 1]) > 0.1

    cpnum1 = cpnum1[flag]
    e1 = e1[flag]
    pos1 = pos1[flag]
    pos2 = pos2[flag]
    sigma_pos1_sq = sigma_pos1_sq[flag]
    sigma_pos2_sq = sigma_pos2_sq[flag]

    if cpnum1.numel() == 0:
        empty = torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
        return empty, None, None

    vector01 = pos1.unsqueeze(1) - sparse_projector.coor_coarse.unsqueeze(0)
    vector12 = (pos2 - pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)

    theta = _compton_theta_from_e1(e1, e0, ELECTRON_REST_MEV)
    klein_nishina = e0 / (e0 - e1) + (e0 - e1) / e0
    beta_cos = (vector01 * vector12).sum(2) / torch.clamp(distance01 * distance12, min=1e-7)
    beta = torch.acos(torch.clamp(beta_cos, -1.0 + 1e-7, 1.0 - 1e-7))

    angle_sigma_ene = _compute_angle_sigma_ene_strict(e1, e0, ene_resolution, ELECTRON_REST_MEV, beta, theta)
    angle_sigma_pos = _compute_angle_sigma_pos_strict(vector01, vector12, beta, sigma_pos1_sq, sigma_pos2_sq)
    angle_sigma = torch.sqrt(torch.clamp(angle_sigma_pos ** 2 + angle_sigma_ene ** 2, min=1e-12))

    t_compton = torch.exp(-((beta - theta.unsqueeze(-1)) ** 2) / (2 * angle_sigma ** 2))
    t_compton = t_compton * (klein_nishina.unsqueeze(-1) - torch.sin(beta) ** 2)

    if model_compton_generator is not None:
        raise NotImplementedError("model_compton_generator is not supported in sparse Compton mode.")

    t_single = reduce_fine_rows_to_coarse(sysmat[cpnum1 - 1, :], sparse_projector)
    t = t_compton * t_single

    flag_nan = torch.isnan(t).sum(dim=1)
    flag_zero = t.sum(dim=1) == 0
    valid = (flag_nan + flag_zero) == 0
    cpnum1 = cpnum1[valid]
    t = t[valid, :]
    t_compton = t_compton[valid, :]
    t_single = t_single[valid, :]

    if t.size(0) == 0:
        empty = torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
        return empty, None, None

    t, t_compton, t_single, stable = _filter_unstable_event_kernels_sparse(t, t_compton, t_single)
    cpnum1 = cpnum1[stable]
    event_rows = pack_sparse_event_rows(cpnum1, t_compton)
    return event_rows.cpu(), None, None


def get_compton_backproj_list_mp_sparse(
    rank,
    world_size,
    sysmat,
    detector,
    sparse_projector,
    list_origin_chunk,
    delta_r1,
    delta_r2,
    e0,
    ene_resolution,
    ene_threshold_max,
    ene_threshold_min,
    ene_threshold_sum,
    result_dict,
    num_workers,
    start_time,
    flag_save_t,
    model_compton_generator=None,
):
    del world_size, start_time, flag_save_t
    with torch.no_grad():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        sysmat = sysmat.to(device)
        detector = detector.to(device)
        sparse_projector = sparse_projector.to(device)

        sub_chunks = torch.chunk(list_origin_chunk, num_workers, dim=0)
        t_parts = []
        for sub_chunk in sub_chunks:
            if sub_chunk.numel() == 0:
                continue
            t_chunk, _, _ = get_compton_backproj_list_single_sparse(
                sysmat,
                detector,
                sparse_projector,
                sub_chunk.to(device),
                delta_r1,
                delta_r2,
                e0,
                ene_resolution,
                ene_threshold_max,
                ene_threshold_min,
                ene_threshold_sum,
                device,
                model_compton_generator=model_compton_generator,
            )
            if t_chunk.numel() > 0:
                t_parts.append(t_chunk)

        if t_parts:
            result_dict[rank] = torch.cat(t_parts, dim=0)
        else:
            result_dict[rank] = torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
