"""Sparse storage adapter for the shared density-basis Compton response."""

import torch

from compton_event_response import (
    ComptonEventSettings,
    build_compton_cone_weights,
    build_detector_position_variance,
    prepare_compton_events,
)
from compton_sparse_ops import pack_sparse_event_rows, reduce_fine_rows_to_coarse


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
    input_energies_already_smeared=False,
):
    """Filter events and store their cone component on the selected sparse grid.

    The density-basis system-matrix multiplication and row normalization are
    repeated by ``materialize_sparse_event_rows_to_fine`` during MLEM.  The
    same multiplication is used here only to apply an identical support test.
    """
    del device
    if model_compton_generator is not None:
        raise NotImplementedError("model_compton_generator is not supported in sparse Compton mode.")

    settings = ComptonEventSettings(
        energy_mev=e0,
        energy_resolution=ene_resolution,
        energy_threshold_max_mev=ene_threshold_max,
        energy_threshold_min_mev=ene_threshold_min,
        energy_threshold_sum_mev=ene_threshold_sum,
        delta_r1_mm=delta_r1,
        delta_r2_mm=delta_r2,
    )
    detector_sigma_r1_sq = build_detector_position_variance(detector, delta_r1)
    detector_sigma_r2_sq = build_detector_position_variance(detector, delta_r2)
    prepared, _ = prepare_compton_events(
        list_origin,
        settings,
        detector,
        detector_sigma_r1_sq,
        detector_sigma_r2_sq,
        input_energies_already_smeared=input_energies_already_smeared,
    )
    if prepared is None:
        return torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32), None, None

    cone_coarse = build_compton_cone_weights(prepared, sparse_projector.coor_coarse, settings)
    response_coarse = reduce_fine_rows_to_coarse(sysmat[prepared.cpnum1 - 1, :], sparse_projector)
    raw = cone_coarse * response_coarse
    row_sums = raw.sum(dim=1)
    valid = torch.isfinite(raw).all(dim=1) & torch.isfinite(row_sums) & (row_sums > 0)
    if not bool(torch.any(valid)):
        return torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32), None, None

    # Match the fine-grid MLEM support criterion when full-grid mode is used.
    normalized = raw[valid] / raw[valid].sum(dim=1, keepdim=True)
    support = 1.0 / torch.sum(normalized**2, dim=1)
    stable = support >= settings.min_event_effective_support
    cpnum1 = prepared.cpnum1[valid][stable]
    cone_coarse = cone_coarse[valid][stable]
    if cone_coarse.numel() == 0:
        return torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32), None, None
    return pack_sparse_event_rows(cpnum1, cone_coarse).cpu(), None, None


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
        parts = []
        for sub_chunk in torch.chunk(list_origin_chunk, num_workers, dim=0):
            if sub_chunk.numel() == 0:
                continue
            rows, _, _ = get_compton_backproj_list_single_sparse(
                sysmat, detector, sparse_projector, sub_chunk.to(device),
                delta_r1, delta_r2, e0, ene_resolution, ene_threshold_max,
                ene_threshold_min, ene_threshold_sum, device,
                model_compton_generator=model_compton_generator,
            )
            if rows.numel() > 0:
                parts.append(rows)
        result_dict[rank] = (
            torch.cat(parts, dim=0)
            if parts
            else torch.empty((0, sparse_projector.coarse_pixel_num + 1), dtype=torch.float32)
        )
