"""Multi-node, multi-GPU version of the current six-output 218/440 chain."""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from compton_sparse_ops import build_compton_sparse_projector
from detector_csv import load_detector_coordinates
from fov_config import factor_geometry, load_config, validate_factor_geometry, validate_sensitivity_provenance
from main_local_multi_energy_cntstat import load_cross_factor, load_factors, load_projections
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from reconstruction import (
    forward_project_shard,
    materialize_local_event_blocks,
    run_compton_and_joint_mlem_dist,
    run_single_mlem_dist,
    save_result,
)


DATASET = "ContrastPhantom_DualEnergy_10_30_240_30_225Ac"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed six-output JSCC 218/440 single+Compton reconstruction."
    )
    parser.add_argument("--count-level", default="1e10")
    parser.add_argument("--data-file-name", default=DATASET)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-step", type=int, default=50)
    parser.add_argument("--rotate-num", type=int, default=20)
    parser.add_argument("--energy-resolution-fwhm", type=float, default=0.13)
    parser.add_argument("--energy-resolution-reference-kev", type=float, default=511.0)
    parser.add_argument("--energy-threshold-sum-mev", type=float, default=0.350)
    parser.add_argument("--theta-stride", type=int, default=1)
    parser.add_argument("--z-stride", type=int, default=1)
    parser.add_argument("--event-preprocess-chunks", type=int, default=16)
    parser.add_argument("--materialize-block-events", type=int, default=512)
    parser.add_argument("--materialize-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-events-per-view", type=int, default=0)
    parser.add_argument("--cross-talk-scale", type=float, default=1.0)
    parser.add_argument("--factors-dir", default="Factors")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--cntstat-dir", default="CntStat")
    parser.add_argument("--list-dir", default="List")
    parser.add_argument(
        "--output-dir",
        default="Results/Reconstruction/Distributed_JSCC_ComptonValidation_Geant4_1e10_Iter1000",
    )
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.iterations <= 0 or args.save_step <= 0 or args.iterations % args.save_step:
        raise ValueError("iterations must be positive and divisible by save-step")
    if args.rotate_num != 20:
        raise ValueError("The current Factors and Geant4 chain require rotate-num=20")
    if args.theta_stride <= 0 or args.z_stride <= 0:
        raise ValueError("theta-stride and z-stride must be positive")
    if args.event_preprocess_chunks <= 0 or args.materialize_block_events <= 0:
        raise ValueError("event chunk sizes must be positive")


def setup_distributed(backend):
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"Launch with torchrun; missing environment variables: {missing}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size, device


def detector_bin_bounds(total_bins, rank, world_size):
    start = total_bins * rank // world_size
    end = total_bins * (rank + 1) // world_size
    if start == end:
        raise ValueError(f"world_size={world_size} exceeds detector-bin count={total_bins}")
    return start, end


def read_csv_byte_partition(path, rank, world_size, usecols=(0, 1, 2, 3), limit=0):
    """Read a non-overlapping complete-line byte range from a headerless CSV."""
    path = Path(path)
    file_size = path.stat().st_size
    raw_start = file_size * rank // world_size
    raw_end = file_size * (rank + 1) // world_size
    with path.open("rb") as handle:
        if raw_start > 0:
            handle.seek(raw_start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        start = handle.tell()
        chunks = []
        while handle.tell() < raw_end:
            line = handle.readline()
            if not line:
                break
            chunks.append(line)
            if limit and len(chunks) >= limit:
                break
        end = handle.tell()
    if not chunks:
        return np.empty((0, len(usecols)), dtype=np.float32), start, end
    values = np.loadtxt(io.BytesIO(b"".join(chunks)), delimiter=",", usecols=usecols, dtype=np.float32, ndmin=2)
    return np.ascontiguousarray(values, dtype=np.float32), start, end


def load_local_list_views(list_dir, rotate_num, rank, world_size, limit):
    views = []
    byte_ranges = []
    for view in range(rotate_num):
        path = list_dir / f"{view + 1}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        values, start, end = read_csv_byte_partition(
            path, rank, world_size, limit=limit
        )
        views.append(torch.from_numpy(values))
        byte_ranges.append({"view": view + 1, "start": start, "end": end, "rows": len(values)})
        print(f"[rank {rank}] List view {view + 1}: rows={len(values):,}", flush=True)
    return views, byte_ranges


def save_sum(output_dir, name, *images):
    values = np.sum(
        [image.detach().cpu().numpy().reshape(-1).astype(np.float32) for image in images],
        axis=0,
        dtype=np.float32,
    )
    values.tofile(Path(output_dir) / f"Image_{name}")
    return values


def all_gather_equal_shape_tensor(local_tensor, world_size):
    """All-gather a small tensor using the NCCL/GLOO common collective set."""
    gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    return gathered


def main():
    args = parse_args()
    validate_args(args)
    factor_root = REPO_ROOT / args.factors_dir
    coords, points_per_layer, z_layers = factor_geometry(factor_root / "440keV_RotateNum20")
    if args.experiment_config:
        cfg = load_config(args.experiment_config)
        validate_factor_geometry({name: factor_root / folder for name, folder in (
            ("218", "218keV_RotateNum20"), ("440", "440keV_RotateNum20"),
            ("cross", "440keV_to218win_RotateNum20"))}, cfg)
        validate_sensitivity_provenance(factor_root / "440keV_RotateNum20")
        if (args.energy_resolution_fwhm, args.energy_resolution_reference_kev,
                args.energy_threshold_sum_mev) != (.13, 511, .350):
            raise ValueError("Experiment requires the frozen Sensi_d physics settings")
        if args.theta_stride != 1 or args.z_stride != 1:
            raise ValueError("Experiment production requires the full Compton grid")
    rank, local_rank, world_size, device = setup_distributed(args.backend)
    torch.manual_seed(20260728 + rank)
    np.random.seed(20260728 + rank)

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"world_size={world_size}, output={output_dir}", flush=True)
    dist.barrier()

    loader_args = argparse.Namespace(
        e0_list=[0.218, 0.440], intensity_list=[1.0, 1.0], rotate_num=20,
        pixel_num_layer=points_per_layer, pixel_num_z=z_layers, factor_dir_suffix="",
        cross_talk_scale=args.cross_talk_scale, cntstat_dir_suffix="_Geant4JSCC",
        data_file_name=args.data_file_name, ds=1.0, osem_subset_num=1,
        seed=20260728, overwrite_existing=True,
    )
    factors, pixel_num = load_factors(loader_args, REPO_ROOT / args.factors_dir)
    cross = load_cross_factor(loader_args, REPO_ROOT / args.factors_dir, factors, pixel_num)
    projections, input_files, total_counts = load_projections(
        loader_args, REPO_ROOT / args.cntstat_dir, factors, args.count_level, 0
    )
    factor218 = next(item for item in factors if round(item["e0"] * 1000) == 218)
    factor440 = next(item for item in factors if round(item["e0"] * 1000) == 440)
    total_bins = factor440["total_bins"]
    bin_start, bin_end = detector_bin_bounds(total_bins, rank, world_size)

    list_dir = (
        REPO_ROOT / args.list_dir / "218-440keV_RotateNum20_Geant4JSCC" /
        f"List_{args.data_file_name}_{args.count_level}"
    )
    if not list_dir.is_dir():
        raise FileNotFoundError(list_dir)
    if args.dry_run:
        if rank == 0:
            print(
                f"DRY RUN OK: pixels={pixel_num}, bins={total_bins}, "
                f"List={list_dir}, Sensi_d={factor440['factor_dir'] / 'Sensi_d'}"
            )
        return

    rotmat = factor440["rotmat"].to(device)
    rotmat_inv = factor440["rotmat_inv"].to(device)
    sys218 = factor218["sysmat"][bin_start:bin_end].to(device)
    sys440 = factor440["sysmat"][bin_start:bin_end].to(device)
    sys_cross = cross["sysmat"][bin_start:bin_end].to(device)
    proj218 = projections[0][bin_start:bin_end].to(device)
    proj440 = projections[1][bin_start:bin_end].to(device)
    sensi218 = factor218["sensi"].to(device)
    sensi440 = factor440["sensi"].to(device)
    sys440_full = factor440["sysmat"].to(device)

    sensi_d_path = factor440["factor_dir"] / "Sensi_d"
    sensi_d = torch.from_numpy(
        np.fromfile(sensi_d_path, dtype=np.float32).reshape(pixel_num, 1).copy()
    ).to(device)
    if not torch.isfinite(sensi_d).all() or torch.any(sensi_d <= 0):
        raise ValueError(f"Invalid installed Sensi_d: {sensi_d_path}")

    result440 = run_single_mlem_dist(
        "440_SinglePhoton", sys440, proj440, rotmat, rotmat_inv, sensi440,
        args.iterations, args.save_step, rank,
    )
    predicted_cross = forward_project_shard(sys_cross, rotmat, result440.image)
    max_bins_per_rank = (total_bins + world_size - 1) // world_size
    predicted_cross_padded = torch.zeros(
        (max_bins_per_rank, args.rotate_num), dtype=predicted_cross.dtype, device=device
    )
    predicted_cross_padded[:predicted_cross.size(0)] = predicted_cross
    predicted_parts = all_gather_equal_shape_tensor(
        predicted_cross_padded, world_size
    )
    result218 = run_single_mlem_dist(
        "218_SinglePhoton_CrossTalkCorrected", sys218, proj218, rotmat,
        rotmat_inv, sensi218, args.iterations, args.save_step, rank,
        additive_background=predicted_cross,
    )
    save_result(output_dir, "440_SinglePhoton", result440, args.iterations, args.save_step, rank)
    save_result(output_dir, "218_SinglePhoton_CrossTalkCorrected", result218, args.iterations, args.save_step, rank)

    local_views, byte_ranges = load_local_list_views(
        list_dir, args.rotate_num, rank, world_size, args.max_events_per_view
    )
    range_tensor = torch.tensor(
        [[item["start"], item["end"], item["rows"]] for item in byte_ranges],
        dtype=torch.int64,
        device=device,
    )
    gathered_ranges = all_gather_equal_shape_tensor(range_tensor, world_size)
    detector = torch.from_numpy(
        load_detector_coordinates(factor440["factor_dir"] / "Detector.csv", expected_count=total_bins)
    ).to(device)
    coordinates = torch.from_numpy(
        np.loadtxt(factor440["factor_dir"] / "coor_polar_full.csv", delimiter=",", dtype=np.float32)
    )
    projector = build_compton_sparse_projector(
        coordinates, theta_stride=args.theta_stride, z_stride=args.z_stride,
        rotate_num=args.rotate_num, dtype=torch.float32,
    ).to(device)
    energy = 0.440
    resolution = args.energy_resolution_fwhm * (
        args.energy_resolution_reference_kev / 1000.0 / energy
    ) ** 0.5
    threshold_max = 2 * energy ** 2 / (0.511 + 2 * energy) - 0.001
    packed_by_view = []
    for view, events in enumerate(local_views):
        parts = []
        for chunk in torch.chunk(events, args.event_preprocess_chunks, dim=0):
            if chunk.numel() == 0:
                continue
            packed, _, _ = get_compton_backproj_list_single_sparse(
                sys440_full, detector, projector, chunk.to(device), 0.0, 0.0,
                energy, resolution, threshold_max, 0.05,
                args.energy_threshold_sum_mev, device,
                input_energies_already_smeared=True,
            )
            if packed.numel():
                parts.append(packed)
        packed_view = torch.cat(parts, dim=0) if parts else torch.empty(
            (0, projector.coarse_pixel_num + 1), dtype=torch.float32
        )
        packed_by_view.append(packed_view)
        print(f"[rank {rank}] accepted view {view + 1}: {packed_view.size(0):,}", flush=True)
    del local_views, detector

    local_event_blocks, local_accepted = materialize_local_event_blocks(
        packed_by_view, sys440_full, projector, args.materialize_block_events,
        args.materialize_device, rank,
    )
    accepted_tensor = torch.tensor([local_accepted], dtype=torch.int64, device=device)
    dist.all_reduce(accepted_tensor, op=dist.ReduceOp.SUM)
    global_accepted = int(accepted_tensor.item())
    del packed_by_view, sys440_full
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result_d, result_j = run_compton_and_joint_mlem_dist(
        sys440, proj440, rotmat, rotmat_inv, local_event_blocks, sensi440,
        sensi_d, args.iterations, args.save_step, rank,
    )
    save_result(output_dir, "440_ComptonOnly", result_d, args.iterations, args.save_step, rank)
    save_result(output_dir, "440_SinglePlusCompton", result_j, args.iterations, args.save_step, rank)

    if rank == 0:
        sum_single = save_sum(output_dir, "440SinglePlus218Single", result440.image, result218.image)
        sum_joint = save_sum(output_dir, "440SingleComptonPlus218Single", result_j.image, result218.image)
        predicted_cross_full = torch.cat(
            [
                part[:detector_bin_bounds(total_bins, part_rank, world_size)[1]
                     - detector_bin_bounds(total_bins, part_rank, world_size)[0]]
                for part_rank, part in enumerate(predicted_parts)
            ],
            dim=0,
        )
        predicted_cross_full.cpu().numpy().astype(np.float32).tofile(
            output_dir / "PredictedCntStat_218_From440.float32"
        )
        all_byte_ranges = []
        for part_rank, ranges in enumerate(gathered_ranges):
            all_byte_ranges.append([
                {
                    "view": view + 1,
                    "start": int(values[0]),
                    "end": int(values[1]),
                    "rows": int(values[2]),
                }
                for view, values in enumerate(ranges.cpu().tolist())
            ])
        manifest = {
            "factors_dir": str(factor_root.resolve()),
            "pixel_count": len(coords), "z_layers": z_layers,
            "z_extent_centers_mm": [float(coords[:, 2].min()), float(coords[:, 2].max())],
            "algorithm": "distributed six-output 218/440 JSCC MLEM",
            "physics_response": "frozen shared K*B implementation",
            "count_level": args.count_level,
            "dataset": args.data_file_name,
            "iterations": args.iterations,
            "save_step": args.save_step,
            "world_size": world_size,
            "energy_resolution_fwhm_at_reference": args.energy_resolution_fwhm,
            "energy_resolution_reference_keV": args.energy_resolution_reference_kev,
            "energy_resolution_at_440keV": resolution,
            "energy_threshold_sum_MeV": args.energy_threshold_sum_mev,
            "theta_stride": args.theta_stride,
            "z_stride": args.z_stride,
            "input_energies_already_smeared": True,
            "accepted_compton_events": global_accepted,
            "outputs": [
                "Image_440_SinglePhoton",
                "Image_440_ComptonOnly",
                "Image_440_SinglePlusCompton",
                "Image_218_SinglePhoton_CrossTalkCorrected",
                "Image_440SinglePlus218Single",
                "Image_440SingleComptonPlus218Single",
            ],
            "input_cntstat_files": input_files,
            "total_cntstat_counts": total_counts,
            "sensi_d": str(sensi_d_path),
            "list_byte_ranges_by_rank": all_byte_ranges,
            "sum_checks": {
                "440_single_plus_218_single": float(sum_single.sum(dtype=np.float64)),
                "440_joint_plus_218_single": float(sum_joint.sum(dtype=np.float64)),
            },
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Outputs: {output_dir}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    try:
        with torch.no_grad():
            main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
