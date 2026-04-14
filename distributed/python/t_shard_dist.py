import json
import os
import shutil
import time

import numpy as np
import torch

from _path_setup import setup_repo_root
setup_repo_root()
from process_list_plane_strict import get_compton_backproj_list_single


def get_t_shard_tag(args):
    e0_list_str = "_".join(str(round(e0 * 1000)) for e0 in args.e0_list)
    return (
        f"RotateNum{args.rotate_num}_{args.data_file_name}_{e0_list_str}keV_"
        f"{args.count_level}_DS{args.ds}_Delta{args.delta_r1}_ER{args.ene_resolution_662keV}_"
        f"Chunk{args.list_chunk_events}_Seed{args.seed}"
    )


def get_t_shard_base_dir(args):
    return os.path.join(args.t_shard_root, get_t_shard_tag(args))


def get_manifest_path(args, global_rank, e0):
    base_dir = get_t_shard_base_dir(args)
    return os.path.join(base_dir, f"rank_{global_rank:04d}", f"{round(e0 * 1000)}keV", "manifest.json")


def load_t_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_chunk_tensor(file_path, tensor):
    tensor.numpy().astype(np.float32).tofile(file_path)


def prepare_t_shards_for_rank(
    args,
    global_rank,
    local_rank,
    world_size,
    e0,
    intensity,
    ene_resolution,
    ene_threshold_max,
    ene_threshold_min,
    ene_threshold_sum,
    list_local,
    pixel_num,
):
    manifest_path = get_manifest_path(args, global_rank, e0)
    energy_dir = os.path.dirname(manifest_path)

    if args.force_rebuild_t and os.path.isdir(energy_dir):
        shutil.rmtree(energy_dir)

    if (not args.force_rebuild_t) and os.path.exists(manifest_path):
        manifest = load_t_manifest(manifest_path)
        manifest["manifest_path"] = manifest_path
        return manifest

    os.makedirs(energy_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    factor_path = os.path.join(args.factors_dir, f"{round(1000 * e0)}keV_RotateNum{args.rotate_num}")
    detector = torch.from_numpy(
        np.genfromtxt(os.path.join(factor_path, "Detector.csv"), delimiter=",", dtype=np.float32)[:, 1:4]
    ).to(device)
    coor_polar = torch.from_numpy(
        np.genfromtxt(os.path.join(factor_path, "coor_polar_full.csv"), delimiter=",", dtype=np.float32)
    ).to(device)

    sysmat_file_path = os.path.join(factor_path, "SysMat_polar")
    float_size = np.dtype(np.float32).itemsize
    element_count = os.path.getsize(sysmat_file_path) // float_size
    total_bins = element_count // pixel_num
    sysmat_mmap = np.memmap(sysmat_file_path, dtype=np.float32, mode="r", shape=(pixel_num, total_bins))
    sysmat_full_cpu = np.array(sysmat_mmap.T, dtype=np.float32, copy=True)
    sysmat_full_gpu = torch.from_numpy(sysmat_full_cpu).to(device) * intensity

    records = []
    rotate_event_counts = [0 for _ in range(args.rotate_num)]
    start_time = time.time()

    for rotate_idx in range(args.rotate_num):
        rotate_dir = os.path.join(energy_dir, f"rotate_{rotate_idx + 1:03d}")
        os.makedirs(rotate_dir, exist_ok=True)

        chunk_index = 0
        for chunk in torch.split(list_local[rotate_idx], args.list_chunk_events, dim=0):
            if chunk.size(0) == 0:
                continue

            chunk_seed = (
                args.seed
                + global_rank * 1000003
                + round(e0 * 1000) * 1009
                + rotate_idx * 97
                + chunk_index
            )
            torch.manual_seed(chunk_seed)

            t_chunk, _, _ = get_compton_backproj_list_single(
                sysmat_full_gpu,
                detector,
                coor_polar,
                chunk.to(device),
                args.delta_r1,
                args.delta_r2,
                e0,
                ene_resolution,
                ene_threshold_max,
                ene_threshold_min,
                ene_threshold_sum,
                device,
            )

            rows = int(t_chunk.size(0))
            rel_path = os.path.join(f"rotate_{rotate_idx + 1:03d}", f"chunk_{chunk_index:06d}.bin")
            abs_path = os.path.join(energy_dir, rel_path)

            if rows > 0:
                _write_chunk_tensor(abs_path, t_chunk)

            records.append(
                {
                    "rotate": rotate_idx,
                    "chunk_index": chunk_index,
                    "rows": rows,
                    "cols": pixel_num,
                    "path": rel_path.replace("\\", "/"),
                }
            )
            rotate_event_counts[rotate_idx] += rows
            chunk_index += 1
            torch.cuda.empty_cache()

    manifest = {
        "version": 1,
        "rank": global_rank,
        "world_size": world_size,
        "energy_keV": round(e0 * 1000),
        "pixel_num": pixel_num,
        "rotate_num": args.rotate_num,
        "list_chunk_events": args.list_chunk_events,
        "seed": args.seed,
        "rows_per_rotate": rotate_event_counts,
        "total_rows": int(sum(rotate_event_counts)),
        "records": records,
        "created_seconds": time.time() - start_time,
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    del sysmat_full_gpu
    del sysmat_mmap
    torch.cuda.empty_cache()

    manifest["manifest_path"] = manifest_path
    return manifest


def build_subset_plan_for_manifest(manifest, subset_num, manifest_dir):
    rotate_num = int(manifest["rotate_num"])
    subset_plan = [[[] for _ in range(rotate_num)] for _ in range(subset_num)]

    records_by_rotate = [[] for _ in range(rotate_num)]
    for record in manifest["records"]:
        if record["rows"] > 0:
            records_by_rotate[record["rotate"]].append(record)

    for rotate_idx in range(rotate_num):
        rotate_records = sorted(records_by_rotate[rotate_idx], key=lambda item: item["chunk_index"])
        total_rows = sum(item["rows"] for item in rotate_records)
        if total_rows == 0:
            continue

        target_rows = total_rows / subset_num
        subset_idx = 0
        rows_in_subset = 0

        for record_idx, record in enumerate(rotate_records):
            record_copy = dict(record)
            record_copy["path"] = os.path.join(manifest_dir, record["path"])
            subset_plan[subset_idx][rotate_idx].append(record_copy)
            rows_in_subset += record["rows"]

            remaining_records = len(rotate_records) - record_idx - 1
            remaining_subsets = subset_num - subset_idx - 1
            if (
                subset_idx < subset_num - 1
                and rows_in_subset >= target_rows
                and remaining_records >= remaining_subsets
            ):
                subset_idx += 1
                rows_in_subset = 0

    return subset_plan
