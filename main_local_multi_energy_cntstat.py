"""Local multi-energy, multi-output reconstruction using only CntStat data.

The entry point loads every energy's factors once, reconstructs one image per
energy, then forms the combined image by adding those independent images. It
does not load Compton List data and does not model inter-window cross-talk.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from distributed.python.multi_energy_tasks import (
    IterConfig,
    ReconTask,
    build_tasks,
    format_task_table,
)
from main_local_cntstat import (
    Tee,
    build_energy_subdir_name,
    compute_sensitivity_local,
    downsample_projection,
    load_full_sysmat,
    resolve_device,
    resolve_pixel_num,
)
from recon_osem_local_cntstat import run_recon_osem_local_cntstat
from sparse_main_utils import format_scientific_count


DEFAULT_DATASET = "ContrastPhantom_DualEnergy_10_30_240_30_225Ac"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Local CntStat-only multi-output reconstruction: one image per energy "
            "plus their post-reconstruction pixel-wise sum."
        )
    )
    parser.add_argument("--e0-list", type=float, nargs="+", default=[0.218, 0.440],
                        help="Energy channels in MeV.")
    parser.add_argument("--intensity-list", type=float, nargs="+", default=[1.0, 1.0],
                        help="System-matrix scale for each energy; use 1 when GenProj already applies yields.")
    parser.add_argument("--data-file-name", default=DEFAULT_DATASET,
                        help="Dataset stem shared by every energy's CntStat file.")
    parser.add_argument("--count-levels", nargs="+", default=["1e9", "1e10", "1e11"],
                        help="CntStat filename suffixes to process sequentially.")
    parser.add_argument("--rotate-num", type=int, default=20)
    parser.add_argument("--pixel-num-layer", type=int, default=1280)
    parser.add_argument("--pixel-num-z", type=int, default=20)
    parser.add_argument("--ds", type=float, default=1.0,
                        help="Binomial projection downsampling ratio.")
    parser.add_argument("--osem-subset-num", type=int, default=1,
                        help="Detector-bin subsets; use 1 for MLEM.")
    parser.add_argument("--single-sc-iter", type=int, default=500,
                        help="Iterations for each per-energy image.")
    parser.add_argument("--single-sc-save-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--factors-dir", default="./Factors")
    parser.add_argument("--factor-dir-suffix", default="",
                        help="Optional suffix on factor energy directories.")
    parser.add_argument("--cntstat-dir", default="./CntStat")
    parser.add_argument("--cntstat-dir-suffix", default="",
                        help="Optional suffix on CntStat energy directories.")
    parser.add_argument("--output-root", default="./Figure_Local_SC_MultiOutput")
    parser.add_argument("--device", default="auto",
                        help="Compute device: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="Recompute outputs even when valid result files already exist.")
    return parser.parse_args()


def validate_args(args):
    if len(args.e0_list) < 2:
        raise ValueError("--e0-list must contain at least two energies.")
    if len(args.e0_list) != len(args.intensity_list):
        raise ValueError("--e0-list and --intensity-list must have the same length.")
    if len(set(round(e * 1_000_000) for e in args.e0_list)) != len(args.e0_list):
        raise ValueError("--e0-list contains duplicate energies.")
    if any(e <= 0 for e in args.e0_list):
        raise ValueError("Every energy must be positive.")
    if any(weight <= 0 for weight in args.intensity_list):
        raise ValueError("Every intensity must be positive.")
    if not args.count_levels or any(not level.strip() for level in args.count_levels):
        raise ValueError("--count-levels must contain non-empty filename suffixes.")
    if len(set(args.count_levels)) != len(args.count_levels):
        raise ValueError("--count-levels contains duplicates.")
    if not (0 < args.ds <= 1):
        raise ValueError("--ds must be within (0, 1].")
    if args.rotate_num <= 0 or args.pixel_num_layer <= 0 or args.pixel_num_z <= 0:
        raise ValueError("Rotation and image dimensions must be positive.")
    if args.osem_subset_num <= 0:
        raise ValueError("--osem-subset-num must be positive.")

    if args.single_sc_iter <= 0 or args.single_sc_save_step <= 0:
        raise ValueError("Single-energy iterations and save step must be positive.")
    if args.single_sc_iter % args.single_sc_save_step != 0:
        raise ValueError("--single-sc-iter must be divisible by --single-sc-save-step.")


def resolve_path(repo_root, value):
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def build_energy_dir(root, e0, rotate_num, suffix):
    return root / build_energy_subdir_name(e0, rotate_num, suffix)


def load_factors(args, factors_root):
    loaded = []
    pixel_num = None
    total_bins_ref = None
    rotmat_ref = None
    rotmat_inv_ref = None

    for energy_idx, (e0, intensity) in enumerate(zip(args.e0_list, args.intensity_list)):
        factor_dir = build_energy_dir(
            factors_root, e0, args.rotate_num, args.factor_dir_suffix
        )
        sysmat_path = factor_dir / "SysMat_polar"
        rotmat_path = factor_dir / "RotMat_full.csv"
        rotmat_inv_path = factor_dir / "RotMatInv_full.csv"
        for required in (sysmat_path, rotmat_path, rotmat_inv_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        rotmat = torch.from_numpy(np.loadtxt(rotmat_path, delimiter=",", dtype=np.int64))
        rotmat_inv = torch.from_numpy(np.loadtxt(rotmat_inv_path, delimiter=",", dtype=np.int64))
        pixel_num_current = resolve_pixel_num(args, rotmat, rotmat_inv)
        if rotmat.ndim != 2 or rotmat.size(1) != args.rotate_num:
            raise ValueError(
                f"{rotmat_path} has shape {tuple(rotmat.shape)}, expected (*, {args.rotate_num})."
            )
        if rotmat.min().item() < 1 or rotmat.max().item() > pixel_num_current:
            raise ValueError(f"Rotation indices are out of range in {rotmat_path}.")
        if rotmat_inv.min().item() < 1 or rotmat_inv.max().item() > pixel_num_current:
            raise ValueError(f"Inverse rotation indices are out of range in {rotmat_inv_path}.")

        if pixel_num is None:
            pixel_num = pixel_num_current
            rotmat_ref = rotmat
            rotmat_inv_ref = rotmat_inv
        else:
            if pixel_num_current != pixel_num:
                raise ValueError(
                    f"Energy {e0:.3f} MeV uses pixel_num={pixel_num_current}; expected {pixel_num}."
                )
            if not torch.equal(rotmat, rotmat_ref) or not torch.equal(rotmat_inv, rotmat_inv_ref):
                raise ValueError(
                    "All energies must use identical RotMat_full/RotMatInv_full for a pixel-wise sum. "
                    f"Mismatch found at {factor_dir}."
                )

        sysmat, total_bins = load_full_sysmat(str(sysmat_path), pixel_num, intensity)
        if total_bins_ref is None:
            total_bins_ref = total_bins
        elif total_bins != total_bins_ref:
            raise ValueError(
                "The current local OSEM subset implementation requires equal detector-bin counts "
                f"across energies; got {total_bins_ref} and {total_bins}."
            )

        sensi = compute_sensitivity_local(sysmat, rotmat_inv, args.rotate_num)
        if not torch.isfinite(sensi).all() or sensi.min().item() <= 0:
            raise ValueError(
                f"Invalid single-photon sensitivity for {e0:.3f} MeV: "
                f"min={sensi.min().item():.6e}, max={sensi.max().item():.6e}."
            )

        loaded.append(
            {
                "e0": e0,
                "intensity": intensity,
                "factor_dir": factor_dir,
                "sysmat": sysmat,
                "rotmat": rotmat,
                "rotmat_inv": rotmat_inv,
                "sensi": sensi,
                "total_bins": total_bins,
            }
        )
        print(
            f"Loaded factors for {round(e0 * 1000)} keV: {factor_dir} | "
            f"pixels={pixel_num}, detector_bins={total_bins}, "
            f"sensitivity=[{sensi.min().item():.6e}, {sensi.max().item():.6e}]"
        )

    return loaded, pixel_num


def load_projections(args, cntstat_root, loaded_factors, count_level, count_idx):
    projections = []
    input_files = []
    total_count = 0.0

    for energy_idx, factor in enumerate(loaded_factors):
        cntstat_energy_dir = build_energy_dir(
            cntstat_root, factor["e0"], args.rotate_num, args.cntstat_dir_suffix
        )
        proj_path = cntstat_energy_dir / f"CntStat_{args.data_file_name}_{count_level}.csv"
        if not proj_path.is_file():
            raise FileNotFoundError(proj_path)

        projection_np = np.loadtxt(proj_path, delimiter=",", dtype=np.float32)
        expected_size = args.rotate_num * factor["total_bins"]
        if projection_np.size != expected_size:
            raise ValueError(
                f"Projection size mismatch for {proj_path}: got {projection_np.size}, "
                f"expected {expected_size} ({args.rotate_num} x {factor['total_bins']})."
            )
        if not np.isfinite(projection_np).all() or np.min(projection_np) < 0:
            raise ValueError(f"Projection contains negative or non-finite values: {proj_path}")

        projection = torch.from_numpy(
            projection_np.reshape(args.rotate_num, factor["total_bins"]).T.copy()
        )
        projection = downsample_projection(
            projection, args.ds, args.seed + count_idx * 1000 + energy_idx
        )
        energy_count = projection.sum(dtype=torch.float64).item()
        total_count += energy_count
        projections.append(projection)
        input_files.append(str(proj_path))
        print(
            f"Loaded {round(factor['e0'] * 1000)} keV CntStat: {proj_path} | "
            f"events={energy_count:.0f}"
        )

    return projections, input_files, total_count


def build_save_path(args, output_root, count_level, total_count):
    energy_tag = "-".join(str(round(e * 1000)) for e in args.e0_list)
    mode = "SE" if len(args.e0_list) == 1 else "ME"
    dataset_hash = hashlib.sha256(args.data_file_name.encode("utf-8")).hexdigest()[:8]
    folder = (
        f"{mode}_R{args.rotate_num}_E{energy_tag}_D{dataset_hash}_C{count_level}_"
        f"DS{args.ds}_O{args.osem_subset_num}_SI{args.single_sc_iter}_"
        f"POSTSUM_N{format_scientific_count(total_count)}"
    )
    return output_root / folder / "Polar"


def build_post_sum_task(args):
    energy_tag = "_".join(str(round(e * 1000)) for e in args.e0_list)
    if len(args.e0_list) > 1:
        energy_tag = f"({energy_tag})"
    return ReconTask(
        mode="SUM",
        energy_indices=tuple(range(len(args.e0_list))),
        compton_energy_indices=(),
        iter_num=args.single_sc_iter,
        save_iter_step=args.single_sc_save_step,
        output_name=f"S_{energy_tag}keV",
        type_tag="PostSum",
    )


def write_manifest(save_path, args, tasks, input_files, pixel_num, total_count):
    manifest = {
        "algorithm": "local multi-energy CntStat-only MLEM/OSEM with post-reconstruction sum",
        "cross_talk_model": False,
        "shared_image_assumption": False,
        "combined_image_definition": "sum of independently reconstructed per-energy images",
        "subset_sensitivity": "exact per detector-bin subset",
        "energies_MeV": args.e0_list,
        "intensity_list": args.intensity_list,
        "input_cntstat_files": input_files,
        "measured_event_count": total_count,
        "pixel_num": pixel_num,
        "final_image_dtype": "float32",
        "final_image_shape": [pixel_num, 1],
        "tasks": [
            {
                "type": task.type_tag,
                "mode": task.mode,
                "energy_indices": list(task.energy_indices),
                "combination_method": (
                    "post_reconstruction_pixelwise_sum"
                    if task.mode == "SUM"
                    else "iterative_reconstruction"
                ),
                "iterations": task.iter_num,
                "save_iter_step": task.save_iter_step,
                "output_file": f"Image_{task.output_name}",
                "history_file": (
                    f"Image_{task.output_name}_Iter_{task.iter_num}_"
                    f"{task.iter_num // task.save_iter_step}"
                ),
                "history_shape": [task.iter_num // task.save_iter_step, pixel_num],
            }
            for task in tasks
        ],
        "arguments": vars(args),
    }
    with open(save_path / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)


def task_output_paths(save_path, task):
    final_path = save_path / f"Image_{task.output_name}"
    history_path = save_path / (
        f"Image_{task.output_name}_Iter_{task.iter_num}_"
        f"{task.iter_num // task.save_iter_step}"
    )
    return final_path, history_path


def validate_existing_task_output(save_path, task, pixel_num):
    final_path, history_path = task_output_paths(save_path, task)
    expected_final_bytes = pixel_num * np.dtype(np.float32).itemsize
    expected_history_values = (task.iter_num // task.save_iter_step) * pixel_num
    expected_history_bytes = expected_history_values * np.dtype(np.float32).itemsize

    if not final_path.is_file():
        return False, f"missing {final_path.name}"
    if not history_path.is_file():
        return False, f"missing {history_path.name}"
    if final_path.stat().st_size != expected_final_bytes:
        return False, (
            f"{final_path.name} has {final_path.stat().st_size} bytes; "
            f"expected {expected_final_bytes}"
        )
    if history_path.stat().st_size != expected_history_bytes:
        return False, (
            f"{history_path.name} has {history_path.stat().st_size} bytes; "
            f"expected {expected_history_bytes}"
        )

    final_img = np.fromfile(final_path, dtype=np.float32)
    history = np.fromfile(history_path, dtype=np.float32)
    if final_img.size != pixel_num or history.size != expected_history_values:
        return False, "unexpected element count after reading output files"
    if not np.isfinite(final_img).all() or not np.isfinite(history).all():
        return False, "contains non-finite values"
    if final_img.min(initial=0.0) < 0 or history.min(initial=0.0) < 0:
        return False, "contains negative values"
    if not np.array_equal(history[-pixel_num:], final_img):
        return False, "history final frame does not match final image"

    return True, "valid"


def write_float32_atomic(path, array):
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        np.asarray(array, dtype=np.float32).tofile(temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def combine_independent_outputs(
    args,
    save_path,
    reconstruction_tasks,
    combined_task,
    pixel_num,
):
    if not args.overwrite_existing:
        is_valid, reason = validate_existing_task_output(
            save_path, combined_task, pixel_num
        )
        if is_valid:
            print(f"\n[PostSum] Reusing Image_{combined_task.output_name}: {reason}.")
            return

    snapshot_num = combined_task.iter_num // combined_task.save_iter_step
    final_sum = np.zeros(pixel_num, dtype=np.float32)
    history_sum = np.zeros((snapshot_num, pixel_num), dtype=np.float32)

    for task in reconstruction_tasks:
        is_valid, reason = validate_existing_task_output(save_path, task, pixel_num)
        if not is_valid:
            raise RuntimeError(
                f"Cannot build combined image from Image_{task.output_name}: {reason}."
            )
        final_path, history_path = task_output_paths(save_path, task)
        final_sum += np.fromfile(final_path, dtype=np.float32)
        history_sum += np.fromfile(history_path, dtype=np.float32).reshape(
            snapshot_num, pixel_num
        )

    if not np.isfinite(final_sum).all() or not np.isfinite(history_sum).all():
        raise FloatingPointError("The post-reconstruction sum contains non-finite values.")
    if np.any(final_sum < 0) or np.any(history_sum < 0):
        raise ValueError("The post-reconstruction sum contains negative values.")
    if not np.array_equal(history_sum[-1], final_sum):
        raise RuntimeError("The combined final image does not match its final history frame.")

    final_path, history_path = task_output_paths(save_path, combined_task)
    write_float32_atomic(history_path, history_sum)
    write_float32_atomic(final_path, final_sum)
    print(
        f"\n[PostSum] Image_{combined_task.output_name} = "
        f"{' + '.join('Image_' + task.output_name for task in reconstruction_tasks)} | "
        f"min={final_sum.min():.6e} max={final_sum.max():.6e} "
        f"sum={final_sum.sum(dtype=np.float64):.6e}"
    )


def run_count_level(
    args,
    tasks,
    loaded_factors,
    projections,
    save_path,
    device,
    pixel_num,
):
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_str = str(save_path) + os.sep

    for task_idx, task in enumerate(tasks):
        if not args.overwrite_existing:
            is_valid, reason = validate_existing_task_output(save_path, task, pixel_num)
            if is_valid:
                print(
                    f"\n[Task {task_idx + 1}/{len(tasks)}] Reusing existing "
                    f"Image_{task.output_name}: {reason}."
                )
                continue
            print(
                f"\n[Task {task_idx + 1}/{len(tasks)}] Existing "
                f"Image_{task.output_name} is incomplete: {reason}. Running reconstruction."
            )

        energy_indices = list(task.energy_indices)
        print("\n" + "=" * 70)
        print(
            f"[Task {task_idx + 1}/{len(tasks)}] {task.type_tag} -> "
            f"Image_{task.output_name}"
        )
        print("=" * 70)

        s_map_arg = argparse.Namespace(
            s=sum(loaded_factors[idx]["sensi"] for idx in energy_indices)
        )
        iter_arg = argparse.Namespace(
            sc=task.iter_num,
            save_iter_step=task.save_iter_step,
            osem_subset_num=args.osem_subset_num,
            ene_num=len(energy_indices),
            seed=args.seed + task_idx,
        )

        run_recon_osem_local_cntstat(
            [loaded_factors[idx]["sysmat"] for idx in energy_indices],
            [loaded_factors[idx]["rotmat"] for idx in energy_indices],
            [loaded_factors[idx]["rotmat_inv"] for idx in energy_indices],
            [projections[idx] for idx in energy_indices],
            iter_arg,
            s_map_arg,
            save_path_str,
            device,
            output_name=task.output_name,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    validate_args(args)
    repo_root = Path(__file__).resolve().parent
    factors_root = resolve_path(repo_root, args.factors_dir)
    cntstat_root = resolve_path(repo_root, args.cntstat_dir)
    output_root = resolve_path(repo_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    reconstruction_tasks = build_tasks(
        args.e0_list,
        [],
        IterConfig(
            type1_single_sc=(args.single_sc_iter, args.single_sc_save_step),
        ),
    )
    if not reconstruction_tasks:
        raise ValueError("No reconstruction tasks were generated.")
    combined_task = build_post_sum_task(args)
    output_tasks = reconstruction_tasks + [combined_task]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    started = time.time()
    log_path = output_root / f"print_log_local_multi_output_{int(started)}.txt"
    completed_paths = []
    logfile = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, logfile)

    try:
        print(f"Using device: {device}")
        print(f"Args: {args}")
        print("\nReconstruction task plan")
        print(format_task_table(reconstruction_tasks, args.e0_list))
        print("\nDerived output")
        print(format_task_table([combined_task], args.e0_list))
        print(
            "\nModel scope: CntStat-only; no Compton List input, no cross-talk response, "
            "and no cross-talk correction. The combined image is the pixel-wise sum of "
            "the independently reconstructed energy images."
        )

        loaded_factors, pixel_num = load_factors(args, factors_root)
        for count_idx, count_level in enumerate(args.count_levels):
            print("\n" + "#" * 78)
            print(f"Count level {count_idx + 1}/{len(args.count_levels)}: {count_level}")
            print("#" * 78)
            projections, input_files, total_count = load_projections(
                args, cntstat_root, loaded_factors, count_level, count_idx
            )
            save_path = build_save_path(args, output_root, count_level, total_count)
            print(f"Save path: {save_path}")
            run_count_level(
                args,
                reconstruction_tasks,
                loaded_factors,
                projections,
                save_path,
                device,
                pixel_num,
            )
            combine_independent_outputs(
                args,
                save_path,
                reconstruction_tasks,
                combined_task,
                pixel_num,
            )
            write_manifest(
                save_path, args, output_tasks, input_files, pixel_num, total_count
            )
            completed_paths.append(save_path)
            shutil.copy2(log_path, save_path / "print_log.txt")
            del projections

        print(f"\nAll count levels finished in {time.time() - started:.2f}s")
    finally:
        sys.stdout = original_stdout
        logfile.close()

    print(f"Master log: {log_path}")
    for save_path in completed_paths:
        print(f"Output: {save_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
