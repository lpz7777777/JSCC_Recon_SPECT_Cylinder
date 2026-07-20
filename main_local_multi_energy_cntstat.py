"""Local 218/440 CntStat reconstruction with explicit 440-to-218 cross-talk."""

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
    ReconTask,
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
from recon_osem_local_cntstat import (
    forward_project_local_cntstat,
    run_recon_osem_local_cntstat,
)
from sparse_main_utils import format_scientific_count


DEFAULT_DATASET = "ContrastPhantom_DualEnergy_10_30_240_30_225Ac"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Local 218/440 CntStat reconstruction with an explicit fixed "
            "440-to-218 additive-background correction."
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
    parser.add_argument(
        "--cross-talk-scale",
        type=float,
        default=1.0,
        help=(
            "Calibration scale on C(218-window <- 440-source). Keep 1 when "
            "CntStat contains absolute mixed-source counts from GenProj/Geant4."
        ),
    )
    parser.add_argument("--cntstat-dir", default="./CntStat")
    parser.add_argument("--cntstat-dir-suffix", default="",
                        help="Optional suffix on CntStat energy directories.")
    parser.add_argument(
        "--output-root",
        default="./Results/Reconstruction/Figure_Local_SC_MultiOutput",
    )
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
    expected_kev = {218, 440}
    actual_kev = {round(e * 1000) for e in args.e0_list}
    if actual_kev != expected_kev or len(args.e0_list) != 2:
        raise ValueError("Cross-talk-aware mode requires exactly 0.218 and 0.440 MeV.")
    if any(abs(weight - 1.0) > 1e-12 for weight in args.intensity_list):
        raise ValueError(
            "Cross-talk-aware absolute-count mode requires --intensity-list 1 1. "
            "Gamma yields are already present in the observed CntStat amplitudes."
        )
    if not np.isfinite(args.cross_talk_scale) or args.cross_talk_scale <= 0:
        raise ValueError("--cross-talk-scale must be finite and positive.")
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
        expected_response = "A218" if round(e0 * 1000) == 218 else "A440"
        factor_manifest = validate_factor_response(factor_dir, expected_response)
        maps_activity_density = bool(
            factor_manifest and factor_manifest.get("maps_activity_density", False)
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
                "factor_manifest": factor_manifest,
                "maps_activity_density": maps_activity_density,
            }
        )
        print(
            f"Loaded factors for {round(e0 * 1000)} keV: {factor_dir} | "
            f"pixels={pixel_num}, detector_bins={total_bins}, "
            f"sensitivity=[{sensi.min().item():.6e}, {sensi.max().item():.6e}]"
        )

    density_basis_values = {factor["maps_activity_density"] for factor in loaded}
    if len(density_basis_values) != 1:
        raise ValueError("All direct-energy Factors must use the same activity basis.")
    return loaded, pixel_num


def validate_factor_response(factor_dir, expected_response):
    manifest_path = factor_dir / "factor_manifest.json"
    if not manifest_path.is_file():
        print(
            f"WARNING: {manifest_path} is missing; response semantics are checked "
            "only by directory name and dimensions."
        )
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    actual = manifest.get("response")
    if actual != expected_response:
        raise ValueError(
            f"Factor response mismatch in {manifest_path}: got {actual!r}, "
            f"expected {expected_response!r}."
        )
    if manifest.get("includes_225Ac_gamma_yield") not in (False, None):
        raise ValueError(
            f"{manifest_path} already includes gamma yield; this would double-weight CntStat."
        )
    return manifest


def build_cross_factor_dir(root, rotate_num, suffix):
    suffix_text = suffix.strip()
    if suffix_text and not suffix_text.startswith("_"):
        suffix_text = "_" + suffix_text
    return root / f"440keV_to218win_RotateNum{rotate_num}{suffix_text}"


def load_cross_factor(args, factors_root, loaded_factors, pixel_num):
    factor_dir = build_cross_factor_dir(
        factors_root, args.rotate_num, args.factor_dir_suffix
    )
    factor_manifest = validate_factor_response(factor_dir, "C440to218")
    maps_activity_density = bool(
        factor_manifest and factor_manifest.get("maps_activity_density", False)
    )
    sysmat_path = factor_dir / "SysMat_polar"
    rotmat_path = factor_dir / "RotMat_full.csv"
    rotmat_inv_path = factor_dir / "RotMatInv_full.csv"
    for required in (sysmat_path, rotmat_path, rotmat_inv_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    rotmat = torch.from_numpy(np.loadtxt(rotmat_path, delimiter=",", dtype=np.int64))
    rotmat_inv = torch.from_numpy(
        np.loadtxt(rotmat_inv_path, delimiter=",", dtype=np.int64)
    )
    sysmat, total_bins = load_full_sysmat(
        str(sysmat_path), pixel_num, args.cross_talk_scale
    )

    reference = next(
        factor for factor in loaded_factors if round(factor["e0"] * 1000) == 218
    )
    if maps_activity_density != reference["maps_activity_density"]:
        raise ValueError(
            "C440to218 and direct Factors use different integrated-activity/density bases."
        )
    if total_bins != reference["total_bins"]:
        raise ValueError(
            f"C440to218 detector bins={total_bins}; A218 bins={reference['total_bins']}."
        )
    if not torch.equal(rotmat, reference["rotmat"]) or not torch.equal(
        rotmat_inv, reference["rotmat_inv"]
    ):
        raise ValueError("C440to218 and direct responses use different rotation mappings.")

    sensitivity = compute_sensitivity_local(sysmat, rotmat_inv, args.rotate_num)
    if not torch.isfinite(sensitivity).all() or sensitivity.min().item() < 0:
        raise ValueError("C440to218 sensitivity is negative or non-finite.")
    result = {
        "factor_dir": factor_dir,
        "sysmat": sysmat,
        "rotmat": rotmat,
        "rotmat_inv": rotmat_inv,
        "sensi": sensitivity,
        "total_bins": total_bins,
        "factor_manifest": factor_manifest,
        "maps_activity_density": maps_activity_density,
    }
    print(
        f"Loaded C440to218 factors: {factor_dir} | pixels={pixel_num}, "
        f"detector_bins={total_bins}, scale={args.cross_talk_scale:.9g}, "
        f"sensitivity=[{sensitivity.min().item():.6e}, {sensitivity.max().item():.6e}]"
    )
    return result


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
        f"XTALK_BG{args.cross_talk_scale:g}_N{format_scientific_count(total_count)}"
    )
    return output_root / folder / "Polar"


def build_crosstalk_tasks(args):
    idx_218 = next(i for i, energy in enumerate(args.e0_list) if round(energy * 1000) == 218)
    idx_440 = next(i for i, energy in enumerate(args.e0_list) if round(energy * 1000) == 440)

    def task(output_name, type_tag, energy_indices, mode="S"):
        return ReconTask(
            mode=mode,
            energy_indices=tuple(energy_indices),
            compton_energy_indices=(),
            iter_num=args.single_sc_iter,
            save_iter_step=args.single_sc_save_step,
            output_name=output_name,
            type_tag=type_tag,
        )

    return {
        "direct_440": task("S_440keV", "Direct440", (idx_440,)),
        "contaminated_218": task(
            "S_218keV_Contaminated", "Observed218WithoutCorrection", (idx_218,)
        ),
        "corrected_218": task(
            "S_218keV_CrossTalkCorrected", "CrossTalkCorrected218", (idx_218,)
        ),
        "corrected_sum": task(
            "S_(440_218)keV_CrossTalkCorrected",
            "CorrectedPostSum",
            (idx_440, idx_218),
            mode="SUM",
        ),
    }


def write_manifest(
    save_path,
    args,
    tasks,
    input_files,
    pixel_num,
    total_count,
    cross_factor,
    diagnostics,
):
    density_basis = bool(cross_factor.get("maps_activity_density", False))
    image_symbol = "rho" if density_basis else "x"
    if density_basis:
        observation_model = {
            "218_window": "y218 ~ Poisson(B218*rho218 + BC440to218*rho440)",
            "440_window": "y440 ~ Poisson(B440*rho440)",
        }
        image_activity_convention = (
            "activity density in emitted photons per mm3, integrated over all rotation views"
        )
        matrix_basis = "B = A * diag(full polar-cell volume in mm3)"
        gamma_yield_handling = (
            "observed CntStat amplitudes already contain source yields; density-basis "
            "response matrices contain no gamma yield and no Y440/Y218 factor is applied"
        )
    else:
        observation_model = {
            "218_window": "y218 ~ Poisson(A218*x218 + C440to218*x440)",
            "440_window": "y440 ~ Poisson(A440*x440)",
        }
        image_activity_convention = "total activity summed over all rotation views"
        matrix_basis = "A maps integrated activity per polar cell"
        gamma_yield_handling = (
            "observed CntStat amplitudes already contain source yields; matrices are "
            "per emitted photon and no Y440/Y218 factor is applied in reconstruction"
        )
    manifest = {
        "algorithm": (
            "local 218/440 CntStat-only MLEM/OSEM with fixed 440-to-218 "
            "additive-background correction"
        ),
        "cross_talk_model": True,
        "cross_talk_correction": (
            f"{image_symbol}440 is reconstructed from y440; its predicted 218-window "
            "contribution is then "
            "converted to measured per-view CntStat scale and held fixed as additive "
            "background in the 218-window Poisson denominator"
        ),
        "observation_model": observation_model,
        "cross_talk_scale": args.cross_talk_scale,
        "cross_factor_dir": str(cross_factor["factor_dir"]),
        "gamma_yield_handling": gamma_yield_handling,
        "shared_image_assumption": False,
        "combined_image_definition": (
            f"{image_symbol}440 + {image_symbol}218_corrected, pixelwise in a shared basis"
        ),
        "image_activity_convention": image_activity_convention,
        "matrix_activity_basis": matrix_basis,
        "maps_activity_density": density_basis,
        "projection_activity_convention": (
            "measured per-view CntStat; forward projections include 1/rotate_num"
        ),
        "subset_sensitivity": "exact per detector-bin subset",
        "energies_MeV": args.e0_list,
        "intensity_list": args.intensity_list,
        "input_cntstat_files": input_files,
        "measured_event_count": total_count,
        "pixel_num": pixel_num,
        "final_image_dtype": "float32",
        "final_image_shape": [pixel_num, 1],
        "cross_talk_diagnostics": diagnostics,
        "tasks": [
            {
                "type": task.type_tag,
                "mode": task.mode,
                "energy_indices": list(task.energy_indices),
                "combination_method": (
                    "corrected_post_reconstruction_pixelwise_sum"
                    if task.mode == "SUM"
                    else (
                        "fixed_additive_background_poisson_em"
                        if task.type_tag == "CrossTalkCorrected218"
                        else "iterative_reconstruction"
                    )
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
    # Keep the temporary name short: deeply nested run folders plus the
    # combined-output history name can otherwise exceed Windows MAX_PATH.
    temp_path = path.with_name(f".tmp_{os.getpid()}")
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


def reconstruct_or_load(
    args,
    task,
    factor,
    projection,
    save_path,
    device,
    pixel_num,
    seed_offset,
    additive_background=None,
):
    if not args.overwrite_existing:
        is_valid, reason = validate_existing_task_output(save_path, task, pixel_num)
        if is_valid:
            print(f"\n[{task.type_tag}] Reusing Image_{task.output_name}: {reason}.")
            final_path, _ = task_output_paths(save_path, task)
            return torch.from_numpy(np.fromfile(final_path, dtype=np.float32).copy()).reshape(-1, 1)
        print(f"\n[{task.type_tag}] Existing output is incomplete: {reason}.")

    iter_arg = argparse.Namespace(
        sc=task.iter_num,
        save_iter_step=task.save_iter_step,
        osem_subset_num=args.osem_subset_num,
        ene_num=1,
        seed=args.seed + seed_offset,
    )
    s_map_arg = argparse.Namespace(s=factor["sensi"])
    return run_recon_osem_local_cntstat(
        [factor["sysmat"]],
        [factor["rotmat"]],
        [factor["rotmat_inv"]],
        [projection],
        iter_arg,
        s_map_arg,
        str(save_path) + os.sep,
        device,
        output_name=task.output_name,
        additive_background_all=[additive_background],
    ).detach().cpu()


def relative_l2(observed, predicted):
    denominator = torch.linalg.vector_norm(observed.to(torch.float64)).clamp_min(1e-12)
    numerator = torch.linalg.vector_norm(
        observed.to(torch.float64) - predicted.to(torch.float64)
    )
    return (numerator / denominator).item()


def run_count_level(
    args,
    task_map,
    loaded_factors,
    cross_factor,
    projections,
    save_path,
    device,
    pixel_num,
):
    save_path.mkdir(parents=True, exist_ok=True)
    idx_218 = next(i for i, factor in enumerate(loaded_factors) if round(factor["e0"] * 1000) == 218)
    idx_440 = next(i for i, factor in enumerate(loaded_factors) if round(factor["e0"] * 1000) == 440)
    factor_218 = loaded_factors[idx_218]
    factor_440 = loaded_factors[idx_440]
    projection_218 = projections[idx_218]
    projection_440 = projections[idx_440]

    image_440 = reconstruct_or_load(
        args, task_map["direct_440"], factor_440, projection_440,
        save_path, device, pixel_num, seed_offset=1,
    )
    image_218_contaminated = reconstruct_or_load(
        args, task_map["contaminated_218"], factor_218, projection_218,
        save_path, device, pixel_num, seed_offset=2,
    )
    del image_218_contaminated
    if device.type == "cuda":
        torch.cuda.empty_cache()

    predicted_cross_cntstat = forward_project_local_cntstat(
        cross_factor["sysmat"], cross_factor["rotmat"], image_440, device
    )
    predicted_cross_path = save_path / "PredictedCntStat_218_From440.float32"
    write_float32_atomic(predicted_cross_path, predicted_cross_cntstat.numpy())
    np.savetxt(
        save_path / "PredictedCntStat_218_From440.csv",
        predicted_cross_cntstat.numpy().T,
        delimiter=",",
        fmt="%.9g",
    )
    print(
        f"\nPredicted fixed 440-to-218 background: events="
        f"{predicted_cross_cntstat.sum(dtype=torch.float64).item():.6e}"
    )

    image_218_corrected = reconstruct_or_load(
        args,
        task_map["corrected_218"],
        factor_218,
        projection_218,
        save_path,
        device,
        pixel_num,
        seed_offset=3,
        additive_background=predicted_cross_cntstat,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    predicted_218_direct = forward_project_local_cntstat(
        factor_218["sysmat"], factor_218["rotmat"], image_218_corrected, device
    )
    predicted_440 = forward_project_local_cntstat(
        factor_440["sysmat"], factor_440["rotmat"], image_440, device
    )
    predicted_218_total = predicted_218_direct + predicted_cross_cntstat
    diagnostics = {
        "observed_218_events": projection_218.sum(dtype=torch.float64).item(),
        "observed_440_events": projection_440.sum(dtype=torch.float64).item(),
        "predicted_cross_talk_events": predicted_cross_cntstat.sum(dtype=torch.float64).item(),
        "predicted_cross_fraction_of_observed_218": (
            predicted_cross_cntstat.sum(dtype=torch.float64).item()
            / max(projection_218.sum(dtype=torch.float64).item(), 1.0)
        ),
        "fraction_of_218_bins_where_cross_prediction_exceeds_observation": (
            torch.mean((predicted_cross_cntstat > projection_218).to(torch.float32)).item()
        ),
        "relative_l2_residual_218": relative_l2(projection_218, predicted_218_total),
        "relative_l2_residual_440": relative_l2(projection_440, predicted_440),
        "predicted_cross_file": predicted_cross_path.name,
        "predicted_cross_shape": list(predicted_cross_cntstat.shape),
    }
    print(f"Cross-talk diagnostics: {json.dumps(diagnostics, indent=2)}")

    combine_independent_outputs(
        args,
        save_path,
        [task_map["direct_440"], task_map["corrected_218"]],
        task_map["corrected_sum"],
        pixel_num,
    )
    return diagnostics


def main():
    args = parse_args()
    validate_args(args)
    repo_root = Path(__file__).resolve().parent
    factors_root = resolve_path(repo_root, args.factors_dir)
    cntstat_root = resolve_path(repo_root, args.cntstat_dir)
    output_root = resolve_path(repo_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    task_map = build_crosstalk_tasks(args)
    reconstruction_tasks = [
        task_map["direct_440"],
        task_map["contaminated_218"],
        task_map["corrected_218"],
    ]
    output_tasks = reconstruction_tasks + [task_map["corrected_sum"]]

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
        print("\nDerived corrected output")
        print(format_task_table([task_map["corrected_sum"]], args.e0_list))
        print(
            "\nModel scope: CntStat-only with explicit C440to218. The 440 image is "
            "estimated from the 440-window response, its predicted 218-window "
            "contribution is held fixed as an additive Poisson background, and the "
            "corrected output is the pixelwise sum of the 440 and corrected-218 "
            "images in their shared Factors basis."
        )

        loaded_factors, pixel_num = load_factors(args, factors_root)
        cross_factor = load_cross_factor(args, factors_root, loaded_factors, pixel_num)
        for count_idx, count_level in enumerate(args.count_levels):
            print("\n" + "#" * 78)
            print(f"Count level {count_idx + 1}/{len(args.count_levels)}: {count_level}")
            print("#" * 78)
            projections, input_files, total_count = load_projections(
                args, cntstat_root, loaded_factors, count_level, count_idx
            )
            save_path = build_save_path(args, output_root, count_level, total_count)
            print(f"Save path: {save_path}")
            diagnostics = run_count_level(
                args,
                task_map,
                loaded_factors,
                cross_factor,
                projections,
                save_path,
                device,
                pixel_num,
            )
            write_manifest(
                save_path,
                args,
                output_tasks,
                input_files,
                pixel_num,
                total_count,
                cross_factor,
                diagnostics,
            )
            completed_paths.append(save_path)
            shutil.copy2(log_path, save_path / "print_log.txt")
            del projections, diagnostics

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
