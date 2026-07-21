"""Run the 1e9 Geant4 JSCC single/Compton validation requested for 225Ac."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from compton_sparse_ops import build_compton_sparse_projector
from distributed.python.multi_energy_tasks import ReconTask
from main_local_multi_energy_cntstat import (
    forward_project_local_cntstat,
    load_cross_factor,
    load_factors,
    load_projections,
    reconstruct_or_load,
    write_float32_atomic,
)
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from recon_osem_local_sparse_jsccsd_only import run_recon_compton_and_joint_local_sparse


DATASET = "ContrastPhantom_DualEnergy_10_30_240_30_225Ac"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-step", type=int, default=50)
    parser.add_argument("--theta-stride", type=int, default=1)
    parser.add_argument("--z-stride", type=int, default=2)
    parser.add_argument("--list-workers", type=int, default=20)
    parser.add_argument("--t-divide-num", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sensi-d-path", type=Path, default=None)
    parser.add_argument(
        "--max-events-per-view", type=int, default=0,
        help="Diagnostic limit; zero processes every List event.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/Reconstruction/JSCC_ComptonValidation_Geant4_1e9_Iter1000"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def task(name, iterations, save_step, type_tag):
    return ReconTask("S", (0,), (), iterations, save_step, name, type_tag)


def save_sum(output_dir, name, *images):
    result = np.sum([np.asarray(image, dtype=np.float32).reshape(-1) for image in images], axis=0)
    write_float32_atomic(output_dir / f"Image_{name}", result)
    return result


def main():
    cli = parse_args()
    if cli.iterations <= 0 or cli.save_step <= 0 or cli.iterations % cli.save_step:
        raise ValueError("iterations must be positive and divisible by save-step")
    repo = Path(__file__).resolve().parent
    output = (repo / cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20260721)
    np.random.seed(20260721)

    args = argparse.Namespace(
        e0_list=[0.218, 0.440], intensity_list=[1.0, 1.0], rotate_num=20,
        pixel_num_layer=1280, pixel_num_z=20, factor_dir_suffix="",
        cross_talk_scale=1.0, cntstat_dir_suffix="_Geant4JSCC",
        data_file_name=DATASET, ds=1.0, osem_subset_num=1,
        seed=20260721, overwrite_existing=cli.overwrite,
    )
    factors, pixel_num = load_factors(args, repo / "Factors")
    cross = load_cross_factor(args, repo / "Factors", factors, pixel_num)
    projections, input_files, total_counts = load_projections(
        args, repo / "CntStat", factors, "1e9", 0
    )
    factor218 = next(value for value in factors if round(value["e0"] * 1000) == 218)
    factor440 = next(value for value in factors if round(value["e0"] * 1000) == 440)
    proj218 = projections[0]
    proj440 = projections[1]

    t440 = task("440_SinglePhoton", cli.iterations, cli.save_step, "440 single")
    image440 = reconstruct_or_load(
        args, t440, factor440, proj440, output, device, pixel_num, 1
    )
    predicted_cross = forward_project_local_cntstat(
        cross["sysmat"], cross["rotmat"], image440, device
    )
    write_float32_atomic(output / "PredictedCntStat_218_From440.float32", predicted_cross.numpy())
    t218 = task(
        "218_SinglePhoton_CrossTalkCorrected", cli.iterations, cli.save_step,
        "218 corrected single",
    )
    image218 = reconstruct_or_load(
        args, t218, factor218, proj218, output, device, pixel_num, 2,
        additive_background=predicted_cross,
    )

    sensi_d_path = (
        cli.sensi_d_path.resolve()
        if cli.sensi_d_path is not None
        else factor440["factor_dir"] / "Sensi_d"
    )
    if not sensi_d_path.is_file():
        raise FileNotFoundError(f"Formal Cartesian-derived Sensi_d is not installed: {sensi_d_path}")
    sensi_d = torch.from_numpy(np.fromfile(sensi_d_path, dtype=np.float32).copy()).reshape(-1, 1)
    if sensi_d.numel() != pixel_num or torch.any(sensi_d <= 0):
        raise ValueError("Installed Sensi_d has invalid dimensions or values")

    detector = torch.from_numpy(
        np.loadtxt(factor440["factor_dir"] / "Detector.csv", delimiter=",", skiprows=1, dtype=np.float32)[:, 1:4]
    ).to(device)
    coordinates = torch.from_numpy(
        np.loadtxt(factor440["factor_dir"] / "coor_polar_full.csv", delimiter=",", dtype=np.float32)
    )
    projector = build_compton_sparse_projector(
        coordinates, theta_stride=cli.theta_stride, z_stride=cli.z_stride,
        rotate_num=20, dtype=torch.float32,
    )
    projector_device = projector.to(device)
    sysmat_device = factor440["sysmat"].to(device)
    list_dir = repo / "List" / "218-440keV_RotateNum20_Geant4JSCC" / f"List_{DATASET}_1e9"
    energy = 0.440
    resolution = 0.1 * (0.662 / energy) ** 0.5
    threshold_max = 2 * energy**2 / (0.511 + 2 * energy) - 0.001
    t_rotate_all = []
    accepted_compton = 0
    preprocessing_started = time.time()
    for rotate_index in range(20):
        values = np.loadtxt(list_dir / f"{rotate_index + 1}.csv", delimiter=",", dtype=np.float32, usecols=(0, 1, 2, 3))
        events = torch.from_numpy(np.atleast_2d(values))
        if cli.max_events_per_view > 0:
            events = events[:cli.max_events_per_view]
        parts = []
        for chunk in torch.chunk(events, cli.list_workers, dim=0):
            rows, _, _ = get_compton_backproj_list_single_sparse(
                sysmat_device, detector, projector_device, chunk.to(device),
                0.0, 0.0, energy, resolution, threshold_max, 0.05, 0.40,
                device, input_energies_already_smeared=True,
            )
            if rows.numel():
                parts.append(rows)
        rotate_rows = torch.cat(parts, dim=0) if parts else torch.empty((0, projector.coarse_pixel_num + 1))
        t_rotate_all.append(rotate_rows)
        accepted_compton += rotate_rows.size(0)
        print(f"List view {rotate_index + 1}/20: accepted={rotate_rows.size(0)}, total={accepted_compton}")
    del sysmat_device, projector_device, detector
    torch.cuda.empty_cache()
    print(f"List preprocessing finished in {time.time()-preprocessing_started:.1f}s")

    image_d, image_j = run_recon_compton_and_joint_local_sparse(
        factor440["sysmat"], factor440["rotmat"], factor440["rotmat_inv"],
        proj440, t_rotate_all, projector, factor440["sensi"], sensi_d,
        cli.iterations, cli.save_step, cli.t_divide_num, output, device,
    )
    sum_single = save_sum(output, "440SinglePlus218Single", image440, image218)
    sum_joint = save_sum(output, "440SingleComptonPlus218Single", image_j, image218)
    manifest = {
        "dataset": DATASET, "count_level": "1e9", "iterations": cli.iterations,
        "save_step": cli.save_step, "osem_subsets": 1,
        "input_energies_already_smeared": True,
        "compton_theta_stride": cli.theta_stride, "compton_z_stride": cli.z_stride,
        "max_events_per_view": cli.max_events_per_view,
        "accepted_compton_events": accepted_compton,
        "input_cntstat_files": input_files, "total_cntstat_counts": total_counts,
        "outputs": [
            "Image_440_SinglePhoton", "Image_440_ComptonOnly",
            "Image_440_SinglePlusCompton", "Image_218_SinglePhoton_CrossTalkCorrected",
            "Image_440SinglePlus218Single", "Image_440SingleComptonPlus218Single",
        ],
        "sum_checks": {
            "440_single_plus_218_single": float(sum_single.sum(dtype=np.float64)),
            "440_joint_plus_218_single": float(sum_joint.sum(dtype=np.float64)),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validation outputs: {output}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
