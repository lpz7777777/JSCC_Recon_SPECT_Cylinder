"""Compare the production Sensi_d with the actual sparse reconstruction operator."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from compton_sparse_ops import build_compton_sparse_projector, materialize_sparse_event_rows_to_fine
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from spect_sensitivity.io import count_event_rows, iter_event_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", type=Path, default=Path("Factors/440keV_RotateNum20"))
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--source-photons", type=float, default=5e10)
    parser.add_argument("--event-fraction", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--theta-stride", type=int, default=1)
    parser.add_argument("--z-stride", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    factor = args.factor_dir.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    list_file = args.list_file.resolve()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    coordinates = np.loadtxt(factor / "coor_polar_full.csv", delimiter=",", dtype=np.float32)
    rotation = np.loadtxt(factor / "RotMat_full.csv", delimiter=",", dtype=np.int64)
    volumes = np.fromfile(factor / "polar_cell_volume_mm3.float64", dtype=np.float64)
    detector = torch.from_numpy(np.loadtxt(factor / "Detector.csv", delimiter=",", skiprows=1, dtype=np.float32)[:, 1:4]).to(device)
    pixels = coordinates.shape[0]
    sysmat = torch.from_numpy(np.fromfile(factor / "SysMat_polar", dtype=np.float32).reshape(pixels, -1).T.copy()).to(device)
    projector = build_compton_sparse_projector(torch.from_numpy(coordinates), args.theta_stride, args.z_stride, 20).to(device)
    total_rows = count_event_rows((list_file,))[0]
    selected_rows = int(total_rows * args.event_fraction)
    represented_photons = args.source_photons * selected_rows / total_rows
    accumulator = torch.zeros(pixels, dtype=torch.float64, device=device)
    accepted = 0
    for batch in iter_event_batches((list_file,), args.batch_size, selected_rows):
        rows, _, _ = get_compton_backproj_list_single_sparse(
            sysmat, detector, projector, batch.values[:, :4].to(device), 0.0, 0.0,
            0.440, 0.1 * (0.662 / 0.440) ** 0.5,
            2 * 0.440**2 / (0.511 + 2 * 0.440) - 0.001, 0.05, 0.40, device,
            input_energies_already_smeared=True,
        )
        if rows.numel():
            fine, _ = materialize_sparse_event_rows_to_fine(rows.to(device), sysmat, projector)
            accumulator += fine.sum(dim=0, dtype=torch.float64)
            accepted += fine.size(0)
        if batch.next_file_offset % (args.batch_size * 200) == 0:
            print(f"processed={batch.next_file_offset}/{selected_rows} accepted={accepted}")

    raw = (accumulator.cpu().numpy() * volumes.sum() / represented_photons)
    rotated = sum(raw[rotation[:, index] - 1] for index in range(20)) / 20
    installed = np.fromfile(factor / "Sensi_d", dtype=np.float32).astype(np.float64)
    eps_sparse = rotated / volumes
    eps_installed = installed / volumes
    summary = {
        "selected_rows": selected_rows, "represented_photons": represented_photons,
        "accepted_events": accepted, "accepted_per_photon": accepted / represented_photons,
        "sparse_global_efficiency": float(rotated.sum() / volumes.sum()),
        "installed_global_efficiency": float(installed.sum() / volumes.sum()),
        "point_efficiency_correlation": float(np.corrcoef(eps_sparse, eps_installed)[0, 1]),
        "ratio_sparse_over_installed": {
            "min": float(np.min(eps_sparse / eps_installed)),
            "median": float(np.median(eps_sparse / eps_installed)),
            "max": float(np.max(eps_sparse / eps_installed)),
        },
    }
    rotated.astype(np.float32).tofile(output / "Sensi_d_sparse_operator_sample")
    eps_sparse.astype(np.float32).tofile(output / "Sensi_d_sparse_operator_sample_point_efficiency")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
