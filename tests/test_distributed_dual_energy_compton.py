import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "distributed/dual_energy_compton_python"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


main = load_module("dist_dual_main", MODULE_DIR / "main_dist_dual_energy_compton.py")
recon = load_module("dist_dual_recon", MODULE_DIR / "reconstruction.py")


def test_detector_bin_bounds_cover_without_overlap():
    bounds = [main.detector_bin_bounds(10496, rank, 32) for rank in range(32)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 10496
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))


def test_byte_partition_reads_every_row_once(tmp_path):
    path = tmp_path / "events.csv"
    rows = [f"{i},{i + 0.1},{i + 1},{i + 1.1}\n" for i in range(103)]
    path.write_text("".join(rows), encoding="ascii")
    pieces = [main.read_csv_byte_partition(path, rank, 7)[0] for rank in range(7)]
    values = np.concatenate(pieces, axis=0)
    assert values.shape == (103, 4)
    np.testing.assert_array_equal(values[:, 0], np.arange(103, dtype=np.float32))


def test_byte_partition_keeps_line_start_boundary(tmp_path):
    path = tmp_path / "equal_width.csv"
    rows = [f"{i:03d},1,2,3\n" for i in range(8)]
    path.write_text("".join(rows), encoding="ascii")
    pieces = [main.read_csv_byte_partition(path, rank, 8)[0] for rank in range(8)]
    values = np.concatenate(pieces, axis=0)
    np.testing.assert_array_equal(values[:, 0], np.arange(8, dtype=np.float32))


def test_single_rank_mlem_matches_identity_case():
    sysmat = torch.eye(3)
    projection = torch.tensor([[2.0], [3.0], [5.0]])
    rotation = torch.arange(1, 4, dtype=torch.long).reshape(3, 1)
    sensitivity = torch.ones((3, 1))
    result = recon.run_single_mlem_dist(
        "test", sysmat, projection, rotation, rotation, sensitivity,
        iterations=1, save_step=1, rank=0,
    )
    torch.testing.assert_close(result.image, projection)
    torch.testing.assert_close(result.history[-1], projection.squeeze(1))


def test_forward_project_shard_uses_view_average():
    sysmat = torch.eye(2)
    rotation = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    image = torch.tensor([[4.0], [8.0]])
    result = recon.forward_project_shard(sysmat, rotation, image)
    expected = torch.tensor([[2.0, 4.0], [4.0, 2.0]])
    torch.testing.assert_close(result, expected)
