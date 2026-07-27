import unittest
from pathlib import Path

import numpy as np
import torch

from compton_sparse_ops import (
    build_compton_sparse_projector,
    materialize_sparse_event_rows_to_fine,
)
from process_list_plane_sparse import get_compton_backproj_list_single_sparse
from process_list_plane_strict import (
    _build_detector_pos_sigma_sq,
    _compton_theta_from_e1,
    _compute_angle_sigma_ene_strict,
    _compute_angle_sigma_pos_strict,
    get_compton_backproj_list_single,
)
from recon_osem_local_sparse_jsccsd_only import (
    get_weight_compton_sparse,
    safe_em_update,
)
from distributed.python.recon_osem_dist_sparse_jsccsd_only import (
    get_weight_compton_sparse as get_weight_compton_sparse_dist,
    safe_em_update as safe_em_update_dist,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _detectors():
    return torch.tensor(
        [
            [-9.0, 200.0, -3.0],
            [9.0, 200.0, 3.0],
            [-9.0, 230.0, 3.0],
            [9.0, 230.0, -3.0],
            [-9.0, 260.0, -3.0],
            [9.0, 260.0, 3.0],
            [-9.0, 290.0, 3.0],
            [9.0, 290.0, -3.0],
        ],
        dtype=torch.float32,
    )


class ComptonOperatorEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coordinates = torch.from_numpy(
            np.loadtxt(
                REPO_ROOT / "Factors/440keV_RotateNum20/coor_polar_full.csv",
                delimiter=",",
                dtype=np.float32,
            )
        )
        cls.detector = _detectors()
        generator = torch.Generator().manual_seed(20260722)
        cls.sysmat = 0.05 + torch.rand(
            (cls.detector.size(0), cls.coordinates.size(0)), generator=generator
        )
        # Distinct first-detector rows make an off-by-one row lookup observable.
        cls.sysmat *= torch.arange(1, cls.detector.size(0) + 1).float().unsqueeze(1)
        cls.events = torch.tensor(
            [
                [1.0, 0.100, 3.0, 0.340],
                [2.0, 0.130, 5.0, 0.310],
                [3.0, 0.160, 7.0, 0.280],
                [4.0, 0.190, 8.0, 0.250],
            ],
            dtype=torch.float32,
        )

    def test_full_grid_sparse_rows_equal_dense_rows(self):
        projector = build_compton_sparse_projector(
            self.coordinates, theta_stride=1, z_stride=1, rotate_num=20
        )
        dense, _, _ = get_compton_backproj_list_single(
            self.sysmat,
            self.detector,
            self.coordinates,
            self.events,
            0.0,
            0.0,
            0.440,
            0.1 * (0.662 / 0.440) ** 0.5,
            2 * 0.440**2 / (0.511 + 2 * 0.440) - 0.001,
            0.05,
            0.40,
            torch.device("cpu"),
            input_energies_already_smeared=True,
        )
        packed, _, _ = get_compton_backproj_list_single_sparse(
            self.sysmat,
            self.detector,
            projector,
            self.events,
            0.0,
            0.0,
            0.440,
            0.1 * (0.662 / 0.440) ** 0.5,
            2 * 0.440**2 / (0.511 + 2 * 0.440) - 0.001,
            0.05,
            0.40,
            torch.device("cpu"),
            input_energies_already_smeared=True,
        )
        sparse, valid = materialize_sparse_event_rows_to_fine(
            packed, self.sysmat, projector
        )
        self.assertTrue(torch.all(valid))
        self.assertEqual(dense.shape, sparse.shape)
        self.assertTrue(torch.allclose(dense, sparse, rtol=2e-6, atol=2e-8))

        image = torch.linspace(0.5, 1.5, self.coordinates.size(0)).reshape(-1, 1)
        dense_weight = dense.T @ (1.0 / (dense @ image))
        sparse_weight = get_weight_compton_sparse(
            packed, self.sysmat, image, projector
        )
        distributed_weight = get_weight_compton_sparse_dist(
            packed, self.sysmat, image, projector
        )
        self.assertTrue(
            torch.allclose(dense_weight, sparse_weight, rtol=2e-6, atol=2e-8)
        )
        self.assertTrue(
            torch.allclose(sparse_weight, distributed_weight, rtol=0, atol=0)
        )
        sensitivity = dense.sum(dim=0).reshape(-1, 1).clamp_min(1e-12)
        local_update = safe_em_update(image, sparse_weight, sensitivity)
        distributed_update = safe_em_update_dist(
            image, distributed_weight, sensitivity
        )
        self.assertTrue(torch.equal(local_update, distributed_update))

    def test_energy_and_crystal_position_uncertainties_are_both_active(self):
        e1 = torch.tensor([0.13], dtype=torch.float32)
        theta = _compton_theta_from_e1(e1, 0.440, 0.511)
        source = torch.tensor([[[0.0, 0.0, 0.0]]])
        pos1 = self.detector[1].reshape(1, 1, 3)
        pos2 = self.detector[4].reshape(1, 1, 3)
        vector01 = pos1 - source
        vector12 = pos2 - pos1
        distance01 = torch.norm(vector01, dim=2)
        distance12 = torch.norm(vector12, dim=2)
        beta = torch.acos(
            torch.clamp(
                torch.sum(vector01 * vector12, dim=2) / (distance01 * distance12),
                -1 + 1e-7,
                1 - 1e-7,
            )
        )
        sigma_all = _build_detector_pos_sigma_sq(self.detector, 0.0)
        sigma_energy = _compute_angle_sigma_ene_strict(
            e1, 0.440, 0.1 * (0.662 / 0.440) ** 0.5, 0.511, beta, theta
        )
        sigma_position = _compute_angle_sigma_pos_strict(
            vector01,
            vector12,
            beta,
            sigma_all[1].reshape(1, 3),
            sigma_all[4].reshape(1, 3),
            include_pos1_source_leg_sigma=True,
        )
        self.assertGreater(float(sigma_energy.item()), 0.0)
        self.assertGreater(float(sigma_position.item()), 0.0)
        sigma_total = torch.sqrt(sigma_energy**2 + sigma_position**2)
        self.assertGreater(float(sigma_total.item()), float(sigma_energy.item()))
        self.assertGreater(float(sigma_total.item()), float(sigma_position.item()))

    def test_list_mlem_update_matches_direct_formula_and_increases_likelihood(self):
        response = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.7, 0.1], [0.6, 0.1, 0.3]],
            dtype=torch.float64,
        )
        sensitivity = response.sum(dim=0).reshape(-1, 1)
        image = torch.ones((3, 1), dtype=torch.float64)

        def objective(value):
            return torch.log(response @ value).sum() - (sensitivity.T @ value).squeeze()

        previous = objective(image)
        for _ in range(20):
            weight = response.T @ (1.0 / (response @ image))
            direct = image * weight / sensitivity
            updated = safe_em_update(image, weight, sensitivity)
            self.assertTrue(torch.allclose(updated, direct, rtol=0, atol=1e-14))
            current = objective(updated)
            self.assertGreaterEqual(float(current + 1e-12), float(previous))
            image, previous = updated, current


if __name__ == "__main__":
    unittest.main()
