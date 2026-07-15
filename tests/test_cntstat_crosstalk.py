from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import torch

from recon_osem_local_cntstat import (
    build_random_bin_subsets,
    forward_project_local_cntstat,
    get_weight_single,
    run_recon_osem_local_cntstat,
)
from main_local_multi_energy_cntstat import (
    build_crosstalk_tasks,
    run_count_level,
)


class CntStatCrossTalkTests(unittest.TestCase):
    def test_weight_uses_additive_background_in_denominator(self):
        system = torch.eye(2, dtype=torch.float32)
        image = torch.tensor([[5.0], [5.0]])
        background = torch.tensor([[2.0], [3.0]])
        observed = torch.tensor([[7.0], [8.0]])

        weight = get_weight_single(system, observed, image, background)

        torch.testing.assert_close(weight, torch.ones_like(weight))

    def test_random_subsets_keep_all_rows_aligned(self):
        system = torch.tensor(
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]
        )
        projection = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        background = torch.tensor([[101.0], [102.0], [103.0], [104.0]])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)

        systems, projections, backgrounds = build_random_bin_subsets(
            [system], [projection], 2, generator, [background]
        )

        for subset_systems, subset_projections, subset_backgrounds in zip(
            systems, projections, backgrounds
        ):
            row_ids = (subset_systems[0][:, 0] / 10).to(torch.int64)
            torch.testing.assert_close(
                subset_projections[0][:, 0], row_ids.to(torch.float32)
            )
            torch.testing.assert_close(
                subset_backgrounds[0][:, 0], row_ids.to(torch.float32) + 100
            )

    def test_additive_background_mlem_recovers_signal(self):
        system = torch.eye(2, dtype=torch.float32)
        rotmat = torch.tensor([[1], [2]], dtype=torch.int64)
        observed = torch.tensor([[7.0], [8.0]], dtype=torch.float32)
        background = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
        iter_arg = Namespace(sc=100, save_iter_step=10, osem_subset_num=1, seed=7)
        sensitivity = Namespace(s=torch.ones([2, 1], dtype=torch.float32))

        with tempfile.TemporaryDirectory() as output_dir:
            image = run_recon_osem_local_cntstat(
                [system],
                [rotmat],
                [rotmat],
                [observed],
                iter_arg,
                sensitivity,
                output_dir + "/",
                torch.device("cpu"),
                output_name="test",
                additive_background_all=[background],
            )

        torch.testing.assert_close(
            image.cpu(), torch.tensor([[5.0], [5.0]]), atol=1e-5, rtol=1e-5
        )

    def test_multiview_background_uses_measured_cntstat_scale(self):
        system = torch.eye(2, dtype=torch.float32)
        rotmat = torch.tensor([[1, 1], [2, 2]], dtype=torch.int64)
        background = torch.tensor([[2.0, 2.0], [3.0, 3.0]], dtype=torch.float32)
        observed = torch.tensor([[4.5, 4.5], [6.0, 6.0]], dtype=torch.float32)
        iter_arg = Namespace(sc=100, save_iter_step=10, osem_subset_num=1, seed=7)
        sensitivity = Namespace(s=torch.ones([2, 1], dtype=torch.float32))

        with tempfile.TemporaryDirectory() as output_dir:
            image = run_recon_osem_local_cntstat(
                [system],
                [rotmat],
                [rotmat],
                [observed],
                iter_arg,
                sensitivity,
                output_dir + "/",
                torch.device("cpu"),
                output_name="multiview_test",
                additive_background_all=[background],
            )

        torch.testing.assert_close(
            image.cpu(), torch.tensor([[5.0], [6.0]]), atol=1e-5, rtol=1e-5
        )

    def test_cross_forward_projection_respects_rotation(self):
        cross = torch.tensor([[1.0, 10.0]], dtype=torch.float32)
        rotmat = torch.tensor([[1, 2], [2, 1]], dtype=torch.int64)
        image_440 = torch.tensor([[2.0], [3.0]], dtype=torch.float32)

        projection = forward_project_local_cntstat(
            cross, rotmat, image_440, torch.device("cpu")
        )

        torch.testing.assert_close(projection, torch.tensor([[16.0, 11.5]]))

    def test_full_two_window_chain_recovers_corrected_images(self):
        direct = torch.eye(2, dtype=torch.float32)
        cross = torch.diag(torch.tensor([0.2, 0.5], dtype=torch.float32))
        rotmat = torch.tensor([[1], [2]], dtype=torch.int64)
        image_218_true = torch.tensor([[5.0], [6.0]])
        image_440_true = torch.tensor([[3.0], [4.0]])
        projection_440 = direct @ image_440_true
        projection_218 = direct @ image_218_true + cross @ image_440_true
        args = Namespace(
            e0_list=[0.218, 0.440],
            single_sc_iter=100,
            single_sc_save_step=10,
            osem_subset_num=1,
            seed=11,
            overwrite_existing=True,
            cross_talk_scale=1.0,
        )
        task_map = build_crosstalk_tasks(args)
        factors = [
            {
                "e0": 0.218,
                "sysmat": direct,
                "rotmat": rotmat,
                "rotmat_inv": rotmat,
                "sensi": torch.ones([2, 1]),
                "total_bins": 2,
            },
            {
                "e0": 0.440,
                "sysmat": direct,
                "rotmat": rotmat,
                "rotmat_inv": rotmat,
                "sensi": torch.ones([2, 1]),
                "total_bins": 2,
            },
        ]
        cross_factor = {
            "factor_dir": "synthetic",
            "sysmat": cross,
            "rotmat": rotmat,
            "rotmat_inv": rotmat,
            "sensi": torch.sum(cross, dim=0).reshape(-1, 1),
            "total_bins": 2,
        }

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            diagnostics = run_count_level(
                args,
                task_map,
                factors,
                cross_factor,
                [projection_218, projection_440],
                output_path,
                torch.device("cpu"),
                2,
            )
            corrected = torch.from_file(
                str(output_path / "Image_S_218keV_CrossTalkCorrected"),
                shared=False,
                size=2,
                dtype=torch.float32,
            ).clone()
            contaminated = torch.from_file(
                str(output_path / "Image_S_218keV_Contaminated"),
                shared=False,
                size=2,
                dtype=torch.float32,
            ).clone()
            corrected_sum = torch.from_file(
                str(output_path / "Image_S_(440_218)keV_CrossTalkCorrected"),
                shared=False,
                size=2,
                dtype=torch.float32,
            ).clone()

        torch.testing.assert_close(corrected, image_218_true.squeeze(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(contaminated, projection_218.squeeze(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            corrected_sum,
            (image_218_true + image_440_true).squeeze(),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertLess(diagnostics["relative_l2_residual_218"], 1e-6)
        self.assertLess(diagnostics["relative_l2_residual_440"], 1e-6)


if __name__ == "__main__":
    unittest.main()
