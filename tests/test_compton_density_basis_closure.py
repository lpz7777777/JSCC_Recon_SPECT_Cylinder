import unittest

import torch

from compton_event_response import (
    ComptonEventSettings,
    build_compton_cone_weights,
    build_detector_position_variance,
    normalize_event_response,
    prepare_compton_events,
)


class ComptonDensityBasisClosureTest(unittest.TestCase):
    def setUp(self):
        self.detector = torch.tensor(
            [
                [0.0, 200.0, 0.0],
                [4.0, 230.0, 0.0],
                [-4.0, 260.0, 0.0],
                [0.0, 290.0, 4.0],
            ],
            dtype=torch.float32,
        )
        self.coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [-8.0, 0.0, 0.0],
                [0.0, 0.0, 6.0], [8.0, 0.0, 6.0], [-8.0, 0.0, 6.0],
            ],
            dtype=torch.float32,
        )
        self.volumes = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        point_response = torch.tensor(
            [
                [0.05, 0.08, 0.06, 0.04, 0.03, 0.02],
                [0.03, 0.05, 0.08, 0.06, 0.04, 0.02],
                [0.02, 0.03, 0.05, 0.08, 0.06, 0.04],
                [0.04, 0.02, 0.03, 0.05, 0.08, 0.06],
            ],
            dtype=torch.float32,
        )
        self.density_response = point_response * self.volumes.unsqueeze(0)
        self.settings = ComptonEventSettings(
            energy_mev=0.440,
            energy_resolution=0.13 * (0.511 / 0.440) ** 0.5,
            energy_threshold_max_mev=2 * 0.440**2 / (0.511 + 2 * 0.440) - 0.001,
            energy_threshold_min_mev=0.05,
            energy_threshold_sum_mev=0.35,
        )
        self.events = torch.tensor(
            [[1.0, 0.12, 2.0, 0.32], [2.0, 0.10, 3.0, 0.34]], dtype=torch.float32
        )

    def test_shared_rows_use_density_basis_without_deweighting(self):
        sigma = build_detector_position_variance(self.detector, 0.0)
        prepared, diagnostics = prepare_compton_events(
            self.events, self.settings, self.detector, sigma, sigma,
            input_energies_already_smeared=True,
        )
        self.assertEqual(diagnostics.energy_rejected_events, 0)
        cone = build_compton_cone_weights(prepared, self.coordinates, self.settings)
        normalized, _, _, invalid, low_support = normalize_event_response(
            cone, prepared.cpnum1, self.density_response, 1.0
        )
        self.assertEqual(invalid, 0)
        self.assertEqual(low_support, 0)
        expected = cone * self.density_response[prepared.cpnum1 - 1]
        expected = expected / expected.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(normalized, expected, rtol=2e-6, atol=2e-8))

    def test_independent_uniform_halves_leave_uniform_density_fixed(self):
        # Identical synthetic halves emulate independent uniform draws in the
        # zero-noise limit. The density-basis sensitivity is V/N * sum(p_i).
        sigma = build_detector_position_variance(self.detector, 0.0)
        prepared, _ = prepare_compton_events(
            self.events, self.settings, self.detector, sigma, sigma,
            input_energies_already_smeared=True,
        )
        cone = build_compton_cone_weights(prepared, self.coordinates, self.settings)
        posterior, _, _, _, _ = normalize_event_response(
            cone, prepared.cpnum1, self.density_response, 1.0
        )
        source_volume = float(self.volumes.sum())
        source_photons = 2.0e6
        sensi_d = posterior.sum(dim=0) * source_volume / source_photons
        uniform_density = source_photons / source_volume
        one_step_update = posterior.sum(dim=0) / sensi_d
        self.assertTrue(torch.allclose(one_step_update, torch.full_like(one_step_update, uniform_density)))


if __name__ == "__main__":
    unittest.main()
