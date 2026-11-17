"""Small CPU regressions; synthetic responses are explicitly not physical data."""
import io
from pathlib import Path
import unittest
import numpy as np
import torch
from cartesian import Geometry, SHAPE, axes, field_coordinates, stencil, energy_rows, ViewResponse
from common import read

ROOT = Path(__file__).resolve().parent


class CartesianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.geo = Geometry(ROOT/'cartesian_geometry.npz')
        cls.coords = field_coordinates()

    def test_native_HE_support_and_volume(self):
        self.assertEqual(self.geo.active_count, 227040)
        self.assertEqual(self.geo.fractions_xyz.shape, SHAPE)
        x, y, z = axes()
        self.assertEqual((x[0], x[-1], y[0], y[-1], z[0], z[-1]), (-250, 250, -250, 250, -66, 66))
        self.assertLess(abs(self.geo.fractions_xyz.sum()*64/(np.pi*250*150*120)-1), 1e-12)
        self.assertLess(abs(self.geo.reference_fraction_numpy.sum()*64/(np.pi*255**2*120)-1), 1e-12)
        self.assertTrue(np.array_equal(np.flatnonzero(self.geo.fractions_xyz.ravel()>0), self.geo.active_numpy))

    def test_exact_cartesian_rotation_of_affine_field(self):
        # Affine synthetic response has an analytically known bilinear value.
        rows = torch.tensor((2+self.coords[:, 0]/600+self.coords[:, 1]/900)[None, :])
        for v in range(20):
            phi = v*np.pi/10
            x, y, _ = self.geo.coords_numpy.T
            expected = 2+(x*np.cos(phi)+y*np.sin(phi))/600+(y*np.cos(phi)-x*np.sin(phi))/900
            actual = self.geo.compact(rows, v).numpy()[0]
            # Float32 stencil/fraction are intentional original solver precision.
            np.testing.assert_allclose(actual, expected*self.geo.fraction.numpy(), rtol=2e-7, atol=2e-7)
            i, w = stencil(self.geo.coords_numpy, v)
            np.testing.assert_allclose(w.sum(0), 1, atol=1e-15)
            self.assertGreaterEqual(i.min(), 0)
            self.assertLess(i.max(), 507000)

    def test_volume_exactly_once_and_outside_zero(self):
        rows = torch.full((1, 507000), 64., dtype=torch.float32)
        x = torch.ones((self.geo.active_count, 1))
        for v in [0, 3, 19]:
            actual = float((self.geo.compact(rows, v)@x).double())
            self.assertLess(abs(actual/(np.pi*250*150*120)-1), 2e-6)
        full = self.geo.full_image(x.numpy())
        self.assertTrue((full[self.geo.fractions_xyz == 0] == 0).all())
        self.assertTrue((full[self.geo.fractions_xyz > 0] == 1).all())

    def test_adjoint_sensitivity_cache_and_short_MLEM(self):
        import torch_active_operator as op
        from single_checkpoint_mlem import single_mlem_checkpointed
        # Smaller detector count, complete real grid, explicitly synthetic H.
        q = self.coords
        rows = torch.tensor(np.stack([1+.2*np.cos(q[:, 0]/80),
            1+.1*np.sin(q[:, 1]/40), 1+.1*np.cos(q[:, 2]/10)]), dtype=torch.float32)
        provider = ViewResponse(rows, self.geo)
        original = op.ViewResponse(rows, self.geo, 'none')
        s = provider.sensitivity()
        self.assertTrue(torch.equal(s, original.sensitivity()))
        x = torch.linspace(.2, 1, self.geo.active_count, dtype=torch.float64)[:, None]
        y = torch.tensor([[.3], [.5], [.8]], dtype=torch.float64)
        for v in [0, 1, 9, 19]:
            h = provider.matrix(v).double()
            lhs, rhs = ((h@x)*y).sum(), (x*(h.T@y)).sum()
            self.assertLess(float((lhs-rhs).abs()/lhs.abs()), 1e-12)
        observation = torch.arange(60, dtype=torch.float32).reshape(3, 20)  # includes zero count
        for background in [None, torch.ones_like(observation)*.5]:
            a, ah = single_mlem_checkpointed(provider, observation, s, 3, 1, background)
            b, bh = op.single_mlem(original, observation, s, 3, 1, background)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(ah, bh))
        provider.close()

    def test_actual_v5_event_tiling_preserves_width_and_likelihood(self):
        from compton_event_response import ComptonEventSettings, prepare_compton_events, build_detector_position_variance
        from detector_csv import load_detector_coordinates
        from compton_energy_probability_v5 import ContinuousTransferLaw, normalized_proxy_response
        det = torch.tensor(load_detector_coordinates(ROOT/'Detector.csv', 10496), dtype=torch.float32)
        var = build_detector_position_variance(det, 0.)
        raw = np.load(ROOT/'calibration_rows.npy')[:3].copy()
        settings = ComptonEventSettings(.440, .13*np.sqrt(.511/.440),
            2*.440**2/(.511+2*.440)-.001, .05, .350, geometry_mode='stable_float64')
        p, _ = prepare_compton_events(torch.tensor(raw), settings, det, var, var,
                                      input_energies_already_smeared=True)
        self.assertEqual(p.count, 3)
        law = ContinuousTransferLaw.load(ROOT/'transfer_training_summary.json')
        coords = torch.tensor(self.coords[np.linspace(100, 506000, 67).astype(int)], dtype=torch.float32)
        # Fixed actual events, SYNTHETIC positive detector efficiency. Tests only
        # kernel arithmetic/tile boundaries, not new physical matrix accuracy.
        rng = np.random.default_rng(20261115)
        b = rng.uniform(.1, 2., (10496, 67)).astype(np.float32)
        expected = normalized_proxy_response(p, coords, torch.from_numpy(b), law,
            backend='tail16_mid2', gaussian_float32=True)
        for tile in [1, 17, 32, 67]:
            actual = energy_rows(p, coords, b, law, torch.ones(67, dtype=torch.float64), tile)
            relative = float(torch.linalg.vector_norm((actual-expected).double())/torch.linalg.vector_norm(expected.double()))
            self.assertLessEqual(relative, 1e-5)
        self.assertTrue(np.array_equal(raw, np.load(ROOT/'calibration_rows.npy')[:3]))

    def test_empty_tail_blocks_joint_equivalence(self):
        from jscc_5e10_joint_mlem import joint_mlem
        from torch_active_operator import _event_weight, _update
        class Small:
            geometry = type('G', (), dict(views=3, active_count=7))()
            full_rows = torch.ones((2, 7), dtype=torch.float32)
            def matrix(self, v):
                return self.full_rows*(v+1)
        r = Small()
        data = torch.tensor([[2, 0, 3], [1, 4, 0]], dtype=torch.float32)
        blocks = [[], [torch.ones((32, 7))], [torch.ones((3, 7))*.3]]
        ss = sum(r.matrix(v).sum(0)[:, None] for v in range(3))/3
        sc = torch.ones_like(ss)*.1  # independent synthetic matched-S fixture
        actual, history = joint_mlem(r, data, blocks, ss, sc, 4, 1)
        x, records = torch.ones((7, 1)), []
        for i in range(4):
            weight = torch.zeros_like(x)
            for v in range(3):
                h = r.matrix(v)
                weight += h.T@(data[:, v:v+1]/(h@x).clamp_min(1e-12))
                weight += _event_weight(blocks[v], x, 'cpu')
            x = _update(x, weight, ss+sc)
            records.append(x.clone())
        self.assertTrue(torch.equal(x, actual))
        self.assertTrue(torch.equal(history, torch.stack(records)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
