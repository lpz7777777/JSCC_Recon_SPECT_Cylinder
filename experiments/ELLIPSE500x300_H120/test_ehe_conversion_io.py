"""Storage-layout and exact numerical regression for conversion-only repair."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from ehe_conversion_io import fill_slab, memory_digest, publish_array, reference_layer
from ehe_gpu_pipeline import stencil


class ConversionIOTests(unittest.TestCase):
    def test_every_bin_point_layer_and_volume_matches_original(self):
        rng = np.random.default_rng(20261009)
        bins, points = 137, 43  # includes the incomplete last 64-bin block
        raw = rng.random((bins, 10, 85, 85), dtype=np.float32)
        x = np.r_[[-252., 252., 0.], rng.uniform(-252, 252, points - 3)]
        y = np.r_[[252., -252., 0.], rng.uniform(-252, 252, points - 3)]
        interp = stencil(x, y)
        volume = rng.uniform(27, 432, 40 * points)
        cart = np.full((bins, 40, 85, 85), np.nan, dtype='<f4')
        polar = np.full((40 * points, bins), np.nan, dtype='<f4')
        for slab in range(4):
            fill_slab(cart, polar, raw, slab, volume, interp)
            self.assertTrue(np.array_equal(cart[:, slab * 10:(slab + 1) * 10], raw))
            for k in range(10):
                z = slab * 10 + k
                expected = reference_layer(raw[:, k], volume[z * points:(z + 1) * points], interp)
                self.assertTrue(np.array_equal(polar[z * points:(z + 1) * points], expected))
        self.assertTrue(np.isfinite(cart).all() and np.isfinite(polar).all())

    def test_sequential_bytes_and_no_overwrite(self):
        a = np.arange(17 * 137, dtype='<f4').reshape(17, 137)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'SysMat_polar'
            receipt = publish_array(path, a)
            self.assertEqual(receipt['sha256'], memory_digest(a))
            self.assertEqual(path.read_bytes(), a.tobytes(order='C'))
            with self.assertRaises(ValueError):
                publish_array(path, a + 1)
            self.assertEqual(path.read_bytes(), a.tobytes(order='C'))

    def test_noncontiguous_and_partial_output_refused(self):
        a = np.ones((17, 137), dtype='<f4')
        with self.assertRaises(ValueError):
            memory_digest(a[:, ::2])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'SysMat_polar'
            path.with_name(path.name + '.writing').write_bytes(b'preserve partial evidence')
            with self.assertRaises(ValueError):
                publish_array(path, a)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
