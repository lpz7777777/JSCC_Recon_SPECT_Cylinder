"""Independent geometry and dual-energy source checks for the shortened NEMA body."""
import json
import math
import unittest

import numpy as np

from make_nema_body_h60 import CONFIG, EXPERIMENT, body_boundary, make_truth


class NemaBodyH60Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text())
        cls.experiment = json.loads(EXPERIMENT.read_text())

    def test_standard_arc_cardinal_points(self):
        inner, _, outer, _ = body_boundary(self.config)
        for predicate, radius_top, radius_corner in (
                (inner, 147, 77), (outer, 150, 80)):
            cy = -35
            for x, y in ((0, cy + radius_top), (0, cy-radius_corner),
                         (radius_top, cy), (-radius_top, cy),
                         (70, cy-radius_corner), (-70, cy-radius_corner)):
                self.assertTrue(bool(predicate(x, y)), (x, y))
            self.assertFalse(bool(predicate(0, cy+radius_top+.01)))
            self.assertFalse(bool(predicate(0, cy-radius_corner-.01)))
            self.assertFalse(bool(predicate(radius_top+.01, cy)))

    def test_truth_concentration_and_volume(self):
        x, y, z, body, wall, lung, spheres, activity, centers, _, _ = make_truth(
            self.config, self.experiment)
        self.assertEqual(body.shape, (40, 100, 168))
        self.assertEqual(np.count_nonzero(body.reshape(40, -1).sum(axis=1)), 20)
        self.assertAlmostEqual(float(z.mean()), 0)
        for array in (body, wall, lung, *spheres.values(), *activity.values()):
            self.assertTrue(np.isfinite(array).all())
            self.assertGreaterEqual(float(array.min()), 0)
        all_sphere = sum(spheres.values())
        background = (body == 1) & (lung == 0) & (all_sphere == 0)
        self.assertGreater(int(background.sum()), 0)
        for energy in (218, 440):
            self.assertTrue(np.all(activity[energy][background] == 1))
        for item in centers:
            diameter = int(item["diameter_mm"])
            pure_sphere = spheres[diameter] == 1
            self.assertGreater(int(pure_sphere.sum()), 0, diameter)
            hot_energy = item["hot_energy_keV"]
            other_energy = 440 if hot_energy == 218 else 218
            self.assertTrue(np.all(activity[hot_energy][pure_sphere] == 10))
            self.assertTrue(np.all(activity[other_energy][pure_sphere] == 0))
        xx, yy = np.meshgrid(x, y, indexing="xy")
        outside_ellipse = ((xx/250)**2 + (yy/150)**2) > 1
        self.assertFalse(np.any((body+wall)[:, outside_ellipse] > 0))
        inner_area = .5*math.pi*(147**2+77**2) + 140*77
        outer_area = .5*math.pi*(150**2+80**2) + 140*80
        self.assertLess(abs(body.sum(dtype=np.float64)*27/(inner_area*60)-1), .001)
        self.assertLess(abs(wall.sum(dtype=np.float64)*27/((outer_area-inner_area)*60)-1), .001)


if __name__ == "__main__":
    unittest.main()
