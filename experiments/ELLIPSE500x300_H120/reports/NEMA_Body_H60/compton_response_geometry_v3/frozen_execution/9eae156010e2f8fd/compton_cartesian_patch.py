"""Bounded sampled A patch; interpolate within its physical box only."""
import numpy as np


class CartesianPatch:
    def __init__(self, axes):
        self.axes = tuple(np.asarray(a, dtype=float) for a in axes)
        if len(self.axes) != 3 or any(len(a) < 2 or not np.all(np.diff(a) > 0)
                                      for a in self.axes):
            raise ValueError('Three strictly increasing sampled axes are required')

    def cache(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Invalid patch coordinates')
        indices, fractions = [], []
        for dim, axis in enumerate(self.axes):
            p = points[:, dim]
            if np.any(p < axis[0]-1e-10) or np.any(p > axis[-1]+1e-10):
                raise ValueError('Patch extrapolation is forbidden')
            lower = np.clip(np.searchsorted(axis, p, side='right')-1, 0, len(axis)-2)
            t = (p-axis[lower])/(axis[lower+1]-axis[lower])
            indices.append(lower); fractions.append(np.clip(t, 0, 1))
        return indices, fractions

    @staticmethod
    def evaluate(field_zyx, cache):
        (ix, iy, iz), (tx, ty, tz) = cache
        result = np.zeros(len(ix), dtype=float)
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    weight = ((tx if x else 1-tx)*(ty if y else 1-ty)*
                              (tz if z else 1-tz))
                    result += field_zyx[iz+z, iy+y, ix+x]*weight
        if not np.isfinite(result).all() or np.any(result < -1e-20):
            raise ValueError('Invalid sampled patch A')
        return np.maximum(result, 0)


def patch_specs():
    # Three small boxes at the geometric near-side rim, independent of images.
    # Even-index subsets are 3 mm; odd points provide held-out 1.5 mm samples.
    return [dict(name=name, shape=(17, 9, 5), spacing=(1.5, 1.5, 1.5),
                 shift=(0, 252, z))
            for name, z in (('near_minus', -57), ('near_middle', 0), ('near_plus', 57))]
