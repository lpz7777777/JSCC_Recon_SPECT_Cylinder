"""Cross-acquisition water NEMA scatter model for the known simulated source geometry."""

import numpy as np
from scipy.ndimage import gaussian_filter


SIDE_WINDOWS = ("440_lower", "218_lower", "218_upper")


def detector_groups(layout):
    layout = np.asarray(layout)
    if layout.ndim != 2 or layout.shape[1] != 4 or np.unique(layout, axis=0).shape != layout.shape:
        raise ValueError("Unique physical (face, layer, u, v) detector rows required")
    groups = []
    for group in np.unique(layout[:, :2], axis=0):
        idx = np.flatnonzero(np.all(layout[:, :2] == group, axis=1))
        u, v = (layout[idx, 2:] - layout[idx, 2:].min(axis=0)).astype(int).T
        groups.append((idx, u, v))
    return groups


def smooth_projection(counts, groups, spatial, angular):
    """Column-normalized Gaussian transport over real crystal bins and circular views."""
    values = np.zeros_like(counts, dtype=np.float64)
    for idx, u, v in groups:
        grid = np.zeros((1 + u.max(), 1 + v.max(), counts.shape[1]), np.float64)
        mask = np.zeros(grid.shape[:2], np.float64)
        mask[u, v] = 1
        normalization = gaussian_filter(mask, spatial, mode="constant")[u, v]
        grid[u, v] = counts[idx] / normalization[:, None]
        values[idx] = gaussian_filter(grid, (spatial, spatial, angular),
                                      mode=("constant", "constant", "wrap"))[u, v]
        # A donor-only weak floor within this face/layer keeps likelihood scores defined.
        values[idx] = .999 * values[idx] + .001 * counts[idx].mean()
    if not np.isclose(values.sum(), counts.sum(), rtol=1e-12, atol=1e-12):
        raise ValueError("Projection smoothing failed to conserve counts")
    return values


def fit_source_ratios(target_sides, donor_sides):
    """Poisson scale per emitted energy. The 440 side window identifies 440."""
    y = np.array([target_sides[name].sum() for name in SIDE_WINDOWS], dtype=float)
    donor = np.array([donor_sides[name].sum(axis=(1, 2, 3)) for name in SIDE_WINDOWS], dtype=float)
    if y[0] <= 0 or donor[0, 1] <= 0 or donor[0, 0] != 0 or np.any(donor[1:, 0] <= 0):
        raise ValueError("Need informative side counts and no donor 218-to-440 leakage")
    ratio440 = y[0] / donor[0, 1]
    cross = donor[1:, 1] * ratio440
    own = donor[1:, 0]
    lo, hi = 0., y[1:].sum() / own.sum()
    for _ in range(60):
        mid = (lo + hi) / 2
        mean = own * mid + cross
        score = np.sum(own * (y[1:] / mean - 1))
        if score > 0:
            lo = mid
        else:
            hi = mid
    ratios = np.array([(lo + hi) / 2, ratio440])
    if np.any(ratios <= 0):
        raise ValueError("Target sides do not identify positive scales for both energies")
    return ratios, y, donor @ ratios


def fit_template(donor_tagged_main, layout):
    """Choose bandwidth from two donor-only Poisson thinnings, then use all donor counts."""
    counts = donor_tagged_main[:, 1:].sum(axis=1)
    groups = detector_groups(layout)
    if counts.shape[1] != len(layout) or counts.shape[-1] != 20:
        raise ValueError("Detector/view axes disagree with frozen acquisition")
    summed = counts.sum(axis=0)
    rng = np.random.default_rng(761107)
    half = rng.binomial(summed.astype(np.int64), .5)
    other = summed - half
    scores = []
    for spatial in (.65, 1.3, 2.6, 5.2):
        for angular in (0., .8, 1.6):
            fit = smooth_projection(half, groups, spatial, angular)
            reverse = smooth_projection(other, groups, spatial, angular)
            score = float((other * np.log(fit + 1e-10)).sum() + (half * np.log(reverse + 1e-10)).sum())
            scores.append((score, spatial, angular))
    _, spatial, angular = max(scores)
    smoothed = np.stack([smooth_projection(row, groups, spatial, angular) for row in counts])
    return smoothed, dict(spatial_sigma_crystals=spatial, angular_sigma_views=angular, donor_cv_scores=scores)
