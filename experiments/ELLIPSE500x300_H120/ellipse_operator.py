"""Exact object-frame ellipse embedding for single and Compton responses.

The response matrix has complete circular columns and already contains full
polar-cell volume. Fractions are applied once, in object coordinates.
"""
from __future__ import annotations

import numpy as np


class EllipseOperator:
    def __init__(self, rotation, active_indices, fractions):
        self.rotation = np.asarray(rotation, dtype=np.int64)
        self.active_indices = np.asarray(active_indices, dtype=np.int64)
        self.fractions = np.asarray(fractions, dtype=np.float64)
        n = len(self.fractions)
        if self.rotation.shape[0] != n or np.any(self.active_indices < 0) or np.any(self.active_indices >= n):
            raise ValueError("Ellipse operator index dimensions mismatch")
        if not np.array_equal(self.active_indices, np.flatnonzero(self.fractions > 1e-13)):
            raise ValueError("Active indices and fractions disagree")
        if np.any(self.fractions < 0) or np.any(self.fractions > 1):
            raise ValueError("Invalid ellipse fractions")
        for view in range(self.rotation.shape[1]):
            if not np.array_equal(np.sort(self.rotation[:, view]), np.arange(n)):
                raise ValueError("Rotation is not a permutation")

    def embed(self, density):
        density = np.asarray(density)
        if density.shape != (len(self.active_indices),):
            raise ValueError("Density has wrong number of active cells")
        full = np.zeros(len(self.fractions), dtype=np.result_type(density, self.fractions))
        full[self.active_indices] = density * self.fractions[self.active_indices]
        return full

    def forward(self, response, density, view):
        """Return B @ P_view @ F @ density for any detector/event response B."""
        return np.asarray(response) @ self.embed(density)[self.rotation[:, view]]

    def adjoint(self, response, counts, view):
        """Return the exact transpose of forward with matching shapes."""
        projected = np.asarray(response).T @ np.asarray(counts)
        full = np.empty(len(self.fractions), dtype=projected.dtype)
        full[self.rotation[:, view]] = projected
        return full[self.active_indices] * self.fractions[self.active_indices]


def load_operator(geometry_file):
    with np.load(geometry_file) as geometry:
        return EllipseOperator(geometry["rotation"], geometry["active_indices"],
                               geometry["ellipse_fraction"])
