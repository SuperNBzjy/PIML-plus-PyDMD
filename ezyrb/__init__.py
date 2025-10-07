"""Minimal EZyRB stand-ins used for testing ParametricDMD.

This lightweight module provides small implementations of the :class:`POD`
and :class:`RBF` classes required by the test-suite.  They reproduce the
behaviour that PyDMD relies on without pulling in the full EZyRB dependency.
The goal is to keep the public API compatible enough for the unit tests while
remaining completely NumPy-based.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class POD:
    """Compute a Proper Orthogonal Decomposition basis.

    The implementation mirrors the small subset of functionality that the
    original :mod:`ezyrb` package exposes and that PyDMD uses in its tests.
    """

    def __init__(self, rank: int | None = None) -> None:
        self.rank = rank
        self._basis: np.ndarray | None = None

    def fit(self, snapshots: np.ndarray) -> "POD":
        """Compute the left singular vectors of ``snapshots``.

        Parameters
        ----------
        snapshots:
            Matrix whose columns contain the states to analyse.  The computed
            basis vectors span the dominant column space of this matrix.
        """

        snapshots = np.asarray(snapshots)
        u, _s, _vh = np.linalg.svd(snapshots, full_matrices=False)
        if self.rank is None or self.rank <= 0:
            r = u.shape[1]
        else:
            r = min(self.rank, u.shape[1])
        self._basis = u[:, :r]
        return self

    def reduce(self, snapshots: np.ndarray) -> np.ndarray:
        """Project ``snapshots`` onto the POD basis."""

        if self._basis is None:
            raise RuntimeError("POD basis not computed. Call fit() first.")
        snapshots = np.asarray(snapshots)
        return np.matmul(self._basis.conj().T, snapshots)

    def expand(self, coefficients: np.ndarray) -> np.ndarray:
        """Lift coefficient vectors back to the full space."""

        if self._basis is None:
            raise RuntimeError("POD basis not computed. Call fit() first.")
        coeffs = np.asarray(coefficients)
        if coeffs.ndim == 1:
            return np.matmul(self._basis, coeffs)
        return np.matmul(self._basis, coeffs)


@dataclass
class RBF:
    """Radial Basis Function interpolator.

    The class implements a Gaussian radial basis interpolant.  It supports the
    :py:meth:`fit`/ :py:meth:`predict` workflow used by
    :class:`pydmd.ParametricDMD`.
    """

    epsilon: float = 1.0

    def __post_init__(self) -> None:
        self._centres: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    # Internal -----------------------------------------------------------------
    def _prepare(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        return points

    def _kernel(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-(self.epsilon * r) ** 2)

    def fit(self, points: np.ndarray, values: np.ndarray) -> "RBF":
        """Fit the interpolant to the provided points and values."""

        pts = self._prepare(points)
        vals = np.asarray(values)
        diffs = pts[:, None, :] - pts[None, :, :]
        r = np.linalg.norm(diffs, axis=2)
        phi = self._kernel(r)
        self._weights = np.linalg.solve(phi, vals)
        self._centres = pts
        return self

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the interpolant at ``points``."""

        if self._centres is None or self._weights is None:
            raise RuntimeError("Interpolator not fitted. Call fit() first.")

        pts = self._prepare(points)
        diffs = pts[:, None, :] - self._centres[None, :, :]
        r = np.linalg.norm(diffs, axis=2)
        phi = self._kernel(r)
        return phi @ self._weights


__all__ = ["POD", "RBF"]
