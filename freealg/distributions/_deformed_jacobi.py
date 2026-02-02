# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy

__all__ = ['DeformedJacobi']


# ===============
# Deformed Jacobi
# ===============

class DeformedJacobi(object):
    """
    Deformed Jacobi / general MANOVA benchmark (compact support in [0, 1]).

    This class generates the MANOVA/Jacobi matrix

        B = S2 @ (S1 + S2)^{-1},

    where S1 and S2 are sample covariances with optional finite-atom population
    covariances (diagonal with repeated atom values).

    Notes
    -----
    - Eigenvalues of B lie in [0, 1] (up to numerical tolerance), hence compact
      support.
    - In the identity-population case, the limit is the classical Wachter law.
    - With nontrivial population spectra (finite atoms) on one/both sides, the
      limit is the general MANOVA / deformed Jacobi law; it can exhibit
      multiple bulks.
    - An explicit one-line algebraic polynomial P(z,m)=0 is not generally as
      simple as CP/MP; it is typically obtained by eliminating a small
      self-consistent system (not implemented here).
    """

    # ====
    # init
    # ====

    def __init__(self, c1, c2, t1=None, w1=None, t2=None, w2=None):
        """
        Parameters
        ----------
        c1, c2 : float
            Aspect ratios p/n1 -> c1 and p/n2 -> c2, must be > 0.

        t1, w1 : array_like or None
            Finite-atom population spectrum for Sigma1.
            If provided: t1 are positive atom values, w1 are weights that sum
            to 1. If None, Sigma1 is identity.

        t2, w2 : array_like or None
            Finite-atom population spectrum for Sigma2.
            If provided: t2 are positive atom values, w2 are weights that sum
            to 1. If None, Sigma2 is identity.
        """

        self.c1 = float(c1)
        self.c2 = float(c2)
        if (self.c1 <= 0.0) or (self.c2 <= 0.0):
            raise ValueError('c1 and c2 must be positive.')

        self.t1, self.w1 = self._check_atoms(t1, w1, name='Sigma1')
        self.t2, self.w2 = self._check_atoms(t2, w2, name='Sigma2')

    # ==========
    # check atoms
    # ==========

    @staticmethod
    def _check_atoms(t, w, name='Sigma'):
        if (t is None) and (w is None):
            return None, None
        if (t is None) or (w is None):
            raise ValueError(f'{name}: both t and w must be provided.')

        t = numpy.asarray(t, dtype=numpy.float64).ravel()
        w = numpy.asarray(w, dtype=numpy.float64).ravel()
        if t.size != w.size:
            raise ValueError(f'{name}: t and w must have the same length.')
        if numpy.any(t <= 0.0):
            raise ValueError(f'{name}: atom values must be positive.')
        if numpy.any(w <= 0.0):
            raise ValueError(f'{name}: weights must be positive.')
        s = float(numpy.sum(w))
        if not numpy.isfinite(s) or (abs(s - 1.0) > 1e-10):
            w = w / s
        return t, w

    # =======
    # support
    # =======

    def support(self):
        """
        Return the ambient compact support interval [0, 1].

        Note: the true continuous support is typically a subset of [0, 1] and
        can be multi-interval depending on parameters.
        """

        return [(0.0, 1.0)]

    # ======================
    # population (diag vector)
    # ======================

    @staticmethod
    def _population_diag(p, t, w, rng):
        """
        Build a diagonal population vector of length p with finitely many atom
        values repeated according to weights.
        """

        if (t is None) or (w is None):
            return numpy.ones(p, dtype=numpy.float64)

        # Sample atom labels then map to values.
        idx = rng.choice(t.size, size=p, replace=True, p=w)
        return t[idx]

    # ======
    # matrix
    # ======

    def matrix(self, size, seed=None):
        """
        Generate a MANOVA/Jacobi matrix B whose ESD approximates the deformed
        Jacobi limit.

        Parameters
        ----------
        size : int
            Dimension p of the matrix B.

        seed : int or None
            Random seed.

        Returns
        -------
        B : ndarray, shape (p, p)
            Symmetric matrix with eigenvalues (numerically) in [0, 1].
        """

        p = int(size)
        if p <= 0:
            raise ValueError('size must be a positive integer.')

        rng = numpy.random.default_rng(seed)

        # Choose sample sizes from aspect ratios: p/nk ~ ck -> nk ~ p/ck
        n1 = int(numpy.round(p / self.c1))
        n2 = int(numpy.round(p / self.c2))
        n1 = max(n1, 1)
        n2 = max(n2, 1)

        # Population diagonals
        d1 = self._population_diag(p, self.t1, self.w1, rng)
        d2 = self._population_diag(p, self.t2, self.w2, rng)

        # Data matrices with variance 1/n
        Z1 = rng.standard_normal((p, n1)) / numpy.sqrt(float(n1))
        Z2 = rng.standard_normal((p, n2)) / numpy.sqrt(float(n2))

        # Xk = Sigma_k^{1/2} Zk; with diagonal Sigma, this is scaling rows
        X1 = (numpy.sqrt(d1)[:, None]) * Z1
        X2 = (numpy.sqrt(d2)[:, None]) * Z2

        # Sample covariances
        S1 = X1 @ X1.T
        S2 = X2 @ X2.T

        # B = S2 (S1+S2)^{-1} symmetrized
        M = S1 + S2
        # Solve M^{-1} times S2 without explicit inverse
        B = numpy.linalg.solve(M, S2)
        B = 0.5 * (B + B.T)

        return B

    # =========
    # poly
    # =========

    def poly(self):
        """
        Placeholder for an explicit algebraic polynomial P(z,m)=0.

        For the identity-population (Wachter) case, P is quadratic.
        For general finite-atom populations, P is obtained by eliminating a
        small self-consistent system (not implemented here).
        """

        raise NotImplementedError(
            'DeformedJacobi.poly(): explicit P(z,m)=0 is obtained by '
            'eliminating the MANOVA fixed-point system; not implemented.'
        )
