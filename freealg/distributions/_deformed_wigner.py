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
from .._algebraic_form._sheets_util import _pick_physical_root_scalar

__all__ = ['DeformedWigner']


# ===============
# Deformed Wigner
# ===============

class DeformedWigner(object):
    """
    Deformed Wiger
    """

    # ====
    # init
    # ====

    def __init__(self, t1, t2, w1, sigma=1.0):
        """
        Initialization.
        """

        if not (0.0 <= w1 <= 1.0):
            raise ValueError("w1 must be in [0, 1].")

        self.t1 = t1
        self.t2 = t2
        self.w1 = w1
        self.sigma = sigma

    # ==================
    # roots cubic scalar
    # ==================

    def _roots_cubic_scalar(self, z):
        """
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        sigma = self.sigma

        w2 = 1.0 - w1
        s2 = sigma * sigma
        a1 = t1 - z
        a2 = t2 - z

        c3 = s2 * s2
        c2 = -s2 * (a1 + a2)
        c1 = (a1 * a2) + s2
        c0 = -(w1 * a2 + w2 * a1)

        return numpy.roots([c3, c2, c1, c0])

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z, max_iter=100, tol=1e-12):
        """
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        sigma = self.sigma

        w2 = 1.0 - w1
        s2 = sigma * sigma

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        m = -1.0 / z
        active = numpy.isfinite(m)

        for _ in range(int(max_iter)):
            if not numpy.any(active):
                break

            ma = m[active]
            za = z[active]

            d1 = (t1 - za - s2 * ma)
            d2 = (t2 - za - s2 * ma)

            f = ma - (w1 / d1 + w2 / d2)
            fp = 1.0 - (w1 * s2 / (d1 * d1) + w2 * s2 / (d2 * d2))

            step = f / fp
            ma2 = ma - step
            m[active] = ma2

            conv = numpy.abs(step) < tol * (1.0 + numpy.abs(ma2))
            idx = numpy.where(active)[0]
            active[idx[conv]] = False

        sign = numpy.where(numpy.imag(z) >= 0.0, 1.0, -1.0)
        bad = (sign * numpy.imag(m) <= 0.0) | (~numpy.isfinite(m))

        if numpy.any(bad):
            zf = z.ravel()
            mf = m.ravel()
            bad_idx = numpy.where(bad.ravel())[0]
            for i in bad_idx:
                r = self._roots_cubic_scalar(zf[i])
                mf[i] = _pick_physical_root_scalar(zf[i], r)
            m = mf.reshape(z.shape)

        if scalar:
            return m.reshape(())
        return m

    # =======
    # density
    # =======

    def density(self, x, eta=1e-3):
        """
        """

        x = numpy.asarray(x, dtype=numpy.float64)
        z = x + 1j * float(eta)

        m = self.stieltjes(z)
        rho = numpy.imag(m) / numpy.pi

        return numpy.maximum(rho, 0.0)

    # =====
    # roots
    # =====

    def roots(self, z):
        """
        """

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        zf = z.ravel()
        out = numpy.empty((zf.size, 3), dtype=numpy.complex128)
        for i in range(zf.size):
            out[i, :] = self._roots_cubic_scalar(zf[i])

        out = out.reshape(z.shape + (3,))
        if scalar:
            return out.reshape((3,))
        return out

    # =======
    # support
    # =======

    def support(self, y_probe=1e-6):
        """
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        sigma = self.sigma

        w2 = 1.0 - w1

        p_a = numpy.poly1d([-1.0, t1])
        p_b = numpy.poly1d([-1.0, t2])

        pa2 = p_a * p_a
        pb2 = p_b * p_b

        eq = pa2 * pb2 - (sigma * sigma) * (w1 * pb2 + w2 * pa2)
        u_roots = numpy.roots(eq.coeffs)

        ucrit = []
        for r in u_roots:
            if numpy.isfinite(r) and abs(r.imag) < 1e-10:
                ucrit.append(float(r.real))
        ucrit.sort()

        def G(u):
            return w1 / (t1 - u) + w2 / (t2 - u)

        def z_of_u(u):
            return u - (sigma * sigma) * G(u)

        edges = []
        for u in ucrit:
            x = z_of_u(u)
            if numpy.isfinite(x):
                x = float(numpy.real(x))
                if (len(edges) == 0) or (abs(x - edges[-1]) > 1e-8):
                    edges.append(x)

        if len(edges) < 2:
            return []

        thr = 100.0 * float(y_probe)
        cuts = []
        for i in range(len(edges) - 1):
            xm = 0.5 * (edges[i] + edges[i + 1])
            z = xm + 1j * float(y_probe)
            r = self._roots_cubic_scalar(z)
            m = _pick_physical_root_scalar(z, r)
            if numpy.imag(m) > thr:
                cuts.append((edges[i], edges[i + 1]))

        return cuts

    # ======
    # matrix
    # ======

    def matrix(self, size, seed=None):
        """
        Generate matrix with the spectral density of the distribution.

        Parameters
        ----------

        size : int
            Size :math:`n` of the matrix.

        seed : int, default=None
            Seed for random number generator.

        Returns
        -------

        A : numpy.ndarray
            A matrix of the size :math:`n \\times n`.

        Notes
        -----

        Generate an :math:`n \\times n` matrix
        :math:`\\mathbf{A} = \\mathbf{T} + \\sigma \\mathbf{W}`
        whose ESD converges to
        :math:`H \\boxplus \\mathrm{SC}_{\\sigma^2}`, where
        :math:`H = w_1 \\delta_{t_1} + (1 - w_1) \\delta_{t_2}`.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import DeformedWigner
            >>> dwg = DeformedWigner(1/50)
            >>> A = dwg.matrix(2000)
        """

        n = int(size)
        if n <= 0:
            raise ValueError("size must be a positive integer.")

        # Unpack parameters
        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        sigma = float(self.sigma)

        # RNG
        rng = numpy.random.default_rng(seed)

        # T part
        n1 = int(round(w1 * n))
        n1 = max(0, min(n, n1))

        d = numpy.empty(n, dtype=numpy.float64)
        d[:n1] = t1
        d[n1:] = t2
        rng.shuffle(d)  # randomize positions
        T = numpy.diag(d)

        # W part: Symmetric Wigner with variance 1/n (up to symmetry)
        G = rng.standard_normal((n, n))
        W = (G + G.T) / numpy.sqrt(2.0 * n)

        # Compose
        A = T + sigma * W

        return A

    # ====
    # poly
    # ====

    def poly(self):
        """
        Return a_coeffs for the exact cubic P(z,m)=0 of the two-atom deformed
        Wigner model.

        a_coeffs[i, j] is the coefficient of z^i m^j.
        Shape is (deg_z+1, deg_m+1) = (3, 4).
        """

        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        w2 = 1.0 - w1
        sigma = float(self.sigma)
        s2 = sigma * sigma

        a = numpy.zeros((3, 4), dtype=numpy.complex128)

        # m^0 column (a0(z) = z - (w1 t2 + w2 t1))
        a[0, 0] = -(w1 * t2 + w2 * t1)
        a[1, 0] = 1.0
        a[2, 0] = 0.0

        # m^1 column (a1(z) = z^2 - (t1+t2)z + t1 t2 + s2)
        a[0, 1] = t1 * t2 + s2
        a[1, 1] = -(t1 + t2)
        a[2, 1] = 1.0

        # m^2 column (a2(z) = 2 s2 z - s2 (t1+t2))
        a[0, 2] = -s2 * (t1 + t2)
        a[1, 2] = 2.0 * s2
        a[2, 2] = 0.0

        # m^3 column (a3(z) = s2^2)
        a[0, 3] = s2 * s2
        a[1, 3] = 0.0
        a[2, 3] = 0.0

        return a
