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

__all__ = ['CompoundPoisson']


# ================
# Compound Poisson
# ================

class CompoundPoisson(object):
    """
    Two-atom free compound Poisson model (MP-like, FD-closed).

    Notes
    -----

    This model has atom at zero with mass :math:`\\max(1-\\lambda, 0)`.

    This model is the additive free compound Poisson law whose R-transform is

    .. math::

        R(w) = \\lambda \\left(
            w_1 \\frac{t_1}{1-t_1 w} + (1-w_1) \\frac{t_2}{1-t_2 w}
        \\right),

    where :math:`\\lambda>0` is the total rate (intensity),
    :math:`t_1,t_2>0` are jump sizes, and :math:`w_1 \\in (0,1)` is the mixture
    weight for the first jump.

    The Stieltjes transform :math:`m(z)` satisfies

    .. math::

        R(-m(z)) = z + 1/m(z).

    For two atoms, clearing denominators yields a cubic polynomial in
    :math:`m`:

    .. math::

        a_3(z)m^3 + t_2(z)m^2 + t_1(z)m + a_0(z) = 0.

    FD-closure (free decompression):
        Under the FD rule that scales the argument of R, this family stays
        closed by scaling the jump sizes: :math:`a_i(t) = e^{-t} a_i`
        (keeping :math:`\\lambda,w_1` fixed). If your convention uses
        :math:`R_t(w)=R_0(e^{+t}w)`, then use :math:`a_i(t)=e^{+t}a_i` instead.
    """

    # ====
    # init
    # ====

    def __init__(self, t1, t2, w1, lam):
        """
        Parameters
        ----------
        t1, t2 : float
            Jump sizes (must be > 0). For PSD-like support, keep them > 0.

        w1 : float
            Mixture weight in (0, 1) for t1. Second weight is 1-w1.

        lam : float
            Total rate (intensity), must be > 0.
        """

        t1 = float(t1)
        t2 = float(t2)
        w1 = float(w1)
        lam = float(lam)

        if t1 <= 0.0 or t2 <= 0.0:
            raise ValueError("t1 and t2 must be > 0.")

        if not (0.0 < w1 < 1.0):
            raise ValueError("w1 must be in (0, 1).")

        if lam <= 0.0:
            raise ValueError("lam must be > 0.")

        self.t1 = t1
        self.t2 = t2
        self.w1 = w1
        self.lam = lam

    # ====================
    # roots cubic m scalar
    # ====================

    def _roots_cubic_m_scalar(self, z):
        """
        Return the three roots of the cubic equation in m at scalar z.
        """

        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        lam = float(self.lam)
        w2 = 1.0 - w1

        lam1 = lam * w1
        lam2 = lam * w2

        z = complex(z)

        # Coefficients for:
        #   a3(z)m^3 + t2(z)m^2 + t1(z)m + a0(z) = 0
        c3 = z * t1 * t2
        c2 = (z * (t1 + t2)) + (t1 * t2) * (1.0 - (lam1 + lam2))
        c1 = z + (t1 + t2) - (lam1 * t1 + lam2 * t2)
        c0 = 1.0

        coeffs = numpy.array([c3, c2, c1, c0], dtype=numpy.complex128)
        roots = numpy.roots(coeffs)
        return roots

    # =====================
    # solve m newton scalar
    # =====================

    def _solve_m_newton(self, z, m0=None, max_iter=100, tol=1e-12):
        """
        Solve R(-m) = z + 1/m for scalar z using Newton iterations.
        """

        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        lam = float(self.lam)
        w2 = 1.0 - w1

        lam1 = lam * w1
        lam2 = lam * w2

        z = complex(z)
        if m0 is None:
            m = -1.0 / z
        else:
            m = complex(m0)

        for _ in range(int(max_iter)):
            d1 = 1.0 + t1 * m
            d2 = 1.0 + t2 * m

            # f(m) = -1/m + R(-m) - z
            f = (-1.0 / m) + (lam1 * t1 / d1) + (lam2 * t2 / d2) - z

            # f'(m) = 1/m^2 - sum lam_i a_i^2/(1+a_i m)^2
            fp = (1.0 / (m * m)) - (
                lam1 * (t1 * t1) / (d1 * d1) +
                lam2 * (t2 * t2) / (d2 * d2)
            )

            step = f / fp
            m2 = m - step

            if abs(step) < tol * (1.0 + abs(m2)):
                return m2, True

            m = m2

        return m, False

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z, max_iter=100, tol=1e-12):
        """
        Physical/Herglotz branch of m(z) for the two-atom compound Poisson law.
        Fast masked Newton in m, keeping z's original shape.
        """

        # Unpack parameters
        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        lam = float(self.lam)
        w2 = 1.0 - w1

        lam1 = lam * w1
        lam2 = lam * w2

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        # m initial guess
        m = -1.0 / z
        active = numpy.isfinite(m)

        for _ in range(int(max_iter)):
            if not numpy.any(active):
                break

            idx = numpy.flatnonzero(active)
            ma = m.ravel()[idx]
            za = z.ravel()[idx]

            d1 = 1.0 + t1 * ma
            d2 = 1.0 + t2 * ma

            f = (-1.0 / ma) + (lam1 * t1 / d1) + (lam2 * t2 / d2) - za

            fp = (1.0 / (ma * ma)) - (
                lam1 * (t1 * t1) / (d1 * d1) +
                lam2 * (t2 * t2) / (d2 * d2))

            step = f / fp
            mn = ma - step

            # write back m
            m_flat = m.ravel()
            m_flat[idx] = mn

            converged = numpy.abs(step) < tol * (1.0 + numpy.abs(mn))
            still = (~converged) & numpy.isfinite(mn)

            a_flat = active.ravel()
            a_flat[idx] = still

        # Herglotz sanity: sign(Im z) == sign(Im m)
        sign = numpy.where(numpy.imag(z) >= 0.0, 1.0, -1.0)
        bad = (~numpy.isfinite(m)) | (sign * numpy.imag(m) <= 0.0)

        if numpy.any(bad):
            zb = z.ravel()
            mb = m.ravel()
            bad_idx = numpy.flatnonzero(bad)
            for i in bad_idx:
                zi = zb[i]
                m_roots = self._roots_cubic_m_scalar(zi)
                mb[i] = _pick_physical_root_scalar(zi, m_roots)
            m = mb.reshape(z.shape)

        if scalar:
            return m.reshape(())
        return m

    # =======
    # density
    # =======

    def density(self, x, eta=2e-4, max_iter=100, tol=1e-12, ac_only=True):
        """
        Density rho(x) from Im m(x + i eta) / pi.

        ac_only: bool, default=True
            If `True`, computes the absolutely-continuous (AC) part. This
            matters when for :math:`\\lambda < 1`, the distribution has an
            atom at zero of mass :math:`1 - \\lambda`, and for plotting, this
            often shows as a spike. Setting this option to true prevents this
            issue in plotting. Also, to compute support, this option allows
            proper detection of the support intervals.
        """

        z = numpy.asarray(x, dtype=numpy.float64) + 1j * float(eta)
        m = self.stieltjes(z, max_iter=max_iter, tol=tol)
        rho = numpy.imag(m) / numpy.pi

        # If it has an atom, subtract from density to get the absolutely
        # continuous part before processing the support
        if (ac_only is True) and (self.lam < 1.0):
            zr = z.real
            atom = (1.0 - self.lam) * \
                (float(eta) / (numpy.pi * (zr*zr + float(eta)**2)))
            rho = rho - atom
            rho = numpy.maximum(rho, 0.0)

        return rho

    # =====
    # roots
    # =====

    def roots(self, z):
        """
        Return all algebraic branches (roots of the cubic) at scalar z.
        """

        z = numpy.asarray(z, dtype=numpy.complex128)
        if z.ndim != 0:
            raise ValueError("roots(z) expects scalar z.")
        return self._roots_cubic_m_scalar(z.reshape(()))

    # =======
    # support
    # =======

    def support(self, eta=2e-4, n_probe=4000, thr=5e-4, x_max=None,
                x_pad=0.05, method='probe'):
        """
        Estimate support intervals by probing rho(x) on a grid.

        Parameters
        ----------
        method : {'probe'}
            Only probing is implemented here.
        """

        _ = method  # keep signature compatible

        t1 = float(self.t1)
        t2 = float(self.t2)
        lam = float(self.lam)

        if x_max is None:
            # Heuristic: scale grows ~ O(lam * max(a))
            x_max = (1.0 + lam) * max(t1, t2) * 6.0
        x_max = float(x_max)
        if x_max <= 0.0:
            raise ValueError("x_max must be > 0.")

        x = numpy.linspace(0.0, x_max, int(n_probe))
        rho = self.density(x, eta=eta, ac_only=True)

        mask = rho > float(thr)
        if not numpy.any(mask):
            return []

        idx = numpy.flatnonzero(mask)
        # split contiguous indices into intervals
        splits = numpy.where(numpy.diff(idx) > 1)[0]
        starts = numpy.r_[idx[0], idx[splits + 1]]
        ends = numpy.r_[idx[splits], idx[-1]]

        intervals = []
        for s, e in zip(starts, ends):
            xa = x[int(s)]
            xb = x[int(e)]
            pad = float(x_pad) * (xb - xa)
            intervals.append((float(max(0.0, xa - pad)), float(xb + pad)))

        return intervals

    # ==========
    # rho scalar
    # ==========

    def rho_scalar(self, x, eta=2e-4, max_iter=100, tol=1e-12):
        """
        Scalar density helper (returns float).
        """

        x = float(x)
        z = x + 1j * float(eta)
        m = self.stieltjes(z, max_iter=max_iter, tol=tol)
        return float(numpy.imag(m) / numpy.pi)

    # ======
    # matrix
    # ======

    def matrix(self, size, seed=None):
        """
        Generate a symmetric random matrix whose ESD approximates this law.

        Parameters
        ----------
        size : int
            Matrix size n.

        seed : int, default=None
            RNG seed.

        Returns
        -------
        A : numpy.ndarray
            Symmetric matrix (n x n).

        Notes
        -----

        Use a sum of two independent (rotationally invariant) Wishart terms,
        which are asymptotically free:

        .. math::

            A = s_1 \\frac{1}{m_1} \\mathbf{Z}_1 \\mathbf{Z}_1^{\\intercal} +
            s_2 * \\frac{1}{m_2} \\mathbf{Z}_2 \\mathbf{Z}_2^{\\intefcal},

        where :math:`\\mathbf{Z}_i` are :math:`n \\times m_`i` i.i.d.
        :math:`N(0,1)`. Choose aspect ratios :math:`c_i = n/m_i` and
        scales :math:`s_i` so each term has R-transform

        .. math::

            R_i(w) = \\lambda_i \\frac{a_i}{(1 - a_i w},

        with :math:`\\lambda_1 = \\lambda w1`,
        :math:`\\lambda_2 = \\lambda (1-w_1)`. This is achieved by setting

        .. math::

            c_i = 1 / \\lambda_i,
            s_i = a_i * \\lambda_i.
        """

        n = int(size)
        if n <= 0:
            raise ValueError("size must be a positive integer.")

        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        lam = float(self.lam)
        w2 = 1.0 - w1

        lam1 = lam * w1
        lam2 = lam * w2

        rng = numpy.random.default_rng(seed)

        A = numpy.zeros((n, n), dtype=numpy.float64)

        # term 1
        c1 = 1.0 / lam1
        m1 = max(1, int(round(n / c1)))
        s1 = t1 * lam1
        Z1 = rng.standard_normal((n, m1))
        A += s1 * (Z1 @ Z1.T) / float(m1)

        # term 2
        c2 = 1.0 / lam2
        m2 = max(1, int(round(n / c2)))
        s2 = t2 * lam2
        Z2 = rng.standard_normal((n, m2))
        A += s2 * (Z2 @ Z2.T) / float(m2)

        return A

    # ====
    # poly
    # ====

    def poly(self):
        """
        Return a_coeffs for the exact cubic P(z,m)=0 of the two-atom free
        compound Poisson model.

        a_coeffs[i, j] is the coefficient of z^i m^j.
        Shape is (2, 4) since deg_z=1 and deg_m=3.

        Coefficients match _roots_cubic_m_scalar.
        """

        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        w2 = 1.0 - w1
        lam = float(self.lam)

        lam1 = lam * w1
        lam2 = lam * w2

        a = numpy.zeros((2, 4), dtype=numpy.complex128)

        # c3 = z * t1 * t2
        a[0, 3] = 0.0
        a[1, 3] = t1 * t2

        # c2 = z (t1+t2) + t1 t2 (1 - (lam1+lam2))
        a[0, 2] = (t1 * t2) * (1.0 - (lam1 + lam2))
        a[1, 2] = (t1 + t2)

        # c1 = z + (t1+t2) - (lam1 t1 + lam2 t2)
        a[0, 1] = (t1 + t2) - (lam1 * t1 + lam2 * t2)
        a[1, 1] = 1.0

        # c0 = 1
        a[0, 0] = 1.0
        a[1, 0] = 0.0

        return a
