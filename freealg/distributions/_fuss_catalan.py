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

__all__ = ['FussCatalan']


# ============
# Fuss-Catalan
# ============

class FussCatalan(object):
    """
    Fuss-Catalan (a.k.a. free Bessel / Raney) distribution.

    This is the law of squared singular values of a product of :math:`p`
    independent (square) Ginibre matrices in the large-n limit. For :math:`p=1`
    this reduces to the Marchenko--Pastur law with :math:`c = 1`.

    Parameters
    ----------

    p : int, default=2
        Order of the Fuss-Catalan distribution. :math:`p = 1` is MP
        (:math:`c=1`).

    Notes
    -----

    Let :math:`p \\geq 1` be an integer. The Stieltjes transform :math:`m(z)`
    of the :math:`p`-th Fuss-Catalan distribution solves the algebraic equation

    ..math::

        (-1)^p z^p m(z)^{p+1} - z m(z) - 1 = 0,

    where the physical branch is selected by the Herglotz condition
    Im m(z) > 0 for Im z > 0.

    Equivalently, in terms of w(z) = -m(z):

    .. math::

        z = (1 + w)^{p+1} / w^p.

    The support is a single interval [0, x_max] with

    .. math::

        x_max = (p+1)^{p+1} / p^p.

    (All in the standard normalization where the mean is 1.)

    **Application:**

    This law might be applicable to the Jacobian of neural netowkrs.
    """

    # ====
    # init
    # ====

    def __init__(self, p=2):
        """
        Initialization.
        """

        p = int(p)

        if p < 1:
            raise ValueError("p must be an integer >= 1.")

        self.p = p

    # ============
    # roots scalar
    # ============

    def _roots_scalar(self, z):
        """
        Return all algebraic roots of the defining polynomial at scalar z.
        """

        p = int(self.p)
        z = numpy.asarray(z, dtype=numpy.complex128).reshape(())
        coeffs = numpy.zeros(p + 2, dtype=numpy.complex128)

        # Polynomial in m:
        #   (-1)^p z^p m^{p+1} - z m - 1 = 0
        coeffs[0] = ((-1.0)**p) * (z**p)       # m^{p+1}
        coeffs[-2] = -z                        # m^1
        coeffs[-1] = -1.0                      # m^0

        return numpy.roots(coeffs)

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z, max_iter=100, tol=1e-12):
        """
        Compute the physical Stieltjes transform m(z) by Newton, with robust
        fallback to algebraic roots + physical picking.
        """

        p = int(self.p)

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        # Initial guess: m ~ -1/z for |z| large
        m = -1.0 / z
        active = numpy.isfinite(m)

        # Newton on f(m) = (-1)^p z^p m^{p+1} - z m - 1
        for _ in range(int(max_iter)):
            if not numpy.any(active):
                break

            ma = m[active]
            za = z[active]

            # f and f'
            f = ((-1.0)**p) * (za**p) * (ma**(p + 1)) - za * ma - 1.0
            fp = ((-1.0)**p) * (za**p) * (p + 1) * (ma**p) - za

            step = f / fp
            mn = ma - step
            m[active] = mn

            conv = numpy.abs(step) < tol * (1.0 + numpy.abs(mn))
            idx = numpy.where(active)[0]
            active[idx[conv]] = False

        # Herglotz sanity: sign(Im z) == sign(Im m)
        sign = numpy.where(numpy.imag(z) >= 0.0, 1.0, -1.0)
        bad = (~numpy.isfinite(m)) | (sign * numpy.imag(m) <= 0.0)

        if numpy.any(bad):
            zb = z.ravel()
            mb = m.ravel()
            bad_idx = numpy.flatnonzero(bad)
            for i in bad_idx:
                zi = zb[i]
                r = self._roots_scalar(zi)
                mb[i] = _pick_physical_root_scalar(zi, r)
            m = mb.reshape(z.shape)

        if scalar:
            return m.reshape(())
        return m

    # =======
    # density
    # =======

    def density(self, x, eta=2e-4):
        """
        Density rho(x) from Im m(x + i eta) / pi.
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
        Return all algebraic branches at scalar z.
        """

        z = numpy.asarray(z, dtype=numpy.complex128)
        if z.ndim != 0:
            raise ValueError("roots(z) expects scalar z.")
        return self._roots_scalar(z.reshape(()))

    # =======
    # support
    # =======

    def support(self):
        """
        Return the single support interval [0, x_max].
        """

        p = float(self.p)
        x_max = ((p + 1.0)**(p + 1.0)) / (p**p)
        return [(0.0, float(x_max))]

    # ======
    # matrix
    # ======

    def matrix(self, size, seed=None):
        """
        Generate a PSD random matrix whose ESD approximates Fuss-Catalan(p).

        Construction
        ------------
        Let G_k be i.i.d. n x n Gaussian (Ginibre) matrices. Define

            P = (1/sqrt(n)) G_1 (1/sqrt(n)) G_2 ... (1/sqrt(n)) G_p,
            A = P P^T.

        Then the ESD of A converges to Fuss-Catalan(p) as n->infty.
        """

        p = int(self.p)
        n = int(size)
        if n <= 0:
            raise ValueError("size must be a positive integer.")

        rng = numpy.random.default_rng(seed)

        P = numpy.eye(n, dtype=numpy.float64)
        scale = 1.0 / numpy.sqrt(float(n))
        for _ in range(p):
            G = rng.standard_normal((n, n))
            P = P @ (scale * G)

        A = P @ P.T
        return A

    # ====
    # poly
    # ====

    def poly(self):
        """
        Return coeffs for the exact polynomial P(z,m)=0.

        P(z,m) = (-1)^p z^p m^{p+1} - z m - 1.

        coeffs[i, j] is the coefficient of z^i m^j.
        Shape is (deg_z+1, deg_m+1) = (p+1, p+2).
        """

        p = int(self.p)
        a = numpy.zeros((p + 1, p + 2), dtype=numpy.complex128)

        # constant: -1
        a[0, 0] = -1.0

        # - z m
        a[1, 1] = -1.0

        # (-1)^p z^p m^{p+1}
        a[p, p + 1] = ((-1.0)**p)

        return a
