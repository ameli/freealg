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
from ..visualization._plot_util import plot_density
from .._algebraic_form._sheets_util import _pick_physical_root_scalar
from ._base_distribution import BaseDistribution

__all__ = ['DeformedWigner']


# ===============
# Deformed Wigner
# ===============

class DeformedWigner(BaseDistribution):
    """
    Deformed Wiger distribution

    Parameters
    ----------

    t : array_like
        Array :math:`[t_1, \\dots, t_r]`, where :math:`t_i` is the jump
        location :math:`\\delta_{t_i}` in the discrete distribution :math:`H`
        (see notes below).

    w : array_like
        Array :math:`[w_1, \\dots, w_r]`, where :math:`w_i` is the jump
        weights of :math:`\\delta_{t_i}` in the discrete distribution :math:`H`
        (see notes below). The sum of all weights must be one:
        :math:`\\sum_{i=1}^r w_i = 1`.

    sigma : float, default=0.0
        Semicircle standard deviation :math:`\\sigma` (variance is
        :math:`\\sigma^2`).

    Methods
    -------

    density
        Spectral density of distribution.

    roots
        Roots of polynomial implicitly representing Stieltjes transform

    stieltjes
        Stieltjes transform

    support
        Support intervals of distribution

    sample
        Sample from distribution.

    matrix
        Generate matrix with its empirical spectral density of distribution

    poly
        Polynomial coefficients implicitly representing the Stieltjes

    See Also
    --------

    freealg.distributions.DeformedMarchenkoPastur
    """

    # ====
    # init
    # ====

    def __init__(self, t, w, sigma=1.0):
        """
        Initialization.
        """

        # Convert t and w to numpy arrays
        self.t = numpy.asarray(t, dtype=numpy.float64)
        self.w = numpy.asarray(w, dtype=numpy.float64)

        if self.t.ndim != 1 or self.w.ndim != 1:
            raise ValueError("t and w must be one-dimensional arrays.")

        if self.t.size == 0:
            raise ValueError("t must have at least one element.")

        if self.t.size != self.w.size:
            raise ValueError("t and w must have the same size.")

        if numpy.any(self.w < 0.0):
            raise ValueError("w must be nonnegative.")

        wsum = float(numpy.sum(self.w))
        if abs(wsum - 1.0) > 1e-12:
            raise ValueError("sum of w must be one.")

        self.sigma = sigma

        # Bounds for smallest and largest eigenvalues
        self.lam_lb = float(numpy.min(self.t)) - 2.0 * float(self.sigma)
        self.lam_ub = float(numpy.max(self.t)) + 2.0 * float(self.sigma)

    # =================
    # roots poly scalar
    # =================

    def _roots_poly_scalar(self, z):
        """
        """

        # Unpack parameters
        t = self.t
        w = self.w
        sigma = self.sigma
        r = int(t.size)

        s2 = sigma * sigma

        # Build polynomial in m corresponding to
        # m prod_i (t_i - z - s2 m) - sum_i w_i prod_{j!=i}(t_j - z - s2 m) = 0
        m = numpy.poly1d([1.0, 0.0])

        prefix = [numpy.poly1d([1.0])]
        for i in range(r):
            fac = numpy.poly1d([-s2, float(t[i]) - z])
            prefix.append(prefix[-1] * fac)

        suffix = [None] * (r + 1)
        suffix[r] = numpy.poly1d([1.0])
        for i in range(r - 1, -1, -1):
            fac = numpy.poly1d([-s2, float(t[i]) - z])
            suffix[i] = suffix[i + 1] * fac

        prod = prefix[r]
        term1 = m * prod

        term2 = numpy.poly1d([0.0])
        for i in range(r):
            prod_ex = prefix[i] * suffix[i + 1]
            term2 = term2 + float(w[i]) * prod_ex

        P = term1 - term2

        return numpy.roots(P.c)

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z, max_iter=100, tol=1e-12):
        """
        """

        # Unpack parameters
        t = self.t
        w = self.w
        sigma = self.sigma
        r = int(t.size)

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

            # d_i = t_i - z - s2 m
            d = (t.reshape((r, 1)) - za.reshape((1, -1)) -
                 s2 * ma.reshape((1, -1)))

            f = ma - numpy.sum(w.reshape((r, 1)) / d, axis=0)
            fp = 1.0 - numpy.sum(w.reshape((r, 1)) * s2 / (d * d), axis=0)

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
                rts = self._roots_poly_scalar(zf[i])
                mf[i] = _pick_physical_root_scalar(zf[i], rts)
            m = mf.reshape(z.shape)

        if scalar:
            return m.reshape(())
        return m

    # =======
    # density
    # =======

    def density(self, x=None, eta=1e-3, plot=False, latex=False, save=False,
                eig=None):
        """
        Density of distribution.

        Parameters
        ----------

        x : numpy.array, default=None
            The locations where density is evaluated at. If `None`, an interval
            slightly larger than the supp interval of the spectral density
            is used.

        rho : numpy.array, default=None
            Density. If `None`, it will be computed.

        eta : float, default=1e-3
            The offset :math:`\\eta` from the real axis where the density
            is evaluated using Plemelj formula at :math:`z = x + i \\eta`.

        plot : bool, default=False
            If `True`, density is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        eig : numpy.array, default=None
            A collection of eigenvalues to compare to via histogram. This
            option is relevant only if ``plot=True``.

        Returns
        -------

        rho : numpy.array
            Density.
        """

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_ub - self.lam_lb)
            center = 0.5 * (self.lam_ub + self.lam_lb)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 500)
        else:
            x = numpy.asarray(x, dtype=numpy.float64)

        z = x + 1j * float(eta)
        m = self.stieltjes(z)
        rho = numpy.imag(m) / numpy.pi

        rho = numpy.maximum(rho, 0.0)

        if plot:
            if eig is not None:
                label = 'Theoretical'
            else:
                label = ''
            plot_density(x, rho, atoms=None, label=label, latex=latex,
                         save=save, eig=eig)

        return rho

    # =====
    # roots
    # =====

    def roots(self, z):
        """
        Roots of polynomial implicitly representing Stieltjes transform
        """

        r = int(self.t.size)

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        zf = z.ravel()
        out = numpy.empty((zf.size, r + 1), dtype=numpy.complex128)
        for i in range(zf.size):
            out[i, :] = self._roots_poly_scalar(zf[i])

        out = out.reshape(z.shape + (r + 1,))
        if scalar:
            return out.reshape((r + 1,))
        return out

    # =======
    # support
    # =======

    def support(self, y_probe=1e-6):
        """
        Support intervals of distribution
        """

        r = int(self.t.size)

        # For r = 2, use the closed-form critical point method (quartic).
        if r == 2:

            # Unpack parameters
            t1 = float(self.t[0])
            t2 = float(self.t[1])
            w1 = float(self.w[0])
            sigma = self.sigma

            w2 = 1.0 - w1

            p_a = numpy.poly1d([-1.0, t1])
            p_b = numpy.poly1d([-1.0, t2])

            pa2 = p_a * p_a
            pb2 = p_b * p_b

            eq = pa2 * pb2 - (sigma * sigma) * (w1 * pb2 + w2 * pa2)
            u_roots = numpy.roots(eq.coeffs)

            ucrit = []
            for r0 in u_roots:
                if numpy.isfinite(r0) and abs(r0.imag) < 1e-10:
                    ucrit.append(float(r0.real))
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
                rts = self._roots_poly_scalar(z)
                m = _pick_physical_root_scalar(z, rts)
                if numpy.imag(m) > thr:
                    cuts.append((edges[i], edges[i + 1]))

            return cuts

        # For r != 2, use a probing method on a fine grid.
        x = numpy.linspace(self.lam_lb, self.lam_ub, 2000)
        rho = self.density(x, eta=max(float(y_probe), 1e-8), plot=False)

        thr = 100.0 * float(y_probe)
        mask = numpy.isfinite(rho) & (rho > thr)

        if not numpy.any(mask):
            return []

        idx = numpy.where(mask)[0]
        cuts = []
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
                continue
            cuts.append((float(x[start]), float(x[prev])))
            start = k
            prev = k
        cuts.append((float(x[start]), float(x[prev])))

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
        :math:`H = \\sum_{i=1}^{r} w_i \\delta_{t_i}`.

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
        t = numpy.asarray(self.t, dtype=numpy.float64)
        w = numpy.asarray(self.w, dtype=numpy.float64)
        sigma = float(self.sigma)
        r = int(t.size)

        # RNG
        rng = numpy.random.default_rng(seed)

        # T part
        wn = w * float(n)
        n_int = numpy.floor(wn).astype(int)
        rem = int(n - numpy.sum(n_int))
        if rem > 0:
            frac = wn - n_int
            idx = numpy.argsort(frac)[::-1]
            n_int[idx[:rem]] += 1

        d = numpy.empty(n, dtype=numpy.float64)
        k = 0
        for i in range(r):
            ni = int(n_int[i])
            if ni <= 0:
                continue
            d[k:k+ni] = float(t[i])
            k += ni
        if k < n:
            d[k:] = float(t[-1])

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
        Polynomial coefficients implicitly representing the Stieltjes

        coeffs[i, j] is the coefficient of z^i m^j.
        """

        t = self.t
        w = self.w
        sigma = float(self.sigma)
        r = int(t.size)

        s2 = sigma * sigma

        # Multivariate polynomial dict: (kz, km) -> coef
        def add_poly(A, B):
            C = dict(A)
            for k, v in B.items():
                C[k] = C.get(k, 0.0) + v
            return {k: v for k, v in C.items() if v != 0.0}

        def mul_poly(A, B):
            C = {}
            for (az, am), av in A.items():
                for (bz, bm), bv in B.items():
                    k = (az + bz, am + bm)
                    C[k] = C.get(k, 0.0) + av * bv
            return {k: v for k, v in C.items() if v != 0.0}

        def scale_poly(A, s):
            if s == 0.0:
                return {}
            return {k: s * v for k, v in A.items()}

        one = {(0, 0): 1.0}
        # z_poly = {(1, 0): 1.0}
        m_poly = {(0, 1): 1.0}

        # Build prod_i (t_i - z - s2 m)
        factors = []
        prod = one
        for i in range(r):
            fac = {
                (0, 0): float(t[i]),
                (1, 0): -1.0,
                (0, 1): -s2
            }
            factors.append(fac)
            prod = mul_poly(prod, fac)

        # Term1: m * prod
        term1 = mul_poly(m_poly, prod)

        # Term2: sum_i w_i prod_{j!=i} (t_j - z - s2 m)
        term2 = {}
        for i in range(r):
            p_ex = one
            for j in range(r):
                if j == i:
                    continue
                p_ex = mul_poly(p_ex, factors[j])
            term2 = add_poly(term2, scale_poly(p_ex, float(w[i])))

        P = add_poly(term1, scale_poly(term2, -1.0))

        max_kz = max(kz for (kz, km) in P.keys()) if P else 0
        max_km = max(km for (kz, km) in P.keys()) if P else 0

        coeffs = numpy.zeros((max_kz + 1, max_km + 1), dtype=numpy.complex128)
        for (kz, km), v in P.items():
            coeffs[int(kz), int(km)] = v

        # Clean tiny numerical noise
        if coeffs.size > 0:
            max_abs = float(numpy.max(numpy.abs(coeffs)))
            if max_abs > 0.0:
                coeffs[numpy.abs(coeffs) < 1.0e-12 * max_abs] = 0.0

        return coeffs
