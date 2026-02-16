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
from ..visualization._sheets_util import _pick_physical_root_scalar
from ._base_distribution import BaseDistribution

__all__ = ['DeformedMarchenkoPastur']


# =========================
# Deformed Marchenko Pastur
# =========================

class DeformedMarchenkoPastur(BaseDistribution):
    """
    Deformed Marchenko-Pastur distribution

    Parameters
    ----------

    t : array_like
        Array :math:`[t_1, \\dots, t_r]`, where :math:`t_i` is the jump
        location :math:`\\delta_{t_i}` in the discrete distribution :math:`H`
        (see notes below). For PSD matrix, :math:`t_i > 0`.

    w : array_like
        Array :math:`[w_1, \\dots, w_r]`, where :math:`w_i` is the jump
        weights of :math:`\\delta_{t_i}` in the discrete distribution :math:`H`
        (see notes below). The sum of all weights must be one:
        :math:`\\sum_{i=1}^r w_i = 1`.

    c : float
        Ratio parameter of Marchenko-Pastur distribution, must be >= 0.

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

    freealg.distributions.DeformedWigner

    Notes
    -----

    Silverstein / companion Stieltjes variable

    For sample-covariance, free multiplicative convolution with :math:`MP_c`:
    Let :math:`u(z)` be the *companion* Stieltjes transform (often denoted
    :math:`\\underline{m})`. It satisfies the Silverstein equation:

    .. math::

        z = -1/u + c * \\mathbb{E}_H [ t / (1 + t u) ].

    For :math:`H = \\sum_i w_i \\delta_{t_i}`:

    .. math::

        z = -1/u + c \\sum_i \\frac{w_i*t_i}{1+t_i u}.

    Then the (ordinary) Stieltjes transform m(z) of
    :math:`\\mu = H \\boxtimes \\mathrm{MP}_c` is

    .. math::

        u = -(1-c)/z + c m

    (equivalently :math:`m = (u + (1-c)/z)/c` for :math:`c>0`).

    This module solves for u (degree r+1 when H has r atoms), then maps to m.

    Reference for the Silverstein equation form:

    .. math::

        z = -1/u + c \\int t/(1 + t u) dH(t).
    """

    # ====
    # init
    # ====

    def __init__(self, t, w, c=1.0):
        """
        Initialization.
        """

        t = numpy.asarray(t, dtype=numpy.float64)
        w = numpy.asarray(w, dtype=numpy.float64)

        if t.ndim != 1 or w.ndim != 1:
            raise ValueError("t and w must be one-dimensional arrays.")

        if t.size == 0:
            raise ValueError("t must contain at least one atom.")

        if t.size != w.size:
            raise ValueError("t and w must have the same length.")

        if c < 0.0:
            raise ValueError("c must be >= 0.")

        if numpy.any(t < 0.0):
            raise ValueError("All entries of t must be >= 0 for a covariance "
                             "model.")

        if numpy.any(w < 0.0):
            raise ValueError("All entries of w must be >= 0.")

        w_sum = float(numpy.sum(w))
        if not numpy.isfinite(w_sum) or abs(w_sum - 1.0) > 1e-12:
            raise ValueError("Weights w must sum to 1.")

        # Store
        self.t = t
        self.w = w
        self.c = c

        # Bounds for smallest and largest eigenvalues
        if c > 1.0:
            # In this case, there is an atom at the origin
            self.lam_lb = 0.0
        else:
            self.lam_lb = float(numpy.min(t)) * (1 - numpy.sqrt(c))**2
        self.lam_ub = float(numpy.max(t)) * (1 + numpy.sqrt(c))**2

    # ===================
    # roots poly u scalar
    # ===================

    def _roots_poly_u_scalar(self, z):
        """
        Solve the polynomial for :math:`u = \\underline{m}(z)` for
        :math:`H = \\sum_i w_i \\delta_{t_i}`.

        Note: despite the name, for r != 2 the polynomial is not cubic.
        """

        t = self.t
        w = self.w
        c = float(self.c)
        r = int(t.size)

        # Build prod_i (1 + t_i u) using explicit poly1d factors:
        # (1 + t_i u) corresponds to poly in u with coefficients [t_i, 1].
        prefix = [numpy.poly1d([1.0])]
        for i in range(r):
            fac = numpy.poly1d([float(t[i]), 1.0])     # t_i*u + 1
            prefix.append(prefix[-1] * fac)

        suffix = [None] * (r + 1)
        suffix[r] = numpy.poly1d([1.0])
        for i in range(r - 1, -1, -1):
            fac = numpy.poly1d([float(t[i]), 1.0])     # t_i*u + 1
            suffix[i] = suffix[i + 1] * fac

        prod = prefix[r]  # poly1d

        # Term1: (-1 - z u) * prod  -> polynomial (-z)*u + (-1)
        term1 = numpy.poly1d([-z, -1.0]) * prod

        # Term2: c u sum_i w_i t_i prod_{j!=i} (1 + t_j u)
        s = numpy.poly1d([0.0])
        for i in range(r):
            prod_ex = prefix[i] * suffix[i + 1]
            s = s + float(w[i] * t[i]) * prod_ex

        # c*u is poly [c, 0]
        term2 = numpy.poly1d([c, 0.0]) * s

        P = term1 + term2

        return numpy.roots(P.c)

    # ==============
    # solve u Newton
    # ==============

    def _solve_u_newton(self, z, u0=None, max_iter=100, tol=1e-12):
        """
        """

        # Unpack parameters
        t = self.t
        w = self.w
        c = float(self.c)

        if u0 is None:
            u = -1.0 / z
        else:
            u = complex(u0)

        for _ in range(int(max_iter)):
            d = 1.0 + t * u
            f = (-1.0 / u) + c * numpy.sum(w * t / d) - z
            fp = (1.0 / (u * u)) - c * numpy.sum(w * (t * t) / (d * d))

            step = f / fp
            u2 = u - step
            if abs(step) < tol * (1.0 + abs(u2)):
                return u2, True
            u = u2

        return u, False

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z, max_iter=100, tol=1e-12):
        """
        Stieltjes transform

        Physical/Herglotz branch of m(z) for
        :math:`\\mu = H \\boxtimes \\mathrm{MP}_c` with
        :math:`H = \\sum_i w_i \\delta_{t_i}`.
        Fast masked Newton in u (companion Stieltjes), keeping z's original
        shape.
        """

        # Unpack parameters
        t = self.t
        w = self.w
        c = float(self.c)

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        if c < 0.0:
            raise ValueError("c must be >= 0.")

        if c == 0.0:
            # Degenerate case: no MP noise, spectrum equals population H
            out = numpy.zeros_like(z, dtype=numpy.complex128)
            for ti, wi in zip(t, w):
                out = out + (wi / (ti - z))
            return out.reshape(()) if scalar else out

        # u initial guess
        u = -1.0 / z
        active = numpy.isfinite(u)

        for _ in range(int(max_iter)):
            if not numpy.any(active):
                break

            # IMPORTANT: use integer indices (works for any ndim; avoids
            # boolean-mask aliasing issues)
            idx = numpy.flatnonzero(active)
            ua = u.ravel()[idx]
            za = z.ravel()[idx]

            # d has shape (r, k)
            d = 1.0 + (t[:, None] * ua[None, :])

            f = (-1.0 / ua) + c * numpy.sum((w * t)[:, None] / d, axis=0) - za
            fp = (1.0 / (ua * ua)) - c * numpy.sum((w * (t * t))[:, None] /
                                                   (d * d), axis=0)

            step = f / fp
            un = ua - step

            # write back u
            u_flat = u.ravel()
            u_flat[idx] = un

            converged = numpy.abs(step) < tol * (1.0 + numpy.abs(un))
            still = (~converged) & numpy.isfinite(un)

            # update active only at the previously-active locations
            a_flat = active.ravel()
            a_flat[idx] = still

        # Herglotz sanity: sign(Im z) == sign(Im u)
        sign = numpy.where(numpy.imag(z) >= 0.0, 1.0, -1.0)
        bad = (~numpy.isfinite(u)) | (sign * numpy.imag(u) <= 0.0)

        if numpy.any(bad):
            zb = z.ravel()
            ub = u.ravel()
            bad_idx = numpy.flatnonzero(bad)
            for i in bad_idx:
                zi = zb[i]
                u_roots = self._roots_poly_u_scalar(zi)
                ub[i] = _pick_physical_root_scalar(zi, u_roots)
            u = ub.reshape(z.shape)

        m = (u + (1.0 - c) / z) / c

        if scalar:
            return m.reshape(())
        return m

    # =======
    # density
    # =======

    def density(self, x=None, eta=1e-3, ac_only=True, plot=False, latex=False,
                save=False, eig=None):
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

        ac_only : bool, default=True
            If `True`, it returns the absolutely-continuous part of density.

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

        Notes
        -----

        * Do not warm-start across x<0 (MP-type support is >=0).
        * Reset warm-start when previous u is (nearly) real.
        * If Newton lands on a non-Herglotz root, fall back to polynomial roots
          and pick.

        If ac_only is True and c < 1, subtract the smeared atom at zero of mass
        (1-c) for visualization.
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

        # Atoms
        atoms = None
        if self.c > 1.0:
            atom_loc = 0.0
            atom_w = 1.0 - 1.0 / self.c
            atoms = [(atom_loc, atom_w)]

        # Optional: remove the atom at zero (only for visualization of AC part)
        if (atoms is not None) and (ac_only is True):
            atom = atom_w * (float(eta) / numpy.pi) / \
                (x * x + float(eta) * float(eta))
            rho = rho - atom
            rho = numpy.maximum(rho, 0.0)

        if plot:
            if eig is not None:
                label = 'Theoretical'
            else:
                label = ''
            plot_density(x, rho, atoms=atoms, label=label, latex=latex,
                         save=save, eig=eig)

        return rho

    # =====
    # roots
    # =====

    def roots(self, z):
        """
        Roots of polynomial implicitly representing Stieltjes transform

        Parameters
        ----------

        z : complex or numpy.ndarray
            A complex scalar or a 1D or 2D array of query points.

        Returns
        -------

        r : numpy.ndarray
            Roots of polynomial with the following array shape:

            * If ``z`` is scalar, returned array is of the shape ``(r+1,)``.
            * If ``z`` is array-like, returned array if of shape
              ``z.shape + (r+1,)``.

        See Also
        --------

        stieltjes
        poly

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 8

            >>> import numpy
            >>> from freealg.distributions import DeformedMarchenkoPastur

            >>> dmp = DeformedMarchenkoPastur(t=[2.0, 5.5], w=[0.75, 1-0.75],
            ...     c=0.1)

            >>> z = numpy.linspace(0, 2, 10) + 2.0j
            >>> r = dmp.roots(z)
        """

        t = numpy.asarray(self.t, dtype=float)
        w = numpy.asarray(self.w, dtype=float)
        c = float(self.c)
        r = int(t.size)

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        if c < 0.0:
            raise ValueError("c must be >= 0.")

        zf = z.ravel()
        out = numpy.empty((zf.size, r + 1), dtype=numpy.complex128)

        if c == 0.0:
            mr = numpy.zeros_like(zf, dtype=numpy.complex128)
            for ti, wi in zip(t, w):
                mr = mr + (wi / (ti - zf))
            out[:, :] = mr[:, None]
        else:
            prod = numpy.array([1.0], dtype=numpy.complex128)
            for ti in t:
                prod = numpy.convolve(
                    prod, numpy.array([1.0, ti], dtype=numpy.complex128))

            s = numpy.zeros(r, dtype=numpy.complex128)
            for wi, ti in zip(w, t):
                q, _ = numpy.polynomial.polynomial.polydiv(
                    prod, numpy.array([1.0, ti], dtype=numpy.complex128))
                s = s + (wi * ti) * q[:r]

            prod_pad = numpy.pad(prod, (0, 1))
            u_prod = numpy.concatenate(
                [numpy.zeros(1, dtype=numpy.complex128), prod])

            term2 = numpy.pad(c * numpy.concatenate(
                [numpy.zeros(1, dtype=numpy.complex128), s]), (0, 1))

            base = (-1.0) * prod_pad + term2
            zpart = (-1.0) * u_prod

            for i in range(zf.size):
                P = base + zf[i] * zpart
                u_roots = numpy.roots(P[::-1])
                out[i, :] = (u_roots + (1.0 - c) / zf[i]) / c

        out = out.reshape(z.shape + (r + 1,))
        if scalar:
            return out.reshape((r + 1,))
        return out

    # =======
    # support
    # =======

    def support(self, eta=2e-4, n_probe=4000, thr=5e-4, x_max=None, x_pad=0.05,
                method='quartic'):
        """
        Support intervals of distribution

        Parameters
        ----------

        eta : float, default=2e-4
            Imaginary offset used in the Stieltjes inversion for density.

        n_probe : int, default=4000
            Number of grid points used to probe the density (when applicable).

        thr : float, default=5e-4
            Density threshold used to detect nonzero regions (when applicable).

        x_max : float or None, default=None
            Optional right endpoint for probing-based methods.

        x_pad : float, default=0.05
            Optional padding used for search/probing range.

        method : {``'quartic'``, ``'probe'``}, default=``'quartic'``
            Method used to estimate the support.

        Returns
        -------

        intervals : list of tuple(float, float)
            List of (left, right) support intervals.

        See Also
        --------

        density

        Notes
        -----

        Estimate support intervals of
        :math:`\\mu = H \\boxtimes \\mathrm{MP}_c` where
        :math:`H = \\sum_i w_i \\delta_{t_i}`.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 5

            >>> from freealg.distributions import DeformedMarchenkoPastur
            >>> dmp = DeformedMarchenkoPastur(t=[2.0, 5.5], w=[0.75, 1-0.75],
            ...     c=0.1)

            >>> print(dmp.support)
            [(1.271942644768898, 2.796717409578293),
             (4.465954791747979, 6.910028011047688)]
        """

        # Unpack parameters
        t = self.t
        w = self.w
        c = float(self.c)
        r = int(t.size)

        if c < 0.0:
            raise ValueError("c must be >= 0.")

        if method not in ('quartic', 'probe'):
            raise ValueError("method must be 'quartic' or 'probe'.")

        # The quartic shortcut is specific to two-atom H.
        if (method == 'quartic') and (r != 2):
            method = 'probe'

        # --- fast endpoint finder via quartic in u (r=2 only) ---
        if method == 'quartic':
            t1 = float(t[0])
            t2 = float(t[1])
            w1 = float(w[0])
            w2 = float(w[1])

            # Build the quartic polynomial:
            #   A(u)^2 B(u)^2 - c u^2 ( w1 t1^2 B(u)^2 + w2 t2^2 A(u)^2 ) = 0
            # where A(u)=1+t1 u, B(u)=1+t2 u.
            u = numpy.poly1d([1.0, 0.0])          # u
            A = 1.0 + float(t1) * u
            B = 1.0 + float(t2) * u
            A2 = A * A
            B2 = B * B
            P = (A2 * B2) - c * (u * u) * \
                (w1 * (t1 * t1) * B2 + w2 * (t2 * t2) * A2)

            u_roots = numpy.roots(P.c)

            # keep real negative roots away from poles u=-1/t1,-1/t2 and from 0
            poles = []
            if float(t1) != 0.0:
                poles.append(-1.0 / float(t1))
            if float(t2) != 0.0:
                poles.append(-1.0 / float(t2))

            u_crit = []
            for rr in u_roots:
                if not numpy.isfinite(rr):
                    continue
                if abs(rr.imag) > 1e-10 * (1.0 + abs(rr.real)):
                    continue
                ur = float(rr.real)
                if ur >= 0.0:
                    continue
                if abs(ur) < 1e-14:
                    continue
                too_close = False
                for p in poles:
                    if abs(ur - p) < 1e-10 * (1.0 + abs(p)):
                        too_close = True
                        break
                if too_close:
                    continue
                u_crit.append(ur)

            u_crit = sorted(set(u_crit))
            if len(u_crit) < 2:
                # Fallback to probing if quartic degenerates numerically
                method = 'probe'
            else:
                def x_of_u(uu):
                    return (-1.0 / uu) + c * (w1 * t1 / (1.0 + t1 * uu) +
                                              w2 * t2 / (1.0 + t2 * uu))

                x_crit = []
                for uu in u_crit:
                    xv = x_of_u(uu)
                    if numpy.isfinite(xv):
                        x_crit.append(float(xv))

                x_crit = sorted(x_crit)
                # endpoints come in pairs; build candidate intervals
                cand = []
                for k in range(0, len(x_crit) - 1, 2):
                    a = x_crit[k]
                    b = x_crit[k + 1]
                    if b > a:
                        cand.append((a, b))

                # validate each candidate interval by checking rho at midpoints
                cuts = []
                for a, b in cand:
                    mid = 0.5 * (a + b)
                    # very cheap check (one evaluation)
                    rh = float(self.density(numpy.array([mid]),
                                            eta=max(eta, 1e-8))[0])
                    if numpy.isfinite(rh) and (rh > 0.0):
                        aa = max(0.0, a)  # MP-type spectra should be >=0
                        cuts.append((aa, b))

                # If everything validated out (rare), fall back to probe.
                if len(cuts) > 0:
                    return cuts
                method = 'probe'

        # Legacy probing (works for any r). Heuristic x-range
        tmax = float(max(numpy.max(numpy.abs(t)), 1e-12))
        if x_max is None:
            s = (1.0 + numpy.sqrt(max(c, 0.0))) ** 2
            x_max = 3.0 * tmax * s + 1.0
        x_max = float(x_max)

        x_min = -float(x_pad) * x_max

        x = numpy.linspace(x_min, x_max, int(n_probe))
        rho = self.density(x, eta=float(eta))

        good = numpy.isfinite(rho) & (rho > float(thr))
        if not numpy.any(good):
            return []

        idx = numpy.where(good)[0]
        breaks = numpy.where(numpy.diff(idx) > 1)[0]
        segments = []
        start = idx[0]
        for b in breaks:
            end = idx[b]
            segments.append((start, end))
            start = idx[b + 1]
        segments.append((start, idx[-1]))

        def rho_scalar(x0):
            return float(self.density(numpy.array([x0]), eta=float(eta))[0])

        cuts = []
        for i0, i1 in segments:
            a0 = float(x[max(i0 - 1, 0)])
            a1 = float(x[i0])
            b0 = float(x[i1])
            b1 = float(x[min(i1 + 1, x.size - 1)])

            # left edge
            lo, hi = a0, a1
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if rho_scalar(mid) > thr:
                    hi = mid
                else:
                    lo = mid
            a = hi

            # right edge
            lo, hi = b0, b1
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if rho_scalar(mid) > thr:
                    lo = mid
                else:
                    hi = mid
            b = lo

            if numpy.isfinite(a) and numpy.isfinite(b) and (b > a + 1e-10):
                cuts.append((max(0.0, a), b))

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

        Generate an :math:`n x n` sample covariance matrix :math:`\\mathbf{S}`
        whose ESD converges to :math:`H \\boxtimes MP_c`, where
        :math:`H = \\sum_i w_i \\delta_{t_i}`.

        Finite :math:`n` construction:

        * :math:`m` is chosen so that :math:`n/m` approx :math:`c` (when
          :math:`c>0`),
        * :math:`Z` has i.i.d. :math:`N(0,1)`,
        * :math:`\\boldsymbol{\\Sigma}` has eigenvalues :math:`t_i` with
          proportions :math:`w_i`,
        * :math:`\\mathbf{S} = (1/m) \\boldsymbol{\\Sigma}^{1/2} \\mathbf{Z}
          \\mathbf{Z}^T \\boldsymbol{\\Sigma}^{1/2}`.
        """

        n = int(size)
        if n <= 0:
            raise ValueError("size must be a positive integer.")

        # Unpack parameters
        t = self.t
        w = self.w
        c = float(self.c)

        rng = numpy.random.default_rng(seed)

        # Choose m so that n/m approx c (for c>0). For c=0, return population
        # Sigma.
        if c == 0.0:
            # Build diagonal Sigma with r atoms
            counts = numpy.floor(w * n).astype(int)
            remainder = n - int(counts.sum())
            if remainder > 0:
                frac = (w * n) - counts
                idx = numpy.argsort(frac)[::-1]
                counts[idx[:remainder]] += 1

            d = numpy.empty(n, dtype=numpy.float64)
            pos = 0
            for ti, ni in zip(t, counts):
                d[pos:pos + ni] = float(ti)
                pos += ni
            rng.shuffle(d)
            return numpy.diag(d)

        # m must be positive integer
        m = int(round(n / c)) if c > 0.0 else n
        m = max(1, m)

        # Build diagonal Sigma^{1/2} with r atoms
        counts = numpy.floor(w * n).astype(int)
        remainder = n - int(counts.sum())
        if remainder > 0:
            frac = (w * n) - counts
            idx = numpy.argsort(frac)[::-1]
            counts[idx[:remainder]] += 1

        s = numpy.empty(n, dtype=numpy.float64)
        pos = 0
        for ti, ni in zip(t, counts):
            s[pos:pos + ni] = numpy.sqrt(float(ti))
            pos += ni
        rng.shuffle(s)

        # Draw Z and form X = Sigma^{1/2} Z / sqrt(m)
        Z = rng.standard_normal((n, m))
        X = (s[:, None] * Z) / numpy.sqrt(m)

        # Sample covariance
        S = X @ X.T

        return S

    # ====
    # poly
    # ====

    def poly(self):
        """
        Polynomial coefficients implicitly representing the Stieltjes

        This is the eliminated polynomial in m (not underline{m}).
        coeffs[i, j] is the coefficient of z^i m^j.
        """

        c = float(self.c)
        t = self.t
        w = self.w
        r = int(t.size)

        # Multivariate Laurent polynomial dict: (kz, km) -> coef
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
        z_poly = {(1, 0): 1.0}
        zinv_poly = {(-1, 0): 1.0}
        m_poly = {(0, 1): 1.0}

        # u = c m + (c-1)/z  (equivalently u = c m - (1-c)/z)
        u_poly = add_poly(scale_poly(m_poly, c),
                          scale_poly(zinv_poly, (c - 1.0)))

        # Build prod_i (1 + t_i u)
        prod = one
        factors = []
        for i in range(r):
            fac = add_poly(one, scale_poly(u_poly, float(t[i])))
            factors.append(fac)
            prod = mul_poly(prod, fac)

        # Term1: (-1 - z u) * prod
        zu = mul_poly(z_poly, u_poly)
        term_factor = add_poly(scale_poly(one, -1.0), scale_poly(zu, -1.0))
        term1 = mul_poly(term_factor, prod)

        # Term2: c u sum_i w_i t_i prod_{j!=i} (1 + t_j u)
        sum_poly = {}
        for i in range(r):
            p_ex = one
            for j in range(r):
                if j == i:
                    continue
                p_ex = mul_poly(p_ex, factors[j])
            sum_poly = add_poly(sum_poly, scale_poly(p_ex, float(w[i] * t[i])))
        term2 = scale_poly(mul_poly(u_poly, sum_poly), c)

        P = add_poly(term1, term2)

        # Clear negative z powers
        min_kz = min(kz for (kz, km) in P.keys())
        if min_kz < 0:
            shift = -min_kz
            P = {(kz + shift, km): v for (kz, km), v in P.items()}

        # Remove any common factor z^k (keep smallest z-power at 0).
        # Note: treat tiny coefficients as zero to avoid spurious kz=0 terms.
        if P:
            max_abs = max(abs(v) for v in P.values())
            tol_abs = 1.0e-12 * max_abs
            keys_nz = [(kz, km) for (kz, km), v in P.items()
                       if abs(v) > tol_abs]
            min_kz2 = min(kz for (kz, km) in keys_nz) if keys_nz else 0
            if min_kz2 > 0:
                P = {(kz - min_kz2, km): v for (kz, km), v in P.items()}

        max_kz = max(kz for (kz, km) in P.keys()) if P else 0
        max_km = max(km for (kz, km) in P.keys()) if P else 0

        coeffs = numpy.zeros((max_kz + 1, max_km + 1), dtype=numpy.complex128)
        for (kz, km), v in P.items():
            coeffs[int(kz), int(km)] = v

        # Clean tiny numerical noise (keeps poly stable and comparable)
        if coeffs.size > 0:
            max_abs = float(numpy.max(numpy.abs(coeffs)))
            if max_abs > 0.0:
                coeffs[numpy.abs(coeffs) < 1.0e-12 * max_abs] = 0.0

        return coeffs
