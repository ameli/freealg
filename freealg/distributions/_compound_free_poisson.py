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

__all__ = ['CompoundFreePoisson']


# =====================
# Compound Free Poisson
# =====================

class CompoundFreePoisson(BaseDistribution):
    """
    Compound free Poisson distribution

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

    lam : float
        Total rate (intensity) :math:`\\lambda > 0`.

    Methods
    -------

    density
        Spectral density of distribution.

    roots
        Roots of polynomial implicitly representing Stieltjes transform

    stieltjes
        Stieltjes transform of distribution

    support
        Support intervals of distribution

    sample
        Sample from distribution

    matrix
        Generate matrix with its empirical spectral density of distribution

    poly
        Polynomial coefficients of the spectral curve of Stieltjes transform

    plot_branches
        Plot branches of the spectral curve of Stieltjes transform.

    See Also
    --------

    freealg.distributions.FreeLevy

    Notes
    -----

    This model has atom at zero with mass :math:`\\max(1-\\lambda, 0)`.

    This model is the additive free compound Poisson law whose R-transform is

    .. math::

        R(w) = \\lambda \\sum_{i=1}^r w_i \\frac{t_i}{1-t_i w},

    where :math:`\\lambda>0` is the total rate (intensity),
    :math:`t_i>0` are jump sizes, and :math:`w_i>0` are weights with
    :math:`\\sum_i w_i = 1`.

    The Stieltjes transform :math:`m(z)` satisfies

    .. math::

        R(-m(z)) = z + 1/m(z).

    For two atoms, clearing denominators yields a cubic polynomial in
    :math:`m`:

    .. math::

        a_3(z)m^3 + t_2(z)m^2 + t_1(z)m + a_0(z) = 0.
    """

    # ====
    # init
    # ====

    def __init__(self, t, w, lam):
        """
        Initialization.
        """

        t = numpy.asarray(t, dtype=numpy.float64)
        w = numpy.asarray(w, dtype=numpy.float64)
        lam = float(lam)

        if t.ndim != 1 or t.size == 0:
            raise ValueError("t must be a one-dimensional non-empty array.")

        if w.shape != t.shape:
            raise ValueError("w and t must have the same shape.")

        if numpy.any(t <= 0.0):
            raise ValueError("t must be > 0.")

        if numpy.any(w <= 0.0):
            raise ValueError("w must be > 0.")

        w_sum = float(numpy.sum(w))
        if abs(w_sum - 1.0) > 1.0e-12:
            raise ValueError("w must sum to 1.")

        if lam <= 0.0:
            raise ValueError("lam must be > 0.")

        self.t = t
        self.w = w
        self.lam = lam

        # Bounds for smallest and largest eigenvalues
        if lam < 1.0:
            # In this case, there is an atom at the origin
            self.lam_lb = 0.0
        else:
            self.lam_lb = numpy.min(t) * (1 - numpy.sqrt(lam))**2
        self.lam_ub = numpy.max(t) * (1 + numpy.sqrt(lam))**2

        # Number of roots (branches)
        self.num_roots = 3

    # ===================
    # roots poly m scalar
    # ===================

    def _roots_poly_m_scalar(self, z):
        """
        Return the roots of the polynomial equation in m at scalar z.
        """

        if int(self.t.size) == 2:
            # When r is two
            t1 = float(self.t[0])
            t2 = float(self.t[1])

            w1 = float(self.w[0])
            w2 = float(self.w[1])

            lam = float(self.lam)
            z = complex(z)

            # cubic coefficients (same as old)
            c3 = z * t1 * t2
            c2 = z * (t1 + t2) + t1 * t2 * (1.0 - lam)
            c1 = z + (t1 + t2) - lam * (w1 * t1 + w2 * t2)
            c0 = 1.0

            roots = numpy.roots(numpy.array([c3, c2, c1, c0],
                                            dtype=numpy.complex128))
            return roots

        else:
            # Any r other than 2
            t = self.t
            w = self.w
            lam = float(self.lam)
            r = int(t.size)

            z = complex(z)

            # Build prod_i (1 + t_i m)
            prefix = [numpy.poly1d([1.0])]
            for i in range(r):
                prefix.append(prefix[-1] * numpy.poly1d([float(t[i]), 1.0]))

            suffix = [None] * (r + 1)
            suffix[r] = numpy.poly1d([1.0])
            for i in range(r - 1, -1, -1):
                suffix[i] = suffix[i + 1] * numpy.poly1d([float(t[i]), 1.0])

            prod = prefix[r]

            # Term1: (-1 - z m) * prod
            term1 = numpy.poly1d([-z, -1.0]) * prod

            # Term2: lam m sum_i w_i t_i prod_{j != i} (1 + t_j m)
            s = numpy.poly1d([0.0])
            for i in range(r):
                prod_ex = prefix[i] * suffix[i + 1]
                s = s + float(w[i] * t[i]) * prod_ex

            term2 = numpy.poly1d([lam, 0.0]) * s
            P = term1 + term2

            roots = numpy.roots(P.c)
            return roots

    # =====================
    # solve m newton scalar
    # =====================

    def _solve_m_newton(self, z, m0=None, max_iter=100, tol=1e-12):
        """
        Solve R(-m) = z + 1/m for scalar z using Newton iterations.
        """

        t = self.t
        w = self.w
        lam = float(self.lam)

        z = complex(z)
        if m0 is None:
            m = -1.0 / z
        else:
            m = complex(m0)

        for _ in range(int(max_iter)):
            d = 1.0 + t * m

            # f(m) = -1/m + R(-m) - z
            f = (-1.0 / m) + lam * numpy.sum(w * t / d) - z

            # f'(m) = 1/m^2 - lam * sum w_i t_i^2/(1+t_i m)^2
            fp = (1.0 / (m * m)) - lam * numpy.sum(w * (t * t) / (d * d))

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
        Stieltjes transform of distribution.

        Parameters
        ----------

        z : complex or numpy.ndarray
            A complex scalar or a 1D or 2D array of query points.

        max_iter : int, default=100
            Maximum number of Newton iterations to solve for Stieltjes
            transform.

        tol : float, default=1e-12
            The tolerance of Newton iterations.

        Returns
        -------

        m : complex or numpy.ndarray
            A complex scalar or array of the same size as the input ``z``, as
            the Stieltjes transform :math:`m = m(z)`.

        See Also
        --------

        density
        roots

        Notes
        -----

        Stieltjes transform is the physical (Herglotz) branch of of the
        polynomial :math:`P(z, m)`.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 10

            >>> import numpy
            >>> from freealg.distributions import CompoundFreePoisson

            >>> # Create an object of the class
            >>> cfp = CompoundFreePoisson(t=[2.0, 5.5], w=[0.75, 1-0.75],
            ...    lam=0.1)

            >>> # Query at given locations
            >>> z = numpy.linspace(-1, 1) + 1j
            >>> m = cfp.stieltjes(z)
        """

        # Unpack parameters
        t = self.t
        w = self.w
        lam = float(self.lam)

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        # Warm-start along 1D horizontal lines z = x + i*eta (prevents isolated
        # branch flips)
        if (not scalar) and (z.ndim == 1) and (z.size >= 3):
            zi = z
            imz = numpy.imag(zi)
            if float(numpy.max(imz) - numpy.min(imz)) < 1.0e-14:
                # process in increasing Re(z) order but return original order
                order = numpy.argsort(numpy.real(zi))
                inv = numpy.empty_like(order)
                inv[order] = numpy.arange(order.size)

                ms = numpy.empty_like(zi)
                m_prev = -1.0 / zi[order[0]]

                for k in order:
                    mk, ok = self._solve_m_newton(zi[k], m0=m_prev,
                                                  max_iter=max_iter, tol=tol)
                    if (not ok) or (not numpy.isfinite(mk)):
                        # fallback to algebraic roots (physical sheet)
                        rts = self._roots_poly_m_scalar(zi[k])
                        mk = _pick_physical_root_scalar(zi[k], rts)

                    ms[k] = mk
                    # keep warm-start only if still safely Herglotz-ish
                    if numpy.imag(mk) > 1.0e-14:
                        m_prev = mk
                    else:
                        m_prev = -1.0 / zi[k]

                return ms

        # m initial guess
        m = -1.0 / z
        active = numpy.isfinite(m)

        for _ in range(int(max_iter)):
            if not numpy.any(active):
                break

            idx = numpy.flatnonzero(active)
            ma = m.ravel()[idx]
            za = z.ravel()[idx]

            # Fast 1D accumulation (avoids (r x n) temporaries)
            f = (-1.0 / ma) - za
            fp = (1.0 / (ma * ma))

            for k in range(int(t.size)):
                tk = float(t[k])
                wk = float(w[k])
                dk = 1.0 + tk * ma
                f += lam * (wk * tk) / dk
                fp -= lam * (wk * tk * tk) / (dk * dk)

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
                m_roots = self._roots_poly_m_scalar(zi)
                mb[i] = _pick_physical_root_scalar(zi, m_roots)
            m = mb.reshape(z.shape)

        # Repair isolated Newton "sheet glitches" where Im(m) nearly vanishes
        # at a single interior grid point (common in density plots). Only for
        # 1D grids at (approximately) constant imaginary offset.
        if (not scalar) and (m.ndim == 1) and (m.size >= 3):
            zi = numpy.imag(z.ravel())
            if numpy.max(numpy.abs(zi - zi[0])) < 1.0e-14:
                im = numpy.imag(m)
                im_max = float(numpy.max(im)) if im.size else 0.0
                if im_max > 0.0:
                    eps = 1.0e-10 * im_max
                    mid = (im[1:-1] < eps) & (im[:-2] > 10.0 * eps) & \
                        (im[2:] > 10.0 * eps)
                    if numpy.any(mid):
                        zb = z.ravel()
                        mb = m.ravel()
                        fix_idx = numpy.flatnonzero(mid) + 1
                        for ii in fix_idx:
                            m_roots = self._roots_poly_m_scalar(zb[int(ii)])
                            mb[int(ii)] = _pick_physical_root_scalar(
                                zb[int(ii)], m_roots)
                        m = mb.reshape(z.shape)

        if scalar:
            return m.reshape(())
        return m

    # =======
    # density
    # =======

    def density(self, x=None, eta=2e-4, max_iter=100, tol=1e-12,
                return_atoms=False, plot=False, latex=False, save=False,
                eig=None):
        """
        Density of distribution.

        Parameters
        ----------

        x : numpy.array, default=None
            The locations where density is evaluated at. If `None`, an interval
            slightly larger than the support interval of the spectral density
            is used.

        eta : float, default=2e-4
            The offset :math:`\\eta` from the real axis where the density
            is evaluated using Plemelj formula at :math:`z = x + i \\eta`.

        max_iter : int, default=100
            Maximum number of Newton iterations to solve for the Stieltjes
            root.

        tol : float, default=1e-12
            Tolerance for Newton iterations to solve for the Stieltjes root.

        return_atoms : bool, default=False
            If `True`,  the atoms (if any) of the distribution will also be
            returned.

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
            Absolutely-continuous part of the spectral density.

        if return_atoms is True:
            atoms : list
                A list of tuples ``(loc, wight)`` containing the location and
                the weight of the atom.

        See Also
        --------

        stieltjes
        sample

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 10

            >>> import numpy
            >>> from freealg.distributions import CompoundFreePoisson

            >>> # Create an object of the class
            >>> cfp = CompoubdFreePoisson(t=[2.0, 5.5], w=[0.75, 1-0.75],
            ...    lam=0.1)

            >>> # Plot density
            >>> x = numpy.linspace(0, 2, 100)
            >>> rho, atoms = cfp.density(x, return_atoms=True, plot=True)
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
        m = self.stieltjes(z, max_iter=max_iter, tol=tol)
        rho = numpy.imag(m) / numpy.pi

        # Atoms
        atoms = []
        if self.lam < 1.0:
            atom_loc = 0.0
            atom_w = 1.0 - self.lam
            atoms = [(atom_loc, atom_w)]

        # Optional: remove the atom at zero (only for visualization of AC part)
        if len(atoms) > 0:
            zr = z.real
            atom = atom_w * (float(eta) / (numpy.pi * (zr*zr + float(eta)**2)))
            rho = rho - atom
            rho = numpy.maximum(rho, 0.0)

        if plot:
            label = 'Absolutely-Continuous'
            plot_density(x, rho, atoms=atoms, label=label, latex=latex,
                         save=save, eig=eig)

        if return_atoms:
            return rho, atoms
        else:
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
            :emphasize-lines: 9

            >>> import numpy
            >>> from freealg.distributions import CompoundFreePoisson

            >>> # Create an object of the class
            >>> cfp = CompoundFreePoisson(t=[2.0, 5.5], w=[0.75, 1-0.75],
            ...    lam=0.1)

            >>> z = numpy.linspace(0, 2, 10) + 2.0j
            >>> r = cfp.roots(z)
        """

        z = numpy.asarray(z, dtype=numpy.complex128)

        if z.ndim == 0:
            return self._roots_poly_m_scalar(z.reshape(()))

        z_flat = z.ravel()
        n = z_flat.size

        t = numpy.asarray(self.t, dtype=float)
        w = numpy.asarray(self.w, dtype=float)
        lam = float(self.lam)
        r = int(t.size)

        prod = numpy.array([1.0], dtype=numpy.complex128)
        for ti in t:
            prod = numpy.convolve(prod, numpy.array([1.0, ti],
                                                    dtype=numpy.complex128))

        s = numpy.zeros(r, dtype=numpy.complex128)
        for wi, ti in zip(w, t):
            q, rem = numpy.polynomial.polynomial.polydiv(
                prod, numpy.array([1.0, ti], dtype=numpy.complex128))
            s += (wi * ti) * q[:r]

        term0 = (-1.0) * prod + lam * numpy.concatenate(
            [numpy.zeros(1, dtype=numpy.complex128), s])

        base = numpy.pad(term0, (0, 1))
        zpart = (-1.0) * numpy.concatenate(
            [numpy.zeros(1, dtype=numpy.complex128), prod])

        deg = int(base.size - 1)
        out = numpy.empty((n, deg), dtype=numpy.complex128)

        for i in range(n):
            c = base + z_flat[i] * zpart
            out[i, :] = numpy.roots(c[::-1])

        return out.reshape(z.shape + (deg,))

    # =======
    # support
    # =======

    def support(self, eta=2e-4, n_probe=4000, thr=5e-4, x_max=None):
        """
        Support intervals of distribution.

        Parameters
        ----------

        eta : float, default=2e-4
            Imaginary offset used in the Stieltjes inversion for density.

        n_probe : int, default=4000
            Number of grid points used to probe the density.

        thr : float, default=5e-4
            Density threshold used to detect nonzero regions.

        x_max : float or None, default=None
            Right endpoint of the probing grid. If None, a heuristic is used.

        Returns
        -------

        intervals : list of tuple(float, float)
            List of (left, right) support intervals estimated from the grid.

        Notes
        -----

        The support is estimated on a real grid by thresholding the density
        :math:`\\rho(x; ``eta)`.

        See Also
        --------

        density

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 8

            >>> import numpy
            >>> from freealg.distributions import CompoundFreePoisson

            >>> # Create an object of the class
            >>> cfp = CompoundFreePoisson(t=[2.0, 5.5], w=[0.75, 1-0.75],
            ...    lam=0.1)

            >>> print(cfp.support)
            [(0.9984996249062267, 3.13165791447862),
             (4.157389347336835, 7.597674418604652)]
        """

        t = self.t
        lam = float(self.lam)

        x_min = 0.0
        if x_max is None:
            # Heuristic: scale grows ~ O(lam * max(t))
            x_max = (1.0 + lam) * float(numpy.max(t)) * 6.0
        x_max = float(x_max)
        if x_max <= 0.0:
            raise ValueError("x_max must be > 0.")

        x = numpy.linspace(x_min, float(x_max), int(n_probe))
        rho = self.density(x, eta=eta)

        mask = rho > float(thr)

        # Prevent one-point gap (due to numerical errors) to split the support
        mask = mask | numpy.r_[False, mask[:-1]] | numpy.r_[mask[1:], False]
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
            intervals.append((float(max(0.0, xa)), float(xb)))

        return intervals

    # ======
    # matrix
    # ======

    def matrix(self, size, seed=None):
        """
        Generate matrix with the spectral density of the distribution.

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

        Use a sum of independent (rotationally invariant) Wishart terms,
        which are asymptotically free:

        .. math::

            A = \\sum_{i=1}^r s_i \\frac{1}{m_i} \\mathbf{Z}_i
            \\mathbf{Z}_i^{\\intercal},

        where :math:`\\mathbf{Z}_i` are :math:`n \\times m_i` i.i.d.
        :math:`N(0,1)`. Choose aspect ratios :math:`c_i = n/m_i` and
        scales :math:`s_i` so each term has R-transform

        .. math::

            R_i(w) = \\lambda_i \\frac{t_i}{(1 - t_i w},

        with :math:`\\lambda_i = \\lambda w_i`. This is achieved by setting

        .. math::

            c_i = 1 / \\lambda_i,
            s_i = t_i \\lambda_i.
        """

        n = int(size)
        if n <= 0:
            raise ValueError("size must be a positive integer.")

        # Unpack parameters
        t = self.t
        w = self.w
        lam = float(self.lam)
        r = int(t.size)

        rng = numpy.random.default_rng(seed)

        A = numpy.zeros((n, n), dtype=numpy.float64)

        for i in range(r):
            lami = lam * float(w[i])
            ci = 1.0 / lami
            mi = max(1, int(round(n / ci)))
            si = float(t[i]) * lami
            Zi = rng.standard_normal((n, mi))
            A += si * (Zi @ Zi.T) / float(mi)

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
        lam = float(self.lam)
        r = int(t.size)

        # Build prod_i (1 + t_i m)
        prod = numpy.poly1d([1.0])
        for i in range(r):
            prod = prod * numpy.poly1d([float(t[i]), 1.0])

        # s(m) = sum_i w_i t_i prod_{j!=i} (1 + t_j m)
        prefix = [numpy.poly1d([1.0])]
        for i in range(r):
            prefix.append(prefix[-1] * numpy.poly1d([float(t[i]), 1.0]))
        suffix = [None] * (r + 1)
        suffix[r] = numpy.poly1d([1.0])
        for i in range(r - 1, -1, -1):
            suffix[i] = suffix[i + 1] * numpy.poly1d([float(t[i]), 1.0])

        s = numpy.poly1d([0.0])
        for i in range(r):
            prod_ex = prefix[i] * suffix[i + 1]
            s = s + float(w[i] * t[i]) * prod_ex

        # P(z,m) = (-1 - z m) prod + lam m s
        term0 = (-1.0) * prod + numpy.poly1d([lam, 0.0]) * s   # z^0 row
        termz = (-1.0) * (numpy.poly1d([1.0, 0.0]) * prod)     # z^1 row

        deg_m = max(int(term0.order), int(termz.order))
        coeffs = numpy.zeros((2, deg_m + 1), dtype=numpy.complex128)

        def _fill_row(row, poly_m):
            cc = numpy.asarray(poly_m.c, dtype=numpy.complex128)
            # poly1d stores descending powers; convert to ascending
            asc = cc[::-1]
            for j in range(asc.size):
                coeffs[row, j] = asc[j]

        _fill_row(0, term0)
        _fill_row(1, termz)

        # Make sure coeff[0, 0] is positive (just a sign convention)
        # Normalize so coeffs[0, 0] is +1 (pure convention)
        if coeffs.size > 0 and coeffs[0, 0] != 0:
            if numpy.real(coeffs[0, 0]) < 0:
                coeffs = -coeffs

        return coeffs
