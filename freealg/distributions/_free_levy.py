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

__all__ = ['FreeLevy']


# =========
# Free Levy
# =========

class FreeLevy(BaseDistribution):
    """
    Free Levy distribution.

    This class extends the compound free Poisson law (
    :class:`freealg.distributions.CompoundFreePoisson`) by adding a drift and
    semicircle parts.

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

    a : float, default=0.0
        Shift (drift) parameter :math:`a` in :math:`\\delta_a` (see notes
        below.)

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

    freealg.distributions.CompoundFreePoisson

    Notes
    -----

    **General Free Levy Law:**

    This model is constructed by the free additive convolution of the following
    three laws:

    .. math::

        \\mu = \\mu_{\\delta_a} \\boxplus \\mu_{\\mathrm{SC}_{\\sigma^2}}
        \\boxplus \\mu_{\\mathrm{CFP}(\\lambda, H)},

    where

    * :math:`\\delta_a` is the shift :math:`X \\mapsto X + a` (given by the
      parameter ``a``);
    * :math:`\\mathrm{SC}_{\\sigma^2}` is the semicircle (free Gaussian) law
      with variance :math:`\\sigma^2` (give by the parameter ``sigma``);
    * :math:`\\mathrm{CFP}(\\lambda, H)` is the compound free Poisson law with
      the rate :math:`\\lambda` and jump :math:`H`, given by the R transform

      .. math::

          R_{\\mathrm{CFP}(\\lambda, H)} =
          \\lambda \\int_{\\mathbb{R}} \\frac{x}{1 - wx} H(\\mathrm{d} x).

      Here, :math:`H > 0` is a positive measure. The free Levy distribution
      represents all free infinitely divisible (FID) distributions (see
      Theorem 13.8 in [1]_).

    **Algebraic Free Levy Law:**

    In this class, we assume :math:`H` is discrete atomic distribution given
    by

    .. math::

        H = \\sum_{i=1}^r w_i \\delta_{t_i},

    where :math:`t_i>0` are jump sizes (given by the parameter ``t`` as a
    list), and :math:`w_i>0` are weights with
    :math:`\\sum_i w_i = 1` (given by the parameter ``w`` as a list). This
    assumption on :math:`H` restricts the free Levy class to *algebraic* FID
    distributions where its Stieltjes transform satisfies a polynomial
    constraint (see below).

    **R transform:**

    The R transform the atomic free Levy law is the sum of all constituent laws

    .. math::

        R = R_{\\delta_a} + R_{\\mathrm{SC}_{\\sigma^2}}
        + R_{\\mathrm{CFP}(\\lambda, H)},

    which for atomic :math:`H` becomes

    .. math::

        R(w) = a + \\sigma^2 w + \\lambda \\sum_{i=1}^r w_i
        \\frac{t_i}{1-t_i w}.

    **Stieltjes Transform:**

    The Stieltjes transform :math:`m(z)` satisfies

    .. math::

        R(-m(z)) = z + \\frac{1}{m(z)}.

    Solving for :math:`m` and clearing the common denominators for the rational
    function representation, we get a polynomial :math:`P(z, m) = 0` with

    .. math::

        P(z, m) = \\sum_{i=1}^{d_z} \\sum_{j=1}^{d_m} c_{ij} z^i m^j = 0,

    where :math:`d_z` and :math:`d_m` are the degrees of :math:`P` in :math:`z`
    and :math:`m`, respectively. The degree :math:`d_z` is always 1.

    For :math:`r` atoms in :math:`H` and :math:`\\sigma = 0`, this yields a
    polynomial with degree :math:`d_m = r+1` in :math:`m`. When
    :math:`\\sigma > 0`, the polynomial is of degree :math:`d_m = r+2`.

    When :math:`\\sigma=0`, this model has an atom at :math:`x = a` with mass
    :math:`\\max(1-\\lambda, 0)`.

    **Functions:**

    * The coefficients :math:`c_{ij}` can be obtained from :func:`poly`
      function.
    * For a given :math:`z`, all :math:`d_m` roots (in :math:`m`) of the
      polynomial :math:`P(z, m) = 0` can be computed by :func:`roots`.
    * Among all roots, only one root corresponds to the *physical* branch,
      known as the Stieltjes transform. This physical root can be computed by
      :func:`stieltjes` function.

    References
    ----------

    .. [1] Alexandru Nica and Roland Speicher (2006).
           `Lectures on the Combinatorics of Free Probability
           <https://doi.org/10.1017/CBO9780511735127>`__.  Cambridge University
           Press, LMS.

    Examples
    --------

    We create a free Levy distribution, plot is density and compute its
    support.

    .. code-block:: python
        :emphasize-lines: 4,5

        >>> from freealg.distributions import FreeLevy

        >>> # Create an object of the class
        >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
        ...               sigma=0.9)

        >>> # Plot density
        >>> rho = fl.density(plot=True)

        >>> # Get the support intervals
        >>> supp = fl.support()
        [(-1.9990360090022503, 3.7017366841710433),
         (4.170457614403601, 7.886856714178547)]

    We can also sample from this distribution: either as a matrix realization,
    or as an array of eigenvalues:

    .. code-block:: python

        >>> # Generate a random matrix realization of this law
        >>> A = fl.matrix(size=2000, seed=0)

        >>> # Sample from eigenvalues of this law
        >>> eig = fl.sample(size=2000)

    Here, we compute the coefficients of the polynomial :math:`P(z, m) = 0`,
    all its roots at a given point :math:`z`, and its physical root (Stieltjes
    transform):

    .. code-block:: python

        >>> # Get the coefficients of the polynomial P(z, m) = 0
        >>> coeffs = fl.poly().real
        array([[ 1.    ,  7.2125, 10.71  ,  6.075 ,  8.91  ],
               [ 0.    ,  1.    ,  7.5   , 11.    , -0.    ]])

        >>> # Compute all roots of the polynomial at a given z
        >>> z = 2.0 + 3.0j
        >>> roots = fl.roots(z)
        array([-2.34243764-3.91061653e+00j, -0.50248052-2.41746562e-02j,
               -0.12025009+2.34338362e-01j, -0.18578573-3.25087974e-03j])

        >>> # Compute the Stieltjes transform at z (the physical root)
        >>> m = fl.stieltjes(z)
        array(-0.12025009+0.23433836j)
    """

    # ====
    # init
    # ====

    def __init__(self, t, w, lam, a=0.0, sigma=0.0):
        """
        Initialization.
        """

        t = numpy.asarray(t, dtype=numpy.float64)
        w = numpy.asarray(w, dtype=numpy.float64)
        lam = float(lam)
        a = float(a)
        sigma = float(sigma)

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

        if sigma < 0.0:
            raise ValueError("sigma must be >= 0.")

        self.t = t
        self.w = w
        self.lam = lam
        self.a = a
        self.sigma = sigma
        self.sc_var = (sigma)**2

        # Bounds for smallest and largest eigenvalues
        if lam < 1.0:
            # In this case, there is an atom at the origin
            self.lam_lb = 0.0
        else:
            self.lam_lb = numpy.min(t) * (1 - numpy.sqrt(lam))**2
        self.lam_ub = numpy.max(t) * (1 + numpy.sqrt(lam))**2

        # Add drift and semicircle (free Gaussian) support padding
        sc_rad = 2.0 * numpy.sqrt(float(self.sc_var))
        self.lam_lb = float(self.lam_lb) + float(a) - float(sc_rad)
        self.lam_ub = float(self.lam_ub) + float(a) + float(sc_rad)

        # Number of roots (branches)
        if self.sigma > 0.0:
            self.num_roots = 4
        else:
            self.num_roots = 3

    # ===================
    # roots poly m scalar
    # ===================

    def _roots_poly_m_scalar(self, z):
        """
        Return the roots of the polynomial equation in m at scalar z.
        """

        if (int(self.t.size) == 2) and (float(self.a) == 0.0) and \
                (float(self.sc_var) == 0.0):
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
            a = float(self.a)
            sc_var = float(self.sc_var)

            # Term1: (-1 + (a - z) m - sc_var m^2) * prod
            term1 = numpy.poly1d([-sc_var, (a - z), -1.0]) * prod

            # Term2: lam m sum_i w_i t_i prod_{j != i} (1 + t_j m)
            s = numpy.poly1d([0.0])
            for i in range(r):
                prod_ex = prefix[i] * suffix[i + 1]
                s = s + float(w[i] * t[i]) * prod_ex

            # Term2: lam m^2 * s
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
            sc_var = float(self.sc_var)

            # f(m) = -1/m + a - sc_var*m + R_jump(-m) - z
            f = (-1.0 / m) + float(self.a) - sc_var * m + \
                lam * numpy.sum(w * t / d) - z

            # f'(m) = 1/m^2 - lam * sum w_i t_i^2/(1+t_i m)^2
            # f'(m) = 1/m^2 - sc_var - lam * sum w_i t_i^2/(1+t_i m)^2
            fp = (1.0 / (m * m)) - sc_var - \
                lam * numpy.sum(w * (t * t) / (d * d))

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
            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=0.9)

            >>> # Query at given locations
            >>> z = numpy.linspace(-1, 1) + 1j
            >>> m = fl.stieltjes(z)
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
            sc_var = float(self.sc_var)

            f = (-1.0 / ma) + float(self.a) - sc_var * ma - za
            fp = (1.0 / (ma * ma)) - sc_var

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
            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=0.9)

            >>> # Plot density
            >>> x = numpy.linspace(0, 2, 100)
            >>> rho, atoms = fl.density(x, return_atoms=True, plot=True)
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
        if (float(self.sigma) == 0.0) and (self.lam < 1.0):
            atom_loc = float(self.a)
            atom_w = 1.0 - self.lam
            atoms = [(atom_loc, atom_w)]

        # Optional: remove the atom at zero (only for visualization of AC part)
        if len(atoms) > 0:
            zr = z.real
            atom = atom_w * (float(eta) /
                             (numpy.pi * ((zr - atom_loc)**2 + float(eta)**2)))

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

    def roots(self, z, chunk_size=20000, eps=1e-14):
        """
        Roots of polynomial implicitly representing Stieltjes transform

        Parameters
        ----------
        z : array_like of complex
            A complex scalar or a 1D or 2D array of query points.

        chunk_size : int, default=20000
            Solve in chunks to limit memory. For 200x200 grids, default is
            fine.

        eps : float, default=1e-14
            Tolerance for trimming leading zeros.

        Returns
        -------

        roots : ndarray
            Shape z.shape + (deg,), where deg is the polynomial degree in m.

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
            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=0.9)

            >>> z = numpy.linspace(0, 2, 10) + 2.0j
            >>> r = fl.roots(z)
        """

        z = numpy.asarray(z, dtype=numpy.complex128)
        if z.ndim == 0:
            # Keep your existing scalar behavior (exactly what you had)
            return self._roots_poly_m_scalar(z.reshape(()))

        z_flat = z.ravel()
        n = z_flat.size

        t = numpy.asarray(self.t, dtype=float)
        w = numpy.asarray(self.w, dtype=float)
        lam = float(self.lam)
        a = float(self.a)
        sc = float(self.sc_var)
        r = int(t.size)

        # Build coefficient templates:
        # P(m; z) = base(m) + z * zpart(m)
        # Coeffs are in ASCENDING powers of m: c[k] is coeff of m^k
        # prod(m) = \prod (1 + t_i m)
        prod = numpy.array([1.0], dtype=numpy.complex128)
        for ti in t:
            # length r+1
            prod = numpy.convolve(prod, numpy.array(
                [1.0, ti], dtype=numpy.complex128))

        # s(m) = \sum w_i t_i \prod_{j \neq i}(1 + t_j m)  (degree r-1)
        # We compute \prod_{j \neq i} by dividing prod by (1 + t_i m)
        s = numpy.zeros(r, dtype=numpy.complex128)  # length r
        for wi, ti in zip(w, t):
            q, rem = numpy.polynomial.polynomial.polydiv(
                prod, numpy.array([1.0, ti], dtype=numpy.complex128))
            s += (wi * ti) * q[:r]

        # term2(m) = lam * m * s(m), length r+1
        term2 = numpy.concatenate(
            [numpy.zeros(1, dtype=numpy.complex128), lam * s])

        # term1(m; z) = (-1 + (a - z)m - sc m^2) * prod(m)
        # m*prod, length r+2
        m_prod = numpy.concatenate(
            [numpy.zeros(1, dtype=numpy.complex128), prod])

        # m^2*prod, length r+3
        m2_prod = numpy.concatenate(
            [numpy.zeros(2, dtype=numpy.complex128), prod])

        # Pad prod to length r+3 to align
        prod_pad = numpy.pad(prod, (0, 2))      # length r+3
        m_prod_pad = numpy.pad(m_prod, (0, 1))  # length r+3

        base = (-1.0) * prod_pad + (a) * m_prod_pad + (-sc) * m2_prod
        zpart = (-1.0) * m_prod_pad

        # add term2, aligned
        base = base + numpy.pad(term2, (0, (base.size - term2.size)))

        # Determine degree of P(m;z)=base(m)+z*zpart(m)
        deg_base = base.size - 1
        while deg_base > 0 and abs(base[deg_base]) < eps:
            deg_base -= 1

        deg_z = zpart.size - 1
        while deg_z > 0 and abs(zpart[deg_z]) < eps:
            deg_z -= 1

        deg = max(deg_base, deg_z)

        base = base[:deg + 1]
        zpart = zpart[:deg + 1]

        if deg < 1:
            raise RuntimeError("Polynomial degree < 1 (unexpected parameter "
                               "regime).")

        # Solve roots in batch via companion matrices
        out = numpy.empty((n, deg), dtype=numpy.complex128)

        def solve_chunk(z_chunk, out_view):
            # A has ascending monic coeffs:
            # like A[:,0] + A[:,1] m + ... + A[:,-2] m^{deg-1} + 1*m^{deg}
            A = base[None, :] + z_chunk[:, None] * zpart[None, :]

            # Normalize per row so leading coefficient is 1
            lead = A[:, -1].copy()
            lead[numpy.abs(lead) < eps] = eps
            A = A / lead[:, None]
            A[:, -1] = 1.0 + 0.0j

            Nc = z_chunk.size
            C = numpy.zeros((Nc, deg, deg), dtype=numpy.complex128)

            # Companion first row expects descending lower
            # coeffs: [-a_{deg-1}, ..., -a0]
            C[:, 0, :] = -A[:, :-1][:, ::-1]

            # Subdiagonal ones
            if deg > 1:
                C[:, 1:, :-1] = \
                    numpy.eye(deg - 1, dtype=numpy.complex128)[None, :, :]

            out_view[:] = numpy.linalg.eigvals(C)

        if n <= int(chunk_size):
            solve_chunk(z_flat, out)
        else:
            cs = int(chunk_size)
            for p in range(0, n, cs):
                q = min(p + cs, n)
                solve_chunk(z_flat[p:q], out[p:q])

        return out.reshape(z.shape + (deg,))

    # =======
    # support
    # =======

    def support(self, eta=2e-4, n_probe=4000, thr=5e-4, x_max=None):
        """
        Support intervals of distribution

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

        x_pad : float, default=0.0
            Optional *search padding* fraction. If nonzero, it should only
            expand the probing grid (x-range), not the returned intervals.

        Returns
        -------

        intervals : list of tuple(float, float)
            List of (left, right) support intervals estimated from the grid.

        Notes
        -----

        The support is estimated on a real grid by thresholding the density
        :math:`\\rho(x; \\eta)`.

        See Also
        --------

        density

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 7

            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=1.0)

            >>> print(fl.support)
            [(-1.938184546136533, 3.5198799699925),
             (4.3766691672918245, 7.750937734433611)]
        """

        t = self.t
        lam = float(self.lam)
        a = float(self.a)
        sigma = float(self.sigma)

        if x_max is None:
            # Heuristic: scale grows ~ O(lam * max(t))
            x_max = (1.0 + lam) * float(numpy.max(t)) * 6.0
        x_max = float(x_max)
        if x_max <= 0.0:
            raise ValueError("x_max must be > 0.")

        # Extend probing to negative axis when semicircle part is present or
        # shift is negative
        x_min = 0.0
        if sigma > 0.0:
            # semicircle radius ~ 2*sigma, add a safety factor
            x_min = a - 6.0 * sigma
        elif a < 0.0:
            x_min = a - 0.25 * abs(x_max)

        x = numpy.linspace(float(x_min), float(x_max), int(n_probe))
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

            left = float(xa)
            if float(self.sigma) == 0.0:
                left = float(max(float(self.a), left))

            intervals.append((left, float(xb)))

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

        See Also
        --------

        sample

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

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 7

            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=1.0)

            >>> A = fl.matrix(size=2000)
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

        # Add drift and semicircle (free Gaussian) component
        if float(self.a) != 0.0:
            A += float(self.a) * numpy.eye(n, dtype=numpy.float64)

        sc_scale = float(self.sigma)
        if sc_scale != 0.0:
            W = rng.standard_normal((n, n))
            W = 0.5 * (W + W.T)
            A += (sc_scale * numpy.sqrt(2.0) / numpy.sqrt(float(n))) * W

        return A

    # ====
    # poly
    # ====

    def poly(self):
        """
        Polynomial coefficients implicitly representing the Stieltjes

        Returns
        -------

        coeffs : numpy.ndarray
            A 2D array of size :math:`(d_z + 1) \\times (d_m + 1)` where
            :math:`d_z = \\deg_z(P)` and :math:`d_m = \\deg_m(P)`.

        Notes
        -----

        ``coeffs[i, j]`` is the coefficient of :math:`z^i m^j`.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 7

            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=1.0)

            >>> coeffs = fl.poly()
            >>> print(coeffs.real)
            [[ 1.      7.2125 10.9     7.5    11.    ]
             [ 0.      1.      7.5    11.     -0.    ]]
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
        a = float(self.a)
        sc_var = float(self.sc_var)

        # P(z,m) = (-1 + a m - sc_var m^2 - z m) prod + lam m s
        term0 = numpy.poly1d([-sc_var, a, -1.0]) * prod + \
            numpy.poly1d([lam, 0.0]) * s
        termz = (-1.0) * (numpy.poly1d([1.0, 0.0]) * prod)  # z^1 row

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
