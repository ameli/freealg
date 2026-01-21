# SPDX-FileCopyrightText: Copyright 2025, Siavash Ameli <sameli@berkeley.edu>
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
from scipy.interpolate import interp1d
from .._free_form._plot_util import plot_density, plot_hilbert, \
    plot_stieltjes, plot_stieltjes_on_disk, plot_samples
from ..visualization import glue_branches

try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import qmc

__all__ = ['MarchenkoPastur']


# ================
# Marchenko Pastur
# ================

class MarchenkoPastur(object):
    """
    Marchenko-Pastur distribution.

    Parameters
    ----------

    lam : float
        Parameter :math:`\\lambda` of the distribution. See Notes.

    Methods
    -------

    density
        Spectral density of distribution.

    hilbert
        Hilbert transform of distribution.

    stieltjes
        Stieltjes transform of distribution.

    sample
        Sample from distribution.

    matrix
        Generate matrix with its empirical spectral density of distribution

    Notes
    -----

    The Marchenko-Pastur distribution has the absolutely-continuous density

    .. math::

        \\mathrm{d} \\rho(x) = \\frac{1}{2 \\pi}
        \\frac{\\sqrt{(\\lambda_{+} - x) (x - \\lambda_{-})}}{\\lambda x}
        \\mathbf{1}_{x \\in [\\lambda_{-}, \\lambda_{+}]} \\mathrm{d}{x}

    where

    * :math:`\\lambda_{\\pm} = (1 \\pm \\sqrt{\\lambda})^2` are the edges of
      the support.
    * :math:`\\lambda > 0` is the shape parameter of the density.

    References
    ----------

    .. [1] Marcenko, V. A., Pastur, L. A. (1967). Distribution of eigenvalues
           for some sets of random matrices. Mathematics of the USSR-Sbornik,
           1(4), 457

    Examples
    --------

    .. code-block:: python

        >>> from freealg.distributions import MarchenkoPastur
        >>> mp = MarchenkoPastur()
    """

    # ====
    # init
    # ====

    def __init__(self, lam, sigma=1.0):
        """
        Initialization.
        """

        self.lam = lam
        self.sigma = sigma

        # self.lam_p = (1 + numpy.sqrt(self.lam))**2
        # self.lam_m = (1 - numpy.sqrt(self.lam))**2
        self.lam_p = sigma**2 * (1.0 + numpy.sqrt(lam))**2
        self.lam_m = sigma**2 * (1.0 - numpy.sqrt(lam))**2

        self.supp = (self.lam_m, self.lam_p)

    # =======
    # density
    # =======

    def density(self, x=None, plot=False, latex=False, save=False, eig=None):
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

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import MarchenkoPastur
            >>> mp = MarchenkoPastur(1/50)
            >>> rho = mp.density(plot=True)

        .. image:: ../_static/images/plots/mp_density.png
            :align: center
            :class: custom-dark
        """

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_p - self.lam_m)
            center = 0.5 * (self.lam_p + self.lam_m)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 500)

        # Unpack parameters
        lam = self.lam
        lam_p = self.lam_p
        lam_m = self.lam_m
        sigma = self.sigma

        rho = numpy.zeros_like(x)
        # mask = numpy.logical_and(x >= self.lam_m, x <= self.lam_p)
        mask = (x > lam_m) & (x < lam_p)

        # rho[mask] = (1.0 / (2.0 * numpy.pi * x[mask] * self.lam)) * \
        #     numpy.sqrt((self.lam_p - x[mask]) * (x[mask] - self.lam_m))

        rho[mask] = numpy.sqrt((lam_p - x[mask]) * (x[mask] - lam_m)) / \
            (lam * x[mask] * 2 * numpy.pi * sigma**2)

        if plot:
            if eig is not None:
                label = 'Theoretical'
            else:
                label = ''
            plot_density(x, rho, label=label, latex=latex, save=save, eig=eig)

        return rho

    # =======
    # support
    # =======

    def support(self):
        """
        supp
        """

        return [self.supp]

    # =======
    # hilbert
    # =======

    def hilbert(self, x=None, plot=False, latex=False, save=False):
        """
        Hilbert transform of the distribution.

        Parameters
        ----------

        x : numpy.array, default=None
            The locations where Hilbert transform is evaluated at. If `None`,
            an interval slightly larger than the supp interval of the
            spectral density is used.

        plot : bool, default=False
            If `True`, Hilbert transform is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        hilb : numpy.array
            Hilbert transform.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import MarchenkoPastur
            >>> mp = MarchenkoPastur(1/50)
            >>> hilb = mp.hilbert(plot=True)

        .. image:: ../_static/images/plots/mp_hilbert.png
            :align: center
            :class: custom-dark
        """

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_p - self.lam_m)
            center = 0.5 * (self.lam_p + self.lam_m)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 500)

        def _P(x):
            return x - 1 + self.lam

        def _Q(x):
            return self.lam * x

        P = _P(x)
        Q = _Q(x)
        Delta2 = P**2 - 4.0 * Q
        Delta = numpy.sqrt(numpy.maximum(Delta2, 0))
        sign = numpy.sign(P)
        hilb = (P - sign * Delta) / (2.0 * Q)

        # using negative sign convention
        hilb = -hilb

        if plot:
            plot_hilbert(x, hilb, support=self.supp, latex=latex, save=save)

        return hilb

    # =======================
    # m mp numeric vectorized
    # =======================

    # def _m_mp_numeric_vectorized(self, z, alt_branch=False, tol=1e-8):
    #     """
    #     Stieltjes transform (principal or secondary branch)
    #     for Marchenko-Pastur distribution on upper half-plane.
    #     """
    #
    #     sigma = 1.0
    #     m = numpy.empty_like(z, dtype=complex)
    #
    #     # When z is too small, do not use quadratic form.
    #     mask = numpy.abs(z) < tol
    #     m[mask] = 1 / (sigma**2 * (1 - self.lam))
    #
    #     # Use quadratic form
    #     not_mask = ~mask
    #     if numpy.any(not_mask):
    #
    #         sign = -1 if alt_branch else 1
    #         A = self.lam * sigma**2 * z[not_mask]
    #         B = z[not_mask] - sigma**2 * (1 - self.lam)
    #         D = B**2 - 4 * A
    #         sqrtD = numpy.sqrt(D)
    #         m1 = (-B + sqrtD) / (2 * A)
    #         m2 = (-B - sqrtD) / (2 * A)
    #
    #         # pick correct branch only for non-masked entries
    #         upper = z[not_mask].imag >= 0
    #         branch = numpy.empty_like(m1)
    #         branch[upper] = numpy.where(sign*m1[upper].imag > 0, m1[upper],
    #                                     m2[upper])
    #         branch[~upper] = numpy.where(sign*m1[~upper].imag < 0,
    #                                      m1[~upper], m2[~upper])
    #         m[not_mask] = branch
    #
    #     return m

    # =============
    # sqrt pos imag
    # =============

    def _sqrt_pos_imag(self, z):
        """
        Square root on a branch cut with always positive imaginary part.
        """

        sq = numpy.sqrt(z)
        sq = numpy.where(sq.imag < 0, -sq, sq)

        return sq

    # ============
    # m mp reflect
    # ============

    # def _m_mp_reflect(self, z, alt_branch=False):
    #     """
    #     Analytic continuation using Schwarz reflection.
    #     """
    #
    #     mask_p = z.imag >= 0.0
    #     mask_n = z.imag < 0.0
    #
    #     m = numpy.zeros_like(z)
    #
    #     f = self._m_mp_numeric_vectorized
    #     m[mask_p] = f(z[mask_p], alt_branch=False)
    #     m[mask_n] = f(z[mask_n], alt_branch=alt_branch)
    #
    #     return m

    # ================
    # stieltjes branch
    # ================

    def _stieltjes_branch(self, z, alt_branch=False, tol=1e-8):
        """
        """

        # Unpack parameters
        lam = self.lam
        sigma = self.sigma

        z = numpy.asarray(z, dtype=complex)
        m = numpy.empty_like(z, dtype=complex)

        def _eval_upper(zu):
            mu = numpy.empty_like(zu, dtype=complex)

            mask = numpy.abs(zu) < tol
            if numpy.any(mask):
                if alt_branch:
                    mu[mask] = numpy.inf + 0.0j
                else:
                    mu[mask] = 1.0 / (sigma**2 * (1.0 - lam))

            not_mask = ~mask
            if numpy.any(not_mask):
                sign = -1 if alt_branch else 1

                A = lam * sigma**2 * zu[not_mask]
                B = zu[not_mask] - sigma**2 * (1.0 - lam)
                D = B**2 - 4.0 * A

                sqrtD = self._sqrt_pos_imag(D)

                r1 = (-B + sqrtD) / (2.0 * A)
                r2 = (-B - sqrtD) / (2.0 * A)

                mu[not_mask] = numpy.where(sign * r1.imag > 0.0, r1, r2)

            return mu

        mask_p = numpy.imag(z) >= 0.0
        if numpy.any(mask_p):
            m[mask_p] = _eval_upper(z[mask_p])

        mask_n = ~mask_p
        if numpy.any(mask_n):
            z_ref = numpy.conjugate(z[mask_n])
            m[mask_n] = numpy.conjugate(_eval_upper(z_ref))

        return m

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z=None, x=None, y=None, alt_branch='both', plot=False,
                  on_disk=False, latex=False, save=False):
        """
        Stieltjes transform of distribution.

        Parameters
        ----------

        x : numpy.array, default=None
            The x axis of the grid where the Stieltjes transform is evaluated.
            If `None`, an interval slightly larger than the supp interval of
            the spectral density is used.

        y : numpy.array, default=None
            The y axis of the grid where the Stieltjes transform is evaluated.
            If `None`, a grid on the interval ``[-1, 1]`` is used.

        alt_branch : {``True``, ``False``, ``'both'``} default=``'both'``
            If `True`, returns non-physical branch. If `False`, returns
            physical branch. If ``'both'``, returns both.

        plot : bool, default=False
            If `True`, Stieltjes transform is plotted.

        on_disk : bool, default=False
            If `True`, the Stieltjes transform is mapped on unit disk. This
            option relevant only if ``plot=True``.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        m1 : numpy.array
            Stieltjes transform on principal branch.

        m12 : numpy.array
            Stieltjes transform on secondary branch.

        Examples
        --------

        .. code-block:: python

            >>> from freealg.distributions import MarchenkoPastur
            >>> mp = MarchenkoPastur(1/50)
            >>> m1, m2 = mp.stieltjes(plot=True)

        .. image:: ../_static/images/plots/mp_stieltjes.png
            :align: center
            :class: custom-dark

        Plot on unit disk using Cayley transform:

        .. code-block:: python

            >>> m1, m2 = mp.stieltjes(plot=True, on_disk=True)

        .. image:: ../_static/images/plots/mp_stieltjes_disk.png
            :align: center
            :class: custom-dark
        """

        if (plot is True) and (on_disk is True):
            n_r = 1000
            n_t = 1000
            r_min, r_max = 0, 2.5
            t_min, t_max = 0, 2.0 * numpy.pi
            r = numpy.linspace(r_min, r_max, n_r)
            t = numpy.linspace(t_min, t_max, n_t + 1)[:-1]
            grid_r, grid_t = numpy.meshgrid(r, t)

            grid_x_D = grid_r * numpy.cos(grid_t)
            grid_y_D = grid_r * numpy.sin(grid_t)
            zeta = grid_x_D + 1j * grid_y_D

            # Cayley transform mapping zeta on D to z on H
            z_H = 1j * (1 + zeta) / (1 - zeta)

            # m1_D = self._m_mp_reflect(z_H, alt_branch=False)
            # m2_D = self._m_mp_reflect(z_H, alt_branch=True)
            m1_D = self._stieltjes_branch(z_H, alt_branch=False)
            m2_D = self._stieltjes_branch(z_H, alt_branch=True)
            m12_D = glue_branches(z_H, m1_D, m2_D)

            plot_stieltjes_on_disk(r, t, m1_D, m12_D, support=self.supp,
                                   latex=latex, save=save)

            if alt_branch == 'both':
                return m1_D, m2_D
            elif alt_branch is True:
                return m2_D
            else:
                return m1_D

        if z is None:
            # Create x if not given
            if x is None:
                radius = 0.5 * (self.lam_p - self.lam_m)
                center = 0.5 * (self.lam_p + self.lam_m)
                scale = 2.0
                x_min = numpy.floor(
                    2.0 * (center - 2.0 * radius * scale)) / 2.0
                x_max = numpy.ceil(
                    2.0 * (center + 2.0 * radius * scale)) / 2.0
                x = numpy.linspace(x_min, x_max, 500)

            # Create y if not given
            if y is None:
                y = numpy.linspace(-1, 1, 400)

            x_grid, y_grid = numpy.meshgrid(x, y)
            z = x_grid + 1j * y_grid              # shape (Ny, Nx)

        # m1 = self._m_mp_reflect(z, alt_branch=False)
        # m2 = self._m_mp_reflect(z, alt_branch=True)
        m1 = self._stieltjes_branch(z, alt_branch=False)
        m2 = self._stieltjes_branch(z, alt_branch=True)

        if plot:
            m12 = glue_branches(z, m1, m2)
            plot_stieltjes(x, y, m1, m12, support=self.supp, latex=latex,
                           save=save)

        if alt_branch == 'both':
            return m1, m2
        elif alt_branch is True:
            return m2
        else:
            return m1

    # ======
    # sample
    # ======

    def sample(self, size, x_min=None, x_max=None, method='qmc', seed=None,
               plot=False, latex=False, save=False):
        """
        Sample from distribution.

        Parameters
        ----------

        size : int
            Size of sample.

        x_min : float, default=None
            Minimum of sample values. If `None`, the left edge of the supp
            is used.

        x_max : float, default=None
            Maximum of sample values. If `None`, the right edge of the supp
            is used.

        method : {``'mc'``, ``'qmc'``}, default= ``'qmc'``
            Method of drawing samples from uniform distribution:

            * ``'mc'``: Monte Carlo
            * ``'qmc'``: Quasi Monte Carlo

        seed : int, default=None,
            Seed for random number generator.

        plot : bool, default=False
            If `True`, samples histogram is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        s : numpy.ndarray
            Samples.

        Notes
        -----

        This method uses inverse transform sampling.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import MarchenkoPastur
            >>> mp = MarchenkoPastur(1/50)
            >>> s = mp.sample(2000)

        .. image:: ../_static/images/plots/mp_samples.png
            :align: center
            :class: custom-dark
        """

        if x_min is None:
            x_min = self.lam_m

        if x_max is None:
            x_max = self.lam_p

        # Grid and PDF
        xs = numpy.linspace(x_min, x_max, size)
        pdf = self.density(xs)

        # CDF (using cumulative trapezoidal rule)
        cdf = cumtrapz(pdf, xs, initial=0)
        cdf /= cdf[-1]  # normalize CDF to 1

        # Inverse CDF interpolator
        inv_cdf = interp1d(cdf, xs, bounds_error=False,
                           fill_value=(x_min, x_max))

        # Random generator
        rng = numpy.random.default_rng(seed)

        # Draw from uniform distribution
        if method == 'mc':
            u = rng.random(size)

        elif method == 'qmc':
            try:
                engine = qmc.Halton(d=1, scramble=True, rng=rng)
            except TypeError:
                # Older scipy versions
                engine = qmc.Halton(d=1, scramble=True, seed=rng)
            u = engine.random(size).ravel()

        else:
            raise NotImplementedError('"method" is invalid.')

        # Draw from distribution by mapping from inverse CDF
        samples = inv_cdf(u).ravel()

        if plot:
            radius = 0.5 * (self.lam_p - self.lam_m)
            center = 0.5 * (self.lam_p + self.lam_m)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 500)
            rho = self.density(x)
            plot_samples(x, rho, x_min, x_max, samples, latex=latex, save=save)

        return samples

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

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import MarchenkoPastur
            >>> mp = MarchenkoPastur(1/50)
            >>> A = mp.matrix(2000)
        """

        # Parameters
        m = int(size / self.lam)

        # Generate random matrix X (n x m) with i.i.d. standard normal entries.
        rng = numpy.random.default_rng(seed)
        X = rng.standard_normal((size, m))

        # Form the sample covariance matrix A = (1/m)*XX^T.
        A = X @ X.T / m

        return A
