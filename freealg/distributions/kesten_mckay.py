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
import networkx as nx
from scipy.interpolate import interp1d
from .._plot_util import plot_density, plot_hilbert, plot_stieltjes, \
    plot_stieltjes_on_disk, plot_samples

try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import qmc

__all__ = ['KestenMcKay']


# ============
# Kesten McKay
# ============

class KestenMcKay(object):
    """
    Kesten-McKay distribution.

    Parameters
    ----------

    d : float
        Parameter :math:`d` of the distribution. See Notes.

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

        \\mathrm{d} \\rho(x) =
        \\frac{\\sqrt{4(d-1) - x^2}}{2 \\pi (d^2 - x^2)}
        \\mathbf{1}_{x \\in [\\lambda_{-}, \\lambda_{+}]} \\mathrm{d}{x}

    where

    * :math:`\\lambda_{\\pm} = \\pm 2 \\sqrt{d-1}` are the edges of
      the support.
    * :math:`d > 1` is the shape parameter of the density.

    References
    ----------

    .. [1] Kesten, H. (1959). Symmetric random walks on groups. Transactions of
           the American Mathematical Society, 92(2), 336–354.

    .. [2] McKay, B. D. (1981). The expected eigenvalue distribution of a large
           regular graph. Linear Algebra and its Applications, 40, 203-216

    Examples
    --------

    .. code-block:: python

        >>> from freealg.distributions import KestenMcKay
        >>> km = KestenMcKay()
    """

    # ====
    # init
    # ====

    def __init__(self, d):
        """
        Initialization.
        """

        self.d = d
        self.lam_p = 2.0 * numpy.sqrt(d - 1.0)
        self.lam_m = -2.0 * numpy.sqrt(d - 1.0)
        self.support = (self.lam_m, self.lam_p)

    # =======
    # density
    # =======

    def density(self, x=None, plot=False, latex=False, save=False):
        """
        Density of distribution.

        Parameters
        ----------

        x : numpy.array, default=None
            The locations where density is evaluated at. If `None`, an interval
            slightly larger than the support interval of the spectral density
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

        Returns
        -------

        rho : numpy.array
            Density.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import KestenMcKay
            >>> km = KestenMcKay(3)
            >>> rho = km.density(plot=True)

        .. image:: ../_static/images/plots/km_density.png
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

        rho = numpy.zeros_like(x)
        mask = numpy.logical_and(x > self.lam_m, x < self.lam_p)

        rho[mask] = (self.d / (2.0 * numpy.pi * (self.d**2 - x[mask]**2))) * \
            numpy.sqrt(4.0 * (self.d - 1.0) - x[mask]**2)

        if plot:
            plot_density(x, rho, label='', latex=latex, save=save)

        return rho

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
            an interval slightly larger than the support interval of the
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

            >>> from freealg.distributions import KestenMcKay
            >>> km = KestenMcKay(3)
            >>> hilb = km.hilbert(plot=True)

        .. image:: ../_static/images/plots/km_hilbert.png
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
            return (self.d - 2.0) * x

        def _Q(x):
            return self.d**2 - x**2

        P = _P(x)
        Q = _Q(x)
        Delta2 = P**2 - 4.0 * Q
        Delta = numpy.sqrt(numpy.maximum(Delta2, 0))
        sign = numpy.sign(P)
        hilb = (P - sign * Delta) / (2.0 * Q)

        # using negative sign convention
        hilb = -hilb

        if plot:
            plot_hilbert(x, hilb, support=self.support, latex=latex, save=save)

        return hilb

    # =======================
    # m mp numeric vectorized
    # =======================

    def _m_mp_numeric_vectorized(self, z, alt_branch=False, tol=1e-8):
        """
        Stieltjes transform (principal or secondary branch)
        for Marchenko–Pastur distribution on upper half-plane.
        """

        m = numpy.empty_like(z, dtype=complex)

        sign = -1 if alt_branch else 1
        A = self.d**2 - z**2
        B = (self.d - 2.0) * z
        D = B**2 - 4 * A
        sqrtD = numpy.sqrt(D)
        m1 = (-B + sqrtD) / (2 * A)
        m2 = (-B - sqrtD) / (2 * A)

        # pick correct branch
        upper = z.imag >= 0
        branch = numpy.empty_like(m1)
        branch[upper] = numpy.where(sign*m1[upper].imag > 0, m1[upper],
                                    m2[upper])
        branch[~upper] = numpy.where(sign*m1[~upper].imag < 0, m1[~upper],
                                     m2[~upper])
        m = branch

        return m

    # ============
    # m mp reflect
    # ============

    def _m_mp_reflect(self, z, alt_branch=False):
        """
        Analytic continuation using Schwarz reflection.
        """

        mask_p = z.imag >= 0.0
        mask_n = z.imag < 0.0

        m = numpy.zeros_like(z)

        f = self._m_mp_numeric_vectorized
        m[mask_p] = f(z[mask_p], alt_branch=False)
        m[mask_n] = f(z[mask_n], alt_branch=alt_branch)

        return m

    # =========
    # stieltjes
    # =========

    def stieltjes(self, x=None, y=None, plot=False, on_disk=False, latex=False,
                  save=False):
        """
        Stieltjes transform of distribution.

        Parameters
        ----------

        x : numpy.array, default=None
            The x axis of the grid where the Stieltjes transform is evaluated.
            If `None`, an interval slightly larger than the support interval of
            the spectral density is used.

        y : numpy.array, default=None
            The y axis of the grid where the Stieltjes transform is evaluated.
            If `None`, a grid on the interval ``[-1, 1]`` is used.

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

            >>> from freealg.distributions import KestenMcKay
            >>> km = KestenMcKay(3)
            >>> m1, m2 = km.stieltjes(plot=True)

        .. image:: ../_static/images/plots/km_stieltjes.png
            :align: center
            :class: custom-dark

        Plot on unit disk using Cayley transform:

        .. code-block:: python

            >>> m1, m2 = mp.stieltjes(plot=True, on_disk=True)

        .. image:: ../_static/images/plots/km_stieltjes_disk.png
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

            m1_D = self._m_mp_reflect(z_H, alt_branch=False)
            m2_D = self._m_mp_reflect(z_H, alt_branch=True)

            plot_stieltjes_on_disk(r, t, m1_D, m2_D, support=self.support,
                                   latex=latex, save=save)

            return m1_D, m2_D

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_p - self.lam_m)
            center = 0.5 * (self.lam_p + self.lam_m)
            scale = 2.0
            x_min = numpy.floor(2.0 * (center - 2.0 * radius * scale)) / 2.0
            x_max = numpy.ceil(2.0 * (center + 2.0 * radius * scale)) / 2.0
            x = numpy.linspace(x_min, x_max, 500)

        # Create y if not given
        if y is None:
            y = numpy.linspace(-1, 1, 400)

        x_grid, y_grid = numpy.meshgrid(x, y)
        z = x_grid + 1j * y_grid              # shape (Ny, Nx)

        m1 = self._m_mp_reflect(z, alt_branch=False)
        m2 = self._m_mp_reflect(z, alt_branch=True)

        if plot:
            plot_stieltjes(x, y, m1, m2, support=self.support, latex=latex,
                           save=save)

        return m1, m2

    # ======
    # sample
    # ======

    def sample(self, size, x_min=None, x_max=None, method='qmc', plot=False,
               latex=False, save=False):
        """
        Sample from distribution.

        Parameters
        ----------

        size : int
            Size of sample.

        x_min : float, default=None
            Minimum of sample values. If `None`, the left edge of the support
            is used.

        x_max : float, default=None
            Maximum of sample values. If `None`, the right edge of the support
            is used.

        method : {``'mc'``, ``'qmc'``}, default= ``'qmc'``
            Method of drawing samples from uniform distirbution:

            * ``'mc'``: Monte Carlo
            * ``'qmc'``: Quasi Monte Carlo

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

            >>> from freealg.distributions import KestenMcKay
            >>> km = KestenMcKay(3)
            >>> s = km.sample(2000)

        .. image:: ../_static/images/plots/km_samples.png
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

        # Draw from uniform distribution
        if method == 'mc':
            u = numpy.random.rand(size)
        elif method == 'qmc':
            engine = qmc.Halton(d=1)
            u = engine.random(size)
        else:
            raise ValueError('"method" is invalid.')

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

    def matrix(self, size):
        """
        Generate matrix with the spectral density of the distribution.

        Parameters
        ----------

        size : int
            Size :math:`n` of the matrix.

        Returns
        -------

        A : numpy.ndarray
            A matrix of the size :math:`n \\times n`.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import KestenMcKay
            >>> km = KestenMcKay(3)
            >>> A = km.matrix(2000)
        """

        n = size
        G = nx.random_regular_graph(self.d, n)
        A = nx.to_numpy_array(G, dtype=float)  # shape (n,n)

        mu = self.d / n
        A_c = A - mu * numpy.ones((n, n))

        return A_c
