# SPDX-FileCopyrightText: Copyright 2025, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import numpy
from scipy.stats import gaussian_kde
# from statsmodels.nonparametric.kde import KDEUnivariate
from functools import partial
from ._util import compute_eig, beta_kde, force_density
from ._jacobi import jacobi_sample_proj, jacobi_kernel_proj, jacobi_approx, \
    jacobi_stieltjes
from ._chebyshev import chebyshev_sample_proj, chebyshev_kernel_proj, \
    chebyshev_approx, chebyshev_stieltjes
from ._damp import jackson_damping, lanczos_damping, fejer_damping, \
    exponential_damping, parzen_damping
from ._plot_util import plot_fit, plot_density, plot_hilbert, plot_stieltjes
from ._pade import fit_pade, eval_pade
from ._decompress import decompress
from ._sample import qmc_sample

__all__ = ['FreeForm']


# =========
# Free Form
# =========

class FreeForm(object):
    """
    Free probability for large matrices.

    Parameters
    ----------

    A : numpy.ndarray
        The 2D symmetric :math:`\\mathbf{A}`. The eigenvalues of this will be
        computed upon calling this class. If a 1D array provided, it is
        assumed to be the eigenvalues of :math:`\\mathbf{A}`.

    support : tuple, default=None
        The support of the density of :math:`\\mathbf{A}`. If `None`, it is
        estimated from the minimum and maximum of the eigenvalues.

    p : float, default=0.001
        The edges of the support of the distribution is detected by the
        :math:`p`-quantile on the left and :math:`(1-p)`-quantile on the right.
        If the argument ``support`` is directly provided, this option is
        ignored. This value should be between 0 and 1, ideally a small
        number close to zero.

    Notes
    -----

    Notes.

    References
    ----------

    .. [1] Reference.

    Attributes
    ----------

    eig : numpy.array
        Eigenvalues of the matrix

    psi : numpy.array
        Jacobi coefficients.

    n   : int
        Initial array size (assuming a square matrix when :math:`\\mathbf{A}`
        is 2D)

    Methods
    -------

    fit
        Fit the Jacobi polynomials to the empirical density.

    density
        Compute the spectral density of the matrix.

    hilbert
        Compute Hilbert transform of the spectral density

    stieltjes
        Compute Stieltjes transform of the spectral density

    decompress
        Free decompression of spectral density

    Examples
    --------

    .. code-block:: python

        >>> from freealg import FreeForm
    """

    # ====
    # init
    # ====

    def __init__(self, A, support=None, p=0.001):
        """
        Initialization.
        """

        self.A = None
        self.eig = None

        # Eigenvalues
        if A.ndim == 1:
            # When A is a 1D array, it is assumed A is the eigenvalue array.
            self.eig = A
            self.n = len(A)
        elif A.ndim == 2:
            # When A is a 2D array, it is assumed A is the actual array,
            # and its eigenvalues will be computed.
            self.A = A
            self.n = A.shape[0]
            assert A.shape[0] == A.shape[1], \
                'Only square matrices are permitted.'
            self.eig = compute_eig(A)

        # Support
        if support is None:
            self.lam_m, self.lam_p = self._detect_support(self.eig, p,
                                                          smoothen=True)
        else:
            self.lam_m = support[0]
            self.lam_p = support[1]
        self.support = (self.lam_m, self.lam_p)

        # Initialize
        self.method = None
        self.psi = None
        self.alpha = None
        self.beta = None
        self._pade_sol = None

    # ==============
    # detect support
    # ==============

    def _detect_support(self, eig, p, smoothen=True):
        """
        """

        # Using quantile directly.
        if smoothen:
            kde = gaussian_kde(eig)
            xs = numpy.linspace(eig.min(), eig.max(), 1000)
            fs = kde(xs)

            cdf = numpy.cumsum(fs)
            cdf /= cdf[-1]

            lam_m = numpy.interp(p, cdf, xs)
            lam_p = numpy.interp(1-p, cdf, xs)
        else:
            lam_m, lam_p = numpy.quantile(eig, [p, 1-p])

        return lam_m, lam_p

    # ===
    # fit
    # ===

    def fit(self, method='jacobi', K=10, alpha=0.0, beta=0.0, reg=0.0,
            projection='gaussian', kernel_bw=None, damp=None, force=False,
            pade_p=0, pade_q=1, odd_side='left', pade_reg=0.0, optimizer='ls',
            plot=False, latex=False, save=False):
        """
        Fit model to eigenvalues.

        Parameters
        ----------

        method : {``'jacobi'``, ``'chebyshev'``}, default= ``'jacobi'``
            Method of approximation, either by Jacobi polynomials or Chebyshev
            polynomials of the second kind.

        K : int, default=10
            Highest polynomial degree

        alpha : float, default=0.0
            Jacobi parameter :math:`\\alpha`. Determines the slope of the
            fitting model on the right side of interval. This should be greater
            then -1. This option is only applicable when ``method='jacobi'``.

        beta : float, default=0.0
            Jacobi parameter :math:`\\beta`. Determines the slope of the
            fitting model on the left side of interval. This should be greater
            then -1. This option is only applicable when ``method='jacobi'``.

        reg : float, default=0.0
            Tikhonov regularization coefficient.

        projection : {``'sample'``, ``'gaussian'``, ``'beta'``}, \
                default= ``'beta'``
            The method of Galerkin projection:

            * ``'sample'``: directly project samples (eigenvalues) to the
              orthogonal polynomials. This method is highly unstable as it
              treats each sample as a delta Dirac function.
            * ``'gaussian'``: computes Gaussian-Kernel KDE from the samples and
              project a smooth KDE to the orthogonal polynomials. This method
              is stable.
            * ``'beta'``: computes Beta-Kernel KDE from the samples and
              project a smooth KDE to the orthogonal polynomials. This method
              is stable.

        kernel_bw : float, default=0.001
            Kernel band-wdth. See scipy.stats.gaussian_kde. This argument is
            relevant if ``projection='kernel'`` is set.

        damp : {``'jackson'``, ``'lanczos'``, ``'fejer``, ``'exponential'``,\
                ``'parzen'``}, default=None
            Damping method to eliminate Gibbs oscillation.

        force : bool, default=False
            If `True`, it forces the density to have unit mass and to be
            strictly positive.

        pade_p : int, default=0
            Degree of polynomial :math:`P(z)` is :math:`q+p` where :math:`p`
            can only be ``-1``, ``0``, or ``1``. See notes below.

        pade_q : int, default=1
            Degree of polynomial :math:`Q(z)` is :math:`q`. See notes below.

        odd_side : {``'left'``, ``'right'``}, default= ``'left'``
            In case of odd number of poles (when :math:`q` is odd), the extra
            pole is set to the left or right side of the support interval,
            while all other poles are split in half to the left and right. Note
            that this is only for the initialization of the poles. The
            optimizer will decide best location by moving them to the left or
            right of the support.

        pade_reg : float, default=0.0
            Regularization for Pade approximation.

        optimizer : {``'ls'``, ``'de'``}, default= ``'ls'``
            Optimizer for Pade approximation, including:

            * ``'ls'``: least square (local, fast)
            * ``'de'``: differential evolution (global, slow)

        plot : bool, default=False
            If `True`, the approximation coefficients and Pade approximation to
            the Hilbert transform are plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        psi : (K+1, ) numpy.ndarray
            Coefficients of fitting Jacobi polynomials

        Notes
        -----

        The Pade approximation for the glue function :math:`G(z)` is

        .. math::

            G(z) = \\frac{P(z)}{Q(z)},

        where :math:`P(z)` and :math:`Q(z)` are polynomials of order
        :math:`p+q` and :math:`q` respectively. Note that :math:`p` can only
        be -1, 0, or 1, effectively making Pade approximation of order
        :math:`q-1:q`, :math:`q:q`, or :math:`q-1:q`.

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if alpha <= -1:
            raise ValueError('"alpha" should be greater then "-1".')

        if beta <= -1:
            raise ValueError('"beta" should be greater then "-1".')

        if not (method in ['jacobi', 'chebyshev']):
            raise ValueError('"method" is invalid.')

        if not (projection in ['sample', 'gaussian', 'beta']):
            raise ValueError('"projection" is invalid.')

        # Project eigenvalues to Jacobi polynomials basis
        if method == 'jacobi':

            if projection == 'sample':
                psi = jacobi_sample_proj(self.eig, support=self.support, K=K,
                                         alpha=alpha, beta=beta, reg=reg)
            else:
                # smooth KDE on a fixed grid
                xs = numpy.linspace(self.lam_m, self.lam_p, 2000)

                if projection == 'gaussian':
                    pdf = gaussian_kde(self.eig, bw_method=kernel_bw)(xs)
                else:
                    pdf = beta_kde(self.eig, xs, self.lam_m, self.lam_p,
                                   kernel_bw)

                # Adaptive KDE
                # k = KDEUnivariate(self.eig)
                # k.fit(bw="silverman", fft=False, weights=None, gridsize=1024,
                #       adaptive=True)
                # pdf = k.evaluate(xs)

                # TEST
                # import matplotlib.pyplot as plt
                # plt.plot(xs, pdf)
                # plt.grid(True)
                # plt.show()

                psi = jacobi_kernel_proj(xs, pdf, support=self.support, K=K,
                                         alpha=alpha, beta=beta, reg=reg)

        elif method == 'chebyshev':

            if projection == 'sample':
                psi = chebyshev_sample_proj(self.eig, support=self.support,
                                            K=K, reg=reg)
            else:
                # smooth KDE on a fixed grid
                xs = numpy.linspace(self.lam_m, self.lam_p, 2000)

                if projection == 'gaussian':
                    pdf = gaussian_kde(self.eig, bw_method=kernel_bw)(xs)
                else:
                    pdf = beta_kde(self.eig, xs, self.lam_m, self.lam_p,
                                   kernel_bw)

                # Adaptive KDE
                # k = KDEUnivariate(self.eig)
                # k.fit(bw="silverman", fft=False, weights=None, gridsize=1024,
                #       adaptive=True)
                # pdf = k.evaluate(xs)

                psi = chebyshev_kernel_proj(xs, pdf, support=self.support,
                                            K=K, reg=reg)
        else:
            raise ValueError('"method" is invalid.')

        # Damping
        if damp is not None:
            if damp == 'jackson':
                g = jackson_damping(K+1)
            elif damp == 'lanczos':
                g = lanczos_damping(K+1)
            elif damp == 'fejer':
                g = fejer_damping(K+1)
            elif damp == 'exponential':
                g = exponential_damping(K+1)
            elif damp == 'parzen':
                g = parzen_damping(K+1)

            psi = psi * g

        if force:
            # A grid to check and enforce positivity and unit mass on it
            grid = numpy.linspace(self.lam_m, self.lam_p, 500)

            if method == 'jacobi':
                approx = partial(jacobi_approx, support=self.support,
                                 alpha=alpha, beta=beta)
            elif method == 'chebyshev':
                approx = partial(chebyshev_approx, support=self.support)
            else:
                raise RuntimeError('"method" is invalid.')

            # Enforce positivity, unit mass, and zero at edges
            psi = force_density(psi, support=self.support, approx=approx,
                                grid=grid, alpha=alpha, beta=beta)

        # Update attributes
        self.method = method
        self.psi = psi
        self.alpha = alpha
        self.beta = beta

        # For holomorphic continuation for the lower half-plane
        x_supp = numpy.linspace(self.lam_m, self.lam_p, 1000)
        g_supp = 2.0 * numpy.pi * self.hilbert(x_supp)

        # Fit a pade approximation
        # self._pade_sol = fit_pade(x_supp, g_supp, self.lam_m,
        #                           self.lam_p, pade_p, pade_q, delta=1e-8,
        #                           B=numpy.inf, S=numpy.inf)
        self._pade_sol = fit_pade(x_supp, g_supp, self.lam_m, self.lam_p,
                                  p=pade_p, q=pade_q, odd_side=odd_side,
                                  pade_reg=pade_reg, safety=1.0, max_outer=40,
                                  xtol=1e-12, ftol=1e-12, optimizer=optimizer,
                                  verbose=0)

        if plot:
            g_supp_approx = eval_pade(x_supp[None, :], self._pade_sol)[0, :]
            plot_fit(psi, x_supp, g_supp, g_supp_approx, support=self.support,
                     latex=latex, save=save)

        return self.psi

    # =======
    # density
    # =======

    def density(self, x=None, plot=False, latex=False, save=False):
        """
        Evaluate density.

        Parameters
        ----------

        x : numpy.array, default=None
            Positions where density to be evaluated at. If `None`, an interval
            slightly larger than the support interval will be used.

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
            Density at locations x.

        See Also
        --------
        hilbert
        stieltjes

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if self.psi is None:
            raise RuntimeError('"fit" the model first.')

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_p - self.lam_m)
            center = 0.5 * (self.lam_p + self.lam_m)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 500)

        # Preallocate density to zero
        rho = numpy.zeros_like(x)

        # Compute density only inside support
        mask = numpy.logical_and(x >= self.lam_m, x <= self.lam_p)

        if self.method == 'jacobi':
            rho[mask] = jacobi_approx(x[mask], self.psi, self.support,
                                      self.alpha, self.beta)
        elif self.method == 'chebyshev':
            rho[mask] = chebyshev_approx(x[mask], self.psi, self.support)
        else:
            raise RuntimeError('"method" is invalid.')

        # Check density is unit mass
        mass = numpy.trapz(rho, x)
        if not numpy.isclose(mass, 1.0, atol=1e-2):
            # raise RuntimeWarning(f'"rho" is not unit mass. mass: {mass}. ' +
            #                      r'Set "force=True".')
            print(f'"rho" is not unit mass. mass: {mass}. Set "force=True".')

        # Check density is positive
        min_rho = numpy.min(rho)
        if min_rho < 0.0 - 1e-3:
            # raise RuntimeWarning(
            #         f'"rho" is not positive. min_rho: {min_rho}. Set ' +
            #         r'"force=True".')
            print(f'"rho" is not positive. min_rho: {min_rho}. Set ' +
                  r'"force=True".')

        if plot:
            plot_density(x, rho, eig=self.eig, support=self.support,
                         label='Estimate', latex=latex, save=save)

        return rho

    # =======
    # hilbert
    # =======

    def hilbert(self, x=None, rho=None, plot=False, latex=False, save=False):
        """
        Compute Hilbert transform of the spectral density.

        Parameters
        ----------

        x : numpy.array, default=None
            The locations where Hilbert transform is evaluated at. If `None`,
            an interval slightly larger than the support interval of the
            spectral density is used.

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

        hilb : numpy.array
            The Hilbert transform on the locations `x`.

        See Also
        --------
        density
        stieltjes

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if self.psi is None:
            raise RuntimeError('"fit" the model first.')

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_p - self.lam_m)
            center = 0.5 * (self.lam_p + self.lam_m)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 500)

        # if (numpy.min(x) > self.lam_m) or (numpy.max(x) < self.lam_p):
        #     raise ValueError('"x" does not encompass support interval.')

        # Preallocate density to zero
        if rho is None:
            rho = self.density(x)

        # mask of support [lam_m, lam_p]
        mask = numpy.logical_and(x >= self.lam_m, x <= self.lam_p)
        x_s = x[mask]
        rho_s = rho[mask]

        # Form the matrix of integrands: rho_s / (t - x_i)
        # Here, we have diff[i,j] = x[i] - x_s[j]
        diff = x[:, None] - x_s[None, :]
        D = rho_s[None, :] / diff

        # Principal‐value: wherever t == x_i, then diff == 0, zero that entry
        # (numpy.isclose handles floating‐point exactly)
        D[numpy.isclose(diff, 0.0)] = 0.0

        # Integrate each row over t using trapezoid rule on x_s
        # Namely, hilb[i] = int rho_s(t)/(t - x[i]) dt
        hilb = numpy.trapz(D, x_s, axis=1) / numpy.pi

        # We use negative sign convention
        hilb = -hilb

        if plot:
            plot_hilbert(x, hilb, support=self.support, latex=latex,
                         save=save)

        return hilb

    # ====
    # glue
    # ====

    def _glue(self, z):
        """
        """

        # Glue function
        g = eval_pade(z, self._pade_sol)

        return g

    # =========
    # stieltjes
    # =========

    def stieltjes(self, x, y, plot=False, latex=False, save=False):
        """
        Compute Stieltjes transform of the spectral density over a 2D Cartesian
        grid on the complex plane.

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

        m_p : numpy.ndarray
            The Stieltjes transform on the principal branch.

        m_m : numpy.ndarray
            The Stieltjes transform continued to the secondary branch.

        See Also
        --------
        density
        hilbert

        Notes
        -----

        Notes.

        References
        ----------

        .. [1] tbd

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if self.psi is None:
            raise RuntimeError('"fit" the model first.')

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

        # Set the number of bases as the number of x points insides support
        mask_sup = numpy.logical_and(x >= self.lam_m, x <= self.lam_p)
        n_base = 2 * numpy.sum(mask_sup)

        # Stieltjes function
        if self.method == 'jacobi':
            stieltjes = partial(jacobi_stieltjes, psi=self.psi,
                                support=self.support, alpha=self.alpha,
                                beta=self.beta, n_base=n_base)
        elif self.method == 'chebyshev':
            stieltjes = partial(chebyshev_stieltjes, psi=self.psi,
                                support=self.support)

        mask_p = y >= 0.0
        mask_m = y < 0.0

        m1 = numpy.zeros_like(z)
        m2 = numpy.zeros_like(z)

        # Upper half-plane
        m1[mask_p, :] = stieltjes(z[mask_p, :])

        # Lower half-plane, use Schwarz reflection
        m1[mask_m, :] = numpy.conjugate(
            stieltjes(numpy.conjugate(z[mask_m, :])))

        # Second Riemann sheet
        m2[mask_p, :] = m1[mask_p, :]
        m2[mask_m, :] = -m1[mask_m, :] + self._glue(z[mask_m, :])

        if plot:
            plot_stieltjes(x, y, m1, m2, self.support, latex=latex, save=save)

        return m1, m2

    # ==============
    # eval stieltjes
    # ==============

    def _eval_stieltjes(self, z):
        """
        Compute Stieltjes transform of the spectral density.

        Parameters
        ----------

        z : numpy.array
            The z values in the complex plan where the Stieltjes transform is
            evaluated.


        Returns
        -------

        m_p : numpy.ndarray
            The Stieltjes transform on the principal branch.

        m_m : numpy.ndarray
            The Stieltjes transform continued to the secondary branch.

        See Also
        --------
        density
        hilbert

        Notes
        -----

        Notes.

        References
        ----------

        .. [1] tbd

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if self.psi is None:
            raise RuntimeError('"fit" the model first.')

        z = numpy.asarray(z)
        shape = z.shape
        if len(shape) == 0:
            shape = (1,)
        z = z.reshape(-1, 1)

        # Stieltjes function
        if self.method == 'jacobi':
            stieltjes = partial(jacobi_stieltjes, psi=self.psi,
                                support=self.support, alpha=self.alpha,
                                beta=self.beta)
        elif self.method == 'chebyshev':
            stieltjes = partial(chebyshev_stieltjes, psi=self.psi,
                                support=self.support)

        mask_p = z.imag >= 0.0
        mask_m = z.imag < 0.0

        m1 = numpy.zeros_like(z)
        m2 = numpy.zeros_like(z)

        # Upper half-plane
        m1[mask_p] = stieltjes(z[mask_p].reshape(-1, 1)).reshape(-1)

        # Lower half-plane, use Schwarz reflection
        m1[mask_m] = numpy.conjugate(
            stieltjes(numpy.conjugate(z[mask_m].reshape(-1, 1)))).reshape(-1)

        # Second Riemann sheet
        m2[mask_p] = m1[mask_p]
        m2[mask_m] = -m1[mask_m] + self._glue(
            z[mask_m].reshape(-1, 1)).reshape(-1)

        m1, m2 = m1.reshape(*shape), m2.reshape(*shape)

        return m1, m2

    # ==========
    # decompress
    # ==========

    def decompress(self, size, x=None, delta=1e-6, iterations=500,
                   step_size=0.1, tolerance=1e-4, plot=False, latex=False,
                   save=False):
        """
        Free decompression of spectral density.

        Parameters
        ----------

        size : int
            Size of the decompressed matrix.

        x : numpy.array, default=None
            Positions where density to be evaluated at. If `None`, an interval
            slightly larger than the support interval will be used.

        delta: float, default=1e-4
            Size of the perturbation into the upper half plane for Plemelj's
            formula.

        iterations: int, default=500
            Maximum number of Newton iterations.

        step_size: float, default=0.1
            Step size for Newton iterations.

        tolerance: float, default=1e-4
            Tolerance for the solution obtained by the Newton solver. Also
            used for the finite difference approximation to the derivative.

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
            Spectral density

        eigs : numpy.array
            Estimated eigenvalues as low-discrepancy samples of the estimated
            spectral density.

        See Also
        --------

        density
        stieltjes

        Notes
        -----

        Work in progress.

        References
        ----------

        .. [1] tbd

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        size = int(size)

        rho, x, (lb, ub) = decompress(self, size, x=x, delta=delta,
                                      iterations=iterations,
                                      step_size=step_size, tolerance=tolerance)
        x, rho = x.ravel(), rho.ravel()

        if plot:
            plot_density(x, rho, support=(lb, ub),
                         label='Decompression', latex=latex, save=save)

        eigs = numpy.sort(qmc_sample(x, rho, size))

        return rho, eigs
