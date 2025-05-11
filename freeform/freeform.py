# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
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
from ._util import compute_eig, force_density
from ._jacobi import jacobi_proj, jacobi_approx, jacobi_stieltjes
from ._chebysev import chebyshev_proj, chebyshev_prrox, chebyshev_stieltjes
from ._damp import jackson_damping, lanczos_damping, fejer_damping, \
    exponential_damping, parzen_damping
from ._plot_util import plot_coeff, plot_density, plot_hilbert, plot_stieltjes

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

    psi : np.array
        Jacobi coefficients.

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

    Examples
    --------

    .. code-block:: python

        >>> from freeform import FreeForm
    """

    # ====
    # init
    # ====

    def __init__(self, A, support=None):
        """
        Initialization.
        """

        self.A = None
        self.eig = None

        # Eigenvalues
        if A.ndim == 1:
            # When A is a 1D array, it is assumed A is the eigenvalue array.
            self.eig = A
        elif A.ndim == 2:
            # When A is a 2D array, it is assumed A is the actual array,
            # and its eigenvalues will be computed.
            self.A = A
            self.eig = compute_eig(A)

        # Support
        if support is None:
            self.lam_m = self.eig.min()
            self.lam_p = self.eig.max()
        else:
            self.lam_m = support[0]
            self.lam_p = support[1]
        self.support = (self.lam_m, self.lam_p)

        # Initialize
        self.method = None
        self.psi = None
        self.alpha = None
        self.beta = None

    # ===
    # fit
    # ===

    def fit(self, method='jacobi', K=10, alpha=0.0, beta=0.0, reg=0, damp=None,
            force=False, plot=False, latex=False, save=False):
        """
        Fit model to eigenvalues.

        Parameters
        ----------

        method : {``'jacobi'``, ``'chebyshev'``}, default=``'jacobi'``
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

        reg : float, default=0
            Regularization coefficient.

        damp : {``'jackson'``, ``'lanczos'``, ``'fejer``, ``'exponential'``,\
                ``'parzen'``}, default=None
            Damping method to eliminate Gibbs oscillation.

        force : bool, default=False
            If `True`, it forces the density to have unit mass and to be
            strictly positive.

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

        psi : (K+1, ) numpy.ndarray
            Coefficients of fitting Jacobi polynomials
        """

        if alpha <= -1:
            raise ValueError('"alpha" should be greater then "-1".')

        if beta <= -1:
            raise ValueError('"beta" should be greater then "-1".')

        # Project eigenvalues to Jacobi polynomials basis
        if method  == 'jacobi':
            psi = jacobi_proj(self.eig, support=self.support, K=K, alpha=alpha,
                              beta=beta, reg=reg)
        elif method == 'chebyshev':
            psi = chebyshev_proj(self.eig, support=self.support, K=K)
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

            self.psi *= g

        if force:
            # A positivity grid
            pos_grid = numpy.linspace(self.lam_m, self.lam_p, 300)

            # Enforce positivity, unit mass, and zero at edges
            psi = force_density(psi, support=self.support, alpha=alpha,
                                beta=beta, pos_grid=pos_grid)

        self.method = method
        self.psi = psi
        self.alpha = alpha
        self.beta = beta

        if plot:
            plot_coeff(psi, latex=latex, save=save)

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
        rho[mask] = jacobi_approx(x[mask], self.psi, self.support, self.alpha,
                                  self.beta)

        # Check density is unit mass
        mass = numpy.trapz(rho, x)
        if not numpy.isclose(mass, 1.0):
            raise RuntimeWarning(f'"rho" is not unit mass. mass: {mass}. ' +
                                 r'Set "force=True".')

        # Check density is positive
        min_rho = numpy.min(rho)
        if min_rho < 0.0:
            raise RuntimeError(f'"rho" is not positive. min_rho: {min_rho}. ' +
                                 r'Set "force=True".')

        if plot:
            plot_density(x, rho, self.eig, self.support, latex=latex,
                         save=save)

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

        if plot:
            plot_hilbert(x, hilb, self.support, latex=latex, save=save)

        return hilb

    # =========
    # stieltjes
    # =========

    def stieltjes(self, x, y, plot=False, latex=False, save=False):
        """
        Compute Stieltjes transform of the spectral density.

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

        # Create y if not given
        if y is None:
            y = numpy.linspace(-1, 1, 400)

        x_grid, y_grid = numpy.meshgrid(x, y)
        z = x_grid + 1j * y_grid              # shape (Ny, Nx)

        # Set the number of bases as the number of x points insides support
        mask_sup = numpy.logical_and(x >= self.lam_m, x <= self.lam_p)
        n_base = 2 * numpy.sum(mask_sup)

        mask_p = y >= 0.0
        mask_n = y < 0.0

        m1 = numpy.zeros_like(z)
        m2 = numpy.zeros_like(z)

        # Upper half-plane
        m1[mask_p, :] = jacobi_stieltjes(z[mask_p, :], self.psi, self.support,
                                         alpha=self.alpha, beta=self.beta,
                                         n_base=n_base)

        # Lower half-plane, use Schwzrt reflection
        m1[mask_n, :] = numpy.conjugate(jacobi_stieltjes(
            numpy.conjugate(z[mask_n, :]), self.psi, self.support,
            alpha=self.alpha, beta=self.beta, n_base=n_base))

        m2[mask_p, :] = m1[mask_p, :]
        m2[mask_m, :] = m1[mask_m, :]  # TEST

        if plot:
            plot_stieltjes(x, y, m1, m2, self.support, latex=latex, save=save)

        return m1, m2
