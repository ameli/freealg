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
from scipy.stats import beta
# from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import texplot
from ._plot_util import _auto_bins

# Fallback to previous API
if not hasattr(numpy, 'trapezoid'):
    numpy.trapezoid = numpy.trapz

__all__ = ['kde', 'force_density']


# ===
# kde
# ===

def kde(eig, xs, lam_m, lam_p, h, kernel='beta', plot=False):
    """
    Kernel density estimation of eigenvalues.

    Parameters
    ----------

    eig : numpy.array
        1D array of samples of size `n`.

    xs : numpy.array
        1D array of evaluation grid (must lie within ``[lam_m, lam_p]``)

    lam_m : float
        Lower end of the support endpoints  with ``lam_m < lam_p``.

    lam_p : float
        Upper end of the support endpoints  with ``lam_m < lam_p``.

    h : float
        Kernel bandwidth in rescaled units where ``0 < h < 1``.

    kernel : {``'gaussian'``, ``'beta'``}, default= ``'beta'``
        Kernel function using either Gaussian or Beta distribution.

    plot : bool, default=False
        If `True`, the KDE is plotted.

    Returns
    -------

    pdf : numpy.ndarray
        Probability distribution function with the same length as ``xs``.

    See Also
    --------

    freealg.supp
    freealg.sample

    References
    ----------

    .. [1] `R-package documentation for Beta kernel
           <https://search.r-project.org/CRAN/refmans/DELTD/html/Beta.html>`__

    .. [2] Chen, S. X. (1999). Beta Kernel estimators for density functions.
           *Computational Statistics and Data Analysis* 31 p. 131--145.

    Notes
    -----

    In Beta kernel density estimation, the shape parameters :math:`a` and
    :math:`b` of the :math:`\\mathrm{Beta}(a, b)` distribution are computed
    for each data point :math:`u` as:

    .. math::

        a = (u / h) + 1.0
        b = ((1.0 - u) / h) + 1.0

    This is a standard way of using Beta kernel (see R-package documentation
    [1]_). These equations are derived from *moment matching* method, where

    .. math::

        \\mathrm{Mean}(\\mathrm{Beta}(a,b)) = u
        \\mathrm{Var}(\\mathrm{Beta}(a,b)) = (1-u) u h

    Solving these two equations for :math:`a` and :math:`b` yields the
    relations above. See [2]_ (page 134).
    """

    if kernel == 'gaussian':
        pdf = gaussian_kde(eig, bw_method=h)(xs)

        # Adaptive KDE
        # k = KDEUnivariate(eig)
        # k.fit(kernel='gau', bw='silverman', fft=False, weights=None,
        #       gridsize=1024, adaptive=True)
        # pdf = k.evaluate(xs)

    elif kernel == 'beta':

        span = lam_p - lam_m
        if span <= 0:
            raise ValueError('"lam_p" must be larger than "lam_m".')

        # map samples and grid to [0, 1]
        u = (eig - lam_m) / span
        t = (xs - lam_m) / span

        # keep only samples strictly inside (0,1)
        if (u.min() < 0) or (u.max() > 1):
            u = u[(u > 0) & (u < 1)]

        n = u.size
        if n == 0:
            return numpy.zeros_like(xs, dtype=float)

        # Shape parameters "a" and "b" or the kernel Beta(a, b), which is
        # computed for each data point "u" (see notes above). These are
        # vectorized.
        a = (u / h) + 1.0
        b = ((1.0 - u) / h) + 1.0

        # # tiny positive number to keep shape parameters > 0
        eps = 1e-6
        a = numpy.clip(a, eps, None)
        b = numpy.clip(b, eps, None)

        # Beta kernel
        pdf_matrix = beta.pdf(t[None, :], a[:, None], b[:, None])

        # Average and re-normalize back to x variable
        pdf = pdf_matrix.sum(axis=0) / (n * span)

        # Exact zeros outside [lam_m, lam_p]
        pdf[(t < 0) | (t > 1)] = 0.0

    else:
        raise NotImplementedError('"kernel" is invalid.')

    if plot:
        with texplot.theme(use_latex=False):
            fig, ax = plt.subplots(figsize=(6, 4))

            x_min = numpy.min(xs)
            x_max = numpy.max(xs)
            bins = numpy.linspace(x_min, x_max, _auto_bins(eig))
            _ = ax.hist(eig, bins, density=True, color='silver',
                        edgecolor='none', label='Samples histogram')
            ax.plot(xs, pdf, color='black', label='KDE')
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$\\rho(x)$')
            ax.set_xlim([xs[0], xs[-1]])
            ax.set_ylim(bottom=0)
            ax.set_title('Kernel Density Estimation')
            ax.legend(fontsize='x-small')
            plt.show()

    return pdf


# =============
# force density
# =============

def force_density(psi0, support, density, grid, alpha=0.0, beta=0.0):
    """
    Starting from psi0 (raw projection), solve
      min  0.5 ||psi - psi0||^2
      s.t. F_pos psi >= 0           (positivity on grid)
           psi[0] = psi0[0]         (mass)
           f(lam_m) psi = 0         (zero at left edge)
           f(lam_p) psi = 0         (zero at right edge)
    """

    lam_m, lam_p = support

    # Objective and gradient
    def fun(psi):
        return 0.5 * numpy.dot(psi-psi0, psi-psi0)

    def grad(psi):
        return psi - psi0

    # Constraints:
    constraints = []

    # Enforce positivity
    constraints.append({'type': 'ineq',
                        'fun': lambda psi: density(grid, psi)})

    # Enforce unit mass
    constraints.append({
        'type': 'eq',
        'fun': lambda psi: numpy.trapz(density(grid, psi), grid) - 1.0
    })

    # Enforce zero at left edge
    if beta <= 0.0 and beta > -0.5:
        constraints.append({
            'type': 'eq',
            'fun': lambda psi: density(numpy.array([lam_m]), psi)[0]
        })

    # Enforce zero at right edge
    if alpha <= 0.0 and alpha > -0.5:
        constraints.append({
            'type': 'eq',
            'fun': lambda psi: density(numpy.array([lam_p]), psi)[0]
        })

    # Solve a small quadratic programming
    res = minimize(fun, psi0, jac=grad,
                   constraints=constraints,
                   # method='trust-constr',
                   method='SLSQP',
                   options={'maxiter': 1000, 'ftol': 1e-9, 'eps': 1e-8})

    psi = res.x

    # Normalize first mode to unit mass
    x = numpy.linspace(lam_m, lam_p, 1000)
    rho = density(x, psi)
    mass = numpy.trapezoid(rho, x)
    psi[0] = psi[0] / mass

    return psi
