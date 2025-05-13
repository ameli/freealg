# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats import qmc

__all__ = ['qmc_sample']


# =============
# quantile func
# =============

def _quantile_func(x, rho):
    """
    Construct a quantile function from evaluations of an estimated density
    on a grid (x, rho(x)).
    """
    cdf = cumulative_trapezoid(rho, x, initial=0)
    cdf /= cdf[-1]
    return interp1d(cdf, x, bounds_error=False, assume_sorted=True)


# ==========
# qmc sample
# ==========

def qmc_sample(x, rho, num_pts):
    """
    Low-discrepancy sampling from a univariate density estimate using
    Quasi-Monte Carlo.

    Parameters
    ----------
    x : numpy.array, shape (n,)
        Sorted abscissae at which the density has been evaluated.

    rho : numpy.array, shape (n,)
        Density values corresponding to `x`. Must be non-negative and define
        a valid probability density (i.e., integrate to 1 over the support).

    num_pts : int
        Number of sample points to generate from the density estimate.

    Returns
    -------
    samples : numpy.array, shape (num_pts,)
        Samples drawn from the estimated density using a one-dimensional Halton
        sequence mapped through the estimated quantile function.

    See Also
    --------
    scipy.stats.qmc.Halton
        Underlying Quasi-Monte Carlo engine used for generating low-discrepancy
        points.

    Examples
    --------
    .. code-block:: python

        >>> import numpy
        >>> from your_module import qmc_sample
        >>> x = numpy.linspace(0, 1, 200)
        >>> rho = 3 * x**2                    # density of Beta(3,1) on [0,1]
        >>> samples = qmc_sample(x, rho, num_pts=1000)
        >>> assert samples.shape == (1000,)
        >>> # Empirical mean should be close to 3/4
        >>> numpy.allclose(samples.mean(), 0.75, atol=0.02)
    """

    quantile = _quantile_func(x, rho)
    engine = qmc.Halton(d=1)
    u = engine.random(num_pts)
    samples = quantile(u)
    return samples.ravel()
