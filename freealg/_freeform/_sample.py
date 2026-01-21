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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.stats import qmc

__all__ = ['sample']


# =============
# quantile func
# =============

def _quantile_func(x, rho, clamp=1e-4, eps=1e-8):
    """
    Construct a quantile function from evaluations of an estimated density
    on a grid (x, rho(x)).
    """

    rho_clamp = rho.copy()
    rho_clamp[rho < clamp] = eps
    cdf = cumulative_trapezoid(rho_clamp, x, initial=0)
    cdf /= cdf[-1]
    cdf_inv = PchipInterpolator(cdf, x, extrapolate=False)

    return cdf_inv


# ======
# sample
# ======

def sample(x, rho, num_pts, method='qmc', seed=None):
    """
    Low-discrepancy sampling from density estimate.

    Parameters
    ----------

    x : numpy.array
        Sorted abscissae at which the density has been evaluated. Shape `(n,)`.

    rho : numpy.array
        Density values corresponding to `x`. Must be non-negative and define
        a valid probability density (i.e., integrate to 1 over the support).
        Shape `(n,)`.

    num_pts : int
        Number of sample points to generate from the density estimate.

    method : {``'mc'``, ``'qmc'``}, default= ``'qmc'``
        Method of drawing samples from uniform distribution:

        * ``'mc'``: Monte Carlo
        * ``'qmc'``: Quasi Monte Carlo

    seed : int, default=None
        Seed for random number generator

    Returns
    -------

    samples : numpy.array, shape (num_pts,)
        Samples drawn from the estimated density using a one-dimensional Halton
        sequence mapped through the estimated quantile function.

    See Also
    --------

    freealg.supp
    freealg.kde

    Notes
    -----

    The underlying Quasi-Monte Carlo engine uses ``scipy.stats.qmc.Halton``
    function for generating low-discrepancy points.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 8

        >>> import numpy
        >>> from freealg import sample

        >>> # density of Beta(3,1) on [0,1]
        >>> x = numpy.linspace(0, 1, 200)
        >>> rho = 3 * x**2

        >>> samples = sample(x, rho, num_pts=1000, method='qmc')
        >>> assert samples.shape == (1000,)

        >>> # Empirical mean should be close to 3/4
        >>> numpy.allclose(samples.mean(), 0.75, atol=0.02)
    """

    rng = numpy.random.default_rng(seed)
    quantile = _quantile_func(x, rho)

    # Draw from uniform distribution
    if method == 'mc':
        u = rng.random(num_pts)

    elif method == 'qmc':
        try:
            engine = qmc.Halton(d=1, scramble=True, rng=rng)
        except TypeError:
            engine = qmc.Halton(d=1, scramble=True, seed=rng)
        u = engine.random(num_pts).ravel()

    else:
        raise NotImplementedError('"method" is invalid.')

    # Draw from distribution by mapping from inverse CDF
    samples = quantile(u)

    return samples.ravel()
