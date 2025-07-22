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

__all__ = ['partial_sum', 'wynn_epsilon']


# ===========
# partial sum
# ===========

def partial_sum(coeffs, x):
    """
    Compute partial sum:

    .. math::

        S_n(x) = \\sum_{n=0}^{N} coeffs[n] * x^n.

    Parameters
    ----------

    coeffs : array_like
        Coefficients [a0, a1, a2, ...] of the power series of the size N.

    x : numpy.array
        A flattened array of the size d.

    Returns
    -------

    Sn : numpy.ndarray
        Partial sums of the size (N, d), where the n-th row is the n-th
        partial sum.
    """

    xn = x.ravel()
    N = len(coeffs)
    d = xn.size

    # Forming partial sum via Horner method
    Sn = numpy.zeros((N, d), dtype=x.dtype)
    sum_ = numpy.zeros((d,), dtype=x.dtype)
    pow_x = numpy.ones((d,), dtype=x.dtype)

    for n in range(N):
        sum_ += coeffs[n] * pow_x
        Sn[n, :] = sum_

        if n < N-1:
            pow_x *= xn

    return Sn


# ============
# wynn epsilon
# ============

def wynn_epsilon(Sn):
    """
    Accelerate conversion of a series using Wynn's epsilon algorithm.

    Parameters
    ----------

    Sn : numpy.ndarray
        A 2D array of the size (N, d), where N is the number of partial sums
        and d is the vector size.

    Returns
    -------

    S : numpy.array
        A 1D array of the size (d,) which is the accelerated value of the
        series at each vector element.

    Notes
    -----

    Given a series of vectors:

    .. math::

        (S_n)_{n=1}^N = (S1, \\dots, S_n)

    this function finds the limit S = \\lim_{n \\to infty} S_n.

    Each :math:`S_i \\in \\mathbb{C}^d` is a vector. However, instead of using
    the vector version of the Wynn's epsilon algorithm, we use the scalar
    version on each component of the vector. The reason for this is that in our
    dataset, each component has its own convergence rate. The convergence rate
    of vector version of the algorithm is bounded by the worse point, and this
    potentially stall convergence for all points. As such, vector version is
    avoided.

    In our dataset, the series is indeed divergent. The Wynn's accelerated
    method computes the principal value of the convergence series.
    """

    # N: number of partial sums, d: vector size
    N, d = Sn.shape

    eps = numpy.zeros((N, N, d), dtype=Sn.dtype)
    eps[0, :, :] = Sn

    tol = numpy.finfo(float).eps

    # Wynn's triangle table
    for k in range(1, N):
        Nk = N - k

        delta = eps[k-1, 1:N-k+1, :] - eps[k-1, :Nk, :]

        # Reciprocal of delta
        rec_delta = numpy.empty_like(delta)

        # Avoid division by zero error
        mask_inf = numpy.abs(delta) < tol
        rec_delta[mask_inf] = numpy.inf
        rec_delta[~mask_inf] = 1.0 / delta[~mask_inf]

        mask_zero = numpy.logical_or(numpy.isinf(delta),
                                     numpy.isnan(delta))
        rec_delta[mask_zero] = 0.0

        eps[k, :Nk, :] = rec_delta

        if k > 1:
            eps[k, :Nk, :] += eps[k-2, 1:Nk+1, :]

    k_even = 2 * ((N - 1) // 2)
    S = eps[k_even, 0, :]

    return S
