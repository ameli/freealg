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

__all__ = ['partial_sum', 'wynn_epsilon', 'wynn_rho', 'levin_u',
           'weniger_delta', 'brezinski_theta']


# ===========
# partial sum
# ===========

def partial_sum(coeffs, x, p=0.0):
    """
    Compute partial sum:

    .. math::

        S_n(x) = \\sum_{n=0}^{N-1} coeffs[n] * x^{n+p}.

    Parameters
    ----------

    coeffs : array_like
        Coefficients [a_0, a_1, a_2, ..., a_{N-1}] of the power series of the
        size N.

    x : numpy.array
        A flattened array of the size d.

    d : float, default=0.0
        Offset power.

    Returns
    -------

    Sn : numpy.ndarray
        Partial sums of the size (N, d), where the n-th row is the n-th
        partial sum.
    """

    x_ = x.ravel()
    N = len(coeffs)
    d = x_.size

    # Forming partial sum via Horner method
    Sn = numpy.zeros((N, d), dtype=x.dtype)
    sum_ = numpy.zeros((d,), dtype=x.dtype)
    pow_x = numpy.ones((d,), dtype=x.dtype)

    if p == 1:
        pow_x *= x_
    elif p != 0:
        pow_x *= x_**p

    for n in range(N):
        sum_ += coeffs[n] * pow_x
        Sn[n, :] = sum_

        if n < N-1:
            pow_x *= x_

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

    # Epsilons of stage k-1 and k-2
    eps_prev = Sn.copy()   # row k-1
    eps_pprev = None       # row k-2

    tol = numpy.finfo(Sn.dtype).eps

    # Wynn's epsilon triangle table
    for k in range(1, N):
        Nk = N - k

        delta = eps_prev[1:Nk+1, :] - eps_prev[:Nk, :]
        small = numpy.abs(delta) <= \
            tol * numpy.maximum(1.0, numpy.abs(eps_prev[1:Nk+1, :]))

        # Reciprocal of delta
        rec_delta = numpy.empty_like(delta)
        rec_delta[small] = 0.0j
        rec_delta[~small] = 1.0 / delta[~small]

        # Current epsilon of row k
        eps_curr = rec_delta
        if k > 1:
            eps_curr += eps_pprev[1:Nk+1, :]

        # Roll rows
        eps_pprev = eps_prev
        eps_prev = eps_curr

    # Last even row
    if (N - 1) % 2 == 0:
        # N is odd, so use step k-1
        S = eps_prev[0, :]
    else:
        # N is even, so use k-2
        S = eps_pprev[0, :]

    return S


# ========
# wynn rho
# ========

def wynn_rho(Sn, beta=0.0):
    """
    Accelerate convergence of a series using Wynn's rho algorithm.

    Parameters
    ----------

    Sn : numpy.ndarray
        A 2D array of shape ``(N, d)``, where *N* is the number of partial
        sums and *d* is the vector size.

    beta : float, default=0.0
        Shift parameter in the rho recursion, usually chosen in the range
        ``0 < beta <= 1``.

    Returns
    -------

    S : numpy.ndarray
        A 1D array of shape ``(d,)`` giving the rho-accelerated estimate
        of the series limit for each component.

    Notes
    -----

    Let ``S_n`` be the *n*-th partial sum of the (possibly divergent)
    sequence.  Wynn's rho algorithm builds a triangular table
    ``rho[k, n]`` (row *k*, column *n*) as follows:

        rho[-1, n] = 0
        rho[ 0, n] = S_n

        rho[k, n] = rho[k-2, n+1] +
                    (n + beta + k - 1) / (rho[k-1, n+1] - rho[k-1, n])

    Only even rows (k even) provide improved approximants.  As with
    ``wynn_epsilon``, we apply the scalar recursion component-wise so that a
    slowly converging component does not stall the others.
    """

    # N: number of partial sums, d: vector size
    N, d = Sn.shape

    # Rho of stage k-1 and k-2
    rho_prev = Sn.copy()   # row k-1
    rho_pprev = None       # row k-2

    tol = numpy.finfo(Sn.dtype).eps

    # Wynn's rho triangule table
    for k in range(1, N):
        Nk = N - k

        delta = rho_prev[1:Nk+1, :] - rho_prev[:Nk, :]
        small = numpy.abs(delta) <= \
            tol * numpy.maximum(1.0, numpy.abs(rho_prev[1:Nk+1, :]))

        coef = (beta + (k - 1) + numpy.arange(Nk))[:, None]   # (Nk, 1)
        coef = numpy.repeat(coef, d, axis=1)                  # (Nk, d)

        # Current rho of row k
        rho_curr = numpy.empty_like(delta)
        rho_curr[small] = 0.0j          # treat near-zero denominator

        if k == 1:
            rho_curr[~small] = coef[~small] / delta[~small]
        else:
            rho_curr[~small] = rho_pprev[1:Nk+1][~small] + \
                coef[~small] / delta[~small]

        # Roll rows
        rho_pprev = rho_prev
        rho_prev = rho_curr

    # Last even row
    if (N - 1) % 2 == 0:
        # N is odd, so use step k-1
        S = rho_prev[0, :]
    else:
        # N is even, so use k-2
        S = rho_pprev[0, :]

    return S


# ========
# levin  u
# ========

def levin_u(Sn, omega=None, beta=0.0):
    """
    Levin u-transform (vector form).

    Parameters
    ----------
    Sn : ndarray, shape (N, d)
        First N partial sums of a vector series.
    omega : None or ndarray, shape (N-1, d), optional
        Remainder estimate.  If None, uses omega_n = S_{n+1} - S_n.
    beta : float, optional
        Levin shift parameter (default 0.0).

    Returns
    -------
    S : ndarray, shape (d,)
        Accelerated sum / antilimit.
    """

    Sn = numpy.asarray(Sn)
    N, d = Sn.shape
    if N < 3:
        raise ValueError("Need at least 3 partial sums for Levin u.")

    # default omega_n  = forward difference
    if omega is None:
        omega = Sn[1:, :] - Sn[:-1, :]
    else:
        omega = numpy.asarray(omega)
        if omega.shape != (N - 1, d):
            raise ValueError("omega must have shape (N-1, d).")

    tol = numpy.finfo(Sn.dtype).eps
    m = N - 2                       # highest possible order

    # binomial coefficients with alternating sign
    Cmk = numpy.empty(m + 1, dtype=Sn.dtype)
    Cmk[0] = 1.0
    for k in range(1, m + 1):
        Cmk[k] = Cmk[k - 1] * (m - k + 1) / k
    Cmk *= (-1.0) ** numpy.arange(m + 1)

    # powers (k + beta)^(m-1)
    if m == 1:
        Pk = numpy.ones(m + 1, dtype=Sn.dtype)
    else:
        Pk = (numpy.arange(m + 1, dtype=Sn.dtype) + beta) ** (m - 1)

    numer = numpy.zeros(d, dtype=Sn.dtype)
    denom = numpy.zeros(d, dtype=Sn.dtype)

    for k in range(m + 1):
        idx = k
        w = omega[idx, :]

        inv_w = numpy.empty_like(w)
        mask = numpy.abs(w) < tol
        inv_w[mask] = 0.0
        inv_w[~mask] = 1.0 / w[~mask]

        coeff = Cmk[k] * Pk[k]
        numer += coeff * Sn[idx, :] * inv_w
        denom += coeff * inv_w

    return numer / denom


# =============
# weniger delta
# =============

def weniger_delta(Sn):
    """
    Weniger's nonlinear delta^2 sequence transformation.

    Parameters
    ----------

    Sn : numpy.ndarray
        Array of shape (N, d) containing the first N partial sums of the
        series.

    Returns
    -------

    S : numpy.ndarray
        Array of shape (d,) giving the delta2 accelerated limit estimate for
        each component.
    """

    N, d = Sn.shape

    # Need at least three partial sums to form delta2
    if N < 3:
        return Sn[-1, :].copy()

    # First and second forward differences
    delta1 = Sn[1:] - Sn[:-1]              # shape (N-1, d)
    delta2 = delta1[1:] - delta1[:-1]      # shape (N-2, d)

    tol = numpy.finfo(Sn.real.dtype).eps

    # Safe reciprocal of delta2
    small = numpy.abs(delta2) <= tol * numpy.maximum(
        1.0, numpy.abs(delta1[:-1]))

    rec_delta2 = numpy.empty_like(delta2)
    rec_delta2[small] = 0.0j
    rec_delta2[~small] = 1.0 / delta2[~small]

    # Delta sequence, length N-2
    delta_sq = Sn[:-2] - (delta1[:-1] ** 2) * rec_delta2

    # Return the last Delta2 term as the accelerated estimate
    S = delta_sq[-1, :]

    return S


# ===============
# brezinski theta
# ===============

def brezinski_theta(Sn):
    """
    Accelerate convergence of a series using Brezinski's theta algorithm.

    Parameters
    ----------

    Sn : numpy.ndarray
        A 2-D array of the size ``(N, d)``, where `N` is the number of partial
        sums and `d` is the vector size.

    Returns
    -------

    S : numpy.ndarray
        A 1-D array of the size ``(d,)``. The theta-accelerated estimate of
        the series limit in each vector component.
    """

    N, d = Sn.shape

    theta_prev = Sn.copy()                        # row k-1
    theta_pprev = numpy.zeros_like(theta_prev)    # row k-2

    tol = numpy.finfo(Sn.dtype).eps

    for k in range(1, N):
        L_prev = theta_prev.shape[0]              # current row length

        if k % 2 == 1:

            # Odd row 2m+1
            if L_prev < 2:
                break

            delta = theta_prev[1:] - theta_prev[:-1]      # len = L
            theta_pp = theta_pprev[1:L_prev]              # len = L

            small = numpy.abs(delta) <= \
                tol * numpy.maximum(1.0, numpy.abs(theta_prev[1:]))

            theta_curr = numpy.empty_like(delta)
            theta_curr[small] = 0.0j
            theta_curr[~small] = theta_pp[~small] + 1.0 / delta[~small]

        else:

            # Even row 2m+2
            if L_prev < 3:
                break

            delta_even = theta_pprev[2:L_prev] - theta_pprev[1:L_prev-1]
            delta_odd = theta_prev[1:L_prev-1] - theta_prev[:L_prev-2]
            delta2_odd = (theta_prev[2:L_prev] - 2.0 * theta_prev[1:L_prev-1]
                          + theta_prev[:L_prev-2])

            small = numpy.abs(delta2_odd) <= tol * numpy.maximum(
                1.0, numpy.abs(theta_prev[1:L_prev-1]))

            theta_curr = numpy.empty_like(delta_odd)
            theta_curr[small] = theta_pprev[1:L_prev-1][small]
            theta_curr[~small] = (
                theta_pprev[1:L_prev-1][~small] +
                (delta_even[~small] * delta_odd[~small]) /
                delta2_odd[~small])

        # roll rows
        theta_pprev = theta_prev
        theta_prev = theta_curr

    if (N - 1) % 2 == 0:
        S = theta_prev[0]
    else:
        S = theta_pprev[0]

    return S
