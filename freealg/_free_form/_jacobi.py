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
from scipy.special import eval_jacobi, roots_jacobi
from scipy.special import gammaln, beta as Beta
from ._series import wynn_epsilon, wynn_rho, levin_u, weniger_delta, \
    brezinski_theta

__all__ = ['jacobi_sample_proj', 'jacobi_kernel_proj', 'jacobi_density',
           'jacobi_stieltjes']


# ==============
# jacobi sq norm
# ==============

def jacobi_sq_norm(k, alpha, beta):
    """
    Norm of P_k
    Special-case k = 0 to avoid gamma(0) issues when alpha + beta + 1 = 0.
    """

    if k == 0:
        return 2.0**(alpha + beta + 1) * Beta(alpha + 1, beta + 1)

    # Use logs instead to avoid overflow in gamma function.
    lg_num = (alpha + beta + 1) * numpy.log(2.0) \
        + gammaln(k + alpha + 1) \
        + gammaln(k + beta + 1)

    lg_den = numpy.log(2*k + alpha + beta + 1) \
        + gammaln(k + 1) \
        + gammaln(k + alpha + beta + 1)

    return numpy.exp(lg_num - lg_den)


# ==================
# jacobi sample proj
# ==================

def jacobi_sample_proj(eig, support, K=10, alpha=0.0, beta=0.0, reg=0.0):
    """
    """

    lam_m, lam_p = support

    # Convert to [-1, 1] interval
    x = (2.0 * eig - (lam_p + lam_m)) / (lam_p - lam_m)

    psi = numpy.empty(K + 1)

    # Empirical moments and coefficients
    for k in range(K + 1):
        moment = numpy.mean(eval_jacobi(k, alpha, beta, x))
        N_k = jacobi_sq_norm(k, alpha, beta)  # normalization

        if k == 0:
            # Do not penalize at k=0, as this  keeps unit mass.
            # k=0 has unit mass, while k>0 has zero mass by orthogonality.
            penalty = 0
        else:
            penalty = reg * (k / (K + 1))**2

        # Add regularization on the diagonal
        psi[k] = moment / (N_k + penalty)

    return psi


# ==================
# jacobi kernel proj
# ==================

def jacobi_kernel_proj(xs, pdf, support, K=10, alpha=0.0, beta=0.0, reg=0.0):
    """
    Same moments as `jacobi_proj`, but the target is a *continuous* density
    given on a grid (xs, pdf).
    """

    lam_m, lam_p = support
    t = (2.0 * xs - (lam_p + lam_m)) / (lam_p - lam_m)      # map to [-1,1]
    psi = numpy.empty(K + 1)

    for k in range(K + 1):
        Pk = eval_jacobi(k, alpha, beta, t)
        N_k = jacobi_sq_norm(k, alpha, beta)

        # \int P_k(t) w(t) \rho(t) dt. w(t) cancels with pdf already being rho
        moment = numpy.trapz(Pk * pdf, xs)

        if k == 0:
            penalty = 0
        else:
            penalty = reg * (k / (K + 1))**2

        psi[k] = moment / (N_k + penalty)

    return psi


# ==============
# jacobi density
# ==============

def jacobi_density(x, psi, support, alpha=0.0, beta=0.0):
    """
    Reconstruct Jacobi approximation of density.

    Parameters
    ----------

    psi : array_like, shape (K+1, )
        Jacobi expansion coefficients.

    x : array_like
        Points (in original eigenvalue scale) to evaluate at.

    support : tuple (lam_m, lam_p)

    alpha : float
        Jacobi parameter.

    beta : float
        Jacobi parameter.

    Returns
    -------

    rho : ndarray
    """

    lam_m, lam_p = support
    t = (2 * x - (lam_p + lam_m)) / (lam_p - lam_m)
    w = (1 - t)**alpha * (1 + t)**beta

    # The function eval_jacobi does not accept complex256 type
    down_cast = False
    if numpy.issubdtype(t.dtype, numpy.complexfloating) and \
            t.itemsize > numpy.dtype(numpy.complex128).itemsize:
        t = t.astype(numpy.complex128)
        down_cast = True

    P = numpy.vstack([eval_jacobi(k, alpha, beta, t) for k in range(len(psi))])

    rho_t = w * (psi @ P)                            # density in t-variable
    rho_x = rho_t * (2.0 / (lam_p - lam_m))          # back to x-variable

    # Case up to complex256
    if down_cast:
        rho_x = rho_x.astype(t.dtype)

    return rho_x


# ================
# jacobi stieltjes
# ================

def jacobi_stieltjes(z, cache, psi, support, alpha=0.0, beta=0.0, n_quad=None,
                     continuation='pade', dtype=numpy.complex128):
    """
    Compute m(z) = sum_k psi_k * m_k(z) where

    .. math::

        m_k(z) = \\int \\frac{w^{(alpha, beta)}(t) P_k^{(alpha, beta)}(t)}{
        (u(z)-t)} \\mathrm{d} t

    Each m_k is evaluated *separately* with a Gauss-Jacobi rule sized
    for that k. This follows the user's request: 1 quadrature rule per P_k.

    Parameters
    ----------

    z : complex or ndarray

    cache : dict
        Pass a dict to enable cross-call caching.

    psi : (K+1,) array_like

    support : (lambda_minus, lambda_plus)

    alpha, beta : float

    n_quad : int, default=None
        Number of Gauss-Jacobi quadrature points.

    continuation : str, default= ``'pade'``
        Method of analytic continuation.

    dtype : numpy.type, default=numpy.complex128
        Data type for complex arrays. This might enhance series acceleration.

    Returns
    -------

    m1 : ndarray
        Same shape as z

    m2 : ndarray
        Same shape as z
    """

    if not isinstance(cache, dict):
        raise TypeError('"cache" must be a dict; pass a persistent dict '
                        '(e.g., self.cache).')

    # Number of quadratures
    if 'n_quad' not in cache:
        if n_quad is None:
            # Set number of quadratures based on Bernstein ellipse. Here using
            # an evaluation point a with distance delta from support, to
            # achieve the quadrature error below tol.
            tol = 1e-16
            delta = 1e-2
            n_quad = int(-numpy.log(tol) / (2.0 * numpy.sqrt(delta)))
        n_quad = max(n_quad, psi.size)
        cache['n_quad'] = n_quad
    else:
        n_quad = cache['n_quad']

    # Quadrature nodes and weights
    if ('t_nodes' not in cache) or ('w_nodes' not in cache):
        t_nodes, w_nodes = roots_jacobi(n_quad, alpha, beta)   # (n_quad,)
        cache['t_nodes'] = t_nodes
        cache['w_nodes'] = w_nodes
    else:
        t_nodes = cache['t_nodes']
        w_nodes = cache['w_nodes']

    z = numpy.asarray(z, dtype=dtype)
    lam_minus, lam_plus = support
    span = lam_plus - lam_minus
    centre = 0.5 * (lam_plus + lam_minus)

    # Map z to u in the standard [-1,1] domain
    u = (2.0 / span) * (z - centre)

    # Cauchy Kernel (flattened for all z)
    u_flat = u.ravel()
    ker = (1.0 / (t_nodes[:, None] - u_flat[None, :])).astype(
        dtype, copy=False)  # (n_quad, Ny*Nx)

    if continuation == 'pade':

        if 'integrand_nodes' not in cache:

            # Compute sum_k psi_k P_k (call it s_node)
            s_nodes = numpy.zeros_like(t_nodes, dtype=dtype)
            for k, psi_k in enumerate(psi):

                # Evaluate P_k at the quadrature nodes
                P_k_nodes = eval_jacobi(k, alpha, beta, t_nodes)   # (n_quad,)
                s_nodes += psi_k * P_k_nodes

            integrand_nodes = (2.0 / span) * (w_nodes * s_nodes).astype(dtype)
            cache['integrand_nodes'] = integrand_nodes

        else:
            integrand_nodes = cache['integrand_nodes']

        Q_flat = (integrand_nodes[:, None] * ker).sum(axis=0)
        m_total = Q_flat.reshape(z.shape)

        return m_total

    else:

        # Continuation is not Pade. This is one of Wynn, Levin, etc. These
        # methods need the series for m for 1, ..., k.

        if 'B' not in cache:
            # All P_k at quadrature nodes (real), row-scale by weights
            P_nodes = numpy.empty((psi.size, n_quad), dtype=w_nodes.dtype)
            for k in range(psi.size):
                P_nodes[k, :] = eval_jacobi(k, alpha, beta, t_nodes)

            # All P_k * w shape (K+1, n_quad)
            B = (2.0 / span) * (P_nodes * w_nodes[None, :]).astype(
                dtype, copy=False)
            cache['B'] = B

        else:
            B = cache['B']

        # Principal branch. 2D matrix for all k
        m_k_all = B @ ker

        # Compute m on secondary branch from the principal branch, which is
        # m_k = m_k + 2 \pi i rho_k(z), and rho(z) is the analytic extension of
        # rho_k(x) using the k-th basis. Basically, rho_k(z) is w * P_k(z).

        # Lower-half-plane jump for ALL k at once (vectorized)
        mask_m = (z.imag <= 0)
        if numpy.any(mask_m):
            idx = numpy.flatnonzero(mask_m.ravel())
            u_m = u_flat[idx].astype(dtype, copy=False)  # complex

            # Scipy's eval_jacobi tops out at complex128 type. If u_m is
            # complex256, downcast to complex128.
            if u_m.dtype.itemsize > numpy.dtype(numpy.complex128).itemsize:
                u_m_eval = u_m.astype(numpy.complex128, copy=False)
                down_cast = True
            else:
                u_m_eval = u_m
                down_cast = False

            # P_k at complex u_m (all means for all k = 1,...,K)
            P_all_m = numpy.empty((psi.size, u_m.size), dtype=dtype)
            for k in range(psi.size):
                P_all_m[k, :] = eval_jacobi(k, alpha, beta, u_m_eval)

            # Jacobi weight. Must match jacobi_density's branch
            w_m = numpy.power(1.0 - u_m, alpha) * numpy.power(1.0 + u_m, beta)

            # rho_k(z) in x-units is (2/span) * w(u) * P_k(u)
            rho_all = ((2.0 / span) * w_m[None, :] * P_all_m).astype(
                dtype, copy=False)

            if down_cast:
                rho_all = rho_all.astype(dtype)

            # compute analytic extension of rho(z) to lower-half plane for when
            # rho is just the k-th Jacobi basis: w(z) P_k(z). For this, we
            m_k_all[:, idx] = m_k_all[:, idx] + (2.0 * numpy.pi * 1j) * rho_all

        # Partial sums S_k = sum_{j<=k} psi_j * m_j
        WQ = (psi[:, None].astype(dtype, copy=False) * m_k_all)
        m_partial = numpy.cumsum(WQ, axis=0)

        if continuation == 'wynn-eps':
            S = wynn_epsilon(m_partial)
        elif continuation == 'wynn-rho':
            S = wynn_rho(m_partial)
        elif continuation == 'levin':
            S = levin_u(m_partial)
        elif continuation == 'weniger':
            S = weniger_delta(m_partial)
        elif continuation == 'brezinski':
            S = brezinski_theta(m_partial)
        else:
            # No acceleration (likely diverges in the lower-half plane)
            S = m_partial[-1, :]

        m_total = S.reshape(z.shape)
        return m_total
