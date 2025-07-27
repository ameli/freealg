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

        #  \int P_k(t) w(t) \rho(t) dt. w(t) cancels with pdf already being rho
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

def jacobi_stieltjes(z, psi, support, alpha=0.0, beta=0.0, n_base=40,
                     continuation='pade', dtype=numpy.complex128):
    """
    Compute m(z) = sum_k psi_k * m_k(z) where

    m_k(z) = \\int w^{(alpha, beta)}(t) P_k^{(alpha, beta)}(t) / (u(z)-t) dt

    Each m_k is evaluated *separately* with a Gauss-Jacobi rule sized
    for that k.  This follows the user's request: 1 quadrature rule per P_k.

    Parameters
    ----------

    z : complex or ndarray

    psi : (K+1,) array_like

    support : (lambda_minus, lambda_plus)

    alpha, beta : float

    n_base : int
        Minimum quadrature size.  For degree-k polynomial we use
        n_quad = max(n_base, k+1).

    continuation : str, default= ``'pade'``
        Methof of analytiv continuation.

    dtype : numpy.type, default=numpy.complex128
        Data type for compelx arrays. This might enhance series acceleration.

    Returns
    -------

    m1 : ndarray
        Same shape as z

    m2 : ndarray
        Same shape as z
    """

    z = numpy.asarray(z, dtype=dtype)
    lam_minus, lam_plus = support
    span = lam_plus - lam_minus
    centre = 0.5 * (lam_plus + lam_minus)

    # Map z -> u in the standard [-1,1] domain
    u = (2.0 / span) * (z - centre)

    m_total = numpy.zeros_like(z, dtype=dtype)

    if continuation != 'pade':
        # Stores  m with the ravel size of z.
        m_partial = numpy.zeros((psi.size, z.size), dtype=dtype)

    for k, psi_k in enumerate(psi):
        # Select quadrature size tailored to this P_k
        n_quad = max(n_base, k + 1)
        t_nodes, w_nodes = roots_jacobi(n_quad, alpha, beta)  # (n_quad,)

        # Evaluate P_k at the quadrature nodes
        P_k_nodes = eval_jacobi(k, alpha, beta, t_nodes)     # (n_quad,)

        # Integrand values at nodes: w_nodes already include the weight
        integrand = w_nodes * P_k_nodes                      # (n_quad,)

        # Evaluate jacobi polynomals of the second kind, Q_k using quadrature
        diff = t_nodes[:, None, None] - u[None, ...]         # (n_quad, Ny, Nx)
        Q_k = (integrand[:, None, None] / diff).sum(axis=0).astype(dtype)

        # Principal branch
        m_k = (2.0 / span) * Q_k

        # Compute secondary branch from the principal branch
        if continuation != 'pade':

            # Compute analytic extension of rho(z) to lower-half plane for
            # when rho is just the k-th Jacobi basis: w(z) P_k(z). FOr this,
            # we create a psi array (called unit_psi_j), with all zeros, except
            # its k-th element is one. Ten we call jacobi_density.
            unit_psi_k = numpy.zeros_like(psi)
            unit_psi_k[k] = 1.0

            # Only lower-half plane
            mask_m = z.imag <= 0
            z_m = z[mask_m]

            # Dnesity here is rho = w(z) P_k
            rho_k = jacobi_density(z_m.ravel(), unit_psi_k, support,
                                   alpha=alpha, beta=beta).reshape(z_m.shape)

            # Secondary branch is principal branch + 2 \pi i rho, using Plemelj
            # (in fact, Riemann-Hirbert jump).
            m_k[mask_m] = m_k[mask_m] + 2.0 * numpy.pi * 1j * rho_k

        # Accumulate with factor 2/span
        m_total += psi_k * m_k

        if continuation != 'pade':
            m_partial[k, :] = m_total.ravel()

    if continuation != 'pade':

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

        m_total = S.reshape(z.shape)

    return m_total
