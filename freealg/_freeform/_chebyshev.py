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
from scipy.special import eval_chebyu
from ._series import partial_sum, wynn_epsilon, wynn_rho, levin_u, \
    weniger_delta, brezinski_theta

__all__ = ['chebyshev_sample_proj', 'chebyshev_kernel_proj',
           'chebyshev_density', 'chebyshev_stieltjes']


# =====================
# chebyshev sample proj
# =====================

def chebyshev_sample_proj(eig, support, K=10, reg=0.0):
    """
    Estimate the coefficients \\psi_k in

        \\rho(x) = w(t) \\sum_{k=0}^K \\psi_k U_k(t),

    where t = (2x-(\\lambda_{-} + \\lambda_{+}))/ (\\lambda_{+} - \\lambda_{-})
    in [-1, 1] and w(t) = \\sqrt{(1 - t^2}.

    Parameters
    ----------

    eig : array_like, shape (N,)
        The raw eigenvalues x_i.

    support : tuple
        The assumed compact support of rho.

    K : int
        Highest Chebyshev-II order.

    reg : float
        Tikhonov-style ridge on each coefficient (defaults to 0).

    Returns
    -------

    psi : ndarray, shape (K+1,)
        The projected coefficients \\psi_k.
    """

    lam_m, lam_p = support

    # Map to [-1,1] interval
    t = (2 * eig - (lam_m + lam_p)) / (lam_p - lam_m)

    # Inner-product norm of each U_k under w(t) = sqrt{1-t^2} is \\pi/2
    norm = numpy.pi / 2

    psi = numpy.empty(K+1)
    for k in range(K+1):

        # empirical moment M_k = (1/N) \\sum U_k(t_i)
        M_k = numpy.mean(eval_chebyu(k, t))

        # Regularization
        if k == 0:
            # Do not penalize at k=0, as this  keeps unit mass.
            # k=0 has unit mass, while k>0 has zero mass by orthogonality.
            penalty = 0
        else:
            penalty = reg * (k / (K + 1))**2

        # Add regularization on the diagonal
        psi[k] = M_k / (norm + penalty)

    return psi


# =====================
# chebyshev kernel proj
# =====================

def chebyshev_kernel_proj(xs, pdf, support, K=10, reg=0.0):
    """
    Projection of a *continuous* density given on a grid (xs, pdf)
    onto the Chebyshev-II basis.

    xs  : 1-D numpy array (original x-axis, not the t-variable)
    pdf : same shape as xs, integrates to 1 on xs
    """

    lam_m, lam_p = support
    t = (2.0 * xs - (lam_m + lam_p)) / (lam_p - lam_m)   # map to [-1,1]

    norm = numpy.pi / 2.0
    psi = numpy.empty(K + 1)

    for k in range(K + 1):
        Pk = eval_chebyu(k, t)                       # U_k(t) on the grid
        moment = numpy.trapezoid(Pk * pdf, xs)       # \int U_k(t) \rho(x) dx

        if k == 0:
            penalty = 0
        else:
            penalty = reg * (k / (K + 1))**2

        psi[k] = moment / (norm + penalty)

    return psi


# =================
# chebyshev density
# =================

def chebyshev_density(x, psi, support):
    """
    Given \\psi_k, evaluate the approximate density \\rho(x).

    Parameters
    ----------

    x : array_like
        Points at which to evaluate \\rho.

    psi : array_like, shape (K+1,)
        Coefficients from chebyshev_proj.

    support : tuple
        Same support used for projection.

    Returns
    -------

    rho_x : ndarray, same shape as x
        Approximated spectral density on the original x-axis.
    """

    lam_m, lam_p = support

    # Map to [-1,1] interval
    t = (2 * numpy.asarray(x) - (lam_m + lam_p)) / (lam_p - lam_m)

    # Weight sqrt{1-t^2} (clip for numerical safety)
    w = numpy.sqrt(numpy.clip(1 - t**2, a_min=0, a_max=None))

    # Summation approximation
    U = numpy.vstack([eval_chebyu(k, t) for k in range(len(psi))]).T
    rho_t = w * (U @ psi)

    # Adjust for dt to dx transformation
    rho_x = rho_t * (2.0 / (lam_p - lam_m))

    return rho_x


# ===================
# chebushev stieltjes
# ===================

def chebyshev_stieltjes(z, psi, support, continuation='pade',
                        dtype=numpy.complex128):
    """
    Compute the Stieltjes transform m(z) for a Chebyshev-II expansion

    rho(x) = (2/(lam_p - lam_m)) * sqrt(1-t(x)^2) * sum_{k=0}^K psi_k U_k(t(x))

    via the closed-form

      \\int_{-1}^1 U_k(t) sqrt(1-t^2)/(u - t) dt = \\pi J(u)^(k+1),

    where

      u = (2(z-center))/span,
      center = (lam_p + lam_m)/2,
      span = lam_p - lam_m,
      J(u) = u - sqrt(u^2-1)

    and then

      m(z) = - (2/ span) * \\sum{k=0}^K \\psi_k * [ \\pi J(u)^(k+1) ].

    Parameters
    ----------

    z : complex or array_like of complex
        Points in the complex plane.

    psi : array_like, shape (K+1,)
        Chebyshev-II coefficients \\psi.

    support : tuple
        The support interval of the original density.

    continuation : str, default= ``'pade'``
        Methof of analytiv continuation.

    dtype : numpy.type, default=numpy.complex128
        Data type for complex arrays. This might enhance series acceleration.

    Returns
    -------

    m_z : ndarray of complex
        The Stieltjes transform m(z) on the same shape as z.
    """

    z = numpy.asarray(z, dtype=dtype)

    lam_m, lam_p = support
    span = lam_p - lam_m
    center = 0.5 * (lam_m + lam_p)

    # Map z to u in the standard [-1,1] domain
    u = (2.0 * (z - center)) / span

    # Inverse-Joukowski: pick branch sqrt with +Im
    root = numpy.sqrt(u*u - 1.0)
    Jm = u - root
    Jp = u + root

    # Make sure J is Herglotz
    J = numpy.zeros_like(Jp)
    J = numpy.where(Jp.imag > 0, Jm, Jp)

    # This depends on the method of analytic continuation
    if continuation == 'pade':
        # Horner summation for S0(J) = sum_{k=0}^K psi_k * J**k
        K = len(psi) - 1
        S0 = numpy.zeros_like(J)
        for k in range(K, -1, -1):
            S0 = psi[k] + J * S0
        S = J * S0

    else:
        # Flatten J before passing to any of the acceleration methods.
        psi_zero = numpy.concatenate([[0.0], psi])
        Sn = partial_sum(psi_zero, J.ravel(), p=0)

        if continuation == 'wynn-eps':
            S = wynn_epsilon(Sn)
        elif continuation == 'wynn-rho':
            S = wynn_rho(Sn)
        elif continuation == 'levin':
            S = levin_u(Sn)
        elif continuation == 'weniger':
            S = weniger_delta(Sn)
        elif continuation == 'brezinski':
            S = brezinski_theta(Sn)
        else:
            raise NotImplementedError('"continuation" is invalid.')

        S = S.reshape(J.shape)

    # Assemble m(z)
    m_z = -(2.0 / span) * numpy.pi * S

    # Check nan or inf
    if numpy.any(numpy.isinf(m_z)):
        raise RuntimeError('"m" is nan.')
    elif numpy.any(numpy.isnan(numpy.abs(m_z))):
        raise RuntimeError('"m" is inf.')

    return m_z
