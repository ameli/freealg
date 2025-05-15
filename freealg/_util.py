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
import scipy
from scipy.stats import beta
from scipy.optimize import minimize

__all__ = ['compute_eig', 'beta_kde', 'force_density']


# ===========
# compute eig
# ===========

def compute_eig(A, lower=False):
    """
    Compute eigenvalues of symmetric matrix.
    """

    eig = scipy.linalg.eigvalsh(A, lower=lower, driver='ev')

    return eig


# ========
# beta kde
# ========

def beta_kde(eig, xs, lam_m, lam_p, h):
    """
    Beta-kernel KDE with automatic guards against NaNs.

    Parameters
    ----------
    eig    : (n,) 1-D array of samples
    xs     : evaluation grid (must lie within [lam_m, lam_p])
    lam_m, lam_p : float, support endpoints  (lam_m < lam_p)
    h      : bandwidth in rescaled units (0 < h < 1)

    Returns
    -------
    pdf    : ndarray  same length as xs
    """

    span = lam_p - lam_m
    if span <= 0:
        raise ValueError("lam_p must be larger than lam_m")

    # map samples and grid to [0,1]
    u = (eig - lam_m) / span
    t = (xs - lam_m) / span

    if u.min() < 0 or u.max() > 1:
        mask = (u > 0) & (u < 1)
        u = u[mask]

    pdf = numpy.zeros_like(xs, dtype=float)
    n = len(u)

    # tiny positive number to keep shape parameters >0
    eps = 1e-6
    for ui in u:
        a = max(ui / h + 1.0, eps)
        b = max((1.0 - ui) / h + 1.0, eps)
        pdf += beta.pdf(t, a, b)

    pdf /= n * span                        # renormalise
    pdf[(t < 0) | (t > 1)] = 0.0           # exact zeros outside

    return pdf


# =============
# force density
# =============

def force_density(psi0, support, approx, grid, alpha=0.0, beta=0.0):
    """
    Starting from psi0 (raw projection), solve
      min  0.5 ||psi - psi0||^2
      s.t. F_pos psi >= 0           (positivity on grid)
           psi[0] = psi0[0]         (mass)
           f(lam_m)·psi = 0         (zero at left edge)
           f(lam_p)·psi = 0         (zero at right edge)
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
                        'fun': lambda psi: approx(grid, psi)})

    # Enforce unit mass
    constraints.append({
        'type': 'eq',
        'fun': lambda psi: numpy.trapz(approx(grid, psi), grid) - 1.0
    })

    # Enforce zero at left edge
    if beta <= 0.0 and beta > -0.5:
        constraints.append({
            'type': 'eq',
            'fun': lambda psi: approx(numpy.array([lam_m]), psi)[0]
        })

    # Enforce zero at right edge
    if alpha <= 0.0 and alpha > -0.5:
        constraints.append({
            'type': 'eq',
            'fun': lambda psi: approx(numpy.array([lam_p]), psi)[0]
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
    rho = approx(x, psi)
    mass = numpy.trapz(rho, x)
    psi[0] = psi[0] / mass

    return psi
