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
import scipy
from scipy.optimize import minimize, LinearConstraint

__all__ = ['compute_eig', 'force_density']


# ===========
# compute eig
# ===========

def compute_eig(A, lower=False):
    """
    Compute eigenvalues of symmetric matrix.
    """

    eig = scipy.linalg.eigvalsh(A, lower=lower, driver='ev')

    return eig


# =============
# force density
# =============

def force_density(psi0, support, alpha, beta, pos_grid, edge_tol=0):
    """
    Starting from psi0 (raw projection), solve
      min  0.5 ||psi - psi0||^2
      s.t. F_pos psi >= 0           (positivity on pos_grid)
           psi[0] = psi0[0]         (mass)
           f(lam_m)·psi = 0         (zero at left edge)
           f(lam_p)·psi = 0         (zero at right edge)
    """

    lam_m, lam_p = support
    span = lam_p - lam_m
    K = len(psi0) - 1

    # Build positivity matrix F_pos: F_pos[i,k] = f_k(x_i)
    eps = 1e-8
    xg = numpy.linspace(lam_m + eps, lam_p - eps, 300)
    tg = (2 * xg - (lam_m + lam_p)) / span
    wg = (1 - tg)**alpha * (1 + tg)**beta
    Pg = numpy.vstack([eval_jacobi(k, alpha, beta, tg) for k in range(K+1)]).T
    F_pos = (2.0 / span) * (wg[:, None] * Pg)    # shape (M, K+1)

    # build edge rows f(lambda)
    def f_row(x0):
        t0 = (2*x0 - (lam_m + lam_p)) / span
        w0 = (1 - t0)**alpha * (1 + t0)**beta
        P0 = numpy.array([eval_jacobi(k, alpha, beta, t0) for k in range(K+1)])
        return (2.0/span) * w0 * P0

    row_m = f_row(lam_m)
    row_p = f_row(lam_p)

    # Objective and gradient
    def fun(psi):
        return 0.5 * numpy.dot(psi-psi0, psi-psi0)

    def grad(psi):
        return psi - psi0

    # Constraints:
    constraints = []

    # Enforce positivity: F_pos psi >= 0
    constraints.append(LinearConstraint(F_pos, lb=0.0, ub=numpy.inf))

    # Enforce unit mass: psi[0] = psi0[0]
    constraints.append({'type': 'eq',
                        'fun': lambda psi: psi[0] - psi0[0],
                        'jac': lambda psi:
                            numpy.concatenate(([1.0], numpy.zeros(K)))})

    # Enforce zero at left edge
    if beta <= 0.0 and beta > -0.5:
        constraints.append({'type': 'eq',
                            'fun': lambda psi: numpy.dot(row_m, psi),
                            'jac': lambda psi: row_m})

    # Enforce zero at right edge
    if alpha <= 0.0 and alpha > -0.5:
        constraints.append({'type': 'eq',
                            'fun': lambda psi: numpy.dot(row_p, psi),
                            'jac': lambda psi: row_p})

    # Solve a small quadratic programming
    res = minimize(fun, psi0, jac=grad,
                   constraints=constraints,
                   method='SLSQP',
                   options={'maxiter': 200, })
    return res.x
