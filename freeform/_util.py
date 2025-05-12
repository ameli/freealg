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
from scipy.optimize import minimize

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
            'fun': lambda psi: approx(numpy.array([lam_m], psi))[0]
        })

    # Enforce zero at right edge
    if alpha <= 0.0 and alpha > -0.5:
        constraints.append({
            'type': 'eq',
            'fun': lambda psi: approx(numpy.array([lam_p], psi))[0]
        })

    # Solve a small quadratic programming
    res = minimize(fun, psi0, jac=grad,
                   constraints=constraints,
                   # method='trust-constr',
                   method='SLSQP',
                   options={'maxiter': 1000, 'ftol': 1e-9, 'eps': 1e-8})

    return res.x
