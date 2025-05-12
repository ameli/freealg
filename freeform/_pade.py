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
from itertools import product
from scipy.optimize import least_squares, differential_evolution

__all__ = ['fit_pade', 'eval_pade']


# ========
# fit pade
# ========

def fit_pade(x, f, lam_m, lam_p, p, q, delta=1e-8, B=numpy.inf, S=numpy.inf,
             B_default=10.0, S_factor=2.0, maxiter_de=200):
    """
    Fit a [p/q] rational P/Q of the form:
      P(x) = s * prod_{i=0..p-1}(x - a_i)
      Q(x) = prod_{j=0..q-1}(x - b_j)

    Constraints:
      a_i ∈ [lam_m, lam_p]
      b_j ∈ (-infty, lam_m - delta] cup [lam_p + delta, infty)

    Approach:
      - Brute‐force all 2^q left/right assignments for denominator roots
      - Global search with differential_evolution, fallback to zeros if needed
      - Local refinement with least_squares

    Returns a dict with keys:
      's'     : optimal scale factor
      'a'     : array of p numerator roots (in [lam_m, lam_p])
      'b'     : array of q denominator roots (outside the interval)
      'resid' : final residual norm
      'signs' : tuple indicating left/right pattern for each b_j
    """

    # Determine finite bounds for DE
    if not numpy.isfinite(B):
        B_eff = B_default
    else:
        B_eff = B
    if not numpy.isfinite(S):
        # scale bound: S_factor * max|f| * interval width + safety
        S_eff = S_factor * numpy.max(numpy.abs(f)) * (lam_p - lam_m) + 1.0
        if S_eff <= 0:
            S_eff = 1.0
    else:
        S_eff = S

    def map_roots(signs, b):
        """Map unconstrained b_j -> real root outside the interval."""
        out = numpy.empty_like(b)
        for j, (s_val, bj) in enumerate(zip(signs, b)):
            if s_val > 0:
                out[j] = lam_p + delta + numpy.exp(bj)
            else:
                out[j] = lam_m - delta - numpy.exp(bj)
        return out

    best = {'resid': numpy.inf}

    # Enumerate all left/right sign patterns
    for signs in product([-1, 1], repeat=q):
        # Residual vector for current pattern
        def resid_vec(z):
            s_val = z[0]
            a = z[1:1+p]
            b = z[1+p:]
            P = s_val * numpy.prod(x[:, None] - a[None, :], axis=1)
            roots_Q = map_roots(signs, b)
            Q = numpy.prod(x[:, None] - roots_Q[None, :], axis=1)
            return P - f * Q

        def obj(z):
            r = resid_vec(z)
            return r.dot(r)

        # Build bounds for DE
        bounds = []
        bounds.append((-S_eff, S_eff))      # s
        bounds += [(lam_m, lam_p)] * p      # a_i
        bounds += [(-B_eff, B_eff)] * q     # b_j

        # 1) Global search
        try:
            de = differential_evolution(obj, bounds,
                                        maxiter=maxiter_de,
                                        polish=False)
            z0 = de.x
        except ValueError:
            # fallback: start at zeros
            z0 = numpy.zeros(1 + p + q)

        # 2) Local refinement
        ls = least_squares(resid_vec, z0, xtol=1e-12, ftol=1e-12)

        rnorm = numpy.linalg.norm(resid_vec(ls.x))
        if rnorm < best['resid']:
            best.update(resid=rnorm, signs=signs, x=ls.x.copy())

    # Unpack best solution
    z_best = best['x']
    s_opt = z_best[0]
    a_opt = z_best[1:1+p]
    b_opt = map_roots(best['signs'], z_best[1+p:])

    return {
        's':     s_opt,
        'a':     a_opt,
        'b':     b_opt,
        'resid': best['resid'],
        'signs': best['signs'],
    }


# =========
# eval pade
# =========

def eval_pade(z, s, a, b):
    """
    """

    Pz = s * numpy.prod([z - aj for aj in a], axis=0)
    Qz = numpy.prod([z - bj for bj in b], axis=0)

    return Pz / Qz
