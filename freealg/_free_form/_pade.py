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
from numpy.linalg import lstsq
from scipy.optimize import least_squares, differential_evolution

__all__ = ['fit_pade', 'eval_pade']


# =============
# default poles
# =============

def _default_poles(q, lam_m, lam_p, safety=1.0, odd_side='left'):
    """
    Generate q real poles outside [lam_m, lam_p].

    * even q  : q/2 on each side (Chebyshev-like layout)
    * odd  q  : (q+1)/2 on the *left*,  (q-1)/2 on the right
                so q=1 => single pole on whichever side `odd_side` says.

    safety >= 1: 1, then poles start half an interval away; >1 pushes them
    farther.
    """

    if q == 0:
        return numpy.empty(0)

    Delta = 0.5 * (lam_p - lam_m)

    # Decide how many poles on each side. m_L and m_R determine how many poles
    # to be on the left and right of the support interval.
    if q % 2 == 0:
        m_L = m_R = q // 2
    else:
        if odd_side == 'left':
            m_L = (q + 1) // 2
            m_R = q // 2
        else:
            m_L = q // 2
            m_R = (q + 1) // 2

    # Chebyshev-extrema offsets  (all positive)
    kL = numpy.arange(m_L)
    tL = (2 * kL + 1) * numpy.pi / (2 * m_L)
    offsL = safety * Delta * (1 + numpy.cos(tL))

    kR = numpy.arange(m_R)
    tR = (2 * kR + 1) * numpy.pi / (2 * m_R + (m_R == 0))
    offsR = safety * Delta * (1 + numpy.cos(tR))

    left = lam_m - offsL
    right = lam_p + offsR

    return numpy.sort(numpy.concatenate([left, right]))


# ============
# encode poles
# ============

def _encode_poles(a, lam_m, lam_p):
    """
    Map real pole a_j => unconstrained s_j,
    so that the default left-of-interval pole stays left.
    """

    # half-width of the interval
    d = 0.5 * (lam_p - lam_m)
    # if a < lam_m, we want s >= 0; if a > lam_p, s < 0
    return numpy.where(
        a < lam_m,
        numpy.log((lam_m - a) / d),   # zero at a = lam_m - d
        -numpy.log((a - lam_p) / d)   # zero at a = lam_p + d
    )


# ============
# decode poles
# ============

def _decode_poles(s, lam_m, lam_p):
    """
    Inverse map s_j => real pole a_j outside the interval.
    """

    d = 0.5 * (lam_p - lam_m)
    return numpy.where(
        s >= 0,
        lam_m - d * numpy.exp(s),     # maps s=0 to a=lam_m-d (left)
        lam_p + d * numpy.exp(-s)     # maps s=0 to a=lam_p+d (right)
    )


# ========
# inner ls
# ========

def _inner_ls(x, f, poles, dpq=1, pade_reg=0.0):
    """
    This is the inner least square (blazing fast).

    dqp is the difference between the order of P (numerator) and Q
    (denominator).
    """

    if poles.size == 0 and dpq == -1:
        return 0.0, 0.0, numpy.empty(0)

    if poles.size == 0:                      # q = 0
        # A = numpy.column_stack((numpy.ones_like(x), x))
        cols = [numpy.ones_like(x)] if dpq >= 0 else []
        if dpq == 1:
            cols.append(x)
        A = numpy.column_stack(cols)
        # ---
        theta, *_ = lstsq(A, f, rcond=None)
        # c, D = theta  # TEST
        if dpq == -1:
            c = 0.0
            D = 0.0
            resid = numpy.empty(0)
        elif dpq == 0:
            c = theta[0]
            D = 0.0
            resid = numpy.empty(0)
        else:  # dpq == 1
            c, D = theta
            resid = numpy.empty(0)
    else:
        # phi = 1.0 / (x[:, None] - poles[None, :])
        # # A = numpy.column_stack((numpy.ones_like(x), x, phi))  # TEST
        # # theta, *_ = lstsq(A, f, rcond=None)
        # # c, D, resid = theta[0], theta[1], theta[2:]
        # phi = 1.0 / (x[:, None] - poles[None, :])
        # cols = [numpy.ones_like(x)] if dpq >= 0 else []
        # if dpq == 1:
        #     cols.append(x)
        #     cols.append(phi)
        #     A = numpy.column_stack(cols)
        #     theta, *_ = lstsq(A, f, rcond=None)
        # if dpq == -1:
        #     c = 0.0
        #     D = 0.0
        #     resid = theta
        # elif dpq == 0:
        #     c = theta[0]
        #     D = 0.0
        #     resid = theta[1:]
        # else:  # dpq == 1
        #     c = theta[0]
        #     D = theta[1]
        #     resid = theta[2:]

        phi = 1.0 / (x[:, None] - poles[None, :])
        cols = [numpy.ones_like(x)] if dpq >= 0 else []
        if dpq == 1:
            cols.append(x)
        cols.append(phi)

        A = numpy.column_stack(cols)

        # theta, *_ = lstsq(A, f, rcond=None) # TEST
        if pade_reg > 0:
            ATA = A.T.dot(A)

            # # add pade_reg * I
            # ATA.flat[:: ATA.shape[1]+1] += pade_reg
            # ATf = A.T.dot(f)
            # theta = numpy.linalg.solve(ATA, ATf)

            # figure out how many elements to skip
            if dpq == 1:
                skip = 2     # skip c and D
            elif dpq == 0:
                skip = 1     # skip c only
            else:
                skip = 0     # all entries are residues

            # add lambda only for the residue positions
            n = ATA.shape[0]
            for i in range(skip, n):
                ATA[i, i] += pade_reg

            # then solve
            ATf = A.T.dot(f)
            theta = numpy.linalg.solve(ATA, ATf)

        else:
            theta, *_ = lstsq(A, f, rcond=None)

        if dpq == -1:
            c, D, resid = 0.0, 0.0, theta
        elif dpq == 0:
            c, D, resid = theta[0], 0.0, theta[1:]
        else:  # dpq == 1
            c, D, resid = theta[0], theta[1], theta[2:]

    return c, D, resid


# =============
# eval rational
# =============

def _eval_rational(z, c, D, poles, resid):
    """
    """

    # z = z[:, None]
    # if poles.size == 0:
    #     term = 0.0
    # else:
    #     term = numpy.sum(resid / (z - poles), axis=1)
    #
    # return c + D * z.ravel() + term

    # ensure z is a 1-D array
    z = numpy.asarray(z)
    z_col = z[:, None]

    if poles.size == 0:
        term = 0.0
    else:
        term = numpy.sum(resid / (z_col - poles[None, :]), axis=1)

    return c + D * z + term


# ========
# fit pade
# ========

def fit_pade(x, f, lam_m, lam_p, p=2, q=2, odd_side='left', pade_reg=0.0,
             safety=1.0, max_outer=40, xtol=1e-12, ftol=1e-12, optimizer='ls',
             verbose=0):
    """
    This is the outer optimiser.
    """

    # Checks
    if not (odd_side in ['left', 'right']):
        raise ValueError('"odd_side" can only be "left" or "right".')

    # Difference between the degrees of numerator and denominator
    dpq = p - q
    if not (dpq in [-1, 0, 1]):
        raise ValueError('"pade_p" and "pade_q" can only differ by "+1", ' +
                         '"0", or "-1".')

    x = numpy.asarray(x, float)
    f = numpy.asarray(f, float)

    poles0 = _default_poles(q, lam_m, lam_p, safety=safety, odd_side=odd_side)
    if q == 0 and dpq <= 0:
        # c, D, resid = _inner_ls(x, f, poles0, pade_reg=pade_reg)  # TEST
        c, D, resid = _inner_ls(x, f, poles0, dpq, pade_reg=pade_reg)
        pade_sol = {
            'c': c, 'D': D, 'poles': poles0, 'resid': resid,
            'outer_iters': 0
        }

        return pade_sol

    s0 = _encode_poles(poles0, lam_m, lam_p)

    # --------
    # residual
    # --------

    def residual(s, dpq=dpq):
        poles = _decode_poles(s, lam_m, lam_p)
        # c, D, resid = _inner_ls(x, f, poles, pade_reg=pade_reg) # TEST
        c, D, resid = _inner_ls(x, f, poles, dpq, pade_reg=pade_reg)
        return _eval_rational(x, c, D, poles, resid) - f

    # ----------------

    # Optimizer
    if optimizer == 'ls':
        # scale = numpy.maximum(1.0, numpy.abs(s0))
        res = least_squares(residual, s0,
                            method='trf',
                            # method='lm',
                            # x_scale=scale,
                            max_nfev=max_outer, xtol=xtol, ftol=ftol,
                            verbose=verbose)

    elif optimizer == 'de':

        # Bounds
        # span = lam_p - lam_m
        # B = 3.0  # multiples of span
        # L = numpy.log(B * span)
        # bounds = [(-L, L)] * len(s0)

        d = 0.5*(lam_p - lam_m)
        # the minimum factor so that lam_m - d*exp(s)=0 is exp(s)=lam_m/d
        min_factor = lam_m/d
        B = max(10.0, min_factor*10.0)
        L = numpy.log(B)
        bounds = [(-L, L)] * len(s0)

        # Global stage
        glob = differential_evolution(lambda s: numpy.sum(residual(s)**2),
                                      bounds, maxiter=50, popsize=10,
                                      polish=False)

        # local polish
        res = least_squares(
                residual, glob.x,
                method='lm',
                max_nfev=max_outer, xtol=xtol, ftol=ftol,
                verbose=verbose)

    else:
        raise RuntimeError('"optimizer" is invalid.')

    poles = _decode_poles(res.x, lam_m, lam_p)
    # c, D, resid = _inner_ls(x, f, poles, pade_reg=pade_reg) # TEST
    c, D, resid = _inner_ls(x, f, poles, dpq, pade_reg=pade_reg)

    pade_sol = {
        'c': c, 'D': D, 'poles': poles, 'resid': resid,
        'outer_iters': res.nfev
    }

    return pade_sol


# =========
# eval pade
# =========

def eval_pade(z, pade_sol):
    """
    """

    # z_arr = numpy.asanyarray(z)                   # shape=(M,N)
    # flat = z_arr.ravel()                          # shape=(M*N,)
    # c, D = pade_sol['c'], pade_sol['D']
    # poles = pade_sol['poles']
    # resid = pade_sol['resid']
    #
    # # _eval_rational takes a 1-D array of z's and returns 1-D outputs
    # flat_out = _eval_rational(flat, c, D, poles, resid)
    #
    # # restore the original shape
    # out = flat_out.reshape(z_arr.shape)         # shape=(M,N)
    #
    # return out

    z = numpy.asanyarray(z)      # complex or real, any shape
    c, D = pade_sol['c'], pade_sol['D']
    poles, resid = pade_sol['poles'], pade_sol['resid']

    out = c + D*z
    for bj, rj in zip(poles, resid):
        out += rj/(z - bj)       # each is an (N,) op, no N*q temp
    return out
