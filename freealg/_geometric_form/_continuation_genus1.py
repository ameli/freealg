# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy
import scipy
from ._elliptic_functions import ellipj

__all__ = ['mobius_X', 'mobius_z', 'mobius_lambda', 'legendre_Y',
           'fit_rational_xy', '_poly_eval', 'eval_rational_xy',
           'eval_rational_xy_select', 'generate_rational_xy']


# =============
# sqrt pos imag
# =============

def sqrt_pos_imag(z):
    """
    Square root on a branch cut with always positive imaginary part.
    """

    sq = numpy.sqrt(z)
    sq = numpy.where(sq.imag < 0, -sq, sq)

    return sq


# ========
# mobius X
# ========

def mobius_X(z, a1, b1, a2, b2):

    A = (a2 - b2) / (a2 - a1)
    return A * (z - a1) / (z - b2)


# ========
# mobius z
# ========

def mobius_z(X, a1, b1, a2, b2):

    A = (a2 - b2) / (a2 - a1)
    return (X * b2 - A * a1) / (X - A)


# =============
# mobius lambda
# =============

def mobius_lambda(a1, b1, a2, b2):

    num = (b1 - a1) * (b2 - a2)
    den = (a2 - a1) * (b2 - b1)

    return num / den


# ==========
# legendre Y
# ==========

def legendre_Y(X, lam):

    D = X * (1.0 - X) * (X - lam)
    Y = sqrt_pos_imag(D)

    flip = numpy.real(X) > 1.0
    Y = numpy.where(flip, -Y, Y)

    return Y


# ===============
# fit rational xy
# ===============

def fit_rational_xy(X, Y, F, deg_p0=12, deg_p1=12, deg_q=12, ridge_lambda=0.0,
                    weights=None):

    n = F.size

    max_deg = max(deg_p0, deg_p1, deg_q)
    xp = numpy.ones((n, max_deg + 1), dtype=complex)
    for k in range(1, max_deg + 1):
        xp[:, k] = xp[:, k - 1] * X

    A0 = xp[:, :deg_p0 + 1]
    A1 = (Y[:, None] * xp[:, :deg_p1 + 1])
    Aq = (-F[:, None] * xp[:, 1:deg_q + 1])

    A = numpy.hstack([A0, A1, Aq])
    bvec = F

    if weights is not None:
        w = numpy.sqrt(numpy.maximum(weights, 0.0))
        A = A * w[:, None]
        bvec = bvec * w

    s = numpy.linalg.norm(A, axis=0)
    s[s == 0] = 1.0
    As = A / s[None, :]

    if ridge_lambda is None:
        ridge_lambda = 0.0

    if ridge_lambda > 0.0:
        alpha = ridge_lambda
        n_coef = As.shape[1]
        A_aug = numpy.vstack(
            [As, numpy.sqrt(alpha) * numpy.eye(n_coef, dtype=complex)])
        b_aug = numpy.concatenate([bvec, numpy.zeros(n_coef, dtype=complex)])
        coef, _, _, _ = numpy.linalg.lstsq(A_aug, b_aug, rcond=None)
    else:
        coef, _, _, _ = numpy.linalg.lstsq(As, bvec, rcond=None)

    coef = coef / s

    i0 = deg_p0 + 1
    i1 = i0 + (deg_p1 + 1)

    p0 = coef[:i0]
    p1 = coef[i0:i1]

    q = numpy.zeros(deg_q + 1, dtype=complex)
    q[0] = 1.0
    q[1:] = coef[i1:]

    return p0, p1, q


# =========
# poly eval
# =========

def _poly_eval(x, c):

    y = numpy.zeros_like(x, dtype=complex)
    for k in range(len(c) - 1, -1, -1):
        y = y * x + c[k]

    return y


# ================
# eval rational xy
# ================

def eval_rational_xy(X, Y, p0, p1, q, alt_branch):
    num0 = _poly_eval(X, p0)
    num1 = _poly_eval(X, p1)
    den = _poly_eval(X, q)

    m_plus = (num0 + Y * num1) / den
    m_minus = (num0 - Y * num1) / den

    if alt_branch:
        return m_minus
    return m_plus


# =======================
# eval rational xy select
# =======================

def eval_rational_xy_select(z, X, Y, p0, p1, q, alt_branch):

    m_plus = eval_rational_xy(X, Y, p0, p1, q, alt_branch=False)
    m_minus = eval_rational_xy(X, Y, p0, p1, q, alt_branch=True)

    mask_p = numpy.imag(z) >= 0.0

    pick_plus_p = (numpy.imag(m_plus) >= 0.0)
    pick_plus_m = (numpy.imag(m_plus) <= 0.0)

    pick_plus = numpy.where(mask_p, pick_plus_p, pick_plus_m)

    m1 = numpy.where(pick_plus, m_plus, m_minus)
    m2 = numpy.where(pick_plus, m_minus, m_plus)

    if alt_branch:
        return m2
    return m1


# ====================
# generate rational xy
# ====================

def generate_rational_xy(m1_fn, a1, b1, a2, b2, deg_p0=12, deg_p1=12, deg_q=12,
                         n_samples=4096, r=0.25, n_r=1, r_min=None,
                         ridge_lambda=0.0):

    if n_samples % 2 != 0:
        raise ValueError('n_samples should be even.')

    if n_r is None or n_r < 1:
        n_r = 1

    lam = mobius_lambda(a1, b1, a2, b2)

    r = float(r)
    if r > 0.6:
        r = 0.30

    if r_min is None:
        r_min = 0.25 * r

    if n_r == 1:
        vs = numpy.array([r], dtype=float)
    else:
        vs = numpy.linspace(r_min, r, n_r)

    K = float(scipy.special.ellipk(lam))
    Kp = float(scipy.special.ellipk(1.0 - lam))

    omega1 = 2.0 * K
    shift2 = 1j * Kp

    n_half = n_samples // 2
    t = (numpy.arange(n_half) + 0.5) / n_half
    u_re = omega1 * t

    X_list = []
    Y_list = []
    F_list = []

    for v in vs:
        for sh in (0.0, shift2):

            u = u_re + sh + 1j * v

            sn, cn, dn, _ = ellipj(u, lam)

            X = lam * (sn * sn)

            Y = legendre_Y(X, lam)
            Y_ref = 1j * lam * (sn * cn * dn)
            if sh != 0.0:
                Y_ref = -Y_ref

            flip = numpy.real(Y * numpy.conjugate(Y_ref)) < 0.0
            Y = numpy.where(flip, -Y, Y)

            z = mobius_z(X, a1, b1, a2, b2)
            f = m1_fn(z)

            X_list.append(X)
            Y_list.append(Y)
            F_list.append(f)

            X_list.append(numpy.conjugate(X))
            Y_list.append(numpy.conjugate(Y))
            F_list.append(numpy.conjugate(f))

    X_all = numpy.concatenate(X_list)
    Y_all = numpy.concatenate(Y_list)
    f_all = numpy.concatenate(F_list)

    p0, p1, q = fit_rational_xy(X_all, Y_all, f_all, deg_p0=deg_p0,
                                deg_p1=deg_p1, deg_q=deg_q,
                                ridge_lambda=ridge_lambda)

    return p0, p1, q, lam
