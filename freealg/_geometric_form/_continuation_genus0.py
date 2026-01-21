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

__all__ = ['joukowski_z', 'joukowski_w', 'fit_pade', 'eval_pade',
           'generate_pade']


# ===========
# joukowski z
# ===========

def joukowski_z(w, a, b):

    c = 0.5 * (a + b)
    d = 0.5 * (b - a)
    z = c + 0.5 * d * (w + 1.0 / w)

    return z


# ===========
# joukowski w
# ===========

def joukowski_w(z, a, b):

    c = 0.5 * (a + b)
    d = 0.5 * (b - a)
    xi = (z - c) / d
    s = numpy.sqrt(xi * xi - 1.0)

    # Stabilize sqrt branch: make s have same sign as xi (helps continuity)
    s = numpy.where(numpy.real(xi) < 0.0, -s, s)

    w1 = xi + s
    w2 = xi - s
    w = numpy.where(numpy.abs(w1) >= 1.0, w1, w2)

    return w


# ========
# fit pade
# ========

def fit_pade(w, m1, deg_p=12, deg_q=12, ridge_lambda=0.0):
    """
    Fit m1 on w-plane using Pade rational approximation.
    """

    n_samples = m1.size

    wp = numpy.ones((n_samples, deg_p + 1), dtype=complex)
    for k in range(1, deg_p + 1):
        wp[:, k] = wp[:, k - 1] * w

    wq = numpy.ones((n_samples, deg_q + 1), dtype=complex)
    for k in range(1, deg_q + 1):
        wq[:, k] = wq[:, k - 1] * w

    A = numpy.hstack([wp, -m1[:, None] * wq[:, 1:]])
    bvec = m1

    # Scale columns for better conditioning on LS
    s = numpy.linalg.norm(A, axis=0)
    s[s == 0] = 1.0
    As = A / s[None, :]

    if ridge_lambda is None:
        ridge_lambda = 0.0

    if ridge_lambda > 0.0:
        # Scale ridge by average diagonal magnitude of AhA
        # Since columns of As have unit norm, this is typically ~1.
        alpha = ridge_lambda

        # Solving augmented least square
        n_coef = As.shape[1]
        A_aug = numpy.vstack([As, numpy.sqrt(alpha) * numpy.eye(
            n_coef, dtype=complex)])
        b_aug = numpy.concatenate([bvec, numpy.zeros(n_coef, dtype=complex)])

        coef, _, _, _ = numpy.linalg.lstsq(A_aug, b_aug, rcond=None)
    else:
        coef, _, _, _ = numpy.linalg.lstsq(As, bvec, rcond=None)

    coef = coef / s

    p = coef[:deg_p + 1]
    q = numpy.zeros(deg_q + 1, dtype=complex)
    q[0] = 1.0
    q[1:] = coef[deg_p + 1:]

    return p, q


# =========
# eval pade
# =========

def eval_pade(w, p, q):

    num = numpy.zeros_like(w, dtype=complex)
    den = numpy.zeros_like(w, dtype=complex)

    for k in range(len(p) - 1, -1, -1):
        num = num * w + p[k]

    for k in range(len(q) - 1, -1, -1):
        den = den * w + q[k]

    return num / den


# =============
# generate pade
# =============

def generate_pade(m1_fn, a, b, deg_p=12, deg_q=12, n_samples=4096, r=1.2,
                  n_r=1, r_min=None, ridge_lambda=0.0):

    if r_min is None:
        r_min = 1.0 + 0.05 * (r - 1.0) if r > 1.0 else 1.0

    if n_r is None or n_r < 1:
        n_r = 1

    if n_samples % 2 != 0:
        raise ValueError('n_samples should be even.')

    if n_r == 1:
        rs = numpy.array([r], dtype=float)
    else:
        rs = numpy.linspace(r_min, r, n_r)

    W_list = []
    M_list = []

    n_half = n_samples // 2

    for r_i in rs:

        # Generate sample points along theta
        theta = numpy.pi * (numpy.arange(n_half) + 0.5) / n_half
        w = r_i * numpy.exp(1j * theta)
        z = joukowski_z(w, a, b)
        m1 = m1_fn(z)

        W_list.append(w)
        M_list.append(m1)

        # Add conjugate points which enforces Schwarz reflection
        W_list.append(numpy.conjugate(w))
        M_list.append(numpy.conjugate(m1))

    w_all = numpy.concatenate(W_list)
    m1_all = numpy.concatenate(M_list)

    # Fit on the sample data from m1
    p, q = fit_pade(w_all, m1_all, deg_p=deg_p, deg_q=deg_q,
                    ridge_lambda=ridge_lambda)

    return p, q
