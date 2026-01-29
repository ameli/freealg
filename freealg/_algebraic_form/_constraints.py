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

__all__ = ['build_moment_constraint_matrix']


# ==========
# series mul
# ==========

def _series_mul(a, b, q_max):

    na = min(len(a), q_max + 1)
    nb = min(len(b), q_max + 1)
    out = numpy.zeros(q_max + 1, dtype=float)
    for i in range(na):
        if a[i] == 0.0:
            continue
        j_max = min(nb - 1, q_max - i)
        if j_max >= 0:
            out[i:i + j_max + 1] += a[i] * b[:j_max + 1]
    return out


# ==========
# series pow
# ==========

def _series_pow(mser, j, q_max):
    if j == 0:
        out = numpy.zeros(q_max + 1, dtype=float)
        out[0] = 1.0
        return out
    out = mser.copy()
    for _ in range(1, j):
        out = _series_mul(out, mser, q_max)
    return out


# ===============================
# build moment constraints matrix
# ===============================

# def build_moment_constraint_matrix(pairs, deg_z, s, mu):
#
#     mu = numpy.asarray(mu, dtype=float).ravel()
#     if mu.size == 0:
#         return numpy.zeros((0, len(pairs)), dtype=float)
#
#     # m(z) = -sum_{p>=0} mu_p / z^{p+1}; t = 1/z so m(t) = -sum mu_p t^{p+1}
#     r = mu.size - 1
#     q_max = r
#
#     mser = numpy.zeros(q_max + 1, dtype=float)
#     for p in range(mu.size):
#         q = p + 1
#         if q <= q_max:
#             mser[q] = -float(mu[p])
#
#     # Precompute (m(t))^j coefficients up to t^{q_max}
#     mpow = []
#     for j in range(s + 1):
#         mpow.append(_series_pow(mser, j, q_max))
#
#     # Constraints: coeff of t^q in Q(t) := t^{deg_z} P(1/t, m(t)) must be 0
#     # Q(t) = sum_{i,j} c_{i,j} * t^{deg_z - i} * (m(t))^j
#     n_coef = len(pairs)
#     B = numpy.zeros((q_max + 1, n_coef), dtype=float)
#
#     for k, (i, j) in enumerate(pairs):
#         shift = deg_z - i
#         if shift < 0:
#             continue
#         mj = mpow[j]
#         for q in range(q_max + 1):
#             qq = q - shift
#             if 0 <= qq <= q_max:
#                 B[q, k] = mj[qq]
#
#     # Drop all-zero rows (can happen if index-set can't support higher
#     # moments)
#     row_norm = numpy.linalg.norm(B, axis=1)
#     keep = row_norm > 0.0
#     B = B[keep, :]
#
#     return B

def build_moment_constraint_matrix(pairs, deg_z, s, mu):

    mu = numpy.asarray(mu, dtype=float).ravel()
    if mu.size == 0:
        return numpy.zeros((0, len(pairs)), dtype=float)

    # mu has entries mu_0..mu_r
    r = mu.size - 1

    # Need t^{r+1} in m(t) = -sum mu_p t^{p+1}, otherwise mu_0 is dropped.
    q_max = r + 1

    mser = numpy.zeros(q_max + 1, dtype=float)
    for p in range(mu.size):
        q = p + 1
        if q <= q_max:
            mser[q] = -float(mu[p])

    mpow = []
    for j in range(s + 1):
        mpow.append(_series_pow(mser, j, q_max))

    n_coef = len(pairs)

    # We only want constraints for l=0..r  -> that's q = 0..r in Q(t)
    B = numpy.zeros((r + 1, n_coef), dtype=float)

    for k, (i, j) in enumerate(pairs):
        shift = deg_z - i
        if shift < 0:
            continue
        mj = mpow[j]
        for q in range(r + 1):
            qq = q - shift
            if 0 <= qq <= q_max:
                B[q, k] = mj[qq]

    row_norm = numpy.linalg.norm(B, axis=1)
    keep = row_norm > 0.0
    return B[keep, :]
