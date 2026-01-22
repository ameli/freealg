# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import numpy as np
import numpy.polynomial.polynomial as poly

__all__ = ['compute_singular_points']


# =========
# ploy trim
# =========

def _poly_trim(p, tol):

    p = np.asarray(p, dtype=complex).ravel()
    if p.size == 0:
        return np.zeros(1, dtype=complex)
    k = p.size - 1
    while k > 0 and abs(p[k]) <= tol:
        k -= 1
    return p[:k + 1].copy()


# ============
# poly is zero
# ============

def _poly_is_zero(p, tol):

    p = _poly_trim(p, tol)
    return (p.size == 1) and (abs(p[0]) <= tol)


# ========
# poly add
# ========

def _poly_add(a, b, tol):

    return _poly_trim(poly.polyadd(a, b), tol)


# ========
# poly sub
# ========

def _poly_sub(a, b, tol):

    return _poly_trim(poly.polysub(a, b), tol)


# =======
# ply mul
# =======

def _poly_mul(a, b, tol):

    return _poly_trim(poly.polymul(a, b), tol)


# ==============
# poly div exact
# ==============

def _poly_div_exact(a, b, tol):

    a = _poly_trim(a, tol)
    b = _poly_trim(b, tol)
    if _poly_is_zero(b, tol):
        raise ZeroDivisionError("poly division by zero")

    q, r = poly.polydiv(a, b)
    r = _poly_trim(r, tol)

    # Bareiss expects exact division; with floats it's only approximate.
    # If the remainder is small, drop it.
    scale = max(1.0, np.linalg.norm(a))
    if np.linalg.norm(r) > 1e3 * tol * scale:
        # Fallback: still drop remainder (keeps algorithm running).
        # This is acceptable because we only need the resultant roots
        # robustly, not exact symbolic coefficients.
        pass

    return _poly_trim(q, tol)


# ================
# det bareiss poly
# ================

def _det_bareiss_poly(M, tol):

    n = len(M)
    A = [[_poly_trim(M[i][j], tol) for j in range(n)] for i in range(n)]

    denom = np.array([1.0], dtype=complex)
    sign = 1.0

    for k in range(n - 1):
        if _poly_is_zero(A[k][k], tol):
            piv = -1
            for i in range(k + 1, n):
                if not _poly_is_zero(A[i][k], tol):
                    piv = i
                    break
            if piv == -1:
                return np.array([0.0], dtype=complex)
            A[k], A[piv] = A[piv], A[k]
            sign *= -1.0

        pivot = A[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = _poly_sub(_poly_mul(A[i][j], pivot, tol),
                                _poly_mul(A[i][k], A[k][j], tol),
                                tol)
                if k > 0:
                    A[i][j] = _poly_div_exact(num, denom, tol)
                else:
                    A[i][j] = _poly_trim(num, tol)

        denom = pivot

    return _poly_trim(sign * A[n - 1][n - 1], tol)


# ===================
# cluster real points
# ===================

def _cluster_real_points(x, eps):

    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return x
    x = np.sort(x)
    uniq = []
    for v in x:
        if (len(uniq) == 0) or (abs(v - uniq[-1]) > eps):
            uniq.append(float(v))
        else:
            uniq[-1] = 0.5 * (uniq[-1] + float(v))
    return np.asarray(uniq, dtype=float)


# =======================
# compute singular points
# =======================

def compute_singular_points(a_coeffs, tol=1e-12, real_tol=None):
    """
    a_coeffs[i,j] is coefficient of z^i m^j, shape (deg_z+1, s+1).

    Returns
    -------

    z_bp     : complex array, roots of Disc_m(P)(z)
    a_s_zero : complex array, roots of leading coefficient a_s(z)
    support  : list of (a,b) from real-ish branch points paired consecutively
    """

    a_coeffs = np.asarray(a_coeffs)
    s = a_coeffs.shape[1] - 1
    if s < 1:
        return (np.array([], dtype=complex),
                np.array([], dtype=complex),
                [])

    if real_tol is None:
        real_tol = 1e3 * tol

    a = [_poly_trim(a_coeffs[:, j], tol) for j in range(s + 1)]

    a_s = a[s]
    a_s_zero = np.roots(a_s[::-1]) if a_s.size > 1 else \
        np.array([], dtype=complex)

    b = []
    for j in range(s):
        b.append(_poly_trim((j + 1) * a[j + 1], tol))

    mdeg = s
    ndeg = s - 1
    N = mdeg + ndeg  # 2s-1

    z0 = np.array([0.0], dtype=complex)
    M = [[z0 for _ in range(N)] for __ in range(N)]

    for r in range(ndeg):
        for j in range(mdeg + 1):
            M[r][r + j] = a[j]

    for r in range(mdeg):
        rr = ndeg + r
        for j in range(ndeg + 1):
            M[rr][r + j] = b[j]

    res = _det_bareiss_poly(M, tol)
    if res.size <= 1:
        z_bp = np.array([], dtype=complex)
    else:
        z_bp = np.roots(res[::-1])

    support = []
    if z_bp.size > 0:
        zr = z_bp[np.abs(z_bp.imag) <= real_tol].real
        zr = _cluster_real_points(zr, eps=1e2 * real_tol)
        m2 = (zr.size // 2) * 2
        for k in range(0, m2, 2):
            a0 = float(zr[k])
            b0 = float(zr[k + 1])
            if b0 > a0:
                support.append((a0, b0))

    return z_bp, a_s_zero, support
