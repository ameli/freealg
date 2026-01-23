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

import numpy

__all__ = ['compute_branch_points']


# =========
# poly trim
# =========

def _poly_trim(p, tol):
    p = numpy.asarray(p, dtype=float)
    if p.size == 0:
        return p
    k = p.size - 1
    while k > 0 and abs(p[k]) <= tol:
        k -= 1
    return p[: k + 1]


# ========
# poly add
# ========

def _poly_add(a, b, tol):

    n = max(len(a), len(b))
    out = numpy.zeros(n, dtype=float)
    out[: len(a)] += a
    out[: len(b)] += b

    return _poly_trim(out, tol)


# ========
# poly sub
# ========

def _poly_sub(a, b, tol):

    n = max(len(a), len(b))
    out = numpy.zeros(n, dtype=float)
    out[: len(a)] += a
    out[: len(b)] -= b

    return _poly_trim(out, tol)


# ========
# poly mul
# ========

def _poly_mul(a, b, tol):

    a = _poly_trim(a, tol)
    b = _poly_trim(b, tol)
    if a.size == 0 or b.size == 0:
        return numpy.zeros(1, dtype=float)
    out = numpy.convolve(a, b)
    return _poly_trim(out, tol)


# ===============
# poly div approx
# ===============

def _poly_div_approx(a, b, tol):
    """
    Polynomial division q,r = a/b in ascending powers (numpy.polynomial
    convention). Returns q (ascending). Remainder is ignored if it is
    small-ish.
    """

    a = _poly_trim(a, tol)
    b = _poly_trim(b, tol)
    if b.size == 0 or (b.size == 1 and abs(b[0]) <= tol):
        raise RuntimeError(
            "division by (near) zero polynomial in branch point resultant")
    # numpy.polydiv uses descending powers, so flip.
    qd, rd = numpy.polydiv(a[::-1], b[::-1])
    q = qd[::-1]
    r = rd[::-1]
    # Accept small remainder (Bareiss should be exact in exact arithmetic).
    # If not small, we still proceed with the quotient (robustness over
    # exactness).
    scale = max(1.0, numpy.linalg.norm(a))
    if numpy.linalg.norm(_poly_trim(r, tol)) > 1e6 * tol * scale:
        pass
    return _poly_trim(q, tol)


# =================
# det baresiss poly
# =================

def _det_bareiss_poly(M, tol):
    """
    Fraction-free determinant for a matrix with polynomial entries in z.
    Polynomials are stored as 1D arrays of ascending coefficients.
    Returns det as ascending coefficients.
    """

    n = len(M)
    A = [[_poly_trim(M[i][j], tol) for j in range(n)] for i in range(n)]
    denom = numpy.array([1.0], dtype=float)

    for k in range(n - 1):
        pivot = A[k][k]
        if pivot.size == 1 and abs(pivot[0]) <= tol:
            swap = None
            for i in range(k + 1, n):
                if not (A[i][k].size == 1 and abs(A[i][k][0]) <= tol):
                    swap = i
                    break
            if swap is None:
                return numpy.zeros(1, dtype=float)
            A[k], A[swap] = A[swap], A[k]
            pivot = A[k][k]

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = _poly_sub(
                    _poly_mul(A[i][j], pivot, tol),
                    _poly_mul(A[i][k], A[k][j], tol),
                    tol,
                )
                if k > 0:
                    A[i][j] = _poly_div_approx(num, denom, tol)
                else:
                    A[i][j] = _poly_trim(num, tol)

        denom = pivot

        for i in range(k + 1, n):
            A[i][k] = numpy.array([0.0], dtype=float)
            A[k][i] = numpy.array([0.0], dtype=float)

    return _poly_trim(A[n - 1][n - 1], tol)


# ======================
# resultant discriminant
# ======================

def _resultant_discriminant(a_coeffs, tol):
    """
    Numerically compute Disc_m(P)(z) as a polynomial in z (ascending coeffs),
    via Sylvester determinant evaluation on a circle + interpolation.

    a_coeffs[i,j] is coeff of z^i m^j, shape (deg_z+1, s+1).
    """

    import numpy

    a_coeffs = numpy.asarray(a_coeffs, dtype=numpy.complex128)
    deg_z = a_coeffs.shape[0] - 1
    s = a_coeffs.shape[1] - 1
    if s < 1 or deg_z < 0:
        return numpy.zeros(1, dtype=numpy.complex128)

    # Degree bound: deg_z(Disc) <= (2s-1)*deg_z
    D = (2 * s - 1) * deg_z
    if D <= 0:
        return numpy.zeros(1, dtype=numpy.complex128)

    def eval_disc(z):
        # Build P(m) coeffs in descending powers of m: p_desc[k] = coeff of
        # m^(s-k)
        p_asc = numpy.zeros(s + 1, dtype=numpy.complex128)
        for j in range(s + 1):
            p_asc[j] = numpy.polyval(a_coeffs[:, j][::-1], z)  # a_j(z)
        p_desc = p_asc[::-1]

        # Q(m) = dP/dm, descending
        q_asc = numpy.zeros(s, dtype=numpy.complex128)
        for j in range(1, s + 1):
            q_asc[j - 1] = j * p_asc[j]
        q_desc = q_asc[::-1]

        # Sylvester matrix of P (deg s) and Q (deg s-1): size (2s-1)x(2s-1)
        n = 2 * s - 1
        S = numpy.zeros((n, n), dtype=numpy.complex128)

        # First (s-1) rows: shifts of P
        for r in range(s - 1):
            S[r, r:r + (s + 1)] = p_desc

        # Next s rows: shifts of Q
        for r in range(s):
            rr = (s - 1) + r
            S[rr, r:r + s] = q_desc

        return numpy.linalg.det(S)

    # Sample points on a circle; scale radius using coefficient magnitude
    # (simple heuristic) (This only affects conditioning of interpolation, not
    # correctness.)
    scale = float(numpy.max(numpy.abs(a_coeffs))) \
        if numpy.max(numpy.abs(a_coeffs)) > 0 else 1.0
    R = 1.0 + 0.1 * scale

    N = D + 1
    k = numpy.arange(N, dtype=float)
    z_samp = R * numpy.exp(2.0j * numpy.pi * k / float(N))
    d_samp = numpy.array([eval_disc(z) for z in z_samp],
                         dtype=numpy.complex128)

    # Interpolate disc(z) = sum_{j=0}^D c[j] z^j  (ascending)
    V = (z_samp[:, None] ** numpy.arange(D + 1)[None, :]).astype(
        numpy.complex128)
    c, _, _, _ = numpy.linalg.lstsq(V, d_samp, rcond=None)

    # Trim tiny coefficients
    c = _poly_trim(c, tol)
    if c.size == 0:
        c = numpy.zeros(1, dtype=numpy.complex128)

    # If numerics leave small imag, kill it (disc should be real-coeff if
    # a_coeffs real)
    if numpy.linalg.norm(c.imag) <= \
            1e3 * tol * max(1.0, numpy.linalg.norm(c.real)):
        c = c.real.astype(numpy.float64)

    return c


# =====================
# compute branch points
# =====================

def compute_branch_points(a_coeffs, tol=1e-12, real_tol=None):
    """
    Compute global branch points of the affine curve P(z,m)=0 by
    z-roots of Disc_m(P)(z) = Res_m(P, dP/dm).

    Returns
    -------
    z_bp : complex ndarray
    a_s_zero : complex ndarray
    info : dict
    """

    a_coeffs = numpy.asarray(a_coeffs, dtype=float)
    s = a_coeffs.shape[1] - 1
    if s < 1:
        if real_tol is None:
            real_tol = 1e3 * tol
        return \
            numpy.array([], dtype=complex), \
            numpy.array([], dtype=complex), \
            {
                "disc": numpy.zeros(1, dtype=float),
                "tol": float(tol),
                "real_tol": float(real_tol),
            }

    if real_tol is None:
        real_tol = 1e3 * tol

    a_s = _poly_trim(a_coeffs[:, s], tol)
    a_s_zero = numpy.roots(a_s[::-1]) if a_s.size > 1 else \
        numpy.array([], dtype=complex)

    disc = _resultant_discriminant(a_coeffs, tol)
    if disc.size <= 1:
        z_bp = numpy.array([], dtype=complex)
    else:
        z_bp = numpy.roots(disc[::-1])

    info = {
        "disc": disc,
        "tol": float(tol),
        "real_tol": float(real_tol),
    }

    return z_bp, a_s_zero, info
