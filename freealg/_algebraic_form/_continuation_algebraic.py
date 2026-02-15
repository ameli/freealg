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
from .._geometric_form._continuation_genus0 import joukowski_z
from ._constraints import build_moment_constraint_matrix

__all__ = ['sample_z_joukowski', 'filter_z_away_from_cuts', 'powers',
           'fit_polynomial_relation', 'sanity_check_stieltjes_branch',
           'eval_P', 'eval_roots']


# ======================
# normalize coefficients
# ======================

def _normalize_coefficients(arr):
    """
    Trim rows and columns on the sides (equivalent to factorizing or reducing
    degree) and normalize so that the sum of the first column is one.
    """

    a = numpy.asarray(arr).copy()

    if a.size == 0:
        return a

    # Trim zero rows (top and bottom)
    non_zero_rows = numpy.any(a != 0, axis=1)
    if not numpy.any(non_zero_rows):
        return a[:0, :0]

    first_row = numpy.argmax(non_zero_rows)
    last_row = len(non_zero_rows) - numpy.argmax(non_zero_rows[::-1])
    a = a[first_row:last_row, :]

    # Trim zero columns (left and right)
    non_zero_cols = numpy.any(a != 0, axis=0)
    if not numpy.any(non_zero_cols):
        return a[:, :0]

    first_col = numpy.argmax(non_zero_cols)
    last_col = len(non_zero_cols) - numpy.argmax(non_zero_cols[::-1])
    a = a[:, first_col:last_col]

    # Normalize so first column sums to 1
    col_sum = numpy.sum(numpy.abs(a[:, 0])) * float(numpy.sign(a[0, 0]))
    if col_sum != 0:
        a = a / col_sum

    return a


# ==================
# sample z joukowski
# ==================

def sample_z_joukowski(a, b, n_samples=4096, r=1.25, n_r=3, r_min=None):

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

    n_half = n_samples // 2
    theta = numpy.pi * (numpy.arange(n_half) + 0.5) / n_half

    z_list = []
    for r_i in rs:
        w = r_i * numpy.exp(1j * theta)
        z = joukowski_z(w, a, b)
        z_list.append(z)
        z_list.append(numpy.conjugate(z))

    return numpy.concatenate(z_list)


# =======================
# filter z away from cuts
# =======================

def filter_z_away_from_cuts(z, cuts, y_eps=1e-2, x_pad=0.0):

    z = numpy.asarray(z, dtype=numpy.complex128).ravel()
    x = numpy.real(z)
    y = numpy.imag(z)

    keep = numpy.ones(z.size, dtype=bool)
    for a, b in cuts:
        aa = a - x_pad
        bb = b + x_pad
        near_real_cut = (numpy.abs(y) <= y_eps) & (x >= aa) & (x <= bb)
        keep &= ~near_real_cut

    return z[keep]


# ======
# powers
# ======

def powers(x, deg):

    n = x.size
    xp = numpy.ones((n, deg + 1), dtype=complex)
    for k in range(1, deg + 1):
        xp[:, k] = xp[:, k - 1] * x
    return xp


# =======================
# fit polynomial relation
# =======================

def fit_polynomial_relation(z, m, s, deg_z, ridge_lambda=0.0, weights=None,
                            triangular=None, normalize=False,
                            mu=None, mu_reg=None):
    """
    Fits polynomial P(z, m) = 0 with samples from the physical branch.
    """

    z = numpy.asarray(z, dtype=complex).ravel()
    m = numpy.asarray(m, dtype=complex).ravel()

    if z.size != m.size:
        raise ValueError('z and m must have the same size.')
    if s < 1:
        raise ValueError('s must be >= 1.')
    if deg_z < 0:
        raise ValueError('deg_z must be >= 0.')

    zp = powers(z, deg_z)
    mp = powers(m, s)

    if weights is None:
        w = None
    else:
        w = numpy.asarray(weights, dtype=float).ravel()
        if w.size != z.size:
            raise ValueError('weights must have the same size as z.')
        w = numpy.sqrt(numpy.maximum(w, 0.0))

    tri = None
    if triangular is not None:
        tri = str(triangular).strip().lower()
        if tri in ['none', '']:
            tri = None

    if tri is None:
        pairs = [(i, j) for j in range(s + 1)
                 for i in range(deg_z + 1)]

    elif tri in ['lower', 'l']:
        pairs = [(i, j) for j in range(s + 1)
                 for i in range(deg_z + 1) if i >= j]

    elif tri in ['upper', 'u']:
        pairs = [(i, j) for j in range(s + 1)
                 for i in range(deg_z + 1) if i <= j]

    elif tri in ['antidiag', 'anti', 'antidiagonal', 'ad']:
        pairs = [(i, j) for j in range(s + 1)
                 for i in range(deg_z + 1) if (i + j) <= deg_z]

        if len(pairs) == 0:
            raise ValueError('antidiag constraint removed all coefficients.')
    else:
        raise ValueError("triangular must be None, 'lower', 'upper', or " +
                         "'antidiag'.")

    n_coef = len(pairs)
    A = numpy.empty((z.size, n_coef), dtype=complex)

    for k, (i, j) in enumerate(pairs):
        A[:, k] = zp[:, i] * mp[:, j]

    if w is not None:
        A = A * w[:, None]

    # Enforce real coefficients by solving: Re(A) c = 0 and Im(A) c = 0
    Ar = numpy.vstack([A.real, A.imag])

    s_col = numpy.max(numpy.abs(Ar), axis=0)
    s_col[s_col == 0.0] = 1.0
    As = Ar / s_col[None, :]

    # Optional moment constraints B c = 0 (hard via nullspace, soft via
    # weighted rows)
    if mu is not None:
        B = build_moment_constraint_matrix(pairs, deg_z, s, mu)
        if B.shape[0] > 0:
            Bs = B / s_col[None, :]

            if mu_reg is None:
                # Hard constraints: solve in nullspace of Bs
                uB, sB, vhB = numpy.linalg.svd(Bs, full_matrices=True)
                tolB = 1e-12 * (sB[0] if sB.size else 1.0)
                rankB = int(numpy.sum(sB > tolB))
                if rankB >= n_coef:
                    raise RuntimeError(
                        'Moment constraints leave no feasible coefficients.')

                N = vhB[rankB:, :].T  # (n_coef, n_free)
                AN = As @ N

                if ridge_lambda > 0.0:
                    L = numpy.sqrt(ridge_lambda) * numpy.eye(N.shape[1],
                                                             dtype=float)
                    AN = numpy.vstack([AN, L])

                _, svals, vhN = numpy.linalg.svd(AN, full_matrices=False)
                y = vhN[-1, :]
                coef_scaled = N @ y

                coef = coef_scaled / s_col

            else:
                mu_reg = float(mu_reg)
                if mu_reg > 0.0:
                    As_aug = As
                    Bs_w = numpy.sqrt(mu_reg) * Bs
                    As_aug = numpy.vstack([As_aug, Bs_w])

                    if ridge_lambda > 0.0:
                        L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef,
                                                                 dtype=float)
                        As_aug = numpy.vstack([As_aug, L])

                    _, svals, vh = numpy.linalg.svd(As_aug,
                                                    full_matrices=False)
                    coef_scaled = vh[-1, :]
                    coef = coef_scaled / s_col
                else:
                    # mu_reg == 0 => ignore constraints
                    if ridge_lambda > 0.0:
                        L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef,
                                                                 dtype=float)
                        As = numpy.vstack([As, L])

                    _, svals, vh = numpy.linalg.svd(As, full_matrices=False)
                    coef_scaled = vh[-1, :]
                    coef = coef_scaled / s_col

        else:
            # B has no effective rows -> proceed unconstrained
            if ridge_lambda > 0.0:
                L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef, dtype=float)
                As = numpy.vstack([As, L])

            _, svals, vh = numpy.linalg.svd(As, full_matrices=False)
            coef_scaled = vh[-1, :]
            coef = coef_scaled / s_col

    else:
        # No moment constraints
        if ridge_lambda > 0.0:
            L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef, dtype=float)
            As = numpy.vstack([As, L])

        _, svals, vh = numpy.linalg.svd(As, full_matrices=False)
        coef_scaled = vh[-1, :]
        coef = coef_scaled / s_col

    full = numpy.zeros((deg_z + 1, s + 1), dtype=complex)
    for k, (i, j) in enumerate(pairs):
        full[i, j] = coef[k]

    if normalize:
        full = _normalize_coefficients(full)

    # Diagnostic metrics
    fit_metrics = {
        's_min': float(svals[-1]),
        'gap_ratio': float(svals[-2] / svals[-1]),
        'n_small': float(int(numpy.sum(svals <= svals[0] * 1e-12))),
    }

    return full, fit_metrics


# =============================
# sanity check stieltjes branch
# =============================

def sanity_check_stieltjes_branch(coeffs, x_min, x_max, eta=0.1,
                                  n_x=64, y0=None, max_bad_frac=0.05):
    """
    Quick sanity check: does P(z,m)=0 admit a continuously trackable root with
    Im(m)>0 along z=x+i*eta.
    """

    x_min = float(x_min)
    x_max = float(x_max)
    eta = float(eta)
    n_x = int(n_x)
    if n_x < 4:
        n_x = 4

    if y0 is None:
        y0 = 10.0 * max(1.0, abs(x_min), abs(x_max))
    y0 = float(y0)

    z0 = 1j * y0
    m0_target = -1.0 / z0

    c0 = _poly_coef_in_m(numpy.array([z0]), coeffs)[0]
    r0 = numpy.roots(c0[::-1])
    if r0.size == 0:
        return {'ok': False, 'frac_bad': 1.0, 'n_test': 0, 'n_bad': 0}

    k0 = int(numpy.argmin(numpy.abs(r0 - m0_target)))
    m_prev = r0[k0]

    xs = numpy.linspace(x_min, x_max, n_x)
    zs = xs + 1j * eta

    n_bad = 0
    n_ok = 0

    for z in zs:
        c = _poly_coef_in_m(numpy.array([z]), coeffs)[0]
        r = numpy.roots(c[::-1])
        if r.size == 0 or not numpy.all(numpy.isfinite(r)):
            n_bad += 1
            continue

        k = int(numpy.argmin(numpy.abs(r - m_prev)))
        m_sel = r[k]
        m_prev = m_sel
        n_ok += 1

        if not numpy.isfinite(m_sel) or (m_sel.imag <= 0.0):
            n_bad += 1

    n_test = n_ok + (n_bad - (n_x - n_ok))
    if n_test <= 0:
        n_test = n_x

    frac_bad = float(n_bad) / float(n_x)
    ok = frac_bad <= float(max_bad_frac)

    status = {
        'ok': ok,
        'frac_bad': frac_bad,
        'n_test': n_x,
        'n_bad': n_bad
    }

    return status


# ======
# eval P
# ======

def eval_P(z, m, coeffs):

    z = numpy.asarray(z, dtype=complex)
    m = numpy.asarray(m, dtype=complex)
    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)

    shp = numpy.broadcast(z, m).shape
    zz = numpy.broadcast_to(z, shp).ravel()
    mm = numpy.broadcast_to(m, shp).ravel()

    zp = powers(zz, deg_z)
    mp = powers(mm, s)

    P = numpy.zeros(zz.size, dtype=complex)
    for j in range(s + 1):
        aj = zp @ coeffs[:, j]
        P = P + aj * mp[:, j]

    return P.reshape(shp)


# ==============
# poly coef in m
# ==============

def _poly_coef_in_m(z, coeffs):

    z = numpy.asarray(z, dtype=complex).ravel()
    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)
    zp = powers(z, deg_z)

    c = numpy.empty((z.size, s + 1), dtype=complex)
    for j in range(s + 1):
        c[:, j] = zp @ coeffs[:, j]
    return c


# ==============
# root quadratic
# ==============

def _roots_quadratic(c0, c1, c2):

    disc = c1 * c1 - 4.0 * c2 * c0
    sq = numpy.sqrt(disc)
    den = 2.0 * c2

    r1 = (-c1 + sq) / den
    r2 = (-c1 - sq) / den
    return numpy.stack([r1, r2], axis=1)


# ============
# cbrt complex
# ============

def _cbrt_complex(z):

    z = numpy.asarray(z, dtype=complex)
    r = numpy.abs(z)
    th = numpy.angle(z)
    return (r ** (1.0 / 3.0)) * numpy.exp(1j * th / 3.0)


# ==========
# root cubic
# ==========

def _roots_cubic(c0, c1, c2, c3):

    c0 = numpy.asarray(c0, dtype=complex)
    c1 = numpy.asarray(c1, dtype=complex)
    c2 = numpy.asarray(c2, dtype=complex)
    c3 = numpy.asarray(c3, dtype=complex)

    a = c2 / c3
    b = c1 / c3
    c = c0 / c3

    p = b - (a * a) / 3.0
    q = (2.0 * a * a * a) / 27.0 - (a * b) / 3.0 + c

    Delta = (q * q) / 4.0 + (p * p * p) / 27.0
    sqrtD = numpy.sqrt(Delta)

    A = -q / 2.0 + sqrtD
    u = _cbrt_complex(A)

    eps = 1e-30
    small = numpy.abs(u) < eps
    if numpy.any(small):
        u2 = _cbrt_complex(-q / 2.0 - sqrtD)
        u = numpy.where(small, u2, u)

    small = numpy.abs(u) < eps
    v = numpy.empty_like(u)
    v[~small] = -p[~small] / (3.0 * u[~small])
    v[small] = _cbrt_complex(-q[small])

    y1 = u + v
    w = complex(-0.5, numpy.sqrt(3.0) / 2.0)
    y2 = w * u + numpy.conjugate(w) * v
    y3 = numpy.conjugate(w) * u + w * v

    x1 = y1 - a / 3.0
    x2 = y2 - a / 3.0
    x3 = y3 - a / 3.0

    return numpy.stack([x1, x2, x3], axis=1)


# ==========
# eval roots
# ==========

def eval_roots(z, coeffs):

    z = numpy.asarray(z, dtype=complex).ravel()
    c = _poly_coef_in_m(z, coeffs)

    s = int(c.shape[1] - 1)
    if s == 1:
        m = -c[:, 0] / c[:, 1]
        return m[:, None]

    if s == 2:
        return _roots_quadratic(c[:, 0], c[:, 1], c[:, 2])

    if s == 3:
        return _roots_cubic(c[:, 0], c[:, 1], c[:, 2], c[:, 3])

    roots = numpy.empty((z.size, s), dtype=complex)
    for i in range(z.size):
        roots[i, :] = numpy.roots(c[i, ::-1])
    return roots
