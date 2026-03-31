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
           'fit_polynomial_relation', 'eval_P', 'eval_roots']


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

def sample_z_joukowski(support, n_samples=4096, r_min=1.8, r_max=2.2, n_r=5,
                       y_scale=1.0, gamma=1.0, log=False, dtype=complex):
    """
    """

    if numpy.isscalar(n_samples):
        n_samples = numpy.tile(n_samples, len(support))

    rs = numpy.linspace(r_min, r_max, n_r)

    z_list = []
    for i, supp in enumerate(support):
        a, b = supp

        if n_samples[i] % 2 != 0:
            raise ValueError('n_samples should be even.')

        n_half = n_samples[i] // 2
        theta = numpy.pi * (numpy.arange(n_half) + 0.5) / n_half

        for r_i in rs:
            w = r_i * numpy.exp(1j * theta)

            if log:
                u = joukowski_z(w, numpy.log(a), numpy.log(b))

                # Option 1: Log only along x axis
                Delta = numpy.log(b / a)
                x = numpy.exp(u.real)
                shape = u.imag / numpy.median(numpy.abs(u.imag))
                y = y_scale * (x ** gamma) * Delta * shape
                z = x + 1j * y

                # Option 2: Full log in z plane
                # z = numpy.exp(u)
            else:
                z = joukowski_z(w, a, b)

            z_list.append(z)
            z_list.append(numpy.conjugate(z))

    return numpy.concatenate(z_list, dtype=dtype)


# =======================
# filter z away from cuts
# =======================

def filter_z_away_from_cuts(z, cuts, cut_eps=0.01, log=False):
    """
    Remove points that are too close to any real cut interval.

    Parameters
    ----------

    z : array_like
        Complex sample points.

    cuts : sequence of (a, b)
        Real cut intervals.

    cut_eps : float, default=0.01
        Fraction of cut width used as the distance threshold.

        * log=False: threshold is ``cut_eps * (b - a)``, where ``(b - a)`` is
          the linear width of the cut interval.

        * log=True: threshold is ``cut_eps * log(b / a)``, where ``log(b / a)``
          is the logarithmic width of the cut interval.

    log : bool
        If True, use a cut-dependent relative threshold:
            eps_j = cut_eps * sqrt(a*b)
        If False, use:
            eps_j = cut_eps

    Returns
    -------

    z_filt : ndarray
        Filtered points.
    """

    z = numpy.asarray(z).ravel()

    if cut_eps is None or cut_eps <= 0:
        return z

    x = z.real
    y = numpy.abs(z.imag)

    keep = numpy.ones(z.size, dtype=bool)

    for a, b in cuts:
        if log:
            # log-aware coordinates: horizontal = log x, vertical = y/x
            dx = numpy.where(x < a, numpy.log(a / x),
                             numpy.where(x > b, numpy.log(x / b), 0.0))
            dy = numpy.abs(numpy.arctan2(y, x))
            dist = numpy.sqrt(dx**2 + dy**2)
            eps_j = cut_eps * numpy.log(b / a)
        else:
            # ordinary linear geometry in z-plane
            dx = numpy.where(x < a, a - x,
                             numpy.where(x > b, x - b, 0.0))
            dy = y
            dist = numpy.sqrt(dx**2 + dy**2)
            eps_j = cut_eps * (b - a)

        keep &= (dist > eps_j)

    return z[keep]


# ======
# powers
# ======

def powers(x, deg, dtype=complex):

    n = x.size
    xp = numpy.ones((n, deg + 1), dtype=dtype)
    for k in range(1, deg + 1):
        xp[:, k] = xp[:, k - 1] * x
    return xp


# ========
# real dot
# ========

def _real_dot(u, v):
    """
    Real Euclidean inner product on complex vectors, consistent with stacking
    [Re; Im].
    """

    return numpy.dot(u.real, v.real) + numpy.dot(u.imag, v.imag)


# =============
# stable powers
# =============

def stable_powers(x, deg, dtype=complex):
    """
    Build a numerically stabilized basis spanning {1, x, ..., x^deg} on the
    sampled points x, using modified Gram-Schmidt with the real stacked inner
    product. This preserves the real-coefficient structure of the fitted
    polynomial.

    Returns
    -------
    q : ndarray
        Basis matrix of shape (n, deg+1), spanning the same column space as
        raw monomials.

    t : ndarray
        Upper-triangular real transform such that

            raw_monomials = q @ t

        Hence coefficients transform back by solving triangular systems.
    """

    x = numpy.asarray(x, dtype=dtype).ravel()
    rdtype = numpy.empty((), dtype=dtype).real.dtype
    n = x.size

    q = numpy.empty((n, deg + 1), dtype=dtype)
    t = numpy.zeros((deg + 1, deg + 1), dtype=rdtype)

    v_prev = numpy.ones(n, dtype=dtype)

    for j in range(deg + 1):
        if j == 0:
            v = numpy.ones(n, dtype=dtype)
        else:
            v = v_prev * x

        w = v.copy()

        # First modified Gram-Schmidt pass
        for k in range(j):
            r = _real_dot(q[:, k], w)
            w = w - r * q[:, k]
            t[k, j] += r

        # Reorthogonalization pass
        for k in range(j):
            r = _real_dot(q[:, k], w)
            w = w - r * q[:, k]
            t[k, j] += r

        nj2 = _real_dot(w, w)
        if nj2 <= 0:
            raise RuntimeError(
                f"Degenerate basis encountered at degree {j}."
            )

        nj = numpy.sqrt(nj2, dtype=rdtype)
        q[:, j] = w / nj
        t[j, j] = nj

        v_prev = v

    return q, t


# =======================
# fit polynomial relation
# =======================

def fit_polynomial_relation(z, m, s, deg_z, ridge_lambda=0.0, weights=None,
                            triangular=None, normalize=False,
                            mu=None, mu_reg=None, dtype=complex):
    """
    Fits polynomial P(z, m) = 0 with samples from the physical branch.

    Note: SVD only accepts float64. So even if dtype is complex256 and float128
    it will cast down.
    """

    rdtype = numpy.empty((), dtype=dtype).real.dtype

    z = numpy.asarray(z, dtype=dtype).ravel()
    m = numpy.asarray(m, dtype=dtype).ravel()

    if z.size != m.size:
        raise ValueError('z and m must have the same size.')
    if s < 1:
        raise ValueError('s must be >= 1.')
    if deg_z < 0:
        raise ValueError('deg_z must be >= 0.')

    # TEST
    # zp = powers(z, deg_z, dtype=dtype)
    # mp = powers(m, s, dtype=dtype)

    # ------------------------------------
    # Scale variables before monomial fit
    # ------------------------------------
    abs_z = numpy.abs(z)
    abs_m = numpy.abs(m)

    # Avoid log(0)
    eps_z = numpy.finfo(rdtype).tiny
    eps_m = numpy.finfo(rdtype).tiny

    z_scale = numpy.exp(
        numpy.mean(numpy.log(numpy.maximum(abs_z, eps_z)), dtype=rdtype)
    ).astype(rdtype)

    m_scale = numpy.exp(
        numpy.mean(numpy.log(numpy.maximum(abs_m, eps_m)), dtype=rdtype)
    ).astype(rdtype)

    if (not numpy.isfinite(z_scale)) or (z_scale == 0):
        z_scale = rdtype.type(1.0)
    if (not numpy.isfinite(m_scale)) or (m_scale == 0):
        m_scale = rdtype.type(1.0)

    zs = z / z_scale
    ms = m / m_scale

    # zp = powers(zs, deg_z, dtype=dtype)
    # mp = powers(ms, s, dtype=dtype)
    zp, tz = stable_powers(zs, deg_z, dtype=dtype)
    mp, tm = stable_powers(ms, s, dtype=dtype)

    if weights is None:
        w = None
    else:
        w = numpy.asarray(weights, dtype=rdtype).ravel()
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
    A = numpy.empty((z.size, n_coef), dtype=dtype)

    for k, (i, j) in enumerate(pairs):
        A[:, k] = zp[:, i] * mp[:, j]

    if w is not None:
        A = A * w[:, None]

    # Enforce real coefficients by solving: Re(A) c = 0 and Im(A) c = 0
    Ar = numpy.vstack([A.real, A.imag])

    s_col = numpy.max(numpy.abs(Ar), axis=0)
    s_col[s_col == 0.0] = 1.0
    As = numpy.asarray(Ar / s_col[None, :], dtype=numpy.float64)  # 64-bit SVD

    # Optional moment constraints B c = 0 (hard via nullspace, soft via
    # weighted rows)
    if mu is not None:
        B = build_moment_constraint_matrix(pairs, deg_z, s, mu)
        if B.shape[0] > 0:
            Bs = numpy.asarray(B / s_col[None, :], dtype=numpy.float64)

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
                                                             dtype=rdtype)
                    AN = numpy.vstack([AN, L], dtype=numpy.float64)

                _, svals, vhN = numpy.linalg.svd(AN, full_matrices=False)
                y = vhN[-1, :]
                coef_scaled = N @ y

                coef = coef_scaled / s_col

            else:
                mu_reg = float(mu_reg)
                if mu_reg > 0.0:
                    As_aug = As
                    Bs_w = numpy.sqrt(mu_reg) * Bs
                    As_aug = numpy.vstack([As_aug, Bs_w], dtype=numpy.float64)

                    if ridge_lambda > 0.0:
                        L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef,
                                                                 dtype=rdtype)
                        As_aug = numpy.vstack([As_aug, L], dtype=numpy.float64)

                    _, svals, vh = numpy.linalg.svd(As_aug,
                                                    full_matrices=False)
                    coef_scaled = vh[-1, :]
                    coef = coef_scaled / s_col
                else:
                    # mu_reg == 0 => ignore constraints
                    if ridge_lambda > 0.0:
                        L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef,
                                                                 dtype=rdtype)
                        As = numpy.vstack([As, L], dtype=numpy.float64)

                    _, svals, vh = numpy.linalg.svd(As, full_matrices=False)
                    coef_scaled = vh[-1, :]
                    coef = coef_scaled / s_col

        else:
            # B has no effective rows -> proceed unconstrained
            if ridge_lambda > 0.0:
                L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef, dtype=rdtype)
                As = numpy.vstack([As, L], dtype=numpy.float64)

            _, svals, vh = numpy.linalg.svd(As, full_matrices=False)
            coef_scaled = vh[-1, :]
            coef = coef_scaled / s_col

    else:
        # No moment constraints
        if ridge_lambda > 0.0:
            L = numpy.sqrt(ridge_lambda) * numpy.eye(n_coef, dtype=rdtype)
            As = numpy.vstack([As, L], dtype=numpy.float64)

        _, svals, vh = numpy.linalg.svd(As, full_matrices=False)
        coef_scaled = vh[-1, :]
        coef = coef_scaled / s_col

    full_basis = numpy.zeros((deg_z + 1, s + 1), dtype=dtype)
    for k, (i, j) in enumerate(pairs):
        full_basis[i, j] = coef[k]

    # Convert from stabilized basis back to raw monomial basis.
    # If raw_z = zp @ tz and raw_m = mp @ tm, then coefficient blocks transform
    # by
    #     C_mono = inv(tz).T @ C_basis @ inv(tm)
    # equivalently:
    #     C_mono = solve(tz.T, C_basis) solve(tm, I)
    tzr = numpy.asarray(tz, dtype=rdtype)
    tmr = numpy.asarray(tm, dtype=rdtype)
    fullr = numpy.asarray(full_basis, dtype=dtype)

    # Correct back-transform with real triangular basis transforms:
    # full_basis = Tz @ full_mono @ Tm.T
    tmp = numpy.linalg.solve(tzr, fullr)
    full = numpy.linalg.solve(tmr, tmp.transpose()).transpose()
    full = numpy.asarray(full, dtype=dtype)

    # Undo variable scaling back to original z and m
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            full[i, j] = full[i, j] / ((z_scale ** i) * (m_scale ** j))

    if normalize:
        full = _normalize_coefficients(full)

    # Diagnostic metrics
    fit_metrics = {
        's_min': float(svals[-1]),
        'gap_ratio': float(svals[-2] / svals[-1]),
        'n_small': float(int(numpy.sum(svals <= svals[0] * 1e-12))),
    }

    return full, fit_metrics


# ======
# eval P
# ======

def eval_P(z, m, coeffs, dtype=complex):

    z = numpy.asarray(z, dtype=dtype)
    m = numpy.asarray(m, dtype=dtype)
    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)

    shp = numpy.broadcast(z, m).shape
    zz = numpy.broadcast_to(z, shp).ravel()
    mm = numpy.broadcast_to(m, shp).ravel()

    zp = powers(zz, deg_z, dtype=dtype)
    mp = powers(mm, s, dtype=dtype)

    P = numpy.zeros(zz.size, dtype=dtype)
    for j in range(s + 1):
        aj = zp @ coeffs[:, j]
        P = P + aj * mp[:, j]

    return P.reshape(shp)


# ==============
# poly coef in m
# ==============

def _poly_coef_in_m(z, coeffs, dtype=complex):

    z = numpy.asarray(z, dtype=dtype).ravel()
    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)
    zp = powers(z, deg_z, dtype=dtype)

    c = numpy.empty((z.size, s + 1), dtype=dtype)
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

def _cbrt_complex(z, dtype=complex):

    z = numpy.asarray(z, dtype=dtype)
    r = numpy.abs(z)
    th = numpy.angle(z)
    return (r ** (1.0 / 3.0)) * numpy.exp(1j * th / 3.0)


# ==========
# root cubic
# ==========

def _roots_cubic(c0, c1, c2, c3, dtype=complex):

    c0 = numpy.asarray(c0, dtype=dtype)
    c1 = numpy.asarray(c1, dtype=dtype)
    c2 = numpy.asarray(c2, dtype=dtype)
    c3 = numpy.asarray(c3, dtype=dtype)

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

def eval_roots(z, coeffs, dtype=complex):

    z = numpy.asarray(z, dtype=dtype).ravel()
    c = _poly_coef_in_m(z, coeffs, dtype=dtype)

    s = int(c.shape[1] - 1)
    if s == 1:
        m = -c[:, 0] / c[:, 1]
        return m[:, None]

    if s == 2:
        return _roots_quadratic(c[:, 0], c[:, 1], c[:, 2])

    if s == 3:
        return _roots_cubic(c[:, 0], c[:, 1], c[:, 2], c[:, 3], dtype=dtype)

    roots = numpy.empty((z.size, s), dtype=dtype)
    for i in range(z.size):
        roots[i, :] = numpy.roots(c[i, ::-1])
    return roots
