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
import numpy.polynomial.polynomial as poly

__all__ = ['compute_support']


# ======================
# poly coeffs in m and z
# ======================

def _poly_coeffs_in_m_at_z(a_coeffs, z):

    s = a_coeffs.shape[1] - 1
    a = numpy.empty(s + 1, dtype=numpy.complex128)
    for j in range(s + 1):
        a[j] = poly.polyval(z, a_coeffs[:, j])
    return a


# ===============
# roots poly in m
# ===============

def _roots_poly_in_m(c_asc, tol=0.0):

    c = numpy.asarray(c_asc, dtype=numpy.complex128).ravel()
    if c.size <= 1:
        return numpy.array([], dtype=numpy.complex128)

    k = c.size - 1
    while k > 0 and abs(c[k]) <= tol:
        k -= 1
    c = c[:k + 1]
    if c.size <= 1:
        return numpy.array([], dtype=numpy.complex128)

    return numpy.roots(c[::-1])


# ================
# dPdm coeffs at z
# ================

def _dPdm_coeffs_at_z(a_coeffs, z):

    a = _poly_coeffs_in_m_at_z(a_coeffs, z)
    s = a.size - 1
    if s <= 0:
        return numpy.array([0.0 + 0.0j], dtype=numpy.complex128)
    d = numpy.empty(s, dtype=numpy.complex128)
    for j in range(1, s + 1):
        d[j - 1] = j * a[j]
    return d


# ==============
# P and partials
# ==============

def _P_and_partials(a_coeffs, z, m):

    s = a_coeffs.shape[1] - 1

    a = numpy.empty(s + 1, dtype=numpy.complex128)
    da = numpy.empty(s + 1, dtype=numpy.complex128)
    for j in range(s + 1):
        a[j] = poly.polyval(z, a_coeffs[:, j])
        da[j] = poly.polyval(z, poly.polyder(a_coeffs[:, j]))

    mpow = 1.0 + 0.0j
    P = 0.0 + 0.0j
    Pz = 0.0 + 0.0j
    for j in range(s + 1):
        P += a[j] * mpow
        Pz += da[j] * mpow
        mpow *= m

    Pm = 0.0 + 0.0j
    Pmm = 0.0 + 0.0j
    Pzm = 0.0 + 0.0j
    for j in range(1, s + 1):
        Pm += j * a[j] * (m ** (j - 1))
        Pzm += j * da[j] * (m ** (j - 1))
    for j in range(2, s + 1):
        Pmm += j * (j - 1) * a[j] * (m ** (j - 2))

    return P, Pz, Pm, Pzm, Pmm, a


# ===========
# newton edge
# ===========

def _newton_edge(a_coeffs, x0, m0, tol=1e-12, max_iter=50):

    x = float(x0)
    m = float(m0)

    for _ in range(max_iter):
        z = x + 0.0j
        P, Pz, Pm, Pzm, Pmm, _ = _P_and_partials(a_coeffs, z, m)

        f0 = float(numpy.real(P))
        f1 = float(numpy.real(Pm))

        j00 = float(numpy.real(Pz))
        j01 = float(numpy.real(Pm))
        j10 = float(numpy.real(Pzm))
        j11 = float(numpy.real(Pmm))

        det = j00 * j11 - j01 * j10
        if det == 0.0 or (not numpy.isfinite(det)):
            return x, m, False

        dx = (-f0 * j11 + f1 * j01) / det
        dm = (-j00 * f1 + j10 * f0) / det

        x += dx
        m += dm

        if abs(dx) + abs(dm) < tol:
            return x, m, True

    return x, m, False


# =============
# cluster edges
# =============

def _cluster_edges(edges, x_tol):

    if len(edges) == 0:
        return numpy.array([], dtype=float)

    edges = numpy.array(sorted(edges), dtype=float)
    out = [edges[0]]
    for e in edges[1:]:
        if abs(e - out[-1]) > x_tol:
            out.append(e)
    return numpy.array(out, dtype=float)


# =======================
# pick physical root at z
# =======================

def _pick_physical_root_at_z(a_coeffs, z, im_sign=+1):

    a = _poly_coeffs_in_m_at_z(a_coeffs, z)
    r = _roots_poly_in_m(a)
    if r.size == 0:
        return numpy.nan + 1j * numpy.nan

    w_ref = -1.0 / z
    idx = int(numpy.argmin(numpy.abs(r - w_ref)))
    w = r[idx]

    # optional strictness: if it violates Herglotz, declare failure
    if not numpy.isfinite(w.real) or not numpy.isfinite(w.imag):
        return w
    if (im_sign * w.imag) <= 0.0:
        return w

    return w


# ===============
# compute support
# ===============

def compute_support(a_coeffs,
                    x_min,
                    x_max,
                    n_scan=4000,
                    y_eps=1e-3,
                    im_sign=+1,
                    root_tol=0.0,
                    edge_rel_tol=1e-6,
                    edge_x_cluster_tol=1e-3,
                    newton_tol=1e-12):
    """
    Fast support from fitted polynomial using branch-point system P=0, Pm=0.

    Returns
    -------
    support : list of (a,b)
    info    : dict (edges, rel_res_curve, etc.)
    """

    a_coeffs = numpy.asarray(a_coeffs)
    x_grid = numpy.linspace(float(x_min), float(x_max), int(n_scan))

    # For each x, find best real critical point m (Pm=0) minimizing rel
    # residual.
    rel = numpy.full(x_grid.size, numpy.inf, dtype=float)
    m_star = numpy.full(x_grid.size, numpy.nan, dtype=float)

    for i, x in enumerate(x_grid):
        z = x + 0.0j
        dcoef = _dPdm_coeffs_at_z(a_coeffs, z)
        mr = _roots_poly_in_m(dcoef, tol=root_tol)

        best = numpy.inf
        best_m = numpy.nan

        for w in mr:
            # accept nearly-real roots; numerical roots can have small imag
            # part
            if abs(w.imag) > 1e-6 * (1.0 + abs(w.real)):
                continue
            m = float(w.real)
            P, _, _, _, _, a = _P_and_partials(a_coeffs, z, m)

            denom = 1.0
            am = 1.0
            for j in range(a.size):
                denom += abs(a[j]) * abs(am)
                am *= m

            r = abs(numpy.real(P)) / denom
            if numpy.isfinite(r) and r < best:
                best = float(r)
                best_m = m

        rel[i] = best
        m_star[i] = best_m

    # Pick candidate edges as local minima of rel(x), below an automatic scale.
    rel_f = rel[numpy.isfinite(rel)]
    if rel_f.size == 0:
        return [], {"edges": numpy.array([], dtype=float), "n_edges": 0}

    med = float(numpy.median(rel_f))
    min_rel = float(numpy.min(rel_f))

    # accept local minima up to a factor above the best one, but never abov
    # background scale
    thr = min(0.1 * med, max(float(edge_rel_tol), 1e4 * min_rel))

    edges0 = []
    seeds = []

    for i in range(1, x_grid.size - 1):
        if not numpy.isfinite(rel[i]):
            continue
        if rel[i] <= rel[i - 1] and rel[i] <= rel[i + 1] and rel[i] < thr and \
                numpy.isfinite(m_star[i]):
            edges0.append(float(x_grid[i]))
            seeds.append((float(x_grid[i]), float(m_star[i])))

    # Refine each seed by 2D Newton (x,m)
    edges = []
    for x0, m0 in seeds:
        xe, me, ok = _newton_edge(a_coeffs, x0, m0, tol=newton_tol)
        if ok and numpy.isfinite(xe) and numpy.isfinite(me):
            edges.append(float(xe))

    edges = _cluster_edges(edges, edge_x_cluster_tol)
    edges.sort()

    # Build support by testing midpoints between consecutive real edges
    support = []
    m_im_tol = 1e-10

    for i in range(edges.size - 1):
        a = float(edges[i])
        b = float(edges[i + 1])
        if b <= a:
            continue

        xmid = 0.5 * (a + b)

        # roots of P(xmid, m) with real coefficients
        a_m = _poly_coeffs_in_m_at_z(a_coeffs, xmid + 0.0j)
        r = _roots_poly_in_m(a_m, tol=root_tol)

        # interval is support iff there exists a non-real root (complex pair)
        if numpy.any(numpy.abs(numpy.imag(r)) > m_im_tol):
            support.append((a, b))

    info = {
        "edges": edges,
        "n_edges": int(edges.size),
        "support": support,
        "n_support": int(len(support)),
        "x_grid": x_grid,
        "rel": rel,
        "thr": float(thr),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "n_scan": int(n_scan),
        "y_eps": float(y_eps),
    }

    return support, info
