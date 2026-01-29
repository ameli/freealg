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
from ._homotopy5 import StieltjesPoly

__all__ = ['compute_support']


# =====================
# poly coeffs in m at z
# =====================

def _poly_coeffs_in_m_at_z(a_coeffs, z):
    s = a_coeffs.shape[1] - 1
    a = numpy.empty(s + 1, dtype=numpy.complex128)
    for j in range(s + 1):
        a[j] = poly.polyval(z, a_coeffs[:, j])
    return a


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

    return P, Pz, Pm, Pzm, Pmm


# ===========
# newton edge
# ===========

def _newton_edge(a_coeffs, x0, m0, tol=1e-12, max_iter=50):
    x = float(x0)
    m = float(m0)

    for _ in range(max_iter):
        z = x + 0.0j
        P, Pz, Pm, Pzm, Pmm = _P_and_partials(a_coeffs, z, m)

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


# ===========
# bisect edge
# ===========

def _bisect_edge(stieltjes_poly, x_lo, x_hi, eta, im_thr, max_iter=60):
    z_lo = x_lo + 1j * eta
    z_hi = x_hi + 1j * eta
    f_lo = float(numpy.imag(stieltjes_poly.evaluate_scalar(z_lo)) - im_thr)
    f_hi = float(numpy.imag(stieltjes_poly.evaluate_scalar(z_hi)) - im_thr)

    if (not numpy.isfinite(f_lo)) or (not numpy.isfinite(f_hi)):
        return 0.5 * (x_lo + x_hi)
    if f_lo == 0.0:
        return float(x_lo)
    if f_hi == 0.0:
        return float(x_hi)
    if f_lo * f_hi > 0.0:
        return 0.5 * (x_lo + x_hi)

    a = float(x_lo)
    b = float(x_hi)
    fa = f_lo
    # fb = f_hi

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        z_c = c + 1j * eta
        fc = float(numpy.imag(stieltjes_poly.evaluate_scalar(z_c)) - im_thr)
        if not numpy.isfinite(fc):
            return c
        if fc == 0.0 or (b - a) < 1e-14 * (1.0 + abs(c)):
            return c
        if fa * fc <= 0.0:
            b = c
            # fb = fc
        else:
            a = c
            fa = fc

    return 0.5 * (a + b)


# ===============
# compute support
# ===============

def compute_support(a_coeffs, x_min, x_max, n_scan=4000, **kwargs):

    a_coeffs = numpy.asarray(a_coeffs, dtype=numpy.complex128)

    x_min = float(x_min)
    x_max = float(x_max)
    n_scan = int(n_scan)

    scale = max(1.0, abs(x_max - x_min), abs(x_min), abs(x_max))
    eta = kwargs.get('eta', None)
    if eta is None:
        eta = 1e-6 * scale
    eta = float(eta)

    vopt = {
        'lam_space': 1.0,
        'lam_asym': 1.0,
        'lam_tiny_im': 200.0,
        'tiny_im': 0.5 * eta,
        'tol_im': 1e-14,
    }
    vopt.update(kwargs.get('viterbi_opt', {}) or {})
    stieltjes = StieltjesPoly(a_coeffs, viterbi_opt=vopt)

    x_grid = numpy.linspace(x_min, x_max, n_scan)
    z_grid = x_grid + 1j * eta
    m_grid = stieltjes(z_grid)
    im_grid = numpy.imag(m_grid)

    max_im = float(numpy.nanmax(im_grid)) \
        if numpy.any(numpy.isfinite(im_grid)) else 0.0

    thr_rel = float(kwargs.get('thr_rel', 1e-3))
    thr_abs = kwargs.get('thr_abs', None)

    im_thr = thr_rel * max_im
    im_thr = max(im_thr, 10.0 * eta)
    if thr_abs is not None:
        im_thr = max(im_thr, float(thr_abs))

    mask = numpy.isfinite(im_grid) & (im_grid > im_thr)

    runs = []
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < mask.size and mask[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1

    edges = []
    for i0, i1 in runs:
        if i0 == 0 or i1 == mask.size - 1:
            continue

        xL = _bisect_edge(stieltjes, x_grid[i0 - 1], x_grid[i0], eta, im_thr)
        xR = _bisect_edge(stieltjes, x_grid[i1], x_grid[i1 + 1], eta, im_thr)

        edges.append(float(xL))
        edges.append(float(xR))

    edge_x_cluster_tol = float(kwargs.get('edge_x_cluster_tol', 1e-8 * scale))
    edges = _cluster_edges(edges, edge_x_cluster_tol)

    refine = bool(kwargs.get('refine', True))
    if refine and edges.size > 0:
        newton_tol = float(kwargs.get('newton_tol', 1e-12))
        edges_ref = []
        for x0 in edges:
            m0 = float(numpy.real(stieltjes.evaluate_scalar(x0 + 1j * eta)))
            xe, _, ok = _newton_edge(a_coeffs, x0, m0, tol=newton_tol)
            edges_ref.append(float(xe)
                             if ok and numpy.isfinite(xe) else float(x0))
        edges = _cluster_edges(edges_ref, edge_x_cluster_tol)

    edges.sort()
    support = []
    for k in range(0, edges.size - 1, 2):
        a = float(edges[k])
        b = float(edges[k + 1])
        if b > a:
            support.append((a, b))

    info = {
        'x_grid': x_grid,
        'eta': eta,
        'm_grid': m_grid,
        'im_grid': im_grid,
        'im_thr': float(im_thr),
        'edges': edges,
        'support': support,
        'x_min': x_min,
        'x_max': x_max,
        'n_scan': n_scan,
        'scale': scale,
    }

    return support, info
