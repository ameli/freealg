# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

# =======
# imports
# =======

import numpy

__all__ = ['estimate_support']


# =================
# contiguous runs
# =================


def _runs_from_mask(mask):
    """
    Convert boolean mask to contiguous index runs.
    """

    mask = numpy.asarray(mask, dtype=bool)

    runs = []
    i = 0
    n = mask.size

    while i < n:
        if not mask[i]:
            i += 1
            continue

        j = i
        while (j + 1 < n) and mask[j + 1]:
            j += 1

        runs.append((i, j))
        i = j + 1

    return runs


# ===================
# edge clustering
# ===================


def _cluster_edges(edges, x_tol):
    """
    Cluster very close edge locations.
    """

    if len(edges) == 0:
        return numpy.array([], dtype=float)

    edges = numpy.array(sorted(edges), dtype=float)
    out = [edges[0]]

    for e in edges[1:]:
        if abs(e - out[-1]) > x_tol:
            out.append(e)

    return numpy.array(out, dtype=float)


# ==========================
# simple log-score smoothing
# ==========================


def _smooth_log_score(score, width=7):
    """
    Smooth log10(score) by moving average.
    """

    score = numpy.asarray(score, dtype=float)

    if width <= 1 or score.size == 0:
        return numpy.log10(numpy.maximum(score, 1e-300))

    width = int(width)
    if width % 2 == 0:
        width += 1

    y = numpy.log10(numpy.maximum(score, 1e-300))
    pad = width // 2
    ypad = numpy.pad(y, pad, mode='edge')
    ker = numpy.ones(width, dtype=float) / float(width)

    return numpy.convolve(ypad, ker, mode='valid')


# ========================
# local peaks in an array
# ========================


def _find_local_peaks(y, i0, i1):
    """
    Return local peak indices in y[i0:i1+1].
    """

    peaks = []

    if i1 - i0 < 2:
        return peaks

    for k in range(i0 + 1, i1):
        if (y[k] >= y[k - 1]) and (y[k] >= y[k + 1]):
            peaks.append(k)

    return peaks


# =========================
# split by deep valleys
# =========================


def _split_runs_by_valley(x_grid, score_grid, runs, *,
                          log=False,
                          valley_rel=0.35,
                          smooth_width=7,
                          min_log_width_mult=1.0):
    """
    Split a run if two peaks are separated by a deep enough valley.

    In log mode, valley comparison is done in log-score and widths are measured
    in log-x.
    """

    if len(runs) == 0:
        return runs

    ys = _smooth_log_score(score_grid, width=smooth_width)

    if x_grid.size >= 2:
        if log:
            dx_ref = abs(numpy.log(float(x_grid[1])) - numpy.log(float(x_grid[0])))
        else:
            dx_ref = abs(float(x_grid[1] - x_grid[0]))
    else:
        dx_ref = 0.0

    min_width = float(min_log_width_mult) * dx_ref

    out = []

    for i0, i1 in runs:
        if i1 - i0 < 4:
            out.append((i0, i1))
            continue

        peaks = _find_local_peaks(ys, i0, i1)

        if len(peaks) <= 1:
            out.append((i0, i1))
            continue

        cut_points = []

        for pL, pR in zip(peaks[:-1], peaks[1:]):
            if pR <= pL + 1:
                continue

            seg = ys[pL + 1:pR]
            if seg.size == 0:
                continue

            kmin = pL + 1 + int(numpy.argmin(seg))
            valley = ys[kmin]
            peak_ref = min(float(ys[pL]), float(ys[pR]))

            if valley <= peak_ref + numpy.log10(float(valley_rel)):
                if log:
                    left_w = numpy.log(float(x_grid[kmin])) - \
                             numpy.log(float(x_grid[i0]))
                    right_w = numpy.log(float(x_grid[i1])) - \
                              numpy.log(float(x_grid[kmin]))
                else:
                    left_w = float(x_grid[kmin] - x_grid[i0])
                    right_w = float(x_grid[i1] - x_grid[kmin])

                if (left_w >= min_width) and (right_w >= min_width):
                    cut_points.append(kmin)

        if len(cut_points) == 0:
            out.append((i0, i1))
        else:
            a = i0
            for c in cut_points:
                out.append((a, c - 1))
                a = c + 1
            out.append((a, i1))

    return out


# ========================
# local-peak run detection
# ========================

def _runs_from_local_peaks(x_grid, score_grid, *,
                           log=True,
                           seed_drop=2.5,
                           smooth_width=7,
                           min_log_width_mult=1.0):
    """
    Build candidate runs around local peaks.

    Each peak gets expanded until the score drops by `seed_drop` decades from
    that peak, or until the score starts rising strongly toward another peak.
    """

    del log

    n = score_grid.size
    if n == 0:
        return []

    ys = _smooth_log_score(score_grid, width=smooth_width)
    peaks = _find_local_peaks(ys, 0, n - 1)

    if len(peaks) == 0:
        k = int(numpy.argmax(ys))
        peaks = [k]

    runs = []

    for kp in peaks:
        ypk = float(ys[kp])
        ythr = ypk - float(seed_drop)

        i0 = kp
        while i0 > 0 and ys[i0 - 1] >= ythr:
            i0 -= 1

        i1 = kp
        while i1 + 1 < n and ys[i1 + 1] >= ythr:
            i1 += 1

        runs.append((i0, i1))

    runs = sorted(runs)
    merged = []

    a0, b0 = runs[0]
    for a1, b1 in runs[1:]:
        if a1 <= b0 + 1:
            b0 = max(b0, b1)
        else:
            merged.append((a0, b0))
            a0, b0 = a1, b1
    merged.append((a0, b0))

    if x_grid.size >= 2:
        dlogx = abs(numpy.log(float(x_grid[1])) - numpy.log(float(x_grid[0])))
    else:
        dlogx = 0.0

    min_width = float(min_log_width_mult) * dlogx

    out = []
    for i0, i1 in merged:
        if numpy.log(float(x_grid[i1])) - numpy.log(float(x_grid[i0])) >= min_width:
            out.append((i0, i1))

    return out


# =======================
# score at a point x
# =======================


def _score_at_x(stieltjes, x, delta, log):
    """
    Evaluate support score at x.
    """

    z = complex(float(x), float(delta))
    im_val = float(numpy.imag(stieltjes.evaluate_scalar(z)))

    if log:
        floor = float(delta / (x * x + delta * delta))
        return im_val / floor
    else:
        return im_val


# ===================
# edge bisection
# ===================


def _bisect_edge(stieltjes, x_lo, x_hi, delta, thr, log=False, max_iter=60):
    """
    Refine one edge by bisection on score(x)-thr.
    """

    def f(x):
        return _score_at_x(stieltjes, x, delta, log) - thr

    a = float(x_lo)
    b = float(x_hi)
    fa = f(a)
    fb = f(b)

    if (not numpy.isfinite(fa)) or (not numpy.isfinite(fb)):
        return 0.5 * (a + b)

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    if fa * fb > 0.0:
        return 0.5 * (a + b)

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)

        if (not numpy.isfinite(fc)) or \
                (fc == 0.0) or \
                ((b - a) < 1e-14 * (1.0 + abs(c))):
            return c

        if fa * fc <= 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return 0.5 * (a + b)


# ==============================
# local adaptive edge refine
# ==============================


def _refine_run_edges(stieltjes, x_grid, score_grid, i0, i1, *,
                      delta, log, edge_drop, edge_rel,
                      n_refine=256):
    """
    Refine the two edges of a candidate run.

    Uses a local threshold based on the run peak, then only searches near the
    run boundaries.
    """

    peak_score = float(numpy.max(score_grid[i0:i1 + 1]))

    if log:
        score_edge = max(peak_score * (10.0 ** (-float(edge_drop))), 2.0)
    else:
        score_edge = max(float(edge_rel) * peak_score, 10.0 * float(delta))

    j0 = i0
    while (j0 > 0) and (score_grid[j0 - 1] > score_edge):
        j0 -= 1

    n_pts = score_grid.size
    j1 = i1
    while (j1 + 1 < n_pts) and (score_grid[j1 + 1] > score_edge):
        j1 += 1

    if j0 == 0:
        xL = float(x_grid[0])
    else:
        xa = float(x_grid[j0 - 1])
        xb = float(x_grid[j0])
        if log:
            xloc = numpy.geomspace(xa, xb, max(16, int(n_refine)))
        else:
            xloc = numpy.linspace(xa, xb, max(16, int(n_refine)))

        sloc = numpy.array(
            [_score_at_x(stieltjes, xx, delta, log) for xx in xloc],
            dtype=float
        )
        mask = numpy.isfinite(sloc - score_edge)
        if numpy.any(mask):
            idx = None
            for k in range(xloc.size - 1):
                f0 = sloc[k] - score_edge
                f1 = sloc[k + 1] - score_edge
                if numpy.isfinite(f0) and numpy.isfinite(f1) and (f0 * f1 <= 0.0):
                    idx = k
                    break
            if idx is None:
                xL = _bisect_edge(stieltjes, xa, xb, delta, score_edge, log=log)
            else:
                xL = _bisect_edge(
                    stieltjes,
                    float(xloc[idx]),
                    float(xloc[idx + 1]),
                    delta,
                    score_edge,
                    log=log
                )
        else:
            xL = 0.5 * (xa + xb)

    if j1 == n_pts - 1:
        xR = float(x_grid[-1])
    else:
        xa = float(x_grid[j1])
        xb = float(x_grid[j1 + 1])
        if log:
            xloc = numpy.geomspace(xa, xb, max(16, int(n_refine)))
        else:
            xloc = numpy.linspace(xa, xb, max(16, int(n_refine)))

        sloc = numpy.array(
            [_score_at_x(stieltjes, xx, delta, log) for xx in xloc],
            dtype=float
        )
        mask = numpy.isfinite(sloc - score_edge)
        if numpy.any(mask):
            idx = None
            for k in range(xloc.size - 1):
                f0 = sloc[k] - score_edge
                f1 = sloc[k + 1] - score_edge
                if numpy.isfinite(f0) and numpy.isfinite(f1) and (f0 * f1 <= 0.0):
                    idx = k
                    break
            if idx is None:
                xR = _bisect_edge(stieltjes, xa, xb, delta, score_edge, log=log)
            else:
                xR = _bisect_edge(
                    stieltjes,
                    float(xloc[idx]),
                    float(xloc[idx + 1]),
                    delta,
                    score_edge,
                    log=log
                )
        else:
            xR = 0.5 * (xa + xb)

    return float(xL), float(xR), float(score_edge), float(peak_score)


# ==========================
# polynomial partials at z,m
# ==========================


def _P_and_partials(coeffs, z, m):
    """
    Evaluate P(z, m) and partial derivatives for

        P(z, m) = sum_j a_j(z) m^j,

    where coeffs[:, j] stores polynomial coefficients of a_j(z) in descending
    powers of z.
    """

    coeffs = numpy.asarray(coeffs)
    if coeffs.ndim != 2:
        raise ValueError('"coeffs" must be a 2D array.')

    deg_z, s1 = coeffs.shape
    s = s1 - 1

    a = numpy.empty(s + 1, dtype=complex)
    da = numpy.empty(s + 1, dtype=complex)

    for j in range(s + 1):
        c = numpy.asarray(coeffs[:, j], dtype=complex)
        a[j] = numpy.polyval(c, z)
        if deg_z > 1:
            dc = numpy.polyder(c)
            da[j] = numpy.polyval(dc, z)
        else:
            da[j] = 0.0 + 0.0j

    P = 0.0 + 0.0j
    Pz = 0.0 + 0.0j
    Pm = 0.0 + 0.0j
    Pzm = 0.0 + 0.0j
    Pmm = 0.0 + 0.0j

    for j in range(s + 1):
        mj = m ** j
        P += a[j] * mj
        Pz += da[j] * mj

    for j in range(1, s + 1):
        mj1 = m ** (j - 1)
        Pm += j * a[j] * mj1
        Pzm += j * da[j] * mj1

    for j in range(2, s + 1):
        Pmm += j * (j - 1) * a[j] * (m ** (j - 2))

    return P, Pz, Pm, Pzm, Pmm


# ===========
# newton edge
# ===========


def _newton_edge(coeffs, x0, m0, tol=1e-12, max_iter=50):
    """
    Solve the real branch-point system

        P(x, m) = 0,
        dP/dm(x, m) = 0

    by Newton iteration, initialized from (x0, m0).
    """

    x = float(x0)
    m = float(m0)

    for _ in range(int(max_iter)):
        try:
            P, Pz, Pm, Pzm, Pmm = _P_and_partials(coeffs, x + 0.0j, m)
        except Exception:
            return float(x0), float(m0), False

        f0 = float(numpy.real(P))
        f1 = float(numpy.real(Pm))

        j00 = float(numpy.real(Pz))
        j01 = float(numpy.real(Pm))
        j10 = float(numpy.real(Pzm))
        j11 = float(numpy.real(Pmm))

        det = j00 * j11 - j01 * j10
        if (not numpy.isfinite(det)) or (abs(det) < 1e-30):
            return x, m, False

        dx = (-f0 * j11 + f1 * j01) / det
        dm = (-j00 * f1 + j10 * f0) / det

        if (not numpy.isfinite(dx)) or (not numpy.isfinite(dm)):
            return x, m, False

        x += dx
        m += dm

        if abs(dx) + abs(dm) < float(tol):
            return x, m, True

    return x, m, False


# ==============================
# optional algebraic edge snap
# ==============================


def _snap_edges_newton(coeffs, stieltjes, edges, delta, x_min, x_max, *,
                       tol=1e-12,
                       max_iter=50,
                       accept_rel=5e-2,
                       accept_abs=None):
    """
    Optionally snap refined edges to algebraic branch points.

    A Newton proposal is accepted only if it is finite, inside the scan range,
    and not too far from the pre-Newton edge. This keeps the correction local.
    """

    edges = numpy.asarray(edges, dtype=float)
    if edges.size == 0:
        return edges, numpy.zeros(0, dtype=bool)

    scale = max(1.0, abs(x_max - x_min), abs(x_min), abs(x_max))
    if accept_abs is None:
        accept_abs = float(accept_rel) * scale

    snapped = []
    ok_mask = []

    for x0 in edges:
        try:
            m0 = float(numpy.real(stieltjes.evaluate_scalar(complex(x0, delta))))
            xe, _, ok = _newton_edge(
                coeffs,
                x0,
                m0,
                tol=tol,
                max_iter=max_iter
            )
        except Exception:
            xe = float(x0)
            ok = False

        accept = bool(ok)
        if accept:
            if (not numpy.isfinite(xe)) or (xe < x_min) or (xe > x_max):
                accept = False
            if abs(float(xe) - float(x0)) > float(accept_abs):
                accept = False

        if accept:
            snapped.append(float(xe))
            ok_mask.append(True)
        else:
            snapped.append(float(x0))
            ok_mask.append(False)

    return numpy.array(snapped, dtype=float), numpy.array(ok_mask, dtype=bool)


# ======================
# merge tiny micro-gaps
# ======================


def _merge_micro_gaps(est_supp, x_grid, log):
    """
    Merge only tiny scan-resolution gaps between adjacent intervals.
    """

    if len(est_supp) < 2 or x_grid.size < 2:
        return est_supp

    merged = []
    a0, b0 = est_supp[0]

    if log:
        gap_tol = 2.0 * abs(numpy.log(float(x_grid[1])) - numpy.log(float(x_grid[0])))
        for a1, b1 in est_supp[1:]:
            gap = numpy.log(float(a1)) - numpy.log(float(b0))
            if gap <= gap_tol:
                b0 = max(float(b0), float(b1))
            else:
                merged.append((float(a0), float(b0)))
                a0, b0 = float(a1), float(b1)
    else:
        gap_tol = 2.0 * abs(float(x_grid[1] - x_grid[0]))
        for a1, b1 in est_supp[1:]:
            gap = float(a1 - b0)
            if gap <= gap_tol:
                b0 = max(float(b0), float(b1))
            else:
                merged.append((float(a0), float(b0)))
                a0, b0 = float(a1), float(b1)

    merged.append((float(a0), float(b0)))
    return merged


# ================
# estimate support
# ================


def estimate_support(coeffs, stieltjes, x_min, x_max, n_scan=1024, log=False,
                     delta=None, thr_rel=1e-4, weak_thr_factor=1e-2,
                     min_log_width_mult=1.0, return_info=True, **kwargs):
    """
    Estimate support intervals from the fitted Stieltjes transform.

    Parameters
    ----------
    coeffs : array_like
        Polynomial coefficients. Used optionally for algebraic Newton edge
        snapping.

    stieltjes : object
        StieltjesPoly-like object with ``__call__`` and
        ``evaluate_scalar(z)`` methods.

    x_min, x_max : float
        Scan range.

    n_scan : int, default=1024
        Number of coarse scan points.

    log : bool, default=False
        If True, scan in geometric x-grid and use the normalized score
        ``Im(m) / floor``.

    delta : float, default=None
        Imaginary offset. If None, it is set automatically.

    thr_rel : float, default=1e-4
        Global relative threshold. Used only in linear mode.

    weak_thr_factor : float, default=1e-2
        Kept only for API compatibility.

    min_log_width_mult : float, default=1.0
        Minimal run width in units of coarse-grid log spacing for log mode.

    return_info : bool, default=True
        Return diagnostics dictionary.

    Other Parameters
    ----------------
    seed_drop : float, default=2.5
        In log mode, peaks seed candidate runs down to this many decades
        below each peak.

    edge_drop : float, default=1.0
        In log mode, edge of a run is placed where the normalized score drops
        by this many decades below the run peak.

    edge_rel : float, default=0.1
        Linear-mode local edge threshold relative to the run peak.

    valley_rel : float, default=0.35
        Split a run if the valley between two peaks is below this fraction
        of the smaller adjacent peak.

    smooth_width : int, default=7
        Smoothing width used only for peak/valley logic.

    n_refine : int, default=256
        Local edge refinement samples.

    edge_x_cluster_tol : float, default=1e-8 * scale
        Cluster nearby edge locations.

    refine : bool, default=True
        If True and coeffs is provided, apply a final Newton snap of each
        refined edge to the algebraic branch-point system.

    newton_tol : float, default=1e-12
        Newton convergence tolerance for algebraic edge snapping.

    newton_max_iter : int, default=50
        Maximum Newton iterations for algebraic edge snapping.

    newton_accept_rel : float, default=5e-2
        Accept a Newton correction only if it stays within this fraction of the
        problem scale from the pre-Newton edge.

    newton_accept_abs : float, optional
        Absolute acceptance radius for the Newton correction. If None, it is
        set from ``newton_accept_rel``.

    merge_micro_gaps : bool, default=True
        Merge only tiny scan-resolution artificial gaps at the very end.

    Notes
    -----
    This version keeps v3's coarse-topology plus local-edge-refinement design,
    then optionally adds a final algebraic Newton snap and a final micro-gap
    merge.
    """

    del weak_thr_factor

    x_min = float(x_min)
    x_max = float(x_max)
    n_scan = int(n_scan)

    if x_min <= 0.0 and log:
        raise ValueError('"x_min" must be positive when log=True.')

    scale = max(1.0, abs(x_max - x_min), abs(x_min), abs(x_max))
    if delta is None:
        delta = 1e-6 * scale
    delta = float(delta)

    seed_drop = float(kwargs.get('seed_drop', 2.5))
    edge_drop = float(kwargs.get('edge_drop', 1.0))
    edge_rel = float(kwargs.get('edge_rel', 0.1))
    valley_rel = float(kwargs.get('valley_rel', 0.35))
    smooth_width = int(kwargs.get('smooth_width', 7))
    n_refine = int(kwargs.get('n_refine', 256))
    thr_abs = kwargs.get('thr_abs', None)

    refine = bool(kwargs.get('refine', True))
    newton_tol = float(kwargs.get('newton_tol', 1e-12))
    newton_max_iter = int(kwargs.get('newton_max_iter', 50))
    newton_accept_rel = float(kwargs.get('newton_accept_rel', 5e-2))
    newton_accept_abs = kwargs.get('newton_accept_abs', None)
    merge_micro_gaps = bool(kwargs.get('merge_micro_gaps', True))

    if log:
        x_grid = numpy.geomspace(x_min, x_max, n_scan)
    else:
        x_grid = numpy.linspace(x_min, x_max, n_scan)

    z_grid = x_grid + 1j * delta
    m_grid = stieltjes(z_grid)
    im_grid = numpy.imag(m_grid)

    if log:
        floor_grid = delta / (x_grid * x_grid + delta * delta)
        score_grid = im_grid / floor_grid
    else:
        floor_grid = None
        score_grid = im_grid

    finite = numpy.isfinite(score_grid)
    if not numpy.any(finite):
        info = {
            'x_grid': x_grid,
            'm_grid': m_grid,
            'im_grid': im_grid,
            'floor_grid': floor_grid,
            'score_grid': score_grid,
            'runs': [],
            'run_info': [],
            'edges_pre_newton': numpy.array([], dtype=float),
            'edges_post_newton': numpy.array([], dtype=float),
            'newton_ok': numpy.array([], dtype=bool),
        }
        return ([], info) if return_info else []

    max_score = float(numpy.nanmax(score_grid))

    if log:
        runs = _runs_from_local_peaks(
            x_grid, score_grid,
            log=True,
            seed_drop=seed_drop,
            smooth_width=smooth_width,
            min_log_width_mult=min_log_width_mult
        )

        runs = _split_runs_by_valley(
            x_grid, score_grid, runs,
            log=True,
            valley_rel=valley_rel,
            smooth_width=smooth_width,
            min_log_width_mult=min_log_width_mult
        )

        score_thr = 2.0
    else:
        score_thr = max(float(thr_rel) * max_score, 10.0 * delta)
        if thr_abs is not None:
            score_thr = max(score_thr, float(thr_abs))

        mask = numpy.isfinite(score_grid) & (score_grid > score_thr)
        runs = _runs_from_mask(mask)

    edges = []
    run_info = []

    for i0, i1 in runs:
        xL, xR, score_edge, peak_score = _refine_run_edges(
            stieltjes, x_grid, score_grid, i0, i1,
            delta=delta, log=log, edge_drop=edge_drop, edge_rel=edge_rel,
            n_refine=n_refine
        )

        if xR > xL:
            edges.extend([xL, xR])
            run_info.append({
                'run_idx': (int(i0), int(i1)),
                'run_x': (float(x_grid[i0]), float(x_grid[i1])),
                'peak_score': float(peak_score),
                'score_edge': float(score_edge),
                'edges_local': (float(xL), float(xR)),
            })

    edge_x_cluster_tol = float(kwargs.get('edge_x_cluster_tol', 1e-8 * scale))
    edges_pre_newton = _cluster_edges(edges, edge_x_cluster_tol)

    if refine and (coeffs is not None) and (edges_pre_newton.size > 0):
        edges_post_newton_raw, newton_ok = _snap_edges_newton(
            coeffs,
            stieltjes,
            edges_pre_newton,
            delta,
            x_min,
            x_max,
            tol=newton_tol,
            max_iter=newton_max_iter,
            accept_rel=newton_accept_rel,
            accept_abs=newton_accept_abs,
        )
        edges = _cluster_edges(edges_post_newton_raw, edge_x_cluster_tol)
    else:
        edges = edges_pre_newton.copy()
        newton_ok = numpy.zeros(edges_pre_newton.size, dtype=bool)

    edges.sort()
    est_supp = []

    for k in range(0, edges.size - 1, 2):
        a = float(edges[k])
        b = float(edges[k + 1])
        if b > a:
            est_supp.append((a, b))

    if merge_micro_gaps:
        est_supp = _merge_micro_gaps(est_supp, x_grid, log)

    info = {
        'x_grid': x_grid,
        'm_grid': m_grid,
        'im_grid': im_grid,
        'floor_grid': floor_grid,
        'score_grid': score_grid,
        'max_score': max_score,
        'score_thr': score_thr,
        'runs': runs,
        'run_info': run_info,
        'edges_pre_newton': edges_pre_newton,
        'edges_post_newton': edges,
        'newton_ok': newton_ok,
        'delta': delta,
    }

    if return_info:
        return est_supp, info
    else:
        return est_supp
