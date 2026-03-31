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

from __future__ import annotations
import numpy

__all__ = ["estimate_support"]


# =============
# run from mask
# =============

def _runs_from_mask(mask):
    """
    Convert boolean mask to contiguous runs.
    """

    mask = numpy.asarray(mask, dtype=bool)
    runs = []
    n = mask.size
    i = 0

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


# ==============
# moving average
# ==============

def _moving_average(y, width):
    """
    Centered moving average with edge padding.
    """

    y = numpy.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()

    width = int(width)
    if width <= 1:
        return y.copy()
    if width % 2 == 0:
        width += 1
    if width > n:
        width = n if (n % 2 == 1) else max(1, n - 1)
    if width <= 1:
        return y.copy()

    pad = width // 2
    ypad = numpy.pad(y, pad, mode="edge")
    ker = numpy.ones(width, dtype=float) / float(width)
    return numpy.convolve(ypad, ker, mode="valid")


# =================
# auto smooth width
# =================

def _auto_smooth_width(n_scan):
    """
    Mild smoothing width from coarse scan size.
    """

    width = max(5, int(round(int(n_scan) / 256.0)))
    if width % 2 == 0:
        width += 1
    return width


# ==============
# otsu threshold
# ==============

def _otsu_threshold(y, n_bins=256):
    """
    Otsu threshold for finite 1D data.
    """

    y = numpy.asarray(y, dtype=float)
    y = y[numpy.isfinite(y)]

    if y.size == 0:
        return 0.0, {"threshold_method": "empty"}

    y_min = float(numpy.min(y))
    y_max = float(numpy.max(y))

    if (not numpy.isfinite(y_min)) or (not numpy.isfinite(y_max)):
        return 0.0, {"threshold_method": "nonfinite"}

    if y_max <= y_min + 1e-15:
        thr = 0.5 * (y_min + y_max)
        return thr, {
            "threshold_method": "degenerate",
            "threshold_score": 0.0,
        }

    hist, edges = numpy.histogram(y, bins=int(n_bins), range=(y_min, y_max))
    hist = hist.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    weight1 = numpy.cumsum(hist)
    weight2 = numpy.cumsum(hist[::-1])[::-1]
    mean1 = numpy.cumsum(hist * centers)
    mean2 = numpy.cumsum((hist * centers)[::-1])[::-1]

    valid = (weight1[:-1] > 0.0) & (weight2[1:] > 0.0)
    if not numpy.any(valid):
        thr = 0.5 * (y_min + y_max)
        return thr, {
            "threshold_method": "midpoint",
            "threshold_score": 0.0,
        }

    mu1 = numpy.zeros(hist.size - 1, dtype=float)
    mu2 = numpy.zeros(hist.size - 1, dtype=float)
    mu1[valid] = mean1[:-1][valid] / weight1[:-1][valid]
    mu2[valid] = mean2[1:][valid] / weight2[1:][valid]

    sigma_b2 = numpy.full(hist.size - 1, -numpy.inf, dtype=float)
    sigma_b2[valid] = weight1[:-1][valid] * weight2[1:][valid] * \
        (mu1[valid] - mu2[valid]) ** 2

    idx = int(numpy.argmax(sigma_b2))
    thr = float(0.5 * (centers[idx] + centers[idx + 1]))

    return thr, {
        "threshold_method": "otsu",
        "threshold_score": float(sigma_b2[idx]),
        "hist": hist,
        "bin_centers": centers,
    }


# =======
# im at x
# =======

def _im_at_x(stieltjes, x, delta):
    """
    Imaginary part of Stieltjes transform at x+i*delta.
    """

    z = complex(float(x), float(delta))
    return float(numpy.imag(stieltjes.evaluate_scalar(z)))


# ==========
# score at x
# ==========

def _score_at_x(stieltjes, x, delta, log):
    """
    Support score at x.
    """

    im_val = _im_at_x(stieltjes, x, delta)

    if log:
        floor = float(delta / (x * x + delta * delta))
        if floor <= 0.0:
            return 0.0
        return im_val / floor

    return im_val


# =================
# signal from score
# =================

def _signal_from_score(score, log):
    """
    Threshold signal from score.
    """

    score = numpy.asarray(score, dtype=float)
    if log:
        return numpy.log10(numpy.maximum(score, 1e-300))
    return score


# ===========
# signal at x
# ===========

def _signal_at_x(stieltjes, x, delta, log):
    """
    Threshold signal at x.
    """

    score = _score_at_x(stieltjes, x, delta, log)
    if log:
        return float(numpy.log10(max(score, 1e-300)))
    return float(score)


# =======================
# bracketed newton signal
# =======================

def _bracketed_newton_signal(stieltjes, x_lo, x_hi, delta, thr, log,
                             max_iter=40, tol=1e-12):
    """
    Refine one threshold crossing by safeguarded Newton.

    The root is for F(x) = signal(x) - thr, constrained to remain inside the
    original bracket. If Newton loses reliability, the step falls back to
    bisection. This keeps the refinement local and prevents jumping across a
    bulk.
    """

    def f(x):
        return _signal_at_x(stieltjes, x, delta, log) - thr

    a = float(min(x_lo, x_hi))
    b = float(max(x_lo, x_hi))
    fa = f(a)
    fb = f(b)

    if (not numpy.isfinite(fa)) or (not numpy.isfinite(fb)):
        return numpy.sqrt(a * b) if log else 0.5 * (a + b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return numpy.sqrt(a * b) if log else 0.5 * (a + b)

    x = numpy.sqrt(a * b) if log else 0.5 * (a + b)

    for _ in range(int(max_iter)):
        fx = f(x)
        if (not numpy.isfinite(fx)):
            x_new = numpy.sqrt(a * b) if log else 0.5 * (a + b)
        else:
            h = max(1e-12 * (1.0 + abs(x)), 1e-8 * max(abs(b - a), 1e-15))
            xl = max(a, x - h)
            xr = min(b, x + h)
            if xr <= xl:
                x_new = numpy.sqrt(a * b) if log else 0.5 * (a + b)
            else:
                fl = f(xl)
                fr = f(xr)
                denom = xr - xl
                df = (fr - fl) / denom if denom > 0.0 else numpy.nan
                if (not numpy.isfinite(df)) or (abs(df) < 1e-14):
                    x_new = numpy.sqrt(a * b) if log else 0.5 * (a + b)
                else:
                    x_try = x - fx / df
                    if (x_try <= a) or (x_try >= b) or \
                            (not numpy.isfinite(x_try)):
                        x_new = numpy.sqrt(a * b) if log else 0.5 * (a + b)
                    else:
                        x_new = float(x_try)

        fx_new = f(x_new)
        if (not numpy.isfinite(fx_new)):
            x_new = numpy.sqrt(a * b) if log else 0.5 * (a + b)
            fx_new = f(x_new)

        if fa * fx_new <= 0.0:
            b = x_new
            fb = fx_new
        else:
            a = x_new
            fa = fx_new

        x = x_new

        width = abs(numpy.log(b / a)) if log else abs(b - a)
        if width <= tol * (1.0 + abs(x)):
            break
        if abs(fx_new) <= 1e-12:
            break

    return float(x)


# ================
# filter tiny runs
# ================

def _filter_tiny_runs(runs, x_grid, log, min_points=2,
                      min_log_width_mult=1.0):
    """
    Reject only scan-resolution tiny runs.
    """

    if len(runs) == 0:
        return []

    if x_grid.size >= 2:
        if log:
            dref = abs(numpy.log(float(x_grid[1])) -
                       numpy.log(float(x_grid[0])))
        else:
            dref = abs(float(x_grid[1] - x_grid[0]))
    else:
        dref = 0.0

    out = []
    for i0, i1 in runs:
        if (i1 - i0 + 1) < int(min_points):
            continue

        if log:
            width = numpy.log(float(x_grid[i1])) - numpy.log(float(x_grid[i0]))
            if width < float(min_log_width_mult) * dref:
                continue

        out.append((int(i0), int(i1)))

    return out


# ===================
# merge adjacent runs
# ===================

def _merge_adjacent_runs(runs, x_grid, log, merge_threshold):
    """
    Merge adjacent runs separated by a small gap.
    """

    if len(runs) <= 1:
        return list(runs)

    thr = float(merge_threshold)
    if thr <= 0.0:
        return list(runs)

    out = [tuple(runs[0])]

    for i0, i1 in runs[1:]:
        a0, a1 = out[-1]
        left = float(x_grid[a1])
        right = float(x_grid[i0])

        if log:
            gap = numpy.log(right) - numpy.log(left)
        else:
            gap = right - left

        if gap <= thr:
            out[-1] = (a0, i1)
        else:
            out.append((i0, i1))

    return out


# ============
# resplit runs
# ============

def _resplit_runs(stieltjes, runs, x_grid, delta, thr, log,
                  resplit_density=16, min_log_width_mult=1.0):
    """
    Densify inside each coarse run and split if a true dip crosses below
    threshold.

    This is aimed at touching bulks that a coarse grid can miss. It is not a
    peak-growth heuristic; it only checks whether the threshold classifier
    itself disconnects when sampled more densely.
    """

    if len(runs) == 0 or int(resplit_density) <= 1:
        return list(runs)

    out = []

    for i0, i1 in runs:
        xa = float(x_grid[i0])
        xb = float(x_grid[i1])
        if xb <= xa:
            continue

        n_local = max(33, int(resplit_density) * max(2, i1 - i0 + 1))
        if log:
            x_local = numpy.geomspace(xa, xb, n_local)
        else:
            x_local = numpy.linspace(xa, xb, n_local)

        sig_local = numpy.array([
            _signal_at_x(stieltjes, xx, delta, log)
            for xx in x_local
        ], dtype=float)

        mask_local = numpy.isfinite(sig_local) & (sig_local > float(thr))
        runs_local = _runs_from_mask(mask_local)
        runs_local = _filter_tiny_runs(
            runs_local,
            x_local,
            log,
            min_points=2,
            min_log_width_mult=min_log_width_mult)

        if len(runs_local) <= 1:
            out.append((i0, i1))
            continue

        for j0, j1 in runs_local:
            xL = float(x_local[j0])
            xR = float(x_local[j1])
            iL = int(numpy.searchsorted(x_grid, xL, side="left"))
            iR = int(numpy.searchsorted(x_grid, xR, side="right") - 1)
            iL = min(max(iL, i0), i1)
            iR = min(max(iR, i0), i1)
            if iR >= iL:
                out.append((iL, iR))

    out.sort(key=lambda t: (t[0], t[1]))

    # remove duplicates / overlaps conservatively
    merged = []
    for i0, i1 in out:
        if not merged:
            merged.append((i0, i1))
            continue
        a0, a1 = merged[-1]
        if i0 <= a1:
            merged[-1] = (a0, max(a1, i1))
        else:
            merged.append((i0, i1))

    return merged


# ================
# estimate support
# ================

def estimate_support(stieltjes, x_min, x_max, n_scan=1024,
                     refine=True, resplit_density=16, merge_threshold=0.0,
                     thr_rel=1e-4,  min_log_width_mult=1.0, log=False,
                     delta=None, **kwargs):
    """
    Estimate support intervals from the fitted Stieltjes transform.

    Parameters
    ----------

    stieltjes : object
        Stieltjes-like object with ``__call__`` and ``evaluate_scalar``.

    x_min, x_max : float
        Scan range.

    n_scan : int, default=1024
        Number of coarse scan points.

    refine : bool, default=True
        Refine each detected edge by a safeguarded Newton solve of the local
        threshold-crossing equation.

    resplit_density : int, default=16
        Densification factor used to re-check each coarse run for touching
        bulks that may be missed by the coarse scan.

    merge_threshold : float, default=0.0
        Merge adjacent detected bulks if the gap is smaller than this value. In
        log mode it is measured in log-x.

    thr_rel : float, default=1e-4
        Linear-mode relative threshold. In log mode the threshold is selected
        automatically from the data.

    min_log_width_mult : float, default=1.0
        Minimum accepted run width in units of log-grid spacing.

    log : bool, default=False
        If True, use geometric x-grid and threshold a log signal built from the
        Poisson-floor-normalized score.

    delta : float, default=None
        Imaginary offset. If None, a scale-adaptive default is used.

    Other Parameters
    ----------------

    smooth_width : int, optional
        Width of a mild moving-average smoother applied to the threshold signal
        before classification. Default is automatic.

    threshold : float, optional
        Override the automatically chosen threshold on the classification
        signal. In log mode this is in ``log10(score)``.

    threshold_method : {'otsu'}, default='otsu'
        Automatic threshold selection method.
    """

    x_min = float(x_min)
    x_max = float(x_max)
    n_scan = int(n_scan)

    if log and x_min <= 0.0:
        raise ValueError('"x_min" must be positive when log=True.')

    scale = max(1.0, abs(x_max - x_min), abs(x_min), abs(x_max))
    if delta is None:
        delta = 1e-6 * scale
    delta = float(delta)

    smooth_width = int(kwargs.get("smooth_width", _auto_smooth_width(n_scan)))
    threshold = kwargs.get("threshold", None)
    threshold_method = str(kwargs.get("threshold_method", "otsu")).lower()

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

    signal_raw = _signal_from_score(score_grid, log)
    signal_grid = _moving_average(signal_raw, smooth_width)

    finite = numpy.isfinite(signal_grid)
    if not numpy.any(finite):
        info = {
            "x_grid": x_grid,
            "m_grid": m_grid,
            "im_grid": im_grid,
            "floor_grid": floor_grid,
            "score_grid": score_grid,
            "signal_raw": signal_raw,
            "signal_grid": signal_grid,
            "threshold": None,
            "threshold_raw": None,
            "threshold_method": "empty",
            "mask": numpy.zeros_like(signal_grid, dtype=bool),
            "runs": [],
            "delta": delta,
        }
        return [], info

    if log:
        if threshold is None:
            if threshold_method != "otsu":
                raise ValueError("Only threshold_method='otsu' is supported.")
            threshold_raw, thr_info = _otsu_threshold(signal_grid[finite])
            threshold = float(threshold_raw)
        else:
            threshold = float(threshold)
            threshold_raw = float(threshold)
            thr_info = {
                "threshold_method": "manual",
                "threshold_score": numpy.nan,
            }
    else:
        max_score = float(numpy.nanmax(score_grid[finite]))
        threshold_raw = max(float(thr_rel) * max_score, 10.0 * delta)
        threshold = float(threshold_raw)
        thr_info = {
            "threshold_method": "relative",
            "threshold_score": numpy.nan,
        }

    mask = numpy.isfinite(signal_grid) & (signal_grid > threshold)
    runs = _runs_from_mask(mask)
    runs = _filter_tiny_runs(
        runs,
        x_grid,
        log,
        min_points=2,
        min_log_width_mult=min_log_width_mult)

    # resolve touching bulks missed by the coarse scan
    runs = _resplit_runs(
        stieltjes,
        runs,
        x_grid,
        delta,
        threshold,
        log,
        resplit_density=resplit_density,
        min_log_width_mult=min_log_width_mult)

    runs = _merge_adjacent_runs(runs, x_grid, log, merge_threshold)

    est_supp = []
    edges_pre_newton = []
    edges_post_newton = []

    for i0, i1 in runs:
        if i0 <= 0:
            xL = float(x_grid[i0])
            left_bracket = (float(x_grid[i0]), float(x_grid[i0]))
        else:
            left_bracket = (float(x_grid[i0 - 1]), float(x_grid[i0]))
            if refine:
                xL = _bracketed_newton_signal(
                    stieltjes,
                    left_bracket[0],
                    left_bracket[1],
                    delta,
                    threshold,
                    log)
            else:
                xL = left_bracket[1]

        if i1 >= x_grid.size - 1:
            xR = float(x_grid[i1])
            right_bracket = (float(x_grid[i1]), float(x_grid[i1]))
        else:
            right_bracket = (float(x_grid[i1]), float(x_grid[i1 + 1]))
            if refine:
                xR = _bracketed_newton_signal(
                    stieltjes,
                    right_bracket[0],
                    right_bracket[1],
                    delta,
                    threshold,
                    log)
            else:
                xR = right_bracket[0]

        edges_pre_newton.extend([left_bracket[1], right_bracket[0]])
        edges_post_newton.extend([xL, xR])

        if xR > xL:
            est_supp.append((float(xL), float(xR)))

    info = {
        "x_grid": x_grid,
        "m_grid": m_grid,
        "im_grid": im_grid,
        "floor_grid": floor_grid,
        "score_grid": score_grid,
        "signal_raw": signal_raw,
        "signal_grid": signal_grid,
        "threshold": threshold,
        "threshold_raw": threshold_raw,
        "threshold_method": thr_info.get("threshold_method", "unknown"),
        "threshold_score": thr_info.get("threshold_score", numpy.nan),
        "mask": mask,
        "runs": runs,
        "edges_pre_newton": numpy.array(edges_pre_newton, dtype=float),
        "edges_post_newton": numpy.array(edges_post_newton, dtype=float),
        "delta": delta,
        "smooth_width": smooth_width,
        "merge_threshold": merge_threshold,
        "resplit_density": resplit_density,
    }

    if "hist" in thr_info:
        info["hist"] = thr_info["hist"]
        info["bin_centers"] = thr_info["bin_centers"]

    return est_supp, info
