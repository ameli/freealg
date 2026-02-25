# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE

"""Hybrid deformation solver.

Same API as ``_deform.py`` / ``_deform2.py`` / ``_deform3.py``.
Combines outputs of deform2 and deform3 *per active component*.

This revision adds a hard/strong penalty for "imaginary-part collapse" relative to
previous time-row values (to avoid choosing deform3 on a component when it produces
near-real converged roots inside a previously active bulk).
"""

import numpy

from ._deform2 import deform_newton as _deform2_newton
from ._deform3 import deform_newton as _deform3_newton

__all__ = ['deform_newton']


def _connected_components(mask):
    """Return list of (i0, i1) inclusive intervals for True runs."""
    mask = numpy.asarray(mask, dtype=bool).ravel()
    n = mask.size
    comps = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while (j + 1) < n and mask[j + 1]:
            j += 1
        comps.append((i, j))
        i = j + 1
    return comps


def _component_score(Wcand, okcand, Wprev, idx, imag_floor, eta):
    """Score a candidate row on one component; lower is better.

    Adds a strong penalty for isolated/partial collapse of |Im w| inside a component
    that was active in the previous row. This targets "ok=True but wrong-sheet near-real"
    roots that create holes.
    """
    sl = slice(idx[0], idx[1] + 1)
    w = Wcand[sl]
    ok = okcand[sl]
    wp = Wprev[sl]

    if w.size == 0:
        return (numpy.inf, numpy.inf, numpy.inf, numpy.inf)

    im = numpy.abs(numpy.imag(w))
    imp = numpy.abs(numpy.imag(wp))
    finite = numpy.isfinite(w.real) & numpy.isfinite(w.imag)
    active = ok & finite

    # Coverage metrics (keep old behavior, but not the sole deciding factor).
    strong = active & (im > imag_floor)
    n_strong = int(numpy.count_nonzero(strong))
    n_active = int(numpy.count_nonzero(active))

    # Continuity relative to previous time row (use only active points).
    if n_active > 0:
        cont = float(numpy.sum(numpy.abs(w[active] - wp[active]))) / n_active
    else:
        cont = numpy.inf

    # Strong collapse detector relative to previous row (eta-based, not imag_floor-based).
    # imag_floor can be too high for weak bulks; use eta-scale thresholds.
    thr_prev = 20.0 * eta
    thr_now = 5.0 * eta
    prev_active = numpy.isfinite(wp.real) & numpy.isfinite(wp.imag) & (imp > thr_prev)
    now_collapsed = ((~active) | (im < thr_now))
    collapse = prev_active & now_collapsed

    # Penalize interior collapses more strongly (likely wrong-sheet holes, not edges).
    n_collapse = int(numpy.count_nonzero(collapse))
    n_collapse_interior = 0
    if w.size >= 3 and n_collapse > 0:
        for k in range(1, w.size - 1):
            if collapse[k] and prev_active[k - 1] and prev_active[k + 1]:
                n_collapse_interior += 1

    # Smoothness on currently active points.
    smooth = 0.0
    if w.size >= 3:
        for k in range(1, w.size - 1):
            if active[k - 1] and active[k] and active[k + 1]:
                smooth += abs(w[k + 1] - 2.0 * w[k] + w[k - 1])
    smooth = float(smooth)

    # Lexicographic score:
    #   1) interior collapses (dominant)
    #   2) all collapses
    #   3) coverage
    #   4) continuity + tiny smoothness tie-breaker
    return (n_collapse_interior, n_collapse, -n_strong, cont + 0.05 * smooth)


def _fill_short_holes_from_other(Wout_row, okout_row, Walt_row, okalt_row,
                                 comp, max_hole=3, imag_floor=0.0):
    """Fill short holes in chosen row if alternate row has a valid value there."""
    i0, i1 = comp
    i = i0
    while i <= i1:
        bad = ((not okout_row[i]) or (abs(Wout_row[i].imag) <= imag_floor))
        if not bad:
            i += 1
            continue
        j = i
        while j <= i1 and (((not okout_row[j]) or (abs(Wout_row[j].imag) <= imag_floor))):
            j += 1
        L = i - 1
        R = j
        hole_len = j - i
        left_good = (L >= i0 and okout_row[L] and (abs(Wout_row[L].imag) > imag_floor))
        right_good = (R <= i1 and okout_row[R] and (abs(Wout_row[R].imag) > imag_floor))
        if hole_len <= max_hole and left_good and right_good:
            for k in range(i, j):
                if okalt_row[k] and numpy.isfinite(Walt_row[k].real) and numpy.isfinite(Walt_row[k].imag):
                    if (abs(Walt_row[k].imag) > imag_floor) or (not okout_row[k]):
                        Wout_row[k] = Walt_row[k]
                        okout_row[k] = True
        i = j


def deform_newton(z_list, t_grid, coeffs, c0, w0_list, dt_max=0.1, sweep=True,
                  time_rel_tol=5.0, active_imag_eps=None, sweep_pad=20,
                  max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6,
                  w_min=1e-14):
    """Hybrid of deform2 and deform3 with component-wise row selection."""

    common_kwargs = dict(
        z_list=z_list,
        t_grid=t_grid,
        coeffs=coeffs,
        c0=c0,
        w0_list=w0_list,
        dt_max=dt_max,
        sweep=sweep,
        time_rel_tol=time_rel_tol,
        active_imag_eps=active_imag_eps,
        sweep_pad=sweep_pad,
        max_iter=max_iter,
        tol=tol,
        armijo=armijo,
        min_lam=min_lam,
        w_min=w_min,
    )

    W2, ok2 = _deform2_newton(**common_kwargs)
    W3, ok3 = _deform3_newton(**common_kwargs)

    W2 = numpy.asarray(W2, dtype=complex)
    W3 = numpy.asarray(W3, dtype=complex)
    ok2 = numpy.asarray(ok2, dtype=bool)
    ok3 = numpy.asarray(ok3, dtype=bool)

    nt, nz = W2.shape
    W = numpy.empty_like(W2)
    ok = numpy.empty_like(ok2)

    W[0, :] = W2[0, :]
    ok[0, :] = ok2[0, :]

    z_arr = numpy.asarray(z_list, dtype=complex).ravel()
    eta = float(abs(z_arr[0].imag)) if z_arr.size else 1.0e-6
    if active_imag_eps is None:
        imag_floor = 50.0 * eta / max(1.0e-30, float(c0))
    else:
        imag_floor = float(active_imag_eps) / max(1.0e-30, float(c0))
    imag_floor *= 0.25  # less aggressive floor in m-space

    for it in range(1, nt):
        W_row = W3[it, :].copy()
        ok_row = ok3[it, :].copy()

        prev_im = numpy.abs(numpy.imag(W[it - 1, :]))
        mask2 = ok2[it, :] & numpy.isfinite(W2[it, :].real) & numpy.isfinite(W2[it, :].imag)             & (numpy.abs(numpy.imag(W2[it, :])) > imag_floor)
        mask3 = ok3[it, :] & numpy.isfinite(W3[it, :].real) & numpy.isfinite(W3[it, :].imag)             & (numpy.abs(numpy.imag(W3[it, :])) > imag_floor)
        mask_prev = ok[it - 1, :] & numpy.isfinite(W[it - 1, :].real) & numpy.isfinite(W[it - 1, :].imag)             & (prev_im > imag_floor)
        comp_mask = mask2 | mask3 | mask_prev

        pad = max(1, int(sweep_pad // 2) if sweep_pad is not None else 10)
        if pad > 0 and numpy.any(comp_mask):
            comp_mask_pad = comp_mask.copy()
            idx = numpy.where(comp_mask)[0]
            for j in idx:
                lo = max(0, j - pad)
                hi = min(nz, j + pad + 1)
                comp_mask_pad[lo:hi] = True
            comp_mask = comp_mask_pad

        comps = _connected_components(comp_mask)
        if not comps:
            W[it, :] = W2[it, :]
            ok[it, :] = ok2[it, :]
            continue

        for comp in comps:
            s2 = _component_score(W2[it, :], ok2[it, :], W[it - 1, :], comp, imag_floor, eta)
            s3 = _component_score(W3[it, :], ok3[it, :], W[it - 1, :], comp, imag_floor, eta)

            use2 = (s2 < s3)
            if use2:
                i0, i1 = comp
                W_row[i0:i1 + 1] = W2[it, i0:i1 + 1]
                ok_row[i0:i1 + 1] = ok2[it, i0:i1 + 1]
                _fill_short_holes_from_other(W_row, ok_row, W3[it, :], ok3[it, :], comp,
                                             max_hole=4, imag_floor=imag_floor)
            else:
                _fill_short_holes_from_other(W_row, ok_row, W2[it, :], ok2[it, :], comp,
                                             max_hole=4, imag_floor=imag_floor)

        outside = ~comp_mask
        if numpy.any(outside):
            collapsed3 = outside & ((~ok_row) | (numpy.abs(numpy.imag(W_row)) <= imag_floor))
            swap = collapsed3 & ok2[it, :]
            W_row[swap] = W2[it, swap]
            ok_row[swap] = ok2[it, swap]

        W[it, :] = W_row
        ok[it, :] = ok_row

    return W, ok
