# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


"""
Numba-accelerated geometric predictor-corrector continuation for free
free decompression.

This version preserves ``_decompress10_5.py`` exactly for non-log mode.
Only for log-aware runs, a scale-aware acceptance metric is used inside the
per-x continuation so the execution model stays embarrassingly parallel in x.

Default behavior is unchanged unless ``log_mode`` is enabled explicitly or
is auto-detected from a strongly geometric positive x-grid.
"""

# =======
# Imports
# =======

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
import numpy

from ._decompress_coeffs2 import _decompress_coeffs_core
# from ._roots import roots_m_numba, MODE_DIRECT
from ._roots2 import roots_m_numba, MODE_DIRECT

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(func):
            return func
        return deco

# Set to False to avoid crash at multiple runs
numba_cache = False

__all__ = ['decompress_newton']


# =========
# solve 2x2
# =========

@njit(cache=numba_cache)
def _solve2x2(a11, a12, a21, a22, b1, b2):

    det = a11 * a22 - a12 * a21
    if abs(det) == 0.0:
        return 0.0 + 0.0j, 0.0 + 0.0j, det, False
    x1 = (b1 * a22 - a12 * b2) / det
    x2 = (a11 * b2 - b1 * a21) / det
    return x1, x2, det, True


# ======================
# eval P dP scalar numba
# ======================

@njit(cache=numba_cache)
def _eval_P_dP_scalar_numba(z, m, coeffs):
    """
    Evaluate P(z, m), dP/dz, dP/dm for one scalar pair.
    """

    deg_z = coeffs.shape[0] - 1
    s = coeffs.shape[1] - 1

    a = numpy.empty(s + 1, dtype=numpy.complex128)
    a_dz = numpy.empty(s + 1, dtype=numpy.complex128)

    for j in range(s + 1):
        val = coeffs[deg_z, j]
        dval = 0.0 + 0.0j
        for i in range(deg_z - 1, -1, -1):
            dval = dval * z + val
            val = val * z + coeffs[i, j]
        a[j] = val
        a_dz[j] = dval

    P = a[s]
    Pm = 0.0 + 0.0j
    for j in range(s - 1, -1, -1):
        Pm = Pm * m + P
        P = P * m + a[j]

    Pz = a_dz[s]
    for j in range(s - 1, -1, -1):
        Pz = Pz * m + a_dz[j]

    return P, Pz, Pm


# =======================
# residual jacobian numba
# =======================

@njit(cache=numba_cache)
def _residual_jacobian_numba(z_fixed, tau, coeffs, zeta, y):
    """
    Residual and Jacobian entries for the 2x2 system.
    """

    P, Pz, Py = _eval_P_dP_scalar_numba(zeta, y, coeffs)
    F1 = P
    F2 = zeta - (tau - 1.0) / y - z_fixed

    J11 = Pz
    J12 = Py
    J21 = 1.0 + 0.0j
    J22 = (tau - 1.0) / (y * y)
    detJ = J11 * J22 - J12 * J21

    return F1, F2, J11, J12, J21, J22, detJ


# =======================
# curve tangent tau numba
# =======================

@njit(cache=numba_cache)
def _curve_tangent_tau_numba(tau, coeffs, zeta, y):
    """
    Tangent d(zeta, y)/d tau along the decompression system at fixed z.
    """

    _F1, _F2, J11, J12, J21, J22, _detJ = _residual_jacobian_numba(
        0.0 + 0.0j, tau, coeffs, zeta, y)
    rhs1 = 0.0 + 0.0j
    rhs2 = 1.0 / y
    dzeta, dy, _det, ok = _solve2x2(J11, J12, J21, J22, rhs1, rhs2)
    return dzeta, dy, J11, J12, J21, J22, ok


# ====================
# newton project numba
# ====================

@njit(cache=numba_cache)
def _newton_project_numba(z_fixed, tau, coeffs, zeta0, y0,
                          max_iter, tol_res, tol_step,
                          armijo, min_lam, step_clip):
    """
    Damped Newton projection onto F(zeta, y; tau, z_fixed) = 0.
    """

    zeta = zeta0
    y = y0
    last_corr = 1.0e300
    last_detJ = complex(numpy.nan, numpy.nan)

    for it in range(max_iter):
        F1, F2, J11, J12, J21, J22, detJ = _residual_jacobian_numba(
            z_fixed, tau, coeffs, zeta, y)
        res_norm = max(abs(F1), abs(F2))
        last_detJ = detJ
        if res_norm <= tol_res:
            return zeta, y, True, it, res_norm, last_corr, last_detJ

        dzeta, dy, _det, ok = _solve2x2(J11, J12, J21, J22, -F1, -F2)
        if not ok:
            return zeta, y, False, it, res_norm, 1.0e300, last_detJ

        if step_clip > 0.0:
            step_inf = max(abs(dzeta), abs(dy))
            if step_inf > step_clip:
                scl = step_clip / step_inf
                dzeta *= scl
                dy *= scl

        last_corr = max(abs(dzeta), abs(dy))
        if last_corr <= tol_step:
            return zeta, y, True, it, res_norm, last_corr, last_detJ

        norm0 = res_norm
        lam = 1.0
        accepted = False
        while lam >= min_lam:
            zeta_try = zeta + lam * dzeta
            y_try = y + lam * dy
            if y_try == 0.0:
                lam *= 0.5
                continue

            F1_try, F2_try, _a, _b, _c, _d, _det_try = \
                _residual_jacobian_numba(z_fixed, tau, coeffs,
                                         zeta_try, y_try)
            norm_try = max(abs(F1_try), abs(F2_try))
            if (norm_try <= (1.0 - armijo * lam) * norm0) or \
                    (norm_try < norm0):
                zeta = zeta_try
                y = y_try
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            if res_norm <= 100.0 * tol_res:
                return zeta, y, True, it, res_norm, last_corr, last_detJ
            return zeta, y, False, it, res_norm, last_corr, last_detJ

    F1, F2, _a, _b, _c, _d, detJ = _residual_jacobian_numba(
        z_fixed, tau, coeffs, zeta, y)
    res_norm = max(abs(F1), abs(F2))
    ok = (res_norm <= 100.0 * tol_res)
    return zeta, y, ok, max_iter, res_norm, last_corr, detJ


# ==================
# predict heun numba
# ==================

@njit(cache=numba_cache)
def _predict_heun_numba(tau0, tau1, coeffs, zeta0, y0):
    """
    Embedded Euler/Heun predictor from tau0 to tau1.
    """

    h = tau1 - tau0
    dz0, dy0, J011, J012, J021, J022, ok0 = _curve_tangent_tau_numba(
        tau0, coeffs, zeta0, y0)
    if not ok0:
        return (0.0 + 0.0j, 0.0 + 0.0j, 1.0e300,
                0.0 + 0.0j, 0.0 + 0.0j, False)

    zeta_e = zeta0 + h * dz0
    y_e = y0 + h * dy0

    dz1, dy1, J111, J112, J121, J122, ok1 = _curve_tangent_tau_numba(
        tau1, coeffs, zeta_e, y_e)
    if not ok1:
        return (0.0 + 0.0j, 0.0 + 0.0j, 1.0e300,
                0.0 + 0.0j, 0.0 + 0.0j, False)

    zeta_h = zeta0 + 0.5 * h * (dz0 + dz1)
    y_h = y0 + 0.5 * h * (dy0 + dy1)

    err = max(abs(zeta_h - zeta_e), abs(y_h - y_e))
    detJ0 = J011 * J022 - J012 * J021
    detJ1 = J111 * J122 - J112 * J121
    return zeta_h, y_h, err, detJ0, detJ1, True


# ==============
# safe rel delta
# ==============

@njit(cache=numba_cache)
def _safe_rel_delta(a, b, eps):

    da = abs(a - b)
    sa = abs(a)
    sb = abs(b)
    denom = sa
    if sb > denom:
        denom = sb
    if denom < eps:
        denom = eps
    return da / denom


# =======================
# metric pred scale numba
# =======================

@njit(cache=numba_cache)
def _metric_pred_scale_numba(zeta, y, log_mode):

    if not log_mode:
        return max(1.0, abs(zeta), abs(y))
    return 1.0


# =======================
# metric pred error numba
# =======================

@njit(cache=numba_cache)
def _metric_pred_error_numba(zeta_h, y_h, zeta_e, y_e, raw_err, log_mode,
                             rel_eps):

    if not log_mode:
        return raw_err
    return max(_safe_rel_delta(zeta_h, zeta_e, rel_eps),
               _safe_rel_delta(y_h, y_e, rel_eps))


# =================
# metric move numba
# =================

@njit(cache=numba_cache)
def _metric_move_numba(zeta_new, y_new, zeta_old, y_old, log_mode, rel_eps):

    if not log_mode:
        return max(abs(zeta_new - zeta_old), abs(y_new - y_old), 1.0)
    return max(_safe_rel_delta(zeta_new, zeta_old, rel_eps),
               _safe_rel_delta(y_new, y_old, rel_eps), rel_eps)


# =================
# metric corr numba
# =================

@njit(cache=numba_cache)
def _metric_corr_numba(zeta_corr, y_corr, zeta_pred, y_pred, corr_norm,
                       log_mode, rel_eps):

    if not log_mode:
        return corr_norm
    return max(_safe_rel_delta(zeta_corr, zeta_pred, rel_eps),
               _safe_rel_delta(y_corr, y_pred, rel_eps))


# =======================
# advance one point numba
# =======================

@njit(cache=numba_cache)
def _advance_one_point_numba(z_fixed, t0, t1, coeffs, zeta0, y0,
                             max_iter, tol_res, tol_step,
                             armijo, min_lam, step_clip,
                             dt_max, dt_min,
                             pred_atol, pred_rtol,
                             corr_factor, step_growth,
                             step_shrink, max_reject,
                             det_guard, log_mode, rel_eps):
    """
    Adaptive continuation from t0 to t1 for one fixed z.
    """

    tau_cur = math.exp(t0)
    tau_end = math.exp(t1)
    zeta = zeta0
    y = y0

    h = tau_end - tau_cur
    if h > dt_max:
        h = dt_max
    if h < dt_min:
        h = dt_min
    rejects = 0

    while tau_cur < tau_end:
        remaining = tau_end - tau_cur
        if h > remaining:
            h = remaining
        if h < dt_min:
            return zeta, y, False

        tau_try = tau_cur + h
        zeta_pred, y_pred, raw_err_pred, detJ0, detJ1, ok_pred = \
            _predict_heun_numba(tau_cur, tau_try, coeffs, zeta, y)
        if not ok_pred:
            rejects += 1
            h *= step_shrink
            if rejects > max_reject:
                return zeta, y, False
            continue

        scale = _metric_pred_scale_numba(zeta_pred, y_pred, log_mode)
        pred_tol = pred_atol + pred_rtol * scale
        err_pred = _metric_pred_error_numba(
            zeta_pred, y_pred, zeta + (zeta_pred - zeta),
            y + (y_pred - y), raw_err_pred, log_mode, rel_eps)
        # For non-log this is exactly raw_err_pred through the helper. For
        # log mode, compare Heun/Euler in a scale-aware relative metric.
        if log_mode:
            # Reconstruct the embedded Euler predictor from the returned raw
            # Heun state by one cheap tangent evaluation at tau_cur.
            h_local = tau_try - tau_cur
            dz0, dy0, _a, _b, _c, _d, ok0 = _curve_tangent_tau_numba(
                tau_cur, coeffs, zeta, y)
            if not ok0:
                rejects += 1
                h *= step_shrink
                if rejects > max_reject:
                    return zeta, y, False
                continue
            zeta_e = zeta + h_local * dz0
            y_e = y + h_local * dy0
            err_pred = _metric_pred_error_numba(
                zeta_pred, y_pred, zeta_e, y_e, raw_err_pred,
                True, rel_eps)

        if (err_pred > pred_tol) or (abs(detJ0) < det_guard) or \
                (abs(detJ1) < det_guard):
            rejects += 1
            h *= step_shrink
            if rejects > max_reject:
                return zeta, y, False
            continue

        zc, yc, ok_newton, _it, _res_norm, corr_norm, detJc = \
            _newton_project_numba(
                z_fixed, tau_try, coeffs, zeta_pred, y_pred,
                max_iter, tol_res, tol_step,
                armijo, min_lam, step_clip)

        move = _metric_move_numba(zeta_pred, y_pred, zeta, y,
                                  log_mode, rel_eps)
        corr_measure = _metric_corr_numba(zc, yc, zeta_pred, y_pred,
                                          corr_norm, log_mode, rel_eps)
        corr_ok = (corr_measure <= corr_factor * move)
        det_ok = numpy.isfinite(abs(detJc)) and (abs(detJc) >= det_guard)

        if ok_newton and corr_ok and det_ok and (yc != 0.0):
            zeta = zc
            y = yc
            tau_cur = tau_try
            rejects = 0

            if (err_pred < 0.1 * pred_tol) and (corr_measure < 0.25 * move):
                h = step_growth * h
                if h > dt_max:
                    h = dt_max
            elif (err_pred < pred_tol) and (corr_measure < move):
                h = 1.2 * h
                if h > dt_max:
                    h = dt_max
            continue

        rejects += 1
        h *= step_shrink
        if rejects > max_reject:
            return zeta, y, False

    return zeta, y, True


# ==========================
# eval P dP d2P scalar numba
# ==========================

@njit(cache=numba_cache)
def _eval_P_dP_d2_scalar_numba(z, m, coeffs):

    deg_z = coeffs.shape[0] - 1
    s = coeffs.shape[1] - 1

    a = numpy.empty(s + 1, dtype=numpy.complex128)
    for j in range(s + 1):
        val = coeffs[deg_z, j]
        for i in range(deg_z - 1, -1, -1):
            val = val * z + coeffs[i, j]
        a[j] = val

    P = a[s]
    Pm = 0.0 + 0.0j
    Pmm = 0.0 + 0.0j
    for j in range(s - 1, -1, -1):
        Pmm = Pmm * m + 2.0 * Pm
        Pm = Pm * m + P
        P = P * m + a[j]

    return P, Pm, Pmm


# ==================================
# decompress coeffs raw helper numba
# ==================================

@njit(cache=numba_cache)
def _decompress_coeffs_raw_numba(a, t):

    a_copy = a.copy()
    a_copy[-1, 0] = 0.0 + 0.0j
    return _decompress_coeffs_core(a_copy, t)


# =====================================
# zeta and y from reconstructed w numba
# =====================================

@njit(cache=numba_cache)
def _zeta_y_from_w_numba(z_fixed, tau, w, w_min):

    y = tau * w
    if abs(y) < w_min:
        if y == 0:
            y = complex(w_min, 0.0)
        else:
            y = y + (w_min * y / abs(y))
    zeta = z_fixed + (tau - 1.0) / y
    return zeta, y


# ======================
# roots at fixed z numba
# ======================

@njit(cache=numba_cache)
def _roots_fixed_z_numba(coeffs_t, z_fixed):

    return roots_m_numba(coeffs_t, z_fixed,
                         0.0, 1e-14,
                         MODE_DIRECT,
                         True, 8, 1e-14, 1e-14,
                         True)


# ========================
# nearest root index numba
# ========================

@njit(cache=numba_cache)
def _nearest_root_index_numba(roots, w_ref, exclude_idx):

    best_idx = -1
    best_d = 0.0
    found = False
    for j in range(roots.size):
        if exclude_idx >= 0 and j == exclude_idx:
            continue
        d = abs(roots[j] - w_ref)
        if (not found) or (d < best_d):
            best_d = d
            best_idx = j
            found = True
    return best_idx


# =====================================
# closest real-lock partner index numba
# =====================================

@njit(cache=numba_cache)
def _closest_real_partner_index_numba(roots, idx1):

    if idx1 < 0:
        return -1
    r1 = roots[idx1]
    best_idx = -1
    best_d = 0.0
    found = False
    for j in range(roots.size):
        if j == idx1:
            continue
        d = abs(roots[j].real - r1.real)
        if (not found) or (d < best_d):
            best_d = d
            best_idx = j
            found = True
    return best_idx


# ========================
# advance trajectory numba
# ========================

@njit(cache=numba_cache)
def _advance_trajectory_numba(z_fixed, t, coeffs, w0,
                              max_iter, tol_res, tol_step,
                              armijo, min_lam, step_clip,
                              dt_max, dt_min,
                              pred_atol, pred_rtol,
                              corr_factor, step_growth,
                              step_shrink, max_reject,
                              det_guard, w_min, log_mode, rel_eps,
                              pair_enable, pair_gap_factor):
    """
    Compute one full x-trajectory using the v10 continuation logic.
    """

    n_t = t.size
    W = numpy.empty(n_t, dtype=numpy.complex128)
    ok = numpy.zeros(n_t, dtype=numpy.bool_)
    tau = numpy.exp(t)

    W[0] = w0
    ok[0] = numpy.isfinite(w0)

    zeta_state = z_fixed
    y_state = tau[0] * w0
    if abs(y_state) < w_min:
        y_state = y_state + w_min

    for k in range(1, n_t):
        zeta0 = zeta_state
        y0 = y_state

        if (not numpy.isfinite(zeta0)) or (not numpy.isfinite(y0)) or \
                (abs(y0) < w_min):
            zeta0 = z_fixed
            y0 = tau[k - 1] * w0
            if abs(y0) < w_min:
                y0 = y0 + w_min

        zeta1, y1, success = _advance_one_point_numba(
            z_fixed, t[k - 1], t[k], coeffs, zeta0, y0,
            max_iter, tol_res, tol_step,
            armijo, min_lam, step_clip,
            dt_max, dt_min,
            pred_atol, pred_rtol,
            corr_factor, step_growth,
            step_shrink, max_reject,
            det_guard, log_mode, rel_eps)

        if success and numpy.isfinite(zeta1) and numpy.isfinite(y1):
            w = y1 / tau[k]
            if abs(w) >= w_min:
                w_use = w
                zeta_use = zeta1
                y_use = y1
                ok_use = True

                if pair_enable and (w.imag <= 0.0):
                    coeffs_t = _decompress_coeffs_raw_numba(
                        coeffs, float(t[k]))
                    roots = _roots_fixed_z_numba(coeffs_t, z_fixed)
                    if roots.size >= 2:
                        idx1 = _nearest_root_index_numba(roots, w, -1)
                        idx2 = _closest_real_partner_index_numba(roots, idx1)
                        if idx1 >= 0 and idx2 >= 0:
                            r1 = roots[idx1]
                            r2 = roots[idx2]
                            _P, Fm, Fmm = _eval_P_dP_d2_scalar_numba(
                                z_fixed, r1, coeffs_t)
                            if numpy.isfinite(abs(Fmm)) and abs(Fmm) > 0.0:
                                gap_loc = abs(2.0 * Fm / Fmm)
                                if numpy.isfinite(gap_loc) and (gap_loc > 0.0):

                                    if (abs(r2 - r1) <= pair_gap_factor *
                                            gap_loc) and (r2.imag > 0.0):

                                        zeta_guess, y_guess = \
                                            _zeta_y_from_w_numba(
                                                z_fixed, tau[k], r2, w_min)

                                        zeta_proj, y_proj, ok_proj, _itp, \
                                            _resp, _corrp, _detp =  \
                                            _newton_project_numba(
                                                z_fixed, tau[k], coeffs,
                                                zeta_guess, y_guess,
                                                max_iter, tol_res, tol_step,
                                                armijo, min_lam, step_clip)
                                        if ok_proj and \
                                                numpy.isfinite(zeta_proj) and \
                                                numpy.isfinite(y_proj) and \
                                                (abs(y_proj) >= w_min):
                                            w_proj = y_proj / tau[k]
                                            if numpy.isfinite(w_proj) and \
                                                    (r2.imag > 0.0):
                                                w_use = w_proj
                                                zeta_use = zeta_proj
                                                y_use = y_proj
                                                ok_use = True

                W[k] = w_use
                ok[k] = ok_use
                zeta_state = zeta_use
                y_state = y_use
                continue

        W[k] = W[k - 1]
        ok[k] = False

    return W, ok


# =========
# run chunk
# =========

def _run_chunk(args):
    """
    Run a chunk of x-trajectories.
    """

    z_chunk, t, coeffs, w0_chunk, common = args
    n_t = t.size
    m = z_chunk.size
    W = numpy.full((n_t, m), numpy.nan + 1j * numpy.nan,
                   dtype=numpy.complex128)
    ok = numpy.zeros((n_t, m), dtype=bool)

    for j in range(m):
        Wj, okj = _advance_trajectory_numba(
            z_chunk[j], t, coeffs, w0_chunk[j],
            int(common['max_iter']), float(common['tol_res']),
            float(common['tol_step']), float(common['armijo']),
            float(common['min_lam']), float(common['step_clip']),
            float(common['dt_max']), float(common['dt_min']),
            float(common['pred_atol']), float(common['pred_rtol']),
            float(common['corr_factor']), float(common['step_growth']),
            float(common['step_shrink']), int(common['max_reject']),
            float(common['det_guard']), float(common['w_min']),
            bool(common['log_mode']), float(common['rel_eps']),
            bool(common['pair_enable']), float(common['pair_gap_factor']))
        W[:, j] = Wj
        ok[:, j] = okj

    return W, ok


# ====================
# auto detect log mode
# ====================

def _auto_detect_log_mode(z_query):
    """
    Return True only for a clearly geometric positive real grid.
    """

    if z_query.size < 8:
        return False

    x = numpy.asarray(z_query.real, dtype=float)
    if numpy.any(~numpy.isfinite(x)) or numpy.any(x <= 0.0):
        return False

    ratios = x[1:] / x[:-1]
    if numpy.any(~numpy.isfinite(ratios)) or numpy.any(ratios <= 0.0):
        return False

    log_r = numpy.log(ratios)
    if numpy.max(numpy.abs(log_r - numpy.median(log_r))) > 1e-6:
        return False

    span = x[-1] / x[0]
    return bool(span >= 100.0)


# =================
# decompress newton
# =================

def decompress_newton(z_query, t, coeffs, w0_list=None, max_iter=50,
                      tol=1e-12, armijo=1e-4, min_lam=1e-6, w_min=1e-14,
                      **kwargs):
    """
    Free decompression by geometric continuation on the spectral curve.

    For log-aware runs, only the predictor/corrector acceptance metric is
    changed to a scale-aware relative metric; the continuation equations,
    Newton solve, and x-parallel execution model are unchanged.
    """

    z_query = numpy.asarray(z_query, dtype=numpy.complex128).ravel()
    t = numpy.asarray(t, dtype=float).ravel()
    coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)

    if t.size == 0:
        return (numpy.empty((0, z_query.size), dtype=numpy.complex128),
                numpy.empty((0, z_query.size), dtype=bool))
    if numpy.any(numpy.diff(t) < 0.0):
        raise ValueError('t must be sorted increasing.')
    if w0_list is None:
        raise ValueError('w0_list must be provided (physical branch at t=0).')

    w0_list = numpy.asarray(w0_list, dtype=numpy.complex128).ravel()
    if w0_list.size != z_query.size:
        raise ValueError('w0_list must have the same size as z_query.')

    dt_max = float(kwargs.get('dt_max', 0.02))
    dt_min = float(kwargs.get('dt_min', 1e-6))
    pred_atol = float(kwargs.get('pred_atol', 1e-8))
    pred_rtol = float(kwargs.get('pred_rtol', 5e-3))
    corr_factor = float(kwargs.get('corr_factor', 5.0))
    step_growth = float(kwargs.get('step_growth', 1.5))
    step_shrink = float(kwargs.get('step_shrink', 0.5))
    max_reject = int(kwargs.get('max_reject', 30))
    det_guard = float(kwargs.get('det_guard', 1e-14))
    tol_step = kwargs.get('tol_step', None)
    if tol_step is None:
        tol_step = 0.1 * float(tol)
    else:
        tol_step = float(tol_step)

    step_clip = kwargs.get('step_clip', None)
    if step_clip is None:
        step_clip = -1.0
    else:
        step_clip = float(step_clip)

    verbose = bool(kwargs.get('verbose', False))
    parallel = bool(kwargs.get('parallel', False))
    n_jobs = kwargs.get('n_jobs', None)
    chunk_size = int(kwargs.get('chunk_size', 0))

    log_mode_kw = kwargs.get('log_mode', kwargs.get('log', None))
    if log_mode_kw is None:
        log_mode = _auto_detect_log_mode(z_query)
    else:
        log_mode = bool(log_mode_kw)

    rel_eps = float(kwargs.get('rel_eps', 1e-12))

    pair_enable = bool(kwargs.get('pair_enable', True))
    pair_gap_factor = float(kwargs.get('pair_gap_factor', 10.0))

    n_t = t.size
    n_z = z_query.size

    common = dict(
        max_iter=max_iter,
        tol_res=float(tol),
        tol_step=tol_step,
        armijo=float(armijo),
        min_lam=float(min_lam),
        step_clip=step_clip,
        dt_max=dt_max,
        dt_min=dt_min,
        pred_atol=pred_atol,
        pred_rtol=pred_rtol,
        corr_factor=corr_factor,
        step_growth=step_growth,
        step_shrink=step_shrink,
        max_reject=max_reject,
        det_guard=det_guard,
        w_min=float(w_min),
        log_mode=bool(log_mode),
        rel_eps=rel_eps,
        pair_enable=pair_enable,
        pair_gap_factor=pair_gap_factor)

    if (not parallel) or (n_z <= 1):
        W, ok = _run_chunk((z_query, t, coeffs, w0_list, common))
    else:
        if n_jobs is None or int(n_jobs) <= 0:
            n_jobs = os.cpu_count() or 1
        else:
            n_jobs = int(n_jobs)

        if chunk_size <= 0:
            chunk_size = max(1, 4 * n_jobs)

        futures = []
        index_chunks = []

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for offset in range(chunk_size):
                idx = numpy.arange(offset, n_z, chunk_size, dtype=numpy.int64)
                if idx.size == 0:
                    continue
                index_chunks.append(idx)
                futures.append(ex.submit(
                    _run_chunk,
                    (z_query[idx], t, coeffs, w0_list[idx], common),
                ))

            W = numpy.full((n_t, n_z), numpy.nan + 1j * numpy.nan,
                           dtype=numpy.complex128)
            ok = numpy.zeros((n_t, n_z), dtype=bool)

            for idx, fut in zip(index_chunks, futures):
                Wc, okc = fut.result()
                W[:, idx] = Wc
                ok[:, idx] = okc

    if verbose:
        for k in range(1, n_t):
            print(f't[{k}] success rate: {ok[k].mean():.3f}')

    return W, ok
