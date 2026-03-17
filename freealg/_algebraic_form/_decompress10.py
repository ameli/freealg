# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE

"""
Geometric predictor-corrector continuation for free decompression.

This module keeps the same public API as ``_decompress9.py`` but replaces the
first-order predictor plus recursive tau-bisection logic with a cleaner
curve-continuation method:

* state variables are (zeta, y) on the spectral curve P(zeta, y) = 0,
* continuation parameter is tau = exp(t),
* predictor uses the tangent ODE obtained by implicit differentiation,
* projection back to the curve uses damped 2x2 Newton,
* adaptive step size is driven by an embedded Euler/Heun predictor error and
  by the size of the Newton correction.

No branch-scoring / Viterbi / ad hoc root-selection heuristics are used.
"""

from __future__ import annotations

import math
import numpy

__all__ = ['decompress_newton']


# =========================
# scalar polynomial helpers
# =========================

def _eval_P_dP_scalar(z: complex, m: complex, coeffs: numpy.ndarray):
    """
    Evaluate P(z, m), dP/dz, dP/dm for one scalar pair (z, m).

    The polynomial is
        P(z, m) = sum_{i=0}^{deg_z} sum_{j=0}^{s} coeffs[i, j] z^i m^j.
    """

    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)

    # Horner in z for each m-coefficient a_j(z) and its z-derivative.
    a = numpy.empty(s + 1, dtype=complex)
    a_dz = numpy.empty(s + 1, dtype=complex)
    for j in range(s + 1):
        val = complex(coeffs[deg_z, j])
        dval = 0.0 + 0.0j
        for i in range(deg_z - 1, -1, -1):
            dval = dval * z + val
            val = val * z + coeffs[i, j]
        a[j] = val
        a_dz[j] = dval

    # Horner in m for P and Pm.
    P = a[s]
    Pm = 0.0 + 0.0j
    for j in range(s - 1, -1, -1):
        Pm = Pm * m + P
        P = P * m + a[j]

    Pz = a_dz[s]
    for j in range(s - 1, -1, -1):
        Pz = Pz * m + a_dz[j]

    return P, Pz, Pm


# =============================
# residual / Jacobian / tangent
# =============================

def _residual_jacobian(z_fixed: complex,
                       tau: float,
                       coeffs: numpy.ndarray,
                       zeta: complex,
                       y: complex):
    """
    Residual and Jacobian for
        F1 = P(zeta, y) = 0,
        F2 = zeta - (tau - 1)/y - z_fixed = 0.
    """

    P, Pz, Py = _eval_P_dP_scalar(zeta, y, coeffs)
    F1 = P
    F2 = zeta - (tau - 1.0) / y - z_fixed

    J = numpy.array(
        [[Pz, Py],
         [1.0 + 0.0j, (tau - 1.0) / (y * y)]],
        dtype=complex,
    )
    F = numpy.array([F1, F2], dtype=complex)
    return F, J


def _curve_tangent_tau(tau: float,
                       coeffs: numpy.ndarray,
                       zeta: complex,
                       y: complex):
    """
    Tangent d(zeta, y)/d tau along the decompression system at fixed z.

    With
        F1(zeta, y) = P(zeta, y),
        F2(zeta, y; tau) = zeta - (tau - 1)/y - z,
    implicit differentiation gives
        J [dzeta/dtau, dy/dtau]^T = - dF/dtau,
    where dF/dtau = [0, -1/y]^T, hence rhs = [0, 1/y]^T.
    """

    _, J = _residual_jacobian(0.0 + 0.0j, tau, coeffs, zeta, y)
    rhs = numpy.array([0.0 + 0.0j, 1.0 / y], dtype=complex)
    vec = numpy.linalg.solve(J, rhs)
    return vec[0], vec[1], J


# ================
# Newton projection
# ================

def _newton_project(z_fixed: complex,
                    tau: float,
                    coeffs: numpy.ndarray,
                    zeta0: complex,
                    y0: complex,
                    *,
                    max_iter: int,
                    tol_res: float,
                    tol_step: float,
                    armijo: float,
                    min_lam: float,
                    step_clip: float | None):
    """
    Damped Newton projection onto F(zeta, y; tau, z_fixed) = 0.

    Returns
    -------
    zeta, y : complex
        Corrected point.
    ok : bool
        Whether projection succeeded.
    n_iter : int
        Number of Newton iterations used.
    res_norm : float
        Final infinity-norm residual.
    corr_norm : float
        Max norm of the last Newton update.
    detJ : complex
        Determinant of the last Jacobian.
    """

    zeta = complex(zeta0)
    y = complex(y0)
    last_corr = math.inf
    last_detJ = complex(numpy.nan, numpy.nan)

    for it in range(int(max_iter)):
        F, J = _residual_jacobian(z_fixed, tau, coeffs, zeta, y)
        res_norm = max(abs(F[0]), abs(F[1]))
        last_detJ = numpy.linalg.det(J)
        if res_norm <= tol_res:
            return zeta, y, True, it, res_norm, last_corr, last_detJ

        try:
            delta = numpy.linalg.solve(J, -F)
        except numpy.linalg.LinAlgError:
            return zeta, y, False, it, res_norm, math.inf, last_detJ

        if step_clip is not None:
            step_inf = max(abs(delta[0]), abs(delta[1]))
            if step_inf > float(step_clip):
                delta *= float(step_clip) / step_inf

        last_corr = max(abs(delta[0]), abs(delta[1]))
        if last_corr <= tol_step:
            return zeta, y, True, it, res_norm, last_corr, last_detJ

        norm0 = res_norm
        lam = 1.0
        accepted = False
        while lam >= float(min_lam):
            zeta_try = zeta + lam * delta[0]
            y_try = y + lam * delta[1]
            if y_try == 0:
                lam *= 0.5
                continue
            F_try, _ = _residual_jacobian(z_fixed, tau, coeffs,
                                          zeta_try, y_try)
            norm_try = max(abs(F_try[0]), abs(F_try[1]))
            if norm_try <= (1.0 - float(armijo) * lam) * norm0 or \
                    norm_try < norm0:
                zeta = zeta_try
                y = y_try
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            # If already very small in residual, accept current iterate.
            if res_norm <= 100.0 * tol_res:
                return zeta, y, True, it, res_norm, last_corr, last_detJ
            return zeta, y, False, it, res_norm, last_corr, last_detJ

    F, J = _residual_jacobian(z_fixed, tau, coeffs, zeta, y)
    res_norm = max(abs(F[0]), abs(F[1]))
    last_detJ = numpy.linalg.det(J)
    ok = (res_norm <= 100.0 * tol_res)
    return zeta, y, ok, int(max_iter), res_norm, last_corr, last_detJ


# ==================
# adaptive evolution
# ==================

def _predict_heun(z_fixed: complex,
                  tau0: float,
                  tau1: float,
                  coeffs: numpy.ndarray,
                  zeta0: complex,
                  y0: complex):
    """
    Embedded Euler/Heun predictor from tau0 to tau1.

    Returns
    -------
    zeta_heun, y_heun : complex
        Second-order predictor.
    err_pred : float
        Predictor defect estimated by Euler-Heun difference.
    detJ0, detJ1 : complex
        Jacobian determinants at start and Euler endpoint.
    """

    h = float(tau1 - tau0)
    dz0, dy0, J0 = _curve_tangent_tau(tau0, coeffs, zeta0, y0)
    zeta_e = zeta0 + h * dz0
    y_e = y0 + h * dy0
    dz1, dy1, J1 = _curve_tangent_tau(tau1, coeffs, zeta_e, y_e)

    zeta_h = zeta0 + 0.5 * h * (dz0 + dz1)
    y_h = y0 + 0.5 * h * (dy0 + dy1)

    err = max(abs(zeta_h - zeta_e), abs(y_h - y_e))
    return zeta_h, y_h, float(err), numpy.linalg.det(J0), numpy.linalg.det(J1)


def _advance_one_point(z_fixed: complex,
                       t0: float,
                       t1: float,
                       coeffs: numpy.ndarray,
                       zeta0: complex,
                       y0: complex,
                       *,
                       max_iter: int,
                       tol_res: float,
                       tol_step: float,
                       armijo: float,
                       min_lam: float,
                       step_clip: float | None,
                       dt_max: float,
                       dt_min: float,
                       pred_atol: float,
                       pred_rtol: float,
                       corr_factor: float,
                       step_growth: float,
                       step_shrink: float,
                       max_reject: int,
                       det_guard: float,
                       verbose: bool = False):
    """
    Adaptive continuation from t0 to t1 for one fixed z.
    """

    tau_cur = float(math.exp(t0))
    tau_end = float(math.exp(t1))
    zeta = complex(zeta0)
    y = complex(y0)

    # initial step in tau-space
    h = min(float(dt_max), max(float(dt_min), tau_end - tau_cur))
    rejects = 0

    while tau_cur < tau_end:
        h = min(h, tau_end - tau_cur)
        if h < float(dt_min):
            return zeta, y, False

        tau_try = tau_cur + h
        try:
            zeta_pred, y_pred, err_pred, detJ0, detJ1 = _predict_heun(
                z_fixed, tau_cur, tau_try, coeffs, zeta, y)
        except Exception:
            rejects += 1
            h *= float(step_shrink)
            if rejects > int(max_reject):
                return zeta, y, False
            continue

        # Geometric predictor trust from embedded defect and Jacobian.
        scale = max(1.0, abs(zeta_pred), abs(y_pred))
        pred_tol = float(pred_atol) + float(pred_rtol) * scale
        if err_pred > pred_tol or abs(detJ0) < float(det_guard) or \
                abs(detJ1) < float(det_guard):
            rejects += 1
            h *= float(step_shrink)
            if rejects > int(max_reject):
                return zeta, y, False
            continue

        zc, yc, ok, _, res_norm, corr_norm, detJc = _newton_project(
            z_fixed, tau_try, coeffs, zeta_pred, y_pred,
            max_iter=max_iter, tol_res=tol_res, tol_step=tol_step,
            armijo=armijo, min_lam=min_lam, step_clip=step_clip,
        )

        move = max(abs(zeta_pred - zeta), abs(y_pred - y), 1.0)
        corr_ok = (corr_norm <= float(corr_factor) * move)
        det_ok = (abs(detJc) >= float(det_guard)) if numpy.isfinite(abs(detJc)) else False

        if ok and corr_ok and det_ok and yc != 0:
            zeta = zc
            y = yc
            tau_cur = tau_try
            rejects = 0

            # conservative PI-free growth rule using predictor defect
            if err_pred < 0.1 * pred_tol and corr_norm < 0.25 * move:
                h = min(float(dt_max), float(step_growth) * h)
            elif err_pred < pred_tol and corr_norm < move:
                h = min(float(dt_max), 1.2 * h)
            # else keep h unchanged

            if verbose:
                pass
            continue

        rejects += 1
        h *= float(step_shrink)
        if rejects > int(max_reject):
            return zeta, y, False

    return zeta, y, True


# =================
# decompress newton
# =================

def decompress_newton(z_query, t, coeffs, w0_list=None, max_iter=50,
                      tol=1e-12, armijo=1e-4, min_lam=1e-6, w_min=1e-14,
                      sweep=True, max_substeps=8, **kwargs):
    """
    Free decompression by geometric continuation on the spectral curve.

    This function keeps the same API as earlier versions. The arguments
    ``sweep`` and ``max_substeps`` are accepted for compatibility, but the
    implementation uses adaptive tau stepping rather than recursive
    interval splitting.

    Parameters
    ----------
    z_query : array_like (n_z,)
        Query points z = x + i*delta.
    t : array_like (n_t,)
        Time grid, sorted increasing.
    coeffs : ndarray
        Polynomial coefficients of P(z, m) = 0.
    w0_list : array_like (n_z,)
        Initial branch values m(0, z_query).
    max_iter, tol, armijo, min_lam :
        Newton projector parameters.
    w_min : float
        Guard against division by zero in the characteristic map.
    sweep, max_substeps :
        Retained for API compatibility; not used as control heuristics here.

    Other Parameters
    ----------------
    dt_max : float, default=0.02
        Maximum tau-step used internally.
    dt_min : float, default=1e-6
        Minimum tau-step before declaring failure.
    pred_atol : float, default=1e-8
        Absolute predictor defect tolerance.
    pred_rtol : float, default=5e-3
        Relative predictor defect tolerance.
    corr_factor : float, default=5.0
        Accept if Newton correction is no more than this multiple of the
        predictor move.
    step_growth : float, default=1.5
        Accepted-step growth factor for tau step.
    step_shrink : float, default=0.5
        Rejected-step shrink factor for tau step.
    max_reject : int, default=30
        Maximum consecutive rejected attempts per outer time interval.
    det_guard : float, default=1e-14
        Reject steps whose Jacobian determinant magnitude is smaller than this.
    tol_step : float, default=None
        Newton step-size tolerance. If None, uses 0.1*tol.
    step_clip : float or None, default=None
        Optional clip on each Newton update infinity norm.
    verbose : bool, default=False
        Print per-time summary.

    Returns
    -------
    W : ndarray (n_t, n_z), complex
        Decompressed branch values.
    ok : ndarray (n_t, n_z), bool
        Success flags.
    """

    del sweep, max_substeps  # compatibility only

    z_query = numpy.asarray(z_query, dtype=complex).ravel()
    t = numpy.asarray(t, dtype=float).ravel()
    coeffs = numpy.asarray(coeffs, dtype=complex)

    if t.size == 0:
        return (numpy.empty((0, z_query.size), dtype=complex),
                numpy.empty((0, z_query.size), dtype=bool))
    if numpy.any(numpy.diff(t) < 0.0):
        raise ValueError('t must be sorted increasing.')
    if w0_list is None:
        raise ValueError('w0_list must be provided (physical branch at t=0).')

    w0_list = numpy.asarray(w0_list, dtype=complex).ravel()
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
    if step_clip is not None:
        step_clip = float(step_clip)
    verbose = bool(kwargs.get('verbose', False))

    n_t = t.size
    n_z = z_query.size
    tau = numpy.exp(t)

    W = numpy.full((n_t, n_z), numpy.nan + 1j * numpy.nan, dtype=complex)
    ok = numpy.zeros((n_t, n_z), dtype=bool)

    # initial state at t[0]
    W[0, :] = w0_list
    ok[0, :] = numpy.isfinite(w0_list)

    zeta_state = z_query.astype(complex).copy()
    y_state = tau[0] * w0_list.astype(complex)

    tiny = numpy.abs(y_state) < float(w_min)
    if numpy.any(tiny):
        y_state[tiny] += float(w_min)

    for k in range(1, n_t):
        t0 = float(t[k - 1])
        t1 = float(t[k])
        n_ok = 0

        for j in range(n_z):
            zeta0 = zeta_state[j]
            y0 = y_state[j]

            if (not numpy.isfinite(zeta0)) or (not numpy.isfinite(y0)) or \
                    abs(y0) < float(w_min):
                zeta0 = z_query[j]
                y0 = tau[k - 1] * w0_list[j]
                if abs(y0) < float(w_min):
                    y0 += float(w_min)

            zeta1, y1, success = _advance_one_point(
                z_fixed=z_query[j],
                t0=t0,
                t1=t1,
                coeffs=coeffs,
                zeta0=zeta0,
                y0=y0,
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
                verbose=False,
            )

            if success and numpy.isfinite(zeta1) and numpy.isfinite(y1):
                w = y1 / tau[k]
                if abs(w) < float(w_min):
                    success = False
                else:
                    W[k, j] = w
                    ok[k, j] = True
                    zeta_state[j] = zeta1
                    y_state[j] = y1
                    n_ok += 1

            if not success:
                # Keep the previous time value as a finite fallback so
                # downstream plotting does not receive all-NaN rows.
                W[k, j] = W[k - 1, j]
                ok[k, j] = False

        if verbose:
            print(f"decompress10: k={k}/{n_t-1}, t={t1:.6e}, "
                  f"ok={n_ok}/{n_z}")

    # Final cleanup so downstream plotting always receives finite positive
    # densities in log mode.
    delta = float(abs(z_query[0].imag)) if n_z > 0 else 1e-8
    x_abs = numpy.abs(z_query.real)
    floor_im = delta / (x_abs * x_abs + delta * delta)
    floor_im = numpy.asarray(floor_im, dtype=float)
    floor_im[~numpy.isfinite(floor_im)] = 1.0
    floor_im = numpy.maximum(floor_im, 1e-300)

    for k in range(n_t):
        for j in range(n_z):
            w = W[k, j]
            wr = w.real
            wi = w.imag

            if (not numpy.isfinite(wr)) or (k > 0 and not numpy.isfinite(wr)):
                if k > 0 and numpy.isfinite(W[k - 1, j].real):
                    wr = W[k - 1, j].real
                elif numpy.isfinite(w0_list[j].real):
                    wr = w0_list[j].real
                else:
                    wr = 0.0

            if (not numpy.isfinite(wi)) or (wi <= 0.0):
                if k > 0 and numpy.isfinite(W[k - 1, j].imag) and W[k - 1, j].imag > 0.0:
                    wi = max(W[k - 1, j].imag, floor_im[j])
                elif numpy.isfinite(w0_list[j].imag) and w0_list[j].imag > 0.0:
                    wi = max(w0_list[j].imag, floor_im[j])
                else:
                    wi = floor_im[j]
                ok[k, j] = False

            W[k, j] = complex(float(wr), float(wi))

    return W, ok
