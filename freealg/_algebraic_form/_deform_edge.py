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
from ._continuation_algebraic import eval_roots

__all__ = ['evolve_edges', 'merge_edges', 'evolve_edges_with_births',
           'evolve_edges_from_states', 'scan_edges_at_time',
           'third_cusp_residual']


# ===================
# eval P all partials
# ===================

def _eval_P_all_partials(zeta, y, coeffs):
    """
    Evaluate P and first/second partial derivatives.

    Returns
    -------

    P, Pz, Py, Pzz, Pzy, Pyy : complex
    """

    a = numpy.asarray(coeffs, dtype=numpy.complex128)
    deg_z = a.shape[0] - 1
    deg_y = a.shape[1] - 1

    zeta = numpy.complex128(zeta)
    y = numpy.complex128(y)

    zi = numpy.power(zeta, numpy.arange(deg_z + 1, dtype=numpy.int64))
    yj = numpy.power(y, numpy.arange(deg_y + 1, dtype=numpy.int64))

    P = numpy.sum(a * zi[:, None] * yj[None, :])

    if deg_z >= 1:
        iz = numpy.arange(1, deg_z + 1, dtype=numpy.int64)
        zi_m1 = numpy.power(zeta, iz - 1)
        Pz = numpy.sum((a[iz, :] * iz[:, None]) *
                       zi_m1[:, None] * yj[None, :])
    else:
        Pz = 0.0 + 0.0j

    if deg_y >= 1:
        jy = numpy.arange(1, deg_y + 1, dtype=numpy.int64)
        yj_m1 = numpy.power(y, jy - 1)
        Py = numpy.sum((a[:, jy] * jy[None, :]) *
                       zi[:, None] * yj_m1[None, :])
    else:
        Py = 0.0 + 0.0j

    if deg_z >= 2:
        iz = numpy.arange(2, deg_z + 1, dtype=numpy.int64)
        zi_m2 = numpy.power(zeta, iz - 2)
        Pzz = numpy.sum((a[iz, :] * (iz * (iz - 1))[:, None]) *
                        zi_m2[:, None] * yj[None, :])
    else:
        Pzz = 0.0 + 0.0j

    if deg_y >= 2:
        jy = numpy.arange(2, deg_y + 1, dtype=numpy.int64)
        yj_m2 = numpy.power(y, jy - 2)
        Pyy = numpy.sum((a[:, jy] * (jy * (jy - 1))[None, :]) *
                        zi[:, None] * yj_m2[None, :])
    else:
        Pyy = 0.0 + 0.0j

    if (deg_z >= 1) and (deg_y >= 1):
        iz = numpy.arange(1, deg_z + 1, dtype=numpy.int64)
        jy = numpy.arange(1, deg_y + 1, dtype=numpy.int64)
        zi_m1 = numpy.power(zeta, iz - 1)
        yj_m1 = numpy.power(y, jy - 1)
        coeff = a[numpy.ix_(iz, jy)] * (iz[:, None] * jy[None, :])
        Pzy = numpy.sum(coeff * zi_m1[:, None] * yj_m1[None, :])
    else:
        Pzy = 0.0 + 0.0j

    return complex(P), complex(Pz), complex(Py), \
        complex(Pzz), complex(Pzy), complex(Pyy)


# =========
# solve 2x2
# =========

def _solve2x2(a11, a12, a21, a22, b1, b2, det_guard=1e-14):
    """
    Solve a 2x2 linear system.
    """

    det = a11 * a22 - a12 * a21
    scale = max(abs(a11), abs(a12), abs(a21), abs(a22), 1.0)
    if abs(det) <= float(det_guard) * (scale * scale):
        return 0.0 + 0.0j, 0.0 + 0.0j, det, False

    x1 = (b1 * a22 - a12 * b2) / det
    x2 = (a11 * b2 - b1 * a21) / det
    return x1, x2, det, True


# ==============
# safe rel delta
# ==============

def _safe_rel_delta(a, b, rel_eps=1e-12):
    """
    Relative difference guarded near zero.
    """

    den = max(abs(a), abs(b), float(rel_eps))
    return abs(a - b) / den


# ==========
# state move
# ==========

def _state_move(zeta_new, y_new, zeta_old, y_old, log=False,
                rel_eps=1e-12):
    """
    Norm for state change.
    """

    if not log:
        return max(abs(zeta_new - zeta_old), abs(y_new - y_old), 1.0)

    return max(
        _safe_rel_delta(zeta_new, zeta_old, rel_eps),
        _safe_rel_delta(y_new, y_old, rel_eps),
        float(rel_eps))


# ====================
# outside support seed
# ====================

def _outside_support_seed(x_edge, side, width, log=False, frac=0.02):
    """
    Return a seed point slightly outside a supplied support endpoint.

    Histogram/support endpoints can be slightly inside the true polynomial
    support.  Seeding Newton from the inside can latch to a nearby ghost
    branch point.  For physical edge initialization, probe from the gap side:
    left endpoints from the left, right endpoints from the right.
    """

    x_edge = float(x_edge)
    width = float(abs(width))
    frac = float(frac)

    if log and x_edge > 0.0 and width > 0.0:
        # width is interpreted as log-width by the caller in log mode.
        shift = numpy.exp(frac * width)
        if side < 0:
            return x_edge / shift
        return x_edge * shift

    shift = frac * max(width, abs(x_edge), 1.0e-12)
    if side < 0:
        return x_edge - shift
    return x_edge + shift


# ==================
# deform pushforward
# ==================

def _deform_pushforward(tau, zeta, y, c0, w_min=1e-14):
    """
    Physical edge coordinate for free deformation.

    On the initial curve P(zeta, y)=0, define the companion coordinate
        wbar = c0*y + (c0 - 1)/zeta.
    The target physical coordinate is
        x = tau*zeta + (tau - 1)/wbar.
    This is written as tau*zeta + (tau - 1)*zeta/A with
        A = c0*y*zeta + c0 - 1.
    """

    zeta = complex(zeta)
    y = complex(y)
    c0 = float(c0)
    tau = float(tau)

    A = c0 * y * zeta + (c0 - 1.0)
    if abs(A) < float(w_min):
        if A == 0.0:
            A = complex(float(w_min), 0.0)
        else:
            A = A + float(w_min) * A / abs(A)

    return tau * zeta + (tau - 1.0) * zeta / A


# ========================
# residual jacobian branch
# ========================

def _residual_jacobian_branch(zeta, y, coeffs):
    """
    Residual/Jacobian for t=0 branch point system:
        P(zeta, y) = 0,
        Py(zeta, y) = 0.
    """

    P, Pz, Py, _Pzz, Pzy, Pyy = _eval_P_all_partials(zeta, y, coeffs)

    F1 = P
    F2 = Py

    J11 = Pz
    J12 = Py
    J21 = Pzy
    J22 = Pyy

    return F1, F2, J11, J12, J21, J22


# ==========================
# residual jacobian edge tau
# ==========================

def _residual_jacobian_edge_tau(tau, zeta, y, coeffs, c0):
    """
    Residual/Jacobian for the free-deformation edge system.

    The projection restricted to P(zeta, y)=0 is

        x = tau*zeta + (tau - 1) / wbar,
        wbar = c0*y + (c0 - 1)/zeta.

    With A = c0*y*zeta + c0 - 1, an algebraically cleared critical
    equation is

        tau*A**2*Py
        + (tau - 1)*((c0 - 1)*Py + c0*zeta**2*Pz) = 0.

    At tau=1 this reduces to A**2*Py=0, and the initialized physical
    branch points satisfy Py=0.
    """

    tau = float(tau)
    c0 = float(c0)
    P, Pz, Py, Pzz, Pzy, Pyy = _eval_P_all_partials(zeta, y, coeffs)

    A = c0 * y * zeta + (c0 - 1.0)
    B = (c0 - 1.0) * Py + c0 * (zeta * zeta) * Pz

    F1 = P
    F2 = tau * (A * A) * Py + (tau - 1.0) * B

    Az = c0 * y
    Ay = c0 * zeta

    Bz = (c0 - 1.0) * Pzy + c0 * (2.0 * zeta * Pz +
                                  (zeta * zeta) * Pzz)
    By = (c0 - 1.0) * Pyy + c0 * (zeta * zeta) * Pzy

    J11 = Pz
    J12 = Py
    J21 = tau * (2.0 * A * Az * Py + (A * A) * Pzy) + \
        (tau - 1.0) * Bz
    J22 = tau * (2.0 * A * Ay * Py + (A * A) * Pyy) + \
        (tau - 1.0) * By

    Ftau = (A * A) * Py + B

    return F1, F2, J11, J12, J21, J22, Ftau


# ===================
# newton project real
# ===================

def _newton_project_real(zeta0, y0, residual_jacobian,
                         max_iter=50, tol_res=1e-12, tol_step=None,
                         armijo=1e-4, min_lam=1e-6,
                         det_guard=1e-14, rel_eps=1e-12):
    """
    Damped Newton projection for real edge states.

    For Hermitian/real covariance laws, physical spectral edges live on the
    real slice of the spectral curve. This real projection prevents a nearly
    real physical edge from drifting onto a nearby complex/ghost critical
    branch when the fitted polynomial has small numerical imperfections.
    """

    if tol_step is None:
        tol_step = 0.1 * float(tol_res)

    zeta = float(numpy.real(zeta0))
    y = float(numpy.real(y0))

    for _ in range(int(max_iter)):
        F1, F2, J11, J12, J21, J22 = residual_jacobian(
            complex(zeta), complex(y))

        F1 = float(numpy.real(F1))
        F2 = float(numpy.real(F2))
        J11 = float(numpy.real(J11))
        J12 = float(numpy.real(J12))
        J21 = float(numpy.real(J21))
        J22 = float(numpy.real(J22))

        z_scale = max(1.0, abs(zeta))
        y_scale = max(1.0, abs(y))
        row1_scale = 1.0 + abs(J11) * z_scale + abs(J12) * y_scale
        row2_scale = 1.0 + abs(J21) * z_scale + abs(J22) * y_scale
        res_rel = max(abs(F1) / row1_scale, abs(F2) / row2_scale)
        res_norm = max(abs(F1), abs(F2))
        if (res_norm <= float(tol_res)) or (res_rel <= float(tol_res)):
            return complex(zeta), complex(y), True

        det = J11 * J22 - J12 * J21
        scale = max(abs(J11), abs(J12), abs(J21), abs(J22), 1.0)
        if abs(det) <= float(det_guard) * scale * scale:
            return complex(zeta), complex(y), False

        dzeta = (-F1 * J22 + J12 * F2) / det
        dy = (-J11 * F2 + F1 * J21) / det

        step_norm = max(abs(dzeta) / max(abs(zeta), rel_eps),
                        abs(dy) / max(abs(y), rel_eps))
        if step_norm <= float(tol_step):
            return complex(zeta + dzeta), complex(y + dy), True

        lam = 1.0
        accepted = False
        while lam >= float(min_lam):
            z_try = zeta + lam * dzeta
            y_try = y + lam * dy
            F1_t, F2_t, *_ = residual_jacobian(complex(z_try), complex(y_try))
            res_try = max(abs(float(numpy.real(F1_t))),
                          abs(float(numpy.real(F2_t))))
            if res_try <= (1.0 - float(armijo) * lam) * res_norm:
                zeta = z_try
                y = y_try
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            return complex(zeta), complex(y), False

        if not (numpy.isfinite(zeta) and numpy.isfinite(y)):
            return complex(zeta), complex(y), False

    return complex(zeta), complex(y), False


# ==============
# newton project
# ==============

def _newton_project(zeta0, y0, residual_jacobian, max_iter=50, tol_res=1e-12,
                    tol_step=None, armijo=1e-4, min_lam=1e-6, step_clip=-1.0,
                    det_guard=1e-14, log=False, rel_eps=1e-12):
    """
    Damped Newton projection for a 2x2 complex system.
    """

    if tol_step is None:
        tol_step = 0.1 * float(tol_res)

    zeta = complex(zeta0)
    y = complex(y0)

    for _ in range(int(max_iter)):
        F1, F2, J11, J12, J21, J22 = residual_jacobian(zeta, y)

        # Mixed absolute/relative backward-error test.  Pure absolute
        # tolerances are too strict in log-mode small-edge regimes where the
        # second edge equation is a cancellation of large terms.  Scale each
        # residual by the local linearized row magnitude.
        z_scale = max(1.0, abs(zeta))
        y_scale = max(1.0, abs(y))
        row1_scale = 1.0 + abs(J11) * z_scale + abs(J12) * y_scale
        row2_scale = 1.0 + abs(J21) * z_scale + abs(J22) * y_scale
        res_rel = max(abs(F1) / row1_scale, abs(F2) / row2_scale)
        res_norm = max(abs(F1), abs(F2))
        if (res_norm <= float(tol_res)) or (res_rel <= float(tol_res)):
            return zeta, y, True

        dzeta, dy, _det, ok = _solve2x2(
            J11, J12, J21, J22, -F1, -F2, det_guard=det_guard)
        if not ok:
            return zeta, y, False

        if step_clip is not None and float(step_clip) > 0.0:
            step_inf = max(abs(dzeta), abs(dy))
            if step_inf > float(step_clip):
                scl = float(step_clip) / step_inf
                dzeta *= scl
                dy *= scl

        step_norm = _state_move(
            zeta + dzeta, y + dy, zeta, y, log=log,
            rel_eps=rel_eps)
        if step_norm <= float(tol_step):
            return zeta + dzeta, y + dy, True

        lam = 1.0
        accepted = False
        while lam >= float(min_lam):
            z_try = zeta + lam * dzeta
            y_try = y + lam * dy
            F1_t, F2_t, *_ = residual_jacobian(z_try, y_try)
            res_try = max(abs(F1_t), abs(F2_t))
            if res_try <= (1.0 - float(armijo) * lam) * res_norm:
                zeta = z_try
                y = y_try
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            return zeta, y, False

        if not (numpy.isfinite(zeta.real) and numpy.isfinite(zeta.imag) and
                numpy.isfinite(y.real) and numpy.isfinite(y.imag)):
            return zeta, y, False

    return zeta, y, False


# ===========================
# branch root from real curve
# ===========================

def _branch_root_from_real_curve(x_edge, coeffs, target=None):
    """
    Candidate y on the real slice P(x_edge, y)=0.

    If a Stieltjes target is available, use it to identify the physical sheet
    by continuity and select the polynomial root closest to that target. Only
    use |Py| as a secondary tie-breaker. Without a target, fall back to the
    root with smallest |Py|.
    """

    z = complex(float(x_edge))
    roots = numpy.asarray(eval_roots(numpy.array([z]), coeffs)[0],
                          dtype=numpy.complex128).ravel()
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan

    have_target = (target is not None and numpy.isfinite(target.real) and
                   numpy.isfinite(target.imag))

    scores = []
    for rr in roots:
        _P, _Pz, Py, _Pzz, _Pzy, _Pyy = _eval_P_all_partials(z, rr, coeffs)
        if have_target:
            dist = abs(rr - target)
            scores.append((dist, abs(Py), abs(rr.imag), rr))
        else:
            scores.append((abs(Py), abs(rr.imag), rr))

    scores.sort(key=lambda item: item[:-1])
    return complex(scores[0][-1])


# ============================
# init edge point from support
# ============================

def _init_edge_point_from_support(x_edge, coeffs, stieltjes=None, delta=1e-5,
                                  log=False, max_iter=80, tol=1e-12):
    """
    Initialize (zeta, y) at t=0 for an edge near x_edge.

    This routine uses the Stieltjes evaluator to obtain the physical-branch
    seed whenever available, then projects onto the true branch-point system
    P = 0, Py = 0.
    """

    x_edge = float(x_edge)
    zeta0 = complex(x_edge)

    y_seed = numpy.nan + 1j * numpy.nan
    probe_delta = float(abs(delta))

    if stieltjes is not None:
        try:
            if log:
                probe_delta = max(probe_delta, 1e-12 * max(1.0, abs(x_edge)))
            z_probe = complex(x_edge + 1j * probe_delta)
            y_seed = complex(numpy.asarray(stieltjes(z_probe)).reshape(-1)[0])
        except Exception:
            y_seed = numpy.nan + 1j * numpy.nan

    if not (numpy.isfinite(y_seed.real) and numpy.isfinite(y_seed.imag)):
        y_seed = _branch_root_from_real_curve(x_edge, coeffs, target=None)
    else:
        y_seed = _branch_root_from_real_curve(x_edge, coeffs, target=y_seed)

    if not (numpy.isfinite(y_seed.real) and numpy.isfinite(y_seed.imag)):
        return zeta0, y_seed, False

    def residual_jacobian(zeta, y):
        return _residual_jacobian_branch(zeta, y, coeffs)

    # First try the real physical slice.  For real spectral edges, this avoids
    # initializing on a nearby complex ghost branch.
    zeta_r, y_r, ok_r = _newton_project_real(
        zeta0.real, y_seed.real, residual_jacobian,
        max_iter=max_iter, tol_res=tol, tol_step=0.1 * tol,
        armijo=1e-4, min_lam=1e-6,
        det_guard=1e-14, rel_eps=1e-12)
    if ok_r:
        return complex(zeta_r.real), complex(y_r.real), True

    zeta, y, ok = _newton_project(
        zeta0, y_seed, residual_jacobian,
        max_iter=max_iter, tol_res=tol, tol_step=0.1 * tol,
        armijo=1e-4, min_lam=1e-6, step_clip=-1.0,
        det_guard=1e-14, log=log, rel_eps=1e-12)

    if ok and abs(zeta.imag) <= 100.0 * tol:
        zeta = complex(zeta.real)
    if ok and abs(y.imag) <= 100.0 * tol:
        y = complex(y.real)

    if not ok:
        return numpy.nan + 1j * numpy.nan, numpy.nan + 1j * numpy.nan, False

    return zeta, y, ok


# ===================
# edge tangent in tau
# ===================

def _edge_tangent_tau(tau, zeta, y, coeffs, c0, det_guard=1e-14):
    """
    Tangent d(zeta, y)/d tau along the edge continuation system.
    """

    _F1, _F2, J11, J12, J21, J22, Ftau = _residual_jacobian_edge_tau(
        tau, zeta, y, coeffs, c0)
    rhs1 = 0.0 + 0.0j
    rhs2 = -Ftau
    dzeta, dy, _det, ok = _solve2x2(
        J11, J12, J21, J22, rhs1, rhs2, det_guard=det_guard)
    return dzeta, dy, ok


# ================
# predict Heun tau
# ================

def _predict_heun_tau(tau0, tau1, zeta0, y0, coeffs, c0, det_guard=1e-14):
    """
    Heun predictor in tau.
    """

    h = float(tau1) - float(tau0)
    dz0, dy0, ok0 = _edge_tangent_tau(tau0, zeta0, y0, coeffs, c0,
                                      det_guard=det_guard)
    if not ok0:
        return zeta0, y0, False

    z_e = zeta0 + h * dz0
    y_e = y0 + h * dy0

    dz1, dy1, ok1 = _edge_tangent_tau(tau1, z_e, y_e, coeffs, c0,
                                      det_guard=det_guard)
    if not ok1:
        return z_e, y_e, False

    z_h = zeta0 + 0.5 * h * (dz0 + dz1)
    y_h = y0 + 0.5 * h * (dy0 + dy1)
    return z_h, y_h, True


# ================
# edge newton step
# ================

def _edge_newton_step(t, zeta, y, coeffs, c0, max_iter=50, tol=1e-12,
                      log=False):
    """
    Newton projection onto the edge system at fixed time `t`.
    """

    tau = float(numpy.exp(float(t)))

    def residual_jacobian(zeta_, y_):
        F1, F2, J11, J12, J21, J22, _Pz = \
            _residual_jacobian_edge_tau(tau, zeta_, y_, coeffs, c0)
        return F1, F2, J11, J12, J21, J22

    zeta1, y1, ok = _newton_project_real(
        zeta, y, residual_jacobian,
        max_iter=max_iter, tol_res=tol, tol_step=0.1 * tol,
        armijo=1e-4, min_lam=1e-6,
        det_guard=1e-14, rel_eps=1e-12)

    if ok:
        return complex(zeta1.real), complex(y1.real), True

    zeta1, y1, ok = _newton_project(
        zeta, y, residual_jacobian,
        max_iter=max_iter, tol_res=tol, tol_step=0.1 * tol,
        armijo=1e-4, min_lam=1e-6, step_clip=-1.0,
        det_guard=1e-14, log=log, rel_eps=1e-12)

    return zeta1, y1, ok


# =================
# advance edge step
# =================

def _advance_edge_step(t0, t1, zeta0, y0, coeffs, c0, max_iter=50, tol=1e-12,
                       log=False):
    """
    One predictor/corrector continuation step from t0 to t1.
    """

    tau0 = float(numpy.exp(float(t0)))
    tau1 = float(numpy.exp(float(t1)))

    z_pred, y_pred, ok_pred = _predict_heun_tau(
        tau0, tau1, zeta0, y0, coeffs, c0, det_guard=1e-14)
    if not ok_pred:
        z_pred, y_pred = zeta0, y0

    zeta1, y1, ok = _edge_newton_step(
        t1, z_pred, y_pred, coeffs, c0,
        max_iter=max_iter, tol=tol, log=log)

    if not ok:
        # Fallback: project old state directly at the new time.
        zeta1, y1, ok = _edge_newton_step(
            t1, zeta0, y0, coeffs, c0,
            max_iter=max_iter, tol=tol, log=log)

    return zeta1, y1, ok


# ============
# evolve edges
# ============

def evolve_edges(t_grid, coeffs, support=None, stieltjes=None, c0=1.0,
                 delta=1e-5, dt_max=0.1, max_iter=50, tol=1e-12,
                 return_preimage=False, log=False):
    """
    Evolve spectral edges under free deformation using the fitted polynomial
    P.

    Solves for (zeta(t), y(t)) on the spectral curve:
        P(zeta,y) = 0,
        y^2 * Py(zeta,y) - (exp(t)-1) * Pzeta(zeta,y) = 0,

    then maps to physical coordinate:
        x_edge(t) = zeta - (exp(t)-1)/y.

    Parameters
    ----------

    stieltjes : callable or None, default=None
        Evaluator of the fitted physical Stieltjes branch. When provided, it is
        used only to seed the correct sheet at t=0.

    log : bool, default=False
        Enables relative state metrics appropriate for log-scale problems.
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 1:
        raise ValueError('t_grid must be non-empty.')
    if t_grid.size > 1 and numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError('t_grid must be strictly increasing.')

    if support is None:
        raise ValueError('support must be provided (auto-detection not ' +
                         'implemented).')

    endpoints0 = []
    seedpoints0 = []
    support_list = [(float(a), float(b)) for a, b in support]
    n_bulk = len(support_list)

    for ib, (a, b) in enumerate(support_list):
        endpoints0.append(a)
        endpoints0.append(b)

        if log and a > 0.0 and b > 0.0:
            width = numpy.log(b / a)
        else:
            width = b - a

        # Use outside probes only for the two global exterior edges.
        #
        # For histogram-based support, the global exterior endpoints may be
        # slightly inside the true algebraic support; probing from the exterior
        # avoids latching to a ghost branch. For internal gap endpoints, the
        # same gap-side probe can jump to a nonphysical branch point in the
        # gap, so keep the old exact-endpoint behavior.
        if ib == 0:
            seedpoints0.append(_outside_support_seed(
                a, side=-1, width=width, log=log))
        else:
            seedpoints0.append(a)

        if ib == n_bulk - 1:
            seedpoints0.append(_outside_support_seed(
                b, side=+1, width=width, log=log))
        else:
            seedpoints0.append(b)

    m = len(endpoints0)
    complex_edges = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
    ok = numpy.zeros((t_grid.size, m), dtype=bool)

    if return_preimage:
        zeta_hist = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
        y_hist = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
    else:
        zeta_hist = None
        y_hist = None

    zeta = numpy.empty(m, dtype=numpy.complex128)
    y = numpy.empty(m, dtype=numpy.complex128)

    for j in range(m):
        z0, y0, ok0 = _init_edge_point_from_support(
            seedpoints0[j], coeffs, stieltjes=stieltjes, delta=delta,
            log=log, max_iter=max_iter, tol=tol)

        # If the outside probe fails for any reason, fall back to the exact
        # user-supplied endpoint to preserve the old behavior.
        if not ok0:
            z0, y0, ok0 = _init_edge_point_from_support(
                endpoints0[j], coeffs, stieltjes=stieltjes, delta=delta,
                log=log, max_iter=max_iter, tol=tol)
        zeta[j] = z0
        y[j] = y0
        ok[0, j] = ok0
        complex_edges[0, j] = z0

    if return_preimage:
        zeta_hist[0, :] = zeta
        y_hist[0, :] = y

    for it in range(1, t_grid.size):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])
        dt = t1 - t0

        n_sub = max(1, int(numpy.ceil(dt / float(dt_max))))

        z_cur = zeta.copy()
        y_cur = y.copy()
        ok_cur = ok[it - 1, :].copy()

        for ks in range(1, n_sub + 1):
            ts0 = t0 + dt * ((ks - 1) / float(n_sub))
            ts1 = t0 + dt * (ks / float(n_sub))
            for j in range(m):
                if not ok_cur[j]:
                    continue
                z_new, y_new, okj = _advance_edge_step(
                    ts0, ts1, z_cur[j], y_cur[j], coeffs, c0,
                    max_iter=max_iter, tol=tol, log=log)
                ok_cur[j] = bool(okj)
                z_cur[j] = z_new
                y_cur[j] = y_new
                if not (numpy.isfinite(z_new.real) and
                        numpy.isfinite(y_new.real)
                        and numpy.isfinite(z_new.imag) and
                        numpy.isfinite(y_new.imag)):
                    ok_cur[j] = False

        zeta[:] = z_cur
        y[:] = y_cur
        ok[it, :] = ok_cur

        tau = float(numpy.exp(t1))
        # c = tau - 1.0
        complex_edges[it, :] = numpy.asarray([
            _deform_pushforward(tau, zeta[j], y[j], c0)
            for j in range(m)], dtype=numpy.complex128)

        if return_preimage:
            zeta_hist[it, :] = zeta
            y_hist[it, :] = y

    if return_preimage:
        return complex_edges, ok, zeta_hist, y_hist
    return complex_edges, ok


# ===========
# merge edges
# ===========

def merge_edges(edges, tol=0.0):
    """
    Merge bulks when inner edges cross, without shifting columns.
    """

    edges = numpy.asarray(edges, dtype=float)
    nt, m = edges.shape
    if m % 2 != 0:
        raise ValueError('edges must have even number of columns.')
    k0 = m // 2

    edges2 = edges.copy()
    active_k = numpy.zeros(nt, dtype=int)

    for it in range(nt):
        row = edges2[it, :].copy()
        a = row[0::2].copy()
        b = row[1::2].copy()

        blocks = []
        for j in range(k0):
            if numpy.isfinite(a[j]) and numpy.isfinite(b[j]) and (b[j] > a[j]):
                blocks.append([j, j])

        if len(blocks) == 0:
            active_k[it] = 0
            edges2[it, :] = row
            continue

        def left_edge(block):
            return a[block[0]]

        def right_edge(block):
            return b[block[1]]

        merged = True
        while merged and (len(blocks) > 1):
            merged = False
            new_blocks = [blocks[0]]
            for blk in blocks[1:]:
                prev = new_blocks[-1]
                if numpy.isfinite(right_edge(prev)) and \
                        numpy.isfinite(left_edge(blk)) and \
                        (right_edge(prev) >= left_edge(blk) - float(tol)):

                    bj = prev[1]
                    aj = blk[0]
                    b[bj] = numpy.nan
                    a[aj] = numpy.nan
                    prev[1] = blk[1]
                    merged = True
                else:
                    new_blocks.append(blk)
            blocks = new_blocks

        active_k[it] = len(blocks)
        row2 = row.copy()
        row2[0::2] = a
        row2[1::2] = b
        edges2[it, :] = row2

    return edges2, active_k


# ==============
# is inside bulk
# ==============

def _is_inside_bulk(edges_row, x, tol=0.0):
    """
    Returns True if x lies inside any finite bulk interval [a,b] in edges_row.
    """

    row = numpy.asarray(edges_row, dtype=float)
    a = row[0::2]
    b = row[1::2]
    for aj, bj in zip(a, b):
        if numpy.isfinite(aj) and numpy.isfinite(bj) and (bj > aj):
            if (x >= aj - tol) and (x <= bj + tol):
                return True
    return False


# =======================
# bulk index containing x
# =======================

def _bulk_index_containing_x(edges_row, x, tol=0.0):
    """
    Return bulk index j such that x is inside [a_j, b_j] for the given row.
    """

    row = numpy.asarray(edges_row, dtype=float)
    a = row[0::2]
    b = row[1::2]
    for j, (aj, bj) in enumerate(zip(a, b)):
        if numpy.isfinite(aj) and numpy.isfinite(bj) and (bj > aj):
            if (x >= aj - tol) and (x <= bj + tol):
                return j
    return None


# ==============
# first index ge
# ==============

def _first_index_ge(t_grid, t0):
    """First index i with t_grid[i] >= t0. Returns None if none."""

    i = numpy.searchsorted(t_grid, t0, side='left')
    if i >= t_grid.size:
        return None
    return int(i)


# =======================
# evolve edges with birth
# =======================

def evolve_edges_with_births(t_grid, coeffs, support=None, cusps=None,
                             stieltjes=None, c0=1.0, delta=1e-5, dt_max=0.1,
                             max_iter=50, tol=1e-12,
                             return_preimage=False, split_tol=0.0,
                             seed_eps=1e-6, fill_gap='linear',
                             log=False):
    """
    Evolve edges like evolve_edges(), with deformed-cusp merge/death handling.

    This function intentionally keeps the same name/signature as the free
    counterpart so AlgebraicForm can dispatch both files uniformly.  In the
    deformed case, however, a physical cusp is a merge of two existing adjacent
    edges, not a birth of two new edges.  Therefore this routine never inserts
    new columns.  It evolves the initial physical edges and, for each validated
    cusp supplied by _deform_cusp_wrap, sets the colliding adjacent columns to
    NaN after the merge time.
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 1:
        raise ValueError('t_grid must be non-empty.')
    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError('t_grid must be strictly increasing.')

    if return_preimage:
        edges_ext, ok_ext, zeta_ext, y_ext = evolve_edges(
            t_grid, coeffs, support=support, stieltjes=stieltjes, c0=c0,
            delta=delta, dt_max=dt_max, max_iter=max_iter, tol=tol,
            return_preimage=True, log=log)
    else:
        edges_ext, ok_ext = evolve_edges(
            t_grid, coeffs, support=support, stieltjes=stieltjes, c0=c0,
            delta=delta, dt_max=dt_max, max_iter=max_iter, tol=tol,
            return_preimage=False, log=log)
        zeta_ext = None
        y_ext = None

    if cusps is None or len(cusps) == 0:
        if return_preimage:
            return edges_ext, ok_ext, zeta_ext, y_ext
        return edges_ext, ok_ext

    def _cusp_tuple(c):
        if not isinstance(c, dict):
            return None
        t_star = float(c.get('t', numpy.nan))
        x_star = float(c.get('x', numpy.nan))
        if not (numpy.isfinite(t_star) and numpy.isfinite(x_star)):
            return None
        return x_star, t_star

    def _find_adjacent_pair(row, x_star):
        vals = []
        for j in range(row.size):
            xj = float(numpy.real(row[j]))
            if numpy.isfinite(xj):
                vals.append((xj, j))
        if len(vals) < 2:
            return None

        vals.sort(key=lambda q: q[0])

        # For a deformed cusp/merge, the dying pair is the adjacent pair
        # whose midpoint is closest to the cusp location. Do NOT prefer a
        # bracketing pair. Near a sampled pre-cusp time the true colliding
        # pair can both lie slightly to one side of x_star, while the outer
        # neighbor brackets x_star and would be killed incorrectly.
        best = None
        best_score = None
        for k in range(len(vals) - 1):
            x0, j0 = vals[k]
            x1, j1 = vals[k + 1]
            mid = 0.5 * (x0 + x1)
            gap = abs(x1 - x0)
            mid_dist = abs(mid - x_star)

            # Lexicographic score: first choose the pair whose midpoint is
            # closest to the solved cusp, then prefer the tighter gap. This
            # preserves the physical merge pair without relying on sampling
            # exactly at t_star.
            score = (mid_dist, gap)
            if best_score is None or score < best_score:
                best_score = score
                best = (j0, j1)

        return best

    # Deduplicate very close cusp reports before applying deaths.
    raw = []
    for c in cusps:
        q = _cusp_tuple(c)
        if q is not None:
            raw.append(q)
    raw.sort(key=lambda q: (q[1], q[0]))

    uniq = []
    for x_star, t_star in raw:
        if uniq and abs(t_star - uniq[-1][1]) <= 1e-5 and \
                abs(x_star - uniq[-1][0]) <= 1e-5:
            continue
        uniq.append((x_star, t_star))

    killed = set()
    for x_star, t_star in uniq:
        it_ge = _first_index_ge(t_grid, t_star)
        if it_ge is None:
            continue
        it_prev = max(0, it_ge - 1)

        pair = _find_adjacent_pair(numpy.real(edges_ext[it_prev, :]), x_star)
        if pair is None:
            continue
        j0, j1 = int(pair[0]), int(pair[1])

        # Do not re-apply duplicated or overlapping deaths.
        if j0 in killed or j1 in killed:
            continue
        killed.add(j0)
        killed.add(j1)

        # Deformed cusp is a merge/death: the two colliding inner edges cease
        # to be physical after the cusp.  Keep pre-cusp history intact.
        edges_ext[it_ge:, j0] = numpy.nan + 1j * numpy.nan
        edges_ext[it_ge:, j1] = numpy.nan + 1j * numpy.nan
        ok_ext[it_ge:, j0] = False
        ok_ext[it_ge:, j1] = False

        if return_preimage:
            zeta_ext[it_ge:, j0] = numpy.nan + 1j * numpy.nan
            zeta_ext[it_ge:, j1] = numpy.nan + 1j * numpy.nan
            y_ext[it_ge:, j0] = numpy.nan + 1j * numpy.nan
            y_ext[it_ge:, j1] = numpy.nan + 1j * numpy.nan

    if return_preimage:
        return edges_ext, ok_ext, zeta_ext, y_ext
    return edges_ext, ok_ext


# ===================
# third cusp residual
# ===================

def third_cusp_residual(zeta, y, coeffs, c0=1.0, tau=1.0):
    """
    Absolute residual of the deformation third cusp equation at a real state.

    If C is the cleared deformation edge critical equation, the third cusp
    equation is C_z * P_y - C_y * P_z = 0, i.e. dC/dzeta = 0 along
    P(zeta,y)=0 after clearing the implicit derivative denominator.
    """

    P, Pz, Py, Pzz, Pzy, Pyy = _eval_P_all_partials(complex(zeta), complex(y),
                                                    coeffs)
    zeta = float(numpy.real(zeta))
    y = float(numpy.real(y))
    c0 = float(c0)
    tau = float(tau)
    Pz = float(numpy.real(Pz))
    Py = float(numpy.real(Py))
    Pzz = float(numpy.real(Pzz))
    Pzy = float(numpy.real(Pzy))
    Pyy = float(numpy.real(Pyy))

    A = c0 * y * zeta + (c0 - 1.0)
    Az = c0 * y
    Ay = c0 * zeta
    Bz = (c0 - 1.0) * Pzy + c0 * (2.0 * zeta * Pz + zeta * zeta * Pzz)
    By = (c0 - 1.0) * Pyy + c0 * (zeta * zeta) * Pzy
    Cz = tau * (2.0 * A * Az * Py + A * A * Pzy) + \
        (tau - 1.0) * Bz
    Cy = tau * (2.0 * A * Ay * Py + A * A * Pyy) + \
        (tau - 1.0) * By
    F3 = Cz * Py - Cy * Pz
    return abs(F3)


# ========================
# evolve edges from states
# ========================

def evolve_edges_from_states(t_grid, coeffs, zeta0, y0, c0=1.0, max_iter=50,
                             tol=1e-12, return_preimage=False, log=False):
    """
    Continue already-initialized edge states over an arbitrary monotone t-grid.
    This is the same continuation logic as ``evolve_edges`` but starts from
    provided preimage states instead of inferring t=0 branch points.
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 1:
        raise ValueError('t_grid must be non-empty.')
    if t_grid.size > 1:
        d = numpy.diff(t_grid)
        if not (numpy.all(d >= 0.0) or numpy.all(d <= 0.0)):
            raise ValueError('t_grid must be monotone.')

    zeta = numpy.asarray(zeta0, dtype=numpy.complex128).ravel().copy()
    y = numpy.asarray(y0, dtype=numpy.complex128).ravel().copy()
    if zeta.size != y.size:
        raise ValueError('zeta0 and y0 must have same size.')

    m = zeta.size
    complex_edges = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
    ok = numpy.zeros((t_grid.size, m), dtype=bool)

    if return_preimage:
        zeta_hist = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
        y_hist = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
    else:
        zeta_hist = None
        y_hist = None

    tau0 = float(numpy.exp(float(t_grid[0])))
    c0 = float(c0)
    complex_edges[0, :] = numpy.asarray([
        _deform_pushforward(tau0, zeta[j], y[j], c0)
        for j in range(m)], dtype=numpy.complex128)
    ok[0, :] = numpy.isfinite(zeta.real) & numpy.isfinite(zeta.imag) & \
        numpy.isfinite(y.real) & numpy.isfinite(y.imag) & \
        (numpy.abs(y) > 0.0)
    if return_preimage:
        zeta_hist[0, :] = zeta
        y_hist[0, :] = y

    for it in range(1, t_grid.size):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])
        dt = t1 - t0
        n_sub = max(1, int(numpy.ceil(abs(dt) / 0.1)))

        z_cur = zeta.copy()
        y_cur = y.copy()
        ok_cur = ok[it - 1, :].copy()

        for ks in range(1, n_sub + 1):
            ts0 = t0 + dt * ((ks - 1) / float(n_sub))
            ts1 = t0 + dt * (ks / float(n_sub))
            for j in range(m):
                if not ok_cur[j]:
                    continue
                z_new, y_new, okj = _advance_edge_step(
                    ts0, ts1, z_cur[j], y_cur[j], coeffs, c0,
                    max_iter=max_iter, tol=tol, log=log)
                ok_cur[j] = bool(okj)
                z_cur[j] = z_new
                y_cur[j] = y_new
                if not (numpy.isfinite(z_new.real) and
                        numpy.isfinite(y_new.real) and
                        numpy.isfinite(z_new.imag) and
                        numpy.isfinite(y_new.imag)):
                    ok_cur[j] = False

        zeta[:] = z_cur
        y[:] = y_cur
        ok[it, :] = ok_cur
        tau = float(numpy.exp(float(t1)))
        # c = tau - 1.0
        complex_edges[it, :] = numpy.asarray([
            _deform_pushforward(tau, zeta[j], y[j], c0)
            for j in range(m)], dtype=numpy.complex128)
        if return_preimage:
            zeta_hist[it, :] = zeta
            y_hist[it, :] = y

    if return_preimage:
        return complex_edges, ok, zeta_hist, y_hist
    return complex_edges, ok


# ==================
# scan edges at time
# ==================

def scan_edges_at_time(t, coeffs, x_range, n_scan=1024, stieltjes=None,
                       c0=1.0, delta=1e-5, max_iter=50, tol=1e-12,
                       log=False, dedup_x_tol=1e-6):
    """
    Find all edge points at a fixed time by scanning x and solving the fixed-
    time edge equations. This uses the governing equations directly, not any
    density threshold crossing.
    """

    x_min, x_max = float(x_range[0]), float(x_range[1])
    if not numpy.isfinite(x_min) or not numpy.isfinite(x_max) or \
            (x_max <= x_min):
        raise ValueError('invalid x_range.')

    tau = float(numpy.exp(float(t)))
    # c = tau - 1.0
    xs = numpy.linspace(x_min, x_max, int(n_scan))

    cand = []
    for x in xs:
        roots = numpy.asarray(eval_roots(numpy.array([complex(x)]), coeffs)[0],
                              dtype=numpy.complex128).ravel()
        if roots.size == 0:
            continue
        for rr in roots:
            y0 = rr
            if stieltjes is not None:
                try:
                    target = complex(numpy.asarray(stieltjes(complex(x, delta))
                                                   ).reshape(-1)[0])
                    # keep closest root to target as first guess if possible
                    # but still process all roots below since we want all edge
                    # branches.
                    _ = target
                except Exception:
                    pass
            zeta0 = complex(x)
            zeta1, y1, ok1 = _edge_newton_step(t, zeta0, y0, coeffs, c0,
                                               max_iter=max_iter, tol=tol,
                                               log=log)
            if not ok1:
                continue
            if not (numpy.isfinite(zeta1.real) and
                    numpy.isfinite(zeta1.imag) and
                    numpy.isfinite(y1.real) and
                    numpy.isfinite(y1.imag)):
                continue
            if abs(y1) == 0.0:
                continue
            x1 = float(numpy.real(_deform_pushforward(tau, zeta1, y1, c0)))
            if not numpy.isfinite(x1):
                continue
            # equation residual as sanity check
            F1, F2, *_ = _residual_jacobian_edge_tau(
                tau, zeta1, y1, coeffs, c0)
            res = max(abs(F1), abs(F2))
            if not numpy.isfinite(res) or res > max(1e-8, 100.0 * tol):
                continue
            cand.append((x1, complex(zeta1), complex(y1), float(res)))

    if not cand:
        return numpy.empty((0,), dtype=float), \
                numpy.empty((0,), dtype=numpy.complex128), \
                numpy.empty((0,), dtype=numpy.complex128)

    cand.sort(key=lambda item: item[0])
    x_keep = []
    z_keep = []
    y_keep = []
    for x1, z1, y1, res in cand:
        if (not x_keep) or (abs(x1 - x_keep[-1]) > dedup_x_tol):
            x_keep.append(x1)
            z_keep.append(z1)
            y_keep.append(y1)
        else:
            # keep smaller residual representative
            prev_idx = len(x_keep) - 1
            prev_res = max(abs(_residual_jacobian_edge_tau(
                tau, z_keep[prev_idx], y_keep[prev_idx], coeffs, c0)[0]),
                           abs(_residual_jacobian_edge_tau(
                               tau, z_keep[prev_idx], y_keep[prev_idx],
                               coeffs, c0)[1]))
            if res < prev_res:
                x_keep[-1] = x1
                z_keep[-1] = z1
                y_keep[-1] = y1

    return (numpy.asarray(x_keep, dtype=float),
            numpy.asarray(z_keep, dtype=numpy.complex128),
            numpy.asarray(y_keep, dtype=numpy.complex128))
