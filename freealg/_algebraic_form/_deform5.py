# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE

"""
Predictor-corrector deformation solver on the algebraic curve P(zeta,y)=0.

This solver tracks (zeta, y) on the fitted t=0 spectral curve, where y = m0(zeta),
and enforces the deformation constraint via a 2x2 complex system. It avoids
solving directly in the companion variable wbar, which can suffer branch slips.

API matches deform_newton used by AlgebraicForm.deform.
"""

import numpy

__all__ = ['deform_newton']


# =========================
# polynomial helpers: P, dP
# =========================

def _poly_powers(z, deg):
    z = numpy.asarray(z, dtype=complex).ravel()
    n = z.size
    zp = numpy.ones((n, deg + 1), dtype=complex)
    for k in range(1, deg + 1):
        zp[:, k] = zp[:, k - 1] * z
    return zp


def _eval_P_dP(z, m, coeffs):
    """
    Evaluate P(z,m), dP/dz, dP/dm for polynomial coeffs.
    coeffs shape: (deg_z+1, s+1)
    """
    z = numpy.asarray(z, dtype=complex).ravel()
    m = numpy.asarray(m, dtype=complex).ravel()

    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)

    zp = _poly_powers(z, deg_z)
    a = zp @ coeffs

    if deg_z >= 1:
        idx = numpy.arange(deg_z + 1, dtype=float)
        coeffs_dz = coeffs * idx[:, None]
        zp_m1 = numpy.zeros_like(zp)
        zp_m1[:, 1:] = zp[:, :-1]
        a_dz = zp_m1 @ coeffs_dz
    else:
        a_dz = numpy.zeros_like(a)

    mp = numpy.ones((m.size, s + 1), dtype=complex)
    for j in range(1, s + 1):
        mp[:, j] = mp[:, j - 1] * m

    P = numpy.sum(a * mp, axis=1)
    Pz = numpy.sum(a_dz * mp, axis=1)

    if s >= 1:
        jm = numpy.arange(s + 1, dtype=float)
        mp_m1 = numpy.zeros_like(mp)
        mp_m1[:, 1:] = mp[:, :-1]
        Pm = numpy.sum((a * jm[None, :]) * mp_m1, axis=1)
    else:
        Pm = numpy.zeros_like(P)

    return P, Pz, Pm


# =====================================
# deformation system F1=0, F2=0 in (zeta,y)
# =====================================

def _F_system(z_fixed, tau, coeffs, c0, zeta, y):
    """
    F1 = P(zeta, y)
    F2 = (tau*zeta - z) * (c0*y*zeta + c0 - 1) + (tau - 1) * zeta

    This is a desingularized form of the deformation relation with
        wbar = c0*y + (c0-1)/zeta.
    """
    P, Pz, Py = _eval_P_dP(numpy.array([zeta]), numpy.array([y]), coeffs)
    P = P[0]
    Pz = Pz[0]
    Py = Py[0]

    A = c0 * y * zeta + (c0 - 1.0)
    B = tau * zeta - z_fixed

    F1 = P
    F2 = B * A + (tau - 1.0) * zeta

    # Jacobian wrt (zeta, y)
    # dA/dzeta = c0*y, dA/dy = c0*zeta, dB/dzeta = tau
    J11 = Pz
    J12 = Py
    J21 = tau * A + B * (c0 * y) + (tau - 1.0)
    J22 = B * (c0 * zeta)

    # partial wrt tau for predictor (holding zeta,y fixed)
    dF_dtau = zeta * A + zeta

    return F1, F2, J11, J12, J21, J22, dF_dtau


def _recover_target_m(z_fixed, tau, c0, zeta, y, w_min=1e-14):
    """Map (zeta,y) on initial curve to target m at ratio tau*c0."""
    zeta_safe = zeta
    if abs(zeta_safe) < w_min:
        zeta_safe = complex((w_min if zeta.real >= 0 else -w_min), zeta.imag)
        if zeta_safe == 0:
            zeta_safe = w_min + 0j

    wbar = c0 * y + (c0 - 1.0) / zeta_safe
    c = tau * c0
    z_safe = z_fixed
    if abs(z_safe) < w_min:
        z_safe = complex((w_min if z_fixed.real >= 0 else -w_min), z_fixed.imag)
        if z_safe == 0:
            z_safe = w_min + 0j
    m = (wbar - (c - 1.0) / z_safe) / c
    return m, wbar


# ===========================
# corrector: damped 2x2 Newton
# ===========================

def _newton_corrector(z_fixed, tau, coeffs, c0, zeta0, y0,
                      max_iter=50, tol=1e-12,
                      armijo=1e-4, min_lam=1e-6,
                      w_min=1e-14, enforce_imag=True):
    zeta = complex(zeta0)
    y = complex(y0)

    for it in range(int(max_iter)):
        F1, F2, J11, J12, J21, J22, _ = _F_system(
            z_fixed, tau, coeffs, c0, zeta, y)

        r = max(abs(F1), abs(F2))
        if r <= tol:
            m, _ = _recover_target_m(z_fixed, tau, c0, zeta, y, w_min=w_min)
            if (not numpy.isfinite(m.real)) or (not numpy.isfinite(m.imag)):
                return zeta, y, False, it
            if enforce_imag and (z_fixed.imag > 0.0) and (m.imag <= 0.0):
                return zeta, y, False, it
            return zeta, y, True, it

        try:
            delta = numpy.linalg.solve(
                numpy.array([[J11, J12], [J21, J22]], dtype=complex),
                numpy.array([-F1, -F2], dtype=complex)
            )
        except numpy.linalg.LinAlgError:
            return zeta, y, False, it

        dzeta = delta[0]
        dy = delta[1]

        lam = 1.0
        accepted = False
        while lam >= float(min_lam):
            zeta_try = zeta + lam * dzeta
            y_try = y + lam * dy
            F1t, F2t, _, _, _, _, _ = _F_system(z_fixed, tau, coeffs, c0,
                                                zeta_try, y_try)
            rt = max(abs(F1t), abs(F2t))
            if (numpy.isfinite(rt) and
                    (rt <= (1.0 - float(armijo) * lam) * r or rt < r)):
                zeta = zeta_try
                y = y_try
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            return zeta, y, False, it

    return zeta, y, False, int(max_iter)


def _corrector_multistart(z_fixed, tau, coeffs, c0, zeta_pred, y_pred, zeta_prev, y_prev,
                        max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6,
                        w_min=1e-14):
    """Try several nearby seeds and choose branch-consistent corrector solution."""
    seeds = []
    seeds.append((zeta_pred, y_pred))
    # previous state seed (strong continuation prior)
    seeds.append((zeta_prev, y_prev))
    # damped predictor seeds
    seeds.append((0.5 * (zeta_pred + zeta_prev), 0.5 * (y_pred + y_prev)))
    seeds.append((zeta_prev + 0.5 * (zeta_pred - zeta_prev), y_prev + 0.5 * (y_pred - y_prev)))

    best = None
    best_score = None
    im_prev = float(numpy.imag(y_prev)) if numpy.isfinite(y_prev) else numpy.nan

    for zs, ys in seeds:
        zc, yc, ok, nit = _newton_corrector(
            z_fixed, tau, coeffs, c0, zs, ys,
            max_iter=max_iter, tol=tol, armijo=armijo, min_lam=min_lam,
            w_min=w_min, enforce_imag=False)
        if not ok:
            continue
        m, _ = _recover_target_m(z_fixed, tau, c0, zc, yc, w_min=w_min)
        if (not numpy.isfinite(m.real)) or (not numpy.isfinite(m.imag)):
            continue
        if (z_fixed.imag > 0.0) and (m.imag <= 0.0):
            continue

        # continuity score in (zeta,y) + anti-collapse rule in y/m imag
        dz = abs(zc - zeta_prev)
        dy = abs(yc - y_prev)
        collapse_pen = 0.0
        if numpy.isfinite(im_prev) and (im_prev > 0.0):
            # compare on m-imag (physical quantity)
            im_m = float(numpy.imag(m))
            if (im_prev > 50.0 * max(z_fixed.imag, 1e-12)) and (im_m < 5.0 * max(z_fixed.imag, 1e-12)):
                collapse_pen = 1e6
        score = (collapse_pen, dz + dy, nit)
        if (best_score is None) or (score < best_score):
            best_score = score
            best = (zc, yc, m)

    if best is None:
        return zeta_pred, y_pred, False
    zc, yc, m = best
    return zc, yc, m


# ===========================
# predictor step (generic tangent)
# ===========================

def _predictor_step(z_fixed, tau0, tau1, coeffs, c0, zeta0, y0):
    dtau = float(tau1 - tau0)
    if dtau == 0.0:
        return zeta0, y0

    F1, F2, J11, J12, J21, J22, dF2_dtau = _F_system(
        z_fixed, tau0, coeffs, c0, zeta0, y0)

    # J * [zeta', y'] = - [0, dF2/dtau]
    try:
        deriv = numpy.linalg.solve(
            numpy.array([[J11, J12], [J21, J22]], dtype=complex),
            numpy.array([0.0 + 0.0j, -dF2_dtau], dtype=complex)
        )
        zeta_p = deriv[0]
        y_p = deriv[1]
    except numpy.linalg.LinAlgError:
        # fallback: no predictor movement
        zeta_p = 0.0 + 0.0j
        y_p = 0.0 + 0.0j

    return zeta0 + dtau * zeta_p, y0 + dtau * y_p


# ===========================================
# one point continuation from tau_prev to tau_next
# ===========================================

def _continue_one_point(z_fixed, tau_prev, tau_next, coeffs, c0,
                        zeta0, y0,
                        max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6,
                        w_min=1e-14, max_substeps=8):
    """
    Continue one point in tau with predictor-corrector and recursive step halving.
    """
    def _rec(a_tau, b_tau, a_zeta, a_y, depth):
        zeta_pred, y_pred = _predictor_step(z_fixed, a_tau, b_tau, coeffs, c0,
                                            a_zeta, a_y)
        ms = _corrector_multistart(
            z_fixed, b_tau, coeffs, c0, zeta_pred, y_pred, a_zeta, a_y,
            max_iter=max_iter, tol=tol, armijo=armijo, min_lam=min_lam,
            w_min=w_min)
        if ms[2] is not False:
            zeta_corr, y_corr, _m_corr = ms
            return zeta_corr, y_corr, True
        if depth >= int(max_substeps):
            return a_zeta, a_y, False
        mid = 0.5 * (a_tau + b_tau)
        zeta_mid, y_mid, ok1 = _rec(a_tau, mid, a_zeta, a_y, depth + 1)
        if not ok1:
            return a_zeta, a_y, False
        zeta_end, y_end, ok2 = _rec(mid, b_tau, zeta_mid, y_mid, depth + 1)
        if not ok2:
            return zeta_mid, y_mid, False
        return zeta_end, y_end, True

    return _rec(float(tau_prev), float(tau_next), complex(zeta0), complex(y0), 0)


# =============
# deform_newton
# =============

def deform_newton(z_list, t_grid, coeffs, c0, w0_list, dt_max=0.1, sweep=True,
                  time_rel_tol=5.0, active_imag_eps=None, sweep_pad=20,
                  max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6,
                  w_min=1e-14):
    """
    Deformation solver using predictor-corrector on the (zeta, y) spectral curve.

    Parameters match the older deform_newton API (some heuristic args are
    accepted for compatibility but are not central here).
    """
    z_list = numpy.asarray(z_list, dtype=complex).ravel()
    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    w0_list = numpy.asarray(w0_list, dtype=complex).ravel()

    nt = t_grid.size
    nz = z_list.size

    if nt == 0:
        raise ValueError("t_grid must be non-empty.")
    if abs(float(t_grid[0])) > 1e-15:
        raise ValueError("t_grid must start at 0 (tau=1).")
    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")
    if w0_list.size != nz:
        raise ValueError("w0_list must have the same length as z_list.")

    # Output
    W = numpy.full((nt, nz), numpy.nan + 1j * numpy.nan, dtype=complex)
    ok = numpy.zeros((nt, nz), dtype=bool)

    # Initial state on the fitted curve: zeta=z, y=m0(z)
    zeta_state = z_list.astype(complex).copy()
    y_state = w0_list.astype(complex).copy()
    state_ok = numpy.isfinite(y_state.real) & numpy.isfinite(y_state.imag)

    W[0, :] = w0_list
    ok[0, :] = state_ok

    # March in time independently for each x (branch identity maintained by tau continuation)
    for it in range(1, nt):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])

        # internal t-substeps if requested
        if dt_max is None or dt_max <= 0:
            t_sub = numpy.array([t0, t1], dtype=float)
        else:
            nseg = max(1, int(numpy.ceil((t1 - t0) / float(dt_max))))
            t_sub = numpy.linspace(t0, t1, nseg + 1)

        tau_sub = numpy.exp(t_sub)
        tau1 = float(tau_sub[-1])

        w_row = numpy.full(nz, numpy.nan + 1j * numpy.nan, dtype=complex)
        ok_row = numpy.zeros(nz, dtype=bool)

        for j in range(nz):
            z = z_list[j]

            # seed from previous time state
            zeta = zeta_state[j]
            y = y_state[j]
            if (not state_ok[j]) or (not numpy.isfinite(zeta)) or (not numpy.isfinite(y)):
                zeta = z
                y = w0_list[j]

            point_ok = True
            tau_prev = float(tau_sub[0])
            for ksub in range(1, tau_sub.size):
                tau_next = float(tau_sub[ksub])
                zeta, y, step_ok = _continue_one_point(
                    z, tau_prev, tau_next, coeffs, float(c0), zeta, y,
                    max_iter=max_iter, tol=tol, armijo=armijo,
                    min_lam=min_lam, w_min=w_min, max_substeps=8)
                if not step_ok:
                    point_ok = False
                    break
                tau_prev = tau_next

            if point_ok:
                m, _ = _recover_target_m(z, tau1, float(c0), zeta, y, w_min=w_min)
                # fallback: enforce Herglotz sign for z in C+
                if (z.imag > 0.0) and (m.imag <= 0.0):
                    point_ok = False
                elif (not numpy.isfinite(m.real)) or (not numpy.isfinite(m.imag)):
                    point_ok = False

            if point_ok:
                zeta_state[j] = zeta
                y_state[j] = y
                state_ok[j] = True
                w_row[j] = m
                ok_row[j] = True
            else:
                # local fallback 1: try neighbor-seeded corrector directly at final tau
                tau_final = tau1
                repaired = False
                cand = []
                for jj in (j - 1, j + 1):
                    if 0 <= jj < nz and ok_row[jj] and numpy.isfinite(zeta_state[jj]) and numpy.isfinite(y_state[jj]):
                        cand.append((zeta_state[jj], y_state[jj]))
                if (j - 1) >= 0 and (j + 1) < nz:
                    if ok_row[j - 1] and ok_row[j + 1]:
                        cand.append((0.5 * (zeta_state[j - 1] + zeta_state[j + 1]),
                                     0.5 * (y_state[j - 1] + y_state[j + 1])))
                # previous state seed as last resort
                if numpy.isfinite(zeta_state[j]) and numpy.isfinite(y_state[j]):
                    cand.append((zeta_state[j], y_state[j]))

                for zc, yc in cand:
                    zc2, yc2, okc, _ = _newton_corrector(
                        z, tau_final, coeffs, float(c0), zc, yc,
                        max_iter=max_iter, tol=tol, armijo=armijo,
                        min_lam=min_lam, w_min=w_min, enforce_imag=True)
                    if not okc:
                        continue
                    mc, _ = _recover_target_m(z, tau_final, float(c0), zc2, yc2, w_min=w_min)
                    if (z.imag > 0.0) and (mc.imag <= 0.0):
                        continue
                    if (not numpy.isfinite(mc.real)) or (not numpy.isfinite(mc.imag)):
                        continue
                    zeta_state[j] = zc2
                    y_state[j] = yc2
                    state_ok[j] = True
                    w_row[j] = mc
                    ok_row[j] = True
                    repaired = True
                    break

                if not repaired:
                    # local fallback 2: keep previous-row value instead of zero/NaN to avoid holes
                    if ok[it - 1, j] and numpy.isfinite(W[it - 1, j]):
                        w_row[j] = W[it - 1, j]
                        ok_row[j] = True
                    else:
                        w_row[j] = numpy.nan + 1j * numpy.nan
                        ok_row[j] = False

        W[it, :] = w_row
        ok[it, :] = ok_row

    return W, ok
