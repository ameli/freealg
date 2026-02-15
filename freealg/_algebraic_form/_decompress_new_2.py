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
from ._continuation_algebraic import powers
from ._poly_util import eval_P_partials

__all__ = ['decompress_newton']


# ==========
# fd solve w
# ==========

def fd_solve_w(z, t, coeffs, w_init, max_iter=50, tol=1e-12,
               armijo=1e-4, min_lam=1e-6, w_min=1e-14):
    """
    Damped Newton solve for w from F_t(z,w)=P(z+alpha/w, tau*w)=0.

    Convention: m(z)= \int rho(x)/(x-z) dx, so for z in C^+ we want Im(w)>0.
    """
    z = complex(z)
    w = complex(w_init)

    tau = float(numpy.exp(t))
    alpha = 1.0 - 1.0 / tau

    want_pos_imag = (z.imag > 0.0)

    # quick validity check on init
    if (not numpy.isfinite(w.real)) or (not numpy.isfinite(w.imag)):
        return w, False
    if abs(w) < w_min:
        return w, False
    if want_pos_imag and (w.imag <= 0.0):
        # nudge into upper half-plane (do NOT flip sign; just perturb)
        w = complex(w.real, max(1e-15, abs(w.imag)))

    for _ in range(max_iter):

        if (not numpy.isfinite(w.real)) or (not numpy.isfinite(w.imag)):
            return w, False
        if abs(w) < w_min:
            return w, False
        if want_pos_imag and (w.imag <= 0.0):
            return w, False

        zeta = z + alpha / w
        y = tau * w

        F, Pz, Py = eval_P_partials(zeta, y, coeffs)
        F = complex(F)
        Pz = complex(Pz)
        Py = complex(Py)

        F_abs = abs(F)
        if F_abs <= tol:
            return w, True

        dF = (-alpha / (w * w)) * Pz + tau * Py
        dF = complex(dF)
        if dF == 0.0 or (not numpy.isfinite(dF.real)) or (not numpy.isfinite(dF.imag)):
            return w, False

        step = -F / dF

        # backtracking on |F| decrease
        lam = 1.0
        ok = False
        while lam >= min_lam:
            w_new = w + lam * step

            if (not numpy.isfinite(w_new.real)) or (not numpy.isfinite(w_new.imag)):
                lam *= 0.5
                continue
            if abs(w_new) < w_min:
                lam *= 0.5
                continue
            if want_pos_imag and (w_new.imag <= 0.0):
                lam *= 0.5
                continue

            F_new = eval_P_partials(z + alpha / w_new, tau * w_new, coeffs)[0]
            F_new = complex(F_new)

            # Armijo-like sufficient decrease on residual norm
            if abs(F_new) <= (1.0 - armijo * lam) * F_abs:
                w = w_new
                ok = True
                break

            lam *= 0.5

        if not ok:
            return w, False

    # if max_iter hit, accept only if residual is reasonably small
    F_end = eval_P_partials(z + alpha / w, tau * w, coeffs)[0]
    F_end = complex(F_end)
    return w, (abs(F_end) <= 10.0 * tol)



# ===============
# fd candidates w
# ===============

def fd_candidates_w(z, t, coeffs, w_min=1e-14):
    """
    Return candidate roots w solving P(z + alpha/w, tau*w)=0 with Im(w)>0 (if Im(z)>0).
    """
    z = complex(z)
    tau = float(numpy.exp(t))
    alpha = 1.0 - 1.0 / tau
    want_pos_imag = (z.imag > 0.0)

    a = numpy.asarray(coeffs, dtype=numpy.complex128)
    deg_z = a.shape[0] - 1
    deg_m = a.shape[1] - 1

    beta = tau - 1.0  # since alpha/w = (tau-1)/(tau*w) = beta / y with y=tau*w

    poly_y = numpy.zeros(deg_z + deg_m + 1, dtype=numpy.complex128)

    from math import comb
    for i in range(deg_z + 1):
        for j in range(deg_m + 1):
            aij = a[i, j]
            if aij == 0:
                continue
            for k in range(i + 1):
                p = deg_z + j - k
                poly_y[p] += aij * comb(i, k) * (z ** (i - k)) * (beta ** k)

    coeffs = poly_y[::-1]
    nz_lead = numpy.flatnonzero(numpy.abs(coeffs) > 0)
    if nz_lead.size == 0:
        return []

    coeffs = coeffs[nz_lead[0]:]
    roots_y = numpy.roots(coeffs)

    cands = []
    for y in roots_y:
        if not numpy.isfinite(y.real) or not numpy.isfinite(y.imag):
            continue
        w = y / tau
        if abs(w) < w_min:
            continue
        if want_pos_imag and (w.imag <= 0.0):
            continue
        # residual filter (optional but helps)
        cands.append(complex(w))

    return cands


# ======================
# eval rwo by z homotopy
# ======================

def eval_row_by_z_homotopy(
    t,
    z_targets,
    w_seed_targets,
    R,
    coeffs,
    w_anchor,
    *,
    steps=80,
    max_iter=50,
    tol=1e-12,
    armijo=1e-4,
    min_lam=1e-6,
    w_min=1e-14):
    """
    Evaluate w(t,z) on z_targets in C^+ by z-homotopy from z0=iR,
    but anchored at the TRUE w(t,z0)=w_anchor (computed separately).

    Path is 2-segment:
        z0=iR  ->  x+iR  ->  x+i*eta
    """

    z_targets = numpy.asarray(z_targets, dtype=numpy.complex128)
    w_seed_targets = numpy.asarray(w_seed_targets, dtype=numpy.complex128)

    steps = int(steps)
    if steps < 2:
        steps = 2

    z0 = 1j * float(R)
    eta_floor = float(abs(z_targets[0].imag))
    if eta_floor <= 0.0:
        eta_floor = 1e-6

    w_out = numpy.empty(z_targets.size, dtype=numpy.complex128)
    ok_out = numpy.zeros(z_targets.size, dtype=bool)

    def _pick(cands, z, w_ref):
        # Filter Herglotz for convention: Im(w)>0 on C^+
        cpos = [u for u in cands if u.imag > 0.0]
        if cpos:
            cands = cpos

        # Continuity + asymptotic-at-infinity preference (mass=1): w*z ~ -1
        # This is CRITICAL to avoid choosing the wrong Herglotz-looking sheet.
        best = None
        best_cost = None
        for u in cands:
            cost = abs(u - w_ref) + 1.0 * abs(u * z + 1.0)
            if (best_cost is None) or (cost < best_cost):
                best = u
                best_cost = cost
        return best

    for k in range(z_targets.size):
        zT = z_targets[k]
        xT = float(zT.real)

        zA = complex(xT, float(R))      # horizontal leg endpoint
        zB = complex(xT, eta_floor)     # final point (vertical down)

        w = w_anchor
        ok = True

        # ---- segment 1: z0 -> zA (horizontal at imag=R) ----
        for j in range(1, steps + 1):
            s = j / float(steps)
            z = z0 + s * (zA - z0)

            w_new, ok_new = fd_solve_w(
                z, t, coeffs, w,
                max_iter=max_iter, tol=tol, armijo=armijo,
                min_lam=min_lam, w_min=w_min
            )

            if not ok_new:
                cands = fd_candidates_w(z, t, coeffs, w_min=w_min)
                if cands:
                    w_new = _pick(cands, z, w)
                    ok_new = (w_new is not None)

            if not ok_new:
                ok = False
                break
            w = w_new

        # ---- segment 2: zA -> zB (vertical down at fixed real=xT) ----
        if ok:
            for j in range(1, steps + 1):
                s = j / float(steps)
                z = zA + s * (zB - zA)

                w_new, ok_new = fd_solve_w(
                    z, t, coeffs, w,
                    max_iter=max_iter, tol=tol, armijo=armijo,
                    min_lam=min_lam, w_min=w_min
                )

                if not ok_new:
                    cands = fd_candidates_w(z, t, coeffs, w_min=w_min)
                    if cands:
                        w_new = _pick(cands, z, w)
                        ok_new = (w_new is not None)

                if not ok_new:
                    ok = False
                    break
                w = w_new

        w_out[k] = w

        if not ok:
            # fallback at zT: prefer continuity to the provided per-z time seed
            cands = fd_candidates_w(zT, t, coeffs, w_min=w_min)
            if cands:
                w_out[k] = _pick(cands, zT, w_seed_targets[k])
                ok_out[k] = (w_out[k] is not None)
                if not ok_out[k]:
                    w_out[k] = w_seed_targets[k]
            else:
                w_out[k] = w_seed_targets[k]
                ok_out[k] = False
        else:
            ok_out[k] = True

    return w_out, ok_out


# =================
# decompress newton
# =================

def decompress_newton(
    z_list,
    t_grid,
    coeffs,
    w0_list=None,
    *,
    R=400.0,
    z_hom_steps=160,
    eta_track=1e-3,       # IMPORTANT: track branches at this safe height
    eta_steps=40,         # vertical homotopy steps down to target imag
    max_iter=50,
    tol=1e-12,
    armijo=1e-4,
    min_lam=1e-6,
    w_min=1e-14,
    **_unused_kwargs):
    """
    Robust FD solver:
      (A) For each t, compute w(x+i*eta_track) by z-homotopy from z0=iR.
      (B) For each x, descend vertically from eta_track to target imag (typically 1e-5)
          using continuation in eta (vertical homotopy).
    This prevents multi-bulk cutoffs caused by trying to track directly at tiny eta.
    """

    z_list = numpy.asarray(z_list, dtype=numpy.complex128)
    t_grid = numpy.asarray(t_grid, dtype=float)
    nz = z_list.size
    nt = t_grid.size

    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")

    x_list = z_list.real
    eta_target = float(z_list.imag.max())  # z_query uses constant imag
    if eta_target <= 0.0:
        raise ValueError("This solver assumes z_list is in C^+ (imag>0).")

    eta_track = float(max(eta_track, 10.0 * eta_target))  # ensure track height is above target

    # -----------------
    # damped Newton solve
    # -----------------
    def solve_w_newton(z, t, w_init):
        z = complex(z)
        w = complex(w_init)

        tau = float(numpy.exp(t))
        alpha = 1.0 - 1.0 / tau

        # Herglotz for convention: Im(w)>0 for z in C^+
        if w.imag <= 0.0:
            w = complex(w.real, max(1e-15, abs(w.imag)))

        for _ in range(max_iter):
            if (not numpy.isfinite(w.real)) or (not numpy.isfinite(w.imag)):
                return w, False
            if abs(w) < w_min:
                return w, False
            if w.imag <= 0.0:
                return w, False

            zeta = z + alpha / w
            y = tau * w
            F, Pz, Py = eval_P_partials(zeta, y, coeffs)
            F = complex(F)
            if abs(F) <= tol:
                return w, True

            dF = (-alpha / (w * w)) * complex(Pz) + tau * complex(Py)
            if (dF == 0.0) or (not numpy.isfinite(dF.real)) or (not numpy.isfinite(dF.imag)):
                return w, False

            step = -F / dF
            F_abs = abs(F)

            lam = 1.0
            ok = False
            while lam >= min_lam:
                w_new = w + lam * step
                if (not numpy.isfinite(w_new.real)) or (not numpy.isfinite(w_new.imag)):
                    lam *= 0.5
                    continue
                if abs(w_new) < w_min:
                    lam *= 0.5
                    continue
                if w_new.imag <= 0.0:
                    lam *= 0.5
                    continue

                zeta_new = z + alpha / w_new
                y_new = tau * w_new
                F_new = complex(eval_P_partials(zeta_new, y_new, coeffs)[0])

                if abs(F_new) <= (1.0 - armijo * lam) * F_abs:
                    w = w_new
                    ok = True
                    break
                lam *= 0.5

            if not ok:
                return w, False

        # accept if residual not crazy
        zeta = z + alpha / w
        y = tau * w
        F_end = complex(eval_P_partials(zeta, y, coeffs)[0])
        return w, (abs(F_end) <= 1e3 * tol)

    # -----------------------
    # (A) z-homotopy at safe eta
    # -----------------------
    def row_by_z_homotopy_at_eta(t, w_anchor_prev):
        z0 = 1j * float(R)

        # anchor solve at far point (use previous anchor in time)
        if w_anchor_prev is None:
            w0_seed = -1.0 / z0
        else:
            w0_seed = w_anchor_prev

        w_anchor, ok_anchor = solve_w_newton(z0, t, w0_seed)
        if not ok_anchor:
            return None, None, False

        w_row = numpy.empty(nz, dtype=numpy.complex128)
        ok_row = numpy.ones(nz, dtype=bool)

        for iz in range(nz):
            zt = complex(x_list[iz], eta_track)
            dz = zt - z0
            w = w_anchor

            for k in range(1, int(z_hom_steps) + 1):
                s = k / float(z_hom_steps)
                z = z0 + s * dz

                # enforce imag floor at eta_track (never go near the real axis here)
                if z.imag < eta_track:
                    z = complex(z.real, eta_track)

                w, ok = solve_w_newton(z, t, w)
                if not ok:
                    ok_row[iz] = False
                    break

            w_row[iz] = w if ok_row[iz] else (numpy.nan + 1j * numpy.nan)

        return w_anchor, w_row, bool(ok_row.all())

    # -----------------------
    # (B) vertical homotopy: eta_track -> eta_target
    # -----------------------
    def descend_in_eta(t, w_track_row):
        w_out = numpy.empty(nz, dtype=numpy.complex128)
        ok_out = numpy.ones(nz, dtype=bool)

        if eta_steps <= 0 or eta_track <= eta_target:
            # no descent requested
            for iz in range(nz):
                # one final polish at target imag
                zt = complex(x_list[iz], eta_target)
                w, ok = solve_w_newton(zt, t, w_track_row[iz])
                w_out[iz] = w
                ok_out[iz] = ok
            return w_out, bool(ok_out.all())

        for iz in range(nz):
            w = w_track_row[iz]
            ok = True

            # linear schedule in imag
            for k in range(1, int(eta_steps) + 1):
                eta = eta_track + (eta_target - eta_track) * (k / float(eta_steps))
                z = complex(x_list[iz], eta)
                w, ok = solve_w_newton(z, t, w)
                if not ok:
                    break

            w_out[iz] = w
            ok_out[iz] = ok

        return w_out, bool(ok_out.all())

    # -----------------------
    # main time loop
    # -----------------------
    if w0_list is None:
        w_prev = -1.0 / z_list
    else:
        w_prev = numpy.asarray(w0_list, dtype=numpy.complex128).copy()

    W = numpy.empty((nt, nz), dtype=numpy.complex128)
    OK = numpy.zeros((nt, nz), dtype=bool)

    W[0, :] = w_prev
    OK[0, :] = True

    w_anchor_prev = None

    for it in range(1, nt):
        t = float(t_grid[it])

        w_anchor_prev, w_track_row, ok_track = row_by_z_homotopy_at_eta(t, w_anchor_prev)
        if (not ok_track) or (w_track_row is None):
            # fallback: keep previous time
            W[it, :] = w_prev
            OK[it, :] = False
            continue

        w_row, ok_row = descend_in_eta(t, w_track_row)

        W[it, :] = w_row
        OK[it, :] = ok_row
        w_prev = w_row

    return W, OK


