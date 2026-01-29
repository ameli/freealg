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

__all__ = ['build_time_grid', 'decompress_newton']


# ===============
# build time grid
# ===============

def build_time_grid(sizes, n0, min_n_times=0):
    """
    Build a monotone time grid t for sizes/n0 = exp(t).

    Returns
    -------
    t_all : ndarray
        Sorted time grid (includes t=0 and all requested times).
    idx_req : ndarray
        Indices into t_all for the requested times (same order as sizes).
    """
    sizes = numpy.asarray(sizes, dtype=float)
    alpha = sizes / float(n0)
    t_req = numpy.log(alpha)

    T = float(numpy.max(t_req)) if t_req.size else 0.0
    base = numpy.unique(numpy.r_[0.0, t_req, T])
    t_all = numpy.sort(base)

    N = int(min_n_times) if min_n_times is not None else 0
    while t_all.size < N and t_all.size >= 2:
        gaps = numpy.diff(t_all)
        k = int(numpy.argmax(gaps))
        mid = 0.5 * (t_all[k] + t_all[k + 1])
        t_all = numpy.sort(numpy.unique(numpy.r_[t_all, mid]))

    idx_req = numpy.searchsorted(t_all, t_req)
    return t_all, idx_req


# ===============
# eval P partials
# ===============

def eval_P_partials(z, m, a_coeffs):
    """
    Evaluate P(z,m) and partials dP/dz, dP/dm.

    P is represented by a_coeffs in the monomial basis:
        P(z,m) = sum_{j=0..s} a_j(z) m^j
        a_j(z) = sum_{i=0..deg_z} a_coeffs[i,j] z^i
    """
    z = numpy.asarray(z, dtype=complex)
    m = numpy.asarray(m, dtype=complex)

    deg_z = int(a_coeffs.shape[0] - 1)
    s = int(a_coeffs.shape[1] - 1)

    # Scalar fast path
    if (z.ndim == 0) and (m.ndim == 0):
        zz = complex(z)
        mm = complex(m)

        a = numpy.empty(s + 1, dtype=complex)
        ap = numpy.empty(s + 1, dtype=complex)

        for j in range(s + 1):
            c = a_coeffs[:, j]

            val = 0.0 + 0.0j
            for i in range(deg_z, -1, -1):
                val = val * zz + c[i]
            a[j] = val

            dval = 0.0 + 0.0j
            for i in range(deg_z, 0, -1):
                dval = dval * zz + (i * c[i])
            ap[j] = dval

        # Horner in m
        p = a[s]
        pm = 0.0 + 0.0j
        for j in range(s - 1, -1, -1):
            pm = pm * mm + p
            p = p * mm + a[j]

        pz = ap[s]
        for j in range(s - 1, -1, -1):
            pz = pz * mm + ap[j]

        return p, pz, pm

    # Vector path
    shp = numpy.broadcast(z, m).shape
    zz = numpy.broadcast_to(z, shp).ravel()
    mm = numpy.broadcast_to(m, shp).ravel()

    zp = powers(zz, deg_z)
    mp = powers(mm, s)

    dzp = numpy.zeros_like(zp)
    for i in range(1, deg_z + 1):
        dzp[:, i] = i * zp[:, i - 1]

    P = numpy.zeros(zz.size, dtype=complex)
    Pz = numpy.zeros(zz.size, dtype=complex)
    Pm = numpy.zeros(zz.size, dtype=complex)

    for j in range(s + 1):
        aj = zp @ a_coeffs[:, j]
        P += aj * mp[:, j]

        ajp = dzp @ a_coeffs[:, j]
        Pz += ajp * mp[:, j]

        if j >= 1:
            Pm += (j * aj) * mp[:, j - 1]

    return P.reshape(shp), Pz.reshape(shp), Pm.reshape(shp)


# =========================
# Newton for one (z,t) pair
# =========================

def _newton_one(z, t, a_coeffs, w0, max_iter=50, tol=1e-12,
                armijo=1e-4, min_lam=1e-6, w_min=1e-14):
    """
    Solve F_t(z,w)=0 at a single (t,z) by damped Newton.

    F_t(z,w) := P(z + (1 - e^{-t})/w, e^{t} w) = 0.
    """
    z = complex(z)
    w = complex(w0)

    want_upper = (z.imag > 0.0)

    tau = numpy.exp(float(t))
    beta = 1.0 - 1.0 / tau  # = 1 - e^{-t}

    def eval_F_and_dF(w_val):
        if (not numpy.isfinite(w_val)) or (abs(w_val) <= w_min):
            return numpy.nan + 1j * numpy.nan, numpy.nan + 1j * numpy.nan
        zeta = z + beta / w_val
        y = tau * w_val
        P, Pz, Py = eval_P_partials(zeta, y, a_coeffs)
        dF = (-beta / (w_val * w_val)) * Pz + tau * Py
        return P, dF

    F, dF = eval_F_and_dF(w)
    if not numpy.isfinite(F):
        return w, False

    for _ in range(int(max_iter)):
        if abs(F) <= tol:
            ok = numpy.isfinite(w)
            if want_upper:
                ok = ok and (w.imag > 0.0)
            return w, bool(ok)

        if (not numpy.isfinite(dF)) or (abs(dF) == 0.0):
            break

        step = -F / dF
        if not numpy.isfinite(step):
            break

        lam = 1.0
        absF = abs(F)
        w_new = w
        F_new = F
        dF_new = dF

        while lam >= float(min_lam):
            cand = w + lam * step

            if (not numpy.isfinite(cand)) or (abs(cand) <= w_min):
                lam *= 0.5
                continue
            if want_upper and (cand.imag <= 0.0):
                lam *= 0.5
                continue

            Fcand, dFcand = eval_F_and_dF(cand)
            if not numpy.isfinite(Fcand):
                lam *= 0.5
                continue

            if armijo is None:
                ok_dec = (abs(Fcand) < absF)
            else:
                c = float(armijo)
                ok_dec = (abs(Fcand) <= (1.0 - c * lam) * absF)

            if ok_dec:
                w_new = cand
                F_new = Fcand
                dF_new = dFcand
                break

            lam *= 0.5

        if lam < float(min_lam):
            break

        if abs(w_new - w) <= tol * max(1.0, abs(w)):
            w = w_new
            F = F_new
            ok = numpy.isfinite(w)
            if want_upper:
                ok = ok and (w.imag > 0.0)
            return w, bool(ok)

        w = w_new
        F = F_new
        dF = dF_new

    ok = numpy.isfinite(w)
    if want_upper:
        ok = ok and (w.imag > 0.0)
    return w, False


# =====================
# Main Newton evolution
# =====================

def decompress_newton(
    z_query,
    t_all,
    a_coeffs,
    w0_list=None,
    max_iter=50,
    tol=1e-12,
    armijo=1e-4,
    min_lam=1e-6,
    w_min=1e-14,
    sweep=False,
    R=None,
    z_hom_steps=0,
    homotopy_tol=None,
    **_ignored,
):
    """
    Evolve the physical Stieltjes branch under free decompression.

    If R and z_hom_steps are provided (as in your notebook), the solver uses a
    straight-line z-homotopy from a far anchor point z_anchor = i R to each
    query point z, at every time slice. This is much more reliable than a sweep
    across the real grid for multi-cut densities.

    Returns
    -------
    W : ndarray, shape (len(t_all), len(z_query))
    ok : ndarray, same shape, boolean
    """
    zq = numpy.asarray(z_query, dtype=complex).ravel()
    t_all = numpy.asarray(t_all, dtype=float).ravel()
    if t_all.size == 0:
        raise ValueError('t_all must be non-empty.')

    # Sort t if needed
    if numpy.any(numpy.diff(t_all) < 0.0):
        order = numpy.argsort(t_all)
        t_all = t_all[order]
    else:
        order = None

    n_t = t_all.size
    n_z = zq.size
    W = numpy.empty((n_t, n_z), dtype=complex)
    ok = numpy.zeros((n_t, n_z), dtype=bool)

    if w0_list is None:
        w_init = -1.0 / zq
    else:
        w_init = numpy.asarray(w0_list, dtype=complex).ravel()
        if w_init.size != n_z:
            raise ValueError('w0_list must have same length as z_query.')

    W[0, :] = w_init
    ok[0, :] = numpy.isfinite(w_init)
    pos = (zq.imag > 0.0)
    ok[0, pos] &= (W[0, pos].imag > 0.0)

    use_homotopy = (R is not None) and (int(z_hom_steps) > 0)
    if homotopy_tol is None:
        homotopy_tol = tol

    if use_homotopy:
        Rf = float(R)
        if not numpy.isfinite(Rf) or (Rf <= 0.0):
            raise ValueError('R must be a positive finite number.')
        z_anchor = 1j * Rf
        max_dist = float(numpy.max(numpy.abs(zq - z_anchor)))
        if max_dist == 0.0:
            max_dist = 1.0

        # Initial anchor solve at first time
        w_anchor_prev, ok_anchor_prev = _newton_one(
            z_anchor, float(t_all[0]), a_coeffs, -1.0 / z_anchor,
            max_iter=max_iter, tol=homotopy_tol,
            armijo=armijo, min_lam=min_lam, w_min=w_min)
        if not ok_anchor_prev:
            w_anchor_prev = -1.0 / z_anchor

        def _track_from_anchor(z_target, t, w_seed):
            z_target = complex(z_target)
            if z_target == z_anchor:
                return complex(w_seed), True

            n_steps_max = int(z_hom_steps)
            n_steps = int(max(
                4,
                numpy.ceil(n_steps_max * abs(z_target - z_anchor) / max_dist)
            ))

            s = 0.0
            ds = 1.0 / float(n_steps)
            w_curr = complex(w_seed)
            refine = 0
            refine_max = 6

            while s < 1.0 - 1e-15:
                s_next = min(1.0, s + ds)
                z_next = z_anchor + (z_target - z_anchor) * s_next

                w_next, ok_next = _newton_one(
                    z_next, t, a_coeffs, w_curr,
                    max_iter=max_iter, tol=homotopy_tol,
                    armijo=armijo, min_lam=min_lam, w_min=w_min)

                if ok_next:
                    w_curr = w_next
                    s = s_next
                    ds = min(ds * 1.5, 1.0 - s)
                    refine = 0
                else:
                    ds *= 0.5
                    refine += 1
                    if refine > refine_max or ds < 1e-8:
                        return w_curr, False

            return w_curr, True

    start_k = 1
    if not numpy.isclose(t_all[0], 0.0):
        start_k = 0

    for k in range(start_k, n_t):
        t = float(t_all[k])

        w_prev_time = W[k - 1, :] if k > 0 else w_init
        Wk = numpy.empty(n_z, dtype=complex)
        okk = numpy.zeros(n_z, dtype=bool)

        if use_homotopy:
            # Continue anchor in time
            w_anchor, ok_anchor = _newton_one(
                z_anchor, t, a_coeffs, w_anchor_prev,
                max_iter=max_iter, tol=homotopy_tol,
                armijo=armijo, min_lam=min_lam, w_min=w_min)
            if ok_anchor:
                w_anchor_prev = w_anchor
            else:
                w_anchor = w_anchor_prev

            # Track each z from the anchor
            for j in range(n_z):
                w_h, ok_h = _track_from_anchor(zq[j], t, w_anchor)
                if ok_h:
                    Wk[j] = w_h
                    okk[j] = True
                    continue

                # Fallback: direct Newton from previous time
                w_d, ok_d = _newton_one(
                    zq[j], t, a_coeffs, w_prev_time[j],
                    max_iter=max_iter, tol=tol,
                    armijo=armijo, min_lam=min_lam, w_min=w_min)
                if ok_d:
                    Wk[j] = w_d
                    okk[j] = True
                else:
                    Wk[j] = w_h
                    okk[j] = False

        elif sweep:
            for j in range(n_z):
                if j == 0:
                    w0 = w_prev_time[j]
                else:
                    w0 = Wk[j - 1] if okk[j - 1] else w_prev_time[j]
                w_sol, ok_sol = _newton_one(
                    zq[j], t, a_coeffs, w0,
                    max_iter=max_iter, tol=tol,
                    armijo=armijo, min_lam=min_lam, w_min=w_min)
                Wk[j] = w_sol
                okk[j] = ok_sol

        else:
            for j in range(n_z):
                w_sol, ok_sol = _newton_one(
                    zq[j], t, a_coeffs, w_prev_time[j],
                    max_iter=max_iter, tol=tol,
                    armijo=armijo, min_lam=min_lam, w_min=w_min)
                Wk[j] = w_sol
                okk[j] = ok_sol

        W[k, :] = Wk
        ok[k, :] = okk

    if order is not None:
        inv = numpy.empty_like(order)
        inv[order] = numpy.arange(order.size)
        W = W[inv, :]
        ok = ok[inv, :]

    return W, ok
