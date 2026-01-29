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

__all__ = ['decompress_newton']



# ===============
# eval P partials
# ===============

def eval_P_partials(z, m, a_coeffs):
    """
    Evaluate P(z,m) and its partial derivatives dP/dz and dP/dm.

    This assumes P is represented by `a_coeffs` in the monomial basis

        P(z, m) = sum_{j=0..s} a_j(z) * m^j,
        a_j(z) = sum_{i=0..deg_z} a_coeffs[i, j] * z^i.

    The function returns P, dP/dz, dP/dm with broadcasting over z and m.

    Parameters
    ----------
    z : complex or array_like of complex
        First argument to P.
    m : complex or array_like of complex
        Second argument to P. Must be broadcast-compatible with `z`.
    a_coeffs : ndarray, shape (deg_z+1, s+1)
        Coefficient matrix for P in the monomial basis.

    Returns
    -------
    P : complex or ndarray of complex
        Value P(z,m).
    Pz : complex or ndarray of complex
        Partial derivative dP/dz evaluated at (z,m).
    Pm : complex or ndarray of complex
        Partial derivative dP/dm evaluated at (z,m).

    Notes
    -----
    For scalar (z,m), this uses Horner evaluation for a_j(z) and then Horner
    in m. For array inputs, it uses precomputed power tables via `_powers` for
    simplicity.

    Examples
    --------
    .. code-block:: python

        P, Pz, Pm = eval_P_partials(1.0 + 1j, 0.2 + 0.3j, a_coeffs)
    """

    z = numpy.asarray(z, dtype=complex)
    m = numpy.asarray(m, dtype=complex)

    deg_z = int(a_coeffs.shape[0] - 1)
    s = int(a_coeffs.shape[1] - 1)

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

        p = a[s]
        pm = 0.0 + 0.0j
        for j in range(s - 1, -1, -1):
            pm = pm * mm + p
            p = p * mm + a[j]

        pz = ap[s]
        for j in range(s - 1, -1, -1):
            pz = pz * mm + ap[j]

        return p, pz, pm

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


# ==========
# fd solve w
# ==========

# def fd_solve_w(z, t, a_coeffs, w_init, max_iter=50, tol=1e-12,
#                armijo=1e-4, min_lam=1e-6, w_min=1e-14):
#     """
#     Solve for w = m(t,z) from the implicit FD equation using damped Newton.
#
#     We solve in w the equation
#
#         F(w) = P(z + alpha/w, tau*w) = 0,
#
#     where tau = exp(t) and alpha = 1 - 1/tau.
#
#     A backtracking (Armijo) line search is used to stabilize Newton updates.
#     When Im(z) > 0, the iterate is constrained to remain in the upper
#     half-plane (Im(w) > 0), enforcing the Herglotz branch.
#
#     Parameters
#     ----------
#     z : complex
#         Query point in the complex plane.
#     t : float
#         Time parameter (tau = exp(t)).
#     a_coeffs : ndarray
#         Coefficients defining P(zeta,y) in the monomial basis.
#     w_init : complex
#         Initial guess for w.
#     max_iter : int, optional
#         Maximum number of Newton iterations.
#     tol : float, optional
#         Residual tolerance on |F(w)|.
#     armijo : float, optional
#         Armijo parameter for backtracking sufficient decrease.
#     min_lam : float, optional
#         Minimum damping factor allowed in backtracking.
#     w_min : float, optional
#         Minimum |w| allowed to avoid singularity in z + alpha/w.
#
#     Returns
#     -------
#     w : complex
#         The computed solution (last iterate if not successful).
#     success : bool
#         True if convergence criteria were met, False otherwise.
#
#     Notes
#     -----
#     This function does not choose the correct branch globally by itself; it
#     relies on a good initialization strategy (e.g. time continuation and/or
#     x-sweeps) to avoid converging to a different valid root of the implicit
#     equation.
#
#     Examples
#     --------
#     .. code-block:: python
#
#         w, ok = fd_solve_w(
#             z=0.5 + 1e-6j, t=2.0, a_coeffs=a_coeffs, w_init=m1_fn(0.5 + 1e-6j),
#             max_iter=50, tol=1e-12
#         )
#     """
#
#     z = complex(z)
#     w = complex(w_init)
#
#     tau = float(numpy.exp(t))
#     alpha = 1.0 - 1.0 / tau
#
#     want_pos_imag = (z.imag > 0.0)
#
#     for _ in range(max_iter):
#         if not numpy.isfinite(w.real) or not numpy.isfinite(w.imag):
#             return w, False
#         if abs(w) < w_min:
#             return w, False
#         if want_pos_imag and (w.imag <= 0.0):
#             return w, False
#
#         zeta = z + alpha / w
#         y = tau * w
#
#         F, Pz, Py = eval_P_partials(zeta, y, a_coeffs)
#         F = complex(F)
#         Pz = complex(Pz)
#         Py = complex(Py)
#
#         if abs(F) <= tol:
#             return w, True
#
#         dF = (-alpha / (w * w)) * Pz + tau * Py
#         if dF == 0.0:
#             return w, False
#
#         step = -F / dF
#
#         lam = 1.0
#         F_abs = abs(F)
#         ok = False
#
#         while lam >= min_lam:
#             w_new = w + lam * step
#             if abs(w_new) < w_min:
#                 lam *= 0.5
#                 continue
#             if want_pos_imag and (w_new.imag <= 0.0):
#                 lam *= 0.5
#                 continue
#
#             zeta_new = z + alpha / w_new
#             y_new = tau * w_new
#
#             F_new = eval_P_partials(zeta_new, y_new, a_coeffs)[0]
#             F_new = complex(F_new)
#
#             if abs(F_new) <= (1.0 - armijo * lam) * F_abs:
#                 w = w_new
#                 ok = True
#                 break
#
#             lam *= 0.5
#
#         if not ok:
#             return w, False
#
#     F_end = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
#     return w, (abs(F_end) <= 10.0 * tol)

def fd_solve_w(z, t, a_coeffs, w_init, max_iter=50, tol=1e-12,
               armijo=1e-4, min_lam=1e-6, w_min=1e-14):
    """
    Solve for w = m(t,z) from the implicit FD equation using damped Newton.

    We solve in w the equation

        F(w) = P(z + alpha/w, tau*w) = 0,

    where tau = exp(t) and alpha = 1 - 1/tau.

    A backtracking (Armijo) line search is used to stabilize Newton updates.
    When Im(z) > 0, the iterate is constrained to remain in the upper
    half-plane (Im(w) > 0), enforcing the Herglotz branch.

    Parameters
    ----------
    z : complex
        Query point in the complex plane.
    t : float
        Time parameter (tau = exp(t)).
    a_coeffs : ndarray
        Coefficients defining P(zeta,y) in the monomial basis.
    w_init : complex
        Initial guess for w.
    max_iter : int, optional
        Maximum number of Newton iterations.
    tol : float, optional
        Residual tolerance on |F(w)|.
    armijo : float, optional
        Armijo parameter for backtracking sufficient decrease.
    min_lam : float, optional
        Minimum damping factor allowed in backtracking.
    w_min : float, optional
        Minimum |w| allowed to avoid singularity in z + alpha/w.

    Returns
    -------
    w : complex
        The computed solution (last iterate if not successful).
    success : bool
        True if convergence criteria were met, False otherwise.

    Notes
    -----
    This function does not choose the correct branch globally by itself; it
    relies on a good initialization strategy (e.g. time continuation and/or
    x-sweeps) to avoid converging to a different valid root of the implicit
    equation.

    Examples
    --------
    .. code-block:: python

        w, ok = fd_solve_w(
            z=0.5 + 1e-6j, t=2.0, a_coeffs=a_coeffs, w_init=m1_fn(0.5 + 1e-6j),
            max_iter=50, tol=1e-12
        )
    """

    z = complex(z)
    w = complex(w_init)

    tau = float(numpy.exp(t))
    alpha = 1.0 - 1.0 / tau

    want_pos_imag = (z.imag > 0.0)

    for _ in range(max_iter):

        # ----------------

        # if not numpy.isfinite(w.real) or not numpy.isfinite(w.imag):
        #     return w, False
        # if abs(w) < w_min:
        #     return w, False
        # if want_pos_imag and (w.imag <= 0.0):
        #     return w, False
        #
        # zeta = z + alpha / w
        # y = tau * w
        #
        # F, Pz, Py = eval_P_partials(zeta, y, a_coeffs)
        # F = complex(F)
        # Pz = complex(Pz)
        # Py = complex(Py)
        #
        # if abs(F) <= tol:
        #     return w, True
        #
        # dF = (-alpha / (w * w)) * Pz + tau * Py
        # if dF == 0.0:
        #     return w, False
        #
        # step = -F / dF
        #
        # lam = 1.0
        # F_abs = abs(F)
        # ok = False
        #
        # while lam >= min_lam:
        #     w_new = w + lam * step
        #     if abs(w_new) < w_min:
        #         lam *= 0.5
        #         continue
        #     if want_pos_imag and (w_new.imag <= 0.0):
        #         lam *= 0.5
        #         continue
        #
        #     zeta_new = z + alpha / w_new
        #     y_new = tau * w_new
        #
        #     F_new = eval_P_partials(zeta_new, y_new, a_coeffs)[0]
        #     F_new = complex(F_new)
        #
        #     if abs(F_new) <= (1.0 - armijo * lam) * F_abs:
        #         w = w_new
        #         ok = True
        #         break
        #
        #     lam *= 0.5
        #
        # if not ok:
        #     return w, False

        # ---------------

        # TEST

        # -------------------------
        # Polynomial root selection
        # -------------------------
        # We solve: P(z + alpha/w, tau*w) = 0.
        # Let y = tau*w. Then alpha/w = alpha*tau/y = (tau - 1)/y.
        # So we solve in y:
        #     P(z + beta/y, y) = 0,  beta = tau - 1.
        # Multiply by y^deg_z to clear denominators and get a polynomial in y.

        a = numpy.asarray(a_coeffs, dtype=numpy.complex128)
        deg_z = a.shape[0] - 1
        deg_m = a.shape[1] - 1

        beta = tau - 1.0

        # poly_y[p] stores coeff of y^p after clearing denominators
        poly_y = numpy.zeros(deg_z + deg_m + 1, dtype=numpy.complex128)

        # Build polynomial: sum_{i,j} a[i,j] (z + beta/y)^i y^j * y^{deg_z}
        # Expand (z + beta/y)^i = sum_{k=0}^i C(i,k) z^{i-k} (beta/y)^k
        # Term contributes to power p = deg_z + j - k.
        from math import comb
        for i in range(deg_z + 1):
            for j in range(deg_m + 1):
                aij = a[i, j]
                if aij == 0:
                    continue
                for k in range(i + 1):
                    p = deg_z + j - k
                    poly_y[p] += aij * comb(i, k) * (z ** (i - k)) * (beta ** k)

        # numpy.roots expects highest degree first
        coeffs = poly_y[::-1]

        # If leading coefficients are ~0, trim (rare but safe)
        nz_lead = numpy.flatnonzero(numpy.abs(coeffs) > 0)
        if nz_lead.size == 0:
            return w, False
        coeffs = coeffs[nz_lead[0]:]

        roots_y = numpy.roots(coeffs)

        # Pick root with Im(w)>0 (if z in upper half-plane), closest to time seed
        y_seed = tau * w_init
        best = None
        best_score = None

        for y in roots_y:
            if not numpy.isfinite(y.real) or not numpy.isfinite(y.imag):
                continue

            w_cand = y / tau

            if want_pos_imag and (w_cand.imag <= 0.0):
                continue

            if abs(w_cand) < w_min:
                continue

            # score: stick to time continuation
            score = abs(y - y_seed)

            if (best_score is None) or (score < best_score):
                best = w_cand
                best_score = score

        if best is None:
            return w, False

        w = complex(best)

        # final residual check
        F_end = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
        return w, (abs(F_end) <= 1e3 * tol)

    # -------------------


    F_end = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
    return w, (abs(F_end) <= 10.0 * tol)


# ============
# NEW FUNCTION
# ============

def fd_candidates_w(z, t, a_coeffs, w_min=1e-14):
    """
    Return candidate roots w solving P(z + alpha/w, tau*w)=0 with Im(w)>0 (if Im(z)>0).
    """
    z = complex(z)
    tau = float(numpy.exp(t))
    alpha = 1.0 - 1.0 / tau
    want_pos_imag = (z.imag > 0.0)

    a = numpy.asarray(a_coeffs, dtype=numpy.complex128)
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
        # -------------
        # TEST
        # F = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
        # if abs(F) < 1e-6:
        #     cands.append(complex(w))
        # ---------------
        # TEST
        cands.append(complex(w))
        # ------------------

    return cands


# =================
# decompress newton
# =================

def decompress_newton(z_list, t_grid, a_coeffs, w0_list=None,
                      dt_max=0.1, sweep=True, time_rel_tol=5.0,
                      active_imag_eps=None, sweep_pad=20,
                      max_iter=50, tol=1e-12, armijo=1e-4,
                      min_lam=1e-6, w_min=1e-14,
                      viterbi_opt=None):
    """
    Evolve w = m(t,z) on a fixed z grid and time grid using FD.

    This implementation uses a global 1D Viterbi/DP branch-tracker along the
    spatial grid at every time step to avoid local root mis-selection (multi-bulk
    stability). The inputs sweep/time_rel_tol/active_imag_eps/sweep_pad are kept
    for backward compatibility but are ignored by the Viterbi tracker.

    Parameters
    ----------
    z_list : array_like of complex
        Query points z (typically x + 1j*eta with eta > 0), ordered along x.
    t_grid : array_like of float
        Strictly increasing time grid.
    a_coeffs : ndarray
        Coefficients defining P(z,m) in the monomial basis.
    w0_list : array_like of complex
        Initial values w(t0,z) at t_grid[0].

    viterbi_opt : dict or None
        Options for the Viterbi tracker. Keys (all optional):
            lam_space : float (default 1.0)
            lam_time  : float (default 0.25)
            lam_im    : float (default 1e3)   penalty = lam_im / max(|Im(w)|, eps)
            tol_im    : float (default 1e-12) Herglotz sign tolerance
            edge_k    : int   (default 3)     # of points at each end with asym penalty
            lam_asym  : float (default 0.2)   penalty = lam_asym * |z*w + 1|
            refine_newton : bool (default True) refine chosen path with fd_solve_w

    Returns
    -------
    W : ndarray, shape (len(t_grid), len(z_list))
        Evolved values w(t,z).
    ok : ndarray of bool, same shape as W
        Convergence flags from the accepted solve at each point.
    """

    z_list = numpy.asarray(z_list, dtype=complex).ravel()
    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    nt = t_grid.size
    nz = z_list.size

    if w0_list is None:
        raise ValueError("w0_list must be provided (initial m(z) at t_grid[0]).")

    w0_list = numpy.asarray(w0_list, dtype=complex).ravel()
    if w0_list.size != nz:
        raise ValueError("w0_list must have the same size as z_list.")

    if nt == 0:
        return numpy.empty((0, nz), dtype=complex), numpy.empty((0, nz), dtype=bool)

    # Viterbi options
    opt = {} if viterbi_opt is None else dict(viterbi_opt)
    lam_space = float(opt.get('lam_space', 1.0))
    lam_time = float(opt.get('lam_time', 0.25))
    lam_im = float(opt.get('lam_im', 1.0e3))
    tol_im = float(opt.get('tol_im', 1.0e-12))
    edge_k = int(opt.get('edge_k', 3))
    lam_asym = float(opt.get('lam_asym', 0.2))
    refine_newton = bool(opt.get('refine_newton', True))

    W = numpy.empty((nt, nz), dtype=complex)
    ok = numpy.zeros((nt, nz), dtype=bool)

    W[0, :] = w0_list
    ok[0, :] = True
    w_prev = W[0, :].copy()

    # -----------------
    # helper: candidates
    # -----------------

    def _candidates(iz, t):
        cands = fd_candidates_w(z_list[iz], t, a_coeffs, w_min=w_min)
        if len(cands) == 0:
            # fallback: carry previous value as a candidate
            return [complex(w_prev[iz])]
        return cands

    # -------------------------
    # helper: unary / transition
    # -------------------------

    def _want_pos_imag(z):
        return (complex(z).imag > 0.0)

    def _herglotz_ok(w, z):
        z = complex(z)
        w = complex(w)
        if not _want_pos_imag(z):
            return True
        return (w.imag > -tol_im)

    def _unary_cost(w, iz, t):
        # penalize wrong sign heavily
        if not _herglotz_ok(w, z_list[iz]):
            return 1.0e30

        # time continuity
        wt = complex(w_prev[iz])
        c = lam_time * (abs(w - wt) ** 2)

        # discourage tiny-imag traps (safe substitute for any global Im-reward)
        im = abs(w.imag)
        c += lam_im / max(im, 1e-16)

        # asymptotic anchor only near ends
        if edge_k > 0 and (iz < edge_k or iz >= nz - edge_k):
            z = complex(z_list[iz])
            c += lam_asym * abs(z * w + 1.0)

        return c

    def _trans_cost(w_left, w_right):
        return lam_space * (abs(w_right - w_left) ** 2)

    # -------------
    # time evolution
    # -------------

    for it in range(1, nt):
        t = float(t_grid[it])

        # build candidates list per spatial index
        C = []
        for iz in range(nz):
            C.append(_candidates(iz, t))

        # DP tables with variable state sizes
        dp = []
        prev_idx = []

        # init
        c0 = C[0]
        dp0 = numpy.array([_unary_cost(w, 0, t) for w in c0], dtype=float)
        dp.append(dp0)
        prev_idx.append(numpy.full(dp0.size, -1, dtype=int))

        # forward pass
        for iz in range(1, nz):
            ci = C[iz]
            dp_i = numpy.full(len(ci), numpy.inf, dtype=float)
            prev_i = numpy.full(len(ci), -1, dtype=int)

            dp_prev = dp[iz - 1]
            c_prev = C[iz - 1]

            for j, wj in enumerate(ci):
                u = _unary_cost(wj, iz, t)

                best = numpy.inf
                best_k = -1
                for k, wk in enumerate(c_prev):
                    val = dp_prev[k] + _trans_cost(wk, wj)
                    if val < best:
                        best = val
                        best_k = k

                dp_i[j] = u + best
                prev_i[j] = best_k

            dp.append(dp_i)
            prev_idx.append(prev_i)

        # backtrack
        w_row = numpy.empty(nz, dtype=complex)
        ok_row = numpy.zeros(nz, dtype=bool)

        j = int(numpy.argmin(dp[-1]))
        w_row[-1] = complex(C[-1][j])

        for iz in range(nz - 1, 0, -1):
            j = int(prev_idx[iz][j])
            if j < 0:
                j = 0
            w_row[iz - 1] = complex(C[iz - 1][j])

        # optional Newton refinement on chosen path
        if refine_newton:
            for iz in range(nz):
                w_sol, success = fd_solve_w(
                    z_list[iz], t, a_coeffs, w_row[iz],
                    max_iter=max_iter, tol=tol, armijo=armijo,
                    min_lam=min_lam, w_min=w_min)
                w_row[iz] = w_sol
                ok_row[iz] = success
        else:
            ok_row[:] = True

        W[it, :] = w_row
        ok[it, :] = ok_row
        w_prev = w_row

    return W, ok
