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

__all__ = ['build_time_grid', 'decompress_newton_old', 'decompress_newton']


# ===============
# build time grid
# ===============

def build_time_grid(sizes, n0, min_n_times=0):
    """
    sizes: list/array of requested matrix sizes (e.g. [2000,3000,4000,8000])
    n0:    initial size (self.n)
    min_n_times: minimum number of time points to run Newton sweep on

    Returns
    -------
    t_all: sorted time grid to run solver on
    idx_req: indices of requested times inside t_all (same order as sizes)
    """

    sizes = numpy.asarray(sizes, dtype=float)
    alpha = sizes / float(n0)
    t_req = numpy.log(alpha)

    # Always include t=0 and T=max(t_req)
    T = float(numpy.max(t_req)) if t_req.size else 0.0
    base = numpy.unique(numpy.r_[0.0, t_req, T])
    t_all = numpy.sort(base)

    # Add points only if needed: split largest gaps
    N = int(min_n_times) if min_n_times is not None else 0
    while t_all.size < N and t_all.size >= 2:
        gaps = numpy.diff(t_all)
        k = int(numpy.argmax(gaps))
        mid = 0.5 * (t_all[k] + t_all[k+1])
        t_all = numpy.sort(numpy.unique(numpy.r_[t_all, mid]))

    # Map each requested time to an index in t_all (stable, no float drama)
    # (t_req values came from same construction, so they should match exactly;
    # still: use searchsorted + assert)
    idx_req = numpy.searchsorted(t_all, t_req)
    # optional sanity:
    # assert numpy.allclose(t_all[idx_req], t_req, rtol=0, atol=0)

    return t_all, idx_req


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
#
#         # ----------------
#
#         # if not numpy.isfinite(w.real) or not numpy.isfinite(w.imag):
#         #     return w, False
#         # if abs(w) < w_min:
#         #     return w, False
#         # if want_pos_imag and (w.imag <= 0.0):
#         #     return w, False
#         #
#         # zeta = z + alpha / w
#         # y = tau * w
#         #
#         # F, Pz, Py = eval_P_partials(zeta, y, a_coeffs)
#         # F = complex(F)
#         # Pz = complex(Pz)
#         # Py = complex(Py)
#         #
#         # if abs(F) <= tol:
#         #     return w, True
#         #
#         # dF = (-alpha / (w * w)) * Pz + tau * Py
#         # if dF == 0.0:
#         #     return w, False
#         #
#         # step = -F / dF
#         #
#         # lam = 1.0
#         # F_abs = abs(F)
#         # ok = False
#         #
#         # while lam >= min_lam:
#         #     w_new = w + lam * step
#         #     if abs(w_new) < w_min:
#         #         lam *= 0.5
#         #         continue
#         #     if want_pos_imag and (w_new.imag <= 0.0):
#         #         lam *= 0.5
#         #         continue
#         #
#         #     zeta_new = z + alpha / w_new
#         #     y_new = tau * w_new
#         #
#         #     F_new = eval_P_partials(zeta_new, y_new, a_coeffs)[0]
#         #     F_new = complex(F_new)
#         #
#         #     if abs(F_new) <= (1.0 - armijo * lam) * F_abs:
#         #         w = w_new
#         #         ok = True
#         #         break
#         #
#         #     lam *= 0.5
#         #
#         # if not ok:
#         #     return w, False
#
#         # ---------------
#
#         # TEST
#
#         # -------------------------
#         # Polynomial root selection
#         # -------------------------
#         # We solve: P(z + alpha/w, tau*w) = 0.
#         # Let y = tau*w. Then alpha/w = alpha*tau/y = (tau - 1)/y.
#         # So we solve in y:
#         #     P(z + beta/y, y) = 0,  beta = tau - 1.
#         # Multiply by y^deg_z to clear denominators and get a polynomial in y.
#
#         a = numpy.asarray(a_coeffs, dtype=numpy.complex128)
#         deg_z = a.shape[0] - 1
#         deg_m = a.shape[1] - 1
#
#         beta = tau - 1.0
#
#         # poly_y[p] stores coeff of y^p after clearing denominators
#         poly_y = numpy.zeros(deg_z + deg_m + 1, dtype=numpy.complex128)
#
#         # Build polynomial: sum_{i,j} a[i,j] (z + beta/y)^i y^j * y^{deg_z}
#         # Expand (z + beta/y)^i = sum_{k=0}^i C(i,k) z^{i-k} (beta/y)^k
#         # Term contributes to power p = deg_z + j - k.
#         from math import comb
#         for i in range(deg_z + 1):
#             for j in range(deg_m + 1):
#                 aij = a[i, j]
#                 if aij == 0:
#                     continue
#                 for k in range(i + 1):
#                     p = deg_z + j - k
#                     poly_y[p] += aij * comb(i, k) * (z ** (i - k)) * (beta ** k)
#
#         # numpy.roots expects highest degree first
#         coeffs = poly_y[::-1]
#
#         # If leading coefficients are ~0, trim (rare but safe)
#         nz_lead = numpy.flatnonzero(numpy.abs(coeffs) > 0)
#         if nz_lead.size == 0:
#             return w, False
#         coeffs = coeffs[nz_lead[0]:]
#
#         roots_y = numpy.roots(coeffs)
#
#         # Pick root with Im(w)>0 (if z in upper half-plane), closest to time seed
#         y_seed = tau * w_init
#         best = None
#         best_score = None
#
#         for y in roots_y:
#             if not numpy.isfinite(y.real) or not numpy.isfinite(y.imag):
#                 continue
#
#             w_cand = y / tau
#
#             if want_pos_imag and (w_cand.imag <= 0.0):
#                 continue
#
#             if abs(w_cand) < w_min:
#                 continue
#
#             # score: stick to time continuation
#             score = abs(y - y_seed)
#
#             if (best_score is None) or (score < best_score):
#                 best = w_cand
#                 best_score = score
#
#         if best is None:
#             return w, False
#
#         w = complex(best)
#
#         # final residual check
#         F_end = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
#         return w, (abs(F_end) <= 1e3 * tol)
#
#     # -------------------
#
#
#     F_end = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
#     return w, (abs(F_end) <= 10.0 * tol)

def fd_solve_w(z, t, a_coeffs, w_init, max_iter=50, tol=1e-12,
               armijo=1e-4, min_lam=1e-6, w_min=1e-14):
    """
    Damped Newton solve for w from F_t(z,w)=P(z+alpha/w, tau*w)=0.

    Convention: m(z)=∫ rho(x)/(x-z) dx, so for z in C^+ we want Im(w)>0.
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

        F, Pz, Py = eval_P_partials(zeta, y, a_coeffs)
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

            F_new = eval_P_partials(z + alpha / w_new, tau * w_new, a_coeffs)[0]
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
    F_end = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
    F_end = complex(F_end)
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


# =====================
# decompress newton old
# =====================

# def decompress_newton_old(z_list, t_grid, a_coeffs, w0_list=None,
#                           dt_max=0.1, sweep=True, time_rel_tol=5.0,
#                           max_iter=50, tol=1e-12, armijo=1e-4,
#                           min_lam=1e-6, w_min=1e-14):
#     """
#     Evolve w = m(t,z) on a fixed z grid and time grid using FD.
#
#     Parameters
#     ----------
#     z_list : array_like of complex
#         Query points z (typically x + 1j*eta with eta > 0).
#     t_grid : array_like of float
#         Strictly increasing time grid.
#     a_coeffs : ndarray
#         Coefficients defining P(zeta,y) in the monomial basis used by eval_P.
#     w0_list : array_like of complex
#         Initial values at t_grid[0] (typically m0(z_list) on the physical
#         branch).
#     dt_max : float, optional
#         Maximum internal time step. Larger dt is handled by substepping.
#     sweep : bool, optional
#         If True, use spatial continuation (neighbor seeding) plus a
#         time-consistency check to prevent branch collapse. If False, solve
#         each z independently from the previous-time seed (faster but may
#         branch-switch for small eta).
#     time_rel_tol : float, optional
#         When sweep=True, if the neighbor-seeded solution differs from the
#         previous-time value w_prev by more than time_rel_tol*(1+|w_prev|), we
#         also solve using the previous-time seed and select the closer one.
#     max_iter : int, optional
#         Maximum Newton iterations in fd_solve_w.
#     tol : float, optional
#         Residual tolerance in fd_solve_w.
#     armijo : float, optional
#         Armijo parameter for backtracking in fd_solve_w.
#     min_lam : float, optional
#         Minimum damping factor in fd_solve_w backtracking.
#     w_min : float, optional
#         Minimum |w| allowed to avoid singularity.
#
#     Returns
#     -------
#     W : ndarray, shape (len(t_grid), len(z_list))
#         Evolved values w(t,z).
#     ok : ndarray of bool, same shape as W
#         Convergence flags from the final accepted solve at each point.
#
#     Notes
#     -----
#     For very small eta, the implicit FD equation can have multiple roots in the
#     upper half-plane. The sweep option is a branch-selection mechanism. The
#     time-consistency check is critical at large t to avoid propagating a
#     nearly-real spurious root across x.
#
#     Examples
#     --------
#     .. code-block:: python
#
#         x = numpy.linspace(-0.5, 2.5, 2000)
#         eta = 1e-6
#         z_query = x + 1j*eta
#         w0_list = m1_fn(z_query)
#
#         t_grid = numpy.linspace(0.0, 4.0, 2)
#         W, ok = fd_evolve_on_grid(
#             z_query, t_grid, a_coeffs, w0_list=w0_list,
#             dt_max=0.1, sweep=True, time_rel_tol=5.0,
#             max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6, w_min=1e-14
#         )
#         rho = W.imag / numpy.pi
#     """
#     z_list = numpy.asarray(z_list, dtype=complex).ravel()
#     t_grid = numpy.asarray(t_grid, dtype=float).ravel()
#     nt = t_grid.size
#     nz = z_list.size
#
#     W = numpy.empty((nt, nz), dtype=complex)
#     ok = numpy.zeros((nt, nz), dtype=bool)
#
#     if w0_list is None:
#         raise ValueError(
#             "w0_list must be provided (e.g. m1_fn(z_list) at t=0).")
#     w_prev = numpy.asarray(w0_list, dtype=complex).ravel()
#     if w_prev.size != nz:
#         raise ValueError("w0_list must have same size as z_list.")
#
#     W[0, :] = w_prev
#     ok[0, :] = True
#
#     sweep = bool(sweep)
#     time_rel_tol = float(time_rel_tol)
#
#     for it in range(1, nt):
#         t0 = float(t_grid[it - 1])
#         t1 = float(t_grid[it])
#         dt = t1 - t0
#         if dt <= 0.0:
#             raise ValueError("t_grid must be strictly increasing.")
#
#         # Internal substepping makes time-continuity a strong selector.
#         n_sub = int(numpy.ceil(dt / float(dt_max)))
#         if n_sub < 1:
#             n_sub = 1
#
#         for ks in range(1, n_sub + 1):
#             t = t0 + dt * (ks / float(n_sub))
#
#             w_row = numpy.empty(nz, dtype=complex)
#             ok_row = numpy.zeros(nz, dtype=bool)
#
#             if not sweep:
#                 # Independent solves: each point uses previous-time seed only.
#                 for iz in range(nz):
#                     w, success = fd_solve_w(
#                         z_list[iz], t, a_coeffs, w_prev[iz],
#                         max_iter=max_iter, tol=tol, armijo=armijo,
#                         min_lam=min_lam, w_min=w_min
#                     )
#                     w_row[iz] = w
#                     ok_row[iz] = success
#
#                 w_prev = w_row
#                 continue
#
#             # Center-out sweep seed: pick where previous-time Im is largest.
#             i0 = int(numpy.argmax(numpy.abs(numpy.imag(w_prev))))
#
#             w0, ok0 = fd_solve_w(
#                 z_list[i0], t, a_coeffs, w_prev[i0],
#                 max_iter=max_iter, tol=tol, armijo=armijo,
#                 min_lam=min_lam, w_min=w_min
#             )
#             w_row[i0] = w0
#             ok_row[i0] = ok0
#
#             # -----------------
#             # slove with choice
#             # -----------------
#
#             def solve_with_choice(iz, w_neighbor):
#                 # First try neighbor-seeded Newton (spatial continuity).
#                 w_a, ok_a = fd_solve_w(
#                     z_list[iz], t, a_coeffs, w_neighbor,
#                     max_iter=max_iter, tol=tol, armijo=armijo,
#                     min_lam=min_lam, w_min=w_min
#                 )
#
#                 # Always keep a time-consistent fallback candidate.
#                 w_b, ok_b = fd_solve_w(
#                     z_list[iz], t, a_coeffs, w_prev[iz],
#                     max_iter=max_iter, tol=tol, armijo=armijo,
#                     min_lam=min_lam, w_min=w_min
#                 )
#
#                 if ok_a and ok_b:
#                     # Prefer the root closer to previous-time value (time
#                     # continuation).
#                     da = abs(w_a - w_prev[iz])
#                     db = abs(w_b - w_prev[iz])
#
#                     # If neighbor result is wildly off, reject it.
#                     if da > time_rel_tol * (1.0 + abs(w_prev[iz])):
#                         return w_b, True
#
#                     return (w_a, True) if (da <= db) else (w_b, True)
#
#                 if ok_a:
#                     # If only neighbor succeeded, still guard against extreme
#                     # drift.
#                     da = abs(w_a - w_prev[iz])
#                     if da > time_rel_tol * (1.0 + abs(w_prev[iz])) and ok_b:
#                         return w_b, True
#                     return w_a, True
#
#                 if ok_b:
#                     return w_b, True
#
#                 return w_a, False
#
#             # Sweep right
#             for iz in range(i0 + 1, nz):
#                 w_row[iz], ok_row[iz] = solve_with_choice(iz, w_row[iz - 1])
#
#             # Sweep left
#             for iz in range(i0 - 1, -1, -1):
#                 w_row[iz], ok_row[iz] = solve_with_choice(iz, w_row[iz + 1])
#
#             w_prev = w_row
#
#         W[it, :] = w_prev
#         ok[it, :] = ok_row
#
#     return W, ok


# =================
# decompress newton
# =================

# def decompress_newton(z_list, t_grid, a_coeffs, w0_list=None,
#                       dt_max=0.1, sweep=True, time_rel_tol=5.0,
#                       active_imag_eps=None, sweep_pad=20,
#                       max_iter=50, tol=1e-12, armijo=1e-4,
#                       min_lam=1e-6, w_min=1e-14):
#     """
#     Evolve w = m(t,z) on a fixed z grid and time grid using FD.
#
#     Parameters
#     ----------
#     z_list : array_like of complex
#         Query points z (typically x + 1j*eta with eta > 0), ordered along x.
#     t_grid : array_like of float
#         Strictly increasing time grid.
#     a_coeffs : ndarray
#         Coefficients defining P(zeta,y) in the monomial basis.
#     w0_list : array_like of complex
#         Initial values at t_grid[0] (typically m0(z_list) on the physical
#         branch).
#     dt_max : float, optional
#         Maximum internal time step. Larger dt is handled by substepping.
#     sweep : bool, optional
#         If True, enforce spatial continuity within active (bulk) regions and
#         allow edge activation via padding. If False, solve each z independently
#         from previous-time seeds (may fail to "activate" new support near
#         edges).
#     time_rel_tol : float, optional
#         When sweep=True, reject neighbor-propagated solutions that drift too
#         far from the previous-time value, using a time-consistent fallback.
#     active_imag_eps : float or None, optional
#         Threshold on |Im(w_prev)| to define active/bulk indices. If None, it is
#         set to 50*Im(z_list[0]) (works well when z_list=x+i*eta).
#     sweep_pad : int, optional
#         Number of indices used to dilate the active region. This is crucial for
#         multi-bulk laws so that edges can move and points just outside a bulk
#         can be initialized from the interior.
#     max_iter, tol, armijo, min_lam, w_min : optional
#         Newton/backtracking controls passed to fd_solve_w.
#
#     Returns
#     -------
#     W : ndarray, shape (len(t_grid), len(z_list))
#         Evolved values w(t,z).
#     ok : ndarray of bool, same shape as W
#         Convergence flags from the accepted solve at each point.
#     """
#
#     z_list = numpy.asarray(z_list, dtype=complex).ravel()
#     t_grid = numpy.asarray(t_grid, dtype=float).ravel()
#     nt = t_grid.size
#     nz = z_list.size
#
#     W = numpy.empty((nt, nz), dtype=complex)
#     ok = numpy.zeros((nt, nz), dtype=bool)
#
#     if w0_list is None:
#         raise ValueError(
#             "w0_list must be provided (e.g. m1_fn(z_list) at t=0).")
#     w_prev = numpy.asarray(w0_list, dtype=complex).ravel()
#     if w_prev.size != nz:
#         raise ValueError("w0_list must have same size as z_list.")
#
#     W[0, :] = w_prev
#     ok[0, :] = True
#
#     sweep = bool(sweep)
#     time_rel_tol = float(time_rel_tol)
#     sweep_pad = int(sweep_pad)
#
#     # If z_list is x + i*eta, use eta to set an automatic activity threshold.
#     if active_imag_eps is None:
#         eta0 = float(abs(z_list[0].imag))
#         active_imag_eps = 50.0 * eta0 if eta0 > 0.0 else 1e-10
#     active_imag_eps = float(active_imag_eps)
#
#     # --------------------------------------
#     # TEST
#     # def solve_with_choice(iz, w_seed):
#     #     # Neighbor-seeded candidate (spatial continuity)
#     #     w_a, ok_a = fd_solve_w(
#     #         z_list[iz], t, a_coeffs, w_seed,
#     #         max_iter=max_iter, tol=tol, armijo=armijo,
#     #         min_lam=min_lam, w_min=w_min
#     #     )
#     #
#     #     # Time-seeded candidate (time continuation)
#     #     w_b, ok_b = fd_solve_w(
#     #         z_list[iz], t, a_coeffs, w_prev[iz],
#     #         max_iter=max_iter, tol=tol, armijo=armijo,
#     #         min_lam=min_lam, w_min=w_min
#     #     )
#     #
#     #     if ok_a and ok_b:
#     #         da = abs(w_a - w_prev[iz])
#     #         db = abs(w_b - w_prev[iz])
#     #
#     #         # Reject neighbor result if it drifted too far in one step
#     #         if da > time_rel_tol * (1.0 + abs(w_prev[iz])):
#     #             return w_b, True
#     #
#     #         return (w_a, True) if (da <= db) else (w_b, True)
#     #
#     #     if ok_a:
#     #         da = abs(w_a - w_prev[iz])
#     #         if da > time_rel_tol * (1.0 + abs(w_prev[iz])) and ok_b:
#     #             return w_b, True
#     #         return w_a, True
#     #
#     #     if ok_b:
#     #         return w_b, True
#     #
#     #     return w_a, False
#     # ----------------------------------------
#     # TEST
#     # def solve_with_choice(iz, w_seed):
#     #     # candidate roots at this (t,z)
#     #     cands = fd_candidates_w(z_list[iz], t, a_coeffs, w_min=w_min)
#     #
#     #     # ---------------------
#     #     # TEST
#     #     if iz in (0, nz//2, nz-1):
#     #         ims = [float(w.imag) for w in cands]
#     #         print(f"      iz={iz} ncand={len(cands)} Im(cands) min/med/max="
#     #               f"{(min(ims) if ims else None)}/"
#     #               f"{(numpy.median(ims) if ims else None)}/"
#     #               f"{(max(ims) if ims else None)}")
#     #     # ---------------------
#     #
#     #     if len(cands) == 0:
#     #         # fallback to your existing single-root solver
#     #         w, success = fd_solve_w(
#     #             z_list[iz], t, a_coeffs, w_prev[iz],
#     #             max_iter=max_iter, tol=tol, armijo=armijo,
#     #             min_lam=min_lam, w_min=w_min
#     #         )
#     #         return w, success
#     #
#     #     # cost = spatial continuity + time continuity (tune weights if needed)
#     #     w_time = w_prev[iz]
#     #     w_space = w_seed
#     #     best = None
#     #     best_cost = None
#     #
#     #     for w in cands:
#     #         # TEST
#     #         # cost = abs(w - w_space) + 0.25 * abs(w - w_time)
#     #         # TEST
#     #         # prefer continuity, but also prefer larger Im(w) to stay on the bulk branch
#     #         cost = abs(w - w_space) + 0.25 * abs(w - w_time) - 5.0 * w.imag
#     #         # --------------
#     #
#     #         if (best_cost is None) or (cost < best_cost):
#     #             best = w
#     #             best_cost = cost
#     #
#     #     return best, True
#     # ----------------------------------------
#     # TEST
#     def solve_with_choice(iz, w_neighbor):
#         # Neighbor-seeded Newton (spatial continuity).
#         w_a, ok_a = fd_solve_w(
#             z_list[iz], t, a_coeffs, w_neighbor,
#             max_iter=max_iter, tol=tol, armijo=armijo,
#             min_lam=min_lam, w_min=w_min
#         )
#
#         # Time-seeded Newton (time continuity).
#         w_b, ok_b = fd_solve_w(
#             z_list[iz], t, a_coeffs, w_prev[iz],
#             max_iter=max_iter, tol=tol, armijo=armijo,
#             min_lam=min_lam, w_min=w_min
#         )
#
#         z_here = z_list[iz]
#         w_asymp = -1.0 / z_here  # mass=1 Stieltjes asymptote
#
#         def score(w):
#             # prefer time continuity + correct asymptote (stronger for large |z|)
#             return (
#                 abs(w - w_prev[iz])
#                 + 0.2 * abs(z_here) * abs(w - w_asymp)
#             )
#
#         if ok_a and ok_b:
#             # hard reject neighbor result if it jumped in time
#             da = abs(w_a - w_prev[iz])
#             if da > time_rel_tol * (1.0 + abs(w_prev[iz])):
#                 return w_b, True
#
#             return (w_a, True) if (score(w_a) <= score(w_b)) else (w_b, True)
#
#         if ok_a:
#             # if only neighbor succeeded, still reject if it jumped badly
#             da = abs(w_a - w_prev[iz])
#             if da > time_rel_tol * (1.0 + abs(w_prev[iz])) and ok_b:
#                 return w_b, True
#             return w_a, True
#
#         if ok_b:
#             return w_b, True
#
#         return w_a, False
#     # ----------------------------------------
#
#     for it in range(1, nt):
#         t0 = float(t_grid[it - 1])
#         t1 = float(t_grid[it])
#         dt = t1 - t0
#         if dt <= 0.0:
#             raise ValueError("t_grid must be strictly increasing.")
#
#         # Substep in time to keep continuation safe.
#         n_sub = int(numpy.ceil(dt / float(dt_max)))
#         if n_sub < 1:
#             n_sub = 1
#
#         for ks in range(1, n_sub + 1):
#             t = t0 + dt * (ks / float(n_sub))
#
#             w_row = numpy.empty(nz, dtype=complex)
#             ok_row = numpy.zeros(nz, dtype=bool)
#
#             if not sweep:
#                 # Independent solves: can miss edge activation in multi-bulk
#                 # problems.
#                 for iz in range(nz):
#                     w, success = fd_solve_w(
#                         z_list[iz], t, a_coeffs, w_prev[iz],
#                         max_iter=max_iter, tol=tol, armijo=armijo,
#                         min_lam=min_lam, w_min=w_min
#                     )
#                     w_row[iz] = w
#                     ok_row[iz] = success
#
#                 w_prev = w_row
#                 continue
#
#             # Define "active" region from previous time: inside bulks
#             # Im(w_prev) is O(1), outside bulks Im(w_prev) is ~O(eta). Dilate
#             # by sweep_pad to allow edges to move.
#
#             # ------------------------------
#             # TEST
#             # active = (numpy.abs(numpy.imag(w_prev)) > active_imag_eps)
#             # active_pad = active.copy()
#             # if sweep_pad > 0 and numpy.any(active):
#             #     idx = numpy.flatnonzero(active)
#             #     for i in idx:
#             #         lo = 0 if (i - sweep_pad) < 0 else (i - sweep_pad)
#             #         hi = \
#             #             nz if (i + sweep_pad + 1) > nz else (i + sweep_pad + 1)
#             #         active_pad[lo:hi] = True
#             # ------------------------------
#             # TEST
#             active = (numpy.abs(numpy.imag(w_prev)) > active_imag_eps)
#
#             # Split active indices into contiguous blocks (bulks)
#             pad_label = -numpy.ones(nz, dtype=numpy.int64)   # bulk id per index
#             active_pad = numpy.zeros(nz, dtype=bool)
#
#             idx = numpy.flatnonzero(active)
#             if idx.size > 0:
#                 cuts = numpy.where(numpy.diff(idx) > 1)[0]
#                 blocks = numpy.split(idx, cuts + 1)
#
#                 # Build padded intervals + centers
#                 centers = []
#                 pads = []
#                 for b in blocks:
#                     centers.append(int((b[0] + b[-1]) // 2))
#                     lo = int(max(0, b[0] - sweep_pad))
#                     hi = int(min(nz - 1, b[-1] + sweep_pad))
#                     pads.append((lo, hi))
#
#                 # Union of padded regions
#                 for lo, hi in pads:
#                     active_pad[lo:hi + 1] = True
#
#                 # Assign each padded index to the nearest bulk center (no overlap label)
#                 idx_u = numpy.flatnonzero(active_pad)
#                 c = numpy.asarray(centers, dtype=numpy.int64)
#                 dist = numpy.abs(idx_u[:, None] - c[None, :])
#                 winner = numpy.argmin(dist, axis=1).astype(numpy.int64)
#                 pad_label[idx_u] = winner
#             # ------------------------------
#
#             # ------------------------------
#             # TEST
#             def _ranges(idxs):
#                 if idxs.size == 0:
#                     return []
#                 cuts = numpy.where(numpy.diff(idxs) > 1)[0]
#                 blocks = numpy.split(idxs, cuts + 1)
#                 return [(int(b[0]), int(b[-1])) for b in blocks]
#
#             # print("    pad_label>=0 ranges:", _ranges(numpy.flatnonzero(pad_label >= 0)))
#             # print("    overlap(-2) ranges:", _ranges(numpy.flatnonzero(pad_label == -2)))
#             # ------------------------------
#
#
#
#             # ----------------------------------------------
#
#             # TEST
#             # eta = float(abs(z_list[0].imag))
#             #
#             # # Barrier: points that look like "gap" (tiny Im(w_prev))
#             # barrier_eps = 10.0 * eta   # try 5*eta or 10*eta
#             # barrier = (numpy.abs(numpy.imag(w_prev)) <= barrier_eps)
#
#
#
#             # TEST
#             # -------------------------
#             # --- diagnostics ---
#             active = (numpy.abs(numpy.imag(w_prev)) > active_imag_eps)
#             idx_active = numpy.flatnonzero(active)
#             idx_pad = numpy.flatnonzero(active_pad)
#
#             def _ranges(idxs):
#                 if idxs.size == 0:
#                     return []
#                 cuts = numpy.where(numpy.diff(idxs) > 1)[0]
#                 blocks = numpy.split(idxs, cuts + 1)
#                 return [(b[0], b[-1]) for b in blocks]
#
#             # print(f"[t={t:.6g}] eta={z_list[0].imag:.2e} active_eps={active_imag_eps:.2e} "
#             #       f"active_n={idx_active.size}/{nz} pad_n={idx_pad.size}/{nz} "
#             #       f"active_ranges={_ranges(idx_active)} pad_ranges={_ranges(idx_pad)}")
#
#             # Track the physical “gap” region around between bulks by looking at low Im(w_prev)
#             gap = numpy.abs(numpy.imag(w_prev)) <= 5.0 * z_list[0].imag
#             ig = numpy.flatnonzero(gap)
#             # if ig.size > 0:
#             #     print(f"    gap_ranges(Im<=5eta)={_ranges(ig)} "
#             #           f"Im(w) min/med/max = {numpy.min(w_prev.imag):.3e}/"
#             #           f"{numpy.median(w_prev.imag):.3e}/{numpy.max(w_prev.imag):.3e}")
#
#             # ------------------
#
#
#
#             # Left-to-right: use neighbor seed only within padded active
#             # regions, so we don't propagate a branch across the gap between
#             # bulks.
#             for iz in range(nz):
#                 if iz == 0:
#                     w_seed = w_prev[iz]
#                 else:
#                     # TEST
#                     # if active_pad[iz] and active_pad[iz - 1]:
#                     #     w_seed = w_row[iz - 1]
#                     # else:
#                     #     w_seed = w_prev[iz]
#                     # TEST
#                     # if (active_pad[iz] and active_pad[iz - 1] and
#                     #         (not barrier[iz]) and (not barrier[iz - 1])):
#                     #     w_seed = w_row[iz - 1]
#                     # else:
#                     #     w_seed = w_prev[iz]
#                     # ----------------------
#                     # TEST
#                     # if (active_pad[iz] and active_pad[iz - 1] and
#                     #         (pad_label[iz] == pad_label[iz - 1]) and
#                     #         (pad_label[iz] >= 0)):
#                     # -----------------
#                     # TEST
#                     if (active_pad[iz] and active_pad[iz - 1] and
#                         (pad_label[iz] == pad_label[iz - 1]) and
#                         (pad_label[iz] >= 0)):
#                         w_seed = w_row[iz - 1]
#                     else:
#                         w_seed = w_prev[iz]
#                     # ----------------------
#
#
#                 w_row[iz], ok_row[iz] = solve_with_choice(iz, w_seed)
#
#             # Right-to-left refinement: helps stabilize left edges of bulks.
#             for iz in range(nz - 2, -1, -1):
#                 # TEST
#                 # if active_pad[iz] and active_pad[iz + 1]:
#                 # TEST
#                 # if (active_pad[iz] and active_pad[iz + 1] and
#                 #         (not barrier[iz]) and (not barrier[iz + 1])):
#                 # TEST
#                 # if (active_pad[iz] and active_pad[iz + 1] and
#                 #         (pad_label[iz] == pad_label[iz + 1]) and
#                 #         (pad_label[iz] >= 0)):
#                 # TEST
#                 if (active_pad[iz] and active_pad[iz + 1] and
#                         (pad_label[iz] == pad_label[iz + 1]) and
#                         (pad_label[iz] >= 0)):
#
#                     w_seed = w_row[iz + 1]
#                     w_new, ok_new = solve_with_choice(iz, w_seed)
#                     if ok_new:
#                         # Keep the more time-consistent solution.
#                         if (not ok_row[iz]) or (abs(w_new - w_prev[iz]) <
#                                                 abs(w_row[iz] - w_prev[iz])):
#                             w_row[iz] = w_new
#                             ok_row[iz] = True
#
#
#
#             # TEST
#             # print(f'solved_ok={ok_row.sum()}/{nz}  (this substep)')
#
#             w_prev = w_row
#
#         W[it, :] = w_prev
#         ok[it, :] = ok_row
#
#     return W, ok


def eval_row_by_z_homotopy(
    t,
    z_targets,
    w_seed_targets,
    R,
    a_coeffs,
    w_anchor,
    *,
    steps=80,
    max_iter=50,
    tol=1e-12,
    armijo=1e-4,
    min_lam=1e-6,
    w_min=1e-14,
):
    """
    Evaluate w(t,z) on z_targets in C^+ by z-homotopy from z0=iR,
    but anchored at the TRUE w(t,z0)=w_anchor (computed separately).

    Path is 2-segment:
        z0=iR  ->  x+iR  ->  x+i*eta
    """
    import numpy

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
        # Filter Herglotz for your convention: Im(w)>0 on C^+
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
                z, t, a_coeffs, w,
                max_iter=max_iter, tol=tol, armijo=armijo,
                min_lam=min_lam, w_min=w_min
            )

            if not ok_new:
                cands = fd_candidates_w(z, t, a_coeffs, w_min=w_min)
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
                    z, t, a_coeffs, w,
                    max_iter=max_iter, tol=tol, armijo=armijo,
                    min_lam=min_lam, w_min=w_min
                )

                if not ok_new:
                    cands = fd_candidates_w(z, t, a_coeffs, w_min=w_min)
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
            cands = fd_candidates_w(zT, t, a_coeffs, w_min=w_min)
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
def decompress_newton(
    z_list,
    t_grid,
    a_coeffs,
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
    **_unused_kwargs,
):
    """
    Robust FD solver:
      (A) For each t, compute w(x+i*eta_track) by z-homotopy from z0=iR.
      (B) For each x, descend vertically from eta_track to target imag (typically 1e-5)
          using continuation in eta (vertical homotopy).
    This prevents multi-bulk cutoffs caused by trying to track directly at tiny eta.
    """
    import numpy

    z_list = numpy.asarray(z_list, dtype=numpy.complex128)
    t_grid = numpy.asarray(t_grid, dtype=float)
    nz = z_list.size
    nt = t_grid.size

    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")

    x_list = z_list.real
    eta_target = float(z_list.imag.max())  # your z_query uses constant imag
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

        # Herglotz for your convention: Im(w)>0 for z in C^+
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
            F, Pz, Py = eval_P_partials(zeta, y, a_coeffs)
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
                F_new = complex(eval_P_partials(zeta_new, y_new, a_coeffs)[0])

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
        F_end = complex(eval_P_partials(zeta, y, a_coeffs)[0])
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


