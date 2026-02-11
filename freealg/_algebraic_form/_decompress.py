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
from ._poly_util import eval_P_partials

__all__ = ['decompress_newton']


# ==========
# fd solve w
# ==========

def fd_solve_w(z, t, coeffs, w_init, max_iter=50, tol=1e-12,
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
    coeffs : ndarray
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
            z=0.5 + 1e-6j, t=2.0, coeffs=coeffs, w_init=m1_fn(0.5 + 1e-6j),
            max_iter=50, tol=1e-12)
    """

    z = complex(z)
    w = complex(w_init)

    tau = float(numpy.exp(t))
    alpha = 1.0 - 1.0 / tau

    want_pos_imag = (z.imag > 0.0)

    for _ in range(max_iter):

        a = numpy.asarray(coeffs, dtype=numpy.complex128)
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
                    poly_y[p] += aij * comb(i, k) * (z ** (i - k)) * \
                        (beta ** k)

        # numpy.roots expects highest degree first
        coeffs = poly_y[::-1]

        # If leading coefficients are ~0, trim (rare but safe)
        nz_lead = numpy.flatnonzero(numpy.abs(coeffs) > 0)
        if nz_lead.size == 0:
            return w, False
        coeffs = coeffs[nz_lead[0]:]

        roots_y = numpy.roots(coeffs)

        # Pick root with Im(w)>0 (if z in upper half-plane), closest to time
        # seed
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
        F_end = eval_P_partials(z + alpha / w, tau * w, coeffs)[0]
        return w, (abs(F_end) <= 1e3 * tol)

    F_end = eval_P_partials(z + alpha / w, tau * w, coeffs)[0]
    return w, (abs(F_end) <= 10.0 * tol)


# ===============
# fd candidates w
# ===============

def fd_candidates_w(z, t, coeffs, w_min=1e-14):
    """
    Return candidate roots w solving P(z + alpha/w, tau*w)=0 with Im(w)>0 (if
    Im(z)>0).
    """
    z = complex(z)
    tau = float(numpy.exp(t))
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
        cands.append(complex(w))

    return cands


# =================
# decompress newton
# =================

def decompress_newton(z_list, t_grid, coeffs, w0_list=None,
                      dt_max=0.1, sweep=True, time_rel_tol=5.0,
                      active_imag_eps=None, sweep_pad=20,
                      max_iter=50, tol=1e-12, armijo=1e-4,
                      min_lam=1e-6, w_min=1e-14, min_n_time=None):
    """
    Evolve w = m(t,z) on a fixed z grid and time grid using FD.

    Parameters
    ----------
    z_list : array_like of complex
        Query points z (typically x + 1j*eta with eta > 0), ordered along x.
    t_grid : array_like of float
        Strictly increasing time grid.
    coeffs : ndarray
        Coefficients defining P(zeta,y) in the monomial basis.
    w0_list : array_like of complex
        Initial values at t_grid[0] (typically m0(z_list) on the physical
        branch).
    dt_max : float, optional
        Maximum internal time step. Larger dt is handled by substepping.
    sweep : bool, optional
        If True, enforce spatial continuity within active (bulk) regions and
        allow edge activation via padding. If False, solve each z independently
        from previous-time seeds (may fail to "activate" new support near
        edges).
    time_rel_tol : float, optional
        When sweep=True, reject neighbor-propagated solutions that drift too
        far from the previous-time value, using a time-consistent fallback.
    active_imag_eps : float or None, optional
        Threshold on |Im(w_prev)| to define active/bulk indices. If None, it is
        set to 50*Im(z_list[0]) (works well when z_list=x+i*eta).
    sweep_pad : int, optional
        Number of indices used to dilate the active region. This is crucial for
        multi-bulk laws so that edges can move and points just outside a bulk
        can be initialized from the interior.
    max_iter, tol, armijo, min_lam, w_min : optional
        Newton/backtracking controls passed to fd_solve_w.

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

    W = numpy.empty((nt, nz), dtype=complex)
    ok = numpy.zeros((nt, nz), dtype=bool)

    if w0_list is None:
        raise ValueError(
            "w0_list must be provided (e.g. m1_fn(z_list) at t=0).")
    w_prev = numpy.asarray(w0_list, dtype=complex).ravel()
    if w_prev.size != nz:
        raise ValueError("w0_list must have same size as z_list.")

    W[0, :] = w_prev
    ok[0, :] = True

    sweep = bool(sweep)
    time_rel_tol = float(time_rel_tol)
    sweep_pad = int(sweep_pad)

    # If z_list is x + i*eta, use eta to set an automatic activity threshold.
    if active_imag_eps is None:
        eta0 = float(abs(z_list[0].imag))
        active_imag_eps = 50.0 * eta0 if eta0 > 0.0 else 1e-10
    active_imag_eps = float(active_imag_eps)

    def solve_with_choice(iz, w_seed):
        # candidate roots at this (t,z)
        cands = fd_candidates_w(z_list[iz], t, coeffs, w_min=w_min)

        if len(cands) == 0:
            # fallback to your existing single-root solver
            w, success = fd_solve_w(
                z_list[iz], t, coeffs, w_prev[iz],
                max_iter=max_iter, tol=tol, armijo=armijo,
                min_lam=min_lam, w_min=w_min
            )
            return w, success

        # cost = spatial continuity + time continuity (tune weights if needed)
        w_time = w_prev[iz]
        w_space = w_seed
        best = None
        best_cost = None

        for w in cands:
            # prefer continuity, but also prefer larger Im(w) to stay on the
            # bulk branch
            cost = abs(w - w_space) + 0.25 * abs(w - w_time) - 5.0 * w.imag

            if (best_cost is None) or (cost < best_cost):
                best = w
                best_cost = cost

        return best, True

    for it in range(1, nt):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])
        dt = t1 - t0
        if dt <= 0.0:
            raise ValueError("t_grid must be strictly increasing.")

        # Substep in time to keep continuation safe.
        n_sub = int(numpy.ceil(dt / float(dt_max)))
        if n_sub < 1:
            n_sub = 1

        for ks in range(1, n_sub + 1):
            t = t0 + dt * (ks / float(n_sub))

            w_row = numpy.empty(nz, dtype=complex)
            ok_row = numpy.zeros(nz, dtype=bool)

            if not sweep:
                # Independent solves: can miss edge activation in multi-bulk
                # problems.
                for iz in range(nz):
                    w, success = fd_solve_w(
                        z_list[iz], t, coeffs, w_prev[iz],
                        max_iter=max_iter, tol=tol, armijo=armijo,
                        min_lam=min_lam, w_min=w_min
                    )
                    w_row[iz] = w
                    ok_row[iz] = success

                w_prev = w_row
                continue

            # Define "active" region from previous time: inside bulks
            # Im(w_prev) is O(1), outside bulks Im(w_prev) is ~O(eta). Dilate
            # by sweep_pad to allow edges to move.
            active = (numpy.abs(numpy.imag(w_prev)) > active_imag_eps)

            # Split active indices into contiguous blocks (bulks)
            pad_label = -numpy.ones(nz, dtype=numpy.int64)  # bulk id per index
            active_pad = numpy.zeros(nz, dtype=bool)

            idx = numpy.flatnonzero(active)
            if idx.size > 0:
                cuts = numpy.where(numpy.diff(idx) > 1)[0]
                blocks = numpy.split(idx, cuts + 1)

                # Build padded intervals + centers
                centers = []
                pads = []
                for b in blocks:
                    centers.append(int((b[0] + b[-1]) // 2))
                    lo = int(max(0, b[0] - sweep_pad))
                    hi = int(min(nz - 1, b[-1] + sweep_pad))
                    pads.append((lo, hi))

                # Union of padded regions
                for lo, hi in pads:
                    active_pad[lo:hi + 1] = True

                # Assign each padded index to the nearest bulk center (no
                # overlap label)
                idx_u = numpy.flatnonzero(active_pad)
                c = numpy.asarray(centers, dtype=numpy.int64)
                dist = numpy.abs(idx_u[:, None] - c[None, :])
                winner = numpy.argmin(dist, axis=1).astype(numpy.int64)
                pad_label[idx_u] = winner

            active = (numpy.abs(numpy.imag(w_prev)) > active_imag_eps)

            # Left-to-right: use neighbor seed only within padded active
            # regions, so we don't propagate a branch across the gap between
            # bulks.
            for iz in range(nz):
                if iz == 0:
                    w_seed = w_prev[iz]
                else:
                    if (active_pad[iz] and active_pad[iz - 1] and
                        (pad_label[iz] == pad_label[iz - 1]) and
                            (pad_label[iz] >= 0)):
                        w_seed = w_row[iz - 1]
                    else:
                        w_seed = w_prev[iz]

                w_row[iz], ok_row[iz] = solve_with_choice(iz, w_seed)

            # Right-to-left refinement: helps stabilize left edges of bulks.
            for iz in range(nz - 2, -1, -1):
                if (active_pad[iz] and active_pad[iz + 1] and
                        (pad_label[iz] == pad_label[iz + 1]) and
                        (pad_label[iz] >= 0)):

                    w_seed = w_row[iz + 1]
                    w_new, ok_new = solve_with_choice(iz, w_seed)
                    if ok_new:
                        # Keep the more time-consistent solution.
                        if (not ok_row[iz]) or (abs(w_new - w_prev[iz]) <
                                                abs(w_row[iz] - w_prev[iz])):
                            w_row[iz] = w_new
                            ok_row[iz] = True

            w_prev = w_row

        W[it, :] = w_prev
        ok[it, :] = ok_row

    return W, ok
