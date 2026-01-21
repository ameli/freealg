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

__all__ = ['decompress_newton_old', 'decompress_newton']


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
        if not numpy.isfinite(w.real) or not numpy.isfinite(w.imag):
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

        if abs(F) <= tol:
            return w, True

        dF = (-alpha / (w * w)) * Pz + tau * Py
        if dF == 0.0:
            return w, False

        step = -F / dF

        lam = 1.0
        F_abs = abs(F)
        ok = False

        while lam >= min_lam:
            w_new = w + lam * step
            if abs(w_new) < w_min:
                lam *= 0.5
                continue
            if want_pos_imag and (w_new.imag <= 0.0):
                lam *= 0.5
                continue

            zeta_new = z + alpha / w_new
            y_new = tau * w_new

            F_new = eval_P_partials(zeta_new, y_new, a_coeffs)[0]
            F_new = complex(F_new)

            if abs(F_new) <= (1.0 - armijo * lam) * F_abs:
                w = w_new
                ok = True
                break

            lam *= 0.5

        if not ok:
            return w, False

    F_end = eval_P_partials(z + alpha / w, tau * w, a_coeffs)[0]
    return w, (abs(F_end) <= 10.0 * tol)


# =====================
# decompress newton old
# =====================

def decompress_newton_old(z_list, t_grid, a_coeffs, w0_list=None,
                          dt_max=0.1, sweep=True, time_rel_tol=5.0,
                          max_iter=50, tol=1e-12, armijo=1e-4,
                          min_lam=1e-6, w_min=1e-14):
    """
    Evolve w = m(t,z) on a fixed z grid and time grid using FD.

    Parameters
    ----------
    z_list : array_like of complex
        Query points z (typically x + 1j*eta with eta > 0).
    t_grid : array_like of float
        Strictly increasing time grid.
    a_coeffs : ndarray
        Coefficients defining P(zeta,y) in the monomial basis used by eval_P.
    w0_list : array_like of complex
        Initial values at t_grid[0] (typically m0(z_list) on the physical
        branch).
    dt_max : float, optional
        Maximum internal time step. Larger dt is handled by substepping.
    sweep : bool, optional
        If True, use spatial continuation (neighbor seeding) plus a
        time-consistency check to prevent branch collapse. If False, solve
        each z independently from the previous-time seed (faster but may
        branch-switch for small eta).
    time_rel_tol : float, optional
        When sweep=True, if the neighbor-seeded solution differs from the
        previous-time value w_prev by more than time_rel_tol*(1+|w_prev|), we
        also solve using the previous-time seed and select the closer one.
    max_iter : int, optional
        Maximum Newton iterations in fd_solve_w.
    tol : float, optional
        Residual tolerance in fd_solve_w.
    armijo : float, optional
        Armijo parameter for backtracking in fd_solve_w.
    min_lam : float, optional
        Minimum damping factor in fd_solve_w backtracking.
    w_min : float, optional
        Minimum |w| allowed to avoid singularity.

    Returns
    -------
    W : ndarray, shape (len(t_grid), len(z_list))
        Evolved values w(t,z).
    ok : ndarray of bool, same shape as W
        Convergence flags from the final accepted solve at each point.

    Notes
    -----
    For very small eta, the implicit FD equation can have multiple roots in the
    upper half-plane. The sweep option is a branch-selection mechanism. The
    time-consistency check is critical at large t to avoid propagating a
    nearly-real spurious root across x.

    Examples
    --------
    .. code-block:: python

        x = numpy.linspace(-0.5, 2.5, 2000)
        eta = 1e-6
        z_query = x + 1j*eta
        w0_list = m1_fn(z_query)

        t_grid = numpy.linspace(0.0, 4.0, 2)
        W, ok = fd_evolve_on_grid(
            z_query, t_grid, a_coeffs, w0_list=w0_list,
            dt_max=0.1, sweep=True, time_rel_tol=5.0,
            max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6, w_min=1e-14
        )
        rho = W.imag / numpy.pi
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

    for it in range(1, nt):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])
        dt = t1 - t0
        if dt <= 0.0:
            raise ValueError("t_grid must be strictly increasing.")

        # Internal substepping makes time-continuity a strong selector.
        n_sub = int(numpy.ceil(dt / float(dt_max)))
        if n_sub < 1:
            n_sub = 1

        for ks in range(1, n_sub + 1):
            t = t0 + dt * (ks / float(n_sub))

            w_row = numpy.empty(nz, dtype=complex)
            ok_row = numpy.zeros(nz, dtype=bool)

            if not sweep:
                # Independent solves: each point uses previous-time seed only.
                for iz in range(nz):
                    w, success = fd_solve_w(
                        z_list[iz], t, a_coeffs, w_prev[iz],
                        max_iter=max_iter, tol=tol, armijo=armijo,
                        min_lam=min_lam, w_min=w_min
                    )
                    w_row[iz] = w
                    ok_row[iz] = success

                w_prev = w_row
                continue

            # Center-out sweep seed: pick where previous-time Im is largest.
            i0 = int(numpy.argmax(numpy.abs(numpy.imag(w_prev))))

            w0, ok0 = fd_solve_w(
                z_list[i0], t, a_coeffs, w_prev[i0],
                max_iter=max_iter, tol=tol, armijo=armijo,
                min_lam=min_lam, w_min=w_min
            )
            w_row[i0] = w0
            ok_row[i0] = ok0

            def solve_with_choice(iz, w_neighbor):
                # First try neighbor-seeded Newton (spatial continuity).
                w_a, ok_a = fd_solve_w(
                    z_list[iz], t, a_coeffs, w_neighbor,
                    max_iter=max_iter, tol=tol, armijo=armijo,
                    min_lam=min_lam, w_min=w_min
                )

                # Always keep a time-consistent fallback candidate.
                w_b, ok_b = fd_solve_w(
                    z_list[iz], t, a_coeffs, w_prev[iz],
                    max_iter=max_iter, tol=tol, armijo=armijo,
                    min_lam=min_lam, w_min=w_min
                )

                if ok_a and ok_b:
                    # Prefer the root closer to previous-time value (time
                    # continuation).
                    da = abs(w_a - w_prev[iz])
                    db = abs(w_b - w_prev[iz])

                    # If neighbor result is wildly off, reject it.
                    if da > time_rel_tol * (1.0 + abs(w_prev[iz])):
                        return w_b, True

                    return (w_a, True) if (da <= db) else (w_b, True)

                if ok_a:
                    # If only neighbor succeeded, still guard against extreme
                    # drift.
                    da = abs(w_a - w_prev[iz])
                    if da > time_rel_tol * (1.0 + abs(w_prev[iz])) and ok_b:
                        return w_b, True
                    return w_a, True

                if ok_b:
                    return w_b, True

                return w_a, False

            # Sweep right
            for iz in range(i0 + 1, nz):
                w_row[iz], ok_row[iz] = solve_with_choice(iz, w_row[iz - 1])

            # Sweep left
            for iz in range(i0 - 1, -1, -1):
                w_row[iz], ok_row[iz] = solve_with_choice(iz, w_row[iz + 1])

            w_prev = w_row

        W[it, :] = w_prev
        ok[it, :] = ok_row

    return W, ok


# =================
# decompress newton
# =================

def decompress_newton(z_list, t_grid, a_coeffs, w0_list=None,
                      dt_max=0.1, sweep=True, time_rel_tol=5.0,
                      active_imag_eps=None, sweep_pad=20,
                      max_iter=50, tol=1e-12, armijo=1e-4,
                      min_lam=1e-6, w_min=1e-14):
    """
    Evolve w = m(t,z) on a fixed z grid and time grid using FD.

    Parameters
    ----------
    z_list : array_like of complex
        Query points z (typically x + 1j*eta with eta > 0), ordered along x.
    t_grid : array_like of float
        Strictly increasing time grid.
    a_coeffs : ndarray
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
        # Neighbor-seeded candidate (spatial continuity)
        w_a, ok_a = fd_solve_w(
            z_list[iz], t, a_coeffs, w_seed,
            max_iter=max_iter, tol=tol, armijo=armijo,
            min_lam=min_lam, w_min=w_min
        )

        # Time-seeded candidate (time continuation)
        w_b, ok_b = fd_solve_w(
            z_list[iz], t, a_coeffs, w_prev[iz],
            max_iter=max_iter, tol=tol, armijo=armijo,
            min_lam=min_lam, w_min=w_min
        )

        if ok_a and ok_b:
            da = abs(w_a - w_prev[iz])
            db = abs(w_b - w_prev[iz])

            # Reject neighbor result if it drifted too far in one step
            if da > time_rel_tol * (1.0 + abs(w_prev[iz])):
                return w_b, True

            return (w_a, True) if (da <= db) else (w_b, True)

        if ok_a:
            da = abs(w_a - w_prev[iz])
            if da > time_rel_tol * (1.0 + abs(w_prev[iz])) and ok_b:
                return w_b, True
            return w_a, True

        if ok_b:
            return w_b, True

        return w_a, False

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
                        z_list[iz], t, a_coeffs, w_prev[iz],
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
            active_pad = active.copy()
            if sweep_pad > 0 and numpy.any(active):
                idx = numpy.flatnonzero(active)
                for i in idx:
                    lo = 0 if (i - sweep_pad) < 0 else (i - sweep_pad)
                    hi = \
                        nz if (i + sweep_pad + 1) > nz else (i + sweep_pad + 1)
                    active_pad[lo:hi] = True

            # Left-to-right: use neighbor seed only within padded active
            # regions, so we don't propagate a branch across the gap between
            # bulks.
            for iz in range(nz):
                if iz == 0:
                    w_seed = w_prev[iz]
                else:
                    if active_pad[iz] and active_pad[iz - 1]:
                        w_seed = w_row[iz - 1]
                    else:
                        w_seed = w_prev[iz]

                w_row[iz], ok_row[iz] = solve_with_choice(iz, w_seed)

            # Right-to-left refinement: helps stabilize left edges of bulks.
            for iz in range(nz - 2, -1, -1):
                if active_pad[iz] and active_pad[iz + 1]:
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
