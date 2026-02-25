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

__all__ = ['deform_newton']


# ==========
# solve wbar
# ==========

def _solve_wbar(z, tau, coeffs, c0, wbar0, max_iter=50, tol=1e-12, armijo=1e-4,
                min_lam=1e-6, w_min=1e-14, desingularize=True):
    """
    Solve for wbar = underline{m}_c(z) given tau=c/c0, using the implicit
    relation

        P(zeta(wbar), m(wbar)) = 0,

    where
        zeta = z/tau + (1/tau - 1)/wbar,
        m    = (wbar - (c0-1)/zeta) / c0.

    Parameters
    ----------

    z : complex
        Query point in C^+ (typically x + 1j*eta with eta > 0).

    tau : float
        Ratio tau = c / c0 (tau >= 1 corresponds to "decompression").

    coeffs : ndarray
        Coefficient matrix for the fitted polynomial P at t=0.

    c0 : float
        Initial aspect ratio (at t=0).

    wbar0 : complex
        Initial guess for wbar.

    Returns
    -------
    wbar : complex
        Computed solution.
    success : bool
        Convergence flag.
    """

    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    wbar = complex(wbar0)
    if abs(wbar) < w_min:
        wbar = w_min + 1j * max(w_min, z.imag)

    # Constant used in zeta(wbar)
    beta = (1.0 / float(tau)) - 1.0

    # Degree in m of P(z,m); used to cancel artificial poles after
    # substituting m=(wbar-(c0-1)/zeta)/c0 near zeta=0.
    deg_m = int(numpy.shape(coeffs)[1] - 1) if numpy.ndim(coeffs) >= 2 else 1
    if deg_m < 0:
        deg_m = 0

    for _ in range(int(max_iter)):

        if abs(wbar) < w_min:
            wbar = w_min + 1j * max(w_min, z.imag)

        zeta = (z / tau) + (beta / wbar)

        # Avoid zeta too close to 0 to prevent blow-ups in (c0-1)/zeta
        if abs(zeta) < w_min:
            zeta = (w_min + 1j * z.imag)

        m = (wbar - (c0 - 1.0) / zeta) / c0

        P, Pz, Pm = eval_P_partials(zeta, m, coeffs)

        # Residual (optionally desingularized by zeta^deg_m to cancel the
        # artificial poles introduced by the companion-to-Stieltjes map near
        # zeta=0). This regularizes Newton without changing the target root.
        if desingularize and (deg_m > 0):
            zpow = zeta ** deg_m
            F = zpow * P
        else:
            zpow = 1.0 + 0.0j
            F = P

        if abs(F) <= tol:
            return wbar, True

        # Derivatives
        dzeta_dw = -beta / (wbar * wbar)
        dm_dw = (1.0 / c0) + ((c0 - 1.0) / c0) * (dzeta_dw / (zeta * zeta))

        dP = Pz * dzeta_dw + Pm * dm_dw
        if desingularize and (deg_m > 0):
            dF = zpow * dP + (deg_m * (zeta ** (deg_m - 1)) * dzeta_dw) * P
        else:
            dF = dP

        if abs(dF) < 1e-30:
            # Degenerate Jacobian: fail early
            return wbar, False

        step = -F / dF

        # Backtracking line search (Armijo)
        lam = 1.0
        F0 = abs(F)
        success = False

        while lam >= min_lam:

            w_try = wbar + lam * step

            # Keep in upper half-plane for z in C^+ (Herglotz branch)
            if (z.imag > 0.0) and (w_try.imag <= 0.0):
                w_try = complex(w_try.real, max(w_min, abs(w_try.imag)))

            if abs(w_try) < w_min:
                w_try = w_min + 1j * max(w_min, z.imag)

            zeta_try = (z / tau) + (beta / w_try)
            if abs(zeta_try) < w_min:
                zeta_try = (w_min + 1j * z.imag)

            m_try = (w_try - (c0 - 1.0) / zeta_try) / c0
            P_try, _, _ = eval_P_partials(zeta_try, m_try, coeffs)
            if desingularize and (deg_m > 0):
                F_try = abs((zeta_try ** deg_m) * P_try)
            else:
                F_try = abs(P_try)

            # Armijo decrease
            if F_try <= (1.0 - armijo * lam) * F0:
                wbar = w_try
                success = True
                break

            lam *= 0.5

        if not success:
            return wbar, False

        if abs(step) * lam <= tol * max(1.0, abs(wbar)):
            return wbar, True

    return wbar, False


# =============
# deform newton
# =============

def deform_newton(z_list, t_grid, coeffs, c0, w0_list, dt_max=0.1, sweep=True,
                  time_rel_tol=5.0, active_imag_eps=None, sweep_pad=20,
                  max_iter=50, tol=1e-12, armijo=1e-4, min_lam=1e-6,
                  w_min=1e-14, desingularize=True, active_hysteresis=0.25):
    """
    Evolve m(t,z) under the Silverstein / aspect-ratio scaling flow.

    The fitted polynomial P(z,m)=0 is for the t=0 measure (aspect ratio c0).
    The solver evolves the companion Stieltjes transform wbar(t,z) and returns
    m(t,z) recovered from wbar(t,z).

    Parameters
    ----------

    z_list : array_like of complex
        Query points z (x + 1j*eta with eta > 0), ordered along x.

    t_grid : array_like of float
        Strictly increasing grid in t where tau=exp(t)=c/c0.
        Must start with 0.

    coeffs : ndarray
        Coefficients for P(z,m) at t=0.

    c0 : float
        Initial aspect ratio (at t=0).

    w0_list : array_like of complex
        Initial physical values m0(z_list) at t=0.

    dt_max : float, optional
        Maximum internal step in t (homotopy continuation).

    sweep : bool, optional
        Whether to enforce spatial continuity and allow edge activation.

    time_rel_tol, active_imag_eps, sweep_pad : optional
        Continuation controls (similar to decompress_newton).

    max_iter, tol, armijo, min_lam, w_min : optional
        Newton/backtracking controls.

    Returns
    -------

    W : ndarray, shape (len(t_grid), len(z_list))
        Values m(t,z) on the grid.

    ok : ndarray of bool, same shape as W
        Convergence flags.
    """

    z_list = numpy.asarray(z_list, dtype=complex).ravel()
    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    nt = t_grid.size
    nz = z_list.size

    if nt == 0:
        raise ValueError("t_grid must be non-empty.")

    if abs(t_grid[0]) > 1e-15:
        raise ValueError("t_grid must start at 0 (tau=1).")

    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")

    w0_list = numpy.asarray(w0_list, dtype=complex).ravel()
    if w0_list.size != nz:
        raise ValueError("w0_list must have the same length as z_list.")

    # Initial companion values at t=0:
    # wbar0 = c0*m0 + (c0-1)/z
    wbar_prev = c0 * w0_list + (c0 - 1.0) / z_list

    W = numpy.empty((nt, nz), dtype=complex)
    ok = numpy.zeros((nt, nz), dtype=bool)

    # Store t=0
    W[0, :] = w0_list
    ok[0, :] = True

    if active_imag_eps is None:
        # Use a threshold tied to the imaginary buffer of z.
        active_imag_eps = 50.0 * float(abs(z_list[0].imag))

    # Hysteresis memory for active-region tracking; helps prevent temporary
    # loss of thin/weak bulks during continuation.
    active_prev_mask = numpy.zeros(nz, dtype=bool)

    # --------------
    # solve at index
    # --------------

    def _solve_at_index(iz, t, w_seed, w_time):
        tau = float(numpy.exp(t))
        wbar, success = _solve_wbar(
            z_list[iz], tau, coeffs, float(c0), w_seed,
            max_iter=max_iter, tol=tol, armijo=armijo,
            min_lam=min_lam, w_min=w_min, desingularize=desingularize)

        if not success:
            return wbar, False

        # Optional time-consistency check
        if (time_rel_tol is not None) and (time_rel_tol > 0.0):
            if abs(wbar - w_time) > time_rel_tol * max(1.0, abs(w_time)):
                # fallback to time seed
                wbar2, success2 = _solve_wbar(
                    z_list[iz], tau, coeffs, float(c0), w_time,
                    max_iter=max_iter, tol=tol, armijo=armijo,
                    min_lam=min_lam, w_min=w_min)

                if success2:
                    return wbar2, True

        return wbar, True

    # -------------

    for it in range(1, nt):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])
        dt = t1 - t0

        n_sub = int(numpy.ceil(dt / float(dt_max)))
        if n_sub < 1:
            n_sub = 1

        for ks in range(1, n_sub + 1):
            t = t0 + dt * (ks / float(n_sub))
            wbar_row = numpy.empty(nz, dtype=complex)
            ok_row = numpy.zeros(nz, dtype=bool)

            if not sweep:
                for iz in range(nz):
                    wbar_row[iz], ok_row[iz] = _solve_at_index(
                        iz, t, wbar_prev[iz], wbar_prev[iz])

                wbar_prev = wbar_row
                continue

            # Active region based on Im(wbar_prev), with a small hysteresis
            # to avoid dropping a weak bulk for one step and relocking to a
            # wrong branch on the next solve.
            imag_prev = numpy.abs(numpy.imag(wbar_prev))
            active_on = imag_prev > float(active_imag_eps)
            if active_hysteresis is None:
                active_keep = active_on
            else:
                active_keep = imag_prev > float(active_hysteresis) * float(active_imag_eps)
            active = active_on | (active_prev_mask & active_keep)

            pad_label = -numpy.ones(nz, dtype=numpy.int64)
            active_pad = numpy.zeros(nz, dtype=bool)

            idx = numpy.flatnonzero(active)
            if idx.size > 0:
                cuts = numpy.where(numpy.diff(idx) > 1)[0]
                blocks = numpy.split(idx, cuts + 1)

                centers = []
                pads = []
                for b in blocks:
                    centers.append(int((b[0] + b[-1]) // 2))
                    lo = int(max(0, b[0] - int(sweep_pad)))
                    hi = int(min(nz - 1, b[-1] + int(sweep_pad)))
                    pads.append((lo, hi))

                for lo, hi in pads:
                    active_pad[lo:hi + 1] = True

                idx_u = numpy.flatnonzero(active_pad)
                c_cent = numpy.asarray(centers, dtype=numpy.int64)
                dist = numpy.abs(idx_u[:, None] - c_cent[None, :])
                winner = numpy.argmin(dist, axis=1).astype(numpy.int64)
                pad_label[idx_u] = winner

            # Update active-memory with padded support to keep neighboring
            # points alive across substeps (continuation hysteresis).
            active_prev_mask = active_pad.copy()

            # Left-to-right sweep
            for iz in range(nz):
                if iz == 0:
                    w_seed = wbar_prev[iz]
                else:
                    if (active_pad[iz] and active_pad[iz - 1] and
                            (pad_label[iz] == pad_label[iz - 1]) and
                            (pad_label[iz] >= 0)):
                        w_seed = wbar_row[iz - 1]
                    else:
                        w_seed = wbar_prev[iz]

                wbar_row[iz], ok_row[iz] = _solve_at_index(
                    iz, t, w_seed, wbar_prev[iz])

            # Right-to-left refinement
            for iz in range(nz - 2, -1, -1):
                if (active_pad[iz] and active_pad[iz + 1] and
                        (pad_label[iz] == pad_label[iz + 1]) and
                        (pad_label[iz] >= 0)):

                    w_seed = wbar_row[iz + 1]
                    w_new, ok_new = _solve_at_index(
                        iz, t, w_seed, wbar_prev[iz])

                    if ok_new:
                        if (not ok_row[iz]) or \
                            (abs(w_new - wbar_prev[iz]) <
                             abs(wbar_row[iz] - wbar_prev[iz])):
                            wbar_row[iz] = w_new
                            ok_row[iz] = True

            wbar_prev = wbar_row

        # Convert wbar_prev at time t1 to m(t1,z)
        c = float(c0) * float(numpy.exp(t1))
        W[it, :] = (wbar_prev - (c - 1.0) / z_list) / c
        ok[it, :] = ok_row

    return W, ok
