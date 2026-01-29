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

__all__ = ['decompress_newton']


# ==========================
# polynomial helpers: P, dP
# ==========================

def _poly_powers(z, deg):
    z = numpy.asarray(z, dtype=complex).ravel()
    n = z.size
    zp = numpy.ones((n, deg + 1), dtype=complex)
    for k in range(1, deg + 1):
        zp[:, k] = zp[:, k - 1] * z
    return zp


def _eval_P_dP(z, m, a_coeffs):
    """
    Evaluate P(z,m), dP/dz, dP/dm for polynomial coefficients a_coeffs.

    a_coeffs has shape (deg_z+1, s+1), where
        P(z,m) = sum_{j=0}^s a_j(z) m^j
        a_j(z) = sum_{i=0}^{deg_z} a_coeffs[i,j] z^i
    """
    z = numpy.asarray(z, dtype=complex).ravel()
    m = numpy.asarray(m, dtype=complex).ravel()

    deg_z = int(a_coeffs.shape[0] - 1)
    s = int(a_coeffs.shape[1] - 1)

    zp = _poly_powers(z, deg_z)                 # (n, deg_z+1)
    # a_j(z) for all j
    a = zp @ a_coeffs                            # (n, s+1)

    # derivative a_j'(z)
    if deg_z >= 1:
        # coeffs multiplied by power index
        idx = numpy.arange(deg_z + 1, dtype=float)
        a_coeffs_dz = a_coeffs * idx[:, None]
        # powers z^{i-1}: shift zp right
        zp_m1 = numpy.zeros_like(zp)
        zp_m1[:, 0] = 0.0
        zp_m1[:, 1:] = zp[:, :-1]
        a_dz = zp_m1 @ a_coeffs_dz              # (n, s+1)
    else:
        a_dz = numpy.zeros_like(a)

    # powers of m
    mp = numpy.ones((m.size, s + 1), dtype=complex)
    for j in range(1, s + 1):
        mp[:, j] = mp[:, j - 1] * m

    P = numpy.sum(a * mp, axis=1)

    # dP/dz = sum_j a_j'(z) m^j
    Pz = numpy.sum(a_dz * mp, axis=1)

    # dP/dm = sum_{j>=1} j a_j(z) m^{j-1}
    if s >= 1:
        jm = numpy.arange(s + 1, dtype=float)
        # m^{j-1}: shift mp left
        mp_m1 = numpy.zeros_like(mp)
        mp_m1[:, 0] = 0.0
        mp_m1[:, 1:] = mp[:, :-1]
        Pm = numpy.sum((a * jm[None, :]) * mp_m1, axis=1)
    else:
        Pm = numpy.zeros_like(P)

    return P, Pz, Pm


# ===========================
# 2x2 complex Newton (correct)
# ===========================

def _newton_corrector(z_fixed, tau, a_coeffs, zeta0, y0,
                      max_iter=50, tol=1e-12,
                      armijo=1e-4, min_lam=1e-6):
    """
    Solve for (zeta, y) in:
        F1 = P(zeta, y) = 0
        F2 = zeta - (tau-1)/y - z_fixed = 0
    using damped Newton in C^2 (2 complex unknowns).
    """
    zeta = zeta0
    y = y0

    # A tiny stabilizer for division by y
    eps_y = 0.0

    for it in range(int(max_iter)):
        P, Pz, Py = _eval_P_dP(numpy.array([zeta]), numpy.array([y]), a_coeffs)
        F1 = P[0]
        F2 = zeta - (tau - 1.0) / (y + eps_y) - z_fixed

        # Convergence test on infinity norm
        if max(abs(F1), abs(F2)) <= tol:
            return zeta, y, True, it

        # Jacobian (complex 2x2)
        # dF1/dzeta = Pz, dF1/dy = Py
        # dF2/dzeta = 1, dF2/dy = (tau-1)/y^2
        J11 = Pz[0]
        J12 = Py[0]
        J21 = 1.0 + 0.0j
        J22 = (tau - 1.0) / ((y + eps_y) * (y + eps_y))

        # Solve J * delta = -F
        try:
            delta = numpy.linalg.solve(
                numpy.array([[J11, J12], [J21, J22]], dtype=complex),
                numpy.array([-F1, -F2], dtype=complex)
            )
        except numpy.linalg.LinAlgError:
            return zeta, y, False, it

        dzeta = delta[0]
        dy = delta[1]

        # Damped update (Armijo on ||F||_inf)
        lam = 1.0
        norm0 = max(abs(F1), abs(F2))

        while lam >= float(min_lam):
            zeta_try = zeta + lam * dzeta
            y_try = y + lam * dy

            P_try, _, _ = _eval_P_dP(numpy.array([zeta_try]), numpy.array([y_try]), a_coeffs)
            F1_try = P_try[0]
            F2_try = zeta_try - (tau - 1.0) / (y_try + eps_y) - z_fixed
            norm_try = max(abs(F1_try), abs(F2_try))

            # Armijo-like sufficient decrease
            if norm_try <= (1.0 - float(armijo) * lam) * norm0 or norm_try < norm0:
                zeta = zeta_try
                y = y_try
                break

            lam *= 0.5

        else:
            # failed to find a decreasing step
            return zeta, y, False, it

    return zeta, y, False, int(max_iter)


# ===========================
# predictor step (tangent ODE)
# ===========================

def _predictor_step(z_fixed, tau0, tau1, a_coeffs, zeta0, y0):
    """
    One explicit Euler predictor from tau0 to tau1 along the sheet.

    Uses:
      dy/dzeta = -Pz/Py from P(zeta,y)=0
      zeta' = (1/y) / ( 1 + (tau-1)*(dy/dzeta)/y^2 )
    """
    dtau = float(tau1 - tau0)
    if dtau == 0.0:
        return zeta0, y0

    P, Pz, Py = _eval_P_dP(numpy.array([zeta0]), numpy.array([y0]), a_coeffs)
    Pz = Pz[0]
    Py = Py[0]

    # dy/dzeta on curve (avoid division by zero)
    if Py == 0.0:
        D = 0.0 + 0.0j
    else:
        D = -Pz / Py

    den = 1.0 + (tau0 - 1.0) * D / (y0 * y0)
    if den == 0.0:
        den = 1.0 + 0.0j

    zeta_prime = (1.0 / y0) / den
    y_prime = D * zeta_prime

    zeta1 = zeta0 + dtau * zeta_prime
    y1 = y0 + dtau * y_prime

    return zeta1, y1


# =================
# decompress newton
# =================

def decompress_newton(z_query, t, a_coeffs, w0_list=None,
                     max_iter=50, tol=1e-12,
                     armijo=1e-4, min_lam=1e-6,
                     w_min=1e-14,
                     sweep=True,
                     max_substeps=8,
                     **kwargs):
    """
    Free decompression via characteristic continuation on the algebraic curve
    P(z,m)=0 using predictor-corrector in tau = exp(t).

    Parameters
    ----------
    z_query : array_like (n_z,)
        Query points z = x + i*delta (typically slightly above real axis).
    t : array_like (n_t,)
        Time grid (must be sorted increasing; should include 0).
    a_coeffs : ndarray
        Polynomial coefficients of P(z,m)=0 as in AlgebraicForm.fit.
    w0_list : array_like (n_z,)
        Initial condition m(t=0, z_query) on the physical branch.
    max_iter, tol, armijo, min_lam : float/int
        Newton corrector parameters.
    w_min : float
        Lower bound on |w| used to guard divisions (rare).
    sweep : bool
        If True, advance in time using previous solution as initial guess.
    max_substeps : int
        If corrector fails on a time step, internally bisect in tau and try
        smaller predictor steps (without changing returned t grid).

    Returns
    -------
    W : ndarray (n_t, n_z), complex
        m(t,z) for all t and z_query.
    ok : ndarray (n_t, n_z), bool
        Per-point success flag.
    """
    z_query = numpy.asarray(z_query, dtype=complex).ravel()
    t = numpy.asarray(t, dtype=float).ravel()

    if t.size == 0:
        return numpy.empty((0, z_query.size), dtype=complex), \
            numpy.empty((0, z_query.size), dtype=bool)

    # Ensure sorted increasing
    if numpy.any(numpy.diff(t) < 0.0):
        raise ValueError('t must be sorted increasing.')

    tau = numpy.exp(t)

    n_t = t.size
    n_z = z_query.size

    W = numpy.full((n_t, n_z), numpy.nan + 1j * numpy.nan, dtype=complex)
    ok = numpy.zeros((n_t, n_z), dtype=bool)

    if w0_list is None:
        raise ValueError('w0_list must be provided (physical branch at t=0).')
    w0_list = numpy.asarray(w0_list, dtype=complex).ravel()
    if w0_list.size != n_z:
        raise ValueError('w0_list must have the same size as z_query.')

    # Initialize at tau=1 (t=0)
    # We do not assume t[0] == 0, but if not, we still use w0_list as seed.
    for j in range(n_z):
        W[0, j] = w0_list[j]
        ok[0, j] = numpy.isfinite(W[0, j])

    # Per-point continuation state in (zeta, y=tau*w)
    zeta_state = z_query.copy()
    y_state = (tau[0] * w0_list).copy()

    # Ensure no tiny y to avoid division in F2
    tiny = numpy.abs(y_state) < float(w_min)
    if numpy.any(tiny):
        y_state[tiny] = y_state[tiny] + (float(w_min) + 0.0j)

    # Time sweep
    for k in range(1, n_t):
        tau_prev = float(tau[k - 1])
        tau_next = float(tau[k])

        # Allow internal substepping if needed
        for j in range(n_z):
            if not sweep and k > 1:
                # If sweep=False, restart from t=0 each time
                zeta0 = z_query[j]
                y0 = (tau_prev * w0_list[j])
            else:
                zeta0 = zeta_state[j]
                y0 = y_state[j]

            # If previous was not ok, fall back to trivial seed
            if not numpy.isfinite(zeta0) or not numpy.isfinite(y0):
                zeta0 = z_query[j]
                y0 = tau_prev * w0_list[j]

            # Internal bisection in tau (predictor-corrector)
            success = False
            zeta_sol = zeta0
            y_sol = y0

            # Create a queue of tau intervals to try (depth-first)
            intervals = [(tau_prev, tau_next, zeta0, y0, 0)]
            while intervals:
                a_tau, b_tau, a_zeta, a_y, depth = intervals.pop()

                # Predictor from a_tau to b_tau
                zeta_pred, y_pred = _predictor_step(
                    z_query[j], a_tau, b_tau, a_coeffs, a_zeta, a_y
                )

                # Guard tiny y_pred
                if abs(y_pred) < float(w_min):
                    y_pred = y_pred + (float(w_min) + 0.0j)

                # Corrector at b_tau
                zeta_corr, y_corr, ok1, _ = _newton_corrector(
                    z_fixed=z_query[j], tau=b_tau, a_coeffs=a_coeffs,
                    zeta0=zeta_pred, y0=y_pred,
                    max_iter=max_iter, tol=tol, armijo=armijo, min_lam=min_lam
                )

                if ok1 and numpy.isfinite(zeta_corr) and numpy.isfinite(y_corr):
                    # Accept this segment; if it ends at tau_next, done
                    if b_tau == tau_next:
                        success = True
                        zeta_sol = zeta_corr
                        y_sol = y_corr
                        break
                    else:
                        # Continue from this intermediate point to tau_next
                        intervals.append((b_tau, tau_next, zeta_corr, y_corr, depth))
                        continue

                # If failed and we can subdivide further, split interval
                if depth < int(max_substeps):
                    mid = 0.5 * (a_tau + b_tau)
                    # Subdivide: first do [a, mid] then [mid, b]
                    intervals.append((mid, b_tau, a_zeta, a_y, depth + 1))
                    intervals.append((a_tau, mid, a_zeta, a_y, depth + 1))
                # else give up

            if success:
                w = y_sol / tau_next
                W[k, j] = w
                ok[k, j] = True

                # update state
                zeta_state[j] = zeta_sol
                y_state[j] = y_sol
            else:
                # keep previous (or NaN) and mark fail
                W[k, j] = numpy.nan + 1j * numpy.nan
                ok[k, j] = False
                # do not update state; but keep last good if any

    return W, ok
