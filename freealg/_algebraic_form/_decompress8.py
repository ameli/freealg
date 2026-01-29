# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

"""
Robust Newton solver for Free Decompression (FD) using the characteristic
variables (zeta, y) on the spectral curve P(zeta, y)=0.

This implementation avoids solving directly in w (where zeta = z + alpha / w
introduces a pole at w=0). Instead, for each query z and time t (tau=e^t), we
solve the 2x2 complex system:

    F1(zeta, y) := P(zeta, y) = 0
    F2(zeta, y) := zeta - (tau - 1)/y - z = 0

Then m(t,z) = w = y/tau.

The public API matches the existing decompress_newton used by AlgebraicForm.
"""

import numpy

__all__ = ['decompress_newton']


# =====================
# Polynomial evaluation
# =====================

def _powers(x, deg):
    """
    Returns [1, x, x^2, ..., x^deg] for each element of x.
    """
    x = numpy.asarray(x, dtype=numpy.complex128)
    xp = numpy.ones((x.size, deg + 1), dtype=numpy.complex128)
    for k in range(1, deg + 1):
        xp[:, k] = xp[:, k - 1] * x
    return xp


def _poly_coef_in_y(zeta, a_coeffs):
    """
    For each zeta, compute coefficients a_j(zeta) so that
        P(zeta, y) = sum_{j=0}^s a_j(zeta) y^j
    """
    zeta = numpy.asarray(zeta, dtype=numpy.complex128).ravel()
    deg_z = int(a_coeffs.shape[0] - 1)
    s = int(a_coeffs.shape[1] - 1)

    zp = _powers(zeta, deg_z)
    a = numpy.empty((zeta.size, s + 1), dtype=numpy.complex128)
    for j in range(s + 1):
        a[:, j] = zp @ a_coeffs[:, j]
    return a


def _poly_coef_in_y_dzeta(zeta, a_coeffs):
    """
    For each zeta, compute coefficients da_j/dzeta(zeta) so that
        d/dzeta P(zeta, y) = sum_{j=0}^s (da_j/dzeta)(zeta) y^j
    """
    zeta = numpy.asarray(zeta, dtype=numpy.complex128).ravel()
    deg_z = int(a_coeffs.shape[0] - 1)
    s = int(a_coeffs.shape[1] - 1)

    if deg_z <= 0:
        return numpy.zeros((zeta.size, s + 1), dtype=numpy.complex128)

    # derivative powers: d/dzeta zeta^i = i zeta^(i-1)
    zp = _powers(zeta, deg_z - 1)  # up to zeta^(deg_z-1)
    da = numpy.empty((zeta.size, s + 1), dtype=numpy.complex128)
    for j in range(s + 1):
        col = a_coeffs[:, j]
        # sum_{i=1..deg_z} i*c_{i,j} zeta^(i-1)
        # build weighted coefficients for zp @ ...
        w = numpy.arange(deg_z + 1, dtype=numpy.complex128) * col
        da[:, j] = zp @ w[1:]
    return da


def _eval_P_and_partials(zeta, y, a_coeffs):
    """
    Evaluate P(zeta,y), P_zeta(zeta,y), P_y(zeta,y).

    Returns:
        P, Pz, Py  (arrays of shape (n,))
    """
    zeta = numpy.asarray(zeta, dtype=numpy.complex128).ravel()
    y = numpy.asarray(y, dtype=numpy.complex128).ravel()

    a = _poly_coef_in_y(zeta, a_coeffs)          # (n, s+1)
    da = _poly_coef_in_y_dzeta(zeta, a_coeffs)   # (n, s+1)

    s = int(a.shape[1] - 1)
    # powers of y up to s
    yp = _powers(y, s)  # (n, s+1)

    P = numpy.sum(a * yp, axis=1)

    # P_zeta = sum_j da_j(zeta) y^j
    Pz = numpy.sum(da * yp, axis=1)

    # P_y = sum_{j>=1} j*a_j(zeta) y^{j-1}
    if s == 0:
        Py = numpy.zeros_like(P)
    else:
        Py = numpy.zeros_like(P)
        # yp[:, j-1] available
        for j in range(1, s + 1):
            Py += (j * a[:, j]) * yp[:, j - 1]

    return P, Pz, Py


# =======================
# 2x2 complex Newton step
# =======================

def _newton_2x2(z, tau, zeta0, y0, a_coeffs, max_iter, tol,
                armijo, min_lam, w_min, enforce_imag=True):
    """
    Solve for (zeta,y) at given (z,tau) using damped Newton on the 2x2 complex
    system.

    Returns:
        (zeta, y, ok, iters)
    """
    zeta = numpy.complex128(zeta0)
    y = numpy.complex128(y0)

    # helper to compute residual norm
    def F(zeta_, y_):
        P, Pz, Py = _eval_P_and_partials(numpy.array([zeta_]),
                                         numpy.array([y_]),
                                         a_coeffs)
        P = P[0]
        Pz = Pz[0]
        Py = Py[0]
        F1 = P
        F2 = (zeta_ - (tau - 1.0) / y_ - z)
        # Jacobian entries
        J11 = Pz
        J12 = Py
        J21 = 1.0 + 0.0j
        J22 = (tau - 1.0) / (y_ * y_)  # d/dy of (-(tau-1)/y) is +(tau-1)/y^2
        return F1, F2, J11, J12, J21, J22

    # initial residual
    F1, F2, J11, J12, J21, J22 = F(zeta, y)
    r0 = max(abs(F1), abs(F2))

    if not numpy.isfinite(r0):
        return zeta, y, False, 0

    for it in range(int(max_iter)):
        r = max(abs(F1), abs(F2))
        if r <= tol:
            w = y / tau
            if (abs(w) < w_min) or (not numpy.isfinite(w.real)) or (not numpy.isfinite(w.imag)):
                return zeta, y, False, it
            if enforce_imag and (z.imag > 0.0) and (w.imag <= 0.0):
                return zeta, y, False, it
            return zeta, y, True, it

        # Solve 2x2 complex linear system J * d = -F
        det = J11 * J22 - J12 * J21
        if det == 0 or (not numpy.isfinite(det.real)) or (not numpy.isfinite(det.imag)):
            return zeta, y, False, it

        d_zeta = (-F1 * J22 - (-F2) * J12) / det
        d_y = (J11 * (-F2) - J21 * (-F1)) / det

        # Armijo damping on residual norm
        lam = 1.0
        if armijo is None or armijo <= 0.0:
            lam = 1.0
        else:
            c = float(armijo)
            r_curr = r
            # Try steps until sufficient decrease or min_lam
            while True:
                zeta_try = zeta + lam * d_zeta
                y_try = y + lam * d_y
                F1t, F2t, J11t, J12t, J21t, J22t = F(zeta_try, y_try)
                r_try = max(abs(F1t), abs(F2t))
                if numpy.isfinite(r_try) and (r_try <= (1.0 - c * lam) * r_curr):
                    # accept
                    zeta, y = zeta_try, y_try
                    F1, F2, J11, J12, J21, J22 = F1t, F2t, J11t, J12t, J21t, J22t
                    break
                lam *= 0.5
                if lam < float(min_lam):
                    # accept last trial if it improves, else fail
                    if numpy.isfinite(r_try) and (r_try < r_curr):
                        zeta, y = zeta_try, y_try
                        F1, F2, J11, J12, J21, J22 = F1t, F2t, J11t, J12t, J21t, J22t
                        break
                    return zeta, y, False, it

        # continue loop

    # max_iter exceeded
    return zeta, y, False, int(max_iter)


# =================
# Public entrypoint
# =================

def decompress_newton(z_query, t_all, a_coeffs, w0_list=None,
                     max_iter=50, tol=1e-12,
                     armijo=1e-4, min_lam=1e-6, w_min=1e-14,
                     sweep=True, verbose=False, **kwargs):
    """
    Parameters
    ----------
    z_query : array_like (complex)
        Query points z where m(t,z) should be evaluated (typically x + i*delta)

    t_all : array_like (float)
        Time grid including t=0, increasing.

    a_coeffs : ndarray
        Coefficient matrix for P(z,m) in monomial basis (deg_z+1, s+1)

    w0_list : array_like (complex), optional
        Initial condition m(0, z_query) on the physical branch. If None, this
        function will approximate it as -1/z_query (may be poor near cuts).

    Other parameters mirror the existing solver interface.

    Returns
    -------
    W : ndarray (n_t, n_z) complex
        Estimated m(t,z) on the tracked branch.

    ok : ndarray (n_t, n_z) bool
        Convergence flag for each point.
    """

    z_query = numpy.asarray(z_query, dtype=numpy.complex128).ravel()
    t_all = numpy.asarray(t_all, dtype=float).ravel()

    if t_all.size == 0:
        raise ValueError('t_all is empty.')

    # enforce sorted
    if numpy.any(numpy.diff(t_all) < 0):
        raise ValueError('t_all must be sorted increasing.')

    n_z = z_query.size
    n_t = t_all.size


    # If the caller does not include t=0 as the first element, prepend it.
    # The Newton march below assumes W[0] is the known initial condition at t=0
    # (physical branch), and evolves forward for k=1..n_t-1.
    drop_t0 = False
    if n_t == 0:
        raise ValueError('t_all must be non-empty.')
    if abs(float(t_all[0])) > 0.0:
        t_all = numpy.concatenate(([0.0], t_all))
        n_t = t_all.size
        drop_t0 = True
    if w0_list is None:
        w0 = -1.0 / z_query
    else:
        w0 = numpy.asarray(w0_list, dtype=numpy.complex128).ravel()
        if w0.size != n_z:
            raise ValueError('w0_list must have same length as z_query.')

    # Output arrays
    W = numpy.empty((n_t, n_z), dtype=numpy.complex128)
    ok = numpy.zeros((n_t, n_z), dtype=bool)

    # Initialize at t=0
    W[0, :] = w0
    ok[0, :] = numpy.isfinite(w0.real) & numpy.isfinite(w0.imag)

    # For each time step, solve independently per z with continuation seeds
    for it in range(1, n_t):
        t_prev = float(t_all[it - 1])
        t = float(t_all[it])
        tau_prev = numpy.exp(t_prev)
        tau = numpy.exp(t)

        # seeds from previous time
        w_seed = W[it - 1, :].copy()
        # y seed from previous: y = tau_prev * w
        y_seed = tau_prev * w_seed

        # Optional sweep: use previous x point at the same time as init
        zeta_seed = numpy.empty(n_z, dtype=numpy.complex128)

        # Initialize zeta so that F2 is satisfied initially (good conditioning)
        # zeta = z + (tau-1)/y
        # Guard y=0
        y_safe = y_seed.copy()
        tiny = numpy.abs(y_safe) < 1e-300
        if numpy.any(tiny):
            y_safe[tiny] = (1e-300 + 0.0j)
        zeta_seed[:] = z_query + (tau - 1.0) / y_safe

        # Sweep order: left-to-right
        if sweep:
            order = range(n_z)
        else:
            order = range(n_z)

        # Storage for this time
        w_out = numpy.empty(n_z, dtype=numpy.complex128)
        ok_out = numpy.zeros(n_z, dtype=bool)

        prev_good_idx = None
        for j in order:
            z = z_query[j]
            # choose initial guess
            y0 = y_seed[j]
            zeta0 = zeta_seed[j]

            if sweep and (prev_good_idx is not None):
                # use last successful (zeta,y) as initial, but adjust zeta to
                # satisfy constraint using current z and that y
                y0 = y_last
                y_safe0 = y0 if abs(y0) > 1e-300 else (1e-300 + 0.0j)
                zeta0 = z + (tau - 1.0) / y_safe0

            # Solve
            zeta, y, okj, _ = _newton_2x2(
                z, tau, zeta0, y0, a_coeffs,
                max_iter=max_iter, tol=tol,
                armijo=armijo, min_lam=min_lam, w_min=w_min,
                enforce_imag=True
            )

            if not okj:
                # Fallback 1: asymptotic Stieltjes seed
                w_asym = -1.0 / z
                y0b = tau * w_asym
                y_safe0b = y0b if abs(y0b) > 1e-300 else (1e-300 + 0.0j)
                zeta0b = z + (tau - 1.0) / y_safe0b

                zeta, y, okj, _ = _newton_2x2(
                    z, tau, zeta0b, y0b, a_coeffs,
                    max_iter=max_iter, tol=tol,
                    armijo=armijo, min_lam=min_lam, w_min=w_min,
                    enforce_imag=True
                )

            wj = y / tau
            w_out[j] = wj
            ok_out[j] = bool(okj)

            if okj:
                prev_good_idx = j
                y_last = y

        W[it, :] = w_out
        ok[it, :] = ok_out

        if verbose:
            print(f'[t={t:0.6f}] success={ok_out.mean():0.3f}')
    if drop_t0:
        return W[1:, :], ok[1:, :]
    return W, ok
