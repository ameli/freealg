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
import scipy.optimize
from ._decompress import eval_P_partials

__all__ = ["solve_cusp"]


# ==========
# newton 3x3
# ==========

def _newton_3x3(F, x0, max_iter=60, tol=1e-12, bounds=None, max_step=None):
    x = numpy.array(x0, dtype=float)

    # bounds: list/tuple of (lo, hi) per component (None means unbounded)
    if bounds is not None:
        b = []
        for lo, hi in bounds:
            b.append((None if lo is None else float(lo),
                      None if hi is None else float(hi)))
        bounds = b

    if max_step is not None:
        max_step = numpy.asarray(max_step, dtype=float)
        if max_step.shape != (3,):
            raise ValueError("max_step must have shape (3,)")

    def _apply_bounds(xv):
        if bounds is None:
            return xv
        for i, (lo, hi) in enumerate(bounds):
            if lo is not None and xv[i] < lo:
                xv[i] = lo
            if hi is not None and xv[i] > hi:
                xv[i] = hi
        return xv

    x = _apply_bounds(x.copy())

    fx = F(x)
    if numpy.linalg.norm(fx) <= tol:
        return x, True, fx

    for _ in range(max_iter):
        J = numpy.zeros((3, 3), dtype=float)
        eps = 1e-6
        for j in range(3):
            xp = x.copy()
            xp[j] += eps
            xp = _apply_bounds(xp)
            J[:, j] = (F(xp) - fx) / eps

        try:
            dx = numpy.linalg.solve(J, -fx)
        except numpy.linalg.LinAlgError:
            return x, False, fx

        if max_step is not None:
            dx = numpy.clip(dx, -max_step, max_step)

        lam = 1.0
        improved = False
        for _ls in range(12):
            x_try = x + lam * dx
            x_try = _apply_bounds(x_try)
            f_try = F(x_try)
            if numpy.linalg.norm(f_try) < numpy.linalg.norm(fx):
                x, fx = x_try, f_try
                improved = True
                break
            lam *= 0.5

        if not improved:
            return x, False, fx

        if numpy.linalg.norm(fx) <= tol:
            return x, True, fx

    return x, False, fx


__all__ = ["solve_cusp"]


def _second_partials_fd(zeta, y, a_coeffs, eps_z=None, eps_y=None):
    zeta = float(zeta)
    y = float(y)

    if eps_z is None:
        eps_z = 1e-7 * (1.0 + abs(zeta))
    if eps_y is None:
        eps_y = 1e-7 * (1.0 + abs(y))

    _, Pz_p, Py_p = eval_P_partials(zeta + eps_z, y, a_coeffs)
    _, Pz_m, Py_m = eval_P_partials(zeta - eps_z, y, a_coeffs)
    Pzz = (Pz_p - Pz_m) / (2.0 * eps_z)
    Pzy1 = (Py_p - Py_m) / (2.0 * eps_z)

    _, Pz_p, Py_p = eval_P_partials(zeta, y + eps_y, a_coeffs)
    _, Pz_m, Py_m = eval_P_partials(zeta, y - eps_y, a_coeffs)
    Pzy2 = (Pz_p - Pz_m) / (2.0 * eps_y)
    Pyy = (Py_p - Py_m) / (2.0 * eps_y)

    Pzy = 0.5 * (Pzy1 + Pzy2)
    return float(Pzz), float(Pzy), float(Pyy)


def _cusp_F_real(zeta, y, s, a_coeffs):
    # tau = 1 + exp(s)  => c = tau-1 = exp(s) > 0
    c = float(numpy.exp(float(s)))

    P, Pz, Py = eval_P_partials(float(zeta), float(y), a_coeffs)
    P = float(numpy.real(P))
    Pz = float(numpy.real(Pz))
    Py = float(numpy.real(Py))

    F1 = P
    F2 = (y * y) * Py - c * Pz

    Pzz, Pzy, Pyy = _second_partials_fd(zeta, y, a_coeffs)
    F3 = y * (Pzz * (Py * Py) - 2.0 * Pzy * Pz * Py + Pyy * (Pz * Pz)) + \
        2.0 * (Pz * Pz) * Py

    return numpy.array([F1, F2, F3], dtype=float)


# ================
# poly coeffs in y
# ================

def _poly_coeffs_in_y(a_coeffs, zeta):
    a = numpy.asarray(a_coeffs)
    deg_z = a.shape[0] - 1
    deg_y = a.shape[1] - 1
    z_pows = numpy.power(zeta, numpy.arange(deg_z + 1, dtype=numpy.int64))
    c = numpy.empty((deg_y + 1,), dtype=numpy.complex128)
    for j in range(deg_y + 1):
        c[j] = numpy.dot(a[:, j], z_pows)
    return c  # ascending in y


# ===================
# pick realish root y
# ===================

def _pick_realish_root_y(a_coeffs, zeta):

    c_asc = _poly_coeffs_in_y(a_coeffs, zeta)
    c_desc = c_asc[::-1]  # descending for numpy.roots

    k = 0
    while k < len(c_desc) and abs(c_desc[k]) == 0:
        k += 1
    c_desc = c_desc[k:] if k < len(c_desc) else c_desc

    if len(c_desc) <= 1:
        return 0.0

    roots = numpy.roots(c_desc)
    j = int(numpy.argmin(numpy.abs(numpy.imag(roots))))
    return float(numpy.real(roots[j]))


# ==========
# solve cusp
# ==========

def solve_cusp(
        a_coeffs,
        t_init,
        zeta_init,
        y_init=None,
        max_iter=80,
        tol=1e-12,
        t_bounds=None,
        zeta_bounds=None):
    """
    Exact-derivative cusp solve for (zeta, y, t) with unknowns (zeta, y, s),
    where tau = 1 + exp(s), t = log(tau), x = zeta - (tau-1)/y.

    a_coeffs: array shape (deg_z+1, deg_y+1), P(zeta,y)=
    sum_{i,j} a[i,j]*zeta^i*y^j
    """

    a = numpy.asarray(a_coeffs, dtype=numpy.complex128)
    deg_z = a.shape[0] - 1
    deg_y = a.shape[1] - 1

    def _P_partials_all(zeta, y):
        # returns (P, Pz, Py, Pzz, Pzy, Pyy) as complex
        zeta = numpy.complex128(zeta)
        y = numpy.complex128(y)

        zi = numpy.power(zeta, numpy.arange(deg_z + 1, dtype=numpy.int64))
        yj = numpy.power(y, numpy.arange(deg_y + 1, dtype=numpy.int64))

        P = numpy.sum(a * zi[:, None] * yj[None, :])

        # Pz
        if deg_z >= 1:
            iz = numpy.arange(1, deg_z + 1, dtype=numpy.int64)
            zi_m1 = numpy.power(zeta, iz - 1)
            Pz = numpy.sum(
                    (a[iz, :] * iz[:, None]) * zi_m1[:, None] * yj[None, :])
        else:
            Pz = 0.0 + 0.0j

        # Py
        if deg_y >= 1:
            jy = numpy.arange(1, deg_y + 1, dtype=numpy.int64)
            yj_m1 = numpy.power(y, jy - 1)
            Py = numpy.sum(
                    (a[:, jy] * jy[None, :]) * zi[:, None] * yj_m1[None, :])
        else:
            Py = 0.0 + 0.0j

        # Pzz
        if deg_z >= 2:
            iz = numpy.arange(2, deg_z + 1, dtype=numpy.int64)
            zi_m2 = numpy.power(zeta, iz - 2)
            Pzz = numpy.sum((a[iz, :] * (iz * (iz - 1))[:, None]) *
                            zi_m2[:, None] * yj[None, :])
        else:
            Pzz = 0.0 + 0.0j

        # Pyy
        if deg_y >= 2:
            jy = numpy.arange(2, deg_y + 1, dtype=numpy.int64)
            yj_m2 = numpy.power(y, jy - 2)
            Pyy = numpy.sum((a[:, jy] * (jy * (jy - 1))[None, :]) *
                            zi[:, None] * yj_m2[None, :])
        else:
            Pyy = 0.0 + 0.0j

        # Pzy
        if (deg_z >= 1) and (deg_y >= 1):
            iz = numpy.arange(1, deg_z + 1, dtype=numpy.int64)
            jy = numpy.arange(1, deg_y + 1, dtype=numpy.int64)
            zi_m1 = numpy.power(zeta, iz - 1)
            yj_m1 = numpy.power(y, jy - 1)
            coeff = a[numpy.ix_(iz, jy)] * (iz[:, None] * jy[None, :])
            Pzy = numpy.sum(coeff * zi_m1[:, None] * yj_m1[None, :])
        else:
            Pzy = 0.0 + 0.0j

        return P, Pz, Py, Pzz, Pzy, Pyy

    def _F(vec):
        zeta, y, s = float(vec[0]), float(vec[1]), float(vec[2])
        c = float(numpy.exp(s))  # c = tau - 1 > 0
        P, Pz, Py, Pzz, Pzy, Pyy = _P_partials_all(zeta, y)

        # Work in reals: cusp lives on real zeta,y for real cusp
        P = float(numpy.real(P))
        Pz = float(numpy.real(Pz))
        Py = float(numpy.real(Py))
        Pzz = float(numpy.real(Pzz))
        Pzy = float(numpy.real(Pzy))
        Pyy = float(numpy.real(Pyy))

        F1 = P
        F2 = (y * y) * Py - c * Pz
        F3 = y * (Pzz * (Py * Py) - 2.0 * Pzy * Pz * Py + Pyy * (Pz * Pz)) + \
            2.0 * (Pz * Pz) * Py
        return numpy.array([F1, F2, F3], dtype=float)

    z0 = float(zeta_init)

    # seed y: keep your provided seed; else pick a real-ish root at z0
    if y_init is None:
        # build polynomial in y at fixed z0 and pick root with smallest imag
        zi = numpy.power(z0, numpy.arange(deg_z + 1, dtype=numpy.int64))
        c_asc = numpy.array([numpy.dot(a[:, j], zi) for j in range(deg_y + 1)],
                            dtype=numpy.complex128)
        c_desc = c_asc[::-1]
        kk = 0
        while kk < len(c_desc) and abs(c_desc[kk]) == 0:
            kk += 1
        c_desc = c_desc[kk:] if kk < len(c_desc) else c_desc
        roots = numpy.roots(c_desc) if len(c_desc) > 1 else numpy.array([0.0])
        j = int(numpy.argmin(numpy.abs(numpy.imag(roots))))
        y0 = float(numpy.real(roots[j]))
    else:
        y0 = float(y_init)

    tau0 = float(numpy.exp(float(t_init)))
    c0 = max(tau0 - 1.0, 1e-14)
    s0 = float(numpy.log(c0))

    # bounds for zeta, y, s
    z_lo, z_hi = -numpy.inf, numpy.inf
    if zeta_bounds is not None:
        z_lo, z_hi = float(zeta_bounds[0]), float(zeta_bounds[1])
        if z_hi < z_lo:
            z_lo, z_hi = z_hi, z_lo

    s_lo, s_hi = -numpy.inf, numpy.inf
    if t_bounds is not None:
        t_lo, t_hi = float(t_bounds[0]), float(t_bounds[1])
        if t_hi < t_lo:
            t_lo, t_hi = t_hi, t_lo
        c_lo = max(float(numpy.expm1(t_lo)), 1e-14)
        c_hi = max(float(numpy.expm1(t_hi)), 1e-14)
        s_lo, s_hi = float(numpy.log(c_lo)), float(numpy.log(c_hi))

    # keep y on the seeded sheet (this is crucial)
    y_rad = 4.0 * (1.0 + abs(y0))
    y_lo, y_hi = float(y0 - y_rad), float(y0 + y_rad)

    lb = numpy.array([z_lo, y_lo, s_lo], dtype=float)
    ub = numpy.array([z_hi, y_hi, s_hi], dtype=float)
    x0 = numpy.array([z0, y0, s0], dtype=float)
    x0 = numpy.minimum(numpy.maximum(x0, lb), ub)

    res = scipy.optimize.least_squares(
        _F,
        x0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=int(max_iter) * 100,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        x_scale="jac")

    zeta, y, s = res.x
    c = float(numpy.exp(float(s)))
    tau = 1.0 + c
    t = float(numpy.log(tau))
    x = float(zeta - (tau - 1.0) / y)

    F_final = _F(res.x)
    ok = bool(res.success and
              (numpy.max(numpy.abs(F_final)) <= max(1e-9, 50.0 * tol)))

    return {
        "ok": ok,
        "t": t,
        "tau": float(tau),
        "zeta": float(zeta),
        "y": float(y),
        "x": x,
        "F": F_final,
        "success": bool(res.success)}
