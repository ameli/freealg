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
import scipy.optimize as opt


# ================
# poly coeffs in y
# ================

def _poly_coeffs_in_y(a_coeffs, zeta):
    """
    Build coefficients c_j(zeta) so that P(zeta, y) = sum_j c_j(zeta) y^j.

    Assumes a_coeffs[i, j] multiplies z^i y^j (same layout as eval_P in
    _continuation_algebraic). Returns coefficients in ascending powers of y.
    """

    a = numpy.asarray(a_coeffs)
    deg_z = a.shape[0] - 1
    deg_y = a.shape[1] - 1

    # c_j(zeta) = sum_i a[i,j] zeta^i
    z_pows = numpy.power(zeta, numpy.arange(deg_z + 1, dtype=numpy.int64))
    c = numpy.empty((deg_y + 1,), dtype=numpy.complex128)
    for j in range(deg_y + 1):
        c[j] = numpy.dot(a[:, j], z_pows)

    return c


# ===================
# pick realish root y
# ===================

def _pick_realish_root_y(a_coeffs, zeta):
    """
    Pick a reasonable real-ish root y of P(zeta, y)=0 to seed Newton.

    Returns a float (real part of the selected root).
    """

    c_asc = _poly_coeffs_in_y(a_coeffs, zeta)  # ascending in y
    # numpy.roots wants descending order
    c_desc = c_asc[::-1]
    # strip leading ~0 coefficients
    k = 0
    while k < len(c_desc) and abs(c_desc[k]) == 0:
        k += 1
    c_desc = c_desc[k:] if k < len(c_desc) else c_desc

    if len(c_desc) <= 1:
        return 0.0

    roots = numpy.roots(c_desc)
    # choose the root closest to the real axis
    j = int(numpy.argmin(numpy.abs(numpy.imag(roots))))
    return float(numpy.real(roots[j]))


# =========
# cusp wrap
# =========

def cusp_wrap(self, t_grid, edge_kwargs=None, max_iter=80, tol=1e-12,
              verbose=False):

    if edge_kwargs is None:
        edge_kwargs = {}

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()

    # allow scalar / len-1 input
    if t_grid.size == 1:
        t0 = float(t_grid[0])
        dt = 0.25
        t_grid = numpy.linspace(max(0.0, t0 - dt), t0 + dt, 21)

    if t_grid.size < 5:
        raise ValueError("t_grid too small")

    def gap_at(tt):
        ce, _, _ = self.edge(numpy.array([float(tt)]), verbose=False,
                             **edge_kwargs)
        return float(ce[0, 2].real - ce[0, 1].real)

    # coarse grid gap
    ce, _, _ = self.edge(t_grid, verbose=False, **edge_kwargs)
    gap = ce[:, 2].real - ce[:, 1].real
    m = numpy.isfinite(gap)

    if numpy.count_nonzero(m) < 2:
        return {"success": False, "reason": "gap is not finite on grid"}

    tg = t_grid[m]
    gg = gap[m]

    # candidate bracket indices from coarse grid
    s = numpy.sign(gg)
    idx = numpy.where(s[:-1] * s[1:] < 0)[0]

    bracketed = False
    t_star = None

    # robust: verify sign change using the true gap_at before calling brentq
    if idx.size > 0:
        for ii in idx[:5]:  # try a few brackets
            tL, tR = float(tg[ii]), float(tg[ii + 1])
            gL = gap_at(tL)
            gR = gap_at(tR)
            if numpy.isfinite(gL) and numpy.isfinite(gR) and (gL * gR < 0.0):
                t_star = float(opt.brentq(gap_at, tL, tR, xtol=1e-12,
                                          rtol=1e-12, maxiter=200))
                bracketed = True
                break

    # fallback: minimizer of |gap| on the coarse grid
    if t_star is None:
        i0 = int(numpy.argmin(numpy.abs(gg)))
        t_star = float(tg[i0])
        bracketed = False

    # --- seed (zeta,y) correctly using zeta = x + (tau-1)/y ---
    ce_star, _, _ = self.edge(numpy.array([t_star]), verbose=False,
                              **edge_kwargs)
    x_seed = float(ce_star[0, 1].real)  # inner edge b1
    tau = float(numpy.exp(t_star))
    c = tau - 1.0

    a = numpy.asarray(self.a_coeffs, dtype=numpy.complex128)
    deg_z = a.shape[0] - 1
    deg_y = a.shape[1] - 1

    def poly_in_y(zeta):
        zi = numpy.power(zeta, numpy.arange(deg_z + 1, dtype=numpy.int64))
        c_asc = numpy.array([numpy.dot(a[:, j], zi) for j in range(deg_y + 1)],
                            dtype=numpy.complex128)
        return c_asc

    zeta0 = float(x_seed)
    c_asc = poly_in_y(zeta0)
    roots = numpy.roots(c_asc[::-1])
    j = int(numpy.argmin(numpy.abs(numpy.imag(roots))))
    y0 = float(numpy.real(roots[j]))
    if abs(y0) < 1e-12:
        jj = numpy.argsort(numpy.abs(numpy.imag(roots)))
        for k in jj:
            if abs(numpy.real(roots[k])) > 1e-8:
                y0 = float(numpy.real(roots[k]))
                break

    zeta_seed = float(x_seed + c / y0)
    y_seed = float(y0)

    def P_all(zeta, y):
        zeta = numpy.complex128(zeta)
        y = numpy.complex128(y)
        zi = numpy.power(zeta, numpy.arange(deg_z + 1))
        yj = numpy.power(y, numpy.arange(deg_y + 1))
        P = numpy.sum(a * zi[:, None] * yj[None, :])

        if deg_z >= 1:
            iz = numpy.arange(1, deg_z + 1)
            Pz = numpy.sum((a[iz, :] * iz[:, None]) *
                           numpy.power(zeta, iz - 1)[:, None] * yj[None, :])
        else:
            Pz = 0.0 + 0.0j

        if deg_y >= 1:
            jy = numpy.arange(1, deg_y + 1)
            Py = numpy.sum((a[:, jy] * jy[None, :]) * zi[:, None] *
                           numpy.power(y, jy - 1)[None, :])
        else:
            Py = 0.0 + 0.0j

        if deg_z >= 2:
            iz = numpy.arange(2, deg_z + 1)
            Pzz = numpy.sum((a[iz, :] * (iz * (iz - 1))[:, None]) *
                            numpy.power(zeta, iz - 2)[:, None] * yj[None, :])
        else:
            Pzz = 0.0 + 0.0j

        if deg_y >= 2:
            jy = numpy.arange(2, deg_y + 1)
            Pyy = numpy.sum((a[:, jy] * (jy * (jy - 1))[None, :]) *
                            zi[:, None] * numpy.power(y, jy - 2)[None, :])
        else:
            Pyy = 0.0 + 0.0j

        if (deg_z >= 1) and (deg_y >= 1):
            iz = numpy.arange(1, deg_z + 1)
            jy = numpy.arange(1, deg_y + 1)
            coeff = a[numpy.ix_(iz, jy)] * (iz[:, None] * jy[None, :])
            Pzy = numpy.sum(coeff * numpy.power(zeta, iz - 1)[:, None] *
                            numpy.power(y, jy - 1)[None, :])
        else:
            Pzy = 0.0 + 0.0j

        return P, Pz, Py, Pzz, Pzy, Pyy

    def G(v):
        zeta, y = float(v[0]), float(v[1])
        P, Pz, Py, _, _, _ = P_all(zeta, y)
        P = float(numpy.real(P))
        Pz = float(numpy.real(Pz))
        Py = float(numpy.real(Py))
        F2 = (y * y) * Py - c * Pz
        return numpy.array([P, F2], dtype=float)

    z_rad = 0.5
    y_rad = 5.0 * (1.0 + abs(y_seed))
    lb = numpy.array([zeta_seed - z_rad, y_seed - y_rad], dtype=float)
    ub = numpy.array([zeta_seed + z_rad, y_seed + y_rad], dtype=float)

    res = opt.least_squares(
        G, numpy.array([zeta_seed, y_seed], dtype=float),
        bounds=(lb, ub), method="trf",
        max_nfev=8000, ftol=tol, xtol=tol, gtol=tol, x_scale="jac"
    )

    zeta_star = float(res.x[0])
    y_star = float(res.x[1])
    x_star = float(zeta_star - c / y_star)

    P, Pz, Py, Pzz, Pzy, Pyy = P_all(zeta_star, y_star)
    P = float(numpy.real(P))
    Pz = float(numpy.real(Pz))
    Py = float(numpy.real(Py))
    Pzz = float(numpy.real(Pzz))
    Pzy = float(numpy.real(Pzy))
    Pyy = float(numpy.real(Pyy))

    F2 = (y_star * y_star) * Py - c * Pz
    F3 = y_star * (Pzz * (Py * Py) - 2.0 * Pzy * Pz * Py + Pyy * (Pz * Pz)) + \
        2.0 * (Pz * Pz) * Py
    F = numpy.array([P, float(F2), float(F3)], dtype=float)

    ok = bool(numpy.max(numpy.abs(F)) < 1e-8)

    return {
        "ok": ok,
        "t": float(t_star),
        "tau": float(tau),
        "zeta": float(zeta_star),
        "y": float(y_star),
        "x": float(x_star),
        "F": F,
        "success": True,
        "seed": {
            "t": float(t_star),
            "x": float(x_seed),
            "zeta": float(zeta_seed),
            "y": float(y_seed)
        },
        "merge": {"bracketed": bool(bracketed)},
        "gap_at_t": float(gap_at(t_star)),
        "lsq_success": bool(res.success)}
