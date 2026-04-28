# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

"""
Cusp solver for free deformation.

This is the deformation counterpart of ``_cusp.py``.  It keeps the same
least-squares/Newton-style refinement structure, but replaces the free
-decompression projection geometry by the free-deformation projection

    x = tau*zeta + (tau - 1) / wbar,
    wbar = c0*y + (c0 - 1)/zeta,

on the initial spectral curve P(zeta, y)=0.
"""

# =======
# Imports
# =======

import numpy
import scipy.optimize

__all__ = ["solve_cusp"]


# ===================
# eval P all partials
# ===================

def _eval_P_all_partials(zeta, y, coeffs):
    """
    Evaluate P and first/second partial derivatives.
    """

    a = numpy.asarray(coeffs, dtype=numpy.complex128)
    deg_z = a.shape[0] - 1
    deg_y = a.shape[1] - 1

    zeta = numpy.complex128(zeta)
    y = numpy.complex128(y)

    zi = numpy.power(zeta, numpy.arange(deg_z + 1, dtype=numpy.int64))
    yj = numpy.power(y, numpy.arange(deg_y + 1, dtype=numpy.int64))

    P = numpy.sum(a * zi[:, None] * yj[None, :])

    if deg_z >= 1:
        iz = numpy.arange(1, deg_z + 1, dtype=numpy.int64)
        zi_m1 = numpy.power(zeta, iz - 1)
        Pz = numpy.sum((a[iz, :] * iz[:, None]) *
                       zi_m1[:, None] * yj[None, :])
    else:
        Pz = 0.0 + 0.0j

    if deg_y >= 1:
        jy = numpy.arange(1, deg_y + 1, dtype=numpy.int64)
        yj_m1 = numpy.power(y, jy - 1)
        Py = numpy.sum((a[:, jy] * jy[None, :]) *
                       zi[:, None] * yj_m1[None, :])
    else:
        Py = 0.0 + 0.0j

    if deg_z >= 2:
        iz = numpy.arange(2, deg_z + 1, dtype=numpy.int64)
        zi_m2 = numpy.power(zeta, iz - 2)
        Pzz = numpy.sum((a[iz, :] * (iz * (iz - 1))[:, None]) *
                        zi_m2[:, None] * yj[None, :])
    else:
        Pzz = 0.0 + 0.0j

    if deg_y >= 2:
        jy = numpy.arange(2, deg_y + 1, dtype=numpy.int64)
        yj_m2 = numpy.power(y, jy - 2)
        Pyy = numpy.sum((a[:, jy] * (jy * (jy - 1))[None, :]) *
                        zi[:, None] * yj_m2[None, :])
    else:
        Pyy = 0.0 + 0.0j

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


# ======================
# deformation pushforward
# ======================

def _deform_pushforward(tau, zeta, y, c0, w_min=1e-14):
    """
    Physical coordinate of a point on the initial curve under deformation.
    """

    zeta = complex(zeta)
    y = complex(y)
    tau = float(tau)
    c0 = float(c0)

    A = c0 * y * zeta + (c0 - 1.0)
    if abs(A) < float(w_min):
        if A == 0.0:
            A = complex(float(w_min), 0.0)
        else:
            A = A + float(w_min) * A / abs(A)
    return tau * zeta + (tau - 1.0) * zeta / A


# =========================
# deformation cusp residual
# =========================

def _deform_edge_and_cusp_terms(zeta, y, tau, c0, coeffs):
    """
    Return P, edge residual C, and cusp residual dC along the curve.

    The cleared deformation edge critical equation is

        C = tau*A**2*Py
            + (tau - 1)*((c0 - 1)*Py + c0*zeta**2*Pz),

    where A = c0*y*zeta + c0 - 1. The higher-order critical condition is

        C_z * P_y - C_y * P_z = 0.
    """

    P, Pz, Py, Pzz, Pzy, Pyy = _eval_P_all_partials(zeta, y, coeffs)

    tau = float(tau)
    c0 = float(c0)
    A = c0 * y * zeta + (c0 - 1.0)
    B = (c0 - 1.0) * Py + c0 * (zeta * zeta) * Pz
    C = tau * (A * A) * Py + (tau - 1.0) * B

    Az = c0 * y
    Ay = c0 * zeta
    Bz = (c0 - 1.0) * Pzy + c0 * (2.0 * zeta * Pz +
                                  (zeta * zeta) * Pzz)
    By = (c0 - 1.0) * Pyy + c0 * (zeta * zeta) * Pzy

    Cz = tau * (2.0 * A * Az * Py + (A * A) * Pzy) + \
        (tau - 1.0) * Bz
    Cy = tau * (2.0 * A * Ay * Py + (A * A) * Pyy) + \
        (tau - 1.0) * By

    D = Cz * Py - Cy * Pz
    return P, C, D


# ===========
# cusp F real
# ===========

def _cusp_F_real(zeta, y, s, coeffs, c0):
    # tau = 1 + exp(s), so tau > 1 while allowing bounded optimization.
    tau = 1.0 + float(numpy.exp(float(s)))
    P, C, D = _deform_edge_and_cusp_terms(
        float(zeta), float(y), tau, float(c0), coeffs)
    return numpy.array([
        float(numpy.real(P)),
        float(numpy.real(C)),
        float(numpy.real(D)),
    ], dtype=float)


# =============
# solve cusp
# =============

def solve_cusp(
        coeffs,
        t_init,
        zeta_init,
        y_init=None,
        c0=1.0,
        max_iter=80,
        tol=1e-12,
        t_bounds=None,
        zeta_bounds=None):
    """
    Solve the deformation cusp equations for real (zeta, y, t).

    Unknowns are (zeta, y, s), with tau = 1 + exp(s), t = log(tau).
    The returned physical cusp location is

        x = tau*zeta + (tau - 1)/(c0*y + (c0 - 1)/zeta).
    """

    a = numpy.asarray(coeffs, dtype=numpy.complex128)
    deg_z = a.shape[0] - 1
    deg_y = a.shape[1] - 1
    c0 = float(c0)
    if c0 <= 0.0:
        raise ValueError('c0 must be positive.')

    z0 = float(zeta_init)

    if y_init is None:
        zi = numpy.power(z0, numpy.arange(deg_z + 1, dtype=numpy.int64))
        c_asc = numpy.array([numpy.dot(a[:, j], zi)
                             for j in range(deg_y + 1)],
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
    tau_minus_1 = max(tau0 - 1.0, 1e-14)
    s0 = float(numpy.log(tau_minus_1))

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
        tau_lo = max(float(numpy.exp(t_lo)) - 1.0, 1e-14)
        tau_hi = max(float(numpy.exp(t_hi)) - 1.0, 1e-14)
        s_lo, s_hi = float(numpy.log(tau_lo)), float(numpy.log(tau_hi))

    y_rad = 4.0 * (1.0 + abs(y0))
    y_lo, y_hi = float(y0 - y_rad), float(y0 + y_rad)

    lb = numpy.array([z_lo, y_lo, s_lo], dtype=float)
    ub = numpy.array([z_hi, y_hi, s_hi], dtype=float)
    x0 = numpy.array([z0, y0, s0], dtype=float)
    x0 = numpy.minimum(numpy.maximum(x0, lb), ub)

    def _F(vec):
        return _cusp_F_real(vec[0], vec[1], vec[2], a, c0)

    res = scipy.optimize.least_squares(
        _F,
        x0,
        bounds=(lb, ub),
        method='trf',
        max_nfev=int(max_iter) * 100,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        x_scale='jac')

    zeta, y, s = res.x
    tau = 1.0 + float(numpy.exp(float(s)))
    t = float(numpy.log(tau))
    x = float(numpy.real(_deform_pushforward(tau, zeta, y, c0)))

    F_final = _F(res.x)
    ok = bool(res.success and
              (numpy.max(numpy.abs(F_final)) <= max(1e-9, 50.0 * tol)))

    return {
        'ok': ok,
        't': t,
        'tau': float(tau),
        'zeta': float(zeta),
        'y': float(y),
        'x': x,
        'F': F_final,
        'success': bool(res.success),
        'message': res.message,
        'n_iter': int(res.nfev),
    }
