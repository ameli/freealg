# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE

"""Free Decompression (FD) Newton solver.

This file defines `decompress_newton` used by AlgebraicForm.decompress(...,
method='one').

Implementation notes
--------------------
We solve, for each query point z and time t (tau = exp(t)), the 2x2 system
in variables (zeta, y):

    F1(zeta,y) = P(zeta, y) = 0
    F2(zeta,y) = z - zeta + (tau-1)/y = 0

Then w = m(t,z) = y/tau.

Speed-critical change vs earlier variants: all polynomial evaluations
(P, dP/dzeta, dP/dy) use scalar Horner without allocating power matrices.
"""

from __future__ import annotations

import numpy

__all__ = ['decompress_newton']


# =====================
# scalar poly evaluation
# =====================

def _eval_a_and_da(z: complex, a_coeffs: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Evaluate a_j(z) and a'_j(z) for j=0..s where P(z,y)=sum_j a_j(z) y^j.

    a_coeffs has shape (deg_z+1, s+1) storing coefficients in z ascending:
        a_coeffs[i,j] = coeff of z^i in a_j(z).
    """
    deg_z = a_coeffs.shape[0] - 1
    s = a_coeffs.shape[1] - 1

    a = numpy.empty(s + 1, dtype=numpy.complex128)
    da = numpy.empty(s + 1, dtype=numpy.complex128)

    # Horner for each column j
    for j in range(s + 1):
        col = a_coeffs[:, j]
        # a_j(z)
        v = complex(col[deg_z])
        for i in range(deg_z - 1, -1, -1):
            v = v * z + col[i]
        a[j] = v

        # a'_j(z)
        if deg_z == 0:
            da[j] = 0.0 + 0.0j
        else:
            vp = complex(deg_z * col[deg_z])
            for i in range(deg_z - 1, 0, -1):
                vp = vp * z + (i * col[i])
            da[j] = vp

    return a, da


def _eval_P_Pz_Py(z: complex, y: complex, a_coeffs: numpy.ndarray) -> tuple[complex, complex, complex]:
    """Return P(z,y), Pz(z,y)=\partial_z P, Py(z,y)=\partial_y P (scalars)."""
    a, da = _eval_a_and_da(z, a_coeffs)
    s = a_coeffs.shape[1] - 1

    # Build powers of y incrementally (cheap; s is small)
    ypow = 1.0 + 0.0j
    P = 0.0 + 0.0j
    Pz = 0.0 + 0.0j

    for j in range(s + 1):
        P += a[j] * ypow
        Pz += da[j] * ypow
        ypow *= y

    # Py
    Py = 0.0 + 0.0j
    if s >= 1:
        ypow = 1.0 + 0.0j  # y^(j-1)
        for j in range(1, s + 1):
            Py += (j * a[j]) * ypow
            ypow *= y

    return P, Pz, Py


# =========
# 2x2 newton
# =========

def _newton_2x2(
    z: complex,
    tau: float,
    zeta0: complex,
    y0: complex,
    a_coeffs: numpy.ndarray,
    *,
    max_iter: int,
    tol: float,
    damping: float,
    step_clip: float,
    w_min: float,
    require_imw_pos: bool,
    im_eps: float,
) -> tuple[complex, complex, bool, int]:
    """Solve for (zeta,y) at fixed (z,tau)."""
    zeta = complex(zeta0)
    y = complex(y0)
    tau_m1 = tau - 1.0

    # Avoid singular y
    if abs(y) < w_min:
        y = (w_min + 0.0j)

    for it in range(max_iter):
        P, Pz, Py = _eval_P_Pz_Py(zeta, y, a_coeffs)
        F1 = P
        F2 = z - zeta + tau_m1 / y

        # Stop criterion on both equations
        if (abs(F1) <= tol) and (abs(F2) <= tol):
            w = y / tau
            if (not require_imw_pos) or (z.imag <= 0.0) or (w.imag > im_eps):
                return zeta, y, True, it

        # Jacobian
        # F1_zeta = Pz
        # F1_y    = Py
        # F2_zeta = -1
        # F2_y    = -(tau-1)/y^2
        inv_y2 = 1.0 / (y * y)
        J11 = Pz
        J12 = Py
        J21 = -1.0 + 0.0j
        J22 = -(tau_m1) * inv_y2

        det = J11 * J22 - J12 * J21
        if det == 0.0:
            break

        # d = -J^{-1}F
        # d_zeta = (-F1*J22 + F2*J12)/det
        # d_y    = ( J21*F1 - J11*F2)/det
        d_zeta = (-F1 * J22 + F2 * J12) / det
        d_y = (J21 * F1 - J11 * F2) / det

        # Clip step
        if step_clip is not None and step_clip > 0.0:
            m = max(abs(d_zeta), abs(d_y))
            if m > step_clip:
                s = step_clip / m
                d_zeta *= s
                d_y *= s

        lam = float(damping) if damping is not None else 1.0
        if lam <= 0.0:
            lam = 1.0

        # Simple backtracking to enforce Im(w)>0 for z in C+
        # (and avoid y crossing 0)
        for _ in range(12):
            zeta_try = zeta + lam * d_zeta
            y_try = y + lam * d_y

            if abs(y_try) < w_min:
                lam *= 0.5
                continue

            if require_imw_pos and (z.imag > 0.0):
                w_try = y_try / tau
                if w_try.imag <= im_eps:
                    lam *= 0.5
                    continue

            # Accept
            zeta = zeta_try
            y = y_try
            break

        else:
            # could not find acceptable step
            break

    return zeta, y, False, max_iter


# ===============
# public interface
# ===============

def decompress_newton(
    z_query,
    t_all,
    a_coeffs,
    *,
    w0_list=None,
    max_iter: int = 40,
    tol: float = 1e-13,
    damping: float = 1.0,
    step_clip: float = 5.0,
    w_min: float = 1e-300,
    max_split: int = 4,
    require_imw_pos: bool = True,
    im_eps: float = 1e-14,
    sweep: bool = False,
    verbose: bool = False,
    debug: bool = False,
    debug_idx=None,
    eta_rescale: str | None = None,
    **kwargs,
):
    """Compute w(t,z)=m(t,z) on a time grid using FD and Newton.

    Parameters mirror earlier versions; extra kwargs are ignored.

    eta_rescale:
      - None (default): keep z_query fixed for all times.
      - 'inv_tau': use z.imag/tau at time t. (Useful for mass checks.)

    Returns
    -------
    W : (n_t, n_z) complex
    ok : (n_t, n_z) bool
    """

    z_query = numpy.asarray(z_query, dtype=numpy.complex128).ravel()
    t_all = numpy.asarray(t_all, dtype=float).ravel()
    if t_all.size == 0:
        raise ValueError('t_all is empty.')
    if numpy.any(numpy.diff(t_all) < 0):
        raise ValueError('t_all must be sorted increasing.')

    n_z = z_query.size
    n_t = t_all.size

    if w0_list is None:
        w0 = -1.0 / z_query
    else:
        w0 = numpy.asarray(w0_list, dtype=numpy.complex128).ravel()
        if w0.size != n_z:
            raise ValueError('w0_list must have same length as z_query.')

    W = numpy.empty((n_t, n_z), dtype=numpy.complex128)
    ok = numpy.zeros((n_t, n_z), dtype=bool)
    W[0, :] = w0
    ok[0, :] = numpy.isfinite(w0.real) & numpy.isfinite(w0.imag)

    dbg_set = set()
    if debug_idx is not None:
        try:
            dbg_set = set(int(i) for i in debug_idx)
        except Exception:
            dbg_set = set()

    for it in range(1, n_t):
        t_prev = float(t_all[it - 1])
        t = float(t_all[it])
        tau_prev = float(numpy.exp(t_prev))
        tau = float(numpy.exp(t))

        if eta_rescale == 'inv_tau':
            z_t = z_query.real + 1j * (z_query.imag / tau)
        else:
            z_t = z_query

        # warm start: previous w(t_prev,z) -> y seed
        w_seed = W[it - 1, :].copy()
        y_seed = tau_prev * w_seed

        # build zeta seed that satisfies the constraint initially
        y_safe = y_seed.copy()
        tiny = numpy.abs(y_safe) < w_min
        if numpy.any(tiny):
            y_safe[tiny] = (w_min + 0.0j)
        zeta_seed = z_t + (tau - 1.0) / y_safe

        w_out = numpy.empty(n_z, dtype=numpy.complex128)
        ok_out = numpy.zeros(n_z, dtype=bool)

        # optional sweep within x at same time
        y_last = None
        for j in range(n_z):
            z = z_t[j]

            if sweep and (y_last is not None):
                y0 = y_last
                y_safe0 = y0 if abs(y0) > w_min else (w_min + 0.0j)
                zeta0 = z + (tau - 1.0) / y_safe0
            else:
                y0 = y_seed[j]
                zeta0 = zeta_seed[j]

            zeta, y, okj, nit = _newton_2x2(
                z, tau, zeta0, y0, a_coeffs,
                max_iter=max_iter, tol=tol,
                damping=damping, step_clip=step_clip,
                w_min=w_min, require_imw_pos=require_imw_pos,
                im_eps=im_eps,
            )

            if (not okj) and (max_split is not None) and (max_split > 0):
                # try smaller pseudo-steps by tempering tau -> tau_mid
                # (helps if time steps are too large)
                for k in range(1, int(max_split) + 1):
                    tau_mid = tau_prev + (tau - tau_prev) * (k / float(max_split))
                    zeta_mid, y_mid, ok_mid, _ = _newton_2x2(
                        z, tau_mid, zeta0, y0, a_coeffs,
                        max_iter=max_iter, tol=tol,
                        damping=damping, step_clip=step_clip,
                        w_min=w_min, require_imw_pos=require_imw_pos,
                        im_eps=im_eps,
                    )
                    if ok_mid:
                        # now jump from (tau_mid) seed
                        zeta0b = zeta_mid
                        y0b = y_mid
                        zeta, y, okj, nit = _newton_2x2(
                            z, tau, zeta0b, y0b, a_coeffs,
                            max_iter=max_iter, tol=tol,
                            damping=damping, step_clip=step_clip,
                            w_min=w_min, require_imw_pos=require_imw_pos,
                            im_eps=im_eps,
                        )
                        if okj:
                            break

            wj = y / tau
            w_out[j] = wj
            ok_out[j] = bool(okj)

            if okj:
                y_last = y

            if debug and (j in dbg_set):
                print(
                    f"[t={t:0.6f}] j={j} z={z.real:+.6f}{z.imag:+.2e}j "
                    f"zeta={zeta.real:+.6f}{zeta.imag:+.2e}j "
                    f"y={y.real:+.3e}{y.imag:+.3e}j "
                    f"w={wj.real:+.3e}{wj.imag:+.3e}j ok={okj} it={nit}"
                )

        W[it, :] = w_out
        ok[it, :] = ok_out

        if verbose:
            print(f'[t={t:0.6f}] ok={ok_out.mean():0.3f}')

    return W, ok
