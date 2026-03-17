# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

"""
Numba-friendly variant of _root3.

This module keeps the same conservative philosophy as ``_root3.py``, but the
root initializer is replaced by a Numba-callable companion-matrix eigensolve.
This makes the core solver callable from ``@njit`` code, which is the main
requirement for later integration into homotopy.

The algorithm remains:

* direct companion solve of the polynomial in descending order,
* optional solve of the reversed polynomial,
* Newton polishing on the original polynomial,
* selection by maximum normalized residual.

For the bivariate polynomial

    P(z, m) = sum_{i=0}^{deg_z} sum_{j=0}^{deg_m} coeffs[i, j] z**i m**j,

``roots_m(coeffs, z)`` returns the roots in ``m`` of ``P(z, m)=0``.

For the univariate solver, ``roots(p)`` follows the same convention as
``numpy.roots``:

    p[0] x**n + p[1] x**(n-1) + ... + p[n].
"""


# =======
# Imports
# =======

from __future__ import annotations

import numpy
from numba import njit, prange

__all__ = [
    'MODE_AUTO',
    'MODE_DIRECT',
    'MODE_REVERSE',
    'poly_coeffs_in_m',
    'roots',
    'roots_many',
    'roots_m',
    'roots_m_many',
    'roots_desc_numba',
    'roots_m_numba',
    'normalized_residual',
    'max_normalized_residual',
    'normalized_residual_m',
    'max_normalized_residual_m',
    'max_backward_error',
    'max_backward_error_m',
]

MODE_AUTO = 0
MODE_DIRECT = 1
MODE_REVERSE = 2


@njit(cache=False)
def _eval_poly_in_z_numba(coeffs, z):
    deg_z = coeffs.shape[0] - 1
    deg_m = coeffs.shape[1] - 1
    a = numpy.empty(deg_m + 1, dtype=numpy.complex128)

    for j in range(deg_m + 1):
        val = coeffs[deg_z, j]
        for i in range(deg_z - 1, -1, -1):
            val = val * z + coeffs[i, j]
        a[j] = val

    return a


@njit(parallel=True, cache=False)
def _eval_poly_in_z_many_numba(coeffs, z_array):
    deg_m = coeffs.shape[1] - 1
    out = numpy.empty((z_array.size, deg_m + 1), dtype=numpy.complex128)
    for k in prange(z_array.size):
        out[k, :] = _eval_poly_in_z_numba(coeffs, z_array[k])
    return out


@njit(cache=False)
def _trim_descending_degree_numba(p_desc, abs_tol, rel_tol):
    scale = 0.0
    for i in range(p_desc.size):
        ai = abs(p_desc[i])
        if ai > scale:
            scale = ai
    tol = abs_tol + rel_tol * scale

    start = 0
    n = p_desc.size
    while start < n - 1 and abs(p_desc[start]) <= tol:
        start += 1
    return start


@njit(cache=False)
def _prepare_descending_numba(p_desc, trim_abs_tol, trim_rel_tol):
    start = _trim_descending_degree_numba(p_desc, trim_abs_tol, trim_rel_tol)
    q = p_desc[start:].copy()
    if q.size == 0:
        q = numpy.empty(1, dtype=numpy.complex128)
        q[0] = 0.0 + 0.0j
    return q, q.size - 1


@njit(cache=False)
def _horner_desc_numba(p_desc, x):
    y = p_desc[0]
    for i in range(1, p_desc.size):
        y = y * x + p_desc[i]
    return y


@njit(cache=False)
def _horner_desc_and_derivative_numba(p_desc, x):
    p = p_desc[0]
    dp = 0.0 + 0.0j
    for i in range(1, p_desc.size):
        dp = dp * x + p
        p = p * x + p_desc[i]
    return p, dp


@njit(cache=False)
def _newton_polish_desc_inplace_numba(roots, p_desc, max_iter, step_tol, res_tol):
    for j in range(roots.size):
        x = roots[j]
        if not numpy.isfinite(x.real) or not numpy.isfinite(x.imag):
            continue
        for _ in range(max_iter):
            p, dp = _horner_desc_and_derivative_numba(p_desc, x)
            if abs(p) <= res_tol:
                break
            if abs(dp) == 0.0 or (not numpy.isfinite(dp.real)) or \
                    (not numpy.isfinite(dp.imag)):
                break
            dx = -p / dp
            x_new = x + dx
            if (not numpy.isfinite(x_new.real)) or (not numpy.isfinite(x_new.imag)):
                break
            x = x_new
            if abs(dx) <= step_tol:
                break
        roots[j] = x


@njit(cache=False)
def _sort_roots_inplace_numba(roots):
    n = roots.size
    for i in range(n - 1):
        k = i
        rk = roots[i].real
        ik = roots[i].imag
        for j in range(i + 1, n):
            rj = roots[j].real
            ij = roots[j].imag
            if (rj < rk) or ((rj == rk) and (ij < ik)):
                k = j
                rk = rj
                ik = ij
        if k != i:
            tmp = roots[i]
            roots[i] = roots[k]
            roots[k] = tmp


@njit(cache=False)
def _normalized_residuals_desc_numba(p_desc, roots):
    out = numpy.empty(roots.size, dtype=numpy.float64)
    n = p_desc.size - 1

    for i in range(roots.size):
        x = roots[i]
        if (not numpy.isfinite(x.real)) or (not numpy.isfinite(x.imag)):
            out[i] = numpy.inf
            continue

        num = abs(_horner_desc_numba(p_desc, x))
        den = 0.0
        ax = abs(x)
        pow_ax = 1.0
        # accumulate from constant term upward for fewer pow calls
        for j in range(n, -1, -1):
            den += abs(p_desc[j]) * pow_ax
            pow_ax *= ax

        if den == 0.0:
            out[i] = numpy.inf
        else:
            out[i] = num / den

    return out


@njit(cache=False)
def _max_normalized_residual_desc_numba(p_desc, roots):
    if roots.size == 0:
        return numpy.inf
    vals = _normalized_residuals_desc_numba(p_desc, roots)
    out = vals[0]
    for i in range(1, vals.size):
        if vals[i] > out:
            out = vals[i]
    return out


@njit(cache=False)
def _max_backward_error_desc_numba(p_desc, roots):
    out = 0.0
    for i in range(roots.size):
        r = roots[i]
        if (not numpy.isfinite(r.real)) or (not numpy.isfinite(r.imag)):
            continue
        val = abs(_horner_desc_numba(p_desc, r))
        if val > out:
            out = val
    return out


@njit(cache=False)
def _companion_roots_desc_numba(p_desc):
    n = p_desc.size - 1
    if n <= 0:
        return numpy.empty(0, dtype=numpy.complex128)
    if n == 1:
        out = numpy.empty(1, dtype=numpy.complex128)
        out[0] = -p_desc[1] / p_desc[0]
        return out

    lead = p_desc[0]
    C = numpy.zeros((n, n), dtype=numpy.complex128)
    inv_lead = 1.0 / lead
    for j in range(n):
        C[0, j] = -p_desc[j + 1] * inv_lead
    for i in range(1, n):
        C[i, i - 1] = 1.0 + 0.0j
    return numpy.linalg.eigvals(C)


@njit(cache=False)
def _roots_direct_desc_numba(p_desc):
    return _companion_roots_desc_numba(p_desc)


@njit(cache=False)
def _roots_reversed_desc_numba(p_desc):
    q_desc = p_desc[::-1].copy()
    y = _companion_roots_desc_numba(q_desc)
    x = numpy.empty_like(y)
    for i in range(y.size):
        yi = y[i]
        if abs(yi) == 0.0:
            x[i] = numpy.inf + 0.0j
        else:
            x[i] = 1.0 / yi
    return x


@njit(cache=False)
def _finalize_candidate_numba(p_desc, r, polish, polish_iter, polish_step_tol,
                              polish_res_tol, sort_roots):
    out = r.copy()
    if polish and out.size > 0:
        _newton_polish_desc_inplace_numba(
            out, p_desc, polish_iter, polish_step_tol, polish_res_tol
        )
    if sort_roots and out.size > 0:
        _sort_roots_inplace_numba(out)
    return out


@njit(cache=False)
def roots_desc_numba(p_desc,
                     trim_abs_tol=0.0,
                     trim_rel_tol=1e-14,
                     mode=MODE_AUTO,
                     polish=True,
                     polish_iter=8,
                     polish_step_tol=1e-14,
                     polish_res_tol=1e-14,
                     sort_roots=True):
    q_desc, deg = _prepare_descending_numba(
        p_desc, trim_abs_tol, trim_rel_tol
    )

    if deg <= 0:
        return numpy.empty(0, dtype=numpy.complex128)
    if deg == 1:
        out = numpy.empty(1, dtype=numpy.complex128)
        out[0] = -q_desc[1] / q_desc[0]
        if sort_roots:
            _sort_roots_inplace_numba(out)
        return out

    if mode == MODE_DIRECT:
        return _finalize_candidate_numba(
            q_desc, _roots_direct_desc_numba(q_desc),
            polish, polish_iter, polish_step_tol, polish_res_tol, sort_roots,
        )

    if mode == MODE_REVERSE:
        return _finalize_candidate_numba(
            q_desc, _roots_reversed_desc_numba(q_desc),
            polish, polish_iter, polish_step_tol, polish_res_tol, sort_roots,
        )

    r_direct = _finalize_candidate_numba(
        q_desc, _roots_direct_desc_numba(q_desc),
        polish, polish_iter, polish_step_tol, polish_res_tol, sort_roots,
    )
    r_reverse = _finalize_candidate_numba(
        q_desc, _roots_reversed_desc_numba(q_desc),
        polish, polish_iter, polish_step_tol, polish_res_tol, sort_roots,
    )

    e_direct = _max_normalized_residual_desc_numba(q_desc, r_direct)
    e_reverse = _max_normalized_residual_desc_numba(q_desc, r_reverse)

    if e_reverse < e_direct:
        return r_reverse
    return r_direct


@njit(cache=False)
def roots_m_numba(coeffs,
                  z,
                  trim_abs_tol=0.0,
                  trim_rel_tol=1e-14,
                  mode=MODE_AUTO,
                  polish=True,
                  polish_iter=8,
                  polish_step_tol=1e-14,
                  polish_res_tol=1e-14,
                  sort_roots=True):
    a_asc = _eval_poly_in_z_numba(coeffs, z)
    p_desc = a_asc[::-1].copy()
    return roots_desc_numba(
        p_desc,
        trim_abs_tol,
        trim_rel_tol,
        mode,
        polish,
        polish_iter,
        polish_step_tol,
        polish_res_tol,
        sort_roots,
    )


def poly_coeffs_in_m(coeffs: numpy.ndarray, z: complex) -> numpy.ndarray:
    coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
    return _eval_poly_in_z_numba(coeffs, complex(z))


def _parse_mode(use_reverse: str) -> int:
    mode = str(use_reverse).lower()
    if mode == 'auto':
        return MODE_AUTO
    if mode == 'direct':
        return MODE_DIRECT
    if mode == 'reverse':
        return MODE_REVERSE
    raise ValueError("use_reverse should be 'auto', 'direct', or 'reverse'.")


def roots(p: numpy.ndarray,
          *,
          trim_abs_tol: float = 0.0,
          trim_rel_tol: float = 1e-14,
          use_reverse: str = 'auto',
          polish: bool = True,
          polish_iter: int = 8,
          polish_step_tol: float = 1e-14,
          polish_res_tol: float = 1e-14,
          sort_roots: bool = True) -> numpy.ndarray:
    p_desc = numpy.asarray(p, dtype=numpy.complex128).ravel()
    return roots_desc_numba(
        p_desc,
        float(trim_abs_tol),
        float(trim_rel_tol),
        _parse_mode(use_reverse),
        bool(polish),
        int(polish_iter),
        float(polish_step_tol),
        float(polish_res_tol),
        bool(sort_roots),
    )


def roots_many(p_array: numpy.ndarray,
               *,
               trim_abs_tol: float = 0.0,
               trim_rel_tol: float = 1e-14,
               use_reverse: str = 'auto',
               polish: bool = True,
               polish_iter: int = 8,
               polish_step_tol: float = 1e-14,
               polish_res_tol: float = 1e-14,
               sort_roots: bool = True) -> tuple[numpy.ndarray, numpy.ndarray]:
    p_array = numpy.asarray(p_array, dtype=numpy.complex128)
    if p_array.ndim != 2:
        raise ValueError('p_array should be a 2D array.')

    n_poly, n_coeff = p_array.shape
    roots_out = numpy.full(
        (n_poly, max(0, n_coeff - 1)),
        numpy.nan + 1j * numpy.nan,
        dtype=numpy.complex128,
    )
    n_eff = numpy.zeros(n_poly, dtype=numpy.int64)
    mode = _parse_mode(use_reverse)

    for i in range(n_poly):
        r = roots_desc_numba(
            p_array[i],
            float(trim_abs_tol),
            float(trim_rel_tol),
            mode,
            bool(polish),
            int(polish_iter),
            float(polish_step_tol),
            float(polish_res_tol),
            bool(sort_roots),
        )
        roots_out[i, :r.size] = r
        n_eff[i] = r.size

    return roots_out, n_eff


def roots_m(coeffs: numpy.ndarray,
            z: complex,
            *,
            trim_abs_tol: float = 0.0,
            trim_rel_tol: float = 1e-14,
            use_reverse: str = 'auto',
            polish: bool = True,
            polish_iter: int = 8,
            polish_step_tol: float = 1e-14,
            polish_res_tol: float = 1e-14,
            sort_roots: bool = True) -> numpy.ndarray:
    coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
    return roots_m_numba(
        coeffs,
        complex(z),
        float(trim_abs_tol),
        float(trim_rel_tol),
        _parse_mode(use_reverse),
        bool(polish),
        int(polish_iter),
        float(polish_step_tol),
        float(polish_res_tol),
        bool(sort_roots),
    )


def roots_m_many(coeffs: numpy.ndarray,
                 z_array: numpy.ndarray,
                 *,
                 trim_abs_tol: float = 0.0,
                 trim_rel_tol: float = 1e-14,
                 use_reverse: str = 'auto',
                 polish: bool = True,
                 polish_iter: int = 8,
                 polish_step_tol: float = 1e-14,
                 polish_res_tol: float = 1e-14,
                 sort_roots: bool = True) -> tuple[numpy.ndarray, numpy.ndarray]:
    coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
    z_array = numpy.asarray(z_array, dtype=numpy.complex128).ravel()
    mode = _parse_mode(use_reverse)

    deg_m = coeffs.shape[1] - 1
    roots_out = numpy.full(
        (z_array.size, max(0, deg_m)),
        numpy.nan + 1j * numpy.nan,
        dtype=numpy.complex128,
    )
    n_eff = numpy.zeros(z_array.size, dtype=numpy.int64)

    for i in range(z_array.size):
        r = roots_m_numba(
            coeffs,
            z_array[i],
            float(trim_abs_tol),
            float(trim_rel_tol),
            mode,
            bool(polish),
            int(polish_iter),
            float(polish_step_tol),
            float(polish_res_tol),
            bool(sort_roots),
        )
        roots_out[i, :r.size] = r
        n_eff[i] = r.size

    return roots_out, n_eff


def normalized_residual(p: numpy.ndarray, roots_: numpy.ndarray) -> numpy.ndarray:
    p_desc = numpy.asarray(p, dtype=numpy.complex128).ravel()
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return _normalized_residuals_desc_numba(p_desc, roots_)


def max_normalized_residual(p: numpy.ndarray, roots_: numpy.ndarray) -> float:
    p_desc = numpy.asarray(p, dtype=numpy.complex128).ravel()
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_normalized_residual_desc_numba(p_desc, roots_))


def normalized_residual_m(coeffs: numpy.ndarray,
                          z: complex,
                          roots_: numpy.ndarray) -> numpy.ndarray:
    a_asc = poly_coeffs_in_m(coeffs, z)
    p_desc = a_asc[::-1]
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return _normalized_residuals_desc_numba(p_desc, roots_)


def max_normalized_residual_m(coeffs: numpy.ndarray,
                              z: complex,
                              roots_: numpy.ndarray) -> float:
    a_asc = poly_coeffs_in_m(coeffs, z)
    p_desc = a_asc[::-1]
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_normalized_residual_desc_numba(p_desc, roots_))


def max_backward_error(p: numpy.ndarray, roots_: numpy.ndarray) -> float:
    p_desc = numpy.asarray(p, dtype=numpy.complex128).ravel()
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_backward_error_desc_numba(p_desc, roots_))


def max_backward_error_m(coeffs: numpy.ndarray,
                         z: complex,
                         roots_: numpy.ndarray) -> float:
    a_asc = poly_coeffs_in_m(coeffs, z)
    p_desc = a_asc[::-1]
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_backward_error_desc_numba(p_desc, roots_))
