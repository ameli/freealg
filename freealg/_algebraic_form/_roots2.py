# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# Stable polynomial root solver for univariate polynomials, and for bivariate
# polynomials P(z, m) after fixing z.

"""
Numerical root solver for univariate polynomials, and for bivariate
polynomials P(z, m) after fixing z.

Compared to ``_roots.py``, this version makes four important changes.

1. Degree trimming is disabled by default. Tiny leading coefficients may still
   be root-critical for branch tracking, so trimming is now strictly opt-in.
2. Before forming the companion matrix, the variable is scaled so the roots are
   closer to unit magnitude. This improves the conditioning of the companion
   eigenproblem when roots span a wide dynamic range.
3. Direct and reversed solves are both available in ``MODE_AUTO``. Their
   candidates are polished on the original polynomial, and the more accurate
   whole set is selected conservatively using the maximum normalized residual.
4. Newton polishing uses relative stopping and a simple backtracking safeguard.

The coefficient convention for ``roots(p)`` matches ``numpy.roots``:

    p[0] x**n + p[1] x**(n-1) + ... + p[n].
"""

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


# ====================
# eval poly in z numba
# ====================

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


# =========================
# eval poly in z many numba
# =========================

@njit(parallel=True, cache=False)
def _eval_poly_in_z_many_numba(coeffs, z_array):

    deg_m = coeffs.shape[1] - 1
    out = numpy.empty((z_array.size, deg_m + 1), dtype=numpy.complex128)
    for k in prange(z_array.size):
        out[k, :] = _eval_poly_in_z_numba(coeffs, z_array[k])
    return out


# ============================
# trim descending degree numba
# ============================

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


# ========================
# prepare descending numba
# ========================

@njit(cache=False)
def _prepare_descending_numba(p_desc, trim_abs_tol, trim_rel_tol):

    start = _trim_descending_degree_numba(p_desc, trim_abs_tol, trim_rel_tol)
    q = p_desc[start:].copy()
    if q.size == 0:
        q = numpy.empty(1, dtype=numpy.complex128)
        q[0] = 0.0 + 0.0j
    return q, q.size - 1


# =================
# horner desc numba
# =================

@njit(cache=False)
def _horner_desc_numba(p_desc, x):

    y = p_desc[0]
    for i in range(1, p_desc.size):
        y = y * x + p_desc[i]
    return y


# ================================
# horner desc and derivative numba
# ================================

@njit(cache=False)
def _horner_desc_and_derivative_numba(p_desc, x):

    p = p_desc[0]
    dp = 0.0 + 0.0j
    for i in range(1, p_desc.size):
        dp = dp * x + p
        p = p * x + p_desc[i]
    return p, dp


# =======================
# scalar denom desc numba
# =======================

@njit(cache=False)
def _residual_denom_desc_numba(p_desc, x):

    den = 0.0
    ax = abs(x)
    pow_ax = 1.0
    n = p_desc.size - 1
    for j in range(n, -1, -1):
        den += abs(p_desc[j]) * pow_ax
        pow_ax *= ax
    return den


# ================================
# normalized residual scalar numba
# ================================

@njit(cache=False)
def _normalized_residual_desc_scalar_numba(p_desc, x):

    if (not numpy.isfinite(x.real)) or (not numpy.isfinite(x.imag)):
        return numpy.inf

    num = abs(_horner_desc_numba(p_desc, x))
    den = _residual_denom_desc_numba(p_desc, x)
    if den == 0.0:
        return numpy.inf
    return num / den


# ================================
# newton polish desc inplace numba
# ================================

@njit(cache=False)
def _newton_polish_desc_inplace_numba(roots, p_desc, max_iter,
                                      rel_step_tol, rel_res_tol,
                                      max_backtrack):

    for j in range(roots.size):
        x = roots[j]
        if (not numpy.isfinite(x.real)) or (not numpy.isfinite(x.imag)):
            continue

        res = _normalized_residual_desc_scalar_numba(p_desc, x)
        if not numpy.isfinite(res):
            continue

        for _ in range(max_iter):
            p, dp = _horner_desc_and_derivative_numba(p_desc, x)
            if (not numpy.isfinite(p.real)) or (not numpy.isfinite(p.imag)):
                break
            if (not numpy.isfinite(dp.real)) or (not numpy.isfinite(dp.imag)):
                break
            if abs(dp) == 0.0:
                break

            den = _residual_denom_desc_numba(p_desc, x)
            if den > 0.0 and abs(p) <= rel_res_tol * den:
                break

            dx_full = -p / dp
            if abs(dx_full) <= rel_step_tol * (1.0 + abs(x)):
                x = x + dx_full
                break

            best_x = x
            best_res = res
            step = dx_full
            accepted = False

            for _bt in range(max_backtrack + 1):
                x_try = x + step
                if (not numpy.isfinite(x_try.real)) or \
                        (not numpy.isfinite(x_try.imag)):
                    step = 0.5 * step
                    continue

                res_try = _normalized_residual_desc_scalar_numba(p_desc, x_try)
                if numpy.isfinite(res_try) and res_try <= best_res:
                    best_x = x_try
                    best_res = res_try
                    accepted = True
                    break
                step = 0.5 * step

            if not accepted:
                break

            x = best_x
            res = best_res

            if res <= rel_res_tol:
                break
            if abs(step) <= rel_step_tol * (1.0 + abs(x)):
                break

        roots[j] = x


# ========================
# sort roots inplace numba
# ========================

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


# ===============================
# normalized residuals desc numba
# ===============================

@njit(cache=False)
def _normalized_residuals_desc_numba(p_desc, roots):

    out = numpy.empty(roots.size, dtype=numpy.float64)
    for i in range(roots.size):
        out[i] = _normalized_residual_desc_scalar_numba(p_desc, roots[i])
    return out


# ==================================
# max normalized residual desc numba
# ==================================

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


# =============================
# max backward error desc numba
# =============================

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


# =========================
# root scale estimate numba
# =========================

@njit(cache=False)
def _root_scale_desc_numba(p_desc):
    """
    Estimate a scaling ``x = s y`` so the roots in ``y`` are closer to O(1).

    We use a simple Fujiwara/Cauchy-style bound based on the non-leading
    coefficients. This is inexpensive and robust in Numba.
    """

    n = p_desc.size - 1
    if n <= 0:
        return 1.0

    a0 = abs(p_desc[0])
    if a0 == 0.0 or (not numpy.isfinite(a0)):
        return 1.0

    r = 0.0
    for k in range(1, n + 1):
        ak = abs(p_desc[k])
        if ak == 0.0 or (not numpy.isfinite(ak)):
            continue
        expo = 1.0 / k
        cand = (ak / a0) ** expo
        if cand > r:
            r = cand

    if (not numpy.isfinite(r)) or r <= 0.0:
        return 1.0

    # Avoid over-aggressive tiny scales.
    return max(1.0, r)


# ===========================
# scale descending poly numba
# ===========================

@njit(cache=False)
def _scale_desc_poly_numba(p_desc, scale):
    """
    For p(x)=sum_k a_k x^(n-k), form q(y)=p(scale*y).
    Then q has coefficients b_k = a_k * scale^(n-k).
    """

    n = p_desc.size - 1
    q_desc = numpy.empty_like(p_desc)
    pow_s = 1.0 + 0.0j

    # build from constant upward, then reverse exponent logic
    # q_desc[n] = a_n * scale^0, q_desc[n-1] = a_{n-1} * scale^1, ...
    for j in range(n, -1, -1):
        q_desc[j] = p_desc[j] * pow_s
        pow_s = pow_s * scale
    return q_desc


# ==========================
# companion roots desc numba
# ==========================

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


# =======================
# solve scaled desc numba
# =======================

@njit(cache=False)
def _solve_scaled_desc_numba(p_desc):

    scale = _root_scale_desc_numba(p_desc)
    q_desc = _scale_desc_poly_numba(p_desc, scale)
    y = _companion_roots_desc_numba(q_desc)

    x = numpy.empty_like(y)
    for i in range(y.size):
        x[i] = scale * y[i]
    return x


# =======================
# roots direct desc numba
# =======================

@njit(cache=False)
def _roots_direct_desc_numba(p_desc):

    return _solve_scaled_desc_numba(p_desc)


# =========================
# roots reversed desc numba
# =========================

@njit(cache=False)
def _roots_reversed_desc_numba(p_desc):

    q_desc = p_desc[::-1].copy()
    y = _solve_scaled_desc_numba(q_desc)
    x = numpy.empty_like(y)
    for i in range(y.size):
        yi = y[i]
        if abs(yi) == 0.0:
            x[i] = numpy.inf + 0.0j
        else:
            x[i] = 1.0 / yi
    return x


# ========================
# finalize candidate numba
# ========================

@njit(cache=False)
def _finalize_candidate_numba(p_desc, r, polish, polish_iter,
                              polish_rel_step_tol, polish_rel_res_tol,
                              polish_backtrack, sort_roots):
    out = r.copy()
    if polish and out.size > 0:
        _newton_polish_desc_inplace_numba(
            out, p_desc, polish_iter,
            polish_rel_step_tol, polish_rel_res_tol,
            polish_backtrack)
    if sort_roots and out.size > 0:
        _sort_roots_inplace_numba(out)
    return out


# ========================
# root cluster close numba
# ========================

@njit(cache=False)
def _roots_close_numba(a, b, atol, rtol):
    thr = atol + rtol * max(1.0, abs(a), abs(b))
    return abs(a - b) <= thr


# ======================
# merge roots sets numba
# ======================

@njit(cache=False)
def _merge_root_sets_numba(p_desc, r1, r2, atol, rtol,
                           polish_iter, polish_rel_step_tol,
                           polish_rel_res_tol, polish_backtrack,
                           sort_roots):
    """
    Merge direct and reverse candidates per root rather than picking one set.

    Strategy:
    - start from the concatenation of both sets,
    - polish all candidates on the original polynomial,
    - sort candidates by normalized residual,
    - greedily keep candidates that are not already represented by a nearby
      better one,
    - if fewer than n roots survive, fill from the remaining best candidates.
    """

    n = p_desc.size - 1
    cand = numpy.empty(r1.size + r2.size, dtype=numpy.complex128)
    for i in range(r1.size):
        cand[i] = r1[i]
    for i in range(r2.size):
        cand[r1.size + i] = r2[i]

    if cand.size > 0:
        _newton_polish_desc_inplace_numba(
            cand, p_desc, polish_iter,
            polish_rel_step_tol, polish_rel_res_tol,
            polish_backtrack)

    res = _normalized_residuals_desc_numba(p_desc, cand)

    # Sort indices by residual ascending.
    idx = numpy.arange(cand.size)
    for i in range(idx.size - 1):
        k = i
        best = res[idx[i]]
        for j in range(i + 1, idx.size):
            val = res[idx[j]]
            if val < best:
                best = val
                k = j
        if k != i:
            tmp = idx[i]
            idx[i] = idx[k]
            idx[k] = tmp

    keep = numpy.empty(n, dtype=numpy.complex128)
    n_keep = 0

    for ii in range(idx.size):
        r = cand[idx[ii]]
        if (not numpy.isfinite(r.real)) or (not numpy.isfinite(r.imag)):
            continue

        is_new = True
        for j in range(n_keep):
            if _roots_close_numba(r, keep[j], atol, rtol):
                is_new = False
                break

        if is_new:
            keep[n_keep] = r
            n_keep += 1
            if n_keep == n:
                break

    # If clustering was too aggressive, top up with remaining best residuals.
    if n_keep < n:
        for ii in range(idx.size):
            r = cand[idx[ii]]
            if (not numpy.isfinite(r.real)) or (not numpy.isfinite(r.imag)):
                continue

            present = False
            for j in range(n_keep):
                if r == keep[j]:
                    present = True
                    break
            if not present:
                keep[n_keep] = r
                n_keep += 1
                if n_keep == n:
                    break

    out = keep[:n_keep].copy()
    if sort_roots and out.size > 0:
        _sort_roots_inplace_numba(out)
    return out


# ================
# roots desc numba
# ================

@njit(cache=False)
def roots_desc_numba(p_desc,
                     trim_abs_tol=0.0,
                     trim_rel_tol=0.0,
                     mode=MODE_AUTO,
                     polish=True,
                     polish_iter=20,
                     polish_step_tol=1e-13,
                     polish_res_tol=1e-13,
                     sort_roots=True,
                     merge_atol=1e-12,
                     merge_rtol=1e-8,
                     polish_backtrack=8):

    q_desc, deg = _prepare_descending_numba(p_desc, trim_abs_tol, trim_rel_tol)

    if deg <= 0:
        return numpy.empty(0, dtype=numpy.complex128)
    if deg == 1:
        out = numpy.empty(1, dtype=numpy.complex128)
        out[0] = -q_desc[1] / q_desc[0]
        if polish:
            _newton_polish_desc_inplace_numba(
                out, q_desc, polish_iter,
                polish_step_tol, polish_res_tol,
                polish_backtrack)
        if sort_roots:
            _sort_roots_inplace_numba(out)
        return out

    if mode == MODE_DIRECT:
        return _finalize_candidate_numba(
            q_desc, _roots_direct_desc_numba(q_desc),
            polish, polish_iter, polish_step_tol, polish_res_tol,
            polish_backtrack, sort_roots)

    if mode == MODE_REVERSE:
        return _finalize_candidate_numba(
            q_desc, _roots_reversed_desc_numba(q_desc),
            polish, polish_iter, polish_step_tol, polish_res_tol,
            polish_backtrack, sort_roots)

    r_direct = _finalize_candidate_numba(
        q_desc, _roots_direct_desc_numba(q_desc),
        polish, polish_iter, polish_step_tol, polish_res_tol,
        polish_backtrack, sort_roots)

    r_reverse = _finalize_candidate_numba(
        q_desc, _roots_reversed_desc_numba(q_desc),
        polish, polish_iter, polish_step_tol, polish_res_tol,
        polish_backtrack, sort_roots)

    e_direct = _max_normalized_residual_desc_numba(q_desc, r_direct)
    e_reverse = _max_normalized_residual_desc_numba(q_desc, r_reverse)

    if e_reverse < e_direct:
        return r_reverse
    return r_direct


# =============
# roots m numba
# =============

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
                  sort_roots=True,
                  merge_atol=1e-12,
                  merge_rtol=1e-8,
                  polish_backtrack=8):

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
        merge_atol,
        merge_rtol,
        polish_backtrack)


# ================
# poly coeffs in m
# ================

def poly_coeffs_in_m(coeffs: numpy.ndarray, z: complex) -> numpy.ndarray:

    coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
    return _eval_poly_in_z_numba(coeffs, complex(z))


# ==========
# parse mode
# ==========

def _parse_mode(use_reverse: str) -> int:

    mode = str(use_reverse).lower()
    if mode == 'auto':
        return MODE_AUTO
    if mode == 'direct':
        return MODE_DIRECT
    if mode == 'reverse':
        return MODE_REVERSE
    raise ValueError("use_reverse should be 'auto', 'direct', or 'reverse'.")


# =====
# roots
# =====

def roots(p: numpy.ndarray,
          *,
          trim_abs_tol: float = 0.0,
          trim_rel_tol: float = 0.0,
          use_reverse: str = 'auto',
          polish: bool = True,
          polish_iter: int = 20,
          polish_step_tol: float = 1e-13,
          polish_res_tol: float = 1e-13,
          sort_roots: bool = True,
          merge_atol: float = 1e-12,
          merge_rtol: float = 1e-8,
          polish_backtrack: int = 8) -> numpy.ndarray:

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
        float(merge_atol),
        float(merge_rtol),
        int(polish_backtrack))


# ==========
# roots many
# ==========

def roots_many(p_array: numpy.ndarray,
               *,
               trim_abs_tol: float = 0.0,
               trim_rel_tol: float = 0.0,
               use_reverse: str = 'auto',
               polish: bool = True,
               polish_iter: int = 20,
               polish_step_tol: float = 1e-13,
               polish_res_tol: float = 1e-13,
               sort_roots: bool = True,
               merge_atol: float = 1e-12,
               merge_rtol: float = 1e-8,
               polish_backtrack: int = 8) -> \
                    tuple[numpy.ndarray, numpy.ndarray]:

    p_array = numpy.asarray(p_array, dtype=numpy.complex128)
    if p_array.ndim != 2:
        raise ValueError('p_array should be a 2D array.')

    n_poly, n_coeff = p_array.shape
    roots_out = numpy.full(
        (n_poly, max(0, n_coeff - 1)),
        numpy.nan + 1j * numpy.nan,
        dtype=numpy.complex128)

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
            float(merge_atol),
            float(merge_rtol),
            int(polish_backtrack))

        roots_out[i, :r.size] = r
        n_eff[i] = r.size

    return roots_out, n_eff


# =======
# roots m
# =======

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
            sort_roots: bool = True,
            merge_atol: float = 1e-12,
            merge_rtol: float = 1e-8,
            polish_backtrack: int = 8) -> numpy.ndarray:

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
        float(merge_atol),
        float(merge_rtol),
        int(polish_backtrack))


# ============
# roots m many
# ============

def roots_m_many(coeffs: numpy.ndarray,
                 z_array: numpy.ndarray,
                 *,
                 trim_abs_tol: float = 0.0,
                 trim_rel_tol: float = 0.0,
                 use_reverse: str = 'auto',
                 polish: bool = True,
                 polish_iter: int = 20,
                 polish_step_tol: float = 1e-13,
                 polish_res_tol: float = 1e-13,
                 sort_roots: bool = True,
                 merge_atol: float = 1e-12,
                 merge_rtol: float = 1e-8,
                 polish_backtrack: int = 8) -> \
                         tuple[numpy.ndarray, numpy.ndarray]:

    coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
    z_array = numpy.asarray(z_array, dtype=numpy.complex128).ravel()
    mode = _parse_mode(use_reverse)

    deg_m = coeffs.shape[1] - 1
    roots_out = numpy.full(
        (z_array.size, max(0, deg_m)),
        numpy.nan + 1j * numpy.nan,
        dtype=numpy.complex128)

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
            float(merge_atol),
            float(merge_rtol),
            int(polish_backtrack))

        roots_out[i, :r.size] = r
        n_eff[i] = r.size

    return roots_out, n_eff


# ===================
# normalized residual
# ===================

def normalized_residual(p: numpy.ndarray, roots_: numpy.ndarray) -> \
        numpy.ndarray:

    p_desc = numpy.asarray(p, dtype=numpy.complex128).ravel()
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return _normalized_residuals_desc_numba(p_desc, roots_)


# =======================
# max normalized residual
# =======================

def max_normalized_residual(p: numpy.ndarray, roots_: numpy.ndarray) -> float:

    p_desc = numpy.asarray(p, dtype=numpy.complex128).ravel()
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_normalized_residual_desc_numba(p_desc, roots_))


# =====================
# normalized residual m
# =====================

def normalized_residual_m(coeffs: numpy.ndarray,
                          z: complex,
                          roots_: numpy.ndarray) -> numpy.ndarray:

    a_asc = poly_coeffs_in_m(coeffs, z)
    p_desc = a_asc[::-1]
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return _normalized_residuals_desc_numba(p_desc, roots_)


# =========================
# max normalized residual m
# =========================

def max_normalized_residual_m(coeffs: numpy.ndarray,
                              z: complex,
                              roots_: numpy.ndarray) -> float:

    a_asc = poly_coeffs_in_m(coeffs, z)
    p_desc = a_asc[::-1]
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_normalized_residual_desc_numba(p_desc, roots_))


# ==================
# max backward error
# ==================

def max_backward_error(p: numpy.ndarray, roots_: numpy.ndarray) -> float:

    p_desc = numpy.asarray(p, dtype=numpy.complex128).ravel()
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_backward_error_desc_numba(p_desc, roots_))


# ====================
# max backward error m
# ====================

def max_backward_error_m(coeffs: numpy.ndarray,
                         z: complex,
                         roots_: numpy.ndarray) -> float:

    a_asc = poly_coeffs_in_m(coeffs, z)
    p_desc = a_asc[::-1]
    roots_ = numpy.asarray(roots_, dtype=numpy.complex128).ravel()
    return float(_max_backward_error_desc_numba(p_desc, roots_))
