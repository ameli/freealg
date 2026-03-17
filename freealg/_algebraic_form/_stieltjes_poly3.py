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

from ._roots import roots_m as _roots_backend_m
from ._roots import roots_m_numba as _roots_backend_m_numba
from ._roots import MODE_AUTO as _ROOTS_MODE_AUTO

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # pragma: no cover
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def deco(func):
            return func
        return deco

    def prange(*args):  # pragma: no cover
        return range(*args)

try:
    from ._moments import AlgebraicStieltjesMoments
except Exception:
    AlgebraicStieltjesMoments = None

__all__ = ["StieltjesPoly", "_roots_m"]


# ======================
# poly coeffs in m numba
# ======================

@njit(cache=True)
def _poly_coeffs_in_m_numba(coeffs, z):
    dz = coeffs.shape[0] - 1
    dm = coeffs.shape[1] - 1
    out = numpy.empty(dm + 1, dtype=numpy.complex128)

    for j in range(dm + 1):
        acc = 0.0 + 0.0j
        for i in range(dz, -1, -1):
            acc = acc * z + coeffs[i, j]
        out[j] = acc

    return out


# =======
# roots m
# =======

def _roots_m(coeffs, z):
    """
    """

    return _roots_backend_m(coeffs, z)


# ===============
# poly eval numba
# ===============

@njit(cache=True)
def _poly_eval_numba(coeffs, z, m):
    b = _poly_coeffs_in_m_numba(coeffs, z)
    acc = 0.0 + 0.0j
    for j in range(b.size - 1, -1, -1):
        acc = acc * m + b[j]

    return acc


# ================
# poly der m numba
# ================

@njit(cache=True)
def _poly_der_m_numba(coeffs, z, m):
    b = _poly_coeffs_in_m_numba(coeffs, z)
    if b.size <= 1:
        return 0.0 + 0.0j

    acc = 0.0 + 0.0j
    for j in range(b.size - 1, 0, -1):
        acc = acc * m + j * b[j]

    return acc


# ======================
# poly coeffs in z numba
# ======================

@njit(cache=True)
def _poly_coeffs_in_z_numba(coeffs, m):
    dz = coeffs.shape[0] - 1
    dm = coeffs.shape[1] - 1
    out = numpy.empty(dz + 1, dtype=numpy.complex128)

    for i in range(dz + 1):
        acc = 0.0 + 0.0j
        for j in range(dm, -1, -1):
            acc = acc * m + coeffs[i, j]
        out[i] = acc

    return out


# =================
# poly der z numbda
# =================

@njit(cache=True)
def _poly_der_z_numba(coeffs, z, m):
    c = _poly_coeffs_in_z_numba(coeffs, m)
    if c.size <= 1:
        return 0.0 + 0.0j

    acc = 0.0 + 0.0j
    for i in range(c.size - 1, 0, -1):
        acc = acc * z + i * c[i]

    return acc


# =================
# is finite complex
# =================

def _is_finite_complex(x):
    """
    """

    return numpy.isfinite(numpy.real(x)) and numpy.isfinite(numpy.imag(x))


# =======================
# is finite complex numba
# =======================

@njit(cache=True)
def _is_finite_complex_numba(x):
    return numpy.isfinite(numpy.real(x)) and numpy.isfinite(numpy.imag(x))


# ================
# pick with target
# ================

def _pick_with_target(z, roots, target, tol_im=1e-14, lam_asym=0.2,
                      lam_target=1.0):
    """
    """

    roots = numpy.asarray(roots, dtype=numpy.complex128).ravel()
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan

    sgn = 1.0 if numpy.imag(z) >= 0.0 else -1.0
    ok = (sgn * numpy.imag(roots) > -tol_im)
    cand = roots[ok] if numpy.any(ok) else roots
    cost = lam_asym * numpy.abs(z * cand + 1.0)

    if target is not None and _is_finite_complex(target):
        cost = cost + lam_target * numpy.abs(cand - target)

    return cand[int(numpy.argmin(cost))]


# ======================
# pick with target numba
# ======================

@njit(cache=True)
def _pick_with_target_numba(z, roots, target, tol_im=1e-14, lam_asym=0.2,
                            lam_target=1.0):
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan

    sgn = 1.0 if numpy.imag(z) >= 0.0 else -1.0
    best = roots[0]
    best_cost = numpy.inf
    found = False

    for i in range(roots.size):
        r = roots[i]
        if sgn * numpy.imag(r) > -tol_im:
            cost = lam_asym * abs(z * r + 1.0)
            if _is_finite_complex_numba(target):
                cost = cost + lam_target * abs(r - target)
            if (not found) or (cost < best_cost):
                best = r
                best_cost = cost
                found = True

    if found:
        return best

    best = roots[0]
    best_cost = numpy.inf
    for i in range(roots.size):
        r = roots[i]
        cost = lam_asym * abs(z * r + 1.0)
        if _is_finite_complex_numba(target):
            cost = cost + lam_target * abs(r - target)
        if cost < best_cost:
            best = r
            best_cost = cost

    return best


# ================
# roots pick numba
# ================

@njit(cache=True)
def _roots_pick_numba(coeffs, z_new, w_pred, w_last, tol_im, lam_asym):
    roots = _roots_backend_m_numba(coeffs, z_new, mode=_ROOTS_MODE_AUTO)
    target = w_pred if _is_finite_complex_numba(w_pred) else w_last
    w_pick = _pick_with_target_numba(
        z_new, roots, target, tol_im=tol_im,
        lam_asym=lam_asym, lam_target=1.0)

    return w_pick, _is_finite_complex_numba(w_pick)


# ====================
# newton correct numba
# ====================

@njit(cache=True)
def _newton_correct_numba(coeffs, z, w0, tol=1e-12, max_iter=20, min_pm=1e-12):

    w = w0
    for _ in range(int(max_iter)):
        f = _poly_eval_numba(coeffs, z, w)
        fm = _poly_der_m_numba(coeffs, z, w)

        if (not _is_finite_complex_numba(f)) or \
                (not _is_finite_complex_numba(fm)):
            return w, False

        if abs(fm) < min_pm:
            return w, False

        dw = f / fm
        w = w - dw
        if abs(dw) <= tol * (1.0 + abs(w)):
            return w, True

    f = _poly_eval_numba(coeffs, z, w)
    ok = _is_finite_complex_numba(f) and \
        (abs(f) <= 100.0 * tol * (1.0 + abs(w)))

    return w, ok


# =============
# ode rhs numba
# =============

@njit(cache=True)
def _ode_rhs_numba(coeffs, z, w, min_pm=1e-12):
    pm = _poly_der_m_numba(coeffs, z, w)
    if (not _is_finite_complex_numba(pm)) or abs(pm) < min_pm:
        return numpy.nan + 1j * numpy.nan, False

    pz = _poly_der_z_numba(coeffs, z, w)
    if not _is_finite_complex_numba(pz):
        return numpy.nan + 1j * numpy.nan, False

    return -(pz / pm), True


# ======================
# vertical predict numba
# ======================

@njit(cache=True)
def _vertical_predict_numba(coeffs, x0, sgn, y_hist, w_hist, hist_len, y_new,
                            min_pm=1e-12):
    z_last = complex(x0, sgn * y_hist[hist_len - 1])
    w_last = w_hist[hist_len - 1]
    mz_last, ok_last = _ode_rhs_numba(coeffs, z_last, w_last, min_pm=min_pm)
    if not ok_last:
        return w_last, False

    f_last = 1j * sgn * mz_last
    h = y_new - y_hist[hist_len - 1]

    if hist_len >= 2:
        z_prev = complex(x0, sgn * y_hist[hist_len - 2])
        w_prev = w_hist[hist_len - 2]
        mz_prev, ok_prev = _ode_rhs_numba(coeffs, z_prev, w_prev,
                                          min_pm=min_pm)

        if ok_prev:
            f_prev = 1j * sgn * mz_prev
            return w_last + h * (1.5 * f_last - 0.5 * f_prev), True

    return w_last + h * f_last, True


# =======================
# continue one step numba
# =======================

@njit(cache=True)
def _continue_one_step_numba(coeffs, x0, sgn, y_hist, w_hist, hist_len, y_new,
                             tol_im, lam_asym, newton_tol, newton_iter,
                             min_pm, pc_rel_tol, pc_abs_tol, max_subdivide,
                             log_scale, depth=0):
    w_pred, ok_pred = _vertical_predict_numba(
        coeffs, x0, sgn, y_hist, w_hist, hist_len, y_new, min_pm=min_pm)

    if not ok_pred:
        w_pred = w_hist[hist_len - 1]

    z_new = complex(x0, sgn * y_new)
    w_corr, ok_newt = _newton_correct_numba(
        coeffs, z_new, w_pred, tol=newton_tol, max_iter=newton_iter,
        min_pm=min_pm)

    if ok_newt and _is_finite_complex_numba(w_corr):
        corr = abs(w_corr - w_pred)

        if log_scale:
            scale = max(abs(w_corr), abs(w_pred), abs(w_hist[hist_len - 1]),
                        1.0)
            accept = (corr / scale) <= (pc_rel_tol + pc_abs_tol)
        else:
            scale = 1.0 + abs(w_corr)
            accept = corr <= pc_abs_tol + pc_rel_tol * scale

        if accept:
            pm_corr = _poly_der_m_numba(coeffs, z_new, w_corr)
            if _is_finite_complex_numba(pm_corr) and abs(pm_corr) >= min_pm:
                if sgn * numpy.imag(w_corr) > -tol_im:
                    return w_corr, True

    if depth < max_subdivide:
        y_mid = 0.5 * (y_hist[hist_len - 1] + y_new)
        w_mid, ok_mid = _continue_one_step_numba(
            coeffs, x0, sgn, y_hist, w_hist, hist_len, y_mid, tol_im,
            lam_asym, newton_tol, newton_iter, min_pm, pc_rel_tol,
            pc_abs_tol, max_subdivide, log_scale, depth + 1)

        if ok_mid:
            y_hist2 = numpy.empty(3, dtype=numpy.float64)
            w_hist2 = numpy.empty(3, dtype=numpy.complex128)

            if hist_len == 1:
                y_hist2[0] = y_hist[0]
                y_hist2[1] = y_mid
                w_hist2[0] = w_hist[0]
                w_hist2[1] = w_mid
                hist_len2 = 2
            elif hist_len == 2:
                y_hist2[0] = y_hist[0]
                y_hist2[1] = y_hist[1]
                y_hist2[2] = y_mid
                w_hist2[0] = w_hist[0]
                w_hist2[1] = w_hist[1]
                w_hist2[2] = w_mid
                hist_len2 = 3
            else:
                y_hist2[0] = y_hist[1]
                y_hist2[1] = y_hist[2]
                y_hist2[2] = y_mid
                w_hist2[0] = w_hist[1]
                w_hist2[1] = w_hist[2]
                w_hist2[2] = w_mid
                hist_len2 = 3

            return _continue_one_step_numba(
                coeffs, x0, sgn, y_hist2, w_hist2, hist_len2, y_new, tol_im,
                lam_asym, newton_tol, newton_iter, min_pm, pc_rel_tol,
                pc_abs_tol, max_subdivide, log_scale, depth + 1)

    w_pick, ok = _roots_pick_numba(
        coeffs, z_new, w_pred, w_hist[hist_len - 1], tol_im, lam_asym)

    return w_pick, ok


# ===================
# n levels auto numba
# ===================

@njit(cache=True)
def _n_levels_auto_numba(y_hi, y_lo):
    ratio = max(y_hi / max(y_lo, 1e-300), 1.0)
    return max(12, min(48, int(numpy.ceil(6.0 * numpy.log10(
        ratio + 1e-300))) + 8))


# =========================
# fill levels between numba
# =========================

@njit(cache=True)
def _fill_levels_between_numba(out, y_start, y_stop, n_levels):
    y_hi = max(y_start, y_stop)
    y_lo = min(y_start, y_stop)

    if n_levels <= 1 or numpy.isclose(y_start, y_stop):
        out[0] = y_start
        return 1

    if y_lo <= 0.0:
        step = (y_hi - y_lo) / (n_levels - 1)
        for i in range(n_levels):
            out[i] = y_hi - i * step
    else:
        lg_hi = numpy.log(y_hi)
        lg_lo = numpy.log(y_lo)
        for i in range(n_levels):
            t = i / (n_levels - 1)
            out[i] = numpy.exp((1.0 - t) * lg_hi + t * lg_lo)

    out[0] = y_hi
    out[n_levels - 1] = y_lo

    if y_start < y_stop:
        for i in range(n_levels // 2):
            tmp = out[i]
            out[i] = out[n_levels - 1 - i]
            out[n_levels - 1 - i] = tmp
        out[0] = y_start
        out[n_levels - 1] = y_stop
    else:
        out[0] = y_start
        out[n_levels - 1] = y_stop

    return n_levels


# ======================
# scalar from seed numba
# ======================

@njit(cache=True)
def _scalar_from_seed_numba(coeffs, x0, y_eval_abs, sgn, y_seed, m_seed,
                            tol_im, lam_asym, newton_tol, newton_iter,
                            min_pm, n_levels, pc_rel_tol, pc_abs_tol,
                            max_subdivide, log_scale):
    y_start = abs(y_seed)
    y_stop = abs(y_eval_abs)

    if numpy.isclose(y_start, y_stop):
        return m_seed

    n_use = n_levels
    if n_use <= 0:
        n_use = _n_levels_auto_numba(max(y_start, y_stop),
                                     min(y_start, y_stop))

    ys = numpy.empty(n_use, dtype=numpy.float64)
    n_use = _fill_levels_between_numba(ys, y_start, y_stop, n_use)

    y_hist = numpy.empty(3, dtype=numpy.float64)
    w_hist = numpy.empty(3, dtype=numpy.complex128)
    y_hist[0] = ys[0]
    w_hist[0] = m_seed
    hist_len = 1

    for i in range(1, n_use):
        y_cur = ys[i]
        w_cur, ok = _continue_one_step_numba(
            coeffs, x0, sgn, y_hist, w_hist, hist_len, y_cur, tol_im, lam_asym,
            newton_tol, newton_iter, min_pm, pc_rel_tol, pc_abs_tol,
            max_subdivide, log_scale, 0)

        if not ok:
            return w_cur

        if hist_len < 3:
            y_hist[hist_len] = y_cur
            w_hist[hist_len] = w_cur
            hist_len += 1
        else:
            y_hist[0] = y_hist[1]
            y_hist[1] = y_hist[2]
            y_hist[2] = y_cur
            w_hist[0] = w_hist[1]
            w_hist[1] = w_hist[2]
            w_hist[2] = w_cur

    return w_hist[hist_len - 1]


# =====================
# batch from seed numba
# =====================

@njit(cache=True, parallel=True)
def _batch_from_seed_numba(coeffs, x_arr, y_arr, sgn_arr, y_seed_arr,
                           m_seed_arr, tol_im, lam_asym, newton_tol,
                           newton_iter, min_pm, n_levels, pc_rel_tol,
                           pc_abs_tol, max_subdivide, log_scale, out):
    for k in prange(x_arr.size):
        out[k] = _scalar_from_seed_numba(
            coeffs, x_arr[k], abs(y_arr[k]), sgn_arr[k], y_seed_arr[k],
            m_seed_arr[k], tol_im, lam_asym, newton_tol, newton_iter, min_pm,
            n_levels, pc_rel_tol, pc_abs_tol, max_subdivide, log_scale)


# ==============
# Stieltjes Poly
# ==============

class StieltjesPoly(object):
    """Callable m(z) for P(z,m)=0 using pure vertical continuation.

    Notes
    -----
    Two anchor modes are available.

    ``anchor_mode='asymptotic'``
        Original behavior. Start from large imaginary height and choose the
        top root by asymptotic moment matching.

    ``anchor_mode='empirical'``
        New behavior. At a moderate imaginary height ``anchor_y``, choose the
        polynomial root closest to the empirical finite-n Stieltjes transform
        computed from ``stieltjes_emp``. Then continue vertically from that
        anchor to the requested target height.

    The empirical mode is intended precisely for cases where the correct sheet
    is present in the fitted polynomial, but asymptotic matching at large
    height selects the wrong family.
    """

    # ====
    # init
    # ====

    def __init__(self, coeffs, mom=None, *, stieltjes_opt=None, eps=None,
                 height=2.0, steps=80, order=15, stieltjes_emp=None,
                 anchor_mode=None, dtype=complex):

        self.dtype = dtype
        self.rdtype = numpy.empty((), dtype=self.dtype).real.dtype

        self.coeffs = numpy.asarray(coeffs, dtype=self.dtype)
        if self.coeffs.ndim != 2:
            raise ValueError("coeffs must be a 2D array.")

        self.stieltjes_opt = dict(stieltjes_opt or {})
        self.eps = eps
        self.height = float(height)
        self.steps = int(steps)
        self.order = int(order)

        self._mom = mom
        self._rad = None
        self._m0p = None
        self._m0m = None
        self.stieltjes_emp = stieltjes_emp

        if anchor_mode is None:
            anchor_mode = self.stieltjes_opt.get("anchor_mode", "asymptotic")
        self.anchor_mode = str(anchor_mode)

    # ==============
    # ensure moments
    # ==============

    def _ensure_moments(self):
        """
        """

        if self._rad is not None:
            return

        if self._mom is None:
            if AlgebraicStieltjesMoments is None:
                self._rad = numpy.inf
                self._m0p = None
                self._m0m = None
                return
            self._mom = AlgebraicStieltjesMoments(self.coeffs)

        try:
            if hasattr(self._mom, "radius") and \
                    hasattr(self._mom, "stieltjes"):
                self._rad = 1.0 + self.height * \
                    float(self._mom.radius(self.order))
                self._m0p = self._mom.stieltjes(1j * self._rad, self.order)
                self._m0m = self._mom.stieltjes(-1j * self._rad, self.order)
                return
        except Exception:
            pass

        try:
            mu = numpy.array([self._mom(j) for j in range(self.order + 1)],
                             dtype=self.dtype)
            ratios = []
            for j in range(2, self.order + 1):
                den = mu[j - 1]
                if den != 0:
                    ratios.append(abs(mu[j] / den))
            rad0 = max(ratios) if ratios else 1.0
            self._rad = 1.0 + self.height * float(rad0)
        except Exception:
            self._rad = numpy.inf
            self._m0p = None
            self._m0m = None

    # ===========
    # moments est
    # ===========

    def _moment_est(self, z):
        """
        """

        self._ensure_moments()
        if self._mom is None:
            return -1.0 / complex(z)
        try:
            if hasattr(self._mom, "stieltjes"):
                return self._mom.stieltjes(z, self.order)
        except Exception:
            pass
        try:
            mu = numpy.array([self._mom(j) for j in range(self.order + 1)],
                             dtype=self.dtype)
            z = complex(z)
            return -numpy.sum(mu * (z ** (-numpy.arange(self.order + 1) - 1)))
        except Exception:
            return -1.0 / complex(z)

    # ===========
    # normalize z
    # ===========

    def _normalize_z(self, z):
        """
        """

        z = complex(z)
        if z.imag == 0.0:
            eps_loc = float(self.eps) if self.eps is not None else (
                1e-8 * max(1.0, abs(z)))

            return z + 1j * eps_loc
        return z

    # =======
    # options
    # =======

    def _options(self):
        """
        """

        opt = {
            "tol_im": 1e-14,
            "lam_asym": 0.15,
            "newton_tol": 1e-12,
            "newton_iter": 20,
            "min_pm": 1e-12,
            "n_levels": None,
            "top_factor": 1.5,
            "pc_rel_tol": 0.20,
            "pc_abs_tol": 1e-10,
            "max_subdivide": 2,
            "anchor_mode": self.anchor_mode,
            "anchor_y": 0.7,
            "emp_tol_im": 1e-14,
            "log_scale": False,
            "anchor_ratio": 1.0,
            "anchor_y_min": 1e-6,
            "anchor_y_max": 10.0,
        }

        opt.update(self.stieltjes_opt)
        return opt

    # ==========
    # top height
    # ==========

    def _top_height(self, z_eval, opt):
        """
        """

        self._ensure_moments()
        rad = self._rad

        if rad is None or (not numpy.isfinite(rad)):
            rad = 1.0 + self.height * max(1.0, abs(z_eval))

        return max(float(abs(numpy.imag(z_eval)) * opt["top_factor"]),
                   float(abs(rad)))

    # ==============
    # levels between
    # ==============

    def _levels_between(self, y_start, y_stop, opt):
        """
        """

        y_start = float(abs(y_start))
        y_stop = float(abs(y_stop))
        if numpy.isclose(y_start, y_stop):
            return numpy.array([y_start], dtype=self.rdtype)

        y_hi = max(y_start, y_stop)
        y_lo = min(y_start, y_stop)

        n_levels = opt["n_levels"]
        if n_levels is None:
            ratio = max(y_hi / max(y_lo, 1e-300), 1.0)
            n_levels = max(
                12, min(48, int(
                    numpy.ceil(6.0 * numpy.log10(ratio + 1e-300))) + 8))
        else:
            n_levels = max(2, int(n_levels))

        lev = numpy.geomspace(y_hi, y_lo, n_levels)
        lev[0] = y_hi
        lev[-1] = y_lo
        if y_start < y_stop:
            lev = lev[::-1]
            lev[0] = y_start
            lev[-1] = y_stop
        else:
            lev[0] = y_start
            lev[-1] = y_stop
        return lev

    # ===================
    # empirical stieltjes
    # ===================

    def _empirical_stieltjes(self, z):
        """
        """

        z = complex(z)
        return self.stieltjes_emp(numpy.array([z], dtype=self.dtype))[0]

    # ========
    # top seed
    # ========

    def _top_seed(self, z_eval):
        opt = self._options()
        sgn = 1.0 if numpy.imag(z_eval) >= 0.0 else -1.0
        y_top = self._top_height(z_eval, opt)
        z_top = complex(numpy.real(z_eval), sgn * y_top)

        r_top = _roots_m(self.coeffs, z_top)
        m_top = _pick_with_target(
            z_top, r_top, self._moment_est(z_top),
            tol_im=float(opt["tol_im"]),
            lam_asym=float(opt["lam_asym"]),
            lam_target=2.0)

        return y_top, m_top

    # ==============
    # empirical seed
    # ==============

    def _empirical_seed(self, z_eval):
        opt = self._options()
        if self.stieltjes_emp is None:
            return self._top_seed(z_eval)

        sgn = 1.0 if numpy.imag(z_eval) >= 0.0 else -1.0
        x_abs = float(abs(numpy.real(z_eval)))

        if bool(opt.get("log_scale", False)):
            y_anchor = float(opt["anchor_ratio"]) * \
                max(x_abs, float(self.eps or 0.0))
            y_anchor = max(y_anchor, float(opt["anchor_y_min"]))
            y_anchor = min(y_anchor, float(opt["anchor_y_max"]))
        else:
            y_anchor = float(opt["anchor_y"])

        if y_anchor <= 0.0:
            return self._top_seed(z_eval)

        z_anchor = complex(numpy.real(z_eval), sgn * y_anchor)

        roots = numpy.asarray(_roots_m(self.coeffs, z_anchor),
                              dtype=self.dtype)
        if roots.size == 0:
            return self._top_seed(z_eval)

        ok = (sgn * numpy.imag(roots) > -float(opt["emp_tol_im"]))
        cand = roots[ok] if numpy.any(ok) else roots
        if cand.size == 0:
            return self._top_seed(z_eval)

        m_emp = self._empirical_stieltjes(z_anchor)
        if not _is_finite_complex(m_emp):
            return self._top_seed(z_eval)

        idx = int(numpy.argmin(numpy.abs(cand - m_emp)))
        m_seed = cand[idx]

        return y_anchor, m_seed

    # ================
    # scalar from seed
    # ================

    def _scalar_from_seed(self, z_eval, y_seed, m_seed):
        """
        """

        opt = self._options()
        sgn = 1.0 if numpy.imag(z_eval) >= 0.0 else -1.0
        n_levels = opt["n_levels"]
        if n_levels is None:
            n_levels = 0

        lam_asym = 0.0 if str(opt.get("anchor_mode", self.anchor_mode)) == \
            "empirical" else float(opt["lam_asym"])

        return _scalar_from_seed_numba(
            numpy.asarray(self.coeffs, dtype=numpy.complex128),
            float(numpy.real(z_eval)),
            float(abs(numpy.imag(z_eval))),
            float(sgn),
            float(y_seed),
            complex(m_seed),
            float(opt["tol_im"]),
            lam_asym,
            float(opt["newton_tol"]),
            int(opt["newton_iter"]),
            float(opt["min_pm"]),
            int(n_levels),
            float(opt["pc_rel_tol"]),
            float(opt["pc_abs_tol"]),
            int(opt["max_subdivide"]),
            bool(opt.get("log_scale", False)))

    # ===============
    # scalar from top
    # ===============

    def _scalar_from_top(self, z_eval):
        """
        """

        y_top, m_top = self._top_seed(z_eval)

        if not _is_finite_complex(m_top):
            return m_top

        return self._scalar_from_seed(z_eval, y_top, m_top)

    # ============================
    # scalar from empirical anchor
    # ============================

    def _scalar_from_empirical_anchor(self, z_eval):
        """
        """

        y_anchor, m_seed = self._empirical_seed(z_eval)

        if not _is_finite_complex(m_seed):
            return m_seed

        return self._scalar_from_seed(z_eval, y_anchor, m_seed)

    # ===============
    # evaluate scalar
    # ===============

    def evaluate_scalar(self, z, target=None):
        """
        """

        del target
        z_eval = self._normalize_z(z)
        mode = str(self._options().get("anchor_mode", self.anchor_mode))

        if mode == "empirical":
            return self._scalar_from_empirical_anchor(z_eval)

        return self._scalar_from_top(z_eval)

    # ==============
    # evaluate array
    # ==============

    def _evaluate_array(self, z_flat):
        z_flat = numpy.asarray(z_flat, dtype=self.dtype)
        z_norm = numpy.empty(z_flat.size, dtype=self.dtype)
        for k in range(z_flat.size):
            z_norm[k] = self._normalize_z(z_flat[k])

        mode = str(self._options().get("anchor_mode", self.anchor_mode))
        y_seed = numpy.empty(z_norm.size, dtype=self.rdtype)
        m_seed = numpy.empty(z_norm.size, dtype=self.dtype)

        for k in range(z_norm.size):
            if mode == "empirical":
                y_seed[k], m_seed[k] = self._empirical_seed(z_norm[k])
            else:
                y_seed[k], m_seed[k] = self._top_seed(z_norm[k])

        bad = ~numpy.isfinite(m_seed.real) | ~numpy.isfinite(m_seed.imag)
        out = numpy.empty(z_norm.size, dtype=self.dtype)

        if numpy.any(bad):
            for k in range(z_norm.size):
                if bad[k]:
                    out[k] = m_seed[k]

        good = ~bad
        if not numpy.any(good):
            return out

        opt = self._options()
        n_levels = opt["n_levels"]
        if n_levels is None:
            n_levels = 0

        zg = z_norm[good]
        out_good = numpy.empty(zg.size, dtype=self.dtype)
        sgn = numpy.where(numpy.imag(zg) >= 0.0, 1.0, -1.0).astype(self.rdtype)

        lam_asym = 0.0 if mode == "empirical" else float(opt["lam_asym"])

        _batch_from_seed_numba(
            numpy.asarray(self.coeffs, dtype=numpy.complex128),
            numpy.real(zg).astype(self.rdtype),
            numpy.imag(zg).astype(self.rdtype),
            sgn,
            y_seed[good].astype(self.rdtype),
            m_seed[good].astype(self.dtype),
            float(opt["tol_im"]),
            lam_asym,
            float(opt["newton_tol"]),
            int(opt["newton_iter"]),
            float(opt["min_pm"]),
            int(n_levels),
            float(opt["pc_rel_tol"]),
            float(opt["pc_abs_tol"]),
            int(opt["max_subdivide"]),
            bool(opt.get("log_scale", False)),
            out_good)

        out[good] = out_good
        return out

    # ====
    # call
    # ====

    def __call__(self, z):
        """
        """

        z_arr = numpy.asarray(z, dtype=self.dtype)
        scalar = (z_arr.ndim == 0)

        if scalar:
            return numpy.asarray(
                self.evaluate_scalar(complex(z_arr)), dtype=self.dtype)

        shp = z_arr.shape
        z_flat = z_arr.ravel()
        out = self._evaluate_array(z_flat)

        return out.reshape(shp)
