# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

"""
Pure vertical continuation for algebraic Stieltjes branches P(z,m)=0.

This version adds an optional empirical anchor mode:
  - asymptotic anchor at large |Im z|            (original behavior)
  - empirical anchor at moderate Im z from data  (new)
  - vertical predictor-corrector continuation in y
  - light local bisection only when a step is unreliable
  - arrays are evaluated pointwise by the same scalar solver
"""

# =======
# Imports
# =======

import numpy

try:
    from ._moments import AlgebraicStieltjesMoments
except Exception:
    AlgebraicStieltjesMoments = None

__all__ = ["StieltjesPoly", "_roots_m"]


# ================
# poly coeffs in m
# ================

def _poly_coeffs_in_m(coeffs, z):
    """
    """

    a = numpy.asarray(coeffs, dtype=numpy.complex128)
    dz = a.shape[0] - 1
    dm = a.shape[1] - 1
    z = complex(z)
    out = numpy.empty(dm + 1, dtype=numpy.complex128)

    for j in range(dm + 1):
        acc = 0.0 + 0.0j
        for i in range(dz, -1, -1):
            acc = acc * z + a[i, j]
        out[j] = acc

    return out


# =======
# roots m
# =======

def _roots_m(coeffs, z):
    """
    """

    coeff_m = _poly_coeffs_in_m(coeffs, z)
    c = coeff_m[::-1]
    tol = 1e-15
    while c.size > 1 and abs(c[0]) < tol:
        c = c[1:]
    if c.size <= 1:
        return numpy.array([], dtype=numpy.complex128)

    return numpy.roots(c)


# =========
# poly eval
# =========

def _poly_eval(coeffs, z, m):
    """
    """

    b = _poly_coeffs_in_m(coeffs, z)
    acc = 0.0 + 0.0j
    for j in range(b.size - 1, -1, -1):
        acc = acc * m + b[j]

    return acc


# ==========
# poly der m
# ==========

def _poly_der_m(coeffs, z, m):
    """
    """

    b = _poly_coeffs_in_m(coeffs, z)
    if b.size <= 1:
        return 0.0 + 0.0j
    acc = 0.0 + 0.0j

    for j in range(b.size - 1, 0, -1):
        acc = acc * m + j * b[j]

    return acc


# ================
# poly coeffs in z
# ================

def _poly_coeffs_in_z(coeffs, m):
    """
    """

    a = numpy.asarray(coeffs, dtype=numpy.complex128)
    dz = a.shape[0] - 1
    dm = a.shape[1] - 1
    m = complex(m)
    out = numpy.empty(dz + 1, dtype=numpy.complex128)

    for i in range(dz + 1):
        acc = 0.0 + 0.0j
        for j in range(dm, -1, -1):
            acc = acc * m + a[i, j]
        out[i] = acc

    return out


# ==========
# poly der z
# ==========

def _poly_der_z(coeffs, z, m):
    """
    """

    c = _poly_coeffs_in_z(coeffs, m)
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


# ==============
# newton correct
# ==============

def _newton_correct(coeffs, z, w0, *, tol=1e-12, max_iter=20,
                    min_pm=1e-12):
    """
    """

    w = complex(w0)
    for _ in range(int(max_iter)):
        f = _poly_eval(coeffs, z, w)
        fm = _poly_der_m(coeffs, z, w)

        if (not _is_finite_complex(f)) or (not _is_finite_complex(fm)):
            return w, False

        if abs(fm) < float(min_pm):
            return w, False

        dw = f / fm
        w = w - dw
        if abs(dw) <= float(tol) * (1.0 + abs(w)):
            return w, True

    f = _poly_eval(coeffs, z, w)
    ok = _is_finite_complex(f) and (abs(f) <= 100.0 * float(tol) *
                                    (1.0 + abs(w)))

    return w, ok


# =======
# ode rhs
# =======

def _ode_rhs(coeffs, z, w, *, min_pm=1e-12):
    """
    """

    pm = _poly_der_m(coeffs, z, w)
    if (not _is_finite_complex(pm)) or abs(pm) < float(min_pm):
        return numpy.nan + 1j * numpy.nan, False

    pz = _poly_der_z(coeffs, z, w)
    if not _is_finite_complex(pz):
        return numpy.nan + 1j * numpy.nan, False

    return -(pz / pm), True  # dm/dz


# ================
# vertical predict
# ================

def _vertical_predict(coeffs, x0, sgn, y_hist, w_hist, y_new, *,
                      min_pm=1e-12):
    """
    Adams-Bashforth AB2 in y when history is available, else Euler.
    """

    z_last = complex(x0, sgn * y_hist[-1])
    w_last = complex(w_hist[-1])
    mz_last, ok_last = _ode_rhs(coeffs, z_last, w_last, min_pm=min_pm)
    if not ok_last:
        return w_last, False

    f_last = 1j * sgn * mz_last  # dm/dy
    h = float(y_new - y_hist[-1])

    if len(y_hist) >= 2:
        z_prev = complex(x0, sgn * y_hist[-2])
        w_prev = complex(w_hist[-2])
        mz_prev, ok_prev = _ode_rhs(coeffs, z_prev, w_prev, min_pm=min_pm)

        if ok_prev:
            f_prev = 1j * sgn * mz_prev
            return w_last + h * (1.5 * f_last - 0.5 * f_prev), True

    return w_last + h * f_last, True


# =================
# continue one step
# =================

def _continue_one_step(coeffs, x0, sgn, y_hist, w_hist, y_new, *, tol_im,
                       lam_asym, newton_tol, newton_iter, min_pm,
                       pc_rel_tol, pc_abs_tol, max_subdivide, _depth=0):
    """
    Continue from the last accepted point to y_new with light local
    subdivision.
    """

    w_pred, ok_pred = _vertical_predict(
        coeffs, x0, sgn, y_hist, w_hist, y_new, min_pm=min_pm)

    if not ok_pred:
        w_pred = complex(w_hist[-1])

    z_new = complex(x0, sgn * y_new)
    w_corr, ok_newt = _newton_correct(
        coeffs, z_new, w_pred,
        tol=newton_tol, max_iter=newton_iter, min_pm=min_pm)

    if ok_newt and _is_finite_complex(w_corr):
        corr = abs(w_corr - w_pred)
        scale = 1.0 + abs(w_corr)
        if corr <= float(pc_abs_tol) + float(pc_rel_tol) * scale:
            if sgn * numpy.imag(w_corr) > -10.0 * float(tol_im):
                return w_corr, True

    if _depth < int(max_subdivide):
        y_mid = 0.5 * (float(y_hist[-1]) + float(y_new))
        w_mid, ok_mid = _continue_one_step(
            coeffs, x0, sgn, y_hist, w_hist, y_mid,
            tol_im=tol_im, lam_asym=lam_asym,
            newton_tol=newton_tol, newton_iter=newton_iter, min_pm=min_pm,
            pc_rel_tol=pc_rel_tol, pc_abs_tol=pc_abs_tol,
            max_subdivide=max_subdivide, _depth=_depth + 1)

        if ok_mid:
            y_hist2 = list(y_hist) + [y_mid]
            w_hist2 = list(w_hist) + [w_mid]

            return _continue_one_step(
                coeffs, x0, sgn, y_hist2, w_hist2, y_new,
                tol_im=tol_im, lam_asym=lam_asym,
                newton_tol=newton_tol, newton_iter=newton_iter,
                min_pm=min_pm,
                pc_rel_tol=pc_rel_tol, pc_abs_tol=pc_abs_tol,
                max_subdivide=max_subdivide, _depth=_depth + 1)

    roots = _roots_m(coeffs, z_new)
    target = w_pred if _is_finite_complex(w_pred) else w_hist[-1]
    w_pick = _pick_with_target(
        z_new, roots, target, tol_im=tol_im,
        lam_asym=lam_asym, lam_target=1.0)

    return w_pick, _is_finite_complex(w_pick)


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
        computed from ``emp_eigs``. Then continue vertically from that anchor
        to the requested target height.

    The empirical mode is intended precisely for cases where the correct sheet
    is present in the fitted polynomial, but asymptotic matching at large
    height selects the wrong family.
    """

    # ====
    # init
    # ====

    def __init__(self, coeffs, mom=None, *, stieltjes_opt=None, eps=None,
                 height=2.0, steps=80, order=15, emp_eigs=None,
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

        self.emp_eigs = None if emp_eigs is None else numpy.asarray(
            emp_eigs, dtype=self.rdtype)

        if anchor_mode is None:
            anchor_mode = self.stieltjes_opt.get("anchor_mode", "asymptotic")
        self.anchor_mode = str(anchor_mode)

    # ====================
    # set empirical anchor
    # ====================

    def set_empirical_anchor(
            self, eigs, *, anchor_y=None, tol_im=None,
            log_scale=None, anchor_ratio=None,
            anchor_y_min=None, anchor_y_max=None):
        """
        Enable empirical anchor mode after construction.

        This is useful when callers construct ``StieltjesPoly`` elsewhere and
        do not want to modify that code path. Example::

            af._stieltjes.set_empirical_anchor(eigs[start_idx], anchor_y=0.7)

        Parameters
        ----------

        eigs : array_like
            Eigenvalues of the empirical matrix.

        anchor_y : float, optional
            Moderate positive imaginary height used for the empirical anchor.

        tol_im : float, optional
            Imaginary-part admissibility tolerance at the anchor.
        """

        self.emp_eigs = numpy.asarray(eigs, dtype=self.rdtype)
        self.anchor_mode = "empirical"

        if anchor_y is not None:
            self.stieltjes_opt["anchor_y"] = float(anchor_y)

        if tol_im is not None:
            self.stieltjes_opt["emp_tol_im"] = float(tol_im)

        if log_scale is not None:
            self.stieltjes_opt["log_scale"] = bool(log_scale)

        if anchor_ratio is not None:
            self.stieltjes_opt["anchor_ratio"] = float(anchor_ratio)

        if anchor_y_min is not None:
            self.stieltjes_opt["anchor_y_min"] = float(anchor_y_min)

        if anchor_y_max is not None:
            self.stieltjes_opt["anchor_y_max"] = float(anchor_y_max)

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

        if self.emp_eigs is None:
            return numpy.nan + 1j * numpy.nan
        z = complex(z)

        return numpy.mean(1.0 / (self.emp_eigs - z))

    # ================
    # scalar from seed
    # ================

    def _scalar_from_seed(self, z_eval, y_seed, m_seed):
        """
        """

        opt = self._options()
        sgn = 1.0 if numpy.imag(z_eval) >= 0.0 else -1.0
        ys = self._levels_between(y_seed, abs(numpy.imag(z_eval)), opt)
        y_hist = [float(ys[0])]
        w_hist = [complex(m_seed)]

        for y_cur in ys[1:]:
            w_cur, ok = _continue_one_step(
                self.coeffs,
                numpy.real(z_eval),
                sgn,
                y_hist,
                w_hist,
                float(y_cur),
                tol_im=float(opt["tol_im"]),
                lam_asym=float(opt["lam_asym"]),
                newton_tol=float(opt["newton_tol"]),
                newton_iter=int(opt["newton_iter"]),
                min_pm=float(opt["min_pm"]),
                pc_rel_tol=float(opt["pc_rel_tol"]),
                pc_abs_tol=float(opt["pc_abs_tol"]),
                max_subdivide=int(opt["max_subdivide"]))

            if not ok:
                return w_cur
            y_hist.append(float(y_cur))
            w_hist.append(complex(w_cur))
            if len(y_hist) > 3:
                y_hist = y_hist[-3:]
                w_hist = w_hist[-3:]

        return w_hist[-1]

    # ===============
    # scalar from top
    # ===============

    def _scalar_from_top(self, z_eval):
        """
        """

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

        if not _is_finite_complex(m_top):
            return m_top

        return self._scalar_from_seed(z_eval, y_top, m_top)

    # ============================
    # scalar from empirical anchor
    # ============================

    def _scalar_from_empirical_anchor(self, z_eval):
        """
        """

        opt = self._options()
        if self.emp_eigs is None:
            return self._scalar_from_top(z_eval)

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
            return self._scalar_from_top(z_eval)

        z_anchor = complex(numpy.real(z_eval), sgn * y_anchor)

        roots = numpy.asarray(_roots_m(self.coeffs, z_anchor),
                              dtype=self.dtype)
        if roots.size == 0:
            return self._scalar_from_top(z_eval)

        ok = (sgn * numpy.imag(roots) > -float(opt["emp_tol_im"]))
        cand = roots[ok] if numpy.any(ok) else roots
        if cand.size == 0:
            return self._scalar_from_top(z_eval)

        m_emp = self._empirical_stieltjes(z_anchor)
        if not _is_finite_complex(m_emp):
            return self._scalar_from_top(z_eval)

        idx = int(numpy.argmin(numpy.abs(cand - m_emp)))
        m_seed = cand[idx]

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
        out = numpy.empty(z_flat.size, dtype=self.dtype)

        for k in range(z_flat.size):
            out[k] = self.evaluate_scalar(z_flat[k])

        return out.reshape(shp)
