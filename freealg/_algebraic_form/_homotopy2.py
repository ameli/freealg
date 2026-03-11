
# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# Robust Stieltjes branch evaluation for algebraic P(z,m)=0 using
# 2D sheet continuation in the upper/lower half-plane.
#
# _homotopy8.py: cleaned version of _homotopy7.py,
# but for 1D horizontal lines it no longer treats root selection as
# a single-line labeling problem. Instead, it grows the physical
# Herglotz sheet from a safe high-imaginary row down to the target row,
# with adaptive predictor-corrector continuation and row-wise repair.

import numpy

try:
    from ._moments import AlgebraicStieltjesMoments
except Exception:  # pragma: no cover
    AlgebraicStieltjesMoments = None


__all__ = ["StieltjesPoly"]


def _poly_coeffs_in_m(coeffs, z):
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


def _roots_m(coeffs, z):
    coeff_m = _poly_coeffs_in_m(coeffs, z)
    c = coeff_m[::-1]
    tol = 1e-15
    while c.size > 1 and abs(c[0]) < tol:
        c = c[1:]
    if c.size <= 1:
        return numpy.array([], dtype=numpy.complex128)
    return numpy.roots(c)


def _poly_eval(coeffs, z, m):
    b = _poly_coeffs_in_m(coeffs, z)
    acc = 0.0 + 0.0j
    for j in range(b.size - 1, -1, -1):
        acc = acc * m + b[j]
    return acc


def _poly_der_m(coeffs, z, m):
    b = _poly_coeffs_in_m(coeffs, z)
    if b.size <= 1:
        return 0.0 + 0.0j
    acc = 0.0 + 0.0j
    for j in range(b.size - 1, 0, -1):
        acc = acc * m + j * b[j]
    return acc


def _poly_coeffs_in_z(coeffs, m):
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


def _poly_der_z(coeffs, z, m):
    c = _poly_coeffs_in_z(coeffs, m)
    if c.size <= 1:
        return 0.0 + 0.0j
    acc = 0.0 + 0.0j
    for i in range(c.size - 1, 0, -1):
        acc = acc * z + i * c[i]
    return acc


def _is_finite_complex(x):
    return numpy.isfinite(numpy.real(x)) and numpy.isfinite(numpy.imag(x))



def _pick_with_target(z, roots, target, tol_im=1e-14, lam_asym=0.2, lam_target=1.0):
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


def _newton_correct(coeffs, z, w0, *, tol=1e-12, max_iter=20, min_pm=1e-12):
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
    ok = _is_finite_complex(f) and (abs(f) <= 100.0 * float(tol) * (1.0 + abs(w)))
    return w, ok


def _predict_step(coeffs, z, w, dz, *, min_pm=1e-12):
    pm = _poly_der_m(coeffs, z, w)
    if (not _is_finite_complex(pm)) or abs(pm) < float(min_pm):
        return complex(w), False
    pz = _poly_der_z(coeffs, z, w)
    if not _is_finite_complex(pz):
        return complex(w), False
    return complex(w) - (pz / pm) * dz, True


def _attempt_continuation(coeffs, z1, w_ref, *, tol_im=1e-14, lam_asym=0.1,
                          newton_tol=1e-12, newton_iter=20, min_pm=1e-12):
    w_newton, ok = _newton_correct(
        coeffs, z1, w_ref, tol=newton_tol, max_iter=newton_iter, min_pm=min_pm
    )
    if ok:
        sgn = 1.0 if numpy.imag(z1) >= 0.0 else -1.0
        if sgn * numpy.imag(w_newton) > -10.0 * tol_im:
            return w_newton, True

    roots = _roots_m(coeffs, z1)
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan, False

    w_pick = _pick_with_target(
        z1, roots, target=w_ref, tol_im=tol_im, lam_asym=lam_asym, lam_target=1.0
    )
    return w_pick, _is_finite_complex(w_pick)


def _continue_segment(coeffs, z0, w0, z1, *, tol_im=1e-14, lam_asym=0.1,
                      newton_tol=1e-12, newton_iter=20, min_pm=1e-12,
                      max_depth=12, jump_factor=10.0, _depth=0):
    dz = complex(z1) - complex(z0)
    w_pred, used_pred = _predict_step(coeffs, z0, w0, dz, min_pm=min_pm)
    if not used_pred:
        w_pred = complex(w0)

    w1, ok = _attempt_continuation(
        coeffs,
        z1,
        w_pred,
        tol_im=tol_im,
        lam_asym=lam_asym,
        newton_tol=newton_tol,
        newton_iter=newton_iter,
        min_pm=min_pm,
    )
    if ok and _is_finite_complex(w1):
        jump = abs(w1 - w0)
        scale = 1.0 + abs(w0) + abs(w_pred)
        if (jump <= float(jump_factor) * scale) or (_depth >= int(max_depth)):
            return w1, True

    if _depth >= int(max_depth):
        roots = _roots_m(coeffs, z1)
        if roots.size == 0:
            return numpy.nan + 1j * numpy.nan, False
        return _pick_with_target(
            z1, roots, target=w0, tol_im=tol_im, lam_asym=lam_asym, lam_target=1.0
        ), True

    z_mid = 0.5 * (complex(z0) + complex(z1))
    w_mid, ok_mid = _continue_segment(
        coeffs,
        z0,
        w0,
        z_mid,
        tol_im=tol_im,
        lam_asym=lam_asym,
        newton_tol=newton_tol,
        newton_iter=newton_iter,
        min_pm=min_pm,
        max_depth=max_depth,
        jump_factor=jump_factor,
        _depth=_depth + 1,
    )
    if not ok_mid:
        return w_mid, False

    return _continue_segment(
        coeffs,
        z_mid,
        w_mid,
        z1,
        tol_im=tol_im,
        lam_asym=lam_asym,
        newton_tol=newton_tol,
        newton_iter=newton_iter,
        min_pm=min_pm,
        max_depth=max_depth,
        jump_factor=jump_factor,
        _depth=_depth + 1,
    )


def _viterbi_row(z_list, roots_all, *, anchor=None, prev_row=None, tol_im=1e-14,
                 lam_space=1.0, lam_asym=0.25, lam_anchor=2.0, lam_prev=2.0):
    n, s = roots_all.shape
    big = 1.0e300
    dp = numpy.full((n, s), big, dtype=float)
    back = numpy.zeros((n, s), dtype=numpy.int64)
    unary = numpy.full((n, s), big, dtype=float)

    for k in range(n):
        z = z_list[k]
        r = roots_all[k]
        sgn = 1.0 if numpy.imag(z) >= 0.0 else -1.0
        good = numpy.isfinite(r) & (sgn * numpy.imag(r) > -tol_im)

        if numpy.any(good):
            idx = numpy.where(good)[0]
        else:
            idx = numpy.where(numpy.isfinite(r))[0]

        if idx.size == 0:
            continue

        vals = r[idx]
        cost = lam_asym * numpy.abs(z * vals + 1.0)
        if anchor is not None and _is_finite_complex(anchor[k]):
            cost = cost + lam_anchor * numpy.abs(vals - anchor[k])
        if prev_row is not None and _is_finite_complex(prev_row[k]):
            cost = cost + lam_prev * numpy.abs(vals - prev_row[k])

        unary[k, idx] = cost

    dp[0] = unary[0]

    for k in range(1, n):
        rk = roots_all[k]
        rp = roots_all[k - 1]
        for j in range(s):
            if not numpy.isfinite(unary[k, j]):
                continue
            trans = dp[k - 1] + lam_space * numpy.abs(rk[j] - rp)
            jj = int(numpy.argmin(trans))
            dp[k, j] = unary[k, j] + trans[jj]
            back[k, j] = jj

    path = numpy.empty(n, dtype=numpy.complex128)
    path[:] = numpy.nan + 1j * numpy.nan

    jn = int(numpy.argmin(dp[-1]))
    if not numpy.isfinite(dp[-1, jn]):
        return path

    for k in range(n - 1, -1, -1):
        path[k] = roots_all[k, jn]
        if k > 0:
            jn = int(back[k, jn])
    return path


def _is_flat_imag_line_1d(z):
    z = numpy.asarray(z, dtype=numpy.complex128)
    if z.ndim != 1 or z.size < 2:
        return False
    y = numpy.imag(z)
    if not numpy.all(numpy.isfinite(y)):
        return False
    y0 = float(y[0])
    if y0 == 0.0:
        return False
    if not numpy.all(numpy.sign(y) == numpy.sign(y0)):
        return False
    tol = 1e-14 * max(1.0, abs(y0))
    return numpy.max(numpy.abs(y - y0)) <= tol


class StieltjesPoly(object):
    """Callable m(z) for P(z,m)=0 using 2D sheet continuation."""

    def __init__(self, coeffs, mom=None, *, stieltjes_opt=None, eps=None,
                 height=2.0, steps=80, order=15):
        self.coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
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

    def _ensure_moments(self):
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
            if hasattr(self._mom, "radius") and hasattr(self._mom, "stieltjes"):
                self._rad = 1.0 + self.height * float(self._mom.radius(self.order))
                self._m0p = self._mom.stieltjes(1j * self._rad, self.order)
                self._m0m = self._mom.stieltjes(-1j * self._rad, self.order)
                return
        except Exception:
            pass

        try:
            mu = numpy.array([self._mom(j) for j in range(self.order + 1)],
                             dtype=numpy.complex128)
            ratios = []
            for j in range(2, self.order + 1):
                den = mu[j - 1]
                if den != 0:
                    ratios.append(abs(mu[j] / den))
            rad0 = max(ratios) if ratios else 1.0
            self._rad = 1.0 + self.height * float(rad0)
            zp = 1j * self._rad
            zm = -1j * self._rad
            pows = -numpy.arange(self.order + 1) - 1
            self._m0p = -numpy.sum(mu * (zp ** pows))
            self._m0m = -numpy.sum(mu * (zm ** pows))
        except Exception:
            self._rad = numpy.inf
            self._m0p = None
            self._m0m = None

    def _moment_est(self, z):
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
                             dtype=numpy.complex128)
            z = complex(z)
            return -numpy.sum(mu * (z ** (-numpy.arange(self.order + 1) - 1)))
        except Exception:
            return -1.0 / complex(z)

    def _normalize_z(self, z):
        z = complex(z)
        if z.imag == 0.0:
            eps_loc = float(self.eps) if self.eps is not None else 1e-8 * max(1.0, abs(z))
            return z + 1j * eps_loc
        return z

    def _options(self):
        opt = {
            "tol_im": 1e-14,
            "lam_asym": 0.15,
            "lam_space": 1.0,
            "lam_anchor": 3.0,
            "lam_prev": 3.0,
            "newton_tol": 1e-12,
            "newton_iter": 20,
            "min_pm": 1e-12,
            "max_depth": 12,
            "jump_factor": 10.0,
            "n_levels": None,
            "top_factor": 1.5,
            "row_refine": True,
            "refine_passes": 2,
            "use_viterbi_top": True,
            "use_viterbi_rows": True,
            "fallback_viterbi_1d": True,
        }
        opt.update(self.stieltjes_opt)
        return opt

    def _top_height(self, z_eval):
        self._ensure_moments()
        rad = self._rad
        if rad is None or (not numpy.isfinite(rad)):
            rad = 1.0 + self.height * max(1.0, abs(z_eval))
        return max(float(abs(numpy.imag(z_eval)) * self._options()["top_factor"]),
                   float(abs(rad)))

    def _levels(self, y0, y1, opt):
        y0 = float(abs(y0))
        y1 = float(abs(y1))
        if y0 <= y1:
            return numpy.array([y0], dtype=float)

        n_levels = opt["n_levels"]
        if n_levels is None:
            base = max(6, int(numpy.ceil(numpy.log2(max(y0 / max(y1, 1e-300), 1.0)))) + 3)
            n_levels = max(base, min(24, max(8, self.steps // 8)))
        else:
            n_levels = max(2, int(n_levels))

        lev = numpy.geomspace(y0, y1, n_levels)
        lev[0] = y0
        lev[-1] = y1
        return lev

    def _scalar_from_top(self, z_eval):
        opt = self._options()
        sgn = 1.0 if numpy.imag(z_eval) >= 0.0 else -1.0
        y_top = self._top_height(z_eval)
        z_top = complex(numpy.real(z_eval), sgn * y_top)

        r_top = _roots_m(self.coeffs, z_top)
        m_top = _pick_with_target(
            z_top, r_top, self._moment_est(z_top),
            tol_im=float(opt["tol_im"]),
            lam_asym=float(opt["lam_asym"]),
            lam_target=float(opt["lam_anchor"]),
        )

        if not _is_finite_complex(m_top):
            return m_top

        ys = self._levels(y_top, abs(numpy.imag(z_eval)), opt)
        w = m_top
        z_prev = z_top

        for y in ys[1:]:
            z_cur = complex(numpy.real(z_eval), sgn * float(y))
            w_prev = w

            w, ok = _continue_segment(
                self.coeffs, z_prev, w_prev, z_cur,
                tol_im=float(opt["tol_im"]),
                lam_asym=float(opt["lam_asym"]),
                newton_tol=float(opt["newton_tol"]),
                newton_iter=int(opt["newton_iter"]),
                min_pm=float(opt["min_pm"]),
                max_depth=int(opt["max_depth"]),
                jump_factor=float(opt["jump_factor"]),
            )

            roots = _roots_m(self.coeffs, z_cur)

            if (not ok) or (not _is_finite_complex(w)):
                w = _pick_with_target(
                    z_cur, roots, w_prev,
                    tol_im=float(opt["tol_im"]),
                    lam_asym=float(opt["lam_asym"]),
                    lam_target=1.0,
                )

            # Guard against collapsing from a genuinely bulk branch
            # (moderate/large Im m) onto a nearly-real branch during descent.
            if roots.size > 0 and _is_finite_complex(w_prev) and _is_finite_complex(w):
                im_prev = float(sgn * numpy.imag(w_prev))
                im_cur = float(sgn * numpy.imag(w))

                drop_ratio = float(opt.get("guard_imag_drop_ratio", 0.05))
                alt_imag_ratio = float(opt.get("guard_alt_imag_ratio", 0.25))
                target_relax = float(opt.get("guard_target_relax", 4.0))

                if (im_prev > 1e-12) and (im_cur < drop_ratio * im_prev):
                    cand = numpy.asarray(roots, dtype=numpy.complex128).ravel()
                    good = cand[(sgn * numpy.imag(cand)) > -float(opt["tol_im"])]

                    if good.size > 0:
                        dist = numpy.abs(good - w_prev)
                        best_dist = float(numpy.min(dist))
                        keep = (
                            (sgn * numpy.imag(good) >= alt_imag_ratio * im_prev) &
                            (dist <= target_relax * (best_dist + 1e-30))
                        )

                        if numpy.any(keep):
                            good2 = good[keep]
                            # Among plausible continuations, prefer the one that
                            # stays on the high-imag side rather than collapsing.
                            idx = int(numpy.argmax(sgn * numpy.imag(good2)))
                            w_alt = good2[idx]

                            # Refine on that branch if possible.
                            w_newt, ok_newt = _newton_correct(
                                self.coeffs, z_cur, w_alt,
                                tol=float(opt["newton_tol"]),
                                max_iter=int(opt["newton_iter"]),
                                min_pm=float(opt["min_pm"]),
                            )
                            if ok_newt and _is_finite_complex(w_newt):
                                w = w_newt
                            else:
                                w = w_alt

            # Bottom/bulk rescue:
            # if current branch is nearly real but there is a positive-imag
            # candidate with similar real part and much larger Im, switch to it.
            if roots.size > 0 and _is_finite_complex(w):
                cand = numpy.asarray(roots, dtype=numpy.complex128).ravel()
                good = cand[(sgn * numpy.imag(cand)) > float(opt["tol_im"])]

                if good.size > 0:
                    im_cur = float(sgn * numpy.imag(w))
                    tiny_im = float(opt.get("rescue_tiny_im", 1e-3))
                    min_big = float(opt.get("rescue_min_big_im", 1e-1))
                    real_window = float(opt.get("rescue_real_window", 5.0))
                    imag_ratio = float(opt.get("rescue_imag_ratio", 20.0))

                    if im_cur < tiny_im:
                        re_diff = numpy.abs(numpy.real(good) - numpy.real(w))
                        im_good = sgn * numpy.imag(good)

                        keep = (
                            (re_diff <= real_window) &
                            (im_good >= min_big) &
                            (im_good >= imag_ratio * max(im_cur, 1e-30))
                        )

                        if numpy.any(keep):
                            good2 = good[keep]
                            idx = int(numpy.argmin(numpy.abs(numpy.real(good2) - numpy.real(w))))
                            w_alt = good2[idx]

                            w_newt, ok_newt = _newton_correct(
                                self.coeffs, z_cur, w_alt,
                                tol=float(opt["newton_tol"]),
                                max_iter=int(opt["newton_iter"]),
                                min_pm=float(opt["min_pm"]),
                            )
                            if ok_newt and _is_finite_complex(w_newt):
                                w = w_newt
                            else:
                                w = w_alt

            z_prev = z_cur
        #----------------------

        return w

    def evaluate_scalar(self, z, target=None):
        z_eval = self._normalize_z(z)
        try:
            return self._scalar_from_top(z_eval)
        except Exception:
            roots = _roots_m(self.coeffs, z_eval)
            return _pick_with_target(
                z_eval, roots,
                target if target is not None else self._moment_est(z_eval),
                tol_im=float(self._options()["tol_im"]),
                lam_asym=float(self._options()["lam_asym"]),
                lam_target=1.0,
            )

    def _top_row(self, x, sgn, y_top, opt):
        z_top = x.astype(numpy.complex128) + 1j * sgn * float(y_top)
        s = self.coeffs.shape[1] - 1
        roots_all = numpy.full((x.size, s), numpy.nan + 1j * numpy.nan, dtype=numpy.complex128)
        anchor = numpy.empty(x.size, dtype=numpy.complex128)

        for k in range(x.size):
            roots = _roots_m(self.coeffs, z_top[k])
            roots_all[k, :min(s, roots.size)] = roots[:min(s, roots.size)]
            anchor[k] = self._moment_est(z_top[k])

        if bool(opt.get("use_viterbi_top", True)) and x.size >= 2:
            row = _viterbi_row(
                z_top, roots_all,
                anchor=anchor,
                prev_row=None,
                tol_im=float(opt["tol_im"]),
                lam_space=float(opt["lam_space"]),
                lam_asym=float(opt["lam_asym"]),
                lam_anchor=float(opt["lam_anchor"]),
                lam_prev=0.0,
            )
        else:
            row = numpy.empty(x.size, dtype=numpy.complex128)
            for k in range(x.size):
                row[k] = _pick_with_target(
                    z_top[k], roots_all[k], anchor[k],
                    tol_im=float(opt["tol_im"]),
                    lam_asym=float(opt["lam_asym"]),
                    lam_target=float(opt["lam_anchor"]),
                )
        return z_top, row

    def _descend_row(self, x, sgn, y_prev, y_cur, row_prev, opt):
        n = x.size
        z_prev = x.astype(numpy.complex128) + 1j * sgn * float(y_prev)
        z_cur = x.astype(numpy.complex128) + 1j * sgn * float(y_cur)

        row_pred = numpy.empty(n, dtype=numpy.complex128)
        row_pred[:] = numpy.nan + 1j * numpy.nan

        for k in range(n):
            if not _is_finite_complex(row_prev[k]):
                continue
            row_pred[k], _ = _continue_segment(
                self.coeffs, z_prev[k], row_prev[k], z_cur[k],
                tol_im=float(opt["tol_im"]),
                lam_asym=float(opt["lam_asym"]),
                newton_tol=float(opt["newton_tol"]),
                newton_iter=int(opt["newton_iter"]),
                min_pm=float(opt["min_pm"]),
                max_depth=int(opt["max_depth"]),
                jump_factor=float(opt["jump_factor"]),
            )

        if not bool(opt.get("use_viterbi_rows", True)) or n < 2:
            return z_cur, row_pred

        s = self.coeffs.shape[1] - 1
        roots_all = numpy.full((n, s), numpy.nan + 1j * numpy.nan, dtype=numpy.complex128)
        for k in range(n):
            roots = _roots_m(self.coeffs, z_cur[k])
            roots_all[k, :min(s, roots.size)] = roots[:min(s, roots.size)]

        row = _viterbi_row(
            z_cur, roots_all,
            anchor=row_pred,
            prev_row=row_prev,
            tol_im=float(opt["tol_im"]),
            lam_space=float(opt["lam_space"]),
            lam_asym=float(opt["lam_asym"]),
            lam_anchor=float(opt["lam_anchor"]),
            lam_prev=float(opt["lam_prev"]),
        )

        # Rescue: if row choice disagrees strongly with vertical predictor,
        # and predictor has much larger positive imaginary part, trust predictor.
        rescue_diff = float(opt.get("row_rescue_diff", 1.0))
        rescue_imag_ratio = float(opt.get("row_rescue_imag_ratio", 20.0))
        rescue_min_im = float(opt.get("row_rescue_min_im", 1e-2))

        for k in range(n):
            wk = row[k]
            wp = row_pred[k]
            if (not _is_finite_complex(wk)) or (not _is_finite_complex(wp)):
                continue

            im_row = float(sgn * numpy.imag(wk))
            im_pred = float(sgn * numpy.imag(wp))

            if (
                abs(wk - wp) > rescue_diff and
                im_pred > rescue_min_im and
                im_pred > rescue_imag_ratio * max(im_row, 1e-30)
            ):
                row[k] = wp

        # Strong final-row rescue: if the row solution is nearly real but the
        # independent scalar-from-top solution has large positive imaginary part,
        # trust scalar-from-top.
        scalar_rescue_diff = float(opt.get("scalar_row_rescue_diff", 1.0))
        scalar_rescue_imag_ratio = float(opt.get("scalar_row_rescue_imag_ratio", 20.0))
        scalar_rescue_min_im = float(opt.get("scalar_row_rescue_min_im", 1e-2))
        scalar_rescue_ymax = float(opt.get("scalar_row_rescue_ymax", numpy.inf))

        if float(y_cur) <= scalar_rescue_ymax:
            for k in range(n):
                wk = row[k]
                if not _is_finite_complex(wk):
                    continue

                ws = self._scalar_from_top(z_cur[k])
                if not _is_finite_complex(ws):
                    continue

                im_row = float(sgn * numpy.imag(wk))
                im_scalar = float(sgn * numpy.imag(ws))

                if (
                    abs(wk - ws) > scalar_rescue_diff and
                    im_scalar > scalar_rescue_min_im and
                    im_scalar > scalar_rescue_imag_ratio * max(im_row, 1e-30)
                ):
                    row[k] = ws

        if bool(opt.get("row_refine", True)):
            passes = max(1, int(opt.get("refine_passes", 1)))
            for _ in range(passes):
                for k in range(n):
                    target = row[k]
                    if (k > 0) and _is_finite_complex(row[k - 1]):
                        if _is_finite_complex(target):
                            target = 0.5 * (target + row[k - 1])
                        else:
                            target = row[k - 1]
                    if _is_finite_complex(row_pred[k]):
                        if _is_finite_complex(target):
                            target = 0.5 * (target + row_pred[k])
                        else:
                            target = row_pred[k]
                    roots = roots_all[k]
                    row[k] = _pick_with_target(
                        z_cur[k], roots, target,
                        tol_im=float(opt["tol_im"]),
                        lam_asym=float(opt["lam_asym"]),
                        lam_target=1.0,
                    )

        return z_cur, row

    def _evaluate_line_2d(self, z_line):
        opt = self._options()
        z_line = numpy.asarray(z_line, dtype=numpy.complex128).ravel()
        x = numpy.real(z_line).astype(float)
        y_target = float(abs(numpy.imag(z_line[0])))
        sgn = 1.0 if numpy.imag(z_line[0]) >= 0.0 else -1.0

        y_top = self._top_height(z_line[0])
        ys = self._levels(y_top, y_target, opt)

        _, row = self._top_row(x, sgn, ys[0], opt)
        z_prev = x.astype(numpy.complex128) + 1j * sgn * ys[0]
        row_prev = row

        for y in ys[1:]:
            _, row_prev = self._descend_row(x, sgn, numpy.imag(z_prev[0]) * sgn, y, row_prev, opt)
            z_prev = x.astype(numpy.complex128) + 1j * sgn * float(y)

        return row_prev

    def __call__(self, z):
        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            return self.evaluate_scalar(complex(z))

        z_flat = z.ravel().astype(numpy.complex128)
        for k in range(z_flat.size):
            if numpy.imag(z_flat[k]) == 0.0:
                eps_loc = float(self.eps) if self.eps is not None else 1e-8 * max(1.0, abs(z_flat[k]))
                z_flat[k] = complex(numpy.real(z_flat[k]), eps_loc)

        if _is_flat_imag_line_1d(z_flat):
            try:
                out = self._evaluate_line_2d(z_flat)
            except Exception:
                out = numpy.empty(z_flat.size, dtype=numpy.complex128)
                for k in range(z_flat.size):
                    out[k] = self.evaluate_scalar(z_flat[k])
        else:
            out = numpy.empty(z_flat.size, dtype=numpy.complex128)
            for k in range(z_flat.size):
                out[k] = self.evaluate_scalar(z_flat[k])

        return out.reshape(z.shape)
