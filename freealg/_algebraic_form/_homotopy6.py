# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# Robust Stieltjes branch evaluation for algebraic P(z,m)=0 using
# global 1D dynamic programming (Viterbi) along a complex line.
#
# _homotopy6.py (revised): same public API intent as previous homotopy files,
# with stronger branch locking via sparse homotopy anchors + cut-repair pass.

import numpy

try:
    from ._moments import AlgebraicStieltjesMoments
except Exception:  # pragma: no cover
    AlgebraicStieltjesMoments = None


__all__ = ["StieltjesPoly"]


def _poly_coeffs_in_m(coeffs, z):
    a = coeffs
    dz = a.shape[0] - 1
    s = a.shape[1] - 1
    # Horner in z is a bit more stable than explicit powers for high degree.
    coeff_m = numpy.empty(s + 1, dtype=numpy.complex128)
    z = complex(z)
    for j in range(s + 1):
        acc = 0.0 + 0.0j
        for i in range(dz, -1, -1):
            acc = acc * z + a[i, j]
        coeff_m[j] = acc
    return coeff_m


def _roots_m(coeffs, z):
    coeff_m = _poly_coeffs_in_m(coeffs, z)
    c = coeff_m[::-1]
    # drop leading zeros (highest degree first for numpy.roots)
    while c.size > 1 and numpy.abs(c[0]) == 0:
        c = c[1:]
    if c.size <= 1:
        return numpy.array([], dtype=numpy.complex128)
    return numpy.roots(c)


def _pick_anchor_asym(z, roots, tol_im, lam_asym):
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan
    sgn = 1.0 if numpy.imag(z) >= 0 else -1.0
    ok = (sgn * numpy.imag(roots) > tol_im)
    cand = roots[ok] if numpy.any(ok) else roots
    cost = lam_asym * numpy.abs(z * cand + 1.0)
    return cand[int(numpy.argmin(cost))]


def _select_with_target(z, roots, target, tol_im, lam_asym=0.25, lam_target=1.0):
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan
    sgn = 1.0 if numpy.imag(z) >= 0 else -1.0
    ok = (sgn * numpy.imag(roots) > tol_im)
    cand = roots[ok] if numpy.any(ok) else roots
    cost = lam_asym * numpy.abs(z * cand + 1.0)
    if target is not None and numpy.isfinite(target):
        cost = cost + lam_target * numpy.abs(cand - target)
    return cand[int(numpy.argmin(cost))]


def _viterbi_1d(z_list, roots_all, *, lam_space, lam_asym,
                lam_tiny_im, tiny_im, tol_im,
                lam_edge=0.0, mL=None, mR=None,
                lam_time=0.0, m_prev=None,
                lam_anchor=0.0, m_anchor=None):
    n, s = roots_all.shape
    big = 1.0e300

    cost0 = numpy.zeros((n, s), dtype=float)
    back = numpy.zeros((n, s), dtype=numpy.int64)
    dp = numpy.full((n, s), big, dtype=float)

    for k in range(n):
        z = z_list[k]
        r = roots_all[k]
        sgn = 1.0 if numpy.imag(z) >= 0 else -1.0

        finite = numpy.isfinite(r)
        ok = finite & (sgn * numpy.imag(r) > tol_im)
        cost0[k, ~ok] += big * 1.0e-6

        if lam_tiny_im != 0.0 and tiny_im is not None:
            imabs = numpy.abs(numpy.imag(r))
            hing = numpy.maximum(0.0, float(tiny_im) - imabs)
            cost0[k] += lam_tiny_im * hing * hing

        if lam_asym != 0.0:
            cost0[k] += lam_asym * numpy.abs(z * r + 1.0)

        if (lam_time != 0.0) and (m_prev is not None) and numpy.isfinite(m_prev[k]):
            cost0[k] += lam_time * numpy.abs(r - m_prev[k])

        if (lam_anchor != 0.0) and (m_anchor is not None) and numpy.isfinite(m_anchor[k]):
            cost0[k] += lam_anchor * numpy.abs(r - m_anchor[k])

    if mL is not None and numpy.isfinite(mL):
        cost0[0] += float(lam_edge) * numpy.abs(roots_all[0] - mL)
    if mR is not None and numpy.isfinite(mR):
        cost0[-1] += float(lam_edge) * numpy.abs(roots_all[-1] - mR)

    dp[0] = cost0[0]

    for k in range(1, n):
        r = roots_all[k]
        rp = roots_all[k - 1]
        for j in range(s):
            trans = dp[k - 1] + lam_space * numpy.abs(r[j] - rp)
            idx = int(numpy.argmin(trans))
            dp[k, j] = trans[idx] + cost0[k, j]
            back[k, j] = idx

    jn = int(numpy.argmin(dp[-1]))
    path = numpy.empty(n, dtype=numpy.complex128)
    for k in range(n - 1, -1, -1):
        path[k] = roots_all[k, jn]
        if k > 0:
            jn = int(back[k, jn])
    return path


def _path_energy(z_list, path, *, lam_space, lam_asym, lam_tiny_im, tiny_im,
                 tol_im, lam_edge=0.0, mL=None, mR=None,
                 lam_anchor=0.0, m_anchor=None):
    e = 0.0
    n = path.size
    for k in range(n):
        z = z_list[k]
        w = path[k]
        if not numpy.isfinite(w):
            return numpy.inf
        sgn = 1.0 if numpy.imag(z) >= 0 else -1.0
        if sgn * numpy.imag(w) <= tol_im:
            e += 1.0e12
        if lam_asym != 0.0:
            e += float(lam_asym) * float(numpy.abs(z * w + 1.0))
        if lam_tiny_im != 0.0 and tiny_im is not None:
            hing = max(0.0, float(tiny_im) - abs(float(numpy.imag(w))))
            e += float(lam_tiny_im) * hing * hing
        if (lam_anchor != 0.0) and (m_anchor is not None) and numpy.isfinite(m_anchor[k]):
            e += float(lam_anchor) * float(numpy.abs(w - m_anchor[k]))
        if k > 0:
            e += float(lam_space) * float(numpy.abs(path[k] - path[k - 1]))
    if (mL is not None) and numpy.isfinite(mL):
        e += float(lam_edge) * float(numpy.abs(path[0] - mL))
    if (mR is not None) and numpy.isfinite(mR):
        e += float(lam_edge) * float(numpy.abs(path[-1] - mR))
    return e


class StieltjesPoly(object):
    """Callable m(z) for P(z,m)=0 using robust branch selection."""

    def __init__(self, coeffs, mom=None, *, viterbi_opt=None, eps=None,
                 height=2.0, steps=80, order=15):
        self.coeffs = numpy.asarray(coeffs, dtype=numpy.complex128)
        self.viterbi_opt = dict(viterbi_opt or {})
        self.eps = eps
        self.height = float(height)
        self.steps = int(steps)
        self.order = int(order)
        self._mom = mom
        self._rad = None
        self._m0p = None
        self._m0m = None
        # Lazy init moments only if scalar homotopy anchors are used.

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

        # Support both callable moments mom(k) and object with stieltjes/radius API.
        try:
            if hasattr(self._mom, "radius") and hasattr(self._mom, "stieltjes"):
                self._rad = 1.0 + self.height * float(self._mom.radius(self.order))
                self._m0p = self._mom.stieltjes(1j * self._rad, self.order)
                self._m0m = self._mom.stieltjes(-1j * self._rad, self.order)
                return
        except Exception:
            pass

        # Fallback: callable raw moments mom(k)
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
            self._m0p = -numpy.sum(mu * (zp ** (-numpy.arange(self.order + 1) - 1)))
            self._m0m = -numpy.sum(mu * (zm ** (-numpy.arange(self.order + 1) - 1)))
        except Exception:
            self._rad = numpy.inf
            self._m0p = None
            self._m0m = None

    def _poly_roots(self, z):
        return _roots_m(self.coeffs, z)

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

    def _evaluate_homotopy_scalar(self, z):
        z = complex(z)
        if z.imag == 0.0:
            eps_loc = float(self.eps) if self.eps is not None else 1e-8 * max(1.0, abs(z))
            z_eval = z + 1j * eps_loc
        else:
            z_eval = z
        sgn = 1.0 if z_eval.imag >= 0 else -1.0

        self._ensure_moments()
        rad = self._rad if (self._rad is not None and numpy.isfinite(self._rad)) else (1.0 + 2.0 * max(1.0, abs(z_eval)))
        if abs(z_eval) > rad:
            target = self._moment_est(z_eval)
            return _select_with_target(z_eval, self._poly_roots(z_eval), target,
                                       tol_im=float(self.viterbi_opt.get("tol_im", 1e-14)),
                                       lam_asym=0.25, lam_target=1.0)

        z0 = 1j * rad if sgn > 0 else -1j * rad
        target = self._moment_est(z0)
        tol_im = float(self.viterbi_opt.get("tol_im", 1e-14))
        w_prev = _select_with_target(z0, self._poly_roots(z0), target, tol_im,
                                     lam_asym=0.25, lam_target=1.0)
        # Straight-line continuation.
        nstep = max(12, int(self.steps))
        for tau in numpy.linspace(0.0, 1.0, nstep + 1)[1:]:
            z_tau = z0 + tau * (z_eval - z0)
            w_prev = _select_with_target(z_tau, self._poly_roots(z_tau), w_prev, tol_im,
                                         lam_asym=0.10, lam_target=1.0)
        return w_prev

    def evaluate_scalar(self, z, target=None):
        # Prefer homotopy scalar if moments available; it is much more robust
        # near support edges than asymptotic-only scalar selection.
        try:
            return self._evaluate_homotopy_scalar(z)
        except Exception:
            r = self._poly_roots(z)
            if r.size == 0:
                return numpy.nan + 1j * numpy.nan
            tol_im = float(self.viterbi_opt.get("tol_im", 1e-14))
            lam_asym = float(self.viterbi_opt.get("lam_asym", 1.0))
            return _select_with_target(z, r, target, tol_im, lam_asym=lam_asym, lam_target=1.0)

    def __call__(self, z):
        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        if z.ndim == 1 and z.size >= 2:
            z_list = z.ravel()
            s = self.coeffs.shape[1] - 1
            roots_all = numpy.empty((z_list.size, s), dtype=numpy.complex128)
            ok_all = numpy.ones(z_list.size, dtype=bool)
            for k in range(z_list.size):
                r = self._poly_roots(z_list[k])
                if r.size != s:
                    ok_all[k] = False
                    rr = numpy.empty(s, dtype=numpy.complex128)
                    rr[:] = numpy.nan + 1j * numpy.nan
                    rr[:min(s, r.size)] = r[:min(s, r.size)]
                    roots_all[k] = rr
                else:
                    roots_all[k] = r

            opt = {
                "lam_space": 1.0,
                "lam_asym": 0.75,
                "lam_tiny_im": 300.0,
                "tiny_im": None,
                "tol_im": 1e-14,
                "lam_edge": 25.0,
                "two_pass": True,
                "lam_time": 0.5,
                # new anchor options
                "lam_anchor": 8.0,
                "anchor_stride": None,
                "repair_cuts": True,
                "repair_factor": 4.0,
            }
            opt.update(self.viterbi_opt)
            tol_im = float(opt["tol_im"])

            if opt["tiny_im"] is None:
                opt["tiny_im"] = 0.5 * abs(float(numpy.imag(z_list[0])))

            # endpoint anchors from scalar homotopy (stronger than asymptotic-only)
            mL = self.evaluate_scalar(z_list[0], target=None)
            mR = self.evaluate_scalar(z_list[-1], target=None)

            # Sparse anchors along the line to prevent wrong-sheet picks at bulk edges.
            m_anchor = numpy.full(z_list.size, numpy.nan + 1j * numpy.nan, dtype=numpy.complex128)
            stride = opt.get("anchor_stride", None)
            if stride is None:
                stride = max(1, z_list.size // 24)
            else:
                stride = max(1, int(stride))
            idxs = list(range(0, z_list.size, stride))
            if (z_list.size - 1) not in idxs:
                idxs.append(z_list.size - 1)
            # add a few deterministic edge-biased anchors
            for frac in (0.125, 0.25, 0.5, 0.75, 0.875):
                idxs.append(int(round(frac * (z_list.size - 1))))
            idxs = sorted(set(i for i in idxs if 0 <= i < z_list.size))
            for i in idxs:
                try:
                    m_anchor[i] = self._evaluate_homotopy_scalar(z_list[i])
                except Exception:
                    pass

            m_fwd = _viterbi_1d(
                z_list, roots_all,
                lam_space=float(opt["lam_space"]),
                lam_asym=float(opt["lam_asym"]),
                lam_tiny_im=float(opt["lam_tiny_im"]),
                tiny_im=float(opt["tiny_im"]),
                tol_im=tol_im,
                lam_edge=float(opt["lam_edge"]),
                mL=mL,
                mR=mR,
                lam_time=0.0,
                m_prev=None,
                lam_anchor=float(opt.get("lam_anchor", 0.0)),
                m_anchor=m_anchor,
            )
            m_path = m_fwd

            if bool(opt.get("two_pass", True)) and z_list.size >= 4:
                m_rev_r = _viterbi_1d(
                    z_list[::-1], roots_all[::-1],
                    lam_space=float(opt["lam_space"]),
                    lam_asym=float(opt["lam_asym"]),
                    lam_tiny_im=float(opt["lam_tiny_im"]),
                    tiny_im=float(opt["tiny_im"]),
                    tol_im=tol_im,
                    lam_edge=float(opt["lam_edge"]),
                    mL=mR,
                    mR=mL,
                    lam_time=0.0,
                    m_prev=None,
                    lam_anchor=float(opt.get("lam_anchor", 0.0)),
                    m_anchor=m_anchor[::-1],
                )
                m_rev = m_rev_r[::-1]

                ef = _path_energy(
                    z_list, m_fwd,
                    lam_space=float(opt["lam_space"]),
                    lam_asym=float(opt["lam_asym"]),
                    lam_tiny_im=float(opt["lam_tiny_im"]),
                    tiny_im=float(opt["tiny_im"]),
                    tol_im=tol_im,
                    lam_edge=float(opt["lam_edge"]),
                    mL=mL, mR=mR,
                    lam_anchor=float(opt.get("lam_anchor", 0.0)),
                    m_anchor=m_anchor,
                )
                er = _path_energy(
                    z_list, m_rev,
                    lam_space=float(opt["lam_space"]),
                    lam_asym=float(opt["lam_asym"]),
                    lam_tiny_im=float(opt["lam_tiny_im"]),
                    tiny_im=float(opt["tiny_im"]),
                    tol_im=tol_im,
                    lam_edge=float(opt["lam_edge"]),
                    mL=mL, mR=mR,
                    lam_anchor=float(opt.get("lam_anchor", 0.0)),
                    m_anchor=m_anchor,
                )
                m_seed = m_fwd if ef <= er else m_rev

                m_ref = _viterbi_1d(
                    z_list, roots_all,
                    lam_space=float(opt["lam_space"]),
                    lam_asym=float(opt["lam_asym"]),
                    lam_tiny_im=float(opt["lam_tiny_im"]),
                    tiny_im=float(opt["tiny_im"]),
                    tol_im=tol_im,
                    lam_edge=float(opt["lam_edge"]),
                    mL=mL, mR=mR,
                    lam_time=float(opt.get("lam_time", 0.0)),
                    m_prev=m_seed,
                    lam_anchor=float(opt.get("lam_anchor", 0.0)),
                    m_anchor=m_anchor,
                )
                e_ref = _path_energy(
                    z_list, m_ref,
                    lam_space=float(opt["lam_space"]),
                    lam_asym=float(opt["lam_asym"]),
                    lam_tiny_im=float(opt["lam_tiny_im"]),
                    tiny_im=float(opt["tiny_im"]),
                    tol_im=tol_im,
                    lam_edge=float(opt["lam_edge"]),
                    mL=mL, mR=mR,
                    lam_anchor=float(opt.get("lam_anchor", 0.0)),
                    m_anchor=m_anchor,
                )
                m_path = m_ref if e_ref <= min(ef, er) else m_seed

            # Repair obvious "cuts to zero" by local interpolation targets.
            if bool(opt.get("repair_cuts", True)):
                sgn = 1.0 if numpy.imag(z_list[0]) >= 0 else -1.0
                im = sgn * numpy.imag(m_path)
                floor = max(float(opt["tiny_im"]), 1e-16)
                bad = im <= 0.25 * floor
                # only repair isolated/short cuts surrounded by healthy points
                n = z_list.size
                for i in range(n):
                    if not bad[i]:
                        continue
                    left = i - 1
                    while left >= 0 and bad[left]:
                        left -= 1
                    right = i + 1
                    while right < n and bad[right]:
                        right += 1
                    if left < 0 or right >= n:
                        continue
                    if (sgn * m_path[left].imag <= float(opt["repair_factor"]) * floor or
                            sgn * m_path[right].imag <= float(opt["repair_factor"]) * floor):
                        continue
                    t = 0.5 if right == left + 1 else (i - left) / float(right - left)
                    target = (1.0 - t) * m_path[left] + t * m_path[right]
                    r = roots_all[i][numpy.isfinite(roots_all[i])]
                    if r.size == 0:
                        continue
                    cand = _select_with_target(z_list[i], r, target, tol_im,
                                               lam_asym=0.10, lam_target=2.0)
                    if numpy.isfinite(cand) and (sgn * cand.imag > tol_im):
                        m_path[i] = cand

            if (not numpy.all(ok_all)) or numpy.any(~numpy.isfinite(m_path)):
                out = m_path.copy()
                prev = None
                for i in range(z_list.size):
                    if (not ok_all[i]) or (not numpy.isfinite(out[i])):
                        try:
                            out[i] = self._evaluate_homotopy_scalar(z_list[i])
                        except Exception:
                            out[i] = _select_with_target(z_list[i], self._poly_roots(z_list[i]), prev,
                                                         tol_im, lam_asym=float(opt["lam_asym"]),
                                                         lam_target=1.0)
                    prev = out[i]
                m_path = out

            out = m_path.reshape(z.shape)
            return out.reshape(()) if scalar else out

        out = numpy.empty(z.size, dtype=numpy.complex128)
        zf = z.ravel()
        prev = None
        for i in range(zf.size):
            out[i] = self.evaluate_scalar(zf[i], target=prev)
            prev = out[i]
        out = out.reshape(z.shape)
        return out.reshape(()) if scalar else out
