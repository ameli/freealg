# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# Robust Stieltjes branch evaluation for algebraic P(z,m)=0 using
# global 1D dynamic programming (Viterbi) along a complex line.

import numpy


def _poly_coeffs_in_m(a_coeffs, z):
    a = a_coeffs
    dz = a.shape[0] - 1
    s = a.shape[1] - 1
    zp = numpy.array([z**i for i in range(dz + 1)], dtype=numpy.complex128)
    coeff_m = numpy.empty(s + 1, dtype=numpy.complex128)
    for j in range(s + 1):
        coeff_m[j] = numpy.dot(a[:, j], zp)
    return coeff_m


def _roots_m(a_coeffs, z):
    coeff_m = _poly_coeffs_in_m(a_coeffs, z)
    c = coeff_m[::-1]
    while c.size > 1 and numpy.abs(c[0]) == 0:
        c = c[1:]
    if c.size <= 1:
        return numpy.array([], dtype=numpy.complex128)
    return numpy.roots(c)


def _pick_anchor(z, roots, tol_im, lam_asym):
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan
    sgn = 1.0 if numpy.imag(z) >= 0 else -1.0
    ok = (sgn * numpy.imag(roots) > tol_im)
    if numpy.any(ok):
        cand = roots[ok]
    else:
        cand = roots
    cost = lam_asym * numpy.abs(z * cand + 1.0)
    return cand[int(numpy.argmin(cost))]


def _viterbi_1d(z_list, roots_all, *, lam_space, lam_asym,
                lam_tiny_im, tiny_im, tol_im):
    n, s = roots_all.shape
    big = 1.0e300

    cost0 = numpy.zeros((n, s), dtype=float)
    back = numpy.zeros((n, s), dtype=numpy.int64)
    dp = numpy.full((n, s), big, dtype=float)

    for k in range(n):
        z = z_list[k]
        r = roots_all[k]
        sgn = 1.0 if numpy.imag(z) >= 0 else -1.0

        ok = (sgn * numpy.imag(r) > tol_im)
        cost0[k, ~ok] += big * 1.0e-6

        if lam_tiny_im != 0.0 and tiny_im is not None:
            imabs = numpy.abs(numpy.imag(r))
            hing = numpy.maximum(0.0, float(tiny_im) - imabs)
            cost0[k] += lam_tiny_im * hing

        if lam_asym != 0.0:
            cost0[k] += lam_asym * numpy.abs(z * r + 1.0)

    m0 = _pick_anchor(z_list[0], roots_all[0], tol_im, lam_asym)
    mN = _pick_anchor(z_list[-1], roots_all[-1], tol_im, lam_asym)

    init = cost0[0] + lam_space * numpy.abs(roots_all[0] - m0)
    dp[0] = init

    for k in range(1, n):
        r = roots_all[k]
        rp = roots_all[k - 1]
        for j in range(s):
            trans = dp[k - 1] + lam_space * numpy.abs(r[j] - rp)
            idx = int(numpy.argmin(trans))
            dp[k, j] = trans[idx] + cost0[k, j]
            back[k, j] = idx

    last = dp[-1] + lam_space * numpy.abs(roots_all[-1] - mN)
    jn = int(numpy.argmin(last))

    path = numpy.empty(n, dtype=numpy.complex128)
    for k in range(n - 1, -1, -1):
        path[k] = roots_all[k, jn]
        if k > 0:
            jn = int(back[k, jn])
    return path


class StieltjesPoly(object):
    """Callable m(z) for P(z,m)=0 using robust branch selection."""

    def __init__(self, a_coeffs, *, viterbi_opt=None):
        self.a_coeffs = numpy.asarray(a_coeffs, dtype=numpy.complex128)
        self.viterbi_opt = dict(viterbi_opt or {})

    def evaluate_scalar(self, z, target=None):
        r = _roots_m(self.a_coeffs, z)
        if r.size == 0:
            return numpy.nan + 1j * numpy.nan
        tol_im = float(self.viterbi_opt.get("tol_im", 1e-14))
        lam_asym = float(self.viterbi_opt.get("lam_asym", 1.0))
        sgn = 1.0 if numpy.imag(z) >= 0 else -1.0
        ok = (sgn * numpy.imag(r) > tol_im)
        cand = r[ok] if numpy.any(ok) else r
        cost = lam_asym * numpy.abs(z * cand + 1.0)
        if target is not None and numpy.isfinite(target):
            lam_space = float(self.viterbi_opt.get("lam_space", 1.0))
            cost = cost + lam_space * numpy.abs(cand - target)
        return cand[int(numpy.argmin(cost))]

    def __call__(self, z):
        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        if z.ndim == 1 and z.size >= 2:
            z_list = z.ravel()
            s = self.a_coeffs.shape[1] - 1
            roots_all = numpy.empty((z_list.size, s), dtype=numpy.complex128)
            ok_all = numpy.ones(z_list.size, dtype=bool)
            for k in range(z_list.size):
                r = _roots_m(self.a_coeffs, z_list[k])
                if r.size != s:
                    ok_all[k] = False
                    if r.size == 0:
                        roots_all[k] = numpy.nan + 1j * numpy.nan
                    elif r.size < s:
                        rr = numpy.empty(s, dtype=numpy.complex128)
                        rr[:] = numpy.nan + 1j * numpy.nan
                        rr[:r.size] = r
                        roots_all[k] = rr
                    else:
                        roots_all[k] = r[:s]
                else:
                    roots_all[k] = r

            opt = {
                "lam_space": 1.0,
                "lam_asym": 1.0,
                "lam_tiny_im": 200.0,
                "tiny_im": None,
                "tol_im": 1e-14,
            }
            opt.update(self.viterbi_opt)

            if opt["tiny_im"] is None:
                opt["tiny_im"] = 0.5 * numpy.abs(numpy.imag(z_list[0]))

            m_path = _viterbi_1d(
                z_list, roots_all,
                lam_space=float(opt["lam_space"]),
                lam_asym=float(opt["lam_asym"]),
                lam_tiny_im=float(opt["lam_tiny_im"]),
                tiny_im=float(opt["tiny_im"]),
                tol_im=float(opt["tol_im"]),
            )

            if not numpy.all(ok_all):
                out = m_path.copy()
                prev = None
                for i in range(z_list.size):
                    if not ok_all[i] or not numpy.isfinite(out[i]):
                        out[i] = self.evaluate_scalar(z_list[i], target=prev)
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
