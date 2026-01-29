# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# Robust Stieltjes branch evaluation for algebraic P(z,m)=0.
#
# This version is tailored for empirical polynomial fits where spurious
# small-Im roots can appear outside the true support and create fake bulks
# when using rho = Im m(x+i*eta)/pi.
#
# Core idea: pick the physical branch by combining
#   (i) Herglotz half-plane constraint,
#   (ii) asymptotic constraint z*m ~ -1,
#   (iii) 1D continuity along x (Viterbi),
# and DO NOT globally reward large |Im(m)| (which can fabricate density).

import numpy

__all__ = ["StieltjesPoly"]


# =========================
# Poly -> roots in m for z
# =========================

def _poly_m_coeffs(a_coeffs, z):
    """Return coefficients b_j for \sum_j b_j m^j = 0 at fixed z.

    a_coeffs[i,j] is coeff of z^i m^j.
    Returns b of length (deg_m+1) with b[j] = \sum_i a[i,j] z^i.
    """
    a = numpy.asarray(a_coeffs, dtype=numpy.complex128)
    deg_z = a.shape[0] - 1
    deg_m = a.shape[1] - 1

    # Horner in z for each m-power j
    z = complex(z)
    b = numpy.zeros((deg_m + 1,), dtype=numpy.complex128)
    # b[j] = a[0,j] + a[1,j] z + ... + a[deg_z,j] z^{deg_z}
    # do Horner: (((a[deg_z,j] z + a[deg_z-1,j]) z + ...) z + a[0,j])
    for j in range(deg_m + 1):
        acc = 0.0 + 0.0j
        for i in range(deg_z, -1, -1):
            acc = acc * z + a[i, j]
        b[j] = acc

    return b


def _roots_m(a_coeffs, z):
    """All algebraic roots in m at fixed z."""
    b = _poly_m_coeffs(a_coeffs, z)

    # Drop leading zeros in highest power to keep numpy.roots stable
    # numpy.roots expects highest degree first.
    coeffs = b.copy()
    # find highest nonzero index
    nz = numpy.flatnonzero(numpy.abs(coeffs) > 0.0)
    if nz.size == 0:
        return numpy.array([], dtype=numpy.complex128)
    j_max = int(nz.max())
    coeffs = coeffs[: j_max + 1]

    # reverse to highest-first
    return numpy.roots(coeffs[::-1])


# ==============================
# Physical root selection scalar
# ==============================

def _pick_physical_root_scalar(z, roots, target=None, tol_im=1e-12):
    """Pick the physical root among candidates for a single z.

    Rules:
      1) Prefer roots with sign(Im m) = sign(Im z) (Herglotz).
      2) Break ties by asymptotic closeness to -1/z.
      3) If target provided, also enforce continuity by closeness to target.

    Returns complex root.
    """
    z = complex(z)
    roots = numpy.asarray(roots, dtype=numpy.complex128).ravel()
    if roots.size == 0:
        return numpy.nan + 1j * numpy.nan

    s = numpy.sign(z.imag)
    if s == 0.0:
        s = 1.0

    im_s = numpy.imag(roots) * s
    cand = roots[im_s > -tol_im]
    if cand.size == 0:
        cand = roots

    # asymptotic target
    m_asym = -1.0 / z

    if target is None:
        score = numpy.abs(cand - m_asym)
    else:
        t = complex(target)
        # continuity dominates, asymptotic helps in ambiguous regions
        score = numpy.abs(cand - t) + 0.15 * numpy.abs(cand - m_asym)

    return cand[int(numpy.argmin(score))]


# =====================
# Viterbi along a line
# =====================

def _viterbi_track(z_list, roots_list, mL, mR, *,
                  lam_space=1.0,
                  lam_edge=20.0,
                  lam_asym=0.25,
                  lam_time=0.0,
                  # Hinge penalty against "tiny-im" traps ONLY
                  lam_tiny_im=0.0,
                  tiny_im=1e-7,
                  tol_im=1e-12,
                  m_prev=None):
    """Choose one root per z via dynamic programming.

    z_list: (nz,)
    roots_list: list of arrays of candidate roots, each (k_i,)
    mL,mR: boundary anchors (complex)
    m_prev: optional previous-time chosen path, shape (nz,)

    Returns m_path (nz,) and ok (nz,) boolean indicating finite.
    """
    z_list = numpy.asarray(z_list, dtype=numpy.complex128).ravel()
    nz = z_list.size

    # Determine a fixed K by padding with NaNs (max roots)
    K = max((r.size for r in roots_list), default=0)
    if K == 0:
        return (numpy.full((nz,), numpy.nan + 1j * numpy.nan, dtype=numpy.complex128),
                numpy.zeros((nz,), dtype=bool))

    R = numpy.full((nz, K), numpy.nan + 1j * numpy.nan, dtype=numpy.complex128)
    for i, r in enumerate(roots_list):
        if r.size:
            R[i, : r.size] = r

    # feasible mask: finite and Herglotz half-plane (soft)
    s = numpy.sign(z_list.imag)
    s[s == 0.0] = 1.0
    IM = (numpy.imag(R) * s[:, None])
    feasible = numpy.isfinite(R) & (IM > -tol_im)

    # unary cost
    unary = numpy.full((nz, K), numpy.inf, dtype=numpy.float64)
    # continuity to asymptotic (-1/z) discourages fake branches outside support
    m_asym = -1.0 / z_list

    for i in range(nz):
        zi = z_list[i]
        mi = m_asym[i]
        for k in range(K):
            if not feasible[i, k]:
                continue
            w = R[i, k]
            c = 0.0

            # asymptotic penalty (small weight)
            if lam_asym != 0.0:
                c += float(lam_asym) * float(numpy.abs(zi * w + 1.0))

            # optional hinge against tiny imag (ONLY below threshold)
            if lam_tiny_im != 0.0:
                im = abs(w.imag)
                floor = max(float(tiny_im), 0.25 * abs(zi.imag))
                if im < floor:
                    c += float(lam_tiny_im) * float(((floor / max(im, 1e-16)) - 1.0) ** 2)

            # optional time consistency
            if (lam_time != 0.0) and (m_prev is not None) and numpy.isfinite(m_prev[i]):
                c += float(lam_time) * float(numpy.abs(w - m_prev[i]))

            unary[i, k] = c

    # boundary anchors
    if numpy.isfinite(mL):
        unary[0, :] += float(lam_edge) * numpy.abs(R[0, :] - mL)
    if numpy.isfinite(mR):
        unary[-1, :] += float(lam_edge) * numpy.abs(R[-1, :] - mR)

    # pairwise cost
    dp = numpy.full((nz, K), numpy.inf, dtype=numpy.float64)
    prev = numpy.full((nz, K), -1, dtype=numpy.int64)

    dp[0, :] = unary[0, :]

    for i in range(1, nz):
        for k in range(K):
            if not numpy.isfinite(unary[i, k]):
                continue
            # transition from any j
            best_val = numpy.inf
            best_j = -1
            wk = R[i, k]
            for j in range(K):
                if not numpy.isfinite(dp[i - 1, j]):
                    continue
                wj = R[i - 1, j]
                if not numpy.isfinite(wj):
                    continue
                val = dp[i - 1, j] + float(lam_space) * float(numpy.abs(wk - wj))
                if val < best_val:
                    best_val = val
                    best_j = j
            if best_j >= 0:
                dp[i, k] = best_val + unary[i, k]
                prev[i, k] = best_j

    # backtrack
    k_end = int(numpy.argmin(dp[-1, :]))
    m_path = numpy.full((nz,), numpy.nan + 1j * numpy.nan, dtype=numpy.complex128)
    if not numpy.isfinite(dp[-1, k_end]):
        return m_path, numpy.zeros((nz,), dtype=bool)

    k = k_end
    for i in range(nz - 1, -1, -1):
        m_path[i] = R[i, k]
        k = prev[i, k]
        if (i > 0) and (k < 0):
            # cannot continue
            break

    ok = numpy.isfinite(m_path)
    return m_path, ok


# ============
# StieltjesPoly
# ============

class StieltjesPoly(object):
    """Callable m(z) for P(z,m)=0 using robust branch selection."""

    def __init__(self, a_coeffs, *,
                 viterbi_opt=None):
        self.a_coeffs = numpy.asarray(a_coeffs, dtype=numpy.complex128)
        self.viterbi_opt = dict(viterbi_opt or {})

    # ----------
    # scalar eval
    # ----------

    def evaluate_scalar(self, z, target=None):
        r = _roots_m(self.a_coeffs, z)
        return _pick_physical_root_scalar(z, r, target=target)

    # ---------------
    # vectorized eval
    # ---------------

    def __call__(self, z):
        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        # If 1D, do Viterbi tracking in the given order
        if z.ndim == 1:
            z_list = z.ravel()

            # roots for each point
            roots_list = [_roots_m(self.a_coeffs, zi) for zi in z_list]

            # boundary anchors via scalar selection
            mL = self.evaluate_scalar(z_list[0])
            mR = self.evaluate_scalar(z_list[-1])

            opt = {
                "lam_space": 1.0,
                "lam_edge": 20.0,
                "lam_asym": 0.25,
                "lam_time": 0.0,
                "lam_tiny_im": 0.0,
                "tiny_im": 1e-7,
                "tol_im": 1e-12,
            }
            opt.update(self.viterbi_opt)

            m_path, ok = _viterbi_track(
                z_list,
                roots_list,
                mL,
                mR,
                lam_space=opt["lam_space"],
                lam_edge=opt["lam_edge"],
                lam_asym=opt["lam_asym"],
                lam_time=opt["lam_time"],
                lam_tiny_im=opt["lam_tiny_im"],
                tiny_im=opt["tiny_im"],
                tol_im=opt["tol_im"],
                m_prev=None,
            )

            # fallback pointwise for any failures
            if not numpy.all(ok):
                out = m_path.copy()
                for i in numpy.flatnonzero(~ok):
                    out[i] = self.evaluate_scalar(z_list[i], target=None)
                m_path = out

            out = m_path.reshape(z.shape)
            return out.reshape(()) if scalar else out

        # For nd>1: evaluate pointwise with asymptotic tie-break
        out = numpy.empty(z.size, dtype=numpy.complex128)
        zf = z.ravel()
        prev = None
        for i in range(zf.size):
            out[i] = self.evaluate_scalar(zf[i], target=prev)
            prev = out[i]
        out = out.reshape(z.shape)
        return out.reshape(()) if scalar else out
