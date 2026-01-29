# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# Free Decompression (FD) solver for algebraic Stieltjes transforms.
#
# Public API (used by AlgebraicForm.decompress):
#   build_time_grid(size, n0, min_n_times=0) -> (t_all, idx_req)
#   decompress_newton(z_list, t_grid, a_coeffs, w0_list=None, **opts) -> (W, ok)
#
# Core equation (FD):
#   tau = exp(t) - 1
#   zeta = z - tau*w
#   Solve: P(zeta, w) = 0  where P(z,w)=sum_{i,j} a[i,j] z^i w^j
#   i.e. F(w) := P(z - tau*w, w) = 0.
#
# This rewrite focuses on *robust branch tracking* (multi-start Newton + 2-pass
# Viterbi with "active-region" tiny-im penalty), and optional per-time density
# renormalization for mass preservation when the polynomial fit is imperfect.

from __future__ import annotations

import os
import numpy as np

__all__ = ["build_time_grid", "decompress_newton"]


# =================
# Time grid helper
# =================

def build_time_grid(size, n0, min_n_times=0):
    """
    Build a monotone time grid for FD.

    Parameters
    ----------
    size : array_like
        Requested size ratios, i.e., n(t)/n0 = size. Can include 1.
    n0 : int
        Initial matrix size.
    min_n_times : int, default=0
        Ensures at least this many intermediate time points between
        successive requested times, based on implied integer sizes.

    Returns
    -------
    t_all : numpy.ndarray
        Full time grid (including intermediates), sorted.
    idx_req : numpy.ndarray
        Indices into t_all corresponding to the originally requested times.
    """
    size = np.asarray(size, dtype=float).ravel()
    if size.size == 0:
        raise ValueError("size must be non-empty")
    if np.any(size <= 0.0):
        raise ValueError("size must be > 0")

    t_req = np.log(size)
    order = np.argsort(t_req)
    t_req_sorted = t_req[order]

    n0 = int(n0)
    if n0 <= 0:
        raise ValueError("n0 must be a positive integer")

    t_all = [float(t_req_sorted[0])]
    for k in range(1, t_req_sorted.size):
        t0 = float(t_req_sorted[k - 1])
        t1 = float(t_req_sorted[k])
        if t1 <= t0:
            continue

        if int(min_n_times) <= 0:
            t_all.append(t1)
            continue

        nA = max(1, int(round(n0 * np.exp(t0))))
        nB = max(1, int(round(n0 * np.exp(t1))))
        dn = max(1, nB - nA)
        step_n = max(1, int(np.ceil(dn / float(min_n_times))))
        n_grid = list(range(nA, nB, step_n))
        if n_grid[-1] != nB:
            n_grid.append(nB)
        for nn in n_grid[1:]:
            t_all.append(float(np.log(nn / float(n0))))

    t_all = np.asarray(t_all, dtype=float)

    idx_req_sorted = np.empty(t_req_sorted.size, dtype=int)
    for i, t in enumerate(t_req_sorted):
        idx_req_sorted[i] = int(np.argmin(np.abs(t_all - float(t))))

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    idx_req = idx_req_sorted[inv]
    return t_all, idx_req


# ===================
# Polynomial utilities
# ===================

def _poly_coef_in_w(z, a_coeffs):
    """
    For fixed z, return coefficients c[j] so that P(z,w)=sum_j c[j] w^j.
    a_coeffs[i,j] corresponds to z^i w^j.
    """
    z = complex(z)
    a = np.asarray(a_coeffs, dtype=np.complex128)
    deg_z = int(a.shape[0] - 1)
    # Horner in z for each j
    zp = 1.0 + 0.0j
    c = np.array(a[0, :], dtype=np.complex128)
    for i in range(1, deg_z + 1):
        zp *= z
        c = c + a[i, :] * zp
    return c  # shape (s+1,)


def _eval_P(z, w, a_coeffs):
    c = _poly_coef_in_w(z, a_coeffs)
    # Horner in w
    ww = complex(w)
    out = 0.0 + 0.0j
    for cj in c[::-1]:
        out = out * ww + cj
    return out


def _eval_dP_dw(z, w, a_coeffs):
    """
    d/dw P(z,w)
    """
    c = _poly_coef_in_w(z, a_coeffs)  # c[j] w^j
    ww = complex(w)
    # derivative coefficients: j*c[j]
    out = 0.0 + 0.0j
    for j in range(c.size - 1, 0, -1):
        out = out * ww + (j * c[j])
    return out


def _eval_dP_dz(z, w, a_coeffs):
    """
    d/dz P(z,w)
    """
    z = complex(z)
    w = complex(w)
    a = np.asarray(a_coeffs, dtype=np.complex128)
    deg_z = int(a.shape[0] - 1)
    # compute b[j] = sum_{i>=1} i*a[i,j]*z^{i-1}
    if deg_z <= 0:
        return 0.0 + 0.0j
    b = np.zeros((a.shape[1],), dtype=np.complex128)
    zp = 1.0 + 0.0j
    for i in range(1, deg_z + 1):
        b = b + (i * a[i, :]) * zp
        zp *= z
    # evaluate in w: sum_j b[j] w^j
    out = 0.0 + 0.0j
    for bj in b[::-1]:
        out = out * w + bj
    return out


def _fd_F_and_dF(w, z, tau, a_coeffs):
    """
    F(w) = P(z - tau*w, w).
    dF/dw = dP/dz * (-tau) + dP/dw evaluated at (zeta, w).
    """
    zeta = z - tau * w
    F = _eval_P(zeta, w, a_coeffs)
    dPdw = _eval_dP_dw(zeta, w, a_coeffs)
    dPdz = _eval_dP_dz(zeta, w, a_coeffs)
    dF = dPdw - tau * dPdz
    return F, dF


# =================
# Newton (scalar)
# =================

def _newton_fd_scalar(
    z,
    t,
    a_coeffs,
    w_init,
    *,
    max_iter=60,
    tol=1e-12,
    armijo=True,
    min_lam=1e-6,
    w_min=0.0,
):
    """
    Newton solve for one z at one t.
    Returns (w, ok, n_iter, final_res).
    """
    z = complex(z)
    t = float(t)
    tau = float(np.expm1(t))  # exp(t)-1, stable for small t
    w = complex(w_init)

    # guard against nan seeds
    if not (np.isfinite(w.real) and np.isfinite(w.imag)):
        w = -1.0 / z

    # optional floor on imaginary (avoid falling to lower half due to roundoff)
    if w_min > 0.0 and w.imag < w_min:
        w = complex(w.real, w_min)

    # initial
    F, dF = _fd_F_and_dF(w, z, tau, a_coeffs)
    res0 = abs(F)
    if not np.isfinite(res0):
        return complex(np.nan, np.nan), False, 0, np.inf

    for it in range(max_iter):
        if abs(F) <= tol * (1.0 + res0):
            return w, True, it + 1, abs(F)

        # if derivative is degenerate, bail
        if not np.isfinite(dF.real) or not np.isfinite(dF.imag) or abs(dF) == 0.0:
            break

        step = -F / dF
        lam = 1.0

        if armijo:
            # Armijo on |F| (cheap, robust)
            f0 = abs(F)
            # Try to avoid huge steps
            if abs(step) > 10.0 * (1.0 + abs(w)):
                step = step * (10.0 * (1.0 + abs(w)) / abs(step))

            while lam >= min_lam:
                w_new = w + lam * step
                if w_min > 0.0 and w_new.imag < w_min:
                    w_new = complex(w_new.real, w_min)
                F_new, dF_new = _fd_F_and_dF(w_new, z, tau, a_coeffs)
                f1 = abs(F_new)
                if np.isfinite(f1) and (f1 <= (1.0 - 1e-4 * lam) * f0):
                    w, F, dF = w_new, F_new, dF_new
                    break
                lam *= 0.5
            else:
                # failed to find descent
                break
        else:
            w = w + step
            if w_min > 0.0 and w.imag < w_min:
                w = complex(w.real, w_min)
            F, dF = _fd_F_and_dF(w, z, tau, a_coeffs)

    # final
    F, _ = _fd_F_and_dF(w, z, tau, a_coeffs)
    ok = np.isfinite(F.real) and np.isfinite(F.imag) and (abs(F) <= 1e3 * tol * (1.0 + res0))
    return w, bool(ok), max_iter, abs(F)


# ==========================
# Candidate generation (per z)
# ==========================

def _make_default_seeds(z, w_prev, w_left, w_right):
    seeds = []
    if w_prev is not None and np.isfinite(w_prev.real) and np.isfinite(w_prev.imag):
        seeds.append(complex(w_prev))
    if w_left is not None and np.isfinite(w_left.real) and np.isfinite(w_left.imag):
        seeds.append(complex(w_left))
    if w_right is not None and np.isfinite(w_right.real) and np.isfinite(w_right.imag):
        seeds.append(complex(w_right))
    seeds.append(complex(-1.0 / z))
    return seeds


def _dedup_cands(cands, tol=1e-10):
    if len(cands) == 0:
        return np.empty((0,), dtype=np.complex128)
    out = []
    for w in cands:
        keep = True
        for u in out:
            if abs(w - u) <= tol * (1.0 + abs(u)):
                keep = False
                break
        if keep:
            out.append(w)
    return np.asarray(out, dtype=np.complex128)


def _fd_candidates(
    z,
    t,
    a_coeffs,
    seeds,
    *,
    max_iter=60,
    tol=1e-12,
    armijo=True,
    min_lam=1e-6,
    w_min=0.0,
    keep_best=8,
):
    """
    Multi-start Newton candidates.
    Returns (cands, ok_flags, resids).
    """
    cands = []
    oks = []
    ress = []
    for s in seeds:
        w, ok, _, res = _newton_fd_scalar(
            z, t, a_coeffs, s,
            max_iter=max_iter, tol=tol,
            armijo=armijo, min_lam=min_lam, w_min=w_min
        )
        if np.isfinite(w.real) and np.isfinite(w.imag):
            cands.append(w)
            oks.append(ok)
            ress.append(res)

    if len(cands) == 0:
        return np.empty((0,), np.complex128), np.empty((0,), bool), np.empty((0,), float)

    cands = np.asarray(cands, dtype=np.complex128)
    oks = np.asarray(oks, dtype=bool)
    ress = np.asarray(ress, dtype=float)

    # sort by residual
    idx = np.argsort(ress)
    cands = cands[idx]
    oks = oks[idx]
    ress = ress[idx]

    # keep unique / best
    keep = []
    for i in range(cands.size):
        w = cands[i]
        if len(keep) >= int(keep_best):
            break
        dup = False
        for j in keep:
            if abs(w - cands[j]) <= 1e-10 * (1.0 + abs(cands[j])):
                dup = True
                break
        if not dup:
            keep.append(i)

    cands = cands[keep]
    oks = oks[keep]
    ress = ress[keep]
    return cands, oks, ress


# =====================
# Viterbi (1D tracking)
# =====================

def _viterbi_track(
    z_list,
    cand_list,
    w_prev=None,
    *,
    lam_space=1.0,
    lam_time=0.25,
    lam_asym=0.5,
    lam_tiny_im=0.0,
    tiny_im=1e-7,
    lam_res=0.5,
    edge_k=8,
):
    """
    Track one candidate per z along the 1D grid using DP.

    cand_list: list of arrays of candidates for each iz (variable length)
    Returns: w_path (nz,), ok (nz,)
    """
    nz = z_list.size
    K = max((c.size for c in cand_list), default=0)
    if K == 0:
        return np.full((nz,), np.nan + 1j*np.nan, np.complex128), np.zeros((nz,), bool)

    # pad to rectangular with NaNs
    R = np.full((nz, K), np.nan + 1j*np.nan, dtype=np.complex128)
    for i in range(nz):
        c = cand_list[i]
        if c.size:
            R[i, :c.size] = c

    # unary costs
    unary = np.full((nz, K), np.inf, dtype=np.float64)

    # asymptotic anchors (ends)
    targetL = -1.0 / z_list[0]
    targetR = -1.0 / z_list[-1]

    for i in range(nz):
        zi = z_list[i]
        for k in range(K):
            w = R[i, k]
            if not np.isfinite(w.real) or not np.isfinite(w.imag):
                continue
            c = 0.0

            # residual proxy: prefer smaller |z*w + 1| far from support
            c += lam_asym * float(abs(zi * w + 1.0))

            # time continuity
            if w_prev is not None and np.isfinite(w_prev[i].real) and np.isfinite(w_prev[i].imag):
                c += lam_time * float(abs(w - w_prev[i]))

            # tiny-im penalty (used in pass-2, inside active regions)
            if lam_tiny_im != 0.0:
                im = float(w.imag)
                if im < tiny_im:
                    c += lam_tiny_im * float((tiny_im - im) / max(tiny_im, 1e-30))

            unary[i, k] = c

    # boundary anchoring (stronger at ends)
    if edge_k > 0:
        kk = min(int(edge_k), max(1, nz // 2))
        for i in range(kk):
            unary[i, :] += 10.0 * lam_res * np.abs(R[i, :] - targetL)
        for i in range(nz - kk, nz):
            unary[i, :] += 10.0 * lam_res * np.abs(R[i, :] - targetR)

    # DP
    dp = np.full((nz, K), np.inf, dtype=np.float64)
    prev = np.full((nz, K), -1, dtype=np.int64)

    dp[0, :] = unary[0, :]

    for i in range(1, nz):
        wi = R[i, :]
        wj = R[i - 1, :]
        for k in range(K):
            if not np.isfinite(unary[i, k]) or not np.isfinite(wi[k]):
                continue
            best_val = np.inf
            best_j = -1
            for j in range(K):
                if not np.isfinite(dp[i - 1, j]) or not np.isfinite(wj[j]):
                    continue
                val = dp[i - 1, j] + lam_space * float(abs(wi[k] - wj[j]))
                if val < best_val:
                    best_val = val
                    best_j = j
            if best_j >= 0:
                dp[i, k] = best_val + unary[i, k]
                prev[i, k] = best_j

    k_end = int(np.argmin(dp[-1, :]))
    w_path = np.full((nz,), np.nan + 1j*np.nan, dtype=np.complex128)
    if not np.isfinite(dp[-1, k_end]):
        return w_path, np.zeros((nz,), bool)

    k = k_end
    for i in range(nz - 1, -1, -1):
        w_path[i] = R[i, k]
        k = prev[i, k]
        if i > 0 and k < 0:
            break

    ok = np.isfinite(w_path.real) & np.isfinite(w_path.imag)
    return w_path, ok


# ======================
# Active region detection
# ======================

def _infer_active_mask(w_path, *, imag_floor, q=0.90, pad=8):
    """
    Active region: where imag is meaningfully above floor.
    Two-bulk friendly: uses quantile-based threshold (robust).
    """
    im = np.maximum(np.asarray(w_path.imag, dtype=float), 0.0)
    im_finite = im[np.isfinite(im)]
    if im_finite.size == 0:
        return np.zeros((im.size,), dtype=bool)

    thr = max(float(imag_floor), float(np.quantile(im_finite, q) * 0.10))
    active = im >= thr

    if pad > 0 and active.any():
        idx = np.flatnonzero(active)
        lo = max(0, int(idx[0]) - int(pad))
        hi = min(active.size, int(idx[-1]) + int(pad) + 1)
        active2 = np.zeros_like(active)
        active2[lo:hi] = True

        # also pad each contiguous block
        # (fast enough for nz<=1e4)
        active = active2
    return active


# =====================
# Mass renormalization
# =====================

def _renormalize_density(z_list, w_path, target_mass=1.0):
    """
    Scale imag(w) to match target_mass using trapezoidal rule on x = Re(z).
    """
    x = np.asarray(z_list.real, dtype=float)
    rho = np.maximum(w_path.imag / np.pi, 0.0)
    m = float(np.trapezoid(rho, x))
    if not np.isfinite(m) or m <= 0.0:
        return w_path, m, False
    s = float(target_mass / m)
    # scale only imaginary part (keep Hilbert approx)
    w_new = w_path.real + 1j * (w_path.imag * s)
    return w_new.astype(np.complex128), m, True


# ==========================
# Main decompression API
# ==========================

def decompress_newton(
    z_list: np.ndarray,
    t_grid: np.ndarray,
    a_coeffs: np.ndarray,
    w0_list: np.ndarray | None = None,
    *,
    dt_max: float = 0.05,
    max_iter: int = 80,
    tol: float = 1e-12,
    armijo: bool = True,
    min_lam: float = 1e-6,
    w_min: float = 0.0,
    keep_best: int = 8,
    # branch tracking
    lam_space: float = 1.0,
    lam_time: float = 0.25,
    lam_asym: float = 0.5,
    # pass-2 active region penalty
    active_imag_eps: float = 1e-8,
    lam_tiny_im: float = 5.0,
    active_q: float = 0.90,
    sweep_pad: int = 10,
    # mass
    renorm_mass: bool = True,
    target_mass: float = 1.0,
    # debug
    viterbi_opt: dict | None = None,
    sweep: bool = False,
    time_rel_tol: float = 1e-3,
):
    """
    Solve FD for a set of complex query points z_list over t_grid.

    Parameters (key ones)
    ---------------------
    z_list : (nz,) complex
        Query points (typically x + 1j*delta), in the desired x order.
    t_grid : (nt,) float
        Times (typically log(size ratios)), increasing.
    w0_list : (nz,) complex
        Initial w at t=t_grid[0] (typically physical m(z)).
    renorm_mass : bool
        If True, scales Im(w) at each time to match target_mass.

    Returns
    -------
    W : (nt, nz) complex
    ok : (nt, nz) bool
    """
    z_list = np.asarray(z_list, dtype=np.complex128).ravel()
    t_grid = np.asarray(t_grid, dtype=np.float64).ravel()
    nt = t_grid.size
    nz = z_list.size
    if nz == 0 or nt == 0:
        raise ValueError("z_list and t_grid must be non-empty")

    if w0_list is None:
        w0_list = -1.0 / z_list
    w0_list = np.asarray(w0_list, dtype=np.complex128).ravel()
    if w0_list.size != nz:
        raise ValueError("w0_list must have same length as z_list.")

    # debug / overrides
    vopt = {} if viterbi_opt is None else dict(viterbi_opt)
    lam_space = float(vopt.get("lam_space", lam_space))
    lam_time = float(vopt.get("lam_time", lam_time))
    lam_asym = float(vopt.get("lam_asym", lam_asym))
    lam_tiny_im = float(vopt.get("lam_tiny_im", lam_tiny_im))
    active_q = float(vopt.get("active_q", active_q))
    debug_path = vopt.get("debug_path", None)
    debug_every = int(vopt.get("debug_every", max(1, nt // 10)))
    debug_iz = vopt.get("debug_iz", None)

    W = np.empty((nt, nz), dtype=np.complex128)
    ok = np.ones((nt, nz), dtype=bool)

    W[0, :] = w0_list

    debug_pack = []
    if debug_iz is None:
        debug_iz = []

    for it in range(1, nt):
        t_target = float(t_grid[it])
        t_prev = float(t_grid[it - 1])
        dt = t_target - t_prev
        n_sub = max(1, int(np.ceil(abs(dt) / max(float(dt_max), 1e-12))))
        sub_ts = np.linspace(t_prev, t_target, n_sub + 1)[1:]

        w_prev = W[it - 1].copy()

        # optional "sweep": warm-start by single Newton sweep at same t_target
        # (kept for API compatibility; not required)
        if sweep:
            pass

        for t_sub in sub_ts:
            cand_list = []
            # build candidates for each z
            for iz in range(nz):
                wL = w_prev[iz - 1] if iz > 0 else None
                wR = w_prev[iz + 1] if iz + 1 < nz else None
                seeds = _make_default_seeds(z_list[iz], w_prev[iz], wL, wR)

                # also: push-forward seed from characteristic (often helps)
                tau = float(np.expm1(float(t_sub)))
                # w solves at same z, so zeta ~ z - tau*w_prev
                seeds.append(complex(w_prev[iz]))  # already
                seeds.append(complex(-1.0 / (z_list[iz] - tau * w_prev[iz] + 1e-30)))

                cands, oks, ress = _fd_candidates(
                    z_list[iz], float(t_sub), a_coeffs, seeds,
                    max_iter=max_iter, tol=tol,
                    armijo=armijo, min_lam=min_lam, w_min=w_min,
                    keep_best=keep_best,
                )
                # keep only finite
                cand_list.append(cands)

            # pass-1 viterbi (smooth/continuous)
            w1, ok1 = _viterbi_track(
                z_list, cand_list, w_prev,
                lam_space=lam_space,
                lam_time=lam_time,
                lam_asym=lam_asym,
                lam_tiny_im=0.0,
                tiny_im=active_imag_eps,
                edge_k=int(vopt.get("edge_k", 8)),
            )

            # infer active regions (for 2-bulk cases)
            active = _infer_active_mask(
                w1,
                imag_floor=max(active_imag_eps, 1e-12),
                q=active_q,
                pad=int(vopt.get("sweep_pad", sweep_pad)),
            )

            # pass-2 viterbi: penalize tiny imag inside active region only
            if lam_tiny_im != 0.0 and np.any(active):
                cand_list2 = cand_list  # same candidates
                # create per-node tiny-im penalty by splitting into blocks:
                # we'll do it by running viterbi twice: first on full, then
                # override tiny-im penalty by masking.
                # implement by modifying candidates by adding imaginary floor costs
                # in unary: easiest is to run a custom viterbi with per-index lam.
                # We'll approximate by selecting w2 where active via high penalty,
                # else use pass-1 result.
                # To keep it simple/robust: run global viterbi with penalty, but
                # only at active indices.
                # We'll do it by temporarily replacing non-active candidates with
                # the best-by-residual (pass-1) to avoid over-penalizing gap.
                cand_mod = []
                for i in range(nz):
                    if active[i]:
                        cand_mod.append(cand_list2[i])
                    else:
                        # keep only the best candidate close to pass-1
                        c = cand_list2[i]
                        if c.size == 0:
                            cand_mod.append(c)
                        else:
                            j = int(np.argmin(np.abs(c - w1[i])))
                            cand_mod.append(c[j:j+1])

                w2, ok2 = _viterbi_track(
                    z_list, cand_mod, w_prev,
                    lam_space=lam_space,
                    lam_time=lam_time,
                    lam_asym=lam_asym,
                    lam_tiny_im=lam_tiny_im,
                    tiny_im=max(active_imag_eps, 1e-12),
                    edge_k=int(vopt.get("edge_k", 8)),
                )
            else:
                w2, ok2 = w1, ok1

            # finalize this substep
            w_prev = w2
            ok_sub = ok2

            # mass renorm (optional)
            mass0 = np.nan
            if renorm_mass:
                w_prev, mass0, _ = _renormalize_density(z_list, w_prev, target_mass=target_mass)

            # debug snapshot
            if debug_path is not None and ((it % debug_every) == 0):
                pack = {
                    "it": int(it),
                    "t": float(t_sub),
                    "z_real": z_list.real.copy(),
                    "w": w_prev.copy(),
                    "ok": ok_sub.copy(),
                    "active": active.copy(),
                    "mass": float(mass0) if np.isfinite(mass0) else np.nan,
                }
                if debug_iz:
                    pack["debug_iz"] = np.asarray(debug_iz, dtype=int)
                    pack["w_debug"] = w_prev[np.asarray(debug_iz, dtype=int)].copy()
                debug_pack.append(pack)

        W[it, :] = w_prev
        ok[it, :] = np.isfinite(w_prev.real) & np.isfinite(w_prev.imag)

    if debug_path is not None and len(debug_pack) > 0:
        try:
            # store as npz with object array
            os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
            np.savez_compressed(debug_path, debug=np.array(debug_pack, dtype=object))
        except Exception:
            pass

    return W, ok
