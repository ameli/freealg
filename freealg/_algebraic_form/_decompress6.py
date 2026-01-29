# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
"""
FD decompression with correct characteristic map + robust root selection.

Keeps public API:
  - build_time_grid(size, n0, min_n_times=..., include_t0=True) -> (t_all, idx_req)
  - decompress_newton(z_list, t_grid, a_coeffs, w0_list=None, **newton_opt) -> (W, ok)

IMPORTANT: This implements the characteristic transform consistent with:
  τ(t)=e^t,  α(t)=1-τ^{-1},
  P(z + α w^{-1}, τ w) = 0,
where P(ζ,y)=0 is the algebraic relation for m0.

We construct a polynomial in w:
  Q(w) := w^{deg_z} * P(z + α/w, τ w),
which has degree deg_z + deg_m (no artificial extra zero roots).

Root selection:
  - Herglotz (your sign): Im(w) >= -herglotz_tol for Im(z)>0
  - Homotopy anchor in η: start at η_hi, track down to η_lo
  - Filter roots near anchor, then Viterbi along x
"""

from __future__ import annotations

import math
import numpy as np

__all__ = ["build_time_grid", "decompress_newton"]


def _inside_support_mask(x: np.ndarray, edges_row: np.ndarray, pad: float) -> np.ndarray:
    """
    edges_row: [a1,b1,a2,b2,...] with NaNs allowed (ghost edges).
    Returns mask for x inside union of intervals, with optional padding.
    """
    mask = np.zeros_like(x, dtype=bool)
    m = edges_row.size
    for j in range(0, m, 2):
        a = edges_row[j]
        b = edges_row[j+1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        aa = float(a) - float(pad)
        bb = float(b) + float(pad)
        mask |= (x >= aa) & (x <= bb)
    return mask



def build_time_grid(size, n0, min_n_times=0, include_t0=True):
    n0 = float(n0)
    size = np.asarray(size, dtype=float).ravel()
    if size.size == 0:
        if include_t0:
            return np.array([0.0], dtype=float), np.array([0], dtype=int)
        return np.empty((0,), dtype=float), np.empty((0,), dtype=int)

    t_sizes = np.log(size / n0)

    t_req = t_sizes.copy()
    if include_t0:
        t_req = np.concatenate((np.array([0.0], dtype=float), t_req))

    t_req = np.unique(t_req)
    t_req.sort()

    if int(min_n_times) > 0 and t_req.size < int(min_n_times):
        t_all = np.linspace(float(t_req[0]), float(t_req[-1]), int(min_n_times))
    else:
        t_all = t_req

    t_all = np.unique(t_all)
    t_all.sort()

    idx_req = np.array([int(np.argmin(np.abs(t_all - ts))) for ts in t_sizes], dtype=int)
    return t_all, idx_req


# ===========================
# Polynomial curve utilities
# ===========================

def _poly_w_coeffs(z: complex, t: float, a_coeffs: np.ndarray) -> np.ndarray:
    """
    Build Q(w) coeffs (descending) for:
      Q(w) = w^{deg_z} * P(z + α/w, τ w)
    where τ=e^t, α=1-1/τ.
    """
    a = np.asarray(a_coeffs, dtype=np.complex128)
    deg_z = a.shape[0] - 1
    deg_m = a.shape[1] - 1

    tau = math.exp(float(t))
    alpha = 1.0 - 1.0 / tau
    z = complex(z)

    deg_Q = deg_z + deg_m
    c = np.zeros((deg_Q + 1,), dtype=np.complex128)  # ascending

    # term: a_{i,j} (z + alpha/w)^i (tau*w)^j
    # expand (z + alpha/w)^i = sum_{k=0}^i C(i,k) z^{i-k} (alpha/w)^k
    # multiply by w^{deg_z}: exponent of w is deg_z + j - k (>=0).
    for i in range(deg_z + 1):
        for j in range(deg_m + 1):
            aij = a[i, j]
            if aij == 0:
                continue
            for k in range(i + 1):
                p = deg_z + j - k
                if p < 0 or p > deg_Q:
                    continue
                c[p] += aij * math.comb(i, k) * (z ** (i - k)) * ((alpha) ** k) * (tau ** j)

    nz = np.flatnonzero(np.abs(c) > 0)
    if nz.size == 0:
        return np.array([0.0], dtype=np.complex128)
    p_max = int(nz.max())
    c = c[:p_max + 1]
    return c[::-1].copy()


def _herglotz_ok(w: complex, z: complex, tol: float = 0.0) -> bool:
    z = complex(z)
    w = complex(w)
    if z.imag <= 0.0:
        return True
    return (w.imag >= -float(tol))


def _asym_score(z: complex, w: complex) -> float:
    return float(abs(complex(z) * complex(w) + 1.0))


def _newton_poly_root(coeff_desc: np.ndarray, w0: complex, max_iter: int, tol: float,
                      armijo: bool = True, min_lam: float = 1e-4):
    w = complex(w0)
    dcoeff = np.polyder(coeff_desc)
    for _ in range(int(max_iter)):
        f = np.polyval(coeff_desc, w)
        if not np.isfinite(f):
            return w, False
        if abs(f) <= float(tol) * (1.0 + abs(w)):
            return w, True
        df = np.polyval(dcoeff, w)
        if (not np.isfinite(df)) or df == 0:
            return w, False
        step = f / df

        if not armijo:
            w = w - step
            continue

        f0 = abs(f)
        lam = 1.0
        while lam >= float(min_lam):
            w_try = w - lam * step
            f_try = np.polyval(coeff_desc, w_try)
            if np.isfinite(f_try) and abs(f_try) <= (1.0 - 0.5 * lam) * f0:
                w = w_try
                break
            lam *= 0.5
        else:
            w = w - float(min_lam) * step

    f = np.polyval(coeff_desc, w)
    return w, bool(np.isfinite(f) and abs(f) <= float(tol) * (1.0 + abs(w)))


def _roots_of_Q(z: complex, t: float, a_coeffs: np.ndarray):
    coeff_desc = _poly_w_coeffs(z, t, a_coeffs)
    if coeff_desc.size <= 1:
        return coeff_desc, np.empty((0,), np.complex128)
    r = np.roots(coeff_desc)
    r = r[np.isfinite(r)]
    return coeff_desc, r.astype(np.complex128, copy=False)


def _physical_anchor_for_x(x: float, t: float, a_coeffs: np.ndarray,
                           eta_hi: float, eta_lo: float, n_eta: int,
                           herglotz_tol: float,
                           max_iter: int, tol: float,
                           armijo: bool, min_lam: float):
    etas = np.linspace(float(eta_hi), float(eta_lo), int(n_eta))
    z0 = complex(x, etas[0])

    coeff0, roots0 = _roots_of_Q(z0, t, a_coeffs)
    if roots0.size == 0:
        return -1.0 / z0, False

    # pick best among Herglotz candidates by asymptotic score
    good = [w for w in roots0 if _herglotz_ok(w, z0, herglotz_tol)]
    if len(good) == 0:
        good = list(roots0)
    good = np.asarray(good, dtype=np.complex128)
    sc = np.array([_asym_score(z0, w) for w in good], dtype=float)
    w = good[int(np.argmin(sc))]

    # refine and track down
    w, _ = _newton_poly_root(coeff0, w, max_iter=max_iter, tol=tol, armijo=armijo, min_lam=min_lam)

    for eta in etas[1:]:
        z = complex(x, eta)
        coeff, _ = _roots_of_Q(z, t, a_coeffs)
        w, ok2 = _newton_poly_root(coeff, w, max_iter=max_iter, tol=tol, armijo=armijo, min_lam=min_lam)
        if not ok2:
            _coeff, roots = _roots_of_Q(z, t, a_coeffs)
            if roots.size == 0:
                return w, False
            roots = np.asarray(roots, dtype=np.complex128)
            mask = np.array([_herglotz_ok(ww, z, herglotz_tol) for ww in roots], dtype=bool)
            cand = roots[mask] if np.any(mask) else roots
            jj = int(np.argmin(np.abs(cand - w)))
            w = cand[jj]
            w, ok2 = _newton_poly_root(coeff, w, max_iter=max_iter, tol=tol, armijo=armijo, min_lam=min_lam)
            if not ok2:
                return w, False

    return w, True


def _candidate_filter(roots: np.ndarray, z: complex, herglotz_tol: float,
                      anchor: complex | None, anchor_radius: float,
                      *, w_min: float = 0.0,
                      im_floor: float | None = None):
    if roots.size == 0:
        return np.empty((0,), np.complex128)
    keep = []
    for w in roots:
        if abs(w) <= float(w_min):
            continue
        if not _herglotz_ok(w, z, herglotz_tol):
            continue
        if im_floor is not None and z.imag > 0.0:
            if w.imag < float(im_floor):
                continue
        if anchor is not None:
            if abs(w - anchor) > float(anchor_radius) * (1.0 + abs(anchor)):
                continue
        keep.append(complex(w))
    if len(keep) == 0:
        return np.empty((0,), np.complex128)
    # dedup
    out = []
    for w in keep:
        if all(abs(w-u) > 1e-9*(1.0+abs(u)) for u in out):
            out.append(w)
    return np.asarray(out, dtype=np.complex128)


def _viterbi_path(cand_list, z_list, w_prev,
                 lam_time, lam_space, lam_asym, lam_im2,
                 edge_k):
    nz = len(cand_list)
    chosen = np.empty((nz,), dtype=np.complex128)
    ok = np.ones((nz,), dtype=bool)
    big = 1e300

    sizes = np.array([c.size for c in cand_list], dtype=int)
    for i in range(nz):
        if sizes[i] == 0:
            chosen[i] = w_prev[i]
            ok[i] = False

    i = 0
    while i < nz:
        if sizes[i] == 0:
            i += 1
            continue
        j = i
        while j < nz and sizes[j] > 0:
            j += 1

        block = list(range(i, j))
        c0 = cand_list[i]
        m0 = c0.size

        dp = np.full((m0,), big, dtype=float)
        bp = [None] * (j - i)

        node = lam_time * (np.abs(c0 - w_prev[i]) ** 2)
        if edge_k > 0 and (i < edge_k or i >= nz - edge_k):
            node = node + lam_asym * np.array([_asym_score(z_list[i], w) for w in c0], dtype=float)
        if lam_im2 != 0.0:
            node = node + lam_im2 * (np.imag(c0) ** 2)
        dp = node
        bp[0] = np.full((m0,), -1, dtype=int)

        for kpos, idx in enumerate(block[1:], start=1):
            ck = cand_list[idx]
            mk = ck.size
            new_dp = np.full((mk,), big, dtype=float)
            new_bp = np.full((mk,), -1, dtype=int)

            node = lam_time * (np.abs(ck - w_prev[idx]) ** 2)
            if edge_k > 0 and (idx < edge_k or idx >= nz - edge_k):
                node = node + lam_asym * np.array([_asym_score(z_list[idx], w) for w in ck], dtype=float)
            if lam_im2 != 0.0:
                node = node + lam_im2 * (np.imag(ck) ** 2)

            prev_c = cand_list[idx - 1]
            for q in range(mk):
                vals = dp + lam_space * (np.abs(ck[q] - prev_c) ** 2)
                best = int(np.argmin(vals))
                new_dp[q] = float(vals[best] + node[q])
                new_bp[q] = best

            dp = new_dp
            bp[kpos] = new_bp

        end_q = int(np.argmin(dp))
        for kpos in range(len(block) - 1, -1, -1):
            idx = block[kpos]
            chosen[idx] = cand_list[idx][end_q]
            end_q = int(bp[kpos][end_q])

        i = j

    return chosen, ok


# =====================
# Main decompression API
# =====================

def decompress_newton(
    z_list,
    t_grid,
    a_coeffs,
    w0_list=None,
    *,
    dt_max=0.05,
    return_success_rate=False,
    viterbi=True,
    viterbi_opt=None,
    # edge-guided selection (optional)
    edge_use=False,
    edge_support=None,
    edge_pad=0.0,
    im_floor_rel=0.15,
    w_min=1e-14,
    # homotopy (eta) options
    eta_hi=3.0,
    n_eta=24,
    anchor_radius=0.6,
    # newton options for homotopy
    max_iter=60,
    tol=1e-12,
    armijo=True,
    min_lam=1e-4,
    # herglotz convention
    herglotz_tol=0.0,
    **_,
):
    z_list = np.asarray(z_list, dtype=np.complex128).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if z_list.size == 0 or t_grid.size == 0:
        raise ValueError("z_list and t_grid must be non-empty")

    t_grid = np.unique(t_grid)
    t_grid.sort()
    if np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing")

    nt = t_grid.size
    nz = z_list.size
    x = z_list.real
    eta_lo = float(np.median(z_list.imag))

    real_edges = None
    if edge_use:
        try:
            from ._edge import evolve_edges, merge_edges
            if edge_support is None:
                raise ValueError("edge_support must be provided when edge_use=True")
            complex_edges = evolve_edges(t_grid, a_coeffs, support=edge_support)
            # merge_edges in your package expects edges array (nt, 2k) and returns (real_merged_edges, active_k)
            real_edges, _active_k = merge_edges(complex_edges, t_grid)
        except Exception:
            real_edges = None

    if w0_list is None:
        w0_list = -1.0 / z_list
    w_prev = np.asarray(w0_list, dtype=np.complex128).ravel()
    if w_prev.size != nz:
        raise ValueError("w0_list length must match z_list")

    vopt = {} if viterbi_opt is None else dict(viterbi_opt)
    lam_time = float(vopt.get("lam_time", 0.25))
    lam_space = float(vopt.get("lam_space", 1.0))
    lam_asym = float(vopt.get("lam_asym", 0.2))
    lam_im2 = float(vopt.get("lam_im2", 0.0))
    edge_k = int(vopt.get("edge_k", 8))

    W = np.empty((nt, nz), dtype=np.complex128)
    ok = np.ones((nt, nz), dtype=bool)
    W[0, :] = w_prev
    success = np.ones((nt,), dtype=float)
    success[0] = 1.0

    for it in range(1, nt):
        t1 = float(t_grid[it])
        t0 = float(t_grid[it - 1])
        dt = t1 - t0
        n_sub = max(1, int(np.ceil(abs(dt) / max(float(dt_max), 1e-12))))
        sub_ts = np.linspace(t0, t1, n_sub + 1)[1:]

        ok_row = np.ones((nz,), dtype=bool)

        for t_sub in sub_ts:
            anchors = np.empty((nz,), dtype=np.complex128)
            anchor_ok = np.ones((nz,), dtype=bool)
            for iz in range(nz):
                w_a, ok_a = _physical_anchor_for_x(
                    float(x[iz]), float(t_sub), a_coeffs,
                    eta_hi=float(eta_hi), eta_lo=float(eta_lo),
                    n_eta=int(n_eta),
                    herglotz_tol=float(herglotz_tol),
                    max_iter=int(max_iter), tol=float(tol),
                    armijo=bool(armijo), min_lam=float(min_lam),
                )
                anchors[iz] = w_a
                anchor_ok[iz] = ok_a

            cand_list = []
            for iz in range(nz):
                z = z_list[iz]
                coeff, roots = _roots_of_Q(z, float(t_sub), a_coeffs)

                anc = anchors[iz]
                im_floor = None
                if real_edges is not None:
                    inside = _inside_support_mask(np.array([float(x[iz])]), real_edges[int(np.argmin(np.abs(t_grid - float(t_sub))))], pad=float(edge_pad))[0]
                    if inside:
                        # require a fraction of the max positive imaginary root to avoid collapsing early
                        im_pos = np.max(np.maximum(0.0, roots.imag)) if roots.size > 0 else 0.0
                        if im_pos > 0.0:
                            im_floor = float(im_floor_rel) * float(im_pos)
                cands = _candidate_filter(
                    roots, z, herglotz_tol=float(herglotz_tol),
                    anchor=anc if anchor_ok[iz] else None,
                    anchor_radius=float(anchor_radius),
                    w_min=float(w_min),
                    im_floor=im_floor,
                )

                # fallback: take nearest to anchor
                if cands.size == 0 and roots.size > 0:
                    roots = roots.astype(np.complex128, copy=False)
                    roots2 = roots[np.abs(roots) > float(w_min)]
                    if roots2.size == 0:
                        roots2 = roots
                    jj = int(np.argmin(np.abs(roots2 - anc)))
                    cands = np.array([roots2[jj]], dtype=np.complex128)

                # include refined anchor if it lands on a root
                if coeff.size > 1:
                    anc_ref, ok_ref = _newton_poly_root(coeff, anc, max_iter=max_iter, tol=tol,
                                                        armijo=armijo, min_lam=min_lam)
                    if ok_ref and _herglotz_ok(anc_ref, z, tol=herglotz_tol):
                        cands = np.unique(np.concatenate((cands, np.array([anc_ref], dtype=np.complex128))))

                cand_list.append(cands)

            if viterbi:
                w_path, ok_path = _viterbi_path(
                    cand_list, z_list, w_prev,
                    lam_time, lam_space, lam_asym, lam_im2, edge_k
                )
                w_prev = w_path
                ok_row &= ok_path
            else:
                new_row = np.empty((nz,), dtype=np.complex128)
                for iz in range(nz):
                    c = cand_list[iz]
                    if c.size == 0:
                        new_row[iz] = w_prev[iz]
                        ok_row[iz] = False
                        continue
                    jj = int(np.argmin(np.abs(c - w_prev[iz]) ** 2))
                    new_row[iz] = c[jj]
                w_prev = new_row

        W[it, :] = w_prev
        ok[it, :] = ok_row
        success[it] = float(np.mean(ok_row))

    if return_success_rate:
        return W, ok, success
    return W, ok
