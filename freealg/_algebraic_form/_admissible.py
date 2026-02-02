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

__all__ = ['precheck_laurent']


# =====
# L add
# =====

def L_add(A, B):

    C = dict(A)
    for p, c in B.items():
        C[p] = C.get(p, 0.0) + c
        if abs(C[p]) == 0:
            C.pop(p, None)
    return C


# =======
# L scale
# =======

def L_scale(A, s):
    return {p: s*c for p, c in A.items()}


# =====
# L mul
# =====

def L_mul(A, B, pmin, pmax):

    C = {}
    for pa, ca in A.items():
        for pb, cb in B.items():
            p = pa + pb
            if p < pmin or p > pmax:
                continue
            C[p] = C.get(p, 0.0) + ca*cb

    return C


# =====
# L pow
# =====

def L_pow(A, k, pmin, pmax):

    if k == 0:
        return {0: 1.0+0.0j}
    if k == 1:
        return dict(A)
    # fast exponentiation
    out = {0: 1.0+0.0j}
    base = dict(A)
    e = k
    while e > 0:
        if e & 1:
            out = L_mul(out, base, pmin, pmax)
        e >>= 1
        if e:
            base = L_mul(base, base, pmin, pmax)
    return out


# ============
# L inv series
# ============

def L_inv_series(A, pmin, pmax):
    """
    Invert a *power series* around power 0: requires A[0] != 0 and no negative
    powers.

    Returns power series B with nonnegative powers only.
    """

    if any(p < 0 for p in A.keys()):
        raise ValueError("L_inv_series expects no negative powers")

    a0 = A.get(0, 0.0)
    if abs(a0) < 1e-18:
        raise ValueError("Need nonzero constant term to invert")

    # compute up to pmax (nonnegative)
    B = {0: 1.0/a0}
    for n in range(1, pmax+1):
        s = 0.0+0.0j
        for k in range(1, n+1):
            s += A.get(k, 0.0) * B.get(n-k, 0.0)
        B[n] = -s/a0

    # clip
    B = {p: c for p, c in B.items() if 0 <= p <= pmax}
    return B


# ==================
# Ptau Laurent alpha
# ==================

def Ptau_Laurent_alpha(a, tau, alpha, mu, pmin, pmax):
    """
    Build Laurent series of P_tau(z,m) at z=1/w,
    m(w)=-(alpha w + mu1 w^2 + ...).

    Returns dict power->coeff for powers in [pmin,pmax].
    """

    # m(w)
    m = {1: -(alpha+0.0j)}
    for k, muk in enumerate(mu, start=1):
        m[k+1] = -(muk+0.0j)

    # y = tau m
    y = L_scale(m, tau)

    # m = w*m1(w) with m1(w) = -(alpha + mu1 w + mu2 w^2 + ...)
    m1 = {0: -(alpha+0.0j)}
    for k, muk in enumerate(mu, start=1):
        m1[k] = -(muk+0.0j)

    inv_m1 = L_inv_series(m1, pmin=0, pmax=max(0, pmax))

    # 1/m = w^{-1} * inv_m1
    inv_m = {p-1: c for p, c in inv_m1.items() if (p-1) >= pmin}

    # z = 1/w
    z = {-1: 1.0+0.0j}

    c = (1.0 - 1.0/tau)
    zeta = L_add(z, L_scale(inv_m, c))  # z + c*(1/m)

    deg_z = a.shape[0]-1
    deg_m = a.shape[1]-1

    out = {}
    zeta_pows = [L_pow(zeta, i, pmin, pmax) for i in range(deg_z+1)]
    y_pows = [L_pow(y,    j, pmin, pmax) for j in range(deg_m+1)]

    for i in range(deg_z+1):
        for j in range(deg_m+1):
            coeff = a[i, j]
            if abs(coeff) < 1e-18:
                continue
            term = L_mul(zeta_pows[i], y_pows[j], pmin, pmax)
            out = L_add(out, L_scale(term, coeff))

    return out


# ======================
# solve laurent alpha ls
# ======================

def solve_laurent_alpha_ls(a, tau, K=8, L=None, max_iter=40, tol_step=1e-12):
    """
    Solve for x = [alpha, mu1..muK] so that Laurent coeffs vanish
    for powers p in [-L, ..., K]. Uses LS on both Re and Im parts (robust).
    """

    deg_z = a.shape[0]-1
    if L is None:
        L = deg_z + 2

    pmin, pmax = -L, K
    powers = list(range(pmin, pmax+1))

    # unknowns: alpha + mu1..muK
    x = numpy.zeros(K+1, dtype=numpy.float64)
    x[0] = 1.0  # alpha init

    def build_out(xx):
        alpha = float(xx[0])
        mu = xx[1:]
        return Ptau_Laurent_alpha(a, tau, alpha, mu, pmin, pmax)

    # LS Newton / Gauss-Newton
    for it in range(max_iter):
        out = build_out(x)

        # complex residual vector r_p = coeff(p)
        r = numpy.array([out.get(p, 0.0+0.0j) for p in powers],
                        dtype=numpy.complex128)

        # stack real+imag (this is the key fix)
        F = numpy.concatenate([r.real, r.imag], axis=0)
        nrm = numpy.linalg.norm(F)

        if nrm < 1e-12:
            return x, True, out, powers

        # Jacobian by FD
        eps = 1e-6
        J = numpy.zeros((F.size, x.size), dtype=numpy.float64)
        for k in range(x.size):
            x2 = x.copy()
            x2[k] += eps
            out2 = build_out(x2)
            r2 = numpy.array([out2.get(p, 0.0+0.0j) for p in powers],
                             dtype=numpy.complex128)
            F2 = numpy.concatenate([r2.real, r2.imag], axis=0)
            J[:, k] = (F2 - F) / eps

        # LS step
        dx, *_ = numpy.linalg.lstsq(J, -F, rcond=None)

        if numpy.linalg.norm(dx) < tol_step:
            return x, False, out, powers

        x += dx

    return x, False, out, powers


# ================
# precheck laurent
# ================

def precheck_laurent(a, tau, K_list=(6, 8, 10), L=3, tol=1e-8, verbose=True):
    """
    For fixed tau, try several K and pick the best (smallest max |coeff|).
    Returns dict with bestK, alpha, max_abs, ok, and per-K details.
    """

    deg_z = a.shape[0]-1
    if L is None:
        L = deg_z + 2

    best = None
    perK = []

    for K in K_list:
        x, ok_solve, out, powers = solve_laurent_alpha_ls(a, tau, K=K, L=L)
        alpha = x[0]
        coeffs = numpy.array([out.get(p, 0.0+0.0j) for p in powers],
                             dtype=numpy.complex128)
        max_abs = float(numpy.max(numpy.abs(coeffs)))
        worst_p = powers[int(numpy.argmax(numpy.abs(coeffs)))]
        perK.append((K, alpha, max_abs, worst_p, ok_solve))

        if best is None or max_abs < best["max_abs"]:
            best = {
                "K": K,
                "alpha": alpha,
                "max_abs": max_abs,
                "worst_p": worst_p,
                "ok_solve": ok_solve
            }

    alphas = numpy.array([row[1] for row in perK], dtype=float)  # alpha per K
    alpha_std = float(numpy.std(alphas))
    alpha_span = float(numpy.max(alphas) - numpy.min(alphas))

    ok = (best["max_abs"] <= tol) and best["ok_solve"]
    ok = ok and (alpha_std < 1e-3)   # or use alpha_span < 3e-3

    if verbose:
        print(f"--- tau={tau} --- ok={ok} bestK={best['K']} "
              f"max_abs={best['max_abs']:.3e} "
              f"alpha={best['alpha']:.12g} worst_p={best['worst_p']}")

        print(f"  alpha_std={alpha_std:.3e}, alpha_span={alpha_span:.3e}, "
              f"alphas={alphas}")

        for (K, alpha, max_abs, worst_p, ok_solve) in perK:
            print(f"  K={K:2d} max_abs={max_abs:.3e} worst_p={worst_p:2d} "
                  f"alpha={alpha:.12g} solve_ok={ok_solve}")

    info = {
        "best": best,
        "perK": perK,
        "alpha_std": alpha_std,
        "alpha_span": alpha_span
    }

    return ok, info
