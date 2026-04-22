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
from ._cusp import solve_cusp
# from ._edge3_2 import evolve_edges, merge_edges
from ._edge4 import evolve_edges, merge_edges

__all__ = ['cusp_wrap']


# ========
# norm inf
# ========

def _norm_inf(vec):
    """
    """

    if vec is None:
        return numpy.inf
    v = numpy.asarray(vec, dtype=float).ravel()
    if v.size == 0:
        return numpy.inf
    return float(numpy.max(numpy.abs(v)))


# ========
# seed key
# ========

def _seed_key(t0, z0, zb0, zb1, nd=12):
    """
    """

    # used only to reduce identical seeds; not for cusp dedup
    return (round(float(t0), nd), round(float(z0), nd),
            round(float(zb0), 8), round(float(zb1), 8))


# ===========
# dedup cusps
# ===========

def _dedup_cusps(cusps, t_tol=1e-6, x_tol=1e-6):
    """
    Deduplicate cusps by clustering in (t, x).
    Keeps the lowest-residual representative per cluster.
    """

    if not cusps:
        return []

    # sort by (t, x) for stable clustering
    cusps = sorted(cusps, key=lambda c: (c["t"], c["x"]))

    clusters = []  # each: {"rep": cusp, "members": [...]}

    for c in cusps:
        placed = False
        for cl in clusters:
            r = cl["rep"]
            if abs(c["t"] - r["t"]) <= t_tol and abs(c["x"] - r["x"]) <= x_tol:
                cl["members"].append(c)
                # keep best (smallest norm_inf_F)
                if c["info"]["norm_inf_F"] < r["info"]["norm_inf_F"]:
                    cl["rep"] = c
                placed = True
                break
        if not placed:
            clusters.append({"rep": c, "members": [c]})

    return [cl["rep"] for cl in clusters]


# ====
# edge
# ====

def _edge(t, coeffs, support=None, stieltjes=None, log=False):
    """
    Returns edges that is already merged and real.
    """

    # TEST
    delta = 1e-3
    dt_max = 0.1
    max_iter = 30
    tol = 1e-12

    complex_edges, _ = evolve_edges(
        t, coeffs, support=support, stieltjes=stieltjes, delta=delta,
        dt_max=dt_max, max_iter=max_iter, tol=tol, log=log)

    real_edges = complex_edges.real

    # Remove spurious edges / merges for plotting
    real_merged_edges, _ = merge_edges(real_edges, tol=1e-4)

    return complex_edges, real_merged_edges


# =====================
# make edge based seeds
# =====================

def _make_edge_based_seeds(coeffs, t_grid, support=None,
                           stieltjes=None, log=False, max_take=10):
    """
    Build seeds from all adjacent gaps if edges show >=2 bulks at any time.
    Seed zeta at the midpoint of the smallest gaps.
    """

    seeds = []

    try:
        ce, re = _edge(t_grid, coeffs, support=support,
                           stieltjes=stieltjes, log=log)
    except Exception:
        return seeds

    if re is None or re.ndim != 2 or re.shape[0] == 0:
        return seeds

    if re.shape[1] < 4:
        return seeds  # no adjacent gaps exist

    m = re.shape[1]
    kmax = m // 2

    gap_list = []
    meta = []  # (i, j, b, a)

    for i in range(re.shape[0]):
        row = re[i, :]
        for j in range(kmax - 1):
            b = row[2*j + 1]
            a = row[2*j + 2]
            if numpy.isfinite(a) and numpy.isfinite(b) and (a > b):
                gap_list.append(float(a - b))
                meta.append((i, j, float(b), float(a)))

    if not gap_list:
        return seeds

    order = numpy.argsort(numpy.asarray(gap_list, dtype=float))
    take = min(max_take, order.size)

    for idx in order[:take]:
        i, j, b, a = meta[int(idx)]
        t0 = float(t_grid[i])
        z0 = 0.5 * (a + b)
        seeds.append((t0, z0, None, (b, a)))

    return seeds


# ==================
# make generic seeds
# ==================

def _make_generic_seeds(coeffs, t_grid, support=None,
                        stieltjes=None, log=False, t_count=9,
                        q=(0.2, 0.5, 0.8)):
    """
    Generic multistart: choose a few t0 values and zeta quantiles in the outer
    support. Works for split (k=1 -> 2) as well as a fallback for anything.
    """

    seeds = []
    t_min = float(numpy.min(t_grid))
    t_max = float(numpy.max(t_grid))

    t_seeds = numpy.linspace(t_min, t_max, min(t_count, t_grid.size))
    for t0 in t_seeds:
        try:
            ce0, re0 = _edge(numpy.array([float(t0)]), coeffs, support=support,
                            stieltjes=stieltjes, log=log)
            row = re0[0] if (re0 is not None and re0.shape[1] >= 2) else \
                numpy.real(ce0[0])
            a0 = float(row[0])
            b0 = float(row[1])
        except Exception:
            continue

        if not (numpy.isfinite(a0) and numpy.isfinite(b0) and (b0 > a0)):
            continue

        for qq in q:
            z0 = a0 + float(qq) * (b0 - a0)
            seeds.append((float(t0), float(z0), None, (a0, b0)))

    return seeds


# ============
# unique seeds
# ============

def _unique_seeds(seeds):
    """
    """

    uniq = []
    seen = set()
    for (t0, z0, y0, zb) in seeds:
        key = _seed_key(t0, z0, zb[0], zb[1])
        if key not in seen:
            seen.add(key)
            uniq.append((t0, z0, y0, zb))
    return uniq


# ==============
# run solve cusp
# ==============

def _run_solve_cusp(coeffs, t0, z0, y0, t_bounds, zeta_bounds, max_iter, tol):
    """
    Calls solve_cusp and extracts compact debugging info.
    """

    out = solve_cusp(coeffs, t_init=float(t0), zeta_init=float(z0), y_init=y0,
                     t_bounds=t_bounds, zeta_bounds=zeta_bounds,
                     max_iter=max_iter, tol=tol)

    success = bool(out.get("success", False))
    ok = bool(out.get("ok", False)) if "ok" in out else success

    info = {
        "success": success,
        "ok": ok,
        "norm_inf_F": _norm_inf(out.get("F", None)),
        "message": out.get("message", None),
    }

    # If solve_cusp reports iterations, keep it (optional)
    if "n_iter" in out:
        info["n_iter"] = out["n_iter"]

    if not success:
        return None

    # Minimal actionable payload
    return {
        "t": float(out["t"]),
        "x": float(out["x"]),
        "info": info,
        # keep a few internal vars only if useful for debugging
        "debug": {
            "tau": float(out["tau"]) if "tau" in out else None,
            "zeta": float(out["zeta"]) if "zeta" in out else None,
            "y": float(out["y"]) if "y" in out else None,
        },
    }


# =========
# cusp_wrap
# =========

def cusp_wrap(coeffs, t_grid, support=None, stieltjes=None,
              log=False, max_iter=80, tol=1e-12, verbose=False,
              dedup_t_tol=1e-6, dedup_x_tol=1e-6, max_solutions=None):
    """
    Find cusp points and return a list of independent cusps.

    Returns
    -------

    cusps : list of dict
        Each item has:
          - 't': float
          - 'x': float
          - 'info': dict (compact diagnostics)
          - 'debug': dict (optional internal vars)
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 2:
        raise ValueError("t_grid must contain at least two points")

    t_min = float(numpy.min(t_grid))
    t_max = float(numpy.max(t_grid))
    t_bounds = (t_min, t_max)

    # Build seeds
    seeds = []
    seeds += _make_edge_based_seeds(
        coeffs, t_grid, support=support, stieltjes=stieltjes, log=log,
        max_take=12)
    seeds += _make_generic_seeds(
        coeffs, t_grid, support=support, stieltjes=stieltjes, log=log,
        t_count=9)

    seeds = _unique_seeds(seeds)

    if verbose:
        print(f"[cusp_wrap] seeds={len(seeds)}  t in [{t_min}, {t_max}]")

    # Run solver on all seeds
    sols = []
    for (t0, z0, y0, zb) in seeds:
        sol = _run_solve_cusp(coeffs, t0=t0, z0=z0, y0=y0, t_bounds=t_bounds,
                              zeta_bounds=zb, max_iter=max_iter, tol=tol)

        if sol is None:
            continue

        # Optional quality filter: reject very weak convergences (May need to
        # tune this; this is a safe-ish default.)
        if not numpy.isfinite(sol["info"]["norm_inf_F"]):
            continue

        # Require true convergence
        F_tol = max(1e-10, 1e4 * tol)
        if not sol["info"].get("ok", False):
            if sol["info"]["norm_inf_F"] > F_tol:
                continue

        # ignore boundary-hugging solutions
        t_eps = 1e-8 * (t_max - t_min)
        if sol["t"] <= t_min + t_eps or sol["t"] >= t_max - t_eps:
            continue

        sols.append(sol)

    # De-duplicate independent cusps
    sols = _dedup_cusps(sols, t_tol=dedup_t_tol, x_tol=dedup_x_tol)

    # Sort by time for stable output
    sols = sorted(sols, key=lambda s: (s["t"], s["x"]))

    # Optionally cap number of reported solutions
    if max_solutions is not None:
        sols = sols[:int(max_solutions)]

    if verbose:
        print(f"[cusp_wrap] solutions={len(sols)}")

    return sols
