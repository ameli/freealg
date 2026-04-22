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
from ._edge7 import evolve_edges, evolve_edges_from_states, \
        scan_edges_at_time, third_cusp_residual

__all__ = ['cusp_wrap']


# ========
# norm inf
# ========

def _norm_inf(vec):
    if vec is None:
        return numpy.inf
    v = numpy.asarray(vec, dtype=float).ravel()
    if v.size == 0:
        return numpy.inf
    return float(numpy.max(numpy.abs(v)))


# ===========
# dedup cusps
# ===========

def _dedup_cusps(cusps, t_tol=1e-6, x_tol=1e-6):
    if not cusps:
        return []

    cusps = sorted(cusps, key=lambda c: (c['t'], c['x']))
    clusters = []
    for c in cusps:
        placed = False
        for cl in clusters:
            r = cl['rep']
            if abs(c['t'] - r['t']) <= t_tol and abs(c['x'] - r['x']) <= x_tol:
                cl['members'].append(c)
                if c['info']['norm_inf_F'] < r['info']['norm_inf_F']:
                    cl['rep'] = c
                placed = True
                break
        if not placed:
            clusters.append({'rep': c, 'members': [c]})
    return [cl['rep'] for cl in clusters]


# ==================
# run solve cusp
# ==================

def _run_solve_cusp(coeffs, t0, z0, y0, t_bounds, max_iter, tol):
    if y0 is not None:
        try:
            y0 = float(numpy.real(y0))
        except Exception:
            y0 = None

    out = solve_cusp(coeffs, t_init=float(t0), zeta_init=float(z0), y_init=y0,
                     t_bounds=t_bounds, zeta_bounds=None,
                     max_iter=max_iter, tol=tol)

    success = bool(out.get('success', False))
    ok = bool(out.get('ok', False)) if 'ok' in out else success
    info = {
        'success': success,
        'ok': ok,
        'norm_inf_F': _norm_inf(out.get('F', None)),
        'message': out.get('message', None),
    }
    if 'n_iter' in out:
        info['n_iter'] = out['n_iter']

    if not success:
        return None

    return {
        't': float(out['t']),
        'x': float(out['x']),
        'info': info,
        'debug': {
            'tau': float(out['tau']) if 'tau' in out else None,
            'zeta': float(out['zeta']) if 'zeta' in out else None,
            'y': float(out['y']) if 'y' in out else None,
        },
    }


# =====================
# branch local minima
# =====================

def _branch_candidate_indices(vals, ok_mask):
    vals = numpy.asarray(vals, dtype=float)
    ok_mask = numpy.asarray(ok_mask, dtype=bool)
    idx_valid = numpy.where(ok_mask & numpy.isfinite(vals))[0]
    if idx_valid.size == 0:
        return []

    out = []
    # local minima among valid interior points
    for p in range(1, idx_valid.size - 1):
        i0 = idx_valid[p - 1]
        i1 = idx_valid[p]
        i2 = idx_valid[p + 1]
        if (vals[i1] <= vals[i0]) and (vals[i1] <= vals[i2]):
            out.append(int(i1))

    # if none found, include the global minimum on the valid segment
    if not out:
        out.append(int(idx_valid[numpy.argmin(vals[idx_valid])]))

    # also include endpoints of each valid segment: births/deaths happen there
    seg_starts = [idx_valid[0]]
    seg_ends = []
    for p in range(1, idx_valid.size):
        if idx_valid[p] != idx_valid[p - 1] + 1:
            seg_ends.append(idx_valid[p - 1])
            seg_starts.append(idx_valid[p])
    seg_ends.append(idx_valid[-1])

    out.extend(int(v) for v in seg_starts)
    out.extend(int(v) for v in seg_ends)

    out = sorted(set(out))
    return out


# ========================
# collect edge candidates
# ========================

def _collect_edge_candidates(t_grid, zeta_hist, y_hist, ok, coeffs, tag):
    cand = []
    nt, m = zeta_hist.shape
    for j in range(m):
        vals = numpy.full(nt, numpy.nan, dtype=float)
        for it in range(nt):
            if not ok[it, j]:
                continue
            z = zeta_hist[it, j]
            y = y_hist[it, j]
            if not (numpy.isfinite(z.real) and numpy.isfinite(z.imag) and
                    numpy.isfinite(y.real) and numpy.isfinite(y.imag)):
                continue
            vals[it] = third_cusp_residual(z.real, y.real, coeffs)

        for it in _branch_candidate_indices(vals, ok[:, j]):
            if not numpy.isfinite(vals[it]):
                continue
            cand.append({
                't0': float(t_grid[it]),
                'z0': float(numpy.real(zeta_hist[it, j])),
                'y0': float(numpy.real(y_hist[it, j])),
                'metric': float(vals[it]),
                'branch': int(j),
                'source': tag,
            })
    return cand


# =====================
# dedup raw candidates
# =====================

def _dedup_raw_candidates(cands, t_tol=1e-4, z_tol=1e-4):
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: (c['t0'], c['z0'], c['metric']))
    kept = []
    for c in cands:
        placed = False
        for k in kept:
            if abs(c['t0'] - k['t0']) <= t_tol and \
                    abs(c['z0'] - k['z0']) <= z_tol:
                if c['metric'] < k['metric']:
                    k.update(c)
                placed = True
                break
        if not placed:
            kept.append(dict(c))
    return kept


# =========
# cusp_wrap
# =========

def cusp_wrap(coeffs, t_grid, support=None, stieltjes=None,
              log=False, delta=1e-5, edge_dt_max=0.1,
              edge_max_iter=30, edge_tol=1e-12,
              max_iter=80, tol=1e-12, verbose=False,
              dedup_t_tol=1e-6, dedup_x_tol=1e-6, max_solutions=None):
    """
    Edge-driven cusp search.

    This wrapper does not scan the whole (t, x) domain. It uses the edge
    equations to generate candidate cusp seeds, then refines those seeds with
    the exact 3-equation cusp solve.
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 2:
        raise ValueError('t_grid must contain at least two points')
    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError('t_grid must be strictly increasing')
    if support is None:
        raise ValueError('support must be provided')

    t_min = float(numpy.min(t_grid))
    t_max = float(numpy.max(t_grid))
    t_bounds = (t_min, t_max)

    # Forward edge evolution from t=0 support.
    f_edges, f_ok, f_zeta, f_y = evolve_edges(
        t_grid, coeffs, support=support, stieltjes=stieltjes,
        delta=delta, dt_max=edge_dt_max, max_iter=edge_max_iter,
        tol=edge_tol, return_preimage=True, log=log)

    # x-span for final-time scan.
    # x_fin = numpy.real(f_edges[-1, numpy.isfinite(f_edges[-1, :].real)])
    x_all = numpy.real(f_edges[numpy.isfinite(f_edges.real)])
    if x_all.size == 0:
        return []
    x_min = float(numpy.min(x_all))
    x_max = float(numpy.max(x_all))
    pad = 0.05 * max(1.0, x_max - x_min)

    # Scan all edges at final time directly from the fixed-time edge equations.
    x_scan, z_scan, y_scan = scan_edges_at_time(
        t_max, coeffs, (x_min - pad, x_max + pad), n_scan=2048,
        stieltjes=stieltjes, delta=delta, max_iter=edge_max_iter,
        tol=edge_tol, log=log, dedup_x_tol=max(dedup_x_tol, 1e-6))

    # Backward edge evolution from the scanned final-time states.
    if z_scan.size > 0:
        t_rev = t_grid[::-1].copy()
        b_edges_rev, b_ok_rev, b_zeta_rev, b_y_rev = evolve_edges_from_states(
            t_rev, coeffs, z_scan, y_scan, max_iter=edge_max_iter,
            tol=edge_tol, return_preimage=True, log=log)
        # b_edges = b_edges_rev[::-1, :]
        b_ok = b_ok_rev[::-1, :]
        b_zeta = b_zeta_rev[::-1, :]
        b_y = b_y_rev[::-1, :]
    else:
        # b_edges = numpy.empty((t_grid.size, 0), dtype=numpy.complex128)
        b_ok = numpy.empty((t_grid.size, 0), dtype=bool)
        b_zeta = numpy.empty((t_grid.size, 0), dtype=numpy.complex128)
        b_y = numpy.empty((t_grid.size, 0), dtype=numpy.complex128)

    raw = []
    raw += _collect_edge_candidates(t_grid, f_zeta, f_y, f_ok, coeffs,
                                    'forward')
    raw += _collect_edge_candidates(t_grid, b_zeta, b_y, b_ok, coeffs,
                                    'backward')
    raw = _dedup_raw_candidates(raw, t_tol=max(dedup_t_tol, 1e-4),
                                z_tol=max(dedup_x_tol, 1e-4))

    # Sort by edge-based third-equation residual and keep a modest candidate
    # set.
    raw = sorted(raw, key=lambda c: (c['metric'], c['t0'], c['z0']))
    if max_solutions is None:
        max_candidates = min(32, len(raw))
    else:
        max_candidates = min(max(8, 4 * int(max_solutions)), len(raw))
    raw = raw[:max_candidates]

    if verbose:
        print(f'[cusp_wrap] forward branches={f_zeta.shape[1]} '
              f'backward branches={b_zeta.shape[1]} candidates={len(raw)}')

    sols = []
    F_tol = max(1e-10, 100.0 * tol)
    for c in raw:
        sol = _run_solve_cusp(coeffs, t0=c['t0'], z0=c['z0'], y0=c['y0'],
                              t_bounds=t_bounds, max_iter=max_iter, tol=tol)
        if sol is None:
            continue
        if not numpy.isfinite(sol['info']['norm_inf_F']):
            continue
        if sol['info']['norm_inf_F'] > F_tol:
            continue
        # allow t very near the lower boundary but not outside range
        if sol['t'] < t_min - 1e-12 or sol['t'] > t_max + 1e-12:
            continue
        sols.append(sol)

    sols = _dedup_cusps(sols, t_tol=dedup_t_tol, x_tol=dedup_x_tol)
    sols = sorted(sols, key=lambda s: (s['t'], s['x']))
    if max_solutions is not None:
        sols = sols[:int(max_solutions)]

    if verbose:
        print(f'[cusp_wrap] solutions={len(sols)}')

    return sols
