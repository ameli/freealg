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
from ._continuation_algebraic import eval_roots
from ._decompress_util import eval_P_partials

__all__ = ['evolve_edges', 'merge_edges', 'evolve_edges_with_births']


# ================
# edge newton step
# ================

def _edge_newton_step(t, zeta, y, coeffs, max_iter=30, tol=1e-12):
    """
    """

    tau = float(numpy.exp(t))
    c = tau - 1.0

    for _ in range(max_iter):
        P, Pz, Py = eval_P_partials(zeta, y, coeffs)

        # F1 = P(zeta,y)
        F1 = complex(P)

        # F2 = y^2 Py - c Pz
        F2 = complex((y * y) * Py - c * Pz)

        if max(abs(F1), abs(F2)) <= tol:
            return zeta, y, True

        # Numerical Jacobian (2x2) in (zeta,y)
        eps_z = 1e-8 * (1.0 + abs(zeta))
        eps_y = 1e-8 * (1.0 + abs(y))

        Pp, Pzp, Pyp = eval_P_partials(zeta + eps_z, y, coeffs)
        F1_zp = (complex(Pp) - F1) / eps_z
        F2_zp = (complex((y * y) * Pyp - c * Pzp) - F2) / eps_z

        Pp, Pzp, Pyp = eval_P_partials(zeta, y + eps_y, coeffs)
        F1_yp = (complex(Pp) - F1) / eps_y
        F2_yp = (complex(((y + eps_y) * (y + eps_y)) * Pyp - c * Pzp) - F2) / \
            eps_y

        # Solve J * [dz, dy] = -F
        det = F1_zp * F2_yp - F1_yp * F2_zp
        if det == 0.0:
            return zeta, y, False

        dz = (-F1 * F2_yp + F1_yp * F2) / det
        dy = (-F1_zp * F2 + F1 * F2_zp) / det

        # Mild damping if update is huge
        lam = 1.0
        if abs(dz) + abs(dy) > 10.0 * (1.0 + abs(zeta) + abs(y)):
            lam = 0.2

        zeta = zeta + lam * dz
        y = y + lam * dy

    return zeta, y, False


# ==================
# pick physical root
# ==================

def _pick_physical_root(z, roots):
    """
    Pick the Herglotz/physical root at a point z in C+.

    Heuristic: choose the root with maximal Im(root) when Im(z)>0,
    then enforce Im(root)>0. Falls back to closest-to -1/z if needed.
    """

    r = numpy.asarray(roots, dtype=complex).ravel()
    if r.size == 0:
        return numpy.nan + 1j * numpy.nan

    if z.imag > 0.0:
        pos = r[numpy.imag(r) > 0.0]
        if pos.size > 0:
            return pos[numpy.argmax(numpy.imag(pos))]

    target = -1.0 / z
    return r[numpy.argmin(numpy.abs(r - target))]


# ============================
# init edge point from support
# ============================

def _init_edge_point_from_support(x_edge, coeffs, delta=1e-5):
    """
    Initialize (zeta,y) at t=0 for an edge near x_edge.

    Uses z = x_edge + i*delta, picks physical root y, then refines zeta on real
    axis.
    """

    z = complex(x_edge + 1j * delta)
    roots = eval_roots(numpy.array([z]), coeffs)[0]
    y = _pick_physical_root(z, roots)

    # Move zeta to real axis as initial guess
    zeta = complex(x_edge)

    # Refine zeta,y to satisfy P=0 and Py=0 at t=0 (branch point)
    # This uses the same Newton system with c=0, i.e. F2 = y^2 Py.
    zeta, y, ok = _edge_newton_step(0.0, zeta, y, coeffs, max_iter=50,
                                    tol=1e-10)

    return zeta, y, ok


# ============
# evolve edges
# ============

def evolve_edges(
        t_grid,
        coeffs,
        support=None,
        delta=1e-5,
        dt_max=0.1,
        max_iter=30,
        tol=1e-12,
        return_preimage=False):
    """
    Evolve spectral edges under free decompression using the fitted polynomial
    P.

    Solves for (zeta(t), y(t)) on the spectral curve:
        P(zeta,y) = 0,
        y^2 * Py(zeta,y) - (exp(t)-1) * Pzeta(zeta,y) = 0,

    then maps to physical coordinate:
        z_edge(t) = zeta - (exp(t)-1)/y.

    If return_preimage=True, also returns zeta_hist and y_hist of shape
    (nt, 2k).
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 1:
        raise ValueError("t_grid must be non-empty.")
    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")

    if support is None:
        raise ValueError("support must be provided (auto-detection not " +
                         "implemented).")

    # Flatten endpoints in fixed order [a1,b1,a2,b2,...]
    endpoints0 = []
    for a, b in support:
        endpoints0.append(float(a))
        endpoints0.append(float(b))

    m = len(endpoints0)
    complex_edges = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
    ok = numpy.zeros((t_grid.size, m), dtype=bool)

    if return_preimage:
        zeta_hist = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
        y_hist = numpy.empty((t_grid.size, m), dtype=numpy.complex128)
    else:
        zeta_hist = None
        y_hist = None

    # Initialize (zeta,y) at t=0 from support endpoints
    zeta = numpy.empty(m, dtype=numpy.complex128)
    y = numpy.empty(m, dtype=numpy.complex128)

    for j in range(m):
        z0, y0, ok0 = _init_edge_point_from_support(endpoints0[j], coeffs,
                                                    delta=delta)
        zeta[j] = z0
        y[j] = y0
        ok[0, j] = ok0
        complex_edges[0, j] = z0  # at t=0, tau-1 = 0 => z_edge = zeta

    if return_preimage:
        zeta_hist[0, :] = zeta
        y_hist[0, :] = y

    # Time stepping
    for it in range(1, t_grid.size):
        t0 = float(t_grid[it - 1])
        t1 = float(t_grid[it])
        dt = t1 - t0

        n_sub = int(numpy.ceil(dt / float(dt_max)))
        if n_sub < 1:
            n_sub = 1

        for ks in range(1, n_sub + 1):
            t = t0 + dt * (ks / float(n_sub))
            for j in range(m):
                zeta[j], y[j], okj = _edge_newton_step(
                    t, zeta[j], y[j], coeffs, max_iter=max_iter, tol=tol
                )
                ok[it, j] = okj

        tau = float(numpy.exp(t1))
        c = tau - 1.0
        complex_edges[it, :] = zeta - c / y

        if return_preimage:
            zeta_hist[it, :] = zeta
            y_hist[it, :] = y

    if return_preimage:
        return complex_edges, ok, zeta_hist, y_hist

    return complex_edges, ok


# ===========
# merge edges
# ===========

def merge_edges(edges, tol=0.0):
    """
    Merge bulks when inner edges cross, without shifting columns.

    Columns are fixed as [a1,b1,a2,b2,...,ak,bk]. When the gap between bulk j
    and bulk j+1 closes (b_j >= a_{j+1} - tol), we annihilate the two inner
    edges by setting b_j and a_{j+1} to NaN. All other columns remain in place.

    This preserves smooth plotting per original edge index (e.g. b2 stays in
    the same column for all t). The number of active bulks is computed as the
    number of connected components after merges.

    Parameters
    ----------

    edges : ndarray, shape (nt, 2k)
        Edge trajectories [a1,b1,a2,b2,...].

    tol : float
        Merge tolerance in x-units.

    Returns
    -------

    edges2 : ndarray, shape (nt, 2k)
        Same shape as input. Inner merged edges are NaN. No columns are
        shifted.

    active_k : ndarray, shape (nt,)
        Number of remaining bulks (connected components) at each time.
    """

    edges = numpy.asarray(edges, dtype=float)
    nt, m = edges.shape
    if m % 2 != 0:
        raise ValueError("edges must have even number of columns.")
    k0 = m // 2

    edges2 = edges.copy()
    active_k = numpy.zeros(nt, dtype=int)

    for it in range(nt):
        row = edges2[it, :].copy()
        a = row[0::2].copy()
        b = row[1::2].copy()

        # Initialize blocks as list of (L_index, R_index) in bulk indices.
        blocks = []
        for j in range(k0):
            if numpy.isfinite(a[j]) and numpy.isfinite(b[j]) and (b[j] > a[j]):
                blocks.append([j, j])

        if len(blocks) == 0:
            active_k[it] = 0
            edges2[it, :] = row
            continue

        # Helper to get current left/right edge value of a block.
        def left_edge(block):
            return a[block[0]]

        def right_edge(block):
            return b[block[1]]

        # Iteratively merge adjacent blocks when they overlap / touch.
        merged = True
        while merged and (len(blocks) > 1):
            merged = False
            new_blocks = [blocks[0]]
            for blk in blocks[1:]:
                prev = new_blocks[-1]
                # If right(prev) crosses left(blk), merge.
                if numpy.isfinite(right_edge(prev)) and \
                        numpy.isfinite(left_edge(blk)) and \
                        (right_edge(prev) >= left_edge(blk) - float(tol)):

                    # Annihilate inner boundary edges in fixed columns:
                    # b_{prev.right_bulk} and a_{blk.left_bulk}
                    bj = prev[1]
                    aj = blk[0]
                    b[bj] = numpy.nan
                    a[aj] = numpy.nan

                    # Merge block indices: left stays prev.left, right becomes
                    # blk.right
                    prev[1] = blk[1]
                    merged = True
                else:
                    new_blocks.append(blk)
            blocks = new_blocks

        active_k[it] = len(blocks)

        # Write back modified a,b into the row without shifting any columns.
        row2 = row.copy()
        row2[0::2] = a
        row2[1::2] = b
        edges2[it, :] = row2

    return edges2, active_k


# ==============
# is inside bulk
# ==============

def _is_inside_bulk(edges_row, x, tol=0.0):
    """
    Returns True if x lies inside any finite bulk interval [a,b] in edges_row.
    edges_row is [a1,b1,a2,b2,...] with possible NaNs.
    """

    row = numpy.asarray(edges_row, dtype=float)
    a = row[0::2]
    b = row[1::2]
    for aj, bj in zip(a, b):
        if numpy.isfinite(aj) and numpy.isfinite(bj) and (bj > aj):
            if (x >= aj - tol) and (x <= bj + tol):
                return True
    return False


# =======================
# bulk index containing x
# =======================

def _bulk_index_containing_x(edges_row, x, tol=0.0):
    """
    Return bulk index j such that x is inside [a_j, b_j] for the given row.
    Returns None if not inside any bulk.
    """

    row = numpy.asarray(edges_row, dtype=float)
    a = row[0::2]
    b = row[1::2]
    for j, (aj, bj) in enumerate(zip(a, b)):
        if numpy.isfinite(aj) and numpy.isfinite(bj) and (bj > aj):
            if (x >= aj - tol) and (x <= bj + tol):
                return j
    return None


# ==============
# first index ge
# ==============

def _first_index_ge(t_grid, t0):
    """First index i with t_grid[i] >= t0. Returns None if none."""

    i = numpy.searchsorted(t_grid, t0, side="left")
    if i >= t_grid.size:
        return None
    else:
        return int(i)


# =======================
# evolve edges with birth
# =======================

def evolve_edges_with_births(t_grid, coeffs, support=None, cusps=None,
                             delta=1e-5, dt_max=0.1, max_iter=30, tol=1e-12,
                             return_preimage=False, split_tol=0.0,
                             seed_eps=1e-6, fill_gap="linear"):
    """
    Evolve edges like evolve_edges(), but also creates new edge columns after
    split cusps (bulk bifurcation).

    Newborn columns are inserted between parent bulk edges (a_j, b_j), so the
    left->right column ordering is preserved for all rows.

    This version:
      - forces newborn edges to meet at the cusp (writes x_star at the cusp
        row)
      - births the two interior split edges at a slightly later time where two
        distinct interior solutions exist (robust)
      - optionally fills the gap between cusp-row and birth-row ("linear" or
        None)
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 1:
        raise ValueError("t_grid must be non-empty.")
    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")

    nt = t_grid.size

    # 1) Baseline evolution (fixed-width)
    if return_preimage:
        base_edges, base_ok, base_zeta, base_y = evolve_edges(
            t_grid, coeffs, support=support, delta=delta, dt_max=dt_max,
            max_iter=max_iter, tol=tol, return_preimage=True)
    else:
        base_edges, base_ok = evolve_edges(
            t_grid, coeffs, support=support, delta=delta, dt_max=dt_max,
            max_iter=max_iter, tol=tol, return_preimage=False)

        base_zeta = None
        base_y = None

    m0 = base_edges.shape[1]

    if cusps is None or len(cusps) == 0:
        if return_preimage:
            return base_edges, base_ok, base_zeta, base_y
        return base_edges, base_ok

    # 2) Collect split cusps and determine parent bulk index j
    split_list = []
    for c in cusps:
        if not isinstance(c, dict):
            continue
        t_star = float(c.get("t", numpy.nan))
        x_star = float(c.get("x", numpy.nan))

        dbg = c.get("debug", {})
        if not isinstance(dbg, dict):
            dbg = {}
        zeta_star = dbg.get("zeta", None)
        y_star = dbg.get("y", None)

        if not (numpy.isfinite(t_star) and numpy.isfinite(x_star)):
            continue
        if zeta_star is None or y_star is None:
            continue

        it_ge = _first_index_ge(t_grid, t_star)
        if it_ge is None:
            continue
        it_prev = max(0, it_ge - 1)

        row_prev = numpy.real(base_edges[it_prev, :])
        j = _bulk_index_containing_x(row_prev, x_star, tol=split_tol)

        if j is None and m0 == 2:
            j = 0

        if j is None:
            continue

        split_list.append({
            "t": t_star,
            "x": x_star,
            "j": int(j),
            "zeta": complex(zeta_star),
            "y": complex(y_star),
        })

    if len(split_list) == 0:
        if return_preimage:
            return base_edges, base_ok, base_zeta, base_y
        return base_edges, base_ok

    split_list.sort(key=lambda d: (d["j"], d["t"]))

    # 3) Insert 2 newborn columns per split cusp
    edges_ext = base_edges.copy()
    ok_ext = base_ok.copy()

    if return_preimage:
        zeta_ext = base_zeta.copy()
        y_ext = base_y.copy()
    else:
        zeta_ext = None
        y_ext = None

    for csp in split_list:
        j = int(csp["j"])
        insert_pos = 2 * j + 1  # between a_j and b_j

        edges_ext = numpy.insert(
            edges_ext, insert_pos, numpy.nan + 1j * numpy.nan, axis=1)
        edges_ext = numpy.insert(
            edges_ext, insert_pos, numpy.nan + 1j * numpy.nan, axis=1)

        ok_ext = numpy.insert(ok_ext, insert_pos, False, axis=1)
        ok_ext = numpy.insert(ok_ext, insert_pos, False, axis=1)

        if return_preimage:
            zeta_ext = numpy.insert(
                zeta_ext, insert_pos, numpy.nan + 1j * numpy.nan, axis=1)
            zeta_ext = numpy.insert(
                zeta_ext, insert_pos, numpy.nan + 1j * numpy.nan, axis=1)

            y_ext = numpy.insert(
                y_ext, insert_pos, numpy.nan + 1j * numpy.nan, axis=1)
            y_ext = numpy.insert(
                y_ext, insert_pos, numpy.nan + 1j * numpy.nan, axis=1)

        csp["colL"] = int(insert_pos)
        csp["colR"] = int(insert_pos + 1)

    # 4) Helper: birth two distinct INTERIOR split edges at time t0
    def _try_birth_two_interior_edges(t0, zeta_star, y_star, x_star, a_parent,
                                      b_parent):
        tau = float(numpy.exp(t0))
        c = tau - 1.0

        if not (numpy.isfinite(a_parent) and numpy.isfinite(b_parent) and
                (b_parent > a_parent)):
            return None

        edge_gap = float(b_parent - a_parent)
        edge_tol = 1e-8 * (1.0 + edge_gap)

        eps_z0 = float(seed_eps) * (1.0 + abs(zeta_star))
        eps_y0 = float(seed_eps) * (1.0 + abs(y_star))

        trials = []
        scales = (1.0, 10.0, 100.0)
        dirs = (
            (1.0, 0.0), (-1.0, 0.0),
            (0.0, 1.0), (0.0, -1.0),
            (1.0, 1.0), (-1.0, 1.0),
            (1.0, -1.0), (-1.0, -1.0),
        )

        for sc in scales:
            dz = sc * eps_z0
            dy = sc * eps_y0
            for (dr, di) in dirs:
                z0 = zeta_star + dz * (dr + 1.0j * di)
                y0 = y_star * \
                    (1.0 + 1.0j * dy * numpy.sign(di if di != 0.0 else dr))

                z1, y1, ok1 = _edge_newton_step(
                    t0, z0, y0, coeffs, max_iter=max_iter, tol=tol
                )
                if not ok1:
                    continue
                if not (numpy.isfinite(z1.real) and numpy.isfinite(y1.real)):
                    continue

                x1 = float((z1 - c / y1).real)

                if not (a_parent + edge_tol < x1 < b_parent - edge_tol):
                    continue

                trials.append((x1, z1, y1))

        if len(trials) < 2:
            return None

        trials.sort(key=lambda r: r[0])

        uniq = []
        x_tol = 1e-7 * (1.0 + abs(float(x_star)))
        for (x1, z1, y1) in trials:
            if not uniq or abs(x1 - uniq[-1][0]) > x_tol:
                uniq.append((x1, z1, y1))

        if len(uniq) < 2:
            return None

        uniq.sort(key=lambda u: abs(u[0] - x_star))
        xL, zL, yL = uniq[0]

        picked = False
        for k in range(1, len(uniq)):
            xR, zR, yR = uniq[k]
            if xR > xL:
                picked = True
                break
        if not picked:
            return None

        return zL, yL, zR, yR

    # 5) Birth + evolve newborn edges for each split cusp
    for csp in split_list:
        t_star = float(csp["t"])
        x_star = float(csp["x"])
        zeta_star = complex(csp["zeta"])
        y_star = complex(csp["y"])
        colL = int(csp["colL"])
        colR = int(csp["colR"])

        it_ge = _first_index_ge(t_grid, t_star)
        if it_ge is None:
            continue

        # Force meeting at cusp row (grid-snapped)
        edges_ext[it_ge, colL] = float(x_star)
        edges_ext[it_ge, colR] = float(x_star)
        ok_ext[it_ge, colL] = True
        ok_ext[it_ge, colR] = True
        if return_preimage:
            zeta_ext[it_ge, colL] = zeta_star
            zeta_ext[it_ge, colR] = zeta_star
            y_ext[it_ge, colL] = y_star
            y_ext[it_ge, colR] = y_star

        # Start searching strictly after cusp
        it0 = min(it_ge + 1, nt - 1)

        max_birth_tries = 20
        birth = None
        it_birth = None

        for it in range(it0, min(nt, it0 + max_birth_tries)):
            t0 = float(t_grid[it])

            a_parent = float(numpy.real(edges_ext[it, colL - 1]))
            b_parent = float(numpy.real(edges_ext[it, colR + 1]))

            birth = _try_birth_two_interior_edges(
                t0, zeta_star, y_star, x_star, a_parent, b_parent)

            if birth is not None:
                it_birth = it
                break

        if birth is None or it_birth is None:
            continue

        t0 = float(t_grid[it_birth])
        zL, yL, zR, yR = birth

        tau0 = float(numpy.exp(t0))
        c0 = tau0 - 1.0

        xL0 = zL - c0 / yL
        xR0 = zR - c0 / yR

        if float(xL0.real) > float(xR0.real):
            zL, yL, zR, yR = zR, yR, zL, yL
            xL0, xR0 = xR0, xL0

        edges_ext[it_birth, colL] = xL0
        edges_ext[it_birth, colR] = xR0
        ok_ext[it_birth, colL] = True
        ok_ext[it_birth, colR] = True

        if return_preimage:
            zeta_ext[it_birth, colL] = zL
            zeta_ext[it_birth, colR] = zR
            y_ext[it_birth, colL] = yL
            y_ext[it_birth, colR] = yR

        # Optional fill between cusp-row and birth-row
        if fill_gap == "linear" and it_birth > it_ge + 1:
            for it in range(it_ge + 1, it_birth):
                w = (t_grid[it] - t_grid[it_ge]) / \
                    (t_grid[it_birth] - t_grid[it_ge])
                edges_ext[it, colL] = (1.0 - w) * \
                    float(x_star) + w * float(xL0.real)
                edges_ext[it, colR] = (1.0 - w) * \
                    float(x_star) + w * float(xR0.real)
                ok_ext[it, colL] = True
                ok_ext[it, colR] = True
                if return_preimage:
                    # keep preimage as NaN in fill region
                    pass

        # Evolve forward from it_birth
        zeta_pair = numpy.array([zL, zR], dtype=numpy.complex128)
        y_pair = numpy.array([yL, yR], dtype=numpy.complex128)
        alive = numpy.array([True, True], dtype=bool)

        for it in range(it_birth + 1, nt):
            t_prev = float(t_grid[it - 1])
            t_cur = float(t_grid[it])
            dt = t_cur - t_prev

            n_sub = int(numpy.ceil(dt / float(dt_max)))
            if n_sub < 1:
                n_sub = 1

            for ks in range(1, n_sub + 1):
                tt = t_prev + dt * (ks / float(n_sub))
                for jj in range(2):
                    if not alive[jj]:
                        continue

                    zeta_pair[jj], y_pair[jj], okj = _edge_newton_step(
                        tt, zeta_pair[jj], y_pair[jj], coeffs,
                        max_iter=max_iter, tol=tol)

                    if not (numpy.isfinite(zeta_pair[jj].real) and
                            numpy.isfinite(y_pair[jj].real)):
                        alive[jj] = False
                        okj = False

                    if jj == 0:
                        ok_ext[it, colL] = okj
                    else:
                        ok_ext[it, colR] = okj

            if not (alive[0] or alive[1]):
                break

            tau = float(numpy.exp(t_cur))
            c = tau - 1.0

            if alive[0]:
                edges_ext[it, colL] = zeta_pair[0] - c / y_pair[0]
                if return_preimage:
                    zeta_ext[it, colL] = zeta_pair[0]
                    y_ext[it, colL] = y_pair[0]

            if alive[1]:
                edges_ext[it, colR] = zeta_pair[1] - c / y_pair[1]
                if return_preimage:
                    zeta_ext[it, colR] = zeta_pair[1]
                    y_ext[it, colR] = y_pair[1]

    if return_preimage:
        return edges_ext, ok_ext, zeta_ext, y_ext
    return edges_ext, ok_ext
