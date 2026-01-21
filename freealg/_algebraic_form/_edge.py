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
from ._decompress import eval_P_partials

__all__ = ['evolve_edges', 'merge_edges']


# ================
# edge newton step
# ================

def _edge_newton_step(t, zeta, y, a_coeffs, max_iter=30, tol=1e-12):
    """
    """

    tau = float(numpy.exp(t))
    c = tau - 1.0

    for _ in range(max_iter):
        P, Pz, Py = eval_P_partials(zeta, y, a_coeffs)

        # F1 = P(zeta,y)
        F1 = complex(P)

        # F2 = y^2 Py - c Pz
        F2 = complex((y * y) * Py - c * Pz)

        if max(abs(F1), abs(F2)) <= tol:
            return zeta, y, True

        # Numerical Jacobian (2x2) in (zeta,y)
        eps_z = 1e-8 * (1.0 + abs(zeta))
        eps_y = 1e-8 * (1.0 + abs(y))

        Pp, Pzp, Pyp = eval_P_partials(zeta + eps_z, y, a_coeffs)
        F1_zp = (complex(Pp) - F1) / eps_z
        F2_zp = (complex((y * y) * Pyp - c * Pzp) - F2) / eps_z

        Pp, Pzp, Pyp = eval_P_partials(zeta, y + eps_y, a_coeffs)
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

def _init_edge_point_from_support(x_edge, a_coeffs, eta=1e-3):
    """
    Initialize (zeta,y) at t=0 for an edge near x_edge.

    Uses z = x_edge + i*eta, picks physical root y, then refines zeta on real
    axis.
    """

    z = complex(x_edge + 1j * eta)
    roots = eval_roots(numpy.array([z]), a_coeffs)[0]
    y = _pick_physical_root(z, roots)

    # Move zeta to real axis as initial guess
    zeta = complex(x_edge)

    # Refine zeta,y to satisfy P=0 and Py=0 at t=0 (branch point)
    # This uses the same Newton system with c=0, i.e. F2 = y^2 Py.
    zeta, y, ok = _edge_newton_step(0.0, zeta, y, a_coeffs, max_iter=50,
                                    tol=1e-10)

    return zeta, y, ok


# ============
# evolve edges
# ============

def evolve_edges(t_grid, a_coeffs, support=None, eta=1e-3, dt_max=0.1,
                 max_iter=30, tol=1e-12):
    """
    Evolve spectral edges under free decompression using the fitted
    polynomial P.

    At time t, edges are computed as critical values of the FD map restricted
    to the spectral curve P(zeta,y)=0. We solve for (zeta(t), y(t)):

        P(zeta,y) = 0,
        y^2 * Py(zeta,y) - (exp(t)-1) * Pzeta(zeta,y) = 0,

    then map to the physical coordinate:
        z_edge(t) = zeta - (exp(t)-1)/y.

    Parameters
    ----------
    t_grid : array_like of float
        Strictly increasing time grid.
    a_coeffs : ndarray
        Coefficients defining P(zeta,y).
    support : list of (float, float), optional
        List of intervals [(a1,b1),...,(ak,bk)] at t=0. If provided, these
        endpoints are used as labels/initial guesses and all are tracked.
        If omitted, this function currently raises ValueError (auto-detection
        is intentionally not implemented here to avoid fragile heuristics).
    eta : float, optional
        Small imaginary part used only to pick an initial physical root near
        each endpoint at t=0.
    dt_max : float, optional
        Maximum internal time step used for substepping in t.
    max_iter : int, optional
        Newton iterations per time step.
    tol : float, optional
        Tolerance for the 2x2 Newton solve.

    Returns
    -------
    edges : ndarray, shape (len(t_grid), 2*k)
        Tracked edges in the order [a1,b1,a2,b2,...] for each time.
    ok : ndarray of bool, same shape as edges
        Flags indicating whether each edge solve succeeded.

    Notes
    -----
    The solve is done by continuation in time. If two edges merge, the Newton
    system may become ill-conditioned near the merge time.

    Examples
    --------
    .. code-block:: python

        t_grid = numpy.linspace(0.0, 3.0, 61)
        support = [(a1,b1)]
        edges, ok = fd_evolve_edges(t_grid, a_coeffs, support=support,
                                    eta=1e-3)

        a_t = edges[:, 0]
        b_t = edges[:, 1]
    """

    t_grid = numpy.asarray(t_grid, dtype=float).ravel()
    if t_grid.size < 1:
        raise ValueError("t_grid must be non-empty.")
    if numpy.any(numpy.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")

    if support is None:
        raise ValueError(
            "support must be provided (auto-detection not implemented).")

    # Flatten endpoints in the order [a1,b1,a2,b2,...]
    endpoints0 = []
    for a, b in support:
        endpoints0.append(float(a))
        endpoints0.append(float(b))

    m = len(endpoints0)
    edges = numpy.empty((t_grid.size, m), dtype=float)
    ok = numpy.zeros((t_grid.size, m), dtype=bool)

    # Initialize spectral points (zeta,y) at t=0 for each endpoint
    zeta = numpy.empty(m, dtype=complex)
    y = numpy.empty(m, dtype=complex)

    for j in range(m):
        z0, y0, ok0 = _init_edge_point_from_support(endpoints0[j], a_coeffs,
                                                    eta=eta)
        zeta[j] = z0
        y[j] = y0
        edges[0, j] = float(numpy.real(z0))  # at t=0, z_edge = zeta
        ok[0, j] = ok0

    # Time continuation
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
                    t, zeta[j], y[j], a_coeffs,
                    max_iter=max_iter, tol=tol
                )
                ok[it, j] = okj

        tau = float(numpy.exp(t1))
        c = tau - 1.0
        z_edge = zeta - c / y

        edges[it, :] = numpy.real(z_edge)
        # ok[it,:] already set in last substep loop

    return edges, ok


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
