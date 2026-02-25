# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy

__all__ = ['_pick_physical_root_scalar', 'track_roots_on_grid',
           'infer_m1_partners_on_cuts', 'build_sheets_from_roots']


# =========================
# pick physical root scalar
# =========================

def _pick_physical_root_scalar(z, roots):
    """
    Pick the Herglotz root: Im(root) has the same sign as Im(z).
    """

    s = 1.0 if (z.imag >= 0.0) else -1.0
    k = int(numpy.argmax(s * roots.imag))
    return roots[k]


# ============
# permutations
# ============

def _permutations(items):

    items = list(items)
    if len(items) <= 1:
        yield tuple(items)
        return
    for i in range(len(items)):
        rest = items[:i] + items[i + 1:]
        for p in _permutations(rest):
            yield (items[i],) + p


# ===================
# track roots on grid
# ===================

def track_roots_on_grid(m_all, z=None, i0=0, j0=0):

    m_all = numpy.asarray(m_all, dtype=numpy.complex128)
    n_y, n_x, s = m_all.shape

    sheets = numpy.full_like(m_all, numpy.nan + 1j * numpy.nan)

    perms = numpy.array(list(_permutations(range(s))), dtype=int)

    def sort_seed(v):
        v = numpy.asarray(v, dtype=numpy.complex128)
        order = numpy.argsort(-numpy.imag(v))
        return v[order]

    v0 = m_all[i0, j0, :]
    if numpy.all(numpy.isfinite(v0)):
        sheets[i0, j0, :] = sort_seed(v0)

    for i in range(i0, n_y):
        for j in range((j0 if i == i0 else 0), n_x):
            if i == i0 and j == j0:
                continue

            v = m_all[i, j, :]
            if not numpy.all(numpy.isfinite(v)):
                continue

            if j > 0 and numpy.all(numpy.isfinite(sheets[i, j - 1, :])):
                ref = sheets[i, j - 1, :]
            elif i > 0 and numpy.all(numpy.isfinite(sheets[i - 1, j, :])):
                ref = sheets[i - 1, j, :]
            else:
                sheets[i, j, :] = sort_seed(v)
                continue

            v_perm = v[perms]
            cost = numpy.abs(v_perm - ref[None, :]).sum(axis=1)
            p = perms[int(numpy.argmin(cost))]
            sheets[i, j, :] = v[p]

    if z is not None:
        z = numpy.asarray(z)
        if z.shape != (n_y, n_x):
            raise ValueError("z must have shape (n_y, n_x) matching m_all.")
        mask_up = numpy.imag(z) > 0.0
        scores = numpy.full(s, -numpy.inf, dtype=numpy.float64)
        for r in range(s):
            v = sheets[:, :, r]
            vv = v[mask_up]
            finite = numpy.isfinite(vv)
            if numpy.any(finite):
                scores[r] = float(numpy.mean(numpy.imag(vv[finite])))
        r_phys = int(numpy.argmax(scores))
        perm = [r_phys] + [r for r in range(s) if r != r_phys]
        sheets = sheets[:, :, perm]

    return sheets


# =========================
# infer m1 partners on cuts
# =========================

def infer_m1_partners_on_cuts(z, sheets, support):

    # sheets: [m1, m2, ..., ms] arrays on the same z-grid
    X = numpy.real(z[0, :])
    ycol = numpy.imag(z[:, 0])

    # pick nearest rows just above and below 0
    i_up = numpy.where(ycol > 0)[0][0]
    i_dn = numpy.where(ycol < 0)[0][-1]

    partners = []
    for (a, b) in support:
        x0 = 0.5 * (a + b)
        j = int(numpy.argmin(numpy.abs(X - x0)))

        m1_up = sheets[0][i_up, j]
        m1_dn = sheets[0][i_dn, j]

        # who matches across the cut?
        d_up_to_dn = [abs(m1_up - sheets[k][i_dn, j])
                      for k in range(len(sheets))]
        d_dn_to_up = [abs(m1_dn - sheets[k][i_up, j])
                      for k in range(len(sheets))]

        # Ignore k=0 (physical sheet) and pick best non-physical partner.
        k1 = min(range(1, len(sheets)),
                 key=lambda k: d_up_to_dn[k] + d_dn_to_up[k])
        partners.append(k1)

    # e.g. [1,2] means I1 swaps with m2, I2 swaps with m3
    return partners


# =======================
# track one sheet on grid
# =======================

def track_one_sheet_on_grid(z, roots, sheet_seed, cuts=None, i0=None, j0=None):
    """
    This is mostly used for visualization of the sheets.
    """

    z = numpy.asarray(z)
    n_y, n_x = z.shape
    s = roots.shape[1]
    if s < 1:
        raise ValueError("s must be >= 1.")

    R = roots.reshape((n_y, n_x, s))

    if i0 is None:
        ycol = numpy.imag(z[:, 0])
        pos = numpy.where(ycol > 0.0)[0]
        i0 = int(pos[0]) if pos.size > 0 else (n_y // 2)

    if j0 is None:
        j0 = n_x // 2

    seed_imag = float(numpy.imag(sheet_seed))
    cand0 = R[i0, j0, :]
    idx0 = int(numpy.argmin(numpy.abs(cand0 - sheet_seed)))

    sheet = numpy.full((n_y, n_x), numpy.nan + 1j * numpy.nan, dtype=complex)
    sheet[i0, j0] = cand0[idx0]

    visited = numpy.zeros((n_y, n_x), dtype=bool)
    q_i = numpy.empty(n_y * n_x, dtype=int)
    q_j = numpy.empty(n_y * n_x, dtype=int)

    head = 0
    tail = 0
    q_i[tail] = i0
    q_j[tail] = j0
    tail += 1
    visited[i0, j0] = True

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    y_unique = numpy.unique(numpy.imag(z[:, 0]))
    if y_unique.size >= 2:
        dy = float(numpy.min(numpy.diff(y_unique)))
        y_eps = 0.49 * dy
    else:
        y_eps = 0.0

    def crosses_cut(x_mid):
        if cuts is None:
            return False
        for a, b in cuts:
            if a <= x_mid <= b:
                return True
        return False

    while head < tail:
        i = int(q_i[head])
        j = int(q_j[head])
        head += 1

        m_prev = sheet[i, j]
        y1 = float(numpy.imag(z[i, j]))
        x1 = float(numpy.real(z[i, j]))

        for di, dj in neighbors:
            i2 = i + di
            j2 = j + dj
            if i2 < 0 or i2 >= n_y or j2 < 0 or j2 >= n_x:
                continue
            if visited[i2, j2]:
                continue

            y2 = float(numpy.imag(z[i2, j2]))
            x2 = float(numpy.real(z[i2, j2]))

            if cuts is not None:
                if (y1 > y_eps and y2 < -y_eps) or \
                        (y1 < -y_eps and y2 > y_eps):
                    x_mid = 0.5 * (x1 + x2)
                    if crosses_cut(x_mid):
                        continue

            cand = R[i2, j2, :]
            d = numpy.abs(cand - m_prev)
            idx = int(numpy.argmin(d))

            if seed_imag != 0.0:
                y_sign = 1.0 if y2 >= 0.0 else -1.0
                target = float(numpy.sign(seed_imag) * y_sign)
                if target != 0.0:
                    sgn = numpy.sign(numpy.imag(cand))
                    ok = (sgn == numpy.sign(target)) | (sgn == 0.0)
                    if numpy.any(ok):
                        ok_idx = numpy.where(ok)[0]
                        idx = int(ok_idx[numpy.argmin(d[ok])])

            sheet[i2, j2] = cand[idx]
            visited[i2, j2] = True
            q_i[tail] = i2
            q_j[tail] = j2
            tail += 1

    return sheet


# =======================
# build sheets from roots
# =======================

def build_sheets_from_roots(z, roots, m1, cuts=None, i0=None, j0=None):

    z = numpy.asarray(z)
    m1 = numpy.asarray(m1)

    n_y, n_x = z.shape
    s = roots.shape[1]
    if s < 1:
        raise ValueError("s must be >= 1.")

    if i0 is None:
        ycol = numpy.imag(z[:, 0])
        pos = numpy.where(ycol > 0.0)[0]
        i0 = int(pos[0]) if pos.size > 0 else (n_y // 2)

    if j0 is None:
        j0 = n_x // 2

    R = roots.reshape((n_y, n_x, s))

    tracked = track_roots_on_grid(R, z=z, i0=0, j0=0)

    k_phys = int(numpy.argmin(numpy.abs(tracked[i0, j0, :] - m1[i0, j0])))
    if k_phys != 0:
        perm = [k_phys] + [k for k in range(s) if k != k_phys]
        tracked = tracked[:, :, perm]
        idxs = perm
    else:
        idxs = list(range(s))

    sheets = [tracked[:, :, k] for k in range(s)]

    if cuts is not None:
        y_unique = numpy.unique(numpy.imag(z[:, 0]))
        if y_unique.size >= 2:
            dy = float(numpy.min(numpy.diff(y_unique)))
            eps_y = 0.49 * dy
        else:
            eps_y = 0.0

        i_cut = numpy.where(numpy.abs(numpy.imag(z[:, 0])) <= eps_y)[0]
        if i_cut.size > 0:
            i_cut = int(
                i_cut[numpy.argmin(numpy.abs(numpy.imag(z[i_cut, 0])))])

            X = numpy.real(z[i_cut, :])
            on_cut = numpy.zeros(n_x, dtype=bool)
            for j in range(n_x):
                xj = float(X[j])
                for a, b in cuts:
                    if a <= xj <= b:
                        on_cut[j] = True
                        break

            sheets[0][i_cut, on_cut] = m1[i_cut, on_cut]

    ycol = numpy.imag(z[:, 0])
    y_unique = numpy.unique(ycol)
    if y_unique.size >= 2:
        dy = float(numpy.min(numpy.diff(y_unique)))
        eps_y = 1.1 * dy
    else:
        eps_y = 0.0

    i_band = numpy.where(numpy.abs(ycol) <= eps_y)[0]
    i_up = numpy.where(ycol > eps_y)[0]
    i_dn = numpy.where(ycol < -eps_y)[0]
    if (i_band.size > 0) and (i_up.size > 0) and (i_dn.size > 0):
        i_up = int(i_up[0])
        i_dn = int(i_dn[-1])
        for r in range(1, len(sheets)):
            for i in i_band:
                if ycol[i] >= 0.0:
                    sheets[r][i, :] = sheets[r][i_up, :]
                else:
                    sheets[r][i, :] = sheets[r][i_dn, :]

    return sheets, idxs
