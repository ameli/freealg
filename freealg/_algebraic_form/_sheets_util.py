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
           'infer_m1_partners_on_cuts']


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
    # sheets: [m1, m2, m3] arrays on the same z-grid
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

        # ignore k=0 (trivial match away from cuts); take best among {1,2}
        k1 = min([1, 2], key=lambda k: d_up_to_dn[k] + d_dn_to_up[k])
        partners.append(k1)

    # e.g. [1,2] means I1 swaps with m2, I2 swaps with m3
    return partners
