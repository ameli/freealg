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
from ._continuation_algebraic import powers

__all__ = ['build_time_grid', 'eval_P_partials']


# ===============
# build time grid
# ===============

def build_time_grid(sizes, n0, min_n_times=0):
    """
    sizes: list/array of requested matrix sizes (e.g. [2000,3000,4000,8000])
    n0:    initial size (self.n)
    min_n_times: minimum number of time points to run Newton sweep on

    Returns
    -------
    t_all: sorted time grid to run solver on
    idx_req: indices of requested times inside t_all (same order as sizes)
    """

    sizes = numpy.asarray(sizes, dtype=float)
    alpha = sizes / float(n0)
    t_req = numpy.log(alpha)

    # Always include t=0 and T=max(t_req)
    T = float(numpy.max(t_req)) if t_req.size else 0.0
    base = numpy.unique(numpy.r_[0.0, t_req, T])
    t_all = numpy.sort(base)

    # Add points only if needed: split largest gaps
    N = int(min_n_times) if min_n_times is not None else 0
    while t_all.size < N and t_all.size >= 2:
        gaps = numpy.diff(t_all)
        k = int(numpy.argmax(gaps))
        mid = 0.5 * (t_all[k] + t_all[k+1])
        t_all = numpy.sort(numpy.unique(numpy.r_[t_all, mid]))

    # Map each requested time to an index in t_all (stable, no float drama)
    # (t_req values came from same construction, so they should match exactly;
    # still: use searchsorted + assert)
    idx_req = numpy.searchsorted(t_all, t_req)
    # optional sanity:
    # assert numpy.allclose(t_all[idx_req], t_req, rtol=0, atol=0)

    return t_all, idx_req


# ===============
# eval P partials
# ===============

def eval_P_partials(z, m, a_coeffs):
    """
    Evaluate P(z,m) and its partial derivatives dP/dz and dP/dm.

    This assumes P is represented by `a_coeffs` in the monomial basis

        P(z, m) = sum_{j=0..s} a_j(z) * m^j,
        a_j(z) = sum_{i=0..deg_z} a_coeffs[i, j] * z^i.

    The function returns P, dP/dz, dP/dm with broadcasting over z and m.

    Parameters
    ----------
    z : complex or array_like of complex
        First argument to P.
    m : complex or array_like of complex
        Second argument to P. Must be broadcast-compatible with `z`.
    a_coeffs : ndarray, shape (deg_z+1, s+1)
        Coefficient matrix for P in the monomial basis.

    Returns
    -------
    P : complex or ndarray of complex
        Value P(z,m).
    Pz : complex or ndarray of complex
        Partial derivative dP/dz evaluated at (z,m).
    Pm : complex or ndarray of complex
        Partial derivative dP/dm evaluated at (z,m).

    Notes
    -----
    For scalar (z,m), this uses Horner evaluation for a_j(z) and then Horner
    in m. For array inputs, it uses precomputed power tables via `_powers` for
    simplicity.

    Examples
    --------
    .. code-block:: python

        P, Pz, Pm = eval_P_partials(1.0 + 1j, 0.2 + 0.3j, a_coeffs)
    """

    z = numpy.asarray(z, dtype=complex)
    m = numpy.asarray(m, dtype=complex)

    deg_z = int(a_coeffs.shape[0] - 1)
    s = int(a_coeffs.shape[1] - 1)

    if (z.ndim == 0) and (m.ndim == 0):
        zz = complex(z)
        mm = complex(m)

        a = numpy.empty(s + 1, dtype=complex)
        ap = numpy.empty(s + 1, dtype=complex)

        for j in range(s + 1):
            c = a_coeffs[:, j]

            val = 0.0 + 0.0j
            for i in range(deg_z, -1, -1):
                val = val * zz + c[i]
            a[j] = val

            dval = 0.0 + 0.0j
            for i in range(deg_z, 0, -1):
                dval = dval * zz + (i * c[i])
            ap[j] = dval

        p = a[s]
        pm = 0.0 + 0.0j
        for j in range(s - 1, -1, -1):
            pm = pm * mm + p
            p = p * mm + a[j]

        pz = ap[s]
        for j in range(s - 1, -1, -1):
            pz = pz * mm + ap[j]

        return p, pz, pm

    shp = numpy.broadcast(z, m).shape
    zz = numpy.broadcast_to(z, shp).ravel()
    mm = numpy.broadcast_to(m, shp).ravel()

    zp = powers(zz, deg_z)
    mp = powers(mm, s)

    dzp = numpy.zeros_like(zp)
    for i in range(1, deg_z + 1):
        dzp[:, i] = i * zp[:, i - 1]

    P = numpy.zeros(zz.size, dtype=complex)
    Pz = numpy.zeros(zz.size, dtype=complex)
    Pm = numpy.zeros(zz.size, dtype=complex)

    for j in range(s + 1):
        aj = zp @ a_coeffs[:, j]
        P += aj * mp[:, j]

        ajp = dzp @ a_coeffs[:, j]
        Pz += ajp * mp[:, j]

        if j >= 1:
            Pm += (j * aj) * mp[:, j - 1]

    return P.reshape(shp), Pz.reshape(shp), Pm.reshape(shp)
