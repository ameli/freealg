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

__all__ = ['poly_trim', 'eval_P_partials']


# =========
# poly trim
# =========

def poly_trim(p, tol):
    """
    """

    p = numpy.asarray(p, dtype=float)
    if p.size == 0:
        return p
    k = p.size - 1
    while k > 0 and abs(p[k]) <= tol:
        k -= 1
    return p[: k + 1]


# ===============
# eval P partials
# ===============

def eval_P_partials(z, m, coeffs):
    """
    Evaluate P(z,m) and its partial derivatives dP/dz and dP/dm.

    This assumes P is represented by `coeffs` in the monomial basis

        P(z, m) = sum_{j=0..s} a_j(z) * m^j,
        a_j(z) = sum_{i=0..deg_z} coeffs[i, j] * z^i.

    The function returns P, dP/dz, dP/dm with broadcasting over z and m.

    Parameters
    ----------
    z : complex or array_like of complex
        First argument to P.
    m : complex or array_like of complex
        Second argument to P. Must be broadcast-compatible with `z`.
    coeffs : ndarray, shape (deg_z+1, s+1)
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

        P, Pz, Pm = eval_P_partials(1.0 + 1j, 0.2 + 0.3j, coeffs)
    """

    z = numpy.asarray(z, dtype=complex)
    m = numpy.asarray(m, dtype=complex)

    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)

    if (z.ndim == 0) and (m.ndim == 0):
        zz = complex(z)
        mm = complex(m)

        a = numpy.empty(s + 1, dtype=complex)
        ap = numpy.empty(s + 1, dtype=complex)

        for j in range(s + 1):
            c = coeffs[:, j]

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
        aj = zp @ coeffs[:, j]
        P += aj * mp[:, j]

        ajp = dzp @ coeffs[:, j]
        Pz += ajp * mp[:, j]

        if j >= 1:
            Pm += (j * aj) * mp[:, j - 1]

    return P.reshape(shp), Pz.reshape(shp), Pm.reshape(shp)
