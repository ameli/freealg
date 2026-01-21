# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


"""
The Jacobi elliptic functions sn, cn, and dn are implemented in:

1. scipy.special: but they do not accept complex inputs
2. mpmath: but they do not have vectorization (very slow)

Because of these, we implemented sn, cn, and dn ourselves. There is no need to
implement the elliptic K function, since scipy.special already has it (and
vectorized) and we only need that for real inputs, not complex inputs. So, no
need to re-implement the scipy.special.ellipk function.
"""


# =======
# Imports
# =======

import numpy
from scipy import special

__all__ = ['ellipj']


# ===========
# jacobi nome
# ===========

def _jacobi_nome(m):
    """
    Compute the Jacobi nome q = exp(-pi K(1-m)/K(m)) and the complete elliptic
    integrals K(m), K(1-m).
    """

    m = float(m)
    K = special.ellipk(m)
    Kp = special.ellipk(1.0 - m)
    q = numpy.exp(-numpy.pi * Kp / K)

    return q, K, Kp


# =======
# theta 1
# =======

def _theta1(v, q, n_terms):
    """
    Jacobi theta_1(v,q) via truncated Fourier series (vectorized for complex
    v).
    """

    v = numpy.asarray(v, dtype=complex)
    n = numpy.arange(n_terms, dtype=float)
    a = 2.0 * n + 1.0
    logq = numpy.log(q)
    w = numpy.exp(((n + 0.5) ** 2) * logq) * ((-1.0) ** n)
    s = numpy.sin(a[:, None] * v[None, :])

    return 2.0 * (w[:, None] * s).sum(axis=0)


# =======
# theta 2
# =======

def _theta2(v, q, n_terms):
    """
    Jacobi theta_2(v,q) via truncated Fourier series (vectorized for complex
    v).
    """

    v = numpy.asarray(v, dtype=complex)
    n = numpy.arange(n_terms, dtype=float)
    a = 2.0 * n + 1.0
    logq = numpy.log(q)
    w = numpy.exp(((n + 0.5) ** 2) * logq)
    c = numpy.cos(a[:, None] * v[None, :])

    return 2.0 * (w[:, None] * c).sum(axis=0)


# =======
# theta 3
# =======

def _theta3(v, q, n_terms):
    """
    Jacobi theta_3(v,q) via truncated Fourier series (vectorized for complex
    v).
    """

    v = numpy.asarray(v, dtype=complex)
    n = numpy.arange(1, n_terms + 1, dtype=float)
    logq = numpy.log(q)
    w = numpy.exp((n ** 2) * logq)
    c = numpy.cos((2.0 * n)[:, None] * v[None, :])

    return 1.0 + 2.0 * (w[:, None] * c).sum(axis=0)


# =======
# theta 4
# =======

def _theta4(v, q, n_terms):
    """
    Jacobi theta_4(v,q) via truncated Fourier series (vectorized for complex
    v).
    """

    v = numpy.asarray(v, dtype=complex)
    n = numpy.arange(1, n_terms + 1, dtype=float)
    logq = numpy.log(q)
    w = numpy.exp((n ** 2) * logq) * ((-1.0) ** n)
    c = numpy.cos((2.0 * n)[:, None] * v[None, :])

    return 1.0 + 2.0 * (w[:, None] * c).sum(axis=0)


# ======
# ellipj
# ======

def ellipj(u, m, n_terms=None):
    """
    Vectorized Jacobi elliptic sn, cn, dn for complex u and real m in (0,1),
    computed from theta functions.
    """

    u = numpy.asarray(u, dtype=complex)
    q, K, Kp = _jacobi_nome(m)

    if n_terms is None:
        if q < 1e-6:
            n_terms = 16
        elif q < 1e-3:
            n_terms = 24
        elif q < 1e-2:
            n_terms = 32
        elif q < 5e-2:
            n_terms = 48
        else:
            n_terms = 80

    v = (numpy.pi / (2.0 * K)) * u
    v_flat = v.ravel()

    th1 = _theta1(v_flat, q, n_terms)
    th2 = _theta2(v_flat, q, n_terms)
    th3 = _theta3(v_flat, q, n_terms)
    th4 = _theta4(v_flat, q, n_terms)

    th2_0 = _theta2(numpy.array([0.0], dtype=complex), q, n_terms)[0]
    th3_0 = _theta3(numpy.array([0.0], dtype=complex), q, n_terms)[0]
    th4_0 = _theta4(numpy.array([0.0], dtype=complex), q, n_terms)[0]

    sn = (th3_0 * th1) / (th2_0 * th4)
    cn = (th4_0 * th2) / (th2_0 * th4)
    dn = (th4_0 * th3) / (th3_0 * th4)

    sn = sn.reshape(v.shape)
    cn = cn.reshape(v.shape)
    dn = dn.reshape(v.shape)

    return sn, cn, dn, q
