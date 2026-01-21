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
from ._continuation_genus0 import joukowski_z

__all__ = ['make_sphere_grid', 'eval_on_sphere']


# ===========================
# stereographic w from sphere
# ===========================

def stereographic_w_from_sphere(X, Y, Z):

    den = 1.0 - Z
    w = (X + 1j * Y) / den
    return w


# ================
# make sphere grid
# ================

def make_sphere_grid(n_theta=100, n_phi=50):

    theta = numpy.linspace(0.0, 2.0 * numpy.pi, n_theta, endpoint=False)
    u = numpy.linspace(0.0, 1.0, n_phi)
    phi = numpy.pi * (u * u)
    TH, PH = numpy.meshgrid(theta, phi)
    X = numpy.sin(PH) * numpy.cos(TH)
    Y = numpy.sin(PH) * numpy.sin(TH)
    Z = numpy.cos(PH)

    return X, Y, Z


# ==============
# eval on sphere
# ==============

def eval_on_sphere(X, Y, Z, a, b, m1_fn, m2_fn, z_eps=1e-9):

    Zc = numpy.minimum(Z, 1.0 - 1e-12)
    w = stereographic_w_from_sphere(X, Y, Zc)
    z = joukowski_z(w, a, b)
    z = z + 1j * z_eps
    m1 = m1_fn(z)
    m2 = m2_fn(z)
    out = numpy.array(m2, copy=True)
    out[numpy.abs(w) <= 1.0] = m1[numpy.abs(w) <= 1.0]

    return out
