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
import scipy
from ._elliptic_functions import ellipj
from ._continuation_genus1 import _poly_eval

__all__ = ['make_torus_grid', 'u_from_angles', 'eval_fitted_m_on_torus']


# ===============
# make torus grid
# ===============

def make_torus_grid(n_theta=101, n_phi=101, R=1.0, r=0.35):
    """
    Returns an embedded torus mesh (X,Y,Z) with angles (TH,PH).
    TH: around major circle, PH: around tube.
    """

    if n_phi % 2 == 0:
        raise ValueError('n_phi should be odd number to avoid rendering ' +
                         'issue at phi=0.')

    # fundamental angles (no endpoint duplicates)
    theta = numpy.linspace(0.0, 2.0*numpy.pi, int(n_theta), endpoint=False)
    phi = numpy.linspace(0.0, 2.0*numpy.pi, int(n_phi),   endpoint=False)

    TH, PH = numpy.meshgrid(theta, phi)  # shapes (n_phi, n_theta)

    # --- wrap/close the grid by appending first row/col ---
    TH = numpy.vstack([TH, TH[0:1, :]])     # add phi seam row
    PH = numpy.vstack([PH, PH[0:1, :]])

    TH = numpy.hstack([TH, TH[:, 0:1]])     # add theta seam col
    PH = numpy.hstack([PH, PH[:, 0:1]])

    # torus embedding
    X = (R + r*numpy.cos(PH)) * numpy.cos(TH)
    Y = (R + r*numpy.cos(PH)) * numpy.sin(TH)
    Z = r * numpy.sin(PH)

    return X, Y, Z, TH, PH


# =============
# u from angles
# =============

def u_from_angles(TH, PH, lam, center=(0.0, 0.0)):
    """
    Map angles (TH,PH) in [0,2pi)^2 to the elliptic u-plane fundamental cell.

    u = u0 + (omega1/2pi)*TH + (omega2/2pi)*PH,
    omega1 = 2K(m), omega2 = 2 i K(1-m).
    """

    m = float(lam)
    K = scipy.special.ellipk(m)
    Kp = scipy.special.ellipk(1.0 - m)
    omega1 = 2.0 * K
    omega2 = 2.0j * Kp

    u0 = complex(center[0], center[1])  # shift inside the fundamental domain
    u = u0 + (omega1/(2.0*numpy.pi))*TH + (omega2/(2.0*numpy.pi))*PH

    return u


# ==========================
# evaluate fitted m on torus
# ==========================

def eval_fitted_m_on_torus(u, a1, b1, a2, b2, p0, p1, q, lam):
    """
    Evaluate the *uniformized* branch m(u) = (p0(X)+Y p1(X))/q(X)
    with X = lam * sn(u)^2 and Y chosen to match the elliptic derivative sign.

    Requires jacobi_ellipj(u, m) that supports complex arrays and returns
    (sn,cn,dn).
    """

    sn, cn, dn, _ = ellipj(u, lam)

    X = lam * (sn * sn)

    # canonical algebraic Y from curve: Y^2 = X(1-X)(X-lam)
    D = X * (1.0 - X) * (X - lam)

    Y = numpy.sqrt(D)

    # enforce "positive imag" convention first
    Y = numpy.where(numpy.imag(Y) < 0.0, -Y, Y)

    # now align sign with elliptic reference:
    # for Legendre normalization: dX/du = 2 i Y_ref, where Y_ref is equal to
    # i*lam*sn*cn*dn (up to consistent conventions)
    Y_ref = 1j * lam * (sn * cn * dn)
    flip = numpy.real(Y * numpy.conjugate(Y_ref)) < 0.0
    Y = numpy.where(flip, -Y, Y)

    num0 = _poly_eval(X, p0)
    num1 = _poly_eval(X, p1)
    den = _poly_eval(X, q)

    m = (num0 + Y * num1) / den
    return m
