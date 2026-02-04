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
from ._poly_util import poly_trim

__all__ = ['detect_atoms']


# =============
# detect atoms
# =============

def detect_atoms(coeffs, m_eval, eta=1e-6, tol=1e-12, real_tol=None,
                 w_tol=1e-10, merge_tol=1e-8):
    """
    Detect atomic components from an algebraic Stieltjes transform P(z,m)=0.

    This routine uses the necessary condition for a finite pole: a_s(z0)=0,
    where a_s(z) is the leading coefficient of P in powers of m. Candidate
    atom locations are the (nearly) real roots of a_s(z). The atom weight is
    estimated numerically from the Stieltjes transform as

        w ~= eta * Im(m(z0 + i*eta)),

    which follows from m(z) ~ -w/(z - z0) near an atom at z0.

    Parameters
    ----------
    coeffs : array_like, shape (deg_z+1, s+1)
        Polynomial coefficients where coeffs[i, j] is the coefficient of
        z^i * m^j.
    m_eval : callable
        Function handle m_eval(z) returning the physical-branch Stieltjes
        transform evaluated at complex z. Must accept scalar complex input and
        return scalar complex output.
    eta : float, default=1e-6
        Small imaginary part used to probe the pole strength.
    tol : float, default=1e-12
        Tolerance for trimming polynomial coefficients.
    real_tol : float or None, default=None
        Tolerance for treating a complex root as real. If None, uses 1e3*tol.
    w_tol : float, default=1e-10
        Minimum atom weight to report.
    merge_tol : float, default=1e-8
        Merge roots whose real parts differ by at most this tolerance.

    Returns
    -------
    atoms : list of (float, float)
        List of (atom_loc, atom_w). Locations are real numbers and weights are
        nonnegative.

    Notes
    -----
    - This is intended for plotting/diagnostics. For fitted polynomials, a_s(z)
      roots can include spurious candidates; the weight filter w_tol is a
      practical guard.
    - If the physical branch selection in m_eval is wrong near z0, the weight
      estimate may be unreliable.
    """

    coeffs = numpy.asarray(coeffs, dtype=float)
    s = coeffs.shape[1] - 1
    if s < 1:
        return []

    if real_tol is None:
        real_tol = 1e3 * tol

    # Leading coefficient in m: a_s(z)
    a_s = poly_trim(coeffs[:, s], tol)
    if a_s.size <= 1:
        return []

    # Candidate locations: roots of a_s(z) (descending -> ascending flip)
    z0_all = numpy.roots(a_s[::-1])

    # Keep nearly-real roots
    mask_real = numpy.abs(z0_all.imag) <= real_tol
    x0 = numpy.sort(z0_all.real[mask_real].astype(float))

    if x0.size == 0:
        return []

    # Merge very close roots (numerical multiplicity / jitter)
    merged = [float(x0[0])]
    for v in x0[1:]:
        if abs(float(v) - merged[-1]) <= merge_tol:
            merged[-1] = 0.5 * (merged[-1] + float(v))
        else:
            merged.append(float(v))

    atoms = []
    for x in merged:
        z = complex(x, float(eta))

        # Weight estimate: w ~= eta * Im(m(z))
        try:
            m = complex(m_eval(z))
        except Exception:
            continue

        w = float(eta) * float(m.imag)
        if w < 0.0 and abs(w) <= 10.0 * w_tol:
            w = 0.0

        if w > w_tol:
            atoms.append((float(x), float(w)))

    return atoms
