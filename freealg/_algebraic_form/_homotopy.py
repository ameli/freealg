# =======
# Imports
# =======

import numpy

__all__ = ['stieltjes_poly']


# =====================
# stieltjes select root
# =====================

def stieltjes_select_root(roots, z, w_prev=None):
    """
    Select the Stieltjes-branch root among candidates at a given z.

    Parameters
    ----------
    roots : array_like of complex
        Candidate roots for m at the given z.
    z : complex
        Evaluation point. The Stieltjes/Herglotz branch satisfies
        sign(Im(m)) = sign(Im(z)) away from the real axis.
    w_prev : complex or None, optional
        Previous continuation value used to enforce continuity. If None,
        the asymptotic target -1/z is used.

    Returns
    -------
    w : complex
        Selected root corresponding to the Stieltjes branch.
    """

    z = complex(z)
    roots = numpy.asarray(roots, dtype=numpy.complex128).ravel()

    if roots.size == 0:
        raise ValueError("roots must contain at least one candidate root.")

    desired_sign = numpy.sign(z.imag)

    if w_prev is None:
        target = -1.0 / z
    else:
        target = complex(w_prev)

    # Apply a soft Herglotz sign filter: prefer roots with Im(w) having the
    # same sign as Im(z), allowing tiny numerical violations near the axis.
    imag_roots = numpy.imag(roots)

    good = roots[numpy.sign(imag_roots) == desired_sign]
    if good.size == 0:
        good = roots[(imag_roots * desired_sign) > -1e-12]

    candidates = good if good.size > 0 else roots
    idx = int(numpy.argmin(numpy.abs(candidates - target)))
    return candidates[idx]


# ==============
# stieltjes poly
# ==============

def stieltjes_poly(z, a, eps=None, height=1e+4, steps=100):
    """
    Evaluate the Stieltjes-branch solution m(z) of an algebraic equation.

    The coefficients `a` define a polynomial relation
        P(z, m) = 0,
    where P is a polynomial in z and m with monomial-basis coefficients
    arranged so that for fixed z, the coefficients of the polynomial in m
    can be assembled from powers of z.

    Parameters
    ----------
    z : complex
        Evaluation point. Must be a single value.
    a : ndarray, shape (L, K)
        Coefficient matrix defining P(z, m) in the monomial basis.
    eps : float or None, optional
        If Im(z) == 0, use z + i*eps as the boundary evaluation point.
        If None and Im(z) == 0, eps is set to 1e-8 * max(1, |z|).
    height : float, optional
        Imaginary height used for the starting point z0 in the same
        half-plane as the evaluation point.
    steps : int, optional
        Number of continuation steps along the homotopy path.

    Returns
    -------
    w : complex
        Value of the Stieltjes-branch solution m(z) (or m(z+i*eps) if z is
        real).
    """

    z = complex(z)
    a = numpy.asarray(a)

    if a.ndim != 2:
        raise ValueError('a must be a 2D array.')

    if steps < 1:
        raise ValueError("steps must be a positive integer.")

    a_l, _ = a.shape

    def poly_coeffs_m(z_val):
        z_powers = z_val ** numpy.arange(a_l)
        return (z_powers @ a)[::-1]

    def poly_roots(z_val):
        coeffs = numpy.asarray(poly_coeffs_m(z_val), dtype=numpy.complex128)
        return numpy.roots(coeffs)

    # If user asked for a real-axis value, interpret as boundary value from C+.
    if z.imag == 0.0:
        if eps is None:
            eps = 1e-8 * max(1.0, abs(z))
        z_eval = z + 1j * float(eps)
    else:
        z_eval = z

    half_sign = numpy.sign(z_eval.imag)
    if half_sign == 0.0:
        half_sign = 1.0

    z0 = 1j * float(half_sign) * float(height)

    # Initialize at z0 via asymptotic / Im-sign selection.
    w_prev = stieltjes_select_root(poly_roots(z0), z0, w_prev=None)

    # Straight-line homotopy from z0 to z_eval.
    for tau in numpy.linspace(0.0, 1.0, int(steps) + 1)[1:]:
        z_tau = z0 + tau * (z_eval - z0)
        w_prev = stieltjes_select_root(poly_roots(z_tau), z_tau, w_prev=w_prev)

    return w_prev
