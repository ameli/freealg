# =======
# Imports
# =======

import numpy
from ._moments import AlgebraicStieltjesMoments

__all__ = ['stieltjes_poly']


# =====================
# select root
# =====================

def select_root(roots, z, target):
    """
    Select the root among Herglotz candidates at a given z closest to a
    given target

    Parameters
    ----------
    roots : array_like of complex
        Candidate roots for m at the given z.
    z : complex
        Evaluation point. The Stieltjes/Herglotz branch satisfies
        sign(Im(m)) = sign(Im(z)) away from the real axis.
    target : complex
        Previous continuation value used to enforce continuity, or
        target value.

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

class StieltjesPoly(object):
    """
    Stieltjes-branch evaluator for an algebraic equation P(z, m) = 0.

    This class represents the Stieltjes-branch solution m(z) of an algebraic
    equation defined by a polynomial relation

        P(z, m) = 0,

    where P is a polynomial in z and m with monomial-basis coefficients.
    The coefficient matrix ``a`` is fixed at construction time, and all
    quantities depending only on ``a`` are precomputed. Evaluation at a
    complex point ``z`` is performed via :meth:`evaluate`. The instance is
    also callable; :meth:`__call__` supports scalar or vector inputs and
    applies :meth:`evaluate` elementwise.

    The Stieltjes branch is selected by initializing in the appropriate
    half-plane using an asymptotic Stieltjes estimate and then performing
    homotopy continuation along a straight-line path in the complex plane.

    Parameters
    ----------
    a : ndarray, shape (L, K)
        Coefficient matrix defining P(z, m) in the monomial basis. For fixed
        z, the coefficients of the polynomial in m are assembled from powers
        of z.
    eps : float or None, optional
        If Im(z) == 0, use z + i*eps as the boundary evaluation point.
        If None and Im(z) == 0, eps is set to 1e-8 * max(1, |z|).
    height : float, default = 2.0
        Imaginary height used for the starting point z0 in the same
        half-plane as the evaluation point.
    steps : int, default = 100
        Number of continuation steps along the homotopy path.
    order : int, default = 15
        Number of moments in Stieltjes estimate

    Methods
    -------
    evaluate(z)
        Evaluate the Stieltjes-branch solution m(z) at a single complex point.

    __call__(z)
        If ``z`` is scalar, returns ``evaluate(z, ...)``.
        If ``z`` is array-like, returns an array of the same shape, where each
        entry is computed by calling ``evaluate`` on the corresponding element.

    Notes
    -----
    If an input ``z`` value is real (Im(z) == 0), the evaluation is interpreted
    as a boundary value by replacing that element with z + i*eps. If ``eps`` is
    None, eps is chosen per element as 1e-8 * max(1, |z|).
    """

    def __init__(self, a, eps=None, height=2.0, steps=100, order=15):
        a = numpy.asarray(a)
        if a.ndim != 2:
            raise ValueError("a must be a 2D array.")

        self.a = a
        self.a_l, _ = a.shape
        self.eps = eps
        self.height = height
        self.steps = steps
        self.order = order

        # Objects depending only on a
        self.mom = AlgebraicStieltjesMoments(a)
        self._zpows_exp = numpy.arange(self.a_l)
        self.rad = 1.0 + self.height * self.mom.radius(self.order)

    def _poly_coeffs_m(self, z_val):
        z_powers = z_val ** self._zpows_exp
        return (z_powers @ self.a)[::-1]

    def _poly_roots(self, z_val):
        coeffs = numpy.asarray(self._poly_coeffs_m(z_val),
                               dtype=numpy.complex128)
        return numpy.roots(coeffs)

    def evaluate(self, z, eps=None, height=2.0, steps=100, order=15):
        """
        Evaluate the Stieltjes-branch solution m(z) at a single point.

        Parameters are as in the original function, except ``a`` is fixed at
        construction time.
        """
        z = complex(z)

        if steps < 1:
            raise ValueError("steps must be a positive integer.")

        # Boundary-value interpretation on the real axis
        if z.imag == 0.0:
            if self.eps is None:
                eps_loc = 1e-8 * max(1.0, abs(z))
            else:
                eps_loc = float(self.eps)
            z_eval = z + 1j * eps_loc
        else:
            z_eval = z

        half_sign = numpy.sign(z_eval.imag)
        if half_sign == 0.0:
            half_sign = 1.0

        # If z is outside radius of convergence, no homotopy
        # necessary
        if numpy.abs(z) > self.rad:
            target = self.mom.stieltjes(z, self.order)
            return select_root(self._poly_roots(z), z, target)

        z0 = 1j * float(half_sign) * self.rad
        target = self.mom.stieltjes(z0, self.order)

        # Initialize at z0
        w_prev = select_root(self._poly_roots(z0), z0, target)

        # Straight-line homotopy continuation
        for tau in numpy.linspace(0.0, 1.0, int(self.steps) + 1)[1:]:
            z_tau = z0 + tau * (z_eval - z0)
            w_prev = select_root(self._poly_roots(z_tau), z_tau, w_prev)

        return w_prev

    def __call__(self, z):
        # Scalar fast-path
        if numpy.isscalar(z):
            return self.evaluate(z)

        # Array-like: evaluate elementwise, preserving shape
        z_arr = numpy.asarray(z)
        out = numpy.empty(z_arr.shape, dtype=numpy.complex128)

        # Iterate over indices so we can pass Python scalars into evaluate()
        for idx in numpy.ndindex(z_arr.shape):
            out[idx] = self.evaluate(z_arr[idx])

        return out


# def stieltjes_poly(z, a, eps=None, height=2., steps=100, order=15):
#     """
#     Evaluate the Stieltjes-branch solution m(z) of an algebraic equation.

#     The coefficients `a` define a polynomial relation
#         P(z, m) = 0,
#     where P is a polynomial in z and m with monomial-basis coefficients
#     arranged so that for fixed z, the coefficients of the polynomial in m
#     can be assembled from powers of z.

#     Parameters
#     ----------
#     z : complex
#         Evaluation point. Must be a single value.
#     a : ndarray, shape (L, K)
#         Coefficient matrix defining P(z, m) in the monomial basis.
#     eps : float or None, optional
#         If Im(z) == 0, use z + i*eps as the boundary evaluation point.
#         If None and Im(z) == 0, eps is set to 1e-8 * max(1, |z|).
#     height : float, default = 2.0
#         Imaginary height used for the starting point z0 in the same
#         half-plane as the evaluation point.
#     steps : int, default = 100
#         Number of continuation steps along the homotopy path.
#     order : int, default = 15
#         Number of moments in Stieltjes estimate

#     Returns
#     -------
#     w : complex
#         Value of the Stieltjes-branch solution m(z) (or m(z+i*eps) if z is
#         real).
#     """

#     z = complex(z)
#     a = numpy.asarray(a)

#     if a.ndim != 2:
#         raise ValueError('a must be a 2D array.')

#     if steps < 1:
#         raise ValueError("steps must be a positive integer.")

#     a_l, _ = a.shape
#     mom = AlgebraicStieltjesMoments(a)

#     def poly_coeffs_m(z_val):
#         z_powers = z_val ** numpy.arange(a_l)
#         return (z_powers @ a)[::-1]

#     def poly_roots(z_val):
#         coeffs = numpy.asarray(poly_coeffs_m(z_val), dtype=numpy.complex128)
#         return numpy.roots(coeffs)

#     # If user asked for a real-axis value, interpret as boundary value from C+.
#     if z.imag == 0.0:
#         if eps is None:
#             eps = 1e-8 * max(1.0, abs(z))
#         z_eval = z + 1j * float(eps)
#     else:
#         z_eval = z

#     half_sign = numpy.sign(z_eval.imag)
#     if half_sign == 0.0:
#         half_sign = 1.0

#     z0 = 1j * float(half_sign) * (1. + height * mom.radius(order))
#     target = mom.stieltjes(z0, order)

#     # Initialize at z0 via asymptotic / Im-sign selection.
#     w_prev = select_root(poly_roots(z0), z0, target)

#     # Straight-line homotopy from z0 to z_eval.
#     for tau in numpy.linspace(0.0, 1.0, int(steps) + 1)[1:]:
#         z_tau = z0 + tau * (z_eval - z0)
#         w_prev = select_root(poly_roots(z_tau), z_tau, w_prev)

#     return w_prev
