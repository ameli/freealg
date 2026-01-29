# =======
# Imports
# =======

import numpy
from ._moments import AlgebraicStieltjesMoments
from tqdm import tqdm

__all__ = ['StieltjesPoly']


# ===========
# select root
# ===========

# def select_root(roots, z, target):
#     """
#     Select the root among Herglotz candidates at a given z closest to a
#     given target.

#     Parameters
#     ----------
#     roots : array_like of complex
#         Candidate roots for m at the given z.
#     z : complex
#         Evaluation point. The Stieltjes/Herglotz branch satisfies
#         sign(Im(m)) = sign(Im(z)) away from the real axis.
#     target : complex
#         Previous continuation value used to enforce continuity, or
#         target value.

#     Returns
#     -------
#     w : complex
#         Selected root corresponding to the Stieltjes branch.
#     """

def select_root(roots, z, target, tol_im=1e-10, tiny_im=1e-6, ratio=50.0):
    z = complex(z)
    roots = numpy.asarray(roots, dtype=numpy.complex128).ravel()

    s = numpy.sign(z.imag)
    if s == 0.0:
        s = 1.0

    # candidates in the correct half-plane (soft)
    im = numpy.imag(roots) * s
    cand = roots[im > -tol_im]
    if cand.size == 0:
        cand = roots
        im = numpy.imag(cand) * s
    else:
        im = numpy.imag(cand) * s

    # if there is a "big-imag" candidate and a "tiny-imag" candidate, pick big-imag
    im_max = float(im.max())
    im_min = float(im.min())

    if im_max > tiny_im and im_min < tiny_im and im_max / max(im_min, 1e-16) > ratio:
        return cand[int(numpy.argmax(im))]

    # otherwise fall back to continuity
    t = complex(target)
    return cand[int(numpy.argmin(numpy.abs(cand - t)))]

# def select_root(roots, z, target, tol_im=1e-10, tiny_im=1e-6, ratio=50.0):
#     z = complex(z)
#     roots = numpy.asarray(roots, dtype=numpy.complex128).ravel()

#     s = numpy.sign(z.imag)
#     if s == 0.0:
#         s = 1.0

#     im = numpy.imag(roots) * s
#     cand = roots[im > -tol_im]
#     if cand.size == 0:
#         cand = roots
#         im = numpy.imag(cand) * s
#     else:
#         im = numpy.imag(cand) * s

#     im_max = float(im.max())
#     im_min = float(im.min())

#     # If there's a clear "big-Im" vs "tiny-Im" split, take big-Im
#     if im_max > tiny_im and im_min < tiny_im and im_max / max(im_min, 1e-16) > ratio:
#         return cand[int(numpy.argmax(im))]

#     # Otherwise continuity
#     t = complex(target)
#     return cand[int(numpy.argmin(numpy.abs(cand - t)))]





# ==============
# stieltjes poly
# ==============

class StieltjesPoly(object):
    """
    Stieltjes-branch evaluator for an algebraic equation P(z, m) = 0.

    Parameters
    ----------
    a : ndarray, shape (L, K)
        Coefficient matrix defining P(z, m) in the monomial basis.
    eps : float or None, optional
        If Im(z) == 0, use z + i*eps as the boundary evaluation point.
        If None and Im(z) == 0, eps is set to 1e-8 * max(1, |z|).
    height : float, default = 2.0
        Imaginary height factor used to build a safe start radius.
    steps : int, default = 100
        Number of continuation steps along the vertical leg.
    order : int, default = 15
        Number of moments in Stieltjes estimate.
    reanchor : int, default = 1
        During 1D line-sweeps, every ``reanchor`` points we re-run a full
        evaluate() at the current point. Setting this to 1 is the most robust.
    """

    def __init__(self, a, eps=None, height=2.0, steps=100, order=15,
                 reanchor=1):
        a = numpy.asarray(a)
        if a.ndim != 2:
            raise ValueError("a must be a 2D array.")

        self.a = a
        self.a_l, _ = a.shape
        self.eps = eps
        self.height = float(height)
        self.steps = int(steps)
        self.order = int(order)
        self.reanchor = reanchor

        self.mom = AlgebraicStieltjesMoments(a)

        # Start point far enough away (imag direction)
        self.rad = 1.0 + self.height * self.mom.radius(self.order)
        self.z0_p = 1j * self.rad
        self.z0_m = -1j * self.rad

        # Moment anchors at z0
        self.m0_p = self.mom.stieltjes(self.z0_p, self.order)
        self.m0_m = self.mom.stieltjes(self.z0_m, self.order)

    def _poly_coeffs_m(self, z_val):
        z_powers = z_val ** numpy.arange(self.a_l)
        return (z_powers @ self.a)[::-1]

    def _poly_roots(self, z_val):
        coeffs = numpy.asarray(self._poly_coeffs_m(z_val),
                               dtype=numpy.complex128)
        return numpy.roots(coeffs)

    def evaluate(self, z):
        """
        Evaluate the Stieltjes-branch solution m(z) at a single point.

        Robust strategy for multi-bulk cubics:
        1) Move to z_mid = x + i*sign(y)*rad (high above the real axis).
        2) At z_mid, pick the root closest to the moment-based estimate m_mom(z_mid).
           (This is the crucial re-anchoring that prevents choosing the wrong sheet.)
        3) Continue vertically from z_mid down to z_eval = x + i*y (or x + i*eps).
        """

        z = complex(z)

        if self.steps < 1:
            raise ValueError("steps must be >= 1.")

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

        # High-imag anchor at same real part
        z_mid = complex(z_eval.real, half_sign * self.rad)

        r_mid = self._poly_roots(z_mid)
        w_prev = r_mid[int(numpy.argmax(numpy.imag(r_mid) * numpy.sign(z_mid.imag)))]


        # Moment-based target at z_mid (THIS fixes the wrong-bulk issue)
        # m_mid_target = self.mom.stieltjes(z_mid, self.order)
        m_mid_target = -1.0 / z_mid

        # Select correct branch at z_mid using the moment target
        w_prev = select_root(self._poly_roots(z_mid), z_mid, m_mid_target)

        # Vertical continuation: z_mid -> z_eval
        for tau in numpy.linspace(0.0, 1.0, int(self.steps) + 1)[1:]:
            z_tau = z_mid + tau * (z_eval - z_mid)
            w_prev = select_root(self._poly_roots(z_tau), z_tau, w_prev)

        return w_prev

    def _is_flat_imag_line_1d(self, z_flat):
        if z_flat.ndim != 1 or z_flat.size < 2:
            return False

        y = numpy.imag(z_flat)
        if not numpy.all(numpy.isfinite(y)):
            return False

        y0 = float(y[0])
        if y0 == 0.0:
            return False

        # all in same half-plane
        if not numpy.all(numpy.sign(y) == numpy.sign(y0)):
            return False

        # nearly constant imaginary part (relative)
        tol = 1e-14 * max(1.0, abs(y0))
        return numpy.max(numpy.abs(y - y0)) <= tol

    def _sweep_line(self, z_sorted, progress=False):
        n = z_sorted.size
        out = numpy.empty_like(z_sorted, dtype=numpy.complex128)

        # Always anchor the first point robustly
        m_prev = self.evaluate(z_sorted[0])
        out[0] = m_prev

        it = range(1, n)
        if progress:
            it = tqdm(it, total=n - 1)

        reanchor = self.reanchor
        do_reanchor = (reanchor is not None) and (int(reanchor) > 0)
        reanchor = int(reanchor) if do_reanchor else 0

        for k in it:
            zk = z_sorted[k]

            # With default reanchor=1, this becomes "evaluate every point"
            if do_reanchor and (k % reanchor == 0):
                m_prev = self.evaluate(zk)
            else:
                rk = self._poly_roots(zk)
                m_prev = select_root(rk, zk, m_prev)

            out[k] = m_prev

        return out

    def __call__(self, z, progress=False):
        # Scalar fast-path
        if numpy.isscalar(z):
            return self.evaluate(z)

        z_arr = numpy.asarray(z, dtype=numpy.complex128)
        out = numpy.empty(z_arr.shape, dtype=numpy.complex128)

        # 1D horizontal line sweep (density-style queries)
        if z_arr.ndim == 1:
            z_flat = z_arr
            if self._is_flat_imag_line_1d(z_flat):
                order = numpy.argsort(numpy.real(z_flat))
                inv = numpy.empty_like(order)
                inv[order] = numpy.arange(order.size)

                z_sorted = z_flat[order]
                out_sorted = self._sweep_line(z_sorted, progress=progress)

                out[:] = out_sorted[inv]
                return out

        # Fallback: elementwise
        if progress:
            indices = tqdm(numpy.ndindex(z_arr.shape), total=z_arr.size)
        else:
            indices = numpy.ndindex(z_arr.shape)

        for idx in indices:
            out[idx] = self.evaluate(z_arr[idx])

        return out
