# =======
# Imports
# =======

import numpy
from tqdm import tqdm
from ._moments import AlgebraicStieltjesMoments

__all__ = ['StieltjesPoly']


# ==============
# Stieltjes Poly
# ==============

class StieltjesPoly(object):
    """
    Stieltjes-branch evaluator for P(z, m)=0 with robust 1D tracking.

    For 1D arrays on a horizontal line (z = x + i*delta), uses Viterbi
    tracking across the whole line to avoid branch mis-selection.
    Otherwise falls back to pointwise evaluate().

    Parameters
    ----------
    a : ndarray
        Polynomial coefficient matrix.
    eps : float or None
        Imaginary offset when Im(z)=0.
    height : float
        Radius factor for safe imaginary height.
    steps : int
        Vertical continuation steps in evaluate().
    order : int
        Moment order.
    lam_im : float
        Viterbi penalty strength for tiny |Im(m)|.
    tol_im : float
        Herglotz tolerance.
    """

    def __init__(self, a, eps=None, height=2.0, steps=100, order=15,
                 lam_im=1.0e3, tol_im=1.0e-12):
        a = numpy.asarray(a)
        if a.ndim != 2:
            raise ValueError("a must be a 2D array.")

        self.a = a
        self.a_l, _ = a.shape

        self.eps = eps
        self.height = float(height)
        self.steps = int(steps)
        self.order = int(order)

        self.lam_im = float(lam_im)
        self.tol_im = float(tol_im)

        self.mom = AlgebraicStieltjesMoments(a)

        self.rad = 1.0 + self.height * self.mom.radius(self.order)
        self.z0_p = 1j * self.rad
        self.z0_m = -1j * self.rad

        self.m0_p = self.mom.stieltjes(self.z0_p, self.order)
        self.m0_m = self.mom.stieltjes(self.z0_m, self.order)

    # -----------
    # poly roots
    # -----------

    def _poly_coeffs_m(self, z_val):
        z_powers = z_val ** numpy.arange(self.a_l)
        return (z_powers @ self.a)[::-1]

    def _poly_roots(self, z_val):
        coeffs = numpy.asarray(self._poly_coeffs_m(z_val),
                               dtype=numpy.complex128)
        return numpy.roots(coeffs)

    # -------------
    # point evaluate
    # -------------

    def _select_root_continuity(self, roots, z, target):
        z = complex(z)
        roots = numpy.asarray(roots, dtype=numpy.complex128).ravel()

        s = numpy.sign(z.imag)
        if s == 0.0:
            s = 1.0

        im = numpy.imag(roots) * s
        cand = roots[im > -self.tol_im]
        if cand.size == 0:
            cand = roots

        t = complex(target)
        return cand[int(numpy.argmin(numpy.abs(cand - t)))]

    def evaluate(self, z):
        z = complex(z)

        if z.imag == 0.0:
            if self.eps is None:
                eps_loc = 1e-8 * max(1.0, abs(z))
            else:
                eps_loc = float(self.eps)
            z_eval = z + 1j * eps_loc
        else:
            z_eval = z

        s = numpy.sign(z_eval.imag)
        if s == 0.0:
            s = 1.0

        z_mid = complex(z_eval.real, s * self.rad)

        # anchor using moment estimate at z_mid
        target = self.mom.stieltjes(z_mid, self.order)
        w_prev = self._select_root_continuity(self._poly_roots(z_mid), z_mid, target)

        # vertical continuation
        for tau in numpy.linspace(0.0, 1.0, int(self.steps) + 1)[1:]:
            z_tau = z_mid + tau * (z_eval - z_mid)
            w_prev = self._select_root_continuity(self._poly_roots(z_tau), z_tau, w_prev)

        return w_prev

    # -----------------
    # viterbi utilities
    # -----------------

    def _is_flat_imag_line_1d(self, z):
        if z.ndim != 1 or z.size < 2:
            return False
        y = numpy.imag(z)
        if not numpy.all(numpy.isfinite(y)):
            return False
        y0 = float(y[0])
        if y0 == 0.0:
            return False
        if not numpy.all(numpy.sign(y) == numpy.sign(y0)):
            return False
        tol = 1e-14 * max(1.0, abs(y0))
        return numpy.max(numpy.abs(y - y0)) <= tol

    def _herglotz_mask(self, roots, z):
        s = numpy.sign(z.imag)
        if s == 0.0:
            s = 1.0
        return (numpy.imag(roots) * s) > -self.tol_im

    def _viterbi_track(self, z_sorted, roots_all):
        z_sorted = numpy.asarray(z_sorted, dtype=numpy.complex128)
        R = numpy.asarray(roots_all, dtype=numpy.complex128)
        N, S = R.shape

        unary = numpy.zeros((N, S), dtype=numpy.float64)
        for k in range(N):
            mask = self._herglotz_mask(R[k], z_sorted[k])
            unary[k, ~mask] += 1e30
            im = numpy.abs(numpy.imag(R[k]))
            unary[k] += self.lam_im / numpy.maximum(im, 1e-16)

        dp = numpy.full((N, S), numpy.inf, dtype=numpy.float64)
        prev = numpy.full((N, S), -1, dtype=numpy.int64)

        dp[0] = unary[0]

        for k in range(1, N):
            diff = R[k][None, :] - R[k-1][:, None]
            cost = dp[k-1][:, None] + (numpy.abs(diff) ** 2)
            i_star = numpy.argmin(cost, axis=0)
            dp[k] = unary[k] + cost[i_star, numpy.arange(S)]
            prev[k] = i_star

        j = int(numpy.argmin(dp[-1]))
        path = numpy.empty(N, dtype=numpy.int64)
        path[-1] = j
        for k in range(N - 1, 0, -1):
            path[k - 1] = prev[k, path[k]]

        return R[numpy.arange(N), path]

    # -----
    # call
    # -----

    def __call__(self, z, progress=False):
        if numpy.isscalar(z):
            return self.evaluate(z)

        z_arr = numpy.asarray(z, dtype=numpy.complex128)

        # Viterbi path only for 1D horizontal lines
        if z_arr.ndim == 1 and self._is_flat_imag_line_1d(z_arr):
            order = numpy.argsort(numpy.real(z_arr))
            inv = numpy.empty_like(order)
            inv[order] = numpy.arange(order.size)

            z_sorted = z_arr[order]
            roots_all = numpy.array([self._poly_roots(zk) for zk in z_sorted],
                                    dtype=numpy.complex128)
            m_sorted = self._viterbi_track(z_sorted, roots_all)
            return m_sorted[inv]

        # Fallback: elementwise
        out = numpy.empty(z_arr.shape, dtype=numpy.complex128)
        it = numpy.ndindex(z_arr.shape)
        if progress:
            it = tqdm(it, total=z_arr.size)
        for idx in it:
            out[idx] = self.evaluate(z_arr[idx])
        return out
