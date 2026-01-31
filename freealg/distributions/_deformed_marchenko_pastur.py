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
from .._algebraic_form._sheets_util import _pick_physical_root_scalar

__all__ = ['DeformedMarchenkoPastur']


# =========================
# Deformed Marchenko Pastur
# =========================

class DeformedMarchenkoPastur(object):
    """
    Deformed Marchenko-Pastur model.

    Notes
    -----

    Silverstein / companion Stieltjes variable

    For sample-covariance, free multiplicative convolution with :math:`MP_c`:
    Let :math:`u(z)` be the *companion* Stieltjes transform (often denoted
    :math:`\\underline{m})`. It satisfies the Silverstein equation:

    .. math::

        z = -1/u + c * E_H[ t / (1 + t u) ].

    For H = w1 \\delta_{t1} + w2 \\delta_{t2}:

    .. math::

        z = -1/u + c*( w1*t1/(1+t1 u) + w2*t2/(1+t2 u) ).

    Then the (ordinary) Stieltjes transform m(z) of \\mu = H \\boxtimes MP_c is

    .. math::

        u = -(1-c)/z + c m

    (equivalently :math:`m = (u + (1-c)/z)/c` for :math:`c>0`).

    This module solves for u (cubic when H has two atoms), then maps to m.

    Reference for the Silverstein equation form:

    .. math::

        z = -1/u + c \\int t/(1 + t u) dH(t).
    """

    # ====
    # init
    # ====

    def __init__(self, t1, t2, w1, c=1.0):
        """
        Initialization.
        """

        if not (0.0 <= w1 <= 1.0):
            raise ValueError("w1 must be in [0, 1].")

        if c < 0.0:
            raise ValueError("c must be >= 0.")

        if t1 < 0.0 or t2 < 0.0:
            raise ValueError("t1 and t2 must be >= 0 for a covariance model.")

        self.t1 = t1
        self.t2 = t2
        self.w1 = w1
        self.c = c

    # ====================
    # roots cubic u scalar
    # ====================

    def _roots_cubic_u_scalar(self, z):
        """
        Solve the cubic for u = \\underline{m}(z) for H = w_1
        \\delta_{t_1} + (1-w_1)
        \\delta_{t_2}.
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        c = self.c

        w2 = 1.0 - w1
        mu1 = w1 * t1 + w2 * t2

        # Cubic coefficients for u:
        # (z t1 t2) u^3 + ( z(t1+t2) + t1 t2(1-c) ) u^2
        # + ( z + (t1+t2) - c*mu1 ) u + 1 = 0
        c3 = z * (t1 * t2)
        c2 = z * (t1 + t2) + (t1 * t2) * (1.0 - c)
        c1 = z + (t1 + t2) - c * mu1
        c0 = 1.0

        return numpy.roots([c3, c2, c1, c0])

    # ==============
    # solve u Newton
    # ==============

    def _solve_u_newton(self, z, u0=None, max_iter=100, tol=1e-12):
        """
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        c = self.c

        w2 = 1.0 - w1
        if u0 is None:
            u = -1.0 / z
        else:
            u = complex(u0)

        for _ in range(int(max_iter)):
            d1 = 1.0 + t1 * u
            d2 = 1.0 + t2 * u

            # f(u) = -1/u + c*(w1*t1/d1 + w2*t2/d2) - z
            f = (-1.0 / u) + c * (w1 * t1 / d1 + w2 * t2 / d2) - z

            # f'(u) = 1/u^2 - c*(w1*t1^2/d1^2 + w2*t2^2/d2^2)
            fp = (1.0 / (u * u)) - c * (w1 * (t1 * t1) / (d1 * d1) +
                                        w2 * (t2 * t2) / (d2 * d2))

            step = f / fp
            u2 = u - step
            if abs(step) < tol * (1.0 + abs(u2)):
                return u2, True
            u = u2

        return u, False

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z, max_iter=100, tol=1e-12):
        """
        Physical/Herglotz branch of m(z) for \\mu = H \\boxtimes MP_c with
        H = w_1 \\delta_{t_1} + (1-w_1) \\delta_{t_2}.
        Fast masked Newton in u (companion Stieltjes), keeping z's original
        shape.
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        c = self.c

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        c = float(c)
        if c < 0.0:
            raise ValueError("c must be >= 0.")

        w2 = 1.0 - w1

        if c == 0.0:
            out = (w1 / (t1 - z)) + (w2 / (t2 - z))
            return out.reshape(()) if scalar else out

        # u initial guess
        u = -1.0 / z
        active = numpy.isfinite(u)

        for _ in range(int(max_iter)):
            if not numpy.any(active):
                break

            # IMPORTANT: use integer indices (works for any ndim; avoids
            # boolean-mask aliasing issues)
            idx = numpy.flatnonzero(active)
            ua = u.ravel()[idx]
            za = z.ravel()[idx]

            d1 = 1.0 + t1 * ua
            d2 = 1.0 + t2 * ua

            f = (-1.0 / ua) + c * (w1 * t1 / d1 + w2 * t2 / d2) - za
            fp = (1.0 / (ua * ua)) - c * (
                w1 * (t1 * t1) / (d1 * d1) +
                w2 * (t2 * t2) / (d2 * d2))

            step = f / fp
            un = ua - step

            # write back u
            u_flat = u.ravel()
            u_flat[idx] = un

            converged = numpy.abs(step) < tol * (1.0 + numpy.abs(un))
            still = (~converged) & numpy.isfinite(un)

            # update active only at the previously-active locations
            a_flat = active.ravel()
            a_flat[idx] = still

        # Herglotz sanity: sign(Im z) == sign(Im u)
        sign = numpy.where(numpy.imag(z) >= 0.0, 1.0, -1.0)
        bad = (~numpy.isfinite(u)) | (sign * numpy.imag(u) <= 0.0)

        if numpy.any(bad):
            zb = z.ravel()
            ub = u.ravel()
            bad_idx = numpy.flatnonzero(bad)
            for i in bad_idx:
                zi = zb[i]
                u_roots = self._roots_cubic_u_scalar(zi)
                ub[i] = _pick_physical_root_scalar(zi, u_roots)
            u = ub.reshape(z.shape)

        m = (u + (1.0 - c) / z) / c

        if scalar:
            return m.reshape(())
        return m

    # =======
    # density
    # =======

    def density(self, x, eta=1e-3, ac_only=True):
        """
        Density via Stieltjes inversion with robust x-continuation.

        Notes
        -----

          - Do not warm-start across x<0 (MP-type support is >=0).
          - Reset warm-start when previous u is (nearly) real.
          - If Newton lands on a non-Herglotz root, fall back to cubic roots +
            pick.

        If ac_only is True and c < 1, subtract the smeared atom at zero of mass
        (1-c) for visualization.
        """

        x = numpy.asarray(x, dtype=numpy.float64)
        z = x + 1j * float(eta)

        m = self.stieltjes(z)
        rho = numpy.imag(m) / numpy.pi

        # Optional: remove the atom at zero (only for visualization of AC part)
        if ac_only and (self.c > 1.0):
            w0 = 1.0 - 1.0 / self.c
            rho = rho - w0 * (float(eta) / numpy.pi) / \
                (x * x + float(eta) * float(eta))

        return numpy.maximum(rho, 0.0)

    # =====
    # roots
    # =====

    def roots(self, z):
        """
        Return all 3 algebraic roots of m(z) (via roots for u then mapping to
        m).
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        c = self.c

        z = numpy.asarray(z, dtype=numpy.complex128)
        scalar = (z.ndim == 0)
        if scalar:
            z = z.reshape((1,))

        c = float(c)
        if c < 0.0:
            raise ValueError("c must be >= 0.")

        zf = z.ravel()
        out = numpy.empty((zf.size, 3), dtype=numpy.complex128)

        if c == 0.0:
            w2 = 1.0 - w1
            mr = (w1 / (t1 - zf)) + (w2 / (t2 - zf))
            out[:, 0] = mr
            out[:, 1] = mr
            out[:, 2] = mr
        else:
            for i in range(zf.size):
                u_roots = self._roots_cubic_u_scalar(zf[i])
                out[i, :] = (u_roots + (1.0 - c) / zf[i]) / c

        out = out.reshape(z.shape + (3,))
        if scalar:
            return out.reshape((3,))
        return out

    # =======
    # support
    # =======

    def support(self, eta=2e-4, n_probe=4000, thr=5e-4, x_max=None, x_pad=0.05,
                method='quartic'):
        """
        Estimate support intervals of μ = H \\boxtimes MP_c where H = w1
        \\delta_{t1} + (1-w1) \\delta_{t2}.

        Parameters
        ----------
        t1, t2 : float
            Atom locations (typically >0).
        w1 : float
            Weight of atom at t1.
        c : float
            MP aspect ratio parameter.
        method : {'quartic','probe'}
            - 'quartic' (default): compute endpoints from the real Silverstein
              critical equation x'(u)=0 (fast; robust for detecting split /
              merged bulks).
            - 'probe': legacy density probing using :func:`density` on a grid
              (can miss tiny gaps due to finite-eta leakage).

        Notes
        -----
        In the companion variable u = \\underline{m}(z), the real mapping is

            x(u) = -1/u + c * ( w1*t1/(1+t1 u) + (1-w1)*t2/(1+t2 u) ),

        and support endpoints occur at critical points where

            x'(u) = 0  <=>  1/u^2 = c * ( w1*t1^2/(1+t1 u)^2 + (1-w1)*t2^2/
            (1+t2 u)^2 ).

        For two atoms, this reduces to a quartic polynomial in u, so endpoints
        can be obtained with a handful of root solves (no expensive probing).
        """

        # Unpack parameters
        t1 = self.t1
        t2 = self.t2
        w1 = self.w1
        c = self.c

        c = float(c)
        if c < 0.0:
            raise ValueError("c must be >= 0.")
        if not (0.0 <= w1 <= 1.0):
            raise ValueError("w1 must be in [0, 1].")

        if method not in ('quartic', 'probe'):
            raise ValueError("method must be 'quartic' or 'probe'.")

        # --- fast endpoint finder via quartic in u ---
        if method == 'quartic':
            w2 = 1.0 - w1

            # Build the quartic polynomial:
            #   A(u)^2 B(u)^2 - c u^2 ( w1 t1^2 B(u)^2 + w2 t2^2 A(u)^2 ) = 0
            # where A(u)=1+t1 u, B(u)=1+t2 u.
            u = numpy.poly1d([1.0, 0.0])          # u
            A = 1.0 + float(t1) * u
            B = 1.0 + float(t2) * u
            A2 = A * A
            B2 = B * B
            P = (A2 * B2) - c * (u * u) * \
                (w1 * (t1 * t1) * B2 + w2 * (t2 * t2) * A2)

            u_roots = numpy.roots(P.c)

            # keep real negative roots away from poles u=-1/t1,-1/t2 and from 0
            poles = []
            if float(t1) != 0.0:
                poles.append(-1.0 / float(t1))
            if float(t2) != 0.0:
                poles.append(-1.0 / float(t2))

            u_crit = []
            for r in u_roots:
                if not numpy.isfinite(r):
                    continue
                if abs(r.imag) > 1e-10 * (1.0 + abs(r.real)):
                    continue
                ur = float(r.real)
                if ur >= 0.0:
                    continue
                if abs(ur) < 1e-14:
                    continue
                too_close = False
                for p in poles:
                    if abs(ur - p) < 1e-10 * (1.0 + abs(p)):
                        too_close = True
                        break
                if too_close:
                    continue
                u_crit.append(ur)

            u_crit = sorted(set(u_crit))
            if len(u_crit) < 2:
                # Fallback to probing if quartic degenerates numerically
                method = 'probe'
            else:
                def x_of_u(uu):
                    return (-1.0 / uu) + c * (w1 * t1 / (1.0 + t1 * uu) +
                                              w2 * t2 / (1.0 + t2 * uu))

                x_crit = []
                for uu in u_crit:
                    xv = x_of_u(uu)
                    if numpy.isfinite(xv):
                        x_crit.append(float(xv))

                x_crit = sorted(x_crit)
                # endpoints come in pairs; build candidate intervals
                cand = []
                for k in range(0, len(x_crit) - 1, 2):
                    a = x_crit[k]
                    b = x_crit[k + 1]
                    if b > a:
                        cand.append((a, b))

                # validate each candidate interval by checking rho at midpoints
                cuts = []
                for a, b in cand:
                    mid = 0.5 * (a + b)
                    # very cheap check (one evaluation)
                    rh = float(self.density(numpy.array([mid]),
                                            eta=max(eta, 1e-8))[0])
                    if numpy.isfinite(rh) and (rh > 0.0):
                        aa = max(0.0, a)  # MP-type spectra should be >=0
                        cuts.append((aa, b))

                # If everything validated out (rare), fall back to probe.
                if len(cuts) > 0:
                    return cuts
                method = 'probe'

        # --- legacy probing (kept as fallback / comparison) ---
        # Heuristic x-range
        tmax = float(max(abs(t1), abs(t2), 1e-12))
        if x_max is None:
            s = (1.0 + numpy.sqrt(max(c, 0.0))) ** 2
            x_max = 3.0 * tmax * s + 1.0
        x_max = float(x_max)

        x_min = -float(x_pad) * x_max

        x = numpy.linspace(x_min, x_max, int(n_probe))
        rho = self.density(x, eta=float(eta))

        good = numpy.isfinite(rho) & (rho > float(thr))
        if not numpy.any(good):
            return []

        idx = numpy.where(good)[0]
        breaks = numpy.where(numpy.diff(idx) > 1)[0]
        segments = []
        start = idx[0]
        for b in breaks:
            end = idx[b]
            segments.append((start, end))
            start = idx[b + 1]
        segments.append((start, idx[-1]))

        def rho_scalar(x0):
            return float(self.density(numpy.array([x0]), eta=float(eta))[0])

        cuts = []
        for i0, i1 in segments:
            a0 = float(x[max(i0 - 1, 0)])
            a1 = float(x[i0])
            b0 = float(x[i1])
            b1 = float(x[min(i1 + 1, x.size - 1)])

            # left edge
            lo, hi = a0, a1
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if rho_scalar(mid) > thr:
                    hi = mid
                else:
                    lo = mid
            a = hi

            # right edge
            lo, hi = b0, b1
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if rho_scalar(mid) > thr:
                    lo = mid
                else:
                    hi = mid
            b = lo

            if numpy.isfinite(a) and numpy.isfinite(b) and (b > a + 1e-10):
                cuts.append((max(0.0, a), b))

        return cuts

    # ======
    # matrix
    # ======

    def matrix(self, size, seed=None):
        """
        Generate matrix with the spectral density of the distribution.

        Parameters
        ----------

        size : int
            Size :math:`n` of the matrix.

        seed : int, default=None
            Seed for random number generator.

        Returns
        -------

        A : numpy.ndarray
            A matrix of the size :math:`n \\times n`.

        Notes
        -----

        Generate an :math:`n x n` sample covariance matrix :math:`\\mathbf{S}`
        whose ESD converges to :math:`H \\boxtimes MP_c`, where
        :math:`H = w_1 \\delta_{t_1} + (1-w_1) \\delta_{t_2}`.

        Finite :math:`n` construction:

        * :math:`m` is chosen so that :math:`n/m` approx :math:`c` (when
          :math:`c>0`),
        * :math:`Z` has i.i.d. :math:`N(0,1)`,
        * :math:`\\boldsymbol{\\Sigma}` has eigenvalues :math:`t_1`,
          :math:`t_2` with proportions
          :math:`w_1`, and :math:`1-w_1`,
        * :math:`\\mathbf{S} = (1/m) \\boldsymbol{\\Sigma}^{1/2} \\mathbf{Z}
          \\mathbf{Z}^T \\boldsymbol{\\Sigma}^{1/2}`.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import MarchenkoPastur
            >>> mp = MarchenkoPastur(1/50)
            >>> A = mp.matrix(2000)
        """

        n = int(size)
        if n <= 0:
            raise ValueError("size must be a positive integer.")

        # Unpack parameters
        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        c = float(self.c)

        rng = numpy.random.default_rng(seed)

        # Choose m so that n/m approx c (for c>0). For c=0, return population
        # Sigma.
        if c == 0.0:
            n1 = int(round(w1 * n))
            n1 = max(0, min(n, n1))
            d = numpy.empty(n, dtype=numpy.float64)
            d[:n1] = t1
            d[n1:] = t2
            rng.shuffle(d)
            return numpy.diag(d)

        # m must be positive integer
        m = int(round(n / c)) if c > 0.0 else n
        m = max(1, m)

        # Build diagonal Sigma^{1/2} with two atoms
        n1 = int(round(w1 * n))
        n1 = max(0, min(n, n1))

        s = numpy.empty(n, dtype=numpy.float64)
        s[:n1] = numpy.sqrt(t1)
        s[n1:] = numpy.sqrt(t2)
        rng.shuffle(s)

        # Draw Z and form X = Sigma^{1/2} Z / sqrt(m)
        Z = rng.standard_normal((n, m))
        X = (s[:, None] * Z) / numpy.sqrt(m)

        # Sample covariance
        S = X @ X.T

        return S

    # ====
    # poly
    # ====

    def poly(self):
        """
        Return a_coeffs for the exact cubic P(z,m)=0 of the two-atom deformed
        MP model.

        This is the eliminated polynomial in m (not underline{m}).
        a_coeffs[i, j] is the coefficient of z^i m^j.
        Shape is (3, 4).
        """

        t1 = float(self.t1)
        t2 = float(self.t2)
        w1 = float(self.w1)
        w2 = 1.0 - w1
        c = float(self.c)

        # mu1 = w1 * t1 + w2 * t2

        a = numpy.zeros((3, 4), dtype=numpy.complex128)

        # NOTE: This polynomial is defined up to a global nonzero factor.
        # The scaling below is chosen so that the m^3 term is (-c^3 t1 t2) z^2.

        # ---- m^3:  (-c^3 t1 t2) z^2
        a[2, 3] = -(c**3) * t1 * t2

        # ---- m^2:  -( 2 c^3 t1 t2 z - 2 c^2 t1 t2 z + c^2 (t1+t2) z^2 )
        a[0, 2] = 0.0
        a[1, 2] = -(2.0 * (c**3) * t1 * t2 - 2.0 * (c**2) * t1 * t2)
        a[2, 2] = -(c**2) * (t1 + t2)

        # ---- m^1:
        #   -c * [ c^2 t1 t2 - 2 c t1 t2 + t1 t2
        #          + z^2
        #          + z*( -c*w1*t1 + 2c*t1 + c*w1*t2 + c*t2 - t1 - t2 ) ]
        a[0, 1] = -c * ((c**2) * t1 * t2 - 2.0 * c * t1 * t2 + t1 * t2)
        a[1, 1] = -c * ((-c * w1 * t1) + (2.0 * c * t1) + (c * w1 * t2) +
                        (c * t2) - t1 - t2)
        a[2, 1] = -c * (1.0)

        # ---- m^0:  -c z + c(1-c) (w2 t1 + w1 t2)
        a[0, 0] = c * (1.0 - c) * (w2 * t1 + w1 * t2)
        a[1, 0] = -c
        a[2, 0] = 0.0

        return a
