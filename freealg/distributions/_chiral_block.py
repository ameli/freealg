# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# ======
# Import
# ======

import numpy
import collections
from .._geometric_form._elliptic_functions import ellipj
from .._geometric_form._continuation_genus1 import mobius_z

__all__ = ['ChiralBlock']


# ============
# Chiral Block
# ============

class ChiralBlock(object):
    """
    Twisted chiral block model.

    Parameters
    ----------

    alpha : float
    beta : float
    """

    # ====
    # init
    # ====

    def __init__(self, alpha, beta, c):
        """
        Initialization.
        """

        self.alpha = alpha
        self.beta = beta
        self.c = c

    # =======
    # density
    # =======

    def density(self, x):
        """
        Absolutely continous density, and the atom.
        """

        # Parameters
        alpha = self.alpha
        beta = self.beta
        c = self.c

        t = (x - alpha) * (x - beta)

        Delta_t = t * t - 2.0 * (c + 1.0) * t + (c - 1.0) * (c - 1.0)
        s = numpy.sqrt(numpy.maximum(0.0, -Delta_t))

        sgn = numpy.sign(x - alpha)
        sgn = numpy.where(sgn == 0.0, 1.0, sgn)
        sd = 1j * s * sgn

        A = t + (c - 1.0)

        xa = x - alpha
        xa_safe = numpy.where(xa == 0.0, numpy.nan, xa)

        u = (-A + sd) / (2.0 * c * xa_safe)
        den = (t - c + 1.0) + sd
        v = -2.0 * xa_safe / den

        m = (c / (1.0 + c)) * u + (1.0 / (1.0 + c)) * v
        rho = m.imag / numpy.pi

        rho = numpy.where(Delta_t < 0.0, rho, 0.0)
        rho = numpy.where(numpy.isfinite(rho), rho, 0.0)
        rho = numpy.maximum(rho, 0.0)

        # Atom location and weight
        if numpy.abs(c - 1.0) < 1e-4:
            atom_loc = None
            atom_w = None
        elif c > 1.0:
            atom_loc = alpha
            atom_w = (c - 1.0) / (c + 1.0)
        elif c < 1.0:
            atom_loc = beta
            atom_w = (1.0 - c) / (c + 1.0)

        return rho, atom_loc, atom_w

    # ===========
    # sqrt like t
    # ===========

    def _sqrt_like_t(self, delta, t):
        """
        """

        s = numpy.sqrt(delta)
        flip = numpy.real(t * numpy.conjugate(s)) < 0.0
        s = numpy.where(flip, -s, s)
        return s

    # =========
    # stieltjes
    # =========

    def stieltjes(self, z, alt_branch=False):
        """
        Physical Stieltjes transform
        """

        # Parameters
        alpha = self.alpha
        beta = self.beta
        c = self.c

        t = (z - alpha) * (z - beta)
        delta = t * t - 2.0 * (c + 1.0) * t + (c - 1.0) * (c - 1.0)

        s = self._sqrt_like_t(delta, t)

        A = t + (c - 1.0)

        za = z - alpha
        za_safe = numpy.where(za == 0.0, numpy.nan, za)

        u_p = (-A + s) / (2.0 * c * za_safe)
        u_m = (-A - s) / (2.0 * c * za_safe)

        den_p = (t - c + 1.0) + s
        den_m = (t - c + 1.0) - s

        v_p = -2.0 * za_safe / den_p
        v_m = -2.0 * za_safe / den_m

        m_p = (c / (1.0 + c)) * u_p + (1.0 / (1.0 + c)) * v_p
        m_m = (c / (1.0 + c)) * u_m + (1.0 / (1.0 + c)) * v_m

        mask_p = numpy.imag(z) >= 0.0
        pick_p = \
            numpy.where(mask_p, numpy.imag(m_p) >= 0.0, numpy.imag(m_p) <= 0.0)

        m1 = numpy.where(pick_p, m_p, m_m)
        m2 = numpy.where(pick_p, m_m, m_p)

        if alt_branch:
            return m2
        return m1

    # =======
    # support
    # =======

    def support(self):
        """
        Support
        """

        # Parameters
        alpha = self.alpha
        beta = self.beta
        c = self.c

        s = numpy.sqrt(c)
        t_min = (s - 1.0) * (s - 1.0)
        t_max = (s + 1.0) * (s + 1.0)

        d = (alpha - beta) * (alpha - beta)

        r_min = numpy.sqrt(d + 4.0 * t_min)
        r_max = numpy.sqrt(d + 4.0 * t_max)

        a1 = 0.5 * (alpha + beta - r_max)
        b1 = 0.5 * (alpha + beta - r_min)
        a2 = 0.5 * (alpha + beta + r_min)
        b2 = 0.5 * (alpha + beta + r_max)

        return [(a1, b1), (a2, b2)]

    # ==================
    # stieltjes on torus
    # ==================

    def stieltjes_on_torus(self, u, lam, a1, b1, a2, b2):
        """
        Exact m on the torus (no fit), continuous, by:
          1) computing the two exact candidates mA(z(u)) and mB(z(u)),
          2) selecting a continuous branch on the torus via BFS continuation,
          3) applying an optimal "half-cycle swap" along the phi-direction
             (choosing the cut location automatically) to ensure global
             consistency without breaking periodicity (fixes the equator-circle
             issue).

        Usage:
            mT_exact = eval_m_on_torus_exact(u, lam, a1, b1, a2, b2, alpha,
            beta, c)
        """

        # ---------------------------
        # core (drop seam duplicates)
        # ---------------------------

        uc = u[:-1, :-1]
        nphi, ntheta = uc.shape

        # ----------------------------
        # map u -> z via X = lam sn^2
        # ----------------------------

        sn, cn, dn, _ = ellipj(uc, lam)
        Xc = lam * (sn * sn)
        zc = mobius_z(Xc, a1, b1, a2, b2)

        # -------------------------------
        # exact branch candidates at z(u)
        # -------------------------------

        mA = self.stieltjes(zc, alt_branch=False)  # candidate A
        mB = self.stieltjes(zc, alt_branch=True)   # candidate B

        finA = numpy.isfinite(mA)
        finB = numpy.isfinite(mB)

        # output core and chosen flags
        mC = numpy.full_like(mA, numpy.nan, dtype=complex)

        # 0->A, 1->B, -1 unset
        chosen = numpy.full((nphi, ntheta), -1, dtype=numpy.int8)

        # -----------------------------------
        # seed: find a point with both finite
        # -----------------------------------

        if finA[0, 0] and finB[0, 0]:
            i0, j0 = 0, 0
        else:
            idx = numpy.argwhere(finA & finB)
            if idx.size == 0:
                raise RuntimeError("No points where both branches are finite.")
            i0, j0 = idx[0]

        # deterministic seed choice (any deterministic rule is fine)
        # prefer candidate whose Im(m) roughly matches sign of Im(z)
        if numpy.imag(zc[i0, j0]) >= 0:
            pickA = (numpy.imag(mA[i0, j0]) >= numpy.imag(mB[i0, j0]))
        else:
            pickA = (numpy.imag(mA[i0, j0]) <= numpy.imag(mB[i0, j0]))

        chosen[i0, j0] = 0 if pickA else 1
        mC[i0, j0] = mA[i0, j0] if pickA else mB[i0, j0]

        # ----------------------------------------------
        # BFS continuation on torus (periodic neighbors)
        # ----------------------------------------------

        q = collections.deque([(i0, j0)])
        while q:
            i, j = q.popleft()
            ref = mC[i, j]
            if not numpy.isfinite(ref):
                continue

            nbrs = [((i - 1) % nphi, j),
                    ((i + 1) % nphi, j),
                    (i, (j - 1) % ntheta),
                    (i, (j + 1) % ntheta)]

            for ii, jj in nbrs:
                if chosen[ii, jj] != -1:
                    continue

                a_ok = finA[ii, jj]
                b_ok = finB[ii, jj]

                if not a_ok and not b_ok:
                    chosen[ii, jj] = 2
                    mC[ii, jj] = numpy.nan
                    continue

                if a_ok and not b_ok:
                    chosen[ii, jj] = 0
                    mC[ii, jj] = mA[ii, jj]
                    q.append((ii, jj))
                    continue

                if b_ok and not a_ok:
                    chosen[ii, jj] = 1
                    mC[ii, jj] = mB[ii, jj]
                    q.append((ii, jj))
                    continue

                # both finite: choose closer to already-selected neighbor
                # (continuation)
                da = abs(mA[ii, jj] - ref)
                db = abs(mB[ii, jj] - ref)
                if da <= db:
                    chosen[ii, jj] = 0
                    mC[ii, jj] = mA[ii, jj]
                else:
                    chosen[ii, jj] = 1
                    mC[ii, jj] = mB[ii, jj]
                q.append((ii, jj))

        # ----------------------------------------------------------------
        # Step 3: choose the correct "half-cycle swap" cut automatically
        #
        # Build the "other-sheet" field mOther (swap at every point)
        # and then choose a contiguous block of phi-rows of length L=nphi/2
        # to swap, with cut location k chosen to minimize the two boundary
        # jumps.
        #
        # This fixes the "entire equator circle wrong" issue.
        # -----------------------------------------------------------------

        if nphi % 2 != 0:
            # If odd, we still do a near-half swap; but nphi is typically even.
            L = nphi // 2
        else:
            L = nphi // 2

        # swapped-everywhere alternative (only valid where chosen is 0/1)
        mOther = numpy.full_like(mC, numpy.nan, dtype=complex)
        ok0 = (chosen == 0)
        ok1 = (chosen == 1)
        mOther[ok0] = mB[ok0]
        mOther[ok1] = mA[ok1]

        # boundary cost between row i and i+1 (mod nphi) when row i uses mC (0)
        # and row i+1 uses mOther (1) and when row i uses mOther (1) and row
        # i+1 uses mC (0).
        def boundary_cost_rowpair(Arow, Brow):
            d = Arow - Brow
            ok = numpy.isfinite(d)
            return numpy.median(numpy.abs(d[ok])) if numpy.any(ok) \
                else numpy.inf

        # row i uses mC, row i+1 uses mOther
        c01 = numpy.full(nphi, numpy.inf, dtype=float)

        # row i uses mOther, row i+1 uses mC
        c10 = numpy.full(nphi, numpy.inf, dtype=float)

        for i in range(nphi):
            ip = (i + 1) % nphi
            c01[i] = boundary_cost_rowpair(mC[i, :],     mOther[ip, :])
            c10[i] = boundary_cost_rowpair(mOther[i, :], mC[ip, :])

        # For a swap-block starting at k (rows k..k+L-1 swapped),
        # the two cut boundaries are:
        #   b1 = k-1  : (unswapped -> swapped) uses c01[b1]
        #   b2 = k+L-1: (swapped   -> unswapped) uses c10[b2]
        best_k = 0
        best_cost = numpy.inf
        for k in range(nphi):
            b1 = (k - 1) % nphi
            b2 = (k + L - 1) % nphi
            cost = c01[b1] + c10[b2]
            if cost < best_cost:
                best_cost = cost
                best_k = k

        # apply that optimal contiguous swap block
        swap_rows = numpy.zeros(nphi, dtype=bool)
        for t in range(L):
            swap_rows[(best_k + t) % nphi] = True

        mC2 = mC.copy()
        mC2[swap_rows, :] = mOther[swap_rows, :]
        mC = mC2

        # -----------------------------
        # rewrap seams to match u shape
        # -----------------------------

        mT = numpy.empty_like(u, dtype=complex)
        mT[:-1, :-1] = mC
        mT[-1, :-1] = mC[0, :]
        mT[:-1, -1] = mC[:, 0]
        mT[-1, -1] = mC[0, 0]

        return mT

    # ======
    # matrix
    # ======

    def matrix(self, size, seed=None):
        """
        Generate matrix with the spectral density of the distribution.

        Parameters
        ----------

        size : int
            Total size :math:`N = n + m` of the returned matrix.

        seed : int, default=None
            Seed for random number generator.

        Returns
        -------

        A : numpy.ndarray
            Symmetric matrix of shape :math:`N \\times N`.

        Notes
        -----

        Generate a :math:`(n+m) x (n+m)` matrix

        .. math::

            H =
            \\begin{bmatrix}
                \\alpha \\mathbf{I}_n & (1/\\sqrt{m})) \\mathbf{X} \\
                (1/\\sqrt{m})) \\mathbf{X}^{\\intercal} & \\beta \\mathbf{I}_m
            \\end{bmatrix}


        where :math:`\\mathbf{X}` has i.i.d. :math:`N(0,1)` entries and
        :math:`n/m` approximates :math:`c`.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import MarchenkoPastur
            >>> mp = MarchenkoPastur(1/50)
            >>> A = mp.matrix(2000)
        """

        N = int(size)
        if N <= 1:
            raise ValueError("size must be an integer >= 2.")

        # Unpack parameters
        alpha = float(self.alpha)
        beta = float(self.beta)
        c = float(self.c)

        rng = numpy.random.default_rng(seed)

        # Choose n,m so that n/m approx c and n+m = N.
        # Solve n = c m and n + m = N -> m = N/(c+1), n = cN/(c+1).
        m = int(round(N / (c + 1.0)))
        m = max(1, min(N - 1, m))
        n = N - m

        # Optionally refine to get ratio closer to c (cheap local search).
        # This keeps deterministic behavior.
        best_n = n
        best_m = m
        best_err = abs((n / float(m)) - c)
        for dm in (-2, -1, 0, 1, 2):
            mm = m + dm
            if mm <= 0 or mm >= N:
                continue
            nn = N - mm
            err = abs((nn / float(mm)) - c)
            if err < best_err:
                best_err = err
                best_n = nn
                best_m = mm
        n = best_n
        m = best_m

        # Draw X (n x m) with i.i.d. entries
        X = rng.standard_normal((n, m))

        # Assemble H
        H = numpy.zeros((N, N), dtype=numpy.float64)

        H[:n, :n] = alpha * numpy.eye(n, dtype=numpy.float64)
        H[n:, n:] = beta * numpy.eye(m, dtype=numpy.float64)

        s = 1.0 / numpy.sqrt(float(m))
        B = s * X
        H[:n, n:] = B
        H[n:, :n] = B.T

        return H
