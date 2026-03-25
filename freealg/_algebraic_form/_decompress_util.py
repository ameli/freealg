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
from ._continuation_algebraic import powers

__all__ = ['build_time_grid', 'eval_P_partials', 'inverse_stieltjes']


# ===============
# build time grid
# ===============

def build_time_grid(sizes, n0, min_n_times=0):
    """
    sizes: list/array of requested matrix sizes (e.g. [2000,3000,4000,8000])
    n0:    initial size (self.n)
    min_n_times: minimum number of time points to run Newton sweep on

    Returns
    -------
    t_all: sorted time grid to run solver on
    idx_req: indices of requested times inside t_all (same order as sizes)
    """

    sizes = numpy.asarray(sizes, dtype=float)
    alpha = sizes / float(n0)
    t_req = numpy.log(alpha)

    # Always include t=0 and T=max(t_req)
    T = float(numpy.max(t_req)) if t_req.size else 0.0
    base = numpy.unique(numpy.r_[0.0, t_req, T])
    t_all = numpy.sort(base)

    # Add points only if needed: split largest gaps
    N = int(min_n_times) if min_n_times is not None else 0
    while t_all.size < N and t_all.size >= 2:
        gaps = numpy.diff(t_all)
        k = int(numpy.argmax(gaps))
        mid = 0.5 * (t_all[k] + t_all[k+1])
        t_all = numpy.sort(numpy.unique(numpy.r_[t_all, mid]))

    # Map each requested time to an index in t_all (stable, no float drama)
    # (t_req values came from same construction, so they should match exactly;
    # still: use searchsorted + assert)
    idx_req = numpy.searchsorted(t_all, t_req)
    # optional sanity:
    # assert numpy.allclose(t_all[idx_req], t_req, rtol=0, atol=0)

    return t_all, idx_req


# ===============
# eval P partials
# ===============

def eval_P_partials(z, m, coeffs):
    """
    Evaluate P(z,m) and its partial derivatives dP/dz and dP/dm.

    This assumes P is represented by `coeffs` in the monomial basis

        P(z, m) = sum_{j=0..s} a_j(z) * m^j,
        a_j(z) = sum_{i=0..deg_z} coeffs[i, j] * z^i.

    The function returns P, dP/dz, dP/dm with broadcasting over z and m.

    Parameters
    ----------
    z : complex or array_like of complex
        First argument to P.
    m : complex or array_like of complex
        Second argument to P. Must be broadcast-compatible with `z`.
    coeffs : ndarray, shape (deg_z+1, s+1)
        Coefficient matrix for P in the monomial basis.

    Returns
    -------
    P : complex or ndarray of complex
        Value P(z,m).
    Pz : complex or ndarray of complex
        Partial derivative dP/dz evaluated at (z,m).
    Pm : complex or ndarray of complex
        Partial derivative dP/dm evaluated at (z,m).

    Notes
    -----
    For scalar (z,m), this uses Horner evaluation for a_j(z) and then Horner
    in m. For array inputs, it uses precomputed power tables via `_powers` for
    simplicity.

    Examples
    --------
    .. code-block:: python

        P, Pz, Pm = eval_P_partials(1.0 + 1j, 0.2 + 0.3j, coeffs)
    """

    z = numpy.asarray(z, dtype=complex)
    m = numpy.asarray(m, dtype=complex)

    deg_z = int(coeffs.shape[0] - 1)
    s = int(coeffs.shape[1] - 1)

    if (z.ndim == 0) and (m.ndim == 0):
        zz = complex(z)
        mm = complex(m)

        a = numpy.empty(s + 1, dtype=complex)
        ap = numpy.empty(s + 1, dtype=complex)

        for j in range(s + 1):
            c = coeffs[:, j]

            val = 0.0 + 0.0j
            for i in range(deg_z, -1, -1):
                val = val * zz + c[i]
            a[j] = val

            dval = 0.0 + 0.0j
            for i in range(deg_z, 0, -1):
                dval = dval * zz + (i * c[i])
            ap[j] = dval

        p = a[s]
        pm = 0.0 + 0.0j
        for j in range(s - 1, -1, -1):
            pm = pm * mm + p
            p = p * mm + a[j]

        pz = ap[s]
        for j in range(s - 1, -1, -1):
            pz = pz * mm + ap[j]

        return p, pz, pm

    shp = numpy.broadcast(z, m).shape
    zz = numpy.broadcast_to(z, shp).ravel()
    mm = numpy.broadcast_to(m, shp).ravel()

    zp = powers(zz, deg_z)
    mp = powers(mm, s)

    dzp = numpy.zeros_like(zp)
    for i in range(1, deg_z + 1):
        dzp[:, i] = i * zp[:, i - 1]

    P = numpy.zeros(zz.size, dtype=complex)
    Pz = numpy.zeros(zz.size, dtype=complex)
    Pm = numpy.zeros(zz.size, dtype=complex)

    for j in range(s + 1):
        aj = zp @ coeffs[:, j]
        P += aj * mp[:, j]

        ajp = dzp @ coeffs[:, j]
        Pz += ajp * mp[:, j]

        if j >= 1:
            Pm += (j * aj) * mp[:, j - 1]

    return P.reshape(shp), Pz.reshape(shp), Pm.reshape(shp)


# =================
# inverse stieltjes
# =================

def inverse_stieltjes(m_stack, delta_ladder, x=None, log=False,
                      nonnegative=True, **inv_stieltjes_opt):
    """
    Recover density from Stieltjes values sampled on a delta ladder.

    Parameters
    ----------

    m_stack : numpy.ndarray
        Complex-valued Stieltjes values sampled on a delta ladder.

        Accepted shapes are:

        * ``(n_levels, n_x)``
        * ``(n_levels, n_req, n_x)``

        where ``n_levels`` is the number of offsets in ``delta_ladder``.

    delta_ladder : array_like
        Positive imaginary offsets :math:`\\delta_j`.

    method : {'direct', 'richardson'}, default='direct'
        Density recovery method.

        * ``'direct'`` uses only the smallest offset:
          :math:`\\rho \\approx \\Im m(x + i\\delta_0)/\\pi`.
        * ``'richardson'`` uses polynomial extrapolation in :math:`\\delta`
          to estimate the :math:`\\delta \\to 0^+` limit.

    nonnegative : bool, default=True
        If `True`, clip the recovered density to be nonnegative.

    Returns
    -------

    rho : numpy.ndarray
        Recovered density. Shape is:

        * ``(n_x,)`` if ``m_stack`` has shape ``(n_levels, n_x)``
        * ``(n_req, n_x)`` if ``m_stack`` has shape ``(n_levels, n_req, n_x)``

    Notes
    -----

    **Richardosn method:**

    Neville table / polynomial extrapolation in delta, evaluated at 0.

    R[k, j] is the extrapolated value using points j, ..., j+k.
    Initialization: R[0, j] = G[j]

    Recurrence:
      R[k, j] =
          ((0 - delta[j]) * R[k-1, j+1]
           - (0 - delta[j+k]) * R[k-1, j]) / (delta[j+k] - delta[j])

    which simplifies to
      R[k, j] =
          (-delta[j]   * R[k-1, j+1]
           + delta[j+k] * R[k-1, j]) / (delta[j+k] - delta[j])

    This uses all levels and is exact whenever G is a polynomial
    in delta of degree <= n_levels - 1.
    """

    # Unpack options
    method = inv_stieltjes_opt.get('method', 'direct')
    reg = inv_stieltjes_opt.get('reg', 1e-10)
    fit_degree = inv_stieltjes_opt.get('fit_degree', 1)
    fit_weight = inv_stieltjes_opt.get('fit_weight', 'small_delta')

    m_stack = numpy.asarray(m_stack)
    delta_ladder = numpy.asarray(delta_ladder, dtype=float)

    if m_stack.ndim not in (2, 3):
        raise ValueError(
            '"m_stack" must have shape (n_levels, n_x) or '
            '(n_levels, n_req, n_x).')

    if delta_ladder.ndim != 1:
        raise ValueError('"delta_ladder" must be a 1D array.')

    if m_stack.shape[0] != delta_ladder.size:
        raise ValueError(
            'First axis of "m_stack" must match size of "delta_ladder".')

    if numpy.any(delta_ladder <= 0.0):
        raise ValueError('"delta_ladder" must be strictly positive.')

    n_levels = delta_ladder.size

    # With one ladder point, do not use Richardson
    if n_levels == 1:
        method = 'direct'

    # Use only the first (smallest) ladder levels
    delta = delta_ladder[:n_levels]
    G = m_stack[:n_levels].imag / numpy.pi

    if method == 'direct':
        rho = G[0]

    elif method == 'richardson':
        table = [G]

        for k in range(1, n_levels):
            prev = table[-1]
            curr = numpy.empty_like(prev[:-1])

            for j in range(n_levels - k):
                dj = delta[j]
                djk = delta[j + k]
                curr[j] = (djk * prev[j] - dj * prev[j + 1]) / (djk - dj)

            table.append(curr)

        rho = table[-1][0]

    elif method == 'polyfit':
        fit_degree = int(fit_degree)
        if fit_degree < 0 or fit_degree >= n_levels:
            raise ValueError(
                '"fit_degree" must be between 0 and n_levels - 1.')

        # Scale deltas for conditioning
        s = delta / delta[0]

        # Vandermonde matrix: [1, s, s^2, ...]
        A = numpy.vander(s, N=fit_degree + 1, increasing=True)

        # Optional weighting: emphasize smaller deltas since we want y -> 0
        if fit_weight == 'small_delta':
            # weight_j ~ 1 / s_j
            w = 1.0 / s
        elif fit_weight == 'uniform' or fit_weight is None:
            w = numpy.ones_like(s)
        else:
            raise ValueError('"fit_weight" must be "small_delta" or '
                             '"uniform".')

        sqrt_w = numpy.sqrt(w)
        Aw = sqrt_w[:, None] * A

        # Ridge penalty only on non-constant coefficients
        R = numpy.eye(fit_degree + 1, dtype=float)
        R[0, 0] = 0.0

        lhs = Aw.T @ Aw + float(reg) * R

        # Flatten all trailing dims of G and solve all x at once
        G2 = G.reshape(n_levels, -1)
        Gw = sqrt_w[:, None] * G2

        coef = numpy.linalg.solve(lhs, Aw.T @ Gw)

        # Intercept = estimate at delta = 0
        rho = coef[0].reshape(G.shape[1:])

    elif method == 'chebfit':
        fit_degree = int(fit_degree)
        if fit_degree < 0 or fit_degree >= n_levels:
            raise ValueError(
                '"fit_degree" must be between 0 and n_levels - 1.')

        # Scale deltas by the smallest one, so s[0] = 1
        s = delta / delta[0]

        # Map s from [0, s_max] to xi in [-1, 1]
        # This keeps the extrapolation target s = 0 at the boundary xi = -1.
        s_max = float(s[-1])
        if s_max <= 0.0:
            raise ValueError('Scaled delta range must be positive.')

        xi = (2.0 * s / s_max) - 1.0
        xi0 = -1.0

        # Chebyshev design matrix
        A = numpy.polynomial.chebyshev.chebvander(xi, fit_degree)

        # Optional weighting: emphasize smaller deltas since we want y -> 0
        if fit_weight == 'small_delta':
            w = 1.0 / s
        elif fit_weight == 'uniform' or fit_weight is None:
            w = numpy.ones_like(s)
        else:
            raise ValueError('"fit_weight" must be "small_delta" or '
                             '"uniform".')

        sqrt_w = numpy.sqrt(w)
        Aw = sqrt_w[:, None] * A

        # Flatten all trailing dims of G and solve all x at once
        G2 = G.reshape(n_levels, -1)
        Gw = sqrt_w[:, None] * G2

        # Ridge penalty only on non-constant coefficients. We use augmented
        # least squares instead of normal equations.
        if reg > 0.0 and fit_degree > 0:
            L = numpy.sqrt(float(reg)) * numpy.eye(fit_degree + 1,
                                                   dtype=float)
            L[0, 0] = 0.0

            A_aug = numpy.vstack((Aw, L))
            G_aug = numpy.vstack((
                Gw,
                numpy.zeros((fit_degree + 1, Gw.shape[1]), dtype=Gw.dtype)))

        else:
            A_aug = Aw
            G_aug = Gw

        coef, _, _, _ = numpy.linalg.lstsq(A_aug, G_aug, rcond=None)

        # Evaluate fitted Chebyshev series at delta = 0 (xi0 = -1)
        eval_row = numpy.polynomial.chebyshev.chebvander(
            numpy.array([xi0], dtype=float), fit_degree)[0]

        rho = (eval_row @ coef).reshape(G.shape[1:])

    elif method == 'poisson':

        if x is None:
            raise ValueError('"x" must be provided for method="poisson".')

        x = numpy.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError('"x" must be a 1D array.')

        if x.size != G.shape[-1]:
            raise ValueError('Last axis of "m_stack" must match size '
                             'of "x".')

        if x.size < 3:
            raise ValueError('"x" must have at least 3 points.')

        if log:
            if numpy.any(x <= 0.0):
                raise ValueError('"log=True" requires strictly positive x.')

            # Uniform grid in v = log x is required
            v = numpy.log(x)
            dv = numpy.diff(v)
            dv0 = float(numpy.mean(dv))
            if not numpy.allclose(dv, dv0, rtol=1e-6, atol=0.0):
                raise ValueError(
                    'For log-grid Poisson inversion, x must be geometric '
                    '(uniform in log x).')

            n_x = x.size

            # Flatten all batch dims except x
            batch_shape = G.shape[1:-1]
            n_batch = int(numpy.prod(batch_shape)) if batch_shape else 1
            G2 = G.reshape(n_levels, n_batch, n_x)

            # Zero-padding to reduce wrap-around
            nfft = 1
            while nfft < 4 * n_x:
                nfft *= 2

            # Grid for s = v - u
            s = (numpy.arange(nfft) - (nfft // 2)) * dv0

            numer = numpy.zeros((n_batch, nfft), dtype=complex)
            denom = numpy.zeros(nfft, dtype=float)

            for j in range(n_levels):
                dj = float(delta[j])

                ems = numpy.exp(-s)

                # Correct kernel for psi(v) = rho(exp(v)):
                # g(exp(v)) = (K_d * psi)(v)
                K = (dj / numpy.pi) * ems / ((1.0 - ems)**2 + dj * dj)

                # Put zero-lag at index 0 for FFT convolution
                K = numpy.fft.ifftshift(K) * dv0
                Khat = numpy.fft.fft(K)

                Gpad = numpy.zeros((n_batch, nfft), dtype=complex)
                Gpad[:, :n_x] = G2[j]
                Ghat = numpy.fft.fft(Gpad, axis=-1)

                numer += numpy.conjugate(Khat)[None, :] * Ghat
                denom += numpy.abs(Khat)**2

            # Regularized deconvolution in Fourier domain
            Psi_hat = numer / (denom[None, :] + float(reg))
            Psi = numpy.fft.ifft(Psi_hat, axis=-1).real[:, :n_x]

            # Psi(v) = rho(exp(v))
            rho2 = Psi
            rho = rho2.reshape(G.shape[1:])

        else:
            # Linear case require a uniform linear grid
            dx = numpy.diff(x)
            dx0 = float(numpy.mean(dx))
            if not numpy.allclose(dx, dx0, rtol=1e-6, atol=0.0):
                raise ValueError(
                    'For linear-grid Poisson inversion, x must be uniform '
                    '(linearly spaced).')

            n_x = x.size

            # Flatten all batch dims except x
            batch_shape = G.shape[1:-1]
            n_batch = int(numpy.prod(batch_shape)) if batch_shape else 1
            G2 = G.reshape(n_levels, n_batch, n_x)

            # Zero-padding to reduce circular wrap-around
            nfft = 1
            while nfft < 4 * n_x:
                nfft *= 2

            # Centered grid for kernel variable s = x - y
            s = (numpy.arange(nfft) - (nfft // 2)) * dx0

            numer = numpy.zeros((n_batch, nfft), dtype=complex)
            denom = numpy.zeros(nfft, dtype=float)

            for j in range(n_levels):
                dj = float(delta[j])

                # Poisson kernel in linear coordinates:
                # g(x) = (P_d * rho)(x),  P_d(s) = (d/pi)/(s^2 + d^2)
                K = (dj / numpy.pi) / (s * s + dj * dj)

                # Discrete convolution weight
                K = numpy.fft.ifftshift(K) * dx0
                Khat = numpy.fft.fft(K)

                Gpad = numpy.zeros((n_batch, nfft), dtype=complex)
                Gpad[:, :n_x] = G2[j]
                Ghat = numpy.fft.fft(Gpad, axis=-1)

                numer += numpy.conjugate(Khat)[None, :] * Ghat
                denom += numpy.abs(Khat)**2

            # Tikhonov-regularized least-squares deconvolution
            Rho_hat = numer / (denom[None, :] + float(reg))
            rho2 = numpy.fft.ifft(Rho_hat, axis=-1).real[:, :n_x]

            rho = rho2.reshape(G.shape[1:])

    else:
        raise ValueError('"method" is invalid.')

    if nonnegative:
        rho = numpy.maximum(rho, 0.0)

    return rho
