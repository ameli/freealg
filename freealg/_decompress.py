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
# from scipy.integrate import solve_ivp

__all__ = ['decompress', 'reverse_characteristics']


# =============
# secant method
# =============

def secant_complex(f, z0, z1, a=0+0j, tol=1e-12, max_iter=100, alpha=0.5,
                   max_bt=12, eps=1e-30, verbose=False):
    """
    Solves :math:``f(z) = a`` for many starting points simultaneously
    using the secant method in the complex plane.

    Parameters
    ----------
    f : callable
        Function that accepts and returns complex `ndarray`s.

    z0, z1 : array_like
        Two initial guesses. ``z1`` may be broadcast to ``z0``.

    a : complex or array_like, optional
        Right‑hand‑side targets (broadcasted to ``z0``). Defaults to ``0+0j``.

    tol : float, optional
        Convergence criterion on ``|f(z) - a|``. Defaults to ``1e-12``.

    max_iter : int, optional
        Maximum number of secant iterations. Defaults to ``100``.

    alpha : float, optional
        Back‑tracking shrink factor (``0 < alpha < 1``). Defaults to ``0.5``.

    max_bt : int, optional
        Maximum back‑tracking trials per iteration. Defaults to ``12``.

    eps : float, optional
        Safeguard added to tiny denominators. Defaults to ``1e-30``.

    verbose : bool, optional
        If *True*, prints progress every 10 iterations.

    Returns
    -------
    roots : ndarray
        Estimated roots, shaped like the broadcast inputs.
    residuals : ndarray
        Final residuals ``|f(root) - a|``.
    iterations : ndarray
        Iteration count for each point.
    """

    # Broadcast inputs
    z0, z1, a = numpy.broadcast_arrays(
        numpy.asarray(z0, numpy.complex128),
        numpy.asarray(z1, numpy.complex128),
        numpy.asarray(a,  numpy.complex128),
    )
    orig_shape = z0.shape
    z0, z1, a = (x.ravel() for x in (z0, z1, a))

    n_points = z0.size
    roots = z1.copy()
    iterations = numpy.zeros(n_points, dtype=int)

    f0 = f(z0) - a
    f1 = f(z1) - a
    residuals = numpy.abs(f1)
    converged = residuals < tol

    # Entering main loop
    for k in range(max_iter):
        active = ~converged
        if not active.any():
            break

        # Secant step
        denom = f1 - f0
        denom = numpy.where(numpy.abs(denom) < eps, denom + eps, denom)
        dz = (z1 - z0) * f1 / denom
        z2 = z1 - dz
        f2 = f(z2) - a

        # Line search by backtracking
        worse = (numpy.abs(f2) >= numpy.abs(f1)) & active
        if worse.any():
            shrink = numpy.ones_like(dz)
            for _ in range(max_bt):
                shrink[worse] *= alpha
                z_try = z1[worse] - shrink[worse] * dz[worse]
                f_try = f(z_try) - a[worse]

                improved = numpy.abs(f_try) < numpy.abs(f1[worse])
                if not improved.any():
                    continue

                idx = numpy.flatnonzero(worse)[improved]
                z2[idx], f2[idx] = z_try[improved], f_try[improved]
                worse[idx] = False
                if not worse.any():
                    break

        # Book‑keeping
        newly_conv = (numpy.abs(f2) < tol) & active
        converged[newly_conv] = True
        iterations[newly_conv] = k + 1
        roots[newly_conv] = z2[newly_conv]
        residuals[newly_conv] = numpy.abs(f2[newly_conv])

        still = active & ~newly_conv
        z0[still], z1[still] = z1[still], z2[still]
        f0[still], f1[still] = f1[still], f2[still]

        if verbose and k % 10 == 0:
            print(f"Iter {k}: {converged.sum()} / {n_points} converged")

    # Non‑converged points
    remaining = ~converged
    roots[remaining] = z1[remaining]
    residuals[remaining] = numpy.abs(f1[remaining])
    iterations[remaining] = max_iter

    return (
        roots.reshape(orig_shape),
        residuals.reshape(orig_shape),
        iterations.reshape(orig_shape),
    )


# ==========
# decompress
# ==========

def decompress(freeform, size, x=None, delta=1e-6, max_iter=500,
               tolerance=1e-12):
    """
    Free decompression of spectral density.

    Parameters
    ----------

    freeform : FreeForm
        The initial freeform object of matrix to be decompressed

    size : int
        Size of the decompressed matrix.

    x : numpy.array, default=None
        Positions where density to be evaluated at. If `None`, an interval
        slightly larger than the support interval will be used.

    delta: float, default=1e-4
        Size of the perturbation into the upper half plane for Plemelj's
        formula.

    max_iter: int, default=500
        Maximum number of secant method iterations.

    tolerance: float, default=1e-12
        Tolerance for the solution obtained by the secant method solver.

    Returns
    -------

    rho : numpy.array
        Spectral density

    See Also
    --------

    density
    stieltjes

    Notes
    -----

    Work in progress.

    References
    ----------

    .. [1] tbd

    Examples
    --------

    .. code-block:: python

        >>> from freealg import FreeForm
    """

    alpha = size / freeform.n
    m = freeform._eval_stieltjes
    # Lower and upper bound on new support
    hilb_lb = (1 / m(freeform.lam_m + delta * 1j)[1]).real
    hilb_ub = (1 / m(freeform.lam_p + delta * 1j)[1]).real
    lb = freeform.lam_m - (alpha - 1) * hilb_lb
    ub = freeform.lam_p - (alpha - 1) * hilb_ub

    # Create x if not given
    if x is None:
        radius = 0.5 * (ub - lb)
        center = 0.5 * (ub + lb)
        scale = 1.25
        x_min = numpy.floor(center - radius * scale)
        x_max = numpy.ceil(center + radius * scale)
        x = numpy.linspace(x_min, x_max, 500)

    # Ensure that input is an array
    x = numpy.asarray(x)
    target = x + delta * 1j
    if numpy.isclose(alpha, 1.0):
        return freeform.density(x), x, freeform.support

    # Characteristic curve map
    def _char_z(z):
        return z + (1 / m(z)[1]) * (1 - alpha)

    z0 = numpy.full(target.shape, numpy.mean(freeform.support) + delta*1j,
                    dtype=numpy.complex128)
    z1 = z0 - numpy.log(alpha) * 1j

    roots, _, _ = secant_complex(
        _char_z, z0, z1,
        a=target,
        tol=tolerance,
        max_iter=max_iter
    )

    # Plemelj's formula
    z = roots
    char_s = m(z)[1] / alpha
    rho = numpy.maximum(0, char_s.imag / numpy.pi)
    rho[numpy.isnan(rho) | numpy.isinf(rho)] = 0

    return rho.reshape(*x.shape), x, (lb, ub)


# =======================
# reverse characteristics
# =======================

def reverse_characteristics(freeform, z_inits, T, iterations=500,
                            step_size=0.1, tolerance=1e-8):
    """
    """

    t_span = (0, T)
    t_eval = numpy.linspace(t_span[0], t_span[1], 50)

    m = freeform._eval_stieltjes

    def _char_z(z, t):
        return z + (1 / m(z)[1]) * (1 - numpy.exp(t))

    target_z, target_t = numpy.meshgrid(z_inits, t_eval)

    z = numpy.full(target_z.shape, numpy.mean(freeform.support) - .1j,
                   dtype=numpy.complex128)

    # Broken Newton steps can produce a lot of warnings. Removing them for now.
    with numpy.errstate(all='ignore'):
        for _ in range(iterations):
            objective = _char_z(z, target_t) - target_z
            mask = numpy.abs(objective) >= tolerance
            if not numpy.any(mask):
                break
            z_m = z[mask]
            t_m = target_t[mask]

            # Perform finite difference approximation
            dfdz = _char_z(z_m+tolerance, t_m) - _char_z(z_m-tolerance, t_m)
            dfdz /= 2*tolerance
            dfdz[dfdz == 0] = 1.0

            # Perform Newton step
            z[mask] = z_m - step_size * objective[mask] / dfdz

    return z
