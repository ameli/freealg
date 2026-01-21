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
import matplotlib.pyplot as plt
import texplot

# Fallback to previous numpy API
if not hasattr(numpy, 'trapezoid'):
    numpy.trapezoid = numpy.trapz

__all__ = ['decompress', 'reverse_characteristics']


# ==========
# derivative
# ==========

def _derivative(f, fz, z, h, fd_order=4, vertical=False, return_second=False):
    """
    Compute first or first and second derivatives.

    Parameters
    ----------

    f : function
        Calling function

    fz : numpy.array
        Function values precomputed at the array z. This is used for second
        order derivative where the stencil also contains the middle point.
        Since this is already computed outside the function call, it is reused
        here, essentially making second-order derivative free.

    z : numpy.array
        A 1D array of complex points to evaluate derivative at.

    h : float or numpy.array
        Stencil size

    fd_order : {2, 4, 6}, default=2
        Order of central finite differencing scheme.

    vertical : bool, default=False
        It `True`, it uses vertical stencil (along y axis) instead of
        horizontal (along x axis).

    return_second : bool, default=False
        If `True`, returns both first and second derivatives.

    Returns
    -------

    df1 : numpy.array
        First derivative

    If ``return_second=True``:

        df2 : numpy.array
            Second derivative

    Notes
    -----

    Uses central finite difference.

    If the function is holomorphic, taking the derivative along horizontal or
    vertical directions should be identical (in theory), however, in practice,
    they might not be exactly the same. Note especially that ``vertical=True``
    is not suitable for points close to branch cut where the stencil points
    fall into two branches.
    """

    h = numpy.asarray(h, dtype=z.dtype)

    if vertical:
        h_ = 1j * h
    else:
        h_ = h

    # Stencil indices
    stencil = numpy.arange(-fd_order//2, fd_order//2 + 1).astype(float)

    # Stencil coefficients
    if fd_order == 2:
        coeff_df1 = [-0.5, +0.0, +0.5]
        coeff_df2 = [+1.0, -2.0, +1.0]

    elif fd_order == 4:
        coeff_df1 = [+1.0 / 12.0, -2.0 / 3.0, +0.0, +2.0 / 3.0, -1.0 / 12.0]
        coeff_df2 = [-1.0 / 12.0, +16.0 / 12.0, -30.0 / 12.0, +16.0 / 12.0,
                     -1.0 / 12.0]

    elif fd_order == 6:
        coeff_df1 = [-1.0 / 60.0, +3.0 / 20.0, -3.0 / 4.0, +0.0, +3.0 / 4.0,
                     -3.0 / 20.0, +1.0 / 60.0]
        coeff_df2 = [+1.0 / 90, -3.0 / 20.0, +3.0 / 2.0, -49.0 / 18.0,
                     +3.0 / 2.0, -3.0 / 20.0, +1.0 / 90.0]

    else:
        raise NotImplementedError('"fd_order" is not valid.')

    # Function values at stencil points. Precomputed to avoid redundancy when
    # both first and second derivatives are needed.
    val = [None] * stencil.size
    for i in range(stencil.size):

        # The center stencil for first derivative is zero, so we do not compute
        # the function value, unless second derivative is needed.
        if coeff_df1[i] == 0.0:
            # This is already computed outside this function, for free. This
            # is only used for second derivative where the stencil has non-zero
            # coefficient in the middle point. This is not used in the first
            # derivative where the coefficient of the middle point is zero.
            val[i] = fz
        else:
            val[i] = f(z + stencil[i] * h_)

    # First derivative
    df1 = numpy.zeros_like(z)
    for i in range(stencil.size):

        # Skip the center of stencil where the coeff is zero.
        if coeff_df1[i] != +0.0:
            df1 += coeff_df1[i] * val[i]
    df1 /= h_

    # Second derivative
    if return_second:
        df2 = numpy.zeros_like(z)
        for i in range(stencil.size):
            df2 += coeff_df2[i] * val[i]
        df2 /= h_**2

    if return_second:
        return df1, df2
    else:
        return df1


# ===================
# adaptive derivative
# ===================

def _adaptive_derivative(f, fz, z, h, err_low=1e-4, err_high=1e-2, factor=2.0,
                         vertical=False, return_second=False):
    """
    Compute first or first and second derivatives using adaptive refinement of
    stencil size.

    Parameters
    ----------

    f : function
        Calling function

    fz : numpy.array
        Function values precomputed at the array z. This is used for second
        order derivative where the stencil also contains the middle point.
        Since this is already computed outside the function call, it is reused
        here, essentially making second-order derivative free.

    z : numpy.array
        A 1D array of complex points to evaluate derivative at.

    h : float or numpy.array
        Stencil size

    err_low : float, default=1e-4
        Threshold criteria to increase stencil size h. Namely, the relative
        error of Richardson test below this threshold is considered as
        being caused by having too small stencil size, and hence stencil size
        h has to be increased.

    err_high : float, default=1e-2
        Threshold criteria to decrease stencil size h. Namely, the relative
        error of Richardson test above this threshold is considered as
        being caused by having too large stencil size, and hence stencil size
        h has to be decreased.

    factor : float, default=2.0
        Factor to increase or decrease stencil size.

    vertical : bool, default=False
        It `True`, it uses vertical stencil (along y axis) instead of
        horizontal (along x axis).

    return_second : bool, default=False
        If `True`, returns both first and second derivatives.

    Returns
    -------

    h : numpy.array
        Updated stencil size

    df1 : numpy.array
        First derivative

    If ``return_second=True``:

        df2 : numpy.array
            Second derivative

    Notes
    -----

    Uses central finite difference. This function uses finite difference of
    order 2 or 4 as follows.

    First, function f is evaluated on 5 points:

        [f(z-2h), f(z-h), f(z), f(z+h), f(z+2h)]

    This is used for either a:

    1. 2-nd order finite difference with the stencil of size h
    2. 2-nd order finite difference with the stencil of size 2h
    3. 4-th order finite difference with the stencil of size h

    To determine which of these should be used, the first derivative for the
    cases 1 and 2 above are compared using the Richardson test:

    1. If the Richardson test determines the stencil is too small, the output
       derivative is a second order finite difference on stencil of size 2h.
       Also, h is grown by factor of two for the next iteration.

    2. If the Richardson test determines the stencil is good (not too small or
       not too large), the output derivative is a fourth order finite
       difference on stencil of size h. Also, h is not changed for the next
       iteration.

    3. If the Richardson test determines the stencil is too large, the output
       derivative is a second order finite difference on stencil of size h.
       Also, h is shrunken by factor of two for the next iteration.

    If the function is holomorphic, taking the derivative along horizontal or
    vertical directions should be identical (in theory), however, in practice,
    they might not be exactly the same. Note especially that ``vertical=True``
    is not suitable for points close to branch cut where the stencil points
    fall into two branches.
    """

    h = h.copy().astype(z.dtype)

    if vertical:
        h_ = 1j * h
    else:
        h_ = h

    # Center index of stencil of order 4
    st_center = 2

    # Stencil indices of order 2 of size h
    st_ord2_h = numpy.array([-1, 0, 1]) + st_center

    # Stencil indices of order 2 of size 2h
    st_ord2_2h = numpy.array([-2, 0, 2]) + st_center

    # Stencil indices of order 4 of size h
    st_ord4_h = numpy.array([-2, -1, 0, 1, 2]) + st_center

    # Stencil coefficients for first derivative, order 2
    coeff_df1_ord2 = [-0.5, +0.0, +0.5]

    # Stencil coefficients for second derivative, order 2
    coeff_df2_ord2 = [+1.0, -2.0, +1.0]

    # Stencil coefficients for first derivative, order 4
    coeff_df1_ord4 = [+1.0 / 12.0, -2.0 / 3.0, 0.0, +2.0 / 3.0, -1.0 / 12.0]

    # Stencil coefficients for second derivative, order 4
    coeff_df2_ord4 = [-1.0 / 12.0, +16.0 / 12.0, -30.0 / 12.0, +16.0 / 12.0,
                      -1.0 / 12.0]

    # Function values at stencil points of order 4. Precomputed to avoid
    # redundancy when both first and second derivatives are needed.
    f_val = [None] * st_ord4_h.size
    for i in range(st_ord4_h.size):

        if coeff_df1_ord4[i] == 0.0:
            # This is already computed outside this function, for free. This
            # is only used for second derivative where the stencil has non-zero
            # coefficient in the middle point. This is not used in the first
            # derivative where the coefficient of the middle point is zero.
            f_val[i] = fz
        else:
            f_val[i] = f(z + st_ord4_h[i] * h_)

    # First derivative, fd order 2, stencil of size h
    df1_ord2_h = numpy.zeros_like(z)
    for i in range(st_ord2_h.size):
        # Skip the center of stencil where the coeff is zero.
        if coeff_df1_ord2[i] != 0.0:
            df1_ord2_h += coeff_df1_ord2[i] * f_val[st_ord2_h[i]]
    df1_ord2_h /= h_

    # First derivative, fd order 2, stencil of size 2h
    df1_ord2_2h = numpy.zeros_like(z)
    for i in range(st_ord2_2h.size):
        # Skip the center of stencil where the coeff is zero.
        if coeff_df1_ord2[i] != 0.0:
            df1_ord2_2h += coeff_df1_ord2[i] * f_val[st_ord2_2h[i]]
    df1_ord2_2h /= (2.0 * h_)

    # First derivative, fd order 4, stencil of size h
    df1_ord4_h = numpy.zeros_like(z)
    for i in range(st_ord4_h.size):
        # Skip the center of stencil where the coeff is zero.
        if coeff_df1_ord4[i] != 0.0:
            df1_ord4_h += coeff_df1_ord4[i] * f_val[st_ord4_h[i]]
    df1_ord4_h /= h_

    if return_second:
        # Second derivative, fd order 2, stencil of size h
        df2_ord2_h = numpy.zeros_like(z)
        for i in range(st_ord2_h.size):
            df2_ord2_h += coeff_df2_ord2[i] * f_val[st_ord2_h[i]]
        df2_ord2_h /= h_**2

        # Second derivative, fd order 2, stencil of size 2h
        df2_ord2_2h = numpy.zeros_like(z)
        for i in range(st_ord2_2h.size):
            df2_ord2_2h += coeff_df2_ord2[i] * f_val[st_ord2_2h[i]]
        df2_ord2_2h /= (2.0 * h_)**2

        # Second derivative, fd order 4, stencil of size h
        df2_ord4_h = numpy.zeros_like(z)
        for i in range(st_ord4_h.size):
            df2_ord4_h += coeff_df2_ord4[i] * f_val[st_ord4_h[i]]
        df2_ord4_h /= h_**2

    # Richardson test
    fd_order = 2
    p = 2**fd_order - 1.0
    abs_err = numpy.abs(df1_ord2_2h - df1_ord2_h) / p
    rel_err = abs_err / numpy.maximum(1.0, numpy.abs(df1_ord2_h))

    # Grow and Shrink limits criteria on relative error
    grow = rel_err < err_low
    shrink = rel_err > err_high

    # Output stencil size
    h[grow] *= factor
    h[shrink] /= factor

    # Output first derivative
    df1 = df1_ord4_h
    df1[grow] = df1_ord2_2h[grow]
    df1[shrink] = df1_ord2_h[shrink]

    # Output second derivative
    if return_second:
        df2 = df2_ord4_h
        df2[grow] = df2_ord2_2h[grow]
        df2[shrink] = df2_ord2_h[shrink]

    if return_second:
        return h, df1, df2
    else:
        return h, df1


# =============
# newton method
# =============

def _newton_method(f, z_init, a, support, enforce_wall=False, tol=1e-4,
                   step_size=0.1, max_iter=500, adaptive_stencil=True,
                   halley=False, vertical=False):
    """
    Solves :math:``f(z) = a`` for many starting points simultaneously using the
    damped Newton method.

    Parameters
    ----------

    f : function
        Caller function.

    z_init : numpy.array
        Initial guess of roots

    a : numpy.array
        Target value of f in the root finding problem f(z) - a = 0.

    support : tuple
        The support of the density. This is needed to enforce no crossing
        through branch cut as wall.

    enforce_wall : bool, default=False
        If `True`, roots are not allowed to cross branch cut. It is recommended
        to enable this feature when the initial guess z_init is above the real
        line. When z_init points are below the real line, this feature is
        effectively not used.

    tol : float, default=1e-4
        Tolerance of terminating the iteration when |f(z) - a| < tol.

    step_size : float, default=0.1
        The step size of Newton (or Halley) iterations, between 0 and 1 where
        1 corresponds to full Newton (or Halley) step. Note that these methods
        are generally very sensitive to the step size. Values between 0.05 and
        0.2 are recommended.

    max_ier : int, default=500
        Maximum number of iterations.

    adaptive_stencil : bool, default=True
        If `True`, the stencil size in finite differencing is adaptively
        selected based on Richardson extrapolation error. If `False`, a fixed
        stencil size of all points are used.

    halley : bool, default=False
        If `True`, Halley method is used instead of Newton method, but only
        when the Newton makes very small increments. Halley method is in
        general not suitable for the most of the trajectory, and can only be
        useful at the end of trajectory for convergence. It is recommended to
        turn this off.

    vertical : bool, default=False
        If `True`, to compute the derivative of holomorphic function, a
        vertical stencil (instead of horizontal) is used. This is not suitable
        for points near branch cut where some points of the stencil might fall
        intro two sides of the branch cut, but it might be suitable for points
        inside the support.

    Returns
    -------

    roots: numpy.array
        Solutions z of f(z) - a = 0.

    residuals: numpy.array
        The residuals f(z) - a

    iterations : numpy.array
        Number of iterations used for each root.
    """

    # Finite difference order
    fd_order = 4

    epsilon = numpy.sqrt(numpy.finfo(float).eps)
    h_base = epsilon**(1.0 / (fd_order + 1.0))

    # Global variables
    root = z_init.copy()
    mask = numpy.ones(z_init.shape, dtype=bool)
    f_val = numpy.zeros(z_init.shape, dtype=z_init.dtype)
    residual = numpy.ones(z_init.shape, dtype=z_init.dtype) * numpy.inf
    iterations = numpy.zeros(z_init.shape, dtype=int)

    # Initialize stencil size
    h = h_base * numpy.maximum(1.0, numpy.abs(z_init))

    # Main loop
    for _ in range(max_iter):

        if not numpy.any(mask):
            # No more active point left
            break

        # Update iterations
        iterations += mask.astype(int)

        # Mask variables using the previous mask (dash m are masked variables)
        a_m = a[mask]
        z_m = root[mask]
        f_m = f(z_m)
        f_val[mask] = f_m
        residual_m = f_m - a_m
        residual[mask] = residual_m

        # Update mask
        mask = numpy.abs(residual) >= tol

        # Mask variables again using the new mask
        a_m = a[mask]
        z_m = root[mask]
        f_m = f_val[mask]
        residual_m = residual[mask]
        h_m = h[mask]

        if adaptive_stencil:
            # Use adaptive stencil size from previous update
            h_m = h[mask]

            # Adaptive stencil size, finite difference order is fixed 2 and 4.
            h_m, df1_m, df2_m = _adaptive_derivative(f, f_m, z_m, h_m,
                                                     vertical=vertical,
                                                     return_second=True)

            # Clip too small or too large stencils
            h_m = numpy.clip(h_m, epsilon, h_base * numpy.abs(z_m))

            # Update global stencil sizes
            h[mask] = h_m

        else:
            # Fixed stencil size for all points
            h_m = h_base * numpy.maximum(1.0, numpy.abs(z_m))

            # Fixed stencil size, but finite difference order can be set.
            fd_order = 4  # can be 2, 4, 6
            df1_m, df2_m = _derivative(f, f_m, z_m, h_m, fd_order=fd_order,
                                       vertical=False, return_second=True)

        # Handling second order zeros
        df1_m[numpy.abs(df1_m) < 1e-12 * numpy.abs(z_m)] = 1e-12

        # Newton and Halley steps
        df0_m = residual_m
        newton_step = df0_m / df1_m

        if halley:
            halley_step = (df0_m * df1_m) / (df1_m**2 - 0.5 * df0_m * df2_m)

            # Criteria on where to use Halley instead of Newton
            use_halley = numpy.abs(newton_step) < 1e-2 * numpy.abs(z_m)
            step = numpy.where(use_halley, halley_step, newton_step)
        else:
            step = newton_step

        # Force new point to not cross branch cut. It has to go from upper to
        # lower half plane only through the support interval.
        if enforce_wall:

            root_m_old = root[mask]
            root_m_new = z_m - step_size * step
            lam_m, lam_p = support

            x0, y0 = root_m_old.real, root_m_old.imag
            x1, y1 = root_m_new.real, root_m_new.imag

            # Find which points are crossing the branch cut (either downward
            # or upward)
            crossed = \
                (y0 * y1 <= 0.0) & \
                ((x0 <= lam_m) | (x0 >= lam_p)) & \
                ((x1 <= lam_m) | (x1 >= lam_p))

            # Remove imaginary component from step
            step[crossed] = step[crossed].real + 0.0j

        # Update root
        root[mask] = z_m - step_size * step

    # Residuals at the final points
    residual = f(root) - a

    return root, residual, iterations


# =============
# secant method
# =============

def _secant_complex(f, z0, z1, a=0+0j, tol=1e-12, max_iter=100,
                    alpha=0.5, max_bt=1, eps=1e-30, step_factor=5.0,
                    post_smooth=True, jump_tol=10.0, dtype=numpy.complex128,
                    verbose=False):
    """
    Solves :math:``f(z) = a`` for many starting points simultaneously using the
    secant method in the complex plane.

    Parameters
    ----------
    f : callable
        Function that accepts and returns complex `ndarray`s.

    z0, z1 : array_like
        Two initial guesses. ``z1`` may be broadcast to ``z0``.

    a : complex or array_like, optional
        Right-hand-side targets (broadcasted to ``z0``). Defaults to ``0+0j``.

    tol : float, optional
        Convergence criterion on ``|f(z) - a|``. Defaults to ``1e-12``.

    max_iter : int, optional
        Maximum number of secant iterations. Defaults to ``100``.

    alpha : float, optional
        Back-tracking shrink factor (``0 < alpha < 1``). Defaults to ``0.5``.

    max_bt : int, optional
        Maximum back-tracking trials per iteration. Defaults to ``0``.

    eps : float, optional
        Safeguard added to tiny denominators. Defaults to ``1e-30``.

    post_smooth : bool, optional
        If True (default) run a single vectorised clean-up pass that
        re-solves points whose final root differs from the *nearest*
        neighbour by more than ``jump_tol`` times the local median jump.

    jump_tol : float, optional
        Sensitivity of the clean-up pass; larger tolerance implies fewer
        re-solves.

    dtype : {``'complex128'``, ``'complex256'``}, default = ``'complex128'``
        Data type for inner computations of complex variables.

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
        numpy.asarray(z0, dtype=dtype),
        numpy.asarray(z1, dtype=dtype),
        numpy.asarray(a, dtype=dtype),
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

        # Step-size limiter
        prev_step = numpy.maximum(numpy.abs(z1 - z0), eps)
        max_step = step_factor * prev_step
        big = numpy.abs(dz) > max_step
        dz[big] *= max_step[big] / numpy.abs(dz[big])

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

        # Book-keeping
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

    # Non-converged points
    remaining = ~converged
    roots[remaining] = z1[remaining]
    residuals[remaining] = numpy.abs(f1[remaining])
    iterations[remaining] = max_iter

    # Optional clean-up pass
    if post_smooth and n_points > 2:
        # absolute jump to *nearest* neighbour (left or right)
        diff_left = numpy.empty_like(roots)
        diff_right = numpy.empty_like(roots)
        diff_left[1:] = numpy.abs(roots[1:] - roots[:-1])
        diff_right[:-1] = numpy.abs(roots[:-1] - roots[1:])
        jump = numpy.minimum(diff_left, diff_right)

        # ignore non-converged points
        median_jump = numpy.median(jump[~remaining])
        bad = (jump > jump_tol * median_jump) & ~remaining

        if bad.any():
            z_first_all = numpy.where(bad & (diff_left <= diff_right),
                                      roots - diff_left,
                                      roots + diff_right)

            # keep only the offending indices
            z_first = z_first_all[bad]
            z_second = z_first + (roots[bad] - z_first) * 1e-2

            # re-solve just the outliers in one vector call
            new_root, new_res, new_iter = _secant_complex(
                f, z_first, z_second, a[bad], tol=tol, max_iter=max_iter,
                alpha=alpha, max_bt=max_bt, eps=eps, step_factor=step_factor,
                dtype=dtype, post_smooth=False,  # avoid recursion
            )

            roots[bad] = new_root
            residuals[bad] = new_res
            iterations[bad] = iterations[bad] + new_iter

            if verbose:
                print(f"Clean-up: re-solved {bad.sum()} outliers")

    return (
        roots.reshape(orig_shape),
        residuals.reshape(orig_shape),
        iterations.reshape(orig_shape),
    )


# ================
# plot diagnostics
# ================

def _plot_diagnostics(freeform, x, roots, residuals, iterations, tolerance,
                      max_iter):
    """
    Plot the results of root-findings, including the residuals, number of
    iterations, and the imaginary part of the roots.
    """

    with texplot.theme(use_latex=False):
        fig, ax = plt.subplots(ncols=3, figsize=(9.5, 3))

        for i in range(ax.size):
            ax[i].axvline(x=freeform.lam_m, color='silver', linestyle=':')
            ax[i].axvline(x=freeform.lam_p, color='silver', linestyle=':')

        ax[0].axhline(y=tolerance, color='silver', linestyle='--',
                      label='tolerance')
        ax[0].plot(x, numpy.abs(residuals), color='black')

        ax[1].axhline(y=max_iter, color='silver', linestyle='--',
                      label='max iter')
        ax[1].plot(x, iterations, color='black')

        ax[2].axhline(y=0, color='silver', linestyle='-', alpha=0.5)
        ax[2].plot(x, roots.imag, color='black')

        ax[0].set_yscale('log')
        ax[0].set_xlim([x[0], x[-1]])
        ax[0].set_title('Residuals')
        ax[0].set_xlabel(r'$x$')
        ax[0].set_ylabel(r'$| f(z_0) - z |$')
        ax[0].legend(fontsize='xx-small')

        ax[1].set_xlim([x[0], x[-1]])
        ax[1].set_title('Iterations')
        ax[1].set_xlabel(r'$x$')
        ax[1].set_ylabel(r'$n$')
        ax[1].legend(fontsize='xx-small')

        ax[2].set_xlim([x[0], x[-1]])
        ax[2].set_title('Roots Imaginary Part')
        ax[2].set_xlabel(r'$x$')
        ax[2].set_ylabel(r'Im$(z_0)$')
        ax[2].set_yscale('symlog', linthresh=1e-5)

        plt.tight_layout()
        plt.show()


# ==========
# decompress
# ==========

def decompress(freeform, alpha, x, roots_init=None, method='newton',
               delta=1e-4, max_iter=500, step_size=0.1, tolerance=1e-4,
               plot_diagnostics=False):
    """
    Free decompression of spectral density.

    Parameters
    ----------

    freeform : FreeForm
        The initial freeform object of matrix to be decompressed

    alpha : float
        Decompression ratio :math:`\\alpha = n / n_s = e^{t}`.

    x : numpy.array
        Positions where density to be evaluated at.

    roots_init : numpy.array, default=None
        Initial guess for roots. If `None`, all root points are allocated at
        a point below the center of support. If given, this is usually the
        root that is found in the previous iteration of the called function.

    method : {``'newton'``, ``'secant'``}, default=``'newton'``
        Root-finding method.

    delta: float, default=1e-4
        Size of the perturbation into the upper half plane for Plemelj's
        formula.

    max_iter: int, default=500
        Maximum number of iterations of the chosen method.

    step_size: float, default=0.1
        Step size for Newton iterations.

    tolerance: float, default=1e-4
        Tolerance for the solution obtained by the secant method solver.

    plot_diagnostics : bool, default=False
        Plots diagnostics including convergence and number of iterations of
        root finding method.

    Returns
    -------

    rho : numpy.array
        Spectral density

    See Also
    --------

    density
    stieltjes

    Examples
    --------

    .. code-block:: python

        >>> from freealg import FreeForm
    """

    # Locations where Stieltjes is sought to be found at.
    target = x + delta * 1j

    if numpy.isclose(alpha, 1.0):
        return freeform.density(x), roots_init

    # Function that returns the second branch of Stieltjes
    m = freeform._eval_stieltjes

    # ------
    # char z
    # ------

    def _char_z(z_0):
        """
        Characteristic curve map. Returns z from initial z_0.
        """

        return z_0 + (1.0 / m(z_0)) * (1.0 - alpha)

    # -----

    # Initialize roots below the real axis
    if roots_init is None:
        roots_init = numpy.full(x.shape, numpy.mean(freeform.support) - 0.1j,
                                dtype=freeform.dtype)

    # Finding roots
    if method == 'newton':
        support = (freeform.lam_m, freeform.lam_p)

        # Using initial points below the real line.
        roots, residuals, iterations = \
            _newton_method(_char_z, roots_init, target, support,
                           enforce_wall=False, tol=tolerance,
                           step_size=step_size, max_iter=max_iter,
                           adaptive_stencil=True, halley=False, vertical=False)

        # Using target points themselves as initial points. Since these points
        # are above the real line, enforce branchcut as wall
        # roots_init_1 = target
        # roots_1, residuals_1, iterations_1 = \
        #     _newton_method(_char_z, roots_init_1, target, support,
        #                    enforce_wall=True, tol=tolerance,
        #                    step_size=step_size, max_iter=max_iter,
        #                    adaptive_stencil=True, halley=False,
        #                    vertical=True)
        #
        # # If any result from residuals1 is better than residuals, use them
        # good = numpy.abs(residuals_1) < numpy.abs(residuals)
        # roots[good] = roots_1[good]
        # residuals[good] = residuals_1[good]
        # iterations[good] = iterations_1[good]

    elif method == 'secant':
        z0 = numpy.full(x.shape, numpy.mean(freeform.support) + 0.1j,
                        dtype=freeform.dtype)
        z1 = z0 - 0.2j

        roots, _, _ = _secant_complex(_char_z, z0, z1, a=target, tol=tolerance,
                                      max_iter=max_iter, dtype=freeform.dtype)
    else:
        raise NotImplementedError('"method" is invalid.')

    # Plemelj's formula
    z = roots
    char_s = numpy.squeeze(m(z)) / alpha
    rho = numpy.maximum(0, char_s.imag / numpy.pi)

    # Check any nans are in the roots
    num_nan = numpy.sum(numpy.isnan(roots))
    if num_nan > 0:
        raise RuntimeWarning(f'"nan" roots detected: num: {num_nan}.')

    # dx = x[1] - x[0]
    # left_idx, right_idx = support_from_density(dx, rho)
    # x, rho = x[left_idx-1:right_idx+1], rho[left_idx-1:right_idx+1]
    rho = rho / numpy.trapezoid(rho, x)

    if plot_diagnostics:
        _plot_diagnostics(freeform, x, roots, residuals, iterations, tolerance,
                          max_iter)

    return rho, roots


# =======================
# reverse characteristics
# =======================

def reverse_characteristics(freeform, z_inits, T, iterations=500,
                            step_size=0.1, tolerance=1e-8,
                            dtype=numpy.complex128):
    """
    """

    t_span = (0, T)
    t_eval = numpy.linspace(t_span[0], t_span[1], 50)

    m = freeform._eval_stieltjes

    def _char_z(z, t):
        return z + (1 / m(z)) * (1 - numpy.exp(t))

    target_z, target_t = numpy.meshgrid(z_inits, t_eval)

    z = numpy.full(target_z.shape, numpy.mean(freeform.support) - 0.1j,
                   dtype=dtype)

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
