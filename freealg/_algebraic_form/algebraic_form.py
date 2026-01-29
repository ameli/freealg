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
from .._util import resolve_complex_dtype, compute_eig
# from .._util import compute_eig
from ._continuation_algebraic import sample_z_joukowski, \
        filter_z_away_from_cuts, fit_polynomial_relation, \
        sanity_check_stieltjes_branch, eval_P
from ._edge import evolve_edges, merge_edges
from ._cusp_wrap import cusp_wrap

# Decompress with Newton
# from ._decompress import build_time_grid, decompress_newton
from ._decompress_util import build_time_grid
# from ._decompress4 import decompress_newton # WORKS (mass issue)
# from ._decompress5 import build_time_grid, decompress_newton
# from ._decompress6 import build_time_grid, decompress_newton
# from ._decompress4_2 import build_time_grid, decompress_newton
# from ._decompress_new_2 import build_time_grid, decompress_newton
# from ._decompress_new import build_time_grid, decompress_newton
# from ._decompress6 import decompress_newton
# from ._decompress7 import decompress_newton
# from ._decompress8 import decompress_newton
from ._decompress9 import decompress_newton  # With Predictor/Corrector

# Decompress with coefficients
from ._decompress2 import decompress_coeffs, plot_candidates

# Homotopy
# from ._homotopy import StieltjesPoly
# from ._homotopy2 import StieltjesPoly
# from ._homotopy3 import StieltjesPoly  # Viterbi
# from ._homotopy4 import StieltjesPoly
from ._homotopy5 import StieltjesPoly

from ._branch_points import compute_branch_points
from ._support import compute_support
from ._moments import Moments, AlgebraicStieltjesMoments
from .._free_form._support import supp
from .._free_form._plot_util import plot_density, plot_hilbert, plot_stieltjes

# Fallback to previous numpy API
if not hasattr(numpy, 'trapezoid'):
    numpy.trapezoid = numpy.trapz

__all__ = ['AlgebraicForm']


# ==============
# Algebraic Form
# ==============

class AlgebraicForm(object):
    """
    Algebraic surrogate for ensemble models.

    Parameters
    ----------

    A : numpy.ndarray
        The 2D symmetric :math:`\\mathbf{A}`. The eigenvalues of this will be
        computed upon calling this class. If a 1D array provided, it is
        assumed to be the eigenvalues of :math:`\\mathbf{A}`.

    support : tuple, default=None
        The support of the density of :math:`\\mathbf{A}`. If `None`, it is
        estimated from the minimum and maximum of the eigenvalues.

    delta: float, default=1e-6
        Size of perturbations into the upper half plane for Plemelj's
        formula.

    dtype : {``'complex128'``, ``'complex256'``}, default = ``'complex128'``
        Data type for inner computations of complex variables:

        * ``'complex128'``: 128-bit complex numbers, equivalent of two double
          precision floating point.
        * ``'complex256'``: 256-bit complex numbers, equivalent of two long
          double precision floating point. This option is only available on
          Linux machines.

        When using series acceleration methods (such as setting
        ``continuation`` in :func:`fit` function to ``wynn-eps``), setting a
        higher precision floating point arithmetics might improve conference.

    **kwargs : dict, optional
        Parameters for the :func:`supp` function can also be prescribed
        here when ``support=None``.

    Attributes
    ----------

    eig : numpy.array
        Eigenvalues of the matrix

    support: tuple
        The predicted (or given) support :math:`(\\lambda_{\\min},
        \\lambda_{\\max})` of the eigenvalue density.

    n : int
      Initial array size (assuming a square matrix when :math:`\\mathbf{A}` is
      2D).

    Methods
    -------

    fit
        Fit the Jacobi polynomials to the empirical density.

    density
        Compute the spectral density of the matrix.

    hilbert
        Compute Hilbert transform of the spectral density

    stieltjes
        Compute Stieltjes transform of the spectral density

    decompress
        Free decompression of spectral density

    eigvalsh
        Estimate the eigenvalues

    cond
        Estimate the condition number

    trace
        Estimate the trace of a matrix power

    slogdet
        Estimate the sign and logarithm of the determinant

    norm
        Estimate the Schatten norm

    Examples
    --------

    .. code-block:: python

        >>> from freealg import FreeForm
    """

    # ====
    # init
    # ====

    def __init__(self, A, support=None, delta=1e-5, dtype='complex128',
                 **kwargs):
        """
        Initialization.
        """

        self.A = None
        self.eig = None
        self._stieltjes = None
        self._moments = None
        self.support = support
        self.est_support = None  # Estimated from polynmial after fitting
        self.delta = delta    # Offset above real axis to apply Plemelj formula

        # Data type for complex arrays
        self.dtype = resolve_complex_dtype(dtype)

        if hasattr(A, 'stieltjes') and callable(getattr(A, 'stieltjes', None)):
            # This is one of the distribution objects, like MarchenkoPastur
            self._stieltjes = A.stieltjes
            self.support = A.support()
            self.n = 1

        elif callable(A):
            # This is a custom function
            self._stieltjes = A
            self.n = 1

        else:
            # Eigenvalues
            if A.ndim == 1:
                # If A is a 1D array, it is assumed A is the eigenvalues array.
                self.eig = A
                self.n = len(A)
            elif A.ndim == 2:
                # When A is a 2D array, it is assumed A is the actual array,
                # and its eigenvalues will be computed.
                self.A = A
                self.n = A.shape[0]
                assert A.shape[0] == A.shape[1], \
                    'Only square matrices are permitted.'
                self.eig = compute_eig(A)

            # Use empirical Stieltjes function
            self._stieltjes = lambda z: \
                numpy.mean(1.0/(self.eig-z[:, numpy.newaxis]), axis=-1)
            self._moments = Moments(self.eig)  # NOTE (never used)

        # broad support
        if self.support is None:
            if self.eig is None:
                raise RuntimeError("Support must be provided without data")

            # Detect support
            self.lam_m, self.lam_p = supp(self.eig, **kwargs)
            self.broad_support = (float(self.lam_m), float(self.lam_p))
        else:
            self.lam_m = float(min([s[0] for s in self.support]))
            self.lam_p = float(max([s[1] for s in self.support]))
            self.broad_support = (self.lam_m, self.lam_p)

        # Initialize
        self.a_coeffs = None               # Polynomial coefficients
        self.status = None                 # Fitting status
        self.cache = {}                    # Cache inner-computations

    # ===
    # fit
    # ===

    def fit(self, deg_m, deg_z, reg=0.0,
            r=[1.25, 6.0, 20.0],
            n_r=[3, 2, 1],
            n_samples=4096,
            y_eps=2e-2,
            x_pad=0.0,
            triangular=None,
            mu=None,
            mu_reg=None,
            normalize=False,
            verbose=False):
        """
        Fit polynomial.

        Parameters
        ----------

        deg_m : int
            Degree :math:`\\mathrm{deg}_m(P)`

        deg_z : int
            Degree :math:`\\mathrm{deg}_z(P)`

        mu : array_like, default=None
            If an array :math:`[\\mu_0, \\mu_1`, \\dots, \\mu_r]` is given,
            it enforces the first :math:`r+1` moments. Note that :math:`\\mu_0`
            should be :math:`1` to ensure unit mass. See also ``mu_reg``.

        mu_reg: float, default=None
            If `None`, the constraints ``mu`` are applied as hard constraint.
            If a positive number, the constraints are applied as a soft
            constraints with regularisation ``mu_reg``.

        Notes
        -----

        When the input data are from an exact model, hard moment constraint is
        preferred over soft constraint as the latter can hurt an already a good
        fit.
        """

        # Very important: reset cache whenever this function is called. This
        # also empties all references holding a cache copy.
        # self.cache.clear()

        z_fits = []

        # Sampling around support, or broad_support. This is only needed to
        # ensure sampled points are not hiting the support itself is not used
        # in any computation. If support is not known, use broad support.
        if self.support is not None:
            possible_support = self.support
        else:
            possible_support = [self.broad_support]

        for sup in possible_support:
            a, b = sup

            for i in range(len(r)):
                z_fits.append(sample_z_joukowski(a, b, n_samples=n_samples,
                                                 r=r[i], n_r=n_r[i]))

        z_fit = numpy.concatenate(z_fits)

        # Remove points too close to any cut
        z_fit = filter_z_away_from_cuts(z_fit, possible_support, y_eps=y_eps,
                                        x_pad=x_pad)

        # Fitting (w_inf = None means adaptive weight selection)
        m1_fit = self._stieltjes(z_fit)
        a_coeffs, fit_metrics = fit_polynomial_relation(
                z_fit, m1_fit, s=deg_m, deg_z=deg_z, ridge_lambda=reg,
                triangular=triangular, normalize=normalize, mu=mu,
                mu_reg=mu_reg)

        self.a_coeffs = a_coeffs

        # Estimate support from the fitted polynomial
        self.est_support, _ = self.estimate_support(a_coeffs)

        # Reporting error
        P_res = numpy.abs(eval_P(z_fit, m1_fit, a_coeffs))
        res_max = numpy.max(P_res[numpy.isfinite(P_res)])
        res_99_9 = numpy.quantile(P_res[numpy.isfinite(P_res)], 0.999)

        # Check polynomial has Stieltjes root
        x_min = self.lam_m - 1.0
        x_max = self.lam_p + 1.0
        status = sanity_check_stieltjes_branch(a_coeffs, x_min, x_max,
                                               eta=max(y_eps, 1e-2), n_x=128,
                                               max_bad_frac=0.05)

        status['res_max'] = float(res_max)
        status['res_99_9'] = float(res_99_9)
        status['fit_metrics'] = fit_metrics
        self.status = status

        # -----------------

        # Inflate a bit to make sure all points are searched
        # x_min, x_max = self._inflate_broad_support(inflate=0.2)
        # scale = float(max(1.0, abs(x_max - x_min), abs(x_min), abs(x_max)))
        # eta = 1e-6 * scale
        #
        # vopt = {
        #     'lam_space': 1.0,
        #     'lam_asym': 1.0,
        #     'lam_tiny_im': 200.0,
        #     'tiny_im': 0.5 * eta,
        #     'tol_im': 1e-14,
        # }

        # NOTE overwrite init
        self._stieltjes = StieltjesPoly(self.a_coeffs)
        # self._stieltjes = StieltjesPoly(self.a_coeffs, viterbi_opt=vopt)

        self._moments_base = AlgebraicStieltjesMoments(a_coeffs)
        self.moments = Moments(self._moments_base)

        if verbose:
            print(f'fit residual max  : {res_max:>0.4e}')
            print(f'fit residual 99.9%: {res_99_9:>0.4e}')

            print('\nCoefficients (real)')
            with numpy.printoptions(precision=8, suppress=True):
                for i in range(a_coeffs.shape[0]):
                    for j in range(a_coeffs.shape[1]):
                        v = a_coeffs[i, j]
                        print(f'{v.real:>+0.8f}', end=' ')
                    print('')

            a_coeffs_img_norm = numpy.linalg.norm(a_coeffs.imag, ord='fro')
            print(f'\nCoefficients (imag) norm: {a_coeffs_img_norm:>0.4e}')

            if not status['ok']:
                print("\nWARNING: sanity check failed:\n" +
                      f"\tfrac_bad: {status['frac_bad']:>0.3f}\n" +
                      f"\tn_bad   : {status['n_bad']}\n" +
                      f"\tn_test  : {status['n_test']}")
            else:
                print('\nStieltjes sanity check: OK')

        return a_coeffs, self.est_support, status

    # =====================
    # inflate broad support
    # =====================

    def _inflate_broad_support(self, inflate=0.0):
        """
        """

        min_supp, max_supp = self.broad_support

        c_supp = 0.5 * (max_supp + min_supp)
        r_supp = 0.5 * (max_supp - min_supp)

        x_min = c_supp - r_supp * (1.0 + inflate)
        x_max = c_supp + r_supp * (1.0 + inflate)

        return x_min, x_max

    # ================
    # estimate support
    # ================

    def estimate_support(self, a_coeffs=None, scan_range=None, n_scan=4000):
        """
        """

        if a_coeffs is None:
            if self.a_coeffs is None:
                raise RuntimeError('Call "fit" first.')
            else:
                a_coeffs = self.a_coeffs

        # Inflate a bit to make sure all points are searched
        if scan_range is not None:
            x_min, x_max = scan_range
        else:
            x_min, x_max = self._inflate_broad_support(inflate=0.2)

        est_support, info = compute_support(a_coeffs, x_min=x_min, x_max=x_max,
                                            n_scan=n_scan)

        return est_support, info

    # ======================
    # estimate branch points
    # ======================

    def estimate_branch_points(self, tol=1e-15, real_tol=None):
        """
        Compute global branch points and zeros of leading a_j
        """

        if self.a_coeffs is None:
            raise RuntimeError('Call "fit" first.')

        bp, leading_zeros, info = compute_branch_points(
            self.a_coeffs, tol=tol, real_tol=real_tol)

        return bp, leading_zeros, info

    # =============
    # generate grid
    # =============

    def _generate_grid(self, scale, extend=1.0, N=500):
        """
        Generate a grid of points to evaluate density / Hilbert / Stieltjes
        transforms.
        """

        radius = 0.5 * (self.lam_p - self.lam_m)
        center = 0.5 * (self.lam_p + self.lam_m)

        x_min = numpy.floor(extend * (center - extend * radius * scale))
        x_max = numpy.ceil(extend * (center + extend * radius * scale))

        x_min /= extend
        x_max /= extend

        return numpy.linspace(x_min, x_max, N)

    # =======
    # density
    # =======

    def density(self, x=None, plot=False, latex=False, save=False):
        """
        Evaluate spectral density.

        Parameters
        ----------

        x : numpy.array, default=None
            Positions where density to be evaluated at. If `None`, an interval
            slightly larger than the support interval will be used.

        plot : bool, default=False
            If `True`, density is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        rho : numpy.array
            Density at locations x.

        See Also
        --------
        hilbert
        stieltjes

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if self.a_coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25)

        # Preallocate density to zero
        z = x.astype(complex) + 1j * self.delta
        rho = self._stieltjes(z).imag / numpy.pi

        if plot:
            plot_density(x, rho, eig=self.eig, support=self.broad_support,
                         label='Estimate', latex=latex, save=save)

        return rho

    # =======
    # hilbert
    # =======

    def hilbert(self, x=None, plot=False, latex=False, save=False):
        """
        Compute Hilbert transform of the spectral density.

        Parameters
        ----------

        x : numpy.array, default=None
            The locations where Hilbert transform is evaluated at. If `None`,
            an interval slightly larger than the support interval of the
            spectral density is used.

        plot : bool, default=False
            If `True`, density is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        hilb : numpy.array
            The Hilbert transform on the locations `x`.

        See Also
        --------
        density
        stieltjes

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if self.a_coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25)

        # Preallocate density to zero
        hilb = -self._stieltjes(x).real / numpy.pi

        if plot:
            plot_hilbert(x, hilb, support=self.broad_support, latex=latex,
                         save=save)

        return hilb

    # =========
    # stieltjes
    # =========

    def stieltjes(self, x=None, y=None, plot=False, latex=False, save=False):
        """
        Compute Stieltjes transform of the spectral density on a grid.

        This function evaluates Stieltjes transform on an array of points, or
        over a 2D Cartesian grid on the complex plane.

        Parameters
        ----------

        x : numpy.array, default=None
            The x axis of the grid where the Stieltjes transform is evaluated.
            If `None`, an interval slightly larger than the support interval of
            the spectral density is used.

        y : numpy.array, default=None
            The y axis of the grid where the Stieltjes transform is evaluated.
            If `None`, a grid on the interval ``[-1, 1]`` is used.

        plot : bool, default=False
            If `True`, density is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        m : numpy.ndarray
            The Stieltjes transform on the principal branch.

        See Also
        --------

        density
        hilbert

        Examples
        --------

        .. code-block:: python

            >>> from freealg import FreeForm
        """

        if self.a_coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(2.0, extend=2.0)[::2]

        # Create y if not given
        if (plot is False) and (y is None):
            # Do not use a Cartesian grid. Create a 1D array z slightly above
            # the real line.
            y = self.delta * 1j
            z = x.astype(complex) + y             # shape (Nx,)
        else:
            # Use a Cartesian grid
            if y is None:
                y = numpy.linspace(-1, 1, 200)
            x_grid, y_grid = numpy.meshgrid(x.real, y.real)
            z = x_grid + 1j * y_grid              # shape (Ny, Nx)

        m = self._stieltjes(z, progress=True)

        if plot:
            plot_stieltjes(x, y, m, m, self.broad_support, latex=latex,
                           save=save)

        return m

    # ==============
    # eval stieltjes
    # ==============

    def _eval_stieltjes(self, z, branches=False):
        """
        Compute Stieltjes transform of the spectral density.

        Parameters
        ----------

        z : numpy.array
            The z values in the complex plan where the Stieltjes transform is
            evaluated.

        branches : bool, default = False
            Return both the principal and secondary branches of the Stieltjes
            transform. The default ``branches=False`` will return only
            the secondary branch.

        Returns
        -------

        m_p : numpy.ndarray
            The Stieltjes transform on the principal branch if
            ``branches=True``.

        m_m : numpy.ndarray
            The Stieltjes transform continued to the secondary branch.
        """

        pass

    # ==========
    # decompress
    # ==========

    def decompress(self, size, x=None, method='one', plot=False, latex=False,
                   save=False, verbose=False, min_n_times=10,
                   newton_opt={'max_iter': 50, 'tol': 1e-12, 'armijo': 1e-4,
                               'min_lam': 1e-6, 'w_min': 1e-14,
                               'sweep': True}):
        """
        Free decompression of spectral density.
        """

        # Check size argument
        if numpy.isscalar(size):
            size = int(size)
        else:
            # Check monotonic increment (either all increasing or decreasing)
            diff = numpy.diff(size)
            if not (numpy.all(diff >= 0) or numpy.all(diff <= 0)):
                raise ValueError('"size" increment should be monotonic.')

        # Decompression ratio equal to e^{t}.
        alpha = numpy.atleast_1d(size) / self.n

        # Lower and upper bound on new support
        hilb_lb = \
            (1.0 / self._stieltjes(self.lam_m + self.delta * 1j).item()).real
        hilb_ub = \
            (1.0 / self._stieltjes(self.lam_p + self.delta * 1j).item()).real
        lb = self.lam_m - (numpy.max(alpha) - 1) * hilb_lb
        ub = self.lam_p - (numpy.max(alpha) - 1) * hilb_ub

        # Create x if not given
        if x is None:
            radius = 0.5 * (ub - lb)
            center = 0.5 * (ub + lb)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 200)
        else:
            x = numpy.asarray(x)

        if method == 'one':

            # Query grid on the real axis + a small imaginary buffer
            z_query = x + 1j * self.delta

            # Initial condition at t = 0 (physical branch)
            w0_list = self._stieltjes(z_query)

            # Ensure there are at least min_n_times time t, including requested
            # times, and especially time t = 0
            t_all, idx_req = build_time_grid(
                size, self.n, min_n_times=min_n_times)

            # Evolve
            W, ok = decompress_newton(
                z_query, t_all, self.a_coeffs,
                w0_list=w0_list, **newton_opt)

            rho_all = W.imag / numpy.pi

            # return only the user-requested ones
            rho = rho_all[idx_req]

            if verbose:
                print("success rate per t:", ok.mean(axis=1))

        elif method == 'two':

            # Preallocate density to zero
            rho = numpy.zeros((alpha.size, x.size), dtype=float)

            # Decompress to each alpha
            for i in range(alpha.size):
                t_i = numpy.log(alpha[i])
                coeffs_i = decompress_coeffs(self.a_coeffs, t_i)

                def mom(k):
                    return self.moments(k, t_i)

                stieltjes_i = StieltjesPoly(coeffs_i, mom)
                rho[i, :] = stieltjes_i(x).imag

            rho = rho / numpy.pi

        else:
            raise ValueError('"method" is invalid.')

        # If the input size was only a scalar, return a 1D rho, otherwise 2D.
        if numpy.isscalar(size):
            rho = numpy.squeeze(rho)

        # Plot only the last size
        if plot:
            if numpy.isscalar(size):
                rho_last = rho
            else:
                rho_last = rho[-1, :]
            plot_density(x, rho_last, support=(lb, ub),
                         label='Decompression', latex=latex, save=save)

        return rho, x

    # ==========
    # candidates
    # ==========

    def candidates(self, size, x=None, verbose=False):

        # Check size argument
        if numpy.isscalar(size):
            size = int(size)
        else:
            # Check monotonic increment (either all increasing or decreasing)
            diff = numpy.diff(size)
            if not (numpy.all(diff >= 0) or numpy.all(diff <= 0)):
                raise ValueError('"size" increment should be monotonic.')

        # Decompression ratio equal to e^{t}.
        alpha = numpy.atleast_1d(size) / self.n

        # Lower and upper bound on new support
        hilb_lb = \
            (1.0 / self._stieltjes(self.lam_m + self.delta * 1j).item()).real
        hilb_ub = \
            (1.0 / self._stieltjes(self.lam_p + self.delta * 1j).item()).real
        lb = self.lam_m - (numpy.max(alpha) - 1) * hilb_lb
        ub = self.lam_p - (numpy.max(alpha) - 1) * hilb_ub

        # Create x if not given
        if x is None:
            radius = 0.5 * (ub - lb)
            center = 0.5 * (ub + lb)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 2000)
        else:
            x = numpy.asarray(x)

        for i in range(alpha.size):
            t_i = numpy.log(alpha[i])
            coeffs_i = decompress_coeffs(self.a_coeffs, t_i)
            plot_candidates(coeffs_i, x, size=int(alpha[i]*self.n),
                            verbose=verbose)

    # ====
    # edge
    # ====

    def edge(self, t, eta=1e-3, dt_max=0.1, max_iter=30, tol=1e-12,
             verbose=False):
        """
        Evolves spectral edges.

        Fix: if t is a scalar or length-1 array, we prepend t=0 internally so
        evolve_edges actually advances from the initialization at t=0.
        """

        if self.support is not None:
            known_support = self.support
        elif self.est_support is not None:
            known_support = self.est_support
        else:
            raise RuntimeError('Call "fit" first.')

        t = numpy.asarray(t, dtype=float).ravel()

        if t.size == 1:
            t1 = float(t[0])
            if t1 == 0.0:
                t_grid = numpy.array([0.0], dtype=float)
                complex_edges, ok_edges = evolve_edges(
                    t_grid, self.a_coeffs, support=known_support, eta=eta,
                    dt_max=dt_max, max_iter=max_iter, tol=tol
                )
            else:
                # prepend 0 and drop it after evolution
                t_grid = numpy.array([0.0, t1], dtype=float)
                complex_edges2, ok_edges2 = evolve_edges(
                    t_grid, self.a_coeffs, support=known_support, eta=eta,
                    dt_max=dt_max, max_iter=max_iter, tol=tol
                )
                complex_edges = complex_edges2[-1:, :]
                ok_edges = ok_edges2[-1:, :]
        else:
            # For vector t, require it starts at 0 for correct initialization
            # (you can relax this if you want by prepending 0 similarly).
            complex_edges, ok_edges = evolve_edges(
                t, self.a_coeffs, support=known_support, eta=eta,
                dt_max=dt_max, max_iter=max_iter, tol=tol
            )

        real_edges = complex_edges.real

        # Remove spurious edges / merges for plotting
        real_merged_edges, active_k = merge_edges(real_edges, tol=1e-4)

        if verbose:
            print("edge success rate:", ok_edges.mean())

        return complex_edges, real_merged_edges, active_k

    # ====
    # cusp
    # ====

    def cusp(self, t_grid):
        """
        """

        return cusp_wrap(self, t_grid, edge_kwargs=None, max_iter=50,
                         tol=1.0e-12)

    # ========
    # eigvalsh
    # ========

    def eigvalsh(self, size=None, seed=None, **kwargs):
        """
        Estimate the eigenvalues.

        This function estimates the eigenvalues of the freeform matrix
        or a larger matrix containing it using free decompression.

        Parameters
        ----------

        size : int, default=None
            The size of the matrix containing :math:`\\mathbf{A}` to estimate
            eigenvalues of. If None, returns estimates of the eigenvalues of
            :math:`\\mathbf{A}` itself.

        seed : int, default=None
            The seed for the Quasi-Monte Carlo sampler.

        **kwargs : dict, optional
            Pass additional options to the underlying
            :func:`FreeForm.decompress` function.

        Returns
        -------

        eigs : numpy.array
            Eigenvalues of decompressed matrix

        See Also
        --------

        FreeForm.decompress
        FreeForm.cond

        Notes
        -----

        All arguments to the `.decompress()` procedure can be provided.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 1

            >>> from freealg import FreeForm
        """

        # if size is None:
        #     size = self.n
        #
        # rho, x = self.decompress(size, **kwargs)
        # eigs = numpy.sort(sample(x, rho, size, method='qmc', seed=seed))
        #
        # return eigs
        pass

    # ====
    # cond
    # ====

    def cond(self, size=None, seed=None, **kwargs):
        """
        Estimate the condition number.

        This function estimates the condition number of the matrix
        :math:`\\mathbf{A}` or a larger matrix containing :math:`\\mathbf{A}`
        using free decompression.

        Parameters
        ----------

        size : int, default=None
            The size of the matrix containing :math:`\\mathbf{A}` to estimate
            eigenvalues of. If None, returns estimates of the eigenvalues of
            :math:`\\mathbf{A}` itself.

        **kwargs : dict, optional
            Pass additional options to the underlying
            :func:`FreeForm.decompress` function.

        Returns
        -------

        c : float
            Condition number

        See Also
        --------

        FreeForm.eigvalsh
        FreeForm.norm
        FreeForm.slogdet
        FreeForm.trace

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 1

            >>> from freealg import FreeForm
        """

        eigs = self.eigvalsh(size=size, **kwargs)
        return eigs.max() / eigs.min()

    # =====
    # trace
    # =====

    def trace(self, size=None, p=1.0, seed=None, **kwargs):
        """
        Estimate the trace of a power.

        This function estimates the trace of the matrix power
        :math:`\\mathbf{A}^p` of the freeform or that of a larger matrix
        containing it.

        Parameters
        ----------

        size : int, default=None
            The size of the matrix containing :math:`\\mathbf{A}` to estimate
            eigenvalues of. If None, returns estimates of the eigenvalues of
            :math:`\\mathbf{A}` itself.

        p : float, default=1.0
            The exponent :math:`p` in :math:`\\mathbf{A}^p`.

        seed : int, default=None
            The seed for the Quasi-Monte Carlo sampler.

        **kwargs : dict, optional
            Pass additional options to the underlying
            :func:`FreeForm.decompress` function.

        Returns
        -------

        trace : float
            matrix trace

        See Also
        --------

        FreeForm.eigvalsh
        FreeForm.cond
        FreeForm.slogdet
        FreeForm.norm

        Notes
        -----

        The trace is highly amenable to subsampling: under free decompression
        the average eigenvalue is assumed constant, so the trace increases
        linearly. Traces of powers fall back to :func:`eigvalsh`.
        All arguments to the `.decompress()` procedure can be provided.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 1

            >>> from freealg import FreeForm
        """

        if numpy.isclose(p, 1.0):
            return numpy.mean(self.eig) * (size / self.n)

        eig = self.eigvalsh(size=size, seed=seed, **kwargs)
        return numpy.sum(eig ** p)

    # =======
    # slogdet
    # =======

    def slogdet(self, size=None, seed=None, **kwargs):
        """
        Estimate the sign and logarithm of the determinant.

        This function estimates the *slogdet* of the freeform or that of
        a larger matrix containing it using free decompression.

        Parameters
        ----------

        size : int, default=None
            The size of the matrix containing :math:`\\mathbf{A}` to estimate
            eigenvalues of. If None, returns estimates of the eigenvalues of
            :math:`\\mathbf{A}` itself.

        seed : int, default=None
            The seed for the Quasi-Monte Carlo sampler.

        Returns
        -------

        sign : float
            Sign of determinant

        ld : float
            natural logarithm of the absolute value of the determinant

        See Also
        --------

        FreeForm.eigvalsh
        FreeForm.cond
        FreeForm.trace
        FreeForm.norm

        Notes
        -----

        All arguments to the `.decompress()` procedure can be provided.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 1

            >>> from freealg import FreeForm
        """

        eigs = self.eigvalsh(size=size, seed=seed, **kwargs)
        sign = numpy.prod(numpy.sign(eigs))
        ld = numpy.sum(numpy.log(numpy.abs(eigs)))
        return sign, ld

    # ====
    # norm
    # ====

    def norm(self, size=None, order=2, seed=None, **kwargs):
        """
        Estimate the Schatten norm.

        This function estimates the norm of the freeform or a larger
        matrix containing it using free decompression.

        Parameters
        ----------

        size : int, default=None
            The size of the matrix containing :math:`\\mathbf{A}` to estimate
            eigenvalues of. If None, returns estimates of the eigenvalues of
            :math:`\\mathbf{A}` itself.

        order : {float, ``''inf``, ``'-inf'``, ``'fro'``, ``'nuc'``}, default=2
            Order of the norm.

            * float :math:`p`: Schatten p-norm.
            * ``'inf'``: Largest absolute eigenvalue
              :math:`\\max \\vert \\lambda_i \\vert)`
            * ``'-inf'``: Smallest absolute eigenvalue
              :math:`\\min \\vert \\lambda_i \\vert)`
            * ``'fro'``: Frobenius norm corresponding to :math:`p=2`
            * ``'nuc'``: Nuclear (or trace) norm corresponding to :math:`p=1`

        seed : int, default=None
            The seed for the Quasi-Monte Carlo sampler.

        **kwargs : dict, optional
            Pass additional options to the underlying
            :func:`FreeForm.decompress` function.

        Returns
        -------

        norm : float
            matrix norm

        See Also
        --------

        FreeForm.eigvalsh
        FreeForm.cond
        FreeForm.slogdet
        FreeForm.trace

        Notes
        -----

        Thes Schatten :math:`p`-norm is defined by

        .. math::

            \\Vert \\mathbf{A} \\Vert_p = \\left(
            \\sum_{i=1}^N \\vert \\lambda_i \\vert^p \\right)^{1/p}.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 1

            >>> from freealg import FreeForm
        """

        eigs = self.eigvalsh(size, seed=seed, **kwargs)

        # Check order type and convert to float
        if order == 'nuc':
            order = 1
        elif order == 'fro':
            order = 2
        elif order == 'inf':
            order = float('inf')
        elif order == '-inf':
            order = -float('inf')
        elif not isinstance(order,
                            (int, float, numpy.integer, numpy.floating)) \
                and not isinstance(order, (bool, numpy.bool_)):
            raise ValueError('"order" is invalid.')

        # Compute norm
        if numpy.isinf(order) and not numpy.isneginf(order):
            norm_ = max(numpy.abs(eigs))

        elif numpy.isneginf(order):
            norm_ = min(numpy.abs(eigs))

        elif isinstance(order, (int, float, numpy.integer, numpy.floating)) \
                and not isinstance(order, (bool, numpy.bool_)):
            norm_q = numpy.sum(numpy.abs(eigs)**order)
            norm_ = norm_q**(1.0 / order)

        return norm_
