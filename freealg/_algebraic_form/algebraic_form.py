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
from .._util import compute_eig
from ._continuation_algebraic import sample_z_joukowski, \
        filter_z_away_from_cuts, fit_polynomial_relation, \
        sanity_check_stieltjes_branch, eval_P
from ._edge import evolve_edges, merge_edges
from ._cusp_wrap import cusp_wrap
from ._decompressible import precheck_laurent
from ._decompress_util import build_time_grid
from ._decompress_coeffs import decompress_coeffs, plot_candidates

# Decompress with Newton
# from ._decompress import decompress_newton
# from ._decompress4 import decompress_newton # WORKS (mass issue)
# from ._decompress5 import decompress_newton
# from ._decompress6 import decompress_newton
# from ._decompress4_2 import decompress_newton
# from ._decompress_new_2 import decompress_newton
# from ._decompress_new import decompress_newton
# from ._decompress6 import decompress_newton
# from ._decompress7 import decompress_newton
# from ._decompress8 import decompress_newton
from ._decompress9 import decompress_newton  # With Predictor/Corrector

# Homotopy
# from ._homotopy import StieltjesPoly
# from ._homotopy2 import StieltjesPoly
# from ._homotopy3 import StieltjesPoly  # Viterbi
# from ._homotopy4 import StieltjesPoly
from ._homotopy5 import StieltjesPoly

from ._branch_points import estimate_branch_points, plot_branch_points
from ._atoms import detect_atoms
from ._support import estimate_support
from .._support import supp as estimate_broad_supp
from ._moments import Moments, AlgebraicStieltjesMoments
from ..visualization._plot_util import plot_density, plot_hilbert, \
    plot_stieltjes
from .._base_form import BaseForm

# Fallback to previous numpy API
if not hasattr(numpy, 'trapezoid'):
    numpy.trapezoid = numpy.trapz

__all__ = ['AlgebraicForm']


# ==============
# Algebraic Form
# ==============

class AlgebraicForm(BaseForm):
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

    **kwargs : dict, optional
        Parameters for the :func:`supp` function can also be prescribed
        here when ``support=None``.

    Attributes
    ----------

    eig : numpy.array
        Eigenvalues of the matrix

    supp: tuple
        The predicted (or given) support :math:`(\\lambda_{\\min},
        \\lambda_{\\max})` of the eigenvalue density.

    n : int
      Initial array size (assuming a square matrix when :math:`\\mathbf{A}` is
      2D).

    Methods
    -------

    fit
        Fit an algebraic structure to the input data

    support
        Estimate the spectral edges of the density

    branch_points
        Compute global branch points and zeros of leading coefficient

    atoms
        Detect atom locations and weights of distribution

    density
        Evaluate spectral density

    hilbert
        Compute Hilbert transform of the spectral density

    stieltjes
        Compute Stieltjes transform of the spectral density

    decompress
        Free decompression of spectral density

    candidate
        Candidate densities of free decompression from all possible roots

    is_decompressible
        Check if the underlying distribution can be decompressed

    edge
        Evolves spectral edges

    cusp
        Find cusp (merge) point of evolving spectral edges

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

        >>> from freealg import AlgebraicForm
    """

    # ====
    # init
    # ====

    def __init__(self, A, support=None, delta=1e-5, dtype='complex128',
                 **kwargs):
        """
        Initialization.
        """

        super().__init__(delta, dtype)

        self._stieltjes = None
        self._moments = None
        self.supp = support
        self.est_supp = None  # Estimated from polynomial after fitting

        if hasattr(A, 'stieltjes') and callable(getattr(A, 'stieltjes', None)):
            # This is one of the distribution objects, like MarchenkoPastur
            self._stieltjes = A.stieltjes
            self.supp = A.support()
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
        if self.supp is None:
            if self.eig is None:
                raise RuntimeError("Support must be provided without data")

            # Detect support
            self.lam_m, self.lam_p = estimate_broad_supp(self.eig, **kwargs)
            self.broad_supp = (float(self.lam_m), float(self.lam_p))
        else:
            self.lam_m = float(min([s[0] for s in self.supp]))
            self.lam_p = float(max([s[1] for s in self.supp]))
            self.broad_supp = (self.lam_m, self.lam_p)

        # Initialize
        self.coeffs = None                 # Polynomial coefficients
        self.status = None                 # Fitting status

    # ===
    # fit
    # ===

    def fit(self, deg_m, deg_z, reg=0.0, r=[1.25, 6.0, 20.0], n_r=[3, 2, 1],
            n_samples=4096, y_eps=2e-2, x_pad=0.0, triangular='auto', mu=None,
            mu_reg=None, normalize=True, verbose=False):
        """
        Fit an algebraic structure to the input data.

        Parameters
        ----------

        deg_m : int
            Degree :math:`\\mathrm{deg}_m(P)`

        deg_z : int
            Degree :math:`\\mathrm{deg}_z(P)`

        mu : array_like, default= ``'auto'``
            Constraint to fit polynomial coefficients based on moments:

            * If an array :math:`[\\mu_0, \\mu_1`, \\dots, \\mu_r]` is given,
              it enforces the first :math:`r+1` moments. Note that
              :math:`\\mu_0 = 1` to ensure unit mass.
            * If instead this option is set to ``'auto'``, and the input ``A``
              is a matrix, it automatically uses the first two moments of the
              eigenvalues of the input matrix as moment constraints.
            * If `None`, no constraint is used.

            See also ``mu_reg``.

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

        # Sampling around support, or broad_support. This is only needed to
        # ensure sampled points are not hitting the support itself is not used
        # in any computation. If support is not known, use broad support.
        if self.supp is not None:
            possible_supp = self.supp
        else:
            possible_supp = [self.broad_supp]

        # Sampling points for fitting
        z_fits = []
        for sup in possible_supp:
            a, b = sup

            for i in range(len(r)):
                z_fits.append(sample_z_joukowski(a, b, n_samples=n_samples,
                                                 r=r[i], n_r=n_r[i]))

        z_fit = numpy.concatenate(z_fits)

        # Remove points too close to any cut
        z_fit = filter_z_away_from_cuts(z_fit, possible_supp, y_eps=y_eps,
                                        x_pad=x_pad)

        # Automatically add mu constraints from eigenvalues
        if mu == 'auto':
            if self.eig is not None:
                mu_0 = 1.0
                mu_1 = numpy.mean(self.eig)
                mu_2 = numpy.var(self.eig)
                mu = [mu_0, mu_1, mu_2]
            else:
                mu = None

        # Fitting (w_inf = None means adaptive weight selection)
        m1_fit = self._stieltjes(z_fit)
        self.coeffs, fit_metrics = fit_polynomial_relation(
                z_fit, m1_fit, s=deg_m, deg_z=deg_z, ridge_lambda=reg,
                triangular=triangular, normalize=normalize, mu=mu,
                mu_reg=mu_reg)

        # Estimate support from the fitted polynomial
        self.est_supp, _ = self.support(self.coeffs)

        # Reporting error
        P_res = numpy.abs(eval_P(z_fit, m1_fit, self.coeffs))
        res_max = numpy.max(P_res[numpy.isfinite(P_res)])
        res_99_9 = numpy.quantile(P_res[numpy.isfinite(P_res)], 0.999)

        # Check polynomial has Stieltjes root
        x_min = self.lam_m - 1.0
        x_max = self.lam_p + 1.0
        status = sanity_check_stieltjes_branch(self.coeffs, x_min, x_max,
                                               eta=max(y_eps, 1e-2), n_x=128,
                                               max_bad_frac=0.05)

        status['res_max'] = float(res_max)
        status['res_99_9'] = float(res_99_9)
        status['fit_metrics'] = fit_metrics
        self.status = status

        # -----------------

        # Inflate a bit to make sure all points are searched
        # x_min, x_max = self._inflate_broad_supp(inflate=0.2)
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
        self._stieltjes = StieltjesPoly(self.coeffs)
        # self._stieltjes = StieltjesPoly(self.coeffs, viterbi_opt=vopt)

        self._moments_base = AlgebraicStieltjesMoments(self.coeffs)
        self.moments = Moments(self._moments_base)

        if verbose:
            print(f'fit residual max  : {res_max:>0.4e}')
            print(f'fit residual 99.9%: {res_99_9:>0.4e}')

            print('\nCoefficients (real)')
            with numpy.printoptions(precision=8, suppress=True):
                for i in range(self.coeffs.shape[0]):
                    for j in range(self.coeffs.shape[1]):
                        v = self.coeffs[i, j]
                        print(f'{v.real:>+0.8f}', end=' ')
                    print('')

            coeffs_img_norm = numpy.linalg.norm(self.coeffs.imag, ord='fro')
            print(f'\nCoefficients (imag) norm: {coeffs_img_norm:>0.4e}')

            if not status['ok']:
                print("\nWARNING: sanity check failed:\n" +
                      f"\tfrac_bad: {status['frac_bad']:>0.3f}\n" +
                      f"\tn_bad   : {status['n_bad']}\n" +
                      f"\tn_test  : {status['n_test']}")
            else:
                print('\nStieltjes sanity check: OK')

        return self.coeffs, self.est_supp, status

    # ==================
    # inflate broad supp
    # ==================

    def _inflate_broad_supp(self, inflate=0.0):
        """
        Inflate the broad support for better post-processing, such as detecting
        branch points, spectral edges, etc.
        """

        if inflate < 0:
            raise ValueError('"inflate" should be non-negative.')

        min_supp, max_supp = self.broad_supp

        c_supp = 0.5 * (max_supp + min_supp)
        r_supp = 0.5 * (max_supp - min_supp)

        x_min = c_supp - r_supp * (1.0 + inflate)
        x_max = c_supp + r_supp * (1.0 + inflate)

        return x_min, x_max

    # =======
    # support
    # =======

    def support(self, coeffs=None, scan_range=None, n_scan=4000):
        """
        Estimate the spectral edges of the density.
        """

        if coeffs is None:
            if self.coeffs is None:
                raise RuntimeError('Call "fit" first.')
            else:
                coeffs = self.coeffs

        # Inflate a bit to make sure all points are searched
        if scan_range is not None:
            x_min, x_max = scan_range
        else:
            x_min, x_max = self._inflate_broad_supp(inflate=0.2)

        est_supp, info = estimate_support(coeffs, x_min=x_min, x_max=x_max,
                                          n_scan=n_scan)

        return est_supp, info

    # =============
    # branch points
    # =============

    def branch_points(self, tol=1e-15, real_tol=None, plot=False, latex=False,
                      save=False):
        """
        Compute global branch points.

        Parameters
        ----------

        plot : bool, default=False
            If `True`, the branch points together with spectral edges and atoms
            will be plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        See Also
        --------

        atoms
        """

        if self.coeffs is None:
            raise RuntimeError('Call "fit" first.')

        bp, info = estimate_branch_points(
            self.coeffs, tol=tol, real_tol=real_tol)

        if plot:
            atoms_list = self.atoms()
            est_supp, _ = self.support()
            plot_branch_points(bp, atoms_list, est_supp, latex=latex,
                               save=save)

        return bp, info

    # =====
    # atoms
    # =====

    def atoms(self, eta=1e-6, tol=1e-12, real_tol=None, w_tol=1e-10,
              merge_tol=1e-8):
        """
        Detect atom locations and weights of distribution

        This routine uses the necessary condition for a finite pole: a_s(z0)=0,
        where a_s(z) is the leading coefficient of P in powers of m. Candidate
        atom locations are the (nearly) real roots of a_s(z). The atom weight
        is estimated numerically from the Stieltjes transform as

            w ~= eta * Im(m(z0 + i*eta)),

        which follows from m(z) ~ -w/(z - z0) near an atom at z0.

        Parameters
        ----------

        eta : float, default=1e-6
            Small imaginary part used to probe the pole strength.

        tol : float, default=1e-12
            Tolerance for trimming polynomial coefficients.

        real_tol : float or None, default=None
            Tolerance for treating a complex root as real. If None, uses
            ``1e3*tol``.

        w_tol : float, default=1e-10
            Minimum atom weight to report.

        merge_tol : float, default=1e-8
            Merge roots whose real parts differ by at most this tolerance.

        Returns
        -------
        atoms : list of (float, float)
            List of ``(atom_loc, atom_w)``. Locations are real numbers and
            weights are nonnegative.

        See Also
        --------

        branch_points

        Notes
        -----
        """

        if self.coeffs is None:
            raise RuntimeError('Call "fit" first.')

        atoms_list = detect_atoms(self.coeffs, self._stieltjes, eta=eta,
                                  tol=tol, real_tol=real_tol, w_tol=w_tol,
                                  merge_tol=merge_tol)

        return atoms_list

    # =======
    # density
    # =======

    def density(self, x=None, eta=2e-4, ac_only=False, plot=False, latex=False,
                save=False):
        """
        Evaluate spectral density.

        Parameters
        ----------

        x : numpy.array, default=None
            Positions where density to be evaluated at. If `None`, an interval
            slightly larger than the support interval will be used.

        eta : float, default=2e-4
            A small number to be used for approximating the local behavior of
            atom with a mollifier of scale ``eta``.

        ac_only : bool, default=True
            If `True`, it returns the absolutely-continuous part of density.

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

        if self.coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25)

        # Preallocate density to zero
        z = x.astype(complex) + 1j * self.delta
        rho = self._stieltjes(z).imag / numpy.pi

        rho_ac = rho
        atoms_list = self.atoms()
        if len(atoms_list) > 0:
            for atom_loc, atom_w in atoms_list:

                # Mollifier to approximate atom function
                atom_func = (atom_w * eta / (numpy.pi)) / \
                    ((x - atom_loc)**2 + eta**2)
                rho_ac = rho_ac - atom_func
            rho_ac = numpy.maximum(rho_ac, 0.0)

            if (ac_only is True):
                rho = rho_ac

        if plot:
            plot_density(x, rho_ac, eig=self.eig, atoms=atoms_list,
                         support=self.broad_supp, label='Estimate',
                         latex=latex, save=save)

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

        if self.coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25)

        # Preallocate density to zero
        hilb = -self._stieltjes(x).real / numpy.pi

        if plot:
            plot_hilbert(x, hilb, support=self.broad_supp, latex=latex,
                         save=save)

        return hilb

    # =========
    # stieltjes
    # =========

    def stieltjes(self, x=None, y=None, plot=False, latex=False, save=False):
        """
        Compute Stieltjes transform of the spectral density

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

        if self.coeffs is None:
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
            plot_stieltjes(x, y, m, m, self.broad_supp, latex=latex,
                           save=save)

        return m

    # ==========
    # decompress
    # ==========

    def decompress(self, size, x=None, method='moc', min_n_times=10,
                   newton_opt={'max_iter': 50, 'tol': 1e-12, 'armijo': 1e-4,
                               'min_lam': 1e-6, 'w_min': 1e-14,
                               'sweep': True},
                   plot=False, latex=False, save=False, verbose=False):
        """
        Free decompression of spectral density.

        Parameters
        ----------

        size : int or array_like
            Size(s) of the decompressed matrix. This can be a scalar or an
            array of sizes. For each matrix size in ``size`` array, a density
            is produced.

        x : numpy.array, default=None
            Positions where density to be evaluated at. If `None`, an interval
            slightly larger than the support interval will be used.

        method : {``'moc'``, ``'coeffs'`}, default= ``'moc'``
            Method of decompression:

            * ``'moc'``: Method of characteristics with Newton iterations.
            * ``'coeffs'``: Evolving polynomial coefficients directly.

        min_n_times : int, default=10
            Minimum number of inner sizes to evolve.

        newton_opt : dict
            A dictionary of settings to pass to Newton iteration solver.

        plot : bool, default=False
            If `True`, density is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        verbose : bool, default=False
            If `True`, it prints verbose be bugging information.

        Returns
        -------

        rho : numpy.array or numpy.ndarray
            Estimated spectral density at locations x. ``rho`` can be a 1D or
            2D array output:

            * If ``size`` is a scalar, ``rho`` is a 1D array of the same size
              as ``x``.
            * If ``size`` is an array of size `n`, ``rho`` is a 2D array with
              `n` rows, where each row corresponds to decompression to a size.
              Number of columns of ``rho`` is the same as the size of ``x``.

        x : numpy.array
            Locations where the spectral density is estimated

        See Also
        --------

        density
        stieltjes

        Examples
        --------

        .. code-block:: python

            >>> from freealg import AlgebraicForm

        """

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25)

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

        if method == 'moc':

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
                z_query, t_all, self.coeffs,
                w0_list=w0_list, **newton_opt)

            rho_all = W.imag / numpy.pi

            # return only the user-requested ones
            rho = rho_all[idx_req]

            if verbose:
                print("success rate per t:", ok.mean(axis=1))

        elif method == 'coeffs':

            # Preallocate density to zero
            rho = numpy.zeros((alpha.size, x.size), dtype=float)

            # Decompress to each alpha
            for i in range(alpha.size):
                t_i = numpy.log(alpha[i])
                coeffs_i = decompress_coeffs(self.coeffs, t_i)

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
        """
        Candidate densities of free decompression from all possible roots
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
            x = numpy.linspace(x_min, x_max, 2000)
        else:
            x = numpy.asarray(x)

        for i in range(alpha.size):
            t_i = numpy.log(alpha[i])
            coeffs_i = decompress_coeffs(self.coeffs, t_i)
            plot_candidates(coeffs_i, x, size=int(alpha[i]*self.n),
                            verbose=verbose)

    # =================
    # is decompressible
    # =================

    def is_decompressible(self, ratio=2, n_ratios=5, K=(6, 8, 10), L=3,
                          tol=1e-8, verbose=False):
        """
        Check if the given distribution can be decompressed.

        To this end, this function checks if the evolved polynomial under the
        free decompression admits a valid Stieltjes root.

        Parameters
        ----------

        ratio : float, default=2
            The maximum ratio of decompressed matrix size to the original size.

        n_ratios : int, default=5
            Number of ratios to test from 1 (no decompression) to the given
            maximum ``ratio``.

        K : sequence of int, default=(6, 8, 10)
            Truncation orders ``K`` used to build the Laurent cancellation
            system. Each ``K`` enforces powers ``p`` in ``[-L, ..., K]``.

        L : int, default=3
            Number of negative powers to enforce. The enforced power range is
            ``p in [-L, ..., K]`` for each ``K``.

        tol : float, default=1e-10
            Pass threshold for the best residual over ``K``. The residual
            is ``max_p |c_p|`` over enforced powers, where ``c_p`` are the
            Laurent coefficients of :math:`P_t(1/w, m(w))`.

        verbose : bool, default=True
            If True, print a per-``t`` summary and the per-``K`` diagnostics.

        Returns
        -------

        status : array
            Boolean array of `True` or `False for each time in ``t``.
            `True` means decompressible, and `False` means not decompressible.

        info : dict
            Dictionary with the following keys

            * ``'ratios'``: List of decompression ratios that is checked.
            * ``'ok'``: status of the decompressiblity at the tested ratio.
            * ``'res'``: details of test for each ratio.

        Raises
        ------

        ValueError
            If ``K`` is empty, or if any ``K`` or ``L`` is not positive.

        See Also
        --------

        branch_points :
            Geometric branch-point estimation; complementary to this asymptotic
            check.

        Notes
        -----

        This is a ""no-FD-run" diagnostic: it only uses the base algebraic
        relation :math:`P(z,m)=0` (stored in ``self.coeffs``) and tests the
        pushed relation under free decompression (FD) at selected expansion
        factors ``t``.

        The FD pushforward used here is the characteristic change-of-variables

        * :math:`\\tau = e^t`
        * :math:`y = \\tau * m`
        * :math:`\\zeta = z + (1 - 1/\\tau) / m`
        * :math:`P_t(z, m) = P(\\zeta, y)`

        A necessary condition for FD tracking to remain well-posed is that, for
        each :math:`\\tau > 1`), the pushed relation admits a "Stieltjes branch
        at infinity", i.e., a solution branch with the large
        :math:`\\vert z \\vert` behavior

        .. math::

            m(z) = -1/z + O(1/z^2)

        as :math:`\\vert z \\vert \\to \\infty` in :math:`\\Im(z) > 0`.

        This routine enforces that asymptotic structure by constructing a
        truncated Laurent expansion in :math:`w = 1/z`. It solves for
        coefficients in

        .. math::

            m(z) = -(alpha/z + mu_1/z^2 + mu_2/z^3 + ...)

        so that the Laurent series of :math:`P_t(1/w, m(w))` cancels on a
        prescribed range of powers. Concretely, for each truncation order
        ``K``, we enforce the cancellation of powers ``p`` in ``[-L, ..., K]``
        and measure the maximum absolute residual among those enforced
        coefficients. The best (smallest) residual across ``K`` is used for the
        main pass/fail decision.

        * ``K`` controls how many asymptotic constraints are enforced. Larger
          ``K`` is usually stricter but can become sensitive to coefficient
          noise (e.g. from a fitted polynomial).
        * ``L`` controls how many negative powers are enforced. A safe default
          is ``deg_z + 2``; here we expose it directly so you can tune it per
          model.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 9,10

            >>> import freealg AlgebraicForm
            >>> from freealg.distributions import CompoundPoisson

            >>> # Create compound free Poisson law
            >>> cp = CompoundPoisson(t1=2.0, t2=5.5, w1=0.75, c=0.1)
            >>> af = AlgebraicForm(cp)

            >>> # Check the decompressibility of compound free Poisson
            >>> status, info = af.is_decompressible(ratio=2, n_ratios=5,
            ...                                     verbose=True)

            >>> status
            True
        """

        if self.coeffs is None:
            raise RuntimeError('"fit" model first.')

        if ratio < 1:
            raise ValueError('"ratio" cannot be smaller than 1.')

        tau = numpy.linspace(1.0, ratio, n_ratios)
        ok = numpy.zeros_like(tau, dtype=bool)
        res = [None] * tau.size

        for i in range(tau.size):
            ok[i], res[i] = precheck_laurent(self.coeffs, tau[i], K_list=K,
                                             L=L, tol=tol, verbose=verbose)

            if verbose:
                print("")

        status = bool(numpy.all(ok))

        info = {
            'ratios': tau,
            'ok': ok,
            'res': res,
        }

        return status, info

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

        if self.supp is not None:
            known_supp = self.supp
        elif self.est_supp is not None:
            known_supp = self.est_supp
        else:
            raise RuntimeError('Call "fit" first.')

        t = numpy.asarray(t, dtype=float).ravel()

        if t.size == 1:
            t1 = float(t[0])
            if t1 == 0.0:
                t_grid = numpy.array([0.0], dtype=float)
                complex_edges, ok_edges = evolve_edges(
                    t_grid, self.coeffs, support=known_supp, eta=eta,
                    dt_max=dt_max, max_iter=max_iter, tol=tol
                )
            else:
                # prepend 0 and drop it after evolution
                t_grid = numpy.array([0.0, t1], dtype=float)
                complex_edges2, ok_edges2 = evolve_edges(
                    t_grid, self.coeffs, support=known_supp, eta=eta,
                    dt_max=dt_max, max_iter=max_iter, tol=tol)

                complex_edges = complex_edges2[-1:, :]
                ok_edges = ok_edges2[-1:, :]
        else:
            # For vector t, require it starts at 0 for correct initialization
            # (you can relax this if you want by prepending 0 similarly).
            complex_edges, ok_edges = evolve_edges(
                t, self.coeffs, support=known_supp, eta=eta,
                dt_max=dt_max, max_iter=max_iter, tol=tol)

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
        Find cusp (merge) point of evolving spectral edges
        """

        return cusp_wrap(self, t_grid, edge_kwargs=None, max_iter=50,
                         tol=1.0e-12)
