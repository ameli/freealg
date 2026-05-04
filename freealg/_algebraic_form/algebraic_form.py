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
from .._base_form import BaseForm
from .._util import compute_eig

# Option 1. Use this version for diffusion example
# from ._continuation_algebraic_old import sample_z_joukowski, \
#         filter_z_away_from_cuts, fit_polynomial_relation, eval_P

# Option 2. This version fixes moment constrains on orthogonal basis
from ._continuation_algebraic import sample_z_joukowski, \
        filter_z_away_from_cuts, fit_polynomial_relation, eval_P

# Edges and cups for decompress
from ._edge import evolve_edges, merge_edges, evolve_edges_with_births
from ._cusp_wrap import cusp_wrap

# Edges and cups for deform
from ._deform_edge import evolve_edges as deform_evolve_edges
from ._deform_edge import evolve_edges_with_births as \
    deform_evolve_edges_with_births
from ._deform_cusp_wrap import cusp_wrap as deform_cusp_wrap

from ._branch_points import estimate_branch_points, plot_branch_points
from ._atoms import detect_atoms, evolve_atoms
from ._support import estimate_support

from .._support import supp as estimate_broad_supp
from ._decompressible import precheck_laurent
from ._decompress_util import build_time_grid, inverse_stieltjes
# from ._decompress_coeffs import decompress_coeffs, plot_decompress_candidates
from ._decompress_coeffs2 import decompress_coeffs, plot_decompress_candidates
from ._deform_coeffs2 import deform_coeffs, plot_deform_candidates
from ._decompress_debug import plot_decompress_vs_candidates

# Stieltjes Poly
# from ._stieltjes_poly1 import StieltjesPoly  # 1D horizontal Viterbi
# from ._stieltjes_poly2 import StieltjesPoly  # 2D vertical + horizontal
from ._stieltjes_poly3 import StieltjesPoly    # 1D vertical, parallel

# Decompress with Newton
# from ._decompress6 import decompress_newton
# from ._decompress7 import decompress_newton
# from ._decompress8 import decompress_newton
# from ._decompress9 import decompress_newton   # With Predictor/Corrector
# from ._decompress10 import decompress_newton  # log aware
# from ._decompress10_6 import decompress_newton  # parallel, log
# from ._decompress10_7 import decompress_newton  # parallel, log
from ._decompress10_6d import decompress_newton  # pair mode, newton at switch

# Deform
# from ._deform5 import deform_newton  # predictor-corrector
from ._deform10_6d import deform_newton  # based on _decompress10_6d, _deform5

from ._moments import Moments, AlgebraicStieltjesMoments
from ..visualization._plot_util import plot_density, plot_hilbert, \
    plot_stieltjes

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

    n : int, default=None
        When ``A`` is the eigenvalues of a matrix (not the matrix itself), and
        if the number of these eigenvalues are more than the intended matrix
        size (such as then the eigenvalues are over-sampled from multiple
        submatrix sizes), then the user should provide the actual size of the
        intended matrix using this argument.

    ratio : float, default=None
        If ``A`` is a Gram (covariance) matrix that is formed by
        :math:`\\mathbf{A} = \\mathbf{X} \\mathbf{X}^{\\intercal}` with
        the rectangular matrix
        :math:`\\mathbb{X} \\in \\mathbb{R}^{p \\times n}`, then this
        argument is the aspect ratio :math:`c = p / n \\in (0, \\infty)`.
        The argument is only used for deformed decompression.

    log : bool, default=False
        If `True`, it is assumed the spectral density is positive-definite and
        its support is best represented in logarithmic scale.

    dtype : {``'complex128'``, ``'complex256'``}, default = ``'complex128'``
        Data type for inner computations of complex variables:

        * ``'complex128'``: 128-bit complex numbers, equivalent of two double
          precision floating point.
        * ``'complex256'``: 256-bit complex numbers, equivalent of two long
          double precision floating point. This option is only available on
          Linux machines.

    stieltjes_opt : dict, default={}
        Dictionary of options to configure the computation of Stieltjes
        transform from the fitted polynomial. These options are passed to
        ``StieltjesPoly`` class.

    inv_stieltjes_opt : dict, default={}
        Dictionary of options for computing density using Plemelj formula (
        inverse Stieltjes). These options are passed to ``inverse_stieltjes``
        function. The dictionary include (but not limited to) these keys:

        * ``delta`` : float, default=1e-5
              Imaginary offset into the upper half-plane used to evaluate the
              Stieltjes transform for density recovery by the inverse Stieltjes
              transform (Plemelj formula).
        * ``delta_ladder_ratio`` : float, default=2.0
              Geometric ratio of the imaginary offsets used for multi-level
              density recovery. The offsets are defined by
              :math:`\\delta_j = \\delta r^j`, where :math:`r` is this
              argument.
        * ``delta_ladder_size`` : int, default=1
              Number of imaginary offsets in the geometric ladder used for
              multi-level density recovery.

    supp_opt : dict, default={}
        Dictionary of parameters to pass to :func:`supp` function when
        ``support=None``.

    Attributes
    ----------

    eig : numpy.array
        Eigenvalues of the matrix

    n : int
      Initial array size (assuming a square matrix when :math:`\\mathbf{A}` is
      2D).

    ratio : float
        The aspect ratio of Gram matrix

    coeffs : numpy.ndarray
        The coefficients of the fitted polynomial. This attribute is available
        after calling :func:`fit`.

    supp: tuple
        The predicted (or given) support :math:`(\\lambda_{\\min},
        \\lambda_{\\max})` of the eigenvalue density.

    broad_supp : tuple
        The tuple :math:`(\\lambda_{-}, \\lambda_{+})` indicating the largest
        and smallest eigenvalues (considering all bulks).

    est_supp : list of tuples
        A list of tuples ``(a, b)`` for the spectral edges of the bulks. This
        attribute is only available after calling :func:`support`. These
        spectral edges are estimated form the fitted polynomial.

    _stieltjes_poly : callable function
        A function that computes Stieltjes transform from the fitted polynomial
        using homotopy method.

    _moments : callable function
        A function that computes evolving moments.

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
        Evaluate spectral density of the fitted spectral curve.

    hilbert
        Compute Hilbert transform of the spectral density

    stieltjes
        Compute Stieltjes transform of the spectral density

    decompress
        Free or deformed decompression of spectral density

    debug_decompress
        Plots decompressed root versus all candidates over time

    candidates
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

    **Generate a matrix:**

    Before working with the :class:`freealg.AlgebraicForm`, we generate a
    matrix. This matrix can be from your empirical data. Here, we generate it
    from Levy distribution (see :class:`freealg.distributions.FreeLevy`):

    .. code-block:: python

        >>> # Create a distribution
        >>> from freealg.distributions import FreeLevy
        >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
        ...               sigma=0.0)

        >>> # Generate a matrix realization of the distribution
        >>> A = fl.matrix(size=6000, seed=0)

        >>> # Sample from this matrix
        >>> As = freealg.submatrix(A, size=1000, seed=0)

    *Create AlgebraicForm object:**

    We now use this sampled matrix ``As`` to create an algebraic form. The goal
    is to predict the ESD of ``A`` (the larger matrix of size 6000) from the
    ESD of ``As`` (the smaller matrix of size 1000).

    Since the matrix is generated from the Levy distribution, we now its
    algebraic structure :math:`P(z, m)` has degrees
    :math:`d_z = \\deg_z(P) = 1` and :math:`d_m = \\deg_m(P) = 3` when
    :math:`\\sigma = 0`.

    .. code-block:: python

        >>> from freealg import AlgebraicForm

        >>> # Fit polynomial P with degrees 1 (in z) and 3 (in m)
        >>> coeffs = af.fit(deg_m=3, deg_z=1, reg=0, verbose=True)
        fit residual max  : 2.3763e-01
        fit residual 99.9%: 1.7992e-02

        Coefficients (real)
        +0.99999637 +0.95497991 +0.12108988 +0.00000000
        +0.00000000 +1.00020387 +1.24161671 +0.30272471

        Coefficients (imag) norm: 0.0000e+00

        Stieltjes sanity check: OK

    **Find properties of the algebraic structure:**

    Once fit, we can inquiry the coefficients of the polynomial, support,
    branch points, etc, of the underlying distribution

    .. code-block:: python

        >>> # Coefficients of the fitted polynomial
        >>> af.coeffs.real
        array([[9.999963e-01, 9.5497991e-01, 1.21089884e-01, 4.76107138e-14],
              [0.0000000e+00, 1.00020387e+00, 1.2416167e+00, 3.02724712e-01]])

        # Support of the distribution of the ESD of As
        >>> est_supp = af.support()
        >>> print(est_supp)
        [(0.019606029040223762, 1.9518480572571149)]

        >>> # Atoms of distribution of the ESD of As
        >>> # Output format is the tuple [(atom_location, atom_weight)]
        >>> atoms = af.atoms()
        >>> print(atoms)
        [(-1.572739585567635e-13, 0.3999999986265035)]

        >>> # Plot branch points
        >>> bp = af.branch_points(tol=1e-16, real_tol=1e-16, plot=True)
        >>> print(bp)
        [ 1.95184806e+00+0.j          6.99218919e-01+0.08189017j
          6.99218919e-01-0.08189017j  1.96060290e-02+0.j
          -1.12675884e-13+0.j        ]

    **Decompress:**

    We now apply :func:`AlgebraicForm.decompress` to predict the ESD of the
    larger matrix ``A`` of size 6000. Before this, we can check if this matrix
    is decompressible:

    .. code-block:: python

        >>> status = af.is_decompressible()
        >>> print(status)
        True

        >>> # Decompress from the size 1000 of As to 6000 of A with several
        >>> # intermediate sizes
        >>> x = numpy.linspace(-1, 8, 300)
        >>> sizes = numpy.arange(As.shape[0], A.shape[0]+1, 500)
        >>> rho, x, atoms = af.decompress(
        ...     sizes, x=x, method='moc', min_n_times=100, newton_opt={},
        ...     return_atoms=True, atom_eps=1e-2, verbose=False, plot=True)
    """

    # ====
    # init
    # ====

    def __init__(self, A, support=None, n=None, ratio=None, log=False,
                 dtype='complex128', stieltjes_opt={}, inv_stieltjes_opt={},
                 supp_opt={}):
        """
        Initialization.
        """

        super().__init__(log=log, dtype=dtype, stieltjes_opt=stieltjes_opt,
                         inv_stieltjes_opt=inv_stieltjes_opt)

        if ratio is not None:
            self.ratio = float(ratio)
        else:
            self.ratio = None
        self._stieltjes_poly = None  # Stieltjes from fitted polynomial
        self.eig = None
        self._moments = None
        self.supp = support
        self.est_supp = None  # Estimated from polynomial after fitting

        if hasattr(A, 'stieltjes') and callable(getattr(A, 'stieltjes', None)):
            # This is one of the distribution objects, like MarchenkoPastur
            self._stieltjes_emp_func = A.stieltjes
            self.supp = A.support()
            self.n = n if n is not None else 1

        elif callable(A):
            # This is a custom function
            self._stieltjes_emp_func = A
            self.n = n if n is not None else 1

        else:
            # Eigenvalues
            if A.ndim == 1:
                # If A is a 1D array, it is assumed A is the eigenvalues array.
                self.eig = A

                # In exceptional case when the input array "A" is not the
                # matrix but the eigenvalues of the matrix, and if the number
                # of eigenvalues is not the same as the intended matrix size,
                # such as when eigenvalues are over-sampled from multiple
                # submatrix samples, then the length of self.eig is not
                # necessarily n. In such case, the user should provide the
                # intended size n.
                self.n = n if n is not None else len(A)

            elif A.ndim == 2:
                # When A is a 2D array, it is assumed A is the actual array,
                # and its eigenvalues will be computed.
                self.A = A

                if self.n is not None:
                    raise ValueError('When "A" is a matrix, "n" should not be '
                                     'given.')
                else:
                    self.n = A.shape[0]

                assert A.shape[0] == A.shape[1], \
                    'Only square matrices are permitted.'
                self.eig = compute_eig(A)

            # This will be defined in _stieltjes_emp
            self._stieltjes_emp_func = None
            self._moments = Moments(self.eig)

        # Check eigenvalues to be positive when log is True
        if (self.eig is not None) and (self._log is True):
            if numpy.any(self.eig <= 0):
                raise ValueError('Eigenvalues are not all positive. This '
                                 'conflicts with setting "log=True".')

        # broad support
        if self.supp is None:
            if self.eig is None:
                raise RuntimeError("Support must be provided without data")

            # Detect support
            self.lam_m, self.lam_p = \
                estimate_broad_supp(self.eig, **supp_opt)[0]
            self.broad_supp = (float(self.lam_m), float(self.lam_p))
        else:
            self.lam_m = float(min([s[0] for s in self.supp]))
            self.lam_p = float(max([s[1] for s in self.supp]))
            self.broad_supp = (self.lam_m, self.lam_p)

        # Initialize
        self.coeffs = None                 # Polynomial coefficients

    # =============
    # stieltjes emp
    # =============

    def _stieltjes_emp(self, z, max_mem=8.0, safety_factor=3.0):
        """
        Compute empirical Stieltjes transform with automatic chunking based on
        a memory budget.

        Parameters
        ----------

        z : array_like
            Complex query points.

        max_mem : float, default=0.5
            Maximum temporary memory budget in GB used for chunked evaluation.

        safety_factor : float, default=3.0
            Safety multiplier to account for intermediate NumPy temporaries
            beyond the main broadcasted array.

        Returns
        -------

        m : numpy.ndarray
            Empirical Stieltjes transform evaluated at `z`.
        """

        if self._stieltjes_emp_func is not None:
            m = self._stieltjes_emp_func(z)

        elif self.eig is not None:

            z = numpy.asarray(z, dtype=self.dtype).ravel()
            m = numpy.empty(z.size, dtype=self.dtype)

            n_eig = self.eig.size
            itemsize = numpy.dtype(self.dtype).itemsize

            # Convert GB to bytes
            max_bytes = float(max_mem) * (1024 ** 3)

            # Estimated bytes per z-point in a chunk
            bytes_per_point = safety_factor * n_eig * itemsize

            # At least one point per chunk
            chunk_size = max(1, int(max_bytes // bytes_per_point))

            for start in range(0, z.size, chunk_size):
                stop = min(start + chunk_size, z.size)
                zc = z[start:stop]
                m[start:stop] = numpy.mean(
                    1.0 / (self.eig[None, :] - zc[:, None]), axis=1)

        else:
            raise RuntimeError('Neither "_stieltjes_emp_func" nor "eig" is '
                               'defined.')

        return m

    # ===
    # fit
    # ===

    def fit(self, deg_m, deg_z, reg=0.0, r_min=1.8, r_max=2.2, n_r=5,
            y_scale=1.0, gamma=1.0,  n_samples=4096, cut_eps=0.01,
            triangular=None, mu='auto', mu_reg=None, normalize=True,
            verbose=False, plot=False):
        """
        Fit an algebraic structure to the input data.

        Parameters
        ----------

        deg_m : int
            :math:`\\mathrm{deg}_m(P)`: degree polynomial :math:`P(z, m)` in
            :math:`m`.

        deg_z : int
            :math:`\\mathrm{deg}_z(P)`: degree polynomial :math:`P(z, m)` in
            :math:`z`.

        reg : float, default=0.0
            Tikhonov regularization parameter for fitting.

        r_min : array_like, default=1.8
            Minimum radius in Joukowski plane where contours around support
            intervals are generated.

        r_max : array_like, default=1.8
            Maximum radius in Joukowski plane where contours around support
            intervals are generated.

        n_r : array_like, default=1.8
            Number of radii in Joukowski plane where contours around support
            intervals are generated.

        y_scale : float, default=1.0
            The scale of y for sampling points in z plane.

        gamma : float, default=1.0
            The exponent in which y components of sampling points increase (
            only in log-scale).

        n_samples : int, default=4.96
            Number of angular samples on each Bernstein ellipse from
            :math:`\\theta = 0` to :math:`\\theta = 2 \\pi`.

        cut_eps : float, default=0.01
            Fraction of cut width used as the distance threshold.

            * log=False: threshold is ``cut_eps * (b - a)``, where ``(b - a)``
              is the linear width of the cut interval.

            * log=True: threshold is ``cut_eps * log(b / a)``, where
              ``log(b / a)`` is the logarithmic width of the cut interval.

        triangular : {None, tuple}, default=None
            Structure of the coefficient index set.

            * ``None``: full rectangular index set.
            * ``(a, b)``: banded support defined by
              :math:`a \\le j - i \\le b`, where each of ``a`` and ``b`` may be
              an integer or ``None``.

              - ``a=None`` means there is no lower bound on :math:`j-i`.
              - ``b=None`` means there is no upper bound on :math:`j-i`.

            Examples:

            * ``(0, None)`` keeps terms with :math:`j \\ge i`.
            * ``(-E, None)`` reproduces the old upper-Hessenberg case
              :math:`i \\le j + E`.
            * ``(a, b)`` with both finite keeps only the two-sided band
              :math:`a \\le j-i \\le b`.

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
            Regularization for applying the moments ``mu``:

            * If `None`, the constraints ``mu`` are applied as hard constraint.
            * If a positive number, the constraints are applied as a soft
              constraints with regularisation ``mu_reg``.
            * If zero, no moment constraint (hard or soft) is applied.

        normalize : bool, default=True
            If `True`, the coefficients are scaled so that ``coeff[0, 0] = 1``.
            This improves the conditioning of the coefficients.

        verbose : bool, default=False
            If `True`, debugging info is printed.

        plot : bool, default=False
            If `True`, some diagnostic plots will be shown.

        Returns
        -------

        coeffs : numpy.ndarray
            A 2D array of the size :math:`(d_z, d_m)` where :math:`d_z` and
            :math:`d_m` are the degrees of :math:`P(z, m)` in :math:`z` and
            :math:`m` respectively.

        See Also
        --------

        support
        branch_points
        atoms
        stieltjes

        Notes
        -----

        The Stieltjes transform :math:`m = m(z)` is the root of the implicit
        relation :math:`P(z, m) = 0`, where :math:`P` is given by

        .. math::

            P(z, m) = \\sum_{i=1}^{d_z} \\sum_{j=1}^{d_m} c_{ij} z^i m^j,

        and :math:`d_z`, :math:`d_m` are the degrees of :math:`P` in :math:`z`
        and :math:`m`, respectively.

        This function returns the :math:`(d_z \\times d_m)` matrix
        :math:`\\mathbf{C} = [c_{ij}]`

        .. math::

            \\mathbf{C} =
            \\begin{bmatrix}
                c_{11} & \\dots & c_{1 d_m} \\\\
                \\vdots & & \\vdots \\\\
                c_{d_z 1} & \\dots & c_{d_z d_m}
            \\end{bmatrix}.

        These coefficients are real, however, they are returned as complex
        numbers with their imaginary part all being zero.

        This matrix is empirically fitted based on the samples of the Stieltjes
        transform. To get this matrix, the :func:`fit` function should have
        been called first.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 13,16

            >>> # Create a distribution
            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=0.9)

            >>> # Create an algebraic form object from the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(fl)

            >>> # Fit polynomial
            >>> af.fit(deg_m=4, deg_z=1)

            >>> # Get the empirically fitted polynomial coefficients
            >>> print(af.coeffs.real)
            [[ 1.      7.2125 12.15   16.875  24.75  ]
             [ 0.      1.      7.5    11.      0.    ]]

            >>> # Compare with the exact polynomial from the distribution law
            >>> print(fl.poly().real)
            [[ 1.      7.2125 12.15   16.875  24.75  ]
             [ 0.      1.      7.5    11.     -0.    ]]
            """

        # Sampling around support, or broad_support. This is only needed to
        # ensure sampled points are not hitting the support itself is not used
        # in any computation. If support is not known, use broad support.
        if self.supp is not None:
            possible_supp = self.supp
        else:
            possible_supp = [self.broad_supp]

        # Sampling points for fitting
        z_fit = sample_z_joukowski(possible_supp, n_samples=n_samples,
                                   r_min=r_min, r_max=r_max, n_r=n_r,
                                   y_scale=y_scale, gamma=gamma, log=self._log,
                                   dtype=self.dtype)

        # Remove points too close to any cut
        z_fit = filter_z_away_from_cuts(z_fit, possible_supp, cut_eps=cut_eps,
                                        log=self._log)

        # Automatically add mu constraints from eigenvalues
        if isinstance(mu, str) and (mu == 'auto'):
            if self.eig is not None:
                mu_0 = 1.0
                mu_1 = numpy.mean(self.eig)
                mu_2 = numpy.mean(self.eig**2)
                mu = [mu_0, mu_1, mu_2]
            else:
                mu = None

        # Fitting (w_inf = None means adaptive weight selection)
        m1_fit = self._stieltjes_emp(z_fit)
        self.coeffs, fit_metrics = fit_polynomial_relation(
                z_fit, m1_fit, s=deg_m, deg_z=deg_z, ridge_lambda=reg,
                weights=None, triangular=triangular, normalize=normalize,
                mu=mu, mu_reg=mu_reg, dtype=self.dtype)

        # Stieltjes transform from fitted polynomial (not from empirical eigs)
        self._stieltjes_poly = StieltjesPoly(
            self.coeffs, stieltjes_opt=self.stieltjes_opt,
            stieltjes_emp=self._stieltjes_emp, dtype=self.dtype)

        # Estimate support from the fitted polynomial
        # self.est_supp = self.support(self.coeffs)

        self._moments_base = AlgebraicStieltjesMoments(self.coeffs)
        self.moments = Moments(self._moments_base)

        if verbose:

            # Reporting error
            P_res = numpy.abs(eval_P(z_fit, m1_fit, self.coeffs))
            res_max = numpy.max(P_res[numpy.isfinite(P_res)])
            res_99_9 = numpy.quantile(P_res[numpy.isfinite(P_res)], 0.999)

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

        if plot:
            edges = [[sup[0], sup[1]] for sup in possible_supp]
            edges = numpy.concatenate(edges)

            import matplotlib.pyplot as plt
            import texplot
            with texplot.theme(use_latex=False):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(z_fit.real, z_fit.imag, 'o', color='black', ms=0.25,
                        label='Sample point')
                ax.plot(edges.real, edges.imag, 'o', color='maroon',
                        label='Spectral edge')

                if self._log:
                    ax.set_xscale('log')
                    ax.set_yscale('symlog', linthresh=1e-5)

                ax.set_title('Sampling Stieltjes Transform')
                ax.set_xlabel(r'$\mathrm{Re}(z)$')
                ax.set_ylabel(r'$\mathrm{Im}(z)$')
                ax.legend(fontsize='x-small')
                ax.grid()
                plt.show()

        return self.coeffs

    # =======
    # support
    # =======

    def support(self, scan_range=None, n_scan=1024, refine=True,
                resplit_density=16, merge_threshold=0.0, thr_rel=1e-4,
                min_log_width_mult=1.0, return_info=False, **kwargs):
        """
        Estimate the spectral edges of the density.

        Parameters
        ----------

        scan_range : tuple, default=None
            A range ``(x_min, x_max)`` on the real axis to scan for the
            spectral edge points. If `None`, an interval is considered is used
            based on an initial broad support guess from the minimum and
            maximum of the eigenvalues.

        n_scan : int, default=1024
            Number of points to scan along the ``scan_range`` interval.

        refine : bool, default=True
            Refine each detected edge by a safeguarded Newton solve of the
            local threshold-crossing equation.

        resplit_density : int, default=16
            Densification factor used to re-check each coarse run for touching
            bulks that may be missed by the coarse scan.

        merge_threshold : float, default=0.0
            Merge adjacent detected bulks if the gap is smaller than this
            value. In log mode it is measured in log-x.

        thr_rel : float, default=1e-4
            Linear-mode relative threshold. In log mode the threshold is
            selected automatically from the data.

        min_log_width_mult : float, default=1.0
            Minimum accepted run width in units of log-grid spacing.

        return_info : bool, default=False
            If `True`, debug info is also returned.

        Returns
        -------

        support : list of tuples
            A list ``[(a1, b1), ..., (ak, bk)]``.

        See Also
        --------

        fit
        branch_points
        atoms

        Notes
        -----

        Support is a list of tuples ``[(a_1, b_1), ..., (a_k, b_k)]``
        where each represent a bulk interval :math:`I_j = (a_j, b_j)`. The
        support of the absolutely-continuous part of the density is then

        .. math::

            I = \\bigcup_{j=1}^k I_j

        The points :math:`a_j` and :math:`b_j` are the spectral edges. Spectral
        edges are part of the branch points of the spectral curve (see
        :func:`branch_points`).

        There are two support variables used in the
        :class:`freealg.AlgebraicForm` class:

        * The ``self.supp`` attribute which is given by the user through
          ``support`` argument when creating the class object.

        * The ``self.est_supp``, which is estimated from the fitted polynomial
          through this function (:func:`support`). All downstream computations
          such as decompression, deformation, etc, uses this variable.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 18

            >>> # Create a distribution with two bulks
            >>> from freealg.distributions import CompoundFreePoisson
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Find the exact support using the distribution (no fitting)
            >>> supp = cfp.support()
            >>> print(supp)
            [(0.9984996249062267, 3.13165791447862),
             (4.157389347336835, 7.597674418604652)]

            >>> # Create AlgebraicForm and fit the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(cfp)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Estimate the support (using the fitted polynomial)
            >>> est_supp = af.support()
            >>> print(est_supp)
            [(1.0055181596689498, 3.1256892314391664),
             (4.162723779878648, 7.595354543299085)]
        """

        # Default kwargs
        # kwargs.setdefault("smooth_width", 7)

        if self._stieltjes_poly is None:
            raise RuntimeError('Call "fit" first.')

        # Inflate a bit to make sure all points are searched
        if scan_range is not None:
            x_min, x_max = scan_range

            if (self._log is True) and (x_min <= 0.0):
                raise ValueError('"x_min" cannot be non-positive when '
                                 '"log=True"')
        else:
            x_min, x_max = self._inflate_broad_supp(inflate=0.2)

        est_supp, info = estimate_support(
            self._stieltjes_poly, x_min=x_min, x_max=x_max, n_scan=n_scan,
            refine=refine, resplit_density=resplit_density,
            merge_threshold=merge_threshold, thr_rel=thr_rel,
            min_log_width_mult=min_log_width_mult, log=self._log,
            delta=self.delta, **kwargs)

        self.est_supp = est_supp

        if return_info:
            return est_supp, info
        else:
            return est_supp

    # =============
    # branch points
    # =============

    def branch_points(self, tol=1e-15, real_tol=None, plot=False, latex=False,
                      save=False, return_info=False):
        """
        Compute global branch points.

        Parameters
        ----------

        tol : float, default=1e-15
            Tolerance of convergence

        real_tol : float, default=None
            Tolerance for treating a complex root as real. If None, uses
            ``1e3*tol``.

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

        return_info : bool, default=False
            If `True`, debugging info is also returned.

        See Also
        --------

        support
        atoms

        Notes
        -----

        This function solves the discriminant equation to obtain branch points
        :math:`z_{\\ast}`. The discriminant :math:`\\Delta(P)` is solved by

        .. math::

            P(z_{\\ast}, m_{\\ast}) = 0,
            \\qquad
            \\partial_m P(z_{\\ast}, m_{\\ast}) = 0.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12

            >>> # Create a distribution with two bulks
            >>> from freealg.distributions import CompoundFreePoisson
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Create AlgebraicForm and fit the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(cfp)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Estimate the branch points (using the fitted polynomial)
            >>> bp = af.branch_points()
            >>> print(bp)
            [6.93519593e+00 3.83043179e+00 3.11397655e+00 1.00372907e+00
             4.39730918e-15]
        """

        if self.coeffs is None:
            raise RuntimeError('Call "fit" first.')

        bp, info = estimate_branch_points(
            self.coeffs, tol=tol, real_tol=real_tol)

        if plot:
            atoms_list = self.atoms()
            est_supp = self.support()
            plot_branch_points(bp, atoms_list, est_supp, log=self._log,
                               latex=latex, save=save)

        if return_info:
            return bp, info
        else:
            return bp

    # =====
    # atoms
    # =====

    def atoms(self, eta=1e-6, tol=1e-12, real_tol=None, w_tol=1e-10,
              merge_tol=1e-8):
        """
        Detect atom locations and weights of distribution

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
            List of tuples of the form ``(atom_loc, atom_w)``. Locations are
            real numbers and weights are nonnegative.

        See Also
        --------

        branch_points
        support

        Notes
        -----

        This routine uses the necessary condition for a finite pole

        .. math::

            a_s(z_0) = 0,

        where :math:`a_s(z)` is the leading coefficient of :math:`P` in powers
        of :math:`m`. Candidate atom locations are the (nearly) real roots of
        :math:`a_s(z)`. The atom weight is estimated numerically from the
        Stieltjes transform as

        .. math::

            w = \\eta \\, \\Im(m(z_0 + i \\eta)),

        which follows from :math:`m(z) \\sim -w/(z - z_0)` near an atom at
        :math:`z_0`.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 14

            >>> # Create a distribution with two bulks
            >>> from freealg.distributions import CompoundFreePoisson
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Create AlgebraicForm and fit the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(cfp)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Estimate the atoms (using the fitted polynomial). This
            >>> # distribution has an atom at x=0 with 90% mass of the total
            >>> # spectral density.
            >>> atoms = af.atoms()
            >>> print(atoms)
             [(1.2538385955639516e-16, 0.9000000001406558)]
        """

        if self.coeffs is None:
            raise RuntimeError('Call "fit" first.')

        atoms_list = detect_atoms(self.coeffs, self._stieltjes_poly, eta=eta,
                                  tol=tol, real_tol=real_tol, w_tol=w_tol,
                                  merge_tol=merge_tol)

        return atoms_list

    # =======
    # density
    # =======

    def density(self, x=None, eta=2e-4, ac_only=False, plot=False, latex=False,
                save=False):
        """
        Evaluate spectral density of the fitted spectral curve.

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

        Notes
        -----

        In the density plot (assuming `plot=True`), the solid curve shows the
        absolutely-continuous part of the spectral density. If the density has
        atom(s), they are shown as an arrow, where the hight of the arrow is
        proportional to its mass, and can be read from the right ordinate of
        the plot.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 14

            >>> # Create a distribution with two bulks
            >>> from freealg.distributions import CompoundFreePoisson
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Create AlgebraicForm and fit the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(cfp)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Plot the density of the fitted spectral curve
            >>> import numpy
            >>> x = numpy.linspace(0, 8, 1000)
            >>> rho = af.density(x, plot=True)
        """

        if self.coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25, log=self._log)

        # Stack of all stieltjes compacted over a ladder of delta
        m_stack = numpy.zeros((self.delta_ladder.size, x.size),
                              dtype=self.dtype)

        # Compute m over a ladder of delta
        for i in range(self.delta_ladder.size):
            z = x.astype(complex) + 1j * self.delta_ladder[i]
            m_stack[i, :] = self._stieltjes_poly(z)

        # Inverse Stieltjes transform
        rho = inverse_stieltjes(m_stack, self.delta_ladder, x=x, log=self._log,
                                nonnegative=True, **self.inv_stieltjes_opt)

        rho_ac = rho
        atoms_list = self.atoms()
        if len(atoms_list) > 0:
            for atom_loc, atom_w in atoms_list:

                # Mollifier to approximate atom function
                atom_mollifier = (atom_w * eta / (numpy.pi)) / \
                    ((x - atom_loc)**2 + eta**2)
                rho_ac = rho_ac - atom_mollifier
            rho_ac = numpy.maximum(rho_ac, 0.0)

            if (ac_only is True):
                rho = rho_ac

        # Remove densities near Poisson kernel delta floor to zero
        # if self._log:
        #     kernel_floor = (self.delta / numpy.pi) / (self.delta**2 + x**2)
        #     factor = 5.0
        #     rho[rho < factor * kernel_floor] = 0.0

        if plot:
            # Pass a copy of rho since in plot function it's zero values will
            # be set to nan.
            plot_density(x, numpy.copy(rho), eig=self.eig, atoms=atoms_list,
                         support=self.est_supp, label='Estimate',
                         log=self._log, latex=latex, save=save)

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

        Notes
        -----

        The Hilbert transform of s spectral density is

        .. math::

            H(x) = \\mathcal{H}[\\rho](x) = \\frac{1}{\\pi} \\mathrm{p.v.}
            \\int_{\\mathbb{R}} \\frac{\\rho(y)}{x - y}\\, \\mathrm{d}y.

        It can be directly computed from the non-tangential limit of the
        Stieltjes transform by

        .. math::

            H(x) = -\\pi \\, \\lim_{\\epsilon \\to 0^{+}}
            \\Re(m(x + i \\epsilon)).

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 14

            >>> # Create a distribution with two bulks
            >>> from freealg.distributions import CompoundFreePoisson
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Create AlgebraicForm and fit the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(cfp)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Plot the density of the fitted spectral curve
            >>> import numpy
            >>> x = numpy.linspace(0, 8, 1000)
            >>> hilb = af.hilbert(x, plot=True)
        """

        if self.coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25, log=self._log)

        # Preallocate density to zero
        hilb = -self._stieltjes_poly(x).real / numpy.pi

        if plot:
            plot_hilbert(x, hilb, support=self.broad_supp, log=self._log,
                         latex=latex, save=save)

        return hilb

    # =========
    # stieltjes
    # =========

    def stieltjes(self, x=None, y=None, plot=False, latex=False, save=False):
        """
        Compute Stieltjes transform of the spectral density

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
        plot_branches

        Notes
        -----

        The Stieltjes transform is defined by

        .. math::

            m(z) = \\int_{\\mathbf{R}} \\frac{\\rho(x)}{x - z} \\,
            \\mathrm{d}x.

        The Stieltjes transform is a Herglotz function, meaning

        .. math::

            \\Im(m(z)) > 0,
            \\quad
            z \\in \\mathbb{C}^{+}.

        Also, it satisfies normalization at infinity:

        .. math::

            m(z) = -\\frac{1}{z},
            \\qquad
            \\vert z \\vert \\to \\infty.

        This function evaluates Stieltjes transform on an array of points, or
        over a 2D Cartesian grid on the complex plane.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 15

            >>> # Create a distribution with two bulks
            >>> from freealg.distributions import CompoundFreePoisson
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Create AlgebraicForm and fit the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(cfp)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Plot the density of the fitted spectral curve
            >>> import numpy
            >>> x = numpy.linspace(0, 8, 100)
            >>> y = numpy.linspace(-2, 2, 100)
            >>> m = af.stieltjes(x, y, plot=True)
        """

        if self.coeffs is None:
            raise RuntimeError('The model needs to be fit using the .fit() ' +
                               'function.')

        # Create x if not given
        if x is None:
            x = self._generate_grid(2.0, extend=2.0, log=self._log)[::2]

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

        m = self._stieltjes_poly(z)

        if plot:
            plot_stieltjes(x, y, m, m, self.broad_supp, latex=latex,
                           save=save)

        return m

    # ==========
    # decompress
    # ==========

    def decompress(self, size, x=None, kind='free', method='moc',
                   atom_eps=None, return_atoms=False, min_n_times=10,
                   newton_opt={'max_iter': 50, 'tol': 1e-12, 'armijo': 1e-4,
                               'min_lam': 1e-6, 'w_min': 1e-14},
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

        kind : {``'free'``, ``'deformed'``}, default= ``'free'``
            The type of operation:

            * ``'free'``: evolve the spectral curve using free decompression
            * ``'deformed'``: evolve the spectral curve using deformed
              deformation.

        method : {``'moc'``, ``'coeffs'``}, default= ``'moc'``
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
        edge
        cusp

        Notes
        -----

        Free or deformed Decompression.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 15

            >>> # Create a distribution with two bulks
            >>> from freealg.distributions import CompoundFreePoisson
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Create AlgebraicForm and fit the distribution
            >>> from freealg import AlgebraicForm
            >>> af = AlgebraicForm(cfp)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Plot the density of the fitted spectral curve
            >>> import numpy
            >>> x = numpy.linspace(0, 8, 500)
            >>> y = numpy.linspace(-2, 2, 300)
            >>> m = af.stieltjes(x, y, plot=True)
        """

        if kind not in ['free', 'deformed']:
            raise ValueError('"kind" should be "free" or "deformed".')

        if kind == 'deformed':
            if self.ratio is None:
                raise ValueError(
                    'In "deformed" kind, "ratio" must be provided.')

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

        # Create x if not given
        if x is None:
            x = self._generate_grid(1.25, log=self._log)
        else:
            x = numpy.asarray(x)

        # Epsilon-neighborhood to exclude atoms from x
        if atom_eps is None:
            atom_eps = 50.0 * float(self.delta)

        # Evolve atoms
        atoms0 = self.atoms()
        atoms_t = evolve_atoms(atoms0, alpha)

        # Remove points too close to atoms to avoid Newton stall at poles
        near_atom = numpy.zeros(x.size, dtype=bool)
        if len(atoms0) > 0:
            # Exclude x grid near atoms
            for x0, _w0 in (atoms0 or []):
                near_atom |= (numpy.abs(x - float(x0)) <= float(atom_eps))

        # The companion Stieltjes introduces a fake pole at zero, which should
        # theoretically cancel perfectly, nut numerically, it wont.
        if kind == 'deformed':
            zero_eps = 50.0 * float(self.delta)
            near_atom |= (numpy.abs(x) <= zero_eps)

        x_safe = x[~near_atom]

        if method == 'moc':

            # Ensure there are at least min_n_times time t, including requested
            # times, and especially time t = 0
            t_all, idx_req = build_time_grid(
                size, self.n, min_n_times=min_n_times)

            # Primary options (more important to tune)
            newton_opt.setdefault('dt_max', 0.01)
            newton_opt.setdefault('tol', 1e-10)
            newton_opt.setdefault('max_iter', 1000)
            newton_opt.setdefault('parallel', True)
            newton_opt.setdefault('n_jobs', None)  # uses all CPUs
            newton_opt.setdefault('log_mode', self._log)

            # Secondary options (less important to tune)
            newton_opt.setdefault('dt_min', 1e-6)
            newton_opt.setdefault('pred_atol', 1e-8)
            newton_opt.setdefault('pred_rtol', 5e-3)
            newton_opt.setdefault('corr_factor', 5.0)
            newton_opt.setdefault('max_reject', 30)
            newton_opt.setdefault('det_guard', 1e-14)
            newton_opt.setdefault('tol_step', 0.1 * float(newton_opt['tol']))

            # Stack all output indexed by (delta ladder, requested time, x)
            m_stack = numpy.zeros(
                (self.delta_ladder.size, idx_req.size, x_safe.size),
                dtype=self.dtype)

            for i in range(self.delta_ladder.size):
                # Query grid on the real axis + a small imaginary buffer
                if self.inv_stieltjes_opt['z_query_delta'] == 'linear':
                    z_query = x_safe.astype(complex) * \
                        (1.0 + 1j * self.delta_ladder[i])
                elif self.inv_stieltjes_opt['z_query_delta'] == 'const':
                    z_query = x_safe.astype(complex) + \
                        1j * self.delta_ladder[i]
                else:
                    raise ValueError('z_query_delta is invalid.')

                # Initial condition at t = 0 (physical branch)
                w0_list = self._stieltjes_poly(z_query)

                # Remove atom from Stieltjes transform (Experimental)
                # if len(atoms0) > 0:
                #     for x0, w0 in atoms0:
                #         w0_list -= (float(w0) / (float(x0) - z_query))

                # Evolve. Output is Stieltjes m(t_all, x_safe)
                if kind == 'free':
                    m, ok = decompress_newton(z_query, t_all, self.coeffs,
                                              w0_list=w0_list, **newton_opt)
                elif kind == 'deformed':
                    m, ok = deform_newton(z_query, t_all, self.coeffs,
                                          self.ratio, w0_list=w0_list,
                                          **newton_opt)

                # Keep only the requested times
                m_stack[i, :, :] = m[idx_req, :]

                if verbose:
                    print("success rate per t:", ok.mean(axis=1))

            # Inverse Stieltjes transform
            rho_safe = inverse_stieltjes(
                m_stack, self.delta_ladder, x=x_safe, log=self._log,
                # nonnegative=True,
                nonnegative=False,
                **self.inv_stieltjes_opt)

            # Back into full x grid (fill with zero, may not be a good idea)
            rho = numpy.full((idx_req.size, x.size), 0.0, dtype=float)
            rho[:, ~near_atom] = rho_safe

        elif method == 'coeffs':

            if kind == 'free':
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

            elif kind == 'deformed':
                raise NotImplementedError('"coeff" method is not implemented.')

        else:
            raise ValueError('"method" is invalid.')

        # If the input size was only a scalar, return a 1D rho, otherwise 2D.
        if numpy.isscalar(size):
            rho = numpy.squeeze(rho)

        # Plot only the last size
        if plot:

            # Density (absolutely-continuous part) at the last time
            if numpy.isscalar(size):
                rho_last = rho
            else:
                rho_last = rho[-1, :]

            # Atoms at the last time
            if len(atoms0) > 0:
                atoms_last = [(loc, w[-1]) for loc, w in atoms_t]
            else:
                atoms_last = None

            if kind == 'free':
                label = 'Free Decompression'
            elif kind == 'deformed':
                label = 'Deformed Decompression'

            # Plot only the last time of atoms and density
            plot_density(x, rho_last, atoms=atoms_last, support=None,
                         label=label, log=self._log, latex=latex, save=save)

        if return_atoms:
            return rho, x, atoms_t
        else:
            return rho, x

    # ================
    # debug decompress
    # ================

    def debug_decompress(self, sizes, x, min_n_times=10, newton_opt=None,
                         t_lim=None, re_lim=None, im_lim=None, pm_lim=None):
        """
        Plots tracked root at for point x by comparing the result from
        decompression and the result from all candidate roots.
        """

        t_all, roots, w, eta = \
            plot_decompress_vs_candidates(self, decompress_newton, sizes,
                                          x, min_n_times=min_n_times,
                                          newton_opt=newton_opt, t_lim=t_lim,
                                          re_lim=re_lim, im_lim=im_lim,
                                          pm_lim=pm_lim)

        return t_all, roots, w, eta

    # ==========
    # candidates
    # ==========

    def candidates(self, size, kind='free', x=None, eig=None, delta=None,
                   markersize=1, ylim=None, latex=False, verbose=False):
        """
        Candidate densities of free decompression from all possible roots

        Parameters
        ----------

        size : int
            The size of matrix to compute the candidate roots of its spectral
            curve.

        kind : {``'free'``, ``'deformed'``}, default= ``'free'``
            The type of operation:

            * ``'free'``: evolve the spectral curve using free decompression
            * ``'deformed'``: evolve the spectral curve using deformed
              deformation.

        x : array_like of float, shape (N,)
            1D array of real x-values (evaluation grid).

        eig : numpy.array, default=None
            Eigenvalues to plot as histogram.

        delta : float, optional
            Small positive imaginary offset used to evaluate
            :math:`m(x + i \\delta)`. If `None`, the ``delta`` attribute of the
            object is used.

        markersize : float, default=3
            Marker size of scatter plot.

        ylim : tuple, default=None
            Limits of the y axis. If `None`, it will be automatically set.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX.

        verbose : bool, default=False
            If `True`, it prints verbose be bugging information.

        See Also
        --------

        density

        Notes
        -----

        Roots are solved from the relation:

        .. math::

            P(z, m) = \\sum_{i=1}^I \\sum_{j=1}^J a_{i, j} z^i m^j,

        where :math:`m(z)` is defined implicitly by :math:`P(z, m(z)) = 0`.

        For each grid point :math:`x_k`, set :math:`z = x_k + i \\delta`,
        form the polynomial in :math:`m` given by :math:`P(z, m) = 0`, solve
        for its roots, and plot the cloud of candidate densities:

        .. math::

            \\frac{1}{\\pi} \\Im(m_{\\mathrm{root}}),

        keeping only roots with :math:`\\Im(m_{\\mathrm{root}}) > 0` (roots are
        not tracked/paired across x-values).

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 20

            >>> from freealg import AlgebraicForm
            >>> from freealg.distributions import CompoundFreePoisson
            >>> from freealg import submatrix

            >>> # Create a distribution with two bulks
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Get a matrix realization of the distribution
            >>> A = cfp.matrix(size=4000, seed=0)

            >>> # Compress the matrix to smaller size
            >>> As = submatrix(A, size=2000)

            >>> # Create AlgebraicForm and fit the smaller matrix
            >>> af = AlgebraicForm(As)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Plot all candidate roots at the decompressed size 4000
            >>> af.candidates(size=4000)
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
        m_lb = self._stieltjes_poly(self.lam_m + self.delta * 1j).item()
        m_ub = self._stieltjes_poly(self.lam_p + self.delta * 1j).item()
        hilb_lb = (1.0 / m_lb).real
        hilb_ub = (1.0 / m_ub).real
        lb = self.lam_m - (numpy.max(alpha) - 1) * hilb_lb
        ub = self.lam_p - (numpy.max(alpha) - 1) * hilb_ub

        # Create x if not given
        if x is None:
            if self._log:
                radius = numpy.sqrt(ub / lb)
                center = numpy.sqrt(ub * lb)
                scale = 1.25
                x_min = center / (radius ** scale)
                x_max = center * (radius ** scale)
                x = numpy.geomspace(x_min, x_max, 2000)
            else:
                radius = 0.5 * (ub - lb)
                center = 0.5 * (ub + lb)
                scale = 1.25
                x_min = numpy.floor(center - radius * scale)
                x_max = numpy.ceil(center + radius * scale)
                x = numpy.linspace(x_min, x_max, 2000)
        else:
            x = numpy.asarray(x)

        # Poisson kernel shift
        delta_ = self.delta if delta is None else delta

        for i in range(alpha.size):
            t_i = numpy.log(alpha[i])

            if kind == 'free':
                coeffs_i = decompress_coeffs(self.coeffs, t_i)
                plot_decompress_candidates(coeffs_i, x, eig=eig, delta=delta_,
                                           size=int(alpha[i]*self.n),
                                           log=self._log,
                                           markersize=markersize, ylim=ylim,
                                           latex=latex, verbose=verbose)

            elif kind == 'deformed':
                if self.ratio is None:
                    raise ValueError(
                        '"ratio" must be provided for kind="deformed".')

                # Initial and target ratios (for deformation)
                c_0 = self.ratio
                c_i = c_0 * alpha[i]

                coeffs_i = deform_coeffs(self.coeffs, t_i, c_0)
                plot_deform_candidates(coeffs_i, x, ax=None, c=c_i, eig=eig,
                                       delta=delta_, size=int(alpha[i]*self.n),
                                       log=self._log, markersize=markersize,
                                       ylim=ylim, latex=latex, verbose=verbose)
            else:
                raise ValueError('"kind" should be "free" or "deformed".')

    # =================
    # is decompressible
    # =================

    def is_decompressible(self, ratio=2, n_ratios=5, K=(6, 8, 10), L=3,
                          tol=1e-8, verbose=False, return_info=False):
        """
        Check if the given distribution can be decompressed.

        To this end, this function checks if the evolved polynomial under the
        free decompression admits a valid Stieltjes root.

        .. note::

            This function is not always reliable!

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

        return_info : bool, default=False
            If `True`, debugging info is also returned.

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
        * :math:`\\zeta = z + (1 - \\tau^{-1}) m^{-1}`
        * :math:`P_t(z, m) = P(\\zeta, y)`

        A necessary condition for FD tracking to remain well-posed is that, for
        each :math:`\\tau > 1`), the pushed relation admits a "Stieltjes branch
        at infinity", i.e., a solution branch with the large
        :math:`\\vert z \\vert` behavior

        .. math::

            m(z) = -\\frac{1}{z} + O(z^{-2})

        as :math:`\\vert z \\vert \\to \\infty` in :math:`\\Im(z) > 0`.

        This routine enforces that asymptotic structure by constructing a
        truncated Laurent expansion in :math:`w = 1/z`. It solves for
        coefficients in

        .. math::

            m(z) = -(\\frac{\\alpha}{z} + \\frac{\\mu_1}{z^2} +
            \\frac{\\mu_2|{z^3} + \\dots)

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
            :emphasize-lines: 20

            >>> from freealg import AlgebraicForm
            >>> from freealg.distributions import CompoundFreePoisson
            >>> from freealg import submatrix

            >>> # Create compound free Poisson law
            >>> cfp = CompoundFreePoisson(t=[2.0, 5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Get a matrix realization of the distribution
            >>> A = cfp.matrix(size=4000, seed=0)

            >>> # Compress the matrix to smaller size
            >>> As = submatrix(A, size=2000)

            >>> # Use the distribution above to create an algebraic form
            >>> af = AlgebraicForm(As)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Check the decompressibility of compound free Poisson
            >>> status = af.is_decompressible()

            >>> print(status)
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

        if return_info:
            return status, info
        else:
            return status

    # ====
    # edge
    # ====

    def edge(self, t, kind='free', supp=None, dt_max=0.1, max_iter=30,
             tol=1e-12, verbose=False, plot=False, latex=False, save=False):
        """
        Evolves spectral edges.

        Parameters
        ----------

        t : float or array_like
            Single scalar or an array of time :math:`t`. Edges are evolved at
            these time points.

        kind : {``'free'``, ``'deformed'``}, default= ``'free'``
            The type of operation:

            * ``'free'``: evolve the spectral curve using free decompression
            * ``'deformed'``: evolve the spectral curve using deformed
              deformation.

        supp : list, default=None
            Estimated support of density as a list of tuples. If not given,
            support is estimated from the fitted polynomial.

        dt_max : float, default=0.1
            Maximum time step during the continuous time evolution.

        max_iter : int, default=30
            Maximum number of iterations to solve for each time point.

        tol : float, default=1e-12
            Tolerance of convergence

        verbose : bool, default=False
            If `True`, debugging information is printed.

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

        complex_edges : numpy.ndarray
            A 2D Array of the size ``(n_t, k)``, where ``n_t`` is the length of
            the input time array ``t``, and ``k`` is the maximum number of
            edges. The i-th column of this array is a x coordinate of the i-th
            branch point, which may or may not be a spectral edge. This array
            is complex.

        real_merged_edges : numpy.ndarray
            A 2D array of the same shape as ``complex_edges``, but the elements
            of this array are real part of the previous array. If
            ``complex_edges`` also has non-zero imaginary parts, the
            corresponding element in ``real_merged_edges`` is set to ``nan``,
            since these branch points points are spectral edges.

        active_k : numpy.array
            A 1D array of the size ``n_t``, the length of time ``t``. Each
            element shows the number of active edges at each time point. For
            example, if the total detected number of edges are 4 (two bulks),
            once two edges merge, leading to one bulk, the active number of
            edges become 2.

        Notes
        -----

        This function evolves all branch points that are initially they were
        spectral edges at `t=0`. Once evolved, some edges may leave the real
        axis, in which, they no longer are spectral edges, but still branch
        points. The output array ``complex_edge`` track all these points (as
        complex number output) regardless they remain on the real axis or move
        to the complex plane.

        In contrast, the array ``real_merged_edges`` are only the real part of
        the previous array, and filters out those branch points that cease to
        be spectral edge: they will be set to ``nan``.

        Fix: if ``t`` is a scalar or length-1 array, we prepend ``t=0``
        internally to advances from the initialization at ``t=0``.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 23

            >>> import numpy
            >>> from freealg import AlgebraicForm
            >>> from freealg.distributions import CompoundFreePoisson
            >>> from freealg import submatrix

            >>> # Create a distribution with two bulks
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.0], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Get a matrix realization of the distribution
            >>> A = cfp.matrix(size=4000, seed=0)

            >>> # Compress the matrix to smaller size
            >>> As = submatrix(A, size=2000)

            >>> # Create AlgebraicForm and fit the smaller matrix
            >>> af = AlgebraicForm(As)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Evolve edges corresponding to size 2000 to 4000
            >>> t_final = numpy.log(A.shape[0] / As.shape[0])
            >>> t = numpy.linspace(0, t_final)
            >>> ce, rc, ne = af.evolve_edges(t)
        """

        if kind not in ['free', 'deformed']:
            raise ValueError('"kind" should be "free" or "deformed".')

        if kind == 'deformed':
            if self.ratio is None:
                raise ValueError(
                    'In "deformed" kind, "ratio" must be provided.')

        if supp is not None:
            known_supp = supp
        elif self.est_supp is not None:
            known_supp = self.est_supp
        else:
            known_supp = self.support(return_info=False)

        t = numpy.asarray(t, dtype=float).ravel()

        if t.size == 1:
            t1 = float(t[0])
            if t1 == 0.0:
                t_grid = numpy.array([0.0], dtype=float)

                if kind == 'free':
                    complex_edges, ok_edges = evolve_edges(
                        t_grid, self.coeffs, support=known_supp,
                        stieltjes=self._stieltjes_poly, delta=self.delta,
                        dt_max=dt_max, max_iter=max_iter, tol=tol,
                        log=self._log)
                elif kind == 'deformed':
                    complex_edges, ok_edges = deform_evolve_edges(
                        t_grid, self.coeffs, support=known_supp,
                        stieltjes=self._stieltjes_poly, c0=self.ratio,
                        delta=self.delta, dt_max=dt_max, max_iter=max_iter,
                        tol=tol, log=self._log)
                cusps = []
            else:
                # Use an internal grid so bifurcations (newborn edges) can be
                # detected
                n_internal = 64  # small, but enough to pass cusp/birth
                t_grid = numpy.linspace(0.0, t1, n_internal)

                cusps, cusps_sol = self.cusp(t_grid, kind=kind,
                                             supp=known_supp, return_info=True)

                if kind == 'free':
                    complex_edges2, ok_edges2 = evolve_edges_with_births(
                        t_grid, self.coeffs, support=known_supp,
                        cusps=cusps_sol, stieltjes=self._stieltjes_poly,
                        delta=self.delta, dt_max=dt_max, max_iter=max_iter,
                        tol=tol, split_tol=0.0, seed_eps=1e-6, log=self._log)
                elif kind == 'deformed':
                    complex_edges2, ok_edges2 = \
                            deform_evolve_edges_with_births(
                                t_grid, self.coeffs, support=known_supp,
                                cusps=cusps_sol,
                                stieltjes=self._stieltjes_poly, c0=self.ratio,
                                delta=self.delta, dt_max=dt_max,
                                max_iter=max_iter, tol=tol, split_tol=0.0,
                                seed_eps=1e-6, log=self._log)

                complex_edges = complex_edges2[-1:, :]
                ok_edges = ok_edges2[-1:, :]
        else:
            cusps, cusps_sol = self.cusp(t, kind=kind, supp=known_supp,
                                         return_info=True)

            if kind == 'free':
                complex_edges, ok_edges = evolve_edges_with_births(
                    t, self.coeffs, support=known_supp, cusps=cusps_sol,
                    stieltjes=self._stieltjes_poly, delta=self.delta,
                    dt_max=dt_max, max_iter=max_iter, tol=tol,
                    split_tol=0.0, seed_eps=1e-6, log=self._log)

            elif kind == 'deformed':
                complex_edges, ok_edges = deform_evolve_edges_with_births(
                    t, self.coeffs, support=known_supp, cusps=cusps_sol,
                    stieltjes=self._stieltjes_poly, c0=self.ratio,
                    delta=self.delta, dt_max=dt_max, max_iter=max_iter,
                    tol=tol, split_tol=0.0, seed_eps=1e-6, log=self._log)

        real_edges = complex_edges.real

        # Remove spurious edges / merges for plotting
        real_merged_edges, active_k = merge_edges(real_edges, tol=1e-4)

        if verbose:
            m_exist = numpy.isfinite(numpy.real(complex_edges))
            rate = numpy.count_nonzero(ok_edges & m_exist) / \
                numpy.count_nonzero(m_exist)
            print("edge success rate:", rate)

        return complex_edges, real_merged_edges, active_k, cusps

    # ====
    # cusp
    # ====

    def cusp(self, t_grid, kind='free', supp=None, max_iter=50, tol=1e-12,
             dedup_t_tol=1e-6, dedup_x_tol=1e-6, return_info=False):
        """
        Find cusp (merge/split) point of evolving spectral edges.

        Parameters
        ----------

        t_grid : array_like
            A time grid to search for cusp points.

        kind : {``'free'``, ``'deformed'``}, default= ``'free'``
            The type of operation:

            * ``'free'``: evolve the spectral curve using free decompression
            * ``'deformed'``: evolve the spectral curve using deformed
              deformation.

        supp : list, default=None
            Estimated support of density as a list of tuples. If not given,
            support is estimated from the fitted polynomial.

        max_iter : int, default=50
            Maximum number of Newton iterations

        tol : float, default=1e-12
            Tolerance in Newton root finding method

        dedup_t_tol : float, default=1e-6
            Tolerance along t axis to identify duplicity (de-duplication.)

        dedup_x_tol : float, default=1e-6
            Tolerance along x axis to identify duplicity (de-duplication.)

        return_info : bool, default=False
            If `True`, a list of debugging information per each cusp point is
            returned.

        Returns
        -------

        cusps : list
            A list of tuples ``(x, t)`` of the location ``x`` and time ``t`` of
            cusp points.

        info : list
            A list of dictionaries, each contain the debugging information for
            a cusp point. This is returned if ``return_info`` is set to `True`.

        See Also
        --------

        edge

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 23

            >>> import numpy
            >>> from freealg import AlgebraicForm
            >>> from freealg.distributions import CompoundFreePoisson
            >>> from freealg import submatrix

            >>> # Create a distribution with two bulks
            >>> cfp = CompoundFreePoisson(t=[2.0,  5.5], w=[0.75, 0.25],
            ...                           lam=0.1)

            >>> # Get a matrix realization of the distribution
            >>> A = cfp.matrix(size=6000, seed=0)

            >>> # Compress the matrix to smaller size
            >>> As = submatrix(A, size=1000)

            >>> # Create AlgebraicForm and fit the smaller matrix
            >>> af = AlgebraicForm(As)
            >>> af.fit(deg_m=3, deg_z=1)

            >>> # Find cusp points
            >>> t_final = numpy.log(A.shape[0] / As.shape[0])
            >>> t = numpy.linspace(0, t_final)
            >>> cusps = af.cusp(t)
        """

        if supp is not None:
            known_supp = supp
        elif self.est_supp is not None:
            known_supp = self.est_supp
        else:
            known_supp = self.support(return_info=False)

        if kind == 'free':
            sol = cusp_wrap(self.coeffs, t_grid, support=known_supp,
                            stieltjes=self._stieltjes_poly, log=self._log,
                            delta=self.delta, edge_dt_max=0.1,
                            edge_max_iter=30, edge_tol=1e-12,
                            max_iter=max_iter, tol=tol,
                            dedup_t_tol=dedup_t_tol, dedup_x_tol=dedup_x_tol)

        elif kind == 'deformed':
            sol = deform_cusp_wrap(self.coeffs, t_grid, support=known_supp,
                                   stieltjes=self._stieltjes_poly,
                                   c0=self.ratio, log=self._log,
                                   delta=self.delta, edge_dt_max=0.1,
                                   edge_max_iter=30, edge_tol=1e-12,
                                   max_iter=max_iter, tol=tol,
                                   dedup_t_tol=dedup_t_tol,
                                   dedup_x_tol=dedup_x_tol)

        else:
            raise ValueError('"kind" should be "free" or "deformed".')

        # Extract x and t from solution
        cusps = []
        for i in range(len(sol)):
            x = sol[i]['x']
            t = sol[i]['t']
            cusps.append((x, t))

        if return_info:
            return cusps, sol
        else:
            return cusps
