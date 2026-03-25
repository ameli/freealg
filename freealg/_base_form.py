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
from ._util import resolve_complex_dtype
from ._sample import sample
from .visualization._plot_util import plot_branches

__all__ = ['BaseForm']


# =========
# Base Form
# =========

class BaseForm(object):
    """
    Base for other "Form" classes. This class itself not a part of API, but the
    inherited member methods are exposed to the API.
    """

    # ====
    # init
    # ====

    def __init__(self, log=False, dtype='complex128', stieltjes_opt={},
                 inv_stieltjes_opt={}):

        self.A = None
        self.eig = None
        self._log = log
        self.n = None
        self.lam_m = None
        self.lam_p = None
        self.broad_supp = None

        # Data type for complex arrays
        self.dtype = resolve_complex_dtype(dtype)

        self.stieltjes_opt = stieltjes_opt
        self.inv_stieltjes_opt = inv_stieltjes_opt

        # Defaults for inverse Stieltjes options for computing density
        self.inv_stieltjes_opt.setdefault('delta', 1e-5)
        self.inv_stieltjes_opt.setdefault('delta_ladder_ratio', 2.0)
        self.inv_stieltjes_opt.setdefault('delta_ladder_size', 1)
        self.inv_stieltjes_opt.setdefault('delta_ladder_grid', 'linear')
        self.inv_stieltjes_opt.setdefault('z_query_delta', 'const')
        self.inv_stieltjes_opt.setdefault('method', 'polyfit')
        self.inv_stieltjes_opt.setdefault('fit_degree', 2)
        self.inv_stieltjes_opt.setdefault('reg', 0.0)
        self.inv_stieltjes_opt.setdefault('fit_weight', 'small_delta')

        if (self._log is False) and \
                (self.inv_stieltjes_opt['z_query_delta'] != 'const'):
            raise ValueError(
                'When "log=False", set "z_query_delta" to "const.')

        if (self.inv_stieltjes_opt['method'] == 'direct') and \
                (self.inv_stieltjes_opt['delta_ladder_size'] > 1):
            raise ValueError('When "inv_stieltjes_opt" is set to "direct", '
                             '"delta_ladder_size" should be set to 1.')

        # Offset above real axis to apply Plemelj formula
        self.delta = self.inv_stieltjes_opt['delta']
        self.delta_ladder = None

        delta_ladder_ratio = inv_stieltjes_opt['delta_ladder_ratio']
        delta_ladder_size = inv_stieltjes_opt['delta_ladder_size']
        delta_ladder_grid = inv_stieltjes_opt['delta_ladder_grid']

        # Type of grid to produces multiple deltas for Plemelj
        if delta_ladder_grid == 'geometric':
            self.delta_ladder = self.delta * (
                    delta_ladder_ratio **
                    numpy.arange(delta_ladder_size, dtype=int))
        elif delta_ladder_grid == 'linear':
            self.delta_ladder = self.delta * \
                numpy.arange(1, delta_ladder_size+1, dtype=int)
        else:
            raise ValueError('delta_ladder_grid is invalid.')

        # Defaults for Stieltjes computation from polynomial
        self.stieltjes_opt.setdefault('n_levels', 100)
        self.stieltjes_opt.setdefault('max_subdivide', 10)
        self.stieltjes_opt.setdefault('log_scale', self._log)
        self.stieltjes_opt.setdefault('anchor_ratio', 1.0)
        self.stieltjes_opt.setdefault('anchor_y_min', max(self.delta, 1e-8))
        self.stieltjes_opt.setdefault('anchor_y_max', 10)

    # =============
    # generate grid
    # =============

    def _generate_grid(self, scale, extend=1.0, N=500, log=False):
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

        if log:
            x = numpy.geomspace(x_min, x_max, N)
        else:
            x = numpy.linspace(x_min, x_max, N)

        return x

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

        if self._log:

            if (min_supp <= 0) or (max_supp <= 0):
                raise ValueError('Log-scale support requires positive broad '
                                 'support.')

            # Geometric mean in log scale
            c_supp = numpy.sqrt(max_supp * min_supp)
            r_supp = numpy.sqrt(max_supp / min_supp)

            x_min = c_supp / (r_supp * (1.0 + inflate))
            x_max = c_supp * (r_supp * (1.0 + inflate))
        else:
            # Arithmetic mean in linear space
            c_supp = 0.5 * (max_supp + min_supp)
            r_supp = 0.5 * (max_supp - min_supp)

            x_min = c_supp - r_supp * (1.0 + inflate)
            x_max = c_supp + r_supp * (1.0 + inflate)

        return x_min, x_max

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

        All arguments to the :func:`decompress` procedure can be provided.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 1

            >>> from freealg import FreeForm
        """

        if size is None:
            size = self.n

        rho, x = self.decompress(size, **kwargs)
        eigs = numpy.sort(sample(x, rho, size, method='qmc', seed=seed))

        return eigs

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

    # =============
    # plot branches
    # =============

    def plot_branches(self, x=None, y=None, latex=False, save=False, **kwargs):
        """
        Plot branches of the spectral curve of Stieltjes transform.

        Parameters
        ----------

        x : numpy.array, default=None
            The x axis of the grid where the Stieltjes transform is evaluated.
            If `None`, an interval slightly larger than the support interval of
            the spectral density is used.

        y : numpy.array, default=None
            The y axis of the grid where the Stieltjes transform is evaluated.
            If `None`, a grid on the interval ``[-1, 1]`` is used.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        **kwargs : dict
            Parameters to pass to
            :func:`freealg.visualization.domain_coloring`.

        See Also
        --------

        stieltjes

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 11

            >>> import numpy
            >>> from freealg import AlgebraicForm

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=0.0)

            >>> # Plot on a grid
            >>> x = numpy.linspace(0, 4)
            >>> y = numpy.linspace(-1, 1)
            >>> fl.plot_branches(x, y)
        """

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_ub - self.lam_lb)
            center = 0.5 * (self.lam_ub + self.lam_lb)
            scale = 2.0
            x_min = numpy.floor(
                2.0 * (center - 2.0 * radius * scale)) / 2.0
            x_max = numpy.ceil(
                2.0 * (center + 2.0 * radius * scale)) / 2.0
            x = numpy.linspace(x_min, x_max, 400)

        # Create y if not given
        if y is None:
            y = numpy.linspace(-1, 1, 400)

        if y.size % 2 != 0:
            raise ValueError('Size of "y" should be even.')

        X, Y = numpy.meshgrid(x, y)
        z = X + 1j * Y

        m1 = self.stieltjes(z)
        roots_ = self.roots(z.ravel())
        support = self.support()

        plot_branches(z, m1, roots_, support, latex=latex, save=save)
