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

__all__ = ['BaseForm']


# =========
# Base Form
# =========

class BaseForm(object):
    """
    Base for other "Form" classes. This class itself not a part of API, but the
    inheritted member methods are exposed to the API.
    """

    # ====
    # init
    # ====

    def __init__(self, delta, dtype='complex128'):

        self.A = None
        self.eig = None
        self.delta = delta    # Offset above real axis to apply Plemelj formula
        self.n = None
        self.lam_m = None
        self.lam_p = None

        # Data type for complex arrays
        self.dtype = resolve_complex_dtype(dtype)

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
