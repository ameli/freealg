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
from .free_form import FreeForm

__all__ = ['eigvalsh', 'cond', 'norm', 'trace', 'slogdet']


# ===============
# subsample apply
# ===============

def _subsample_apply(f, A, output_array=False, seed=None):
    """
    Compute f(A_n) over subsamples A_n of A. If the output of
    f is an array (e.g. eigvals), specify output_array to be True.
    """

    if (A.ndim != 2) or (A.shape[0] != A.shape[1]):
        raise RuntimeError("Only square matrices are permitted.")

    n = A.shape[0]

    # Size of sample matrix
    n_s = int(80.0 * (1.0 + numpy.log(n)))

    # If matrix is not large enough, return eigenvalues
    if n < n_s:
        return f(A), n, n

    # Number of samples
    num_samples = int(10 * (n / n_s)**0.5)

    # Collect eigenvalue samples
    samples = []
    rng = numpy.random.default_rng(seed=seed)
    for _ in range(num_samples):
        indices = rng.choice(n, n_s, replace=False)
        samples.append(f(A[numpy.ix_(indices, indices)]))

    if output_array:
        return numpy.concatenate(samples), n, n_s
    else:
        return numpy.array(samples), n, n_s


# ========
# eigvalsh
# ========

def eigvalsh(A, size=None, psd=None, seed=None, plot=False, **kwargs):
    """
    Estimate the eigenvalues of a matrix.

    This function estimates the eigenvalues of the matrix :math:`\\mathbf{A}`
    or a larger matrix containing :math:`\\mathbf{A}` using free decompression.

    Parameters
    ----------

    A : numpy.ndarray
        The symmetric real-valued matrix :math:`\\mathbf{A}` whose eigenvalues
        (or those of a matrix containing :math:`\\mathbf{A}`) are to be
        computed.

    size : int, default=None
        The size of the matrix containing :math:`\\mathbf{A}` to estimate
        eigenvalues of. If None, returns estimates of the eigenvalues of
        :math:`\\mathbf{A}` itself.

    psd : bool, default=None
        Determines whether the matrix is positive-semidefinite (PSD; all
        eigenvalues are non-negative). If `None`, the matrix is considered PSD
        if all sampled eigenvalues are positive.

    seed : int, default=None
        The seed for sampling rows/columns of matirx as well as the Quasi-Monte
        Carlo sampler for eigenvalues from density.

    plot : bool, default=False
        Print out all relevant plots for diagnosing eigenvalue accuracy.

    **kwargs : dict, optional
        Pass additional options to the underlying
        :func:`FreeForm.decompress` function.

    Returns
    -------

    eigs : numpy.array
        Eigenvalues of decompressed matrix

    See Also
    --------

    cond

    Notes
    -----

    This is a convenience function for the :class:`freealg.FreeForm` class with
    some effective defaults that work well for common random matrix ensembles.
    For improved performance and plotting utilities, consider fine-tuning
    parameters using the FreeForm class.

    References
    ----------

    .. [1] Reference.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 6

        >>> from freealg import cond
        >>> from freealg.distributions import MarchenkoPastur

        >>> mp = MarchenkoPastur(1/50)
        >>> A = mp.matrix(3000)
        >>> eigs = eigvalsh(A)
    """

    samples, n, n_s = _subsample_apply(compute_eig, A, output_array=True,
                                       seed=seed)

    if size is None:
        size = n

    # If all eigenvalues are positive, set PSD flag
    if psd is None:
        psd = samples.min() > 0

    ff = FreeForm(samples)

    # Since we are resampling, we need to provide the correct matrix size
    ff.n = n_s

    # Perform fit and estimate eigenvalues
    order = 1 + int(len(samples)**0.2)
    ff.fit(method='chebyshev', K=order, projection='sample',
           continuation='wynn-eps', force=True, plot=False, latex=False,
           save=False)

    if plot:
        ff.density(plot=True)
        ff.stieltjes(plot=True)

    # Sampling method using Pade seems to need a lower tolerance to properly
    # work. Here we set defaults unless user provides otherwise. Note that the
    # default of tolerance in ff._decompress is much larger (1e-4) for other
    # methods (Newton, and non-sampling projections such as Gaussian and beta)
    # to work properly.
    kwargs.setdefault('tolerance', 1e-9)
    kwargs.setdefault('method', 'secant')

    eigs = ff.eigvalsh(size, seed=seed, plot=plot, **kwargs)

    if psd:
        eigs = numpy.abs(eigs)
        eigs.sort()

    return eigs


# ====
# cond
# ====

def cond(A, size=None, seed=None, **kwargs):
    """
    Estimate the condition number of a Hermitian positive-definite matrix.

    This function estimates the condition number of the matrix
    :math:`\\mathbf{A}` or a larger matrix containing :math:`\\mathbf{A}`
    using free decompression.

    Parameters
    ----------

    A : numpy.ndarray
        The symmetric real-valued matrix :math:`\\mathbf{A}` whose condition
        number (or that of a matrix containing :math:`\\mathbf{A}`) are to be
        computed.

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

    c : float
        Condition number

    See Also
    --------

    eigvalsh
    norm
    slogdet
    trace

    Notes
    -----

    This is a convenience function using :func:`freealg.eigvalsh`.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 6

        >>> from freealg import cond
        >>> from freealg.distributions import MarchenkoPastur

        >>> mp = MarchenkoPastur(1/50)
        >>> A = mp.matrix(3000)
        >>> cond(A)
    """

    eigs = eigvalsh(A, size=size, psd=True, seed=seed, **kwargs)
    return eigs.max() / eigs.min()


# ====
# norm
# ====

def norm(A, size=None, order=2, seed=None, **kwargs):
    """
    Estimate the Schatten norm of a Hermitian matrix.

    This function estimates the norm of the matrix :math:`\\mathbf{A}` or a
    larger matrix containing :math:`\\mathbf{A}` using free decompression.

    Parameters
    ----------

    A : numpy.ndarray
        The symmetric real-valued matrix :math:`\\mathbf{A}` whose condition
        number (or that of a matrix containing :math:`\\mathbf{A}`) are to be
        computed.

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

    eigvalsh
    cond
    slogdet
    trace

    Notes
    -----

    Thes Schatten :math:`p`-norm is defined by

    .. math::

        \\Vert \\mathbf{A} \\Vert_p = \\left(
        \\sum_{i=1}^N \\vert \\lambda_i \\vert^p \\right)^{1/p}.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 6

        >>> from freealg import norm
        >>> from freealg.distributions import MarchenkoPastur

        >>> mp = MarchenkoPastur(1/50)
        >>> A = mp.matrix(3000)
        >>> norm(A, 100_000, order='fro')
    """

    eigs = eigvalsh(A, size=size, seed=seed, **kwargs)

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


# =====
# trace
# =====

def trace(A, N=None, p=1.0, seed=None, **kwargs):
    """
    Estimate the trace of a power of a Hermitian matrix.

    This function estimates the trace of the matrix power :math:`\\mathbf{A}^p`
    or that of a larger matrix containing :math:`\\mathbf{A}`.

    Parameters
    ----------

    A : numpy.ndarray
        The symmetric real-valued matrix :math:`\\mathbf{A}` whose trace of
        a power (or that of a matrix containing :math:`\\mathbf{A}`) is to be
        computed.

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

    eigvalsh
    cond
    slogdet
    norm

    Notes
    -----

    The trace is highly amenable to subsampling: under free decompression
    the average eigenvalue is assumed constant, so the trace increases
    linearly. Traces of powers fall back to :func:`freealg.eigvalsh`.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 6

        >>> from freealg import norm
        >>> from freealg.distributions import MarchenkoPastur

        >>> mp = MarchenkoPastur(1/50)
        >>> A = mp.matrix(3000)
        >>> trace(A, 100_000)
    """

    if numpy.isclose(p, 1.0):
        samples, n, n_s = _subsample_apply(numpy.trace, A, output_array=False)
        if N is None:
            size = n
        return numpy.mean(samples) * (size / n_s)

    eig = eigvalsh(A, size=size, seed=seed, **kwargs)
    return numpy.sum(eig ** p)


# =======
# slogdet
# =======

def slogdet(A, size=None, seed=None, **kwargs):
    """
    Estimate the sign and logarithm of the determinant of a Hermitian matrix.

    This function estimates the *slogdet* of the matrix :math:`\\mathbf{A}` or
    a larger matrix containing :math:`\\mathbf{A}` using free decompression.

    Parameters
    ----------

    A : numpy.ndarray
        The symmetric real-valued matrix :math:`\\mathbf{A}` whose condition
        number (or that of a matrix containing :math:`\\mathbf{A}`) are to be
        computed.

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

    sign : float
        Sign of determinant

    ld : float
        natural logarithm of the absolute value of the determinant

    See Also
    --------

    eigvalsh
    cond
    trace
    norm

    Notes
    -----

    This is a convenience function using :func:`freealg.eigvalsh`.

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 6

        >>> from freealg import norm
        >>> from freealg.distributions import MarchenkoPastur

        >>> mp = MarchenkoPastur(1/50)
        >>> A = mp.matrix(3000)
        >>> sign, ld = slogdet(A, 100_000)
    """

    eigs = eigvalsh(A, size=size, seed=seed, **kwargs)
    sign = numpy.prod(numpy.sign(eigs))
    ld = numpy.sum(numpy.log(numpy.abs(eigs)))

    return sign, ld
