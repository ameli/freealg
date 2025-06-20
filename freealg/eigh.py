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
from ._util import compute_eig
from .freeform import FreeForm

__all__ = ['eigh', 'cond', 'norm', 'trace', 'slogdet']


# ===============
# subsample apply
# ===============

def _subsample_apply(f, A, output_array=False):
    """
    Compute f(A_n) over subsamples A_n of A. If the output of
    f is an array (e.g. eigvals), specify output_array to be True.
    """

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise RuntimeError("Only square matrices are permitted.")

    n = A.shape[0]

    # Size of sample matrix
    n_s = int(80*(1 + numpy.log(n)))
    # If matrix is not large enough, return eigenvalues
    if n < n_s:
        return f(A), n, n

    # Number of samples
    num_samples = int(10 * (n / n_s)**0.5)

    # Collect eigenvalue samples
    samples = []
    for _ in range(num_samples):
        indices = numpy.random.choice(n, n_s, replace=False)
        samples.append(f(A[numpy.ix_(indices, indices)]))

    if output_array:
        return numpy.concatenate(samples).ravel(), n, n_s

    return numpy.array(samples), n, n_s


# ====
# eigh
# ====

def eigh(A, N=None, psd=None, plots=False):
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

    N : int, default=None
        The size of the matrix containing :math:`\\mathbf{A}` to estimate
        eigenvalues of. If None, returns estimates of the eigenvalues of
        :math:`\\mathbf{A}` itself.

    psd : bool, default=None
        Determines whether the matrix is positive-semidefinite (PSD; all
        eigenvalues are non-negative). If `None`, the matrix is considered PSD
        if all sampled eigenvalues are positive.

    plots : bool, default=False
        Print out all relevant plots for diagnosing eigenvalue accuracy.

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
    For improved performance and plotting utilites, consider fine-tuning
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
        >>> eigs = eigh(A)
    """

    samples, n, n_s = _subsample_apply(compute_eig, A, output_array=True)

    if N is None:
        N = n

    # If all eigenvalues are positive, set PSD flag
    if psd is None:
        psd = samples.min() > 0

    ff = FreeForm(samples)
    # Since we are resampling, we need to provide the correct matrix size
    ff.n = n_s

    # Perform fit and estimate eigenvalues
    order = 1 + int(len(samples)**.2)
    ff.fit(method='chebyshev', K=order, projection='sample',
           force=True, plot=False, latex=False, save=False)

    if plots:
        ff.density(plot=True)
        ff.stieltjes(plot=True)

    _, _, eigs = ff.decompress(N, plot=plots)

    if psd:
        eigs = numpy.abs(eigs)
        eigs.sort()

    return eigs


# ====
# cond
# ====

def cond(A, N=None):
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

    N : int, default=None
        The size of the matrix containing :math:`\\mathbf{A}` to estimate
        eigenvalues of. If None, returns estimates of the eigenvalues of
        :math:`\\mathbf{A}` itself.

    Returns
    -------

    c : float
        Condition number

    See Also
    --------

    eigh
    norm
    slogdet
    trace

    Notes
    -----

    This is a convenience function using :func:`freealg.eigh`.

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

    eigs = eigh(A, N, psd=True)
    return eigs.max() / eigs.min()


# ====
# norm
# ====

def norm(A, N=None, order=None):
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

    N : int, default=None
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

    Returns
    -------

    norm : float
        matrix norm

    See Also
    --------

    eigh
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

    eigs = eigh(A, N)

    if (order == 'inf') or numpy.isinf(order):
        norm_ = max(numpy.abs(eigs))

    elif (order == '-inf') or numpy.isneginf(order):
        norm_ = min(numpy.abs(eigs))

    elif (order == 'nuc') or (order == 1.0):
        norm_ = numpy.sum(numpy.abs(eigs))

    elif (order == 'fro') or (order == 2.0):
        norm_2 = numpy.sum(numpy.abs(eigs)**2)
        norm_ = numpy.sqrt(norm_2)

    elif isinstance(order, (int, float, numpy.integer, numpy.floating)) and \
            not isinstance(order, (bool, numpy.bool_)):
        norm_q = numpy.sum(numpy.abs(eigs)**order)
        norm_ = norm_q**(1.0 / order)

    else:
        raise ValueError('"order" is invalid.')

    return norm_


# =====
# trace
# =====

def trace(A, N=None, p=1.0):
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

    N : int, default=None
        The size of the matrix containing :math:`\\mathbf{A}` to estimate
        eigenvalues of. If None, returns estimates of the eigenvalues of
        :math:`\\mathbf{A}` itself.

    p : float, default=1.0
        The exponent :math:`p` in :math:`\\mathbf{A}^p`.


    Returns
    -------

    trace : float
        matrix trace

    See Also
    --------

    eigh
    cond
    slogdet
    norm

    Notes
    -----

    The trace is highly amenable to subsampling: under free decompression
    the average eigenvalue is assumed constant, so the trace increases
    linearly. Traces of powers fall back to :func:`freealg.eigh`.

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
            N = n
        return numpy.mean(samples) * (N / n_s)

    eig = eigh(A, N)
    return numpy.sum(eig ** p)


# =======
# slogdet
# =======

def slogdet(A, N=None):
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

    N : int, default=None
        The size of the matrix containing :math:`\\mathbf{A}` to estimate
        eigenvalues of. If None, returns estimates of the eigenvalues of
        :math:`\\mathbf{A}` itself.

    Returns
    -------

    sign : float
        Sign of determinant

    ld : float
        natural logarithm of the absolute value of the determinant

    See Also
    --------

    eigh
    cond
    trace
    norm

    Notes
    -----

    This is a convenience function using :func:`freealg.eigh`.

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

    eigs = eigh(A, N)
    sign = numpy.prod(numpy.sign(eigs))
    ld = numpy.sum(numpy.log(numpy.abs(eigs)))

    return sign, ld
