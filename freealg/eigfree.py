# SPDX-FileCopyrightText: Copyright 2025, Siavash Ameli <sameli@berkeley.edu>
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

__all__ = ['eigfree']


# ========
# eig free
# ========

def eigfree(A, N=None, psd=None):
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

    psd: bool, default=None
        Determines whether the matrix is positive-semidefinite (PSD; all
        eigenvalues are non-negative). If None, the matrix is considered PSD if
        all sampled eigenvalues are positive.

    Notes
    -----

    This is a convenience function for the FreeForm class with some effective
    defaults that work well for common random matrix ensembles. For improved
    performance and plotting utilites, consider finetuning parameters using
    the FreeForm class.

    References
    ----------

    .. [1] Reference.

    Examples
    --------

    .. code-block:: python

        >>> from freealg import FreeForm
    """

    n = A.shape[0]

    # Size of sample matrix
    n_s = int(80*(1 + numpy.log(n)))

    # If matrix is not large enough, return eigenvalues
    if n < n_s:
        return compute_eig(A)

    if N is None:
        N = n

    # Number of samples
    num_samples = int(10 * (n / n_s)**0.5)

    # Collect eigenvalue samples
    samples = []
    for _ in range(num_samples):
        indices = numpy.random.choice(n, n_s, replace=False)
        samples.append(compute_eig(A[numpy.ix_(indices, indices)]))
    samples = numpy.concatenate(samples).ravel()

    # If all eigenvalues are positive, set PSD flag
    if psd is None:
        psd = samples.min() > 0

    ff = FreeForm(samples)
    # Since we are resampling, we need to provide the correct matrix size
    ff.n = n_s

    # Perform fit and estimate eigenvalues
    order = 1 + int(len(samples)**.2)
    ff.fit(method='chebyshev', K=order, projection='sample', damp='jackson',
           force=True, plot=False, latex=False, save=False, reg=0.05)
    _, _, eigs = ff.decompress(N)

    if psd:
        eigs = numpy.abs(eigs)
        eigs.sort()

    return eigs
