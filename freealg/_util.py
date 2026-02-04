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
import scipy

__all__ = ['resolve_complex_dtype', 'compute_eig', 'submatrix']


# =====================
# resolve complex dtype
# =====================

def resolve_complex_dtype(dtype):
    """
    Convert a user-supplied dtype name to a NumPy dtype object and fall back
    safely if the requested precision is unavailable.
    """

    # Normalise the string
    dtype = str(dtype).lower()

    if not isinstance(numpy.dtype(dtype), numpy.dtype):
        raise ValueError(f'{dtype} is not a recognized numpy dtype.')
    elif not numpy.issubdtype(numpy.dtype(dtype), numpy.complexfloating):
        raise ValueError(f'{dtype} is not a complex dtype.')

    if dtype in {'complex128', '128'}:
        cdtype = numpy.complex128

    elif dtype in ['complex256', '256', 'longcomplex', 'clongcomplex']:

        complex256_found = False
        for name in ['complex256', 'clongcomplex']:
            if hasattr(numpy, name):
                cdtype = getattr(numpy, name)
                complex256_found = True

        if not complex256_found:
            raise RuntimeWarning(
                'NumPy on this platform has no 256-bit complex type. ' +
                'Falling back to complex128.')
            cdtype = numpy.complex128

    else:
        raise ValueError('Unsupported dtype.')

    return cdtype


# ===========
# compute eig
# ===========

def compute_eig(A, lower=False):
    """
    Compute eigenvalues of symmetric matrix.
    """

    eig = scipy.linalg.eigvalsh(A, lower=lower, driver='ev')

    return eig


# =========
# submatrix
# =========

def submatrix(matrix, size, paired=True, seed=None):
    """
    Randomly sample a submatrix from a larger matrix.

    Parameters
    ----------

    matrix : numpy.ndarray
        A 2D square array

    size : int
        Number of rows and columns of the submatrix

    paired : bool, default=True
        If `True`, the rows and columns are sampled with the same random
        indices. If `False`, separate random indices are used for selecting
        rows and columns.

    seed : int, default=None
        Seed for random number generation. If `None`, results will not be
        reproducible.

    Returns
    -------

    sub : numpy.ndarray
        A 2D array with the number of rows/columns specified by ``size``.

    See Also
    --------

    freealg.sample

    Examples
    --------

    .. code-block:: python
        :emphasize-lines: 5

        >>> import numpy
        >>> from freealg import submatrix

        >>> A = numpy.random.randn(1000, 1000)
        >>> B = submatrix(A, size=500, paired=True, seed=0)
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    n = matrix.shape[0]
    if size > n:
        raise ValueError("Submatrix size cannot exceed matrix size.")

    rng = numpy.random.default_rng(seed)

    idx_row = rng.choice(n, size=size, replace=False)
    idx_row = numpy.sort(idx_row)  # optional, preserves original ordering

    if paired:
        idx_col = idx_row
    else:
        idx_col = rng.choice(n, size=size, replace=False)
        idx_col = numpy.sort(idx_col)  # optional, preserves original ordering

    return matrix[numpy.ix_(idx_row, idx_col)]
