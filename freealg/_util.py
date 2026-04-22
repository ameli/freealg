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

def submatrix(matrix, size, block_size=None, paired=True, haar=False,
              seed=None):
    """
    Randomly sample a submatrix from a larger matrix.

    Parameters
    ----------

    matrix : numpy.ndarray
        A 2D square array

    size : int
        Number of rows and columns of the output submatrix.

    block_size : int, default=None
        If given, sampling is performed at the block level where contiguous
        blocks of size ``block_size`` are selected and preserved. The output
        ``size`` should be an integer multiple of ``block_size``.

    paired : bool, default=True
        If `True`, the rows and columns are sampled with the same random
        indices. If `False`, separate random indices are used for selecting
        rows and columns.

    haar : bool, default=False
        If `True`, apply a random orthogonal conjugation to the input matrix
        before sampling, using a Haar-distributed orthogonal matrix. Note that
        this can significantly increase the runtime for large matrices.

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
        raise ValueError('Matrix must be square.')

    n = matrix.shape[0]

    if size > n:
        raise ValueError("Submatrix size cannot exceed matrix size.")

    if block_size is None:
        block_size = 1
    elif block_size <= 0:
        raise ValueError('"block_size" must be positive.')
    elif n % block_size != 0:
        raise ValueError('Matrix size must be divisible by "block_size".')
    elif size % block_size != 0:
        raise ValueError('Submatrix size must be divisible by "block_size".')

    n_blocks = n // block_size
    size_blocks = size // block_size

    if size_blocks > n_blocks:
        raise ValueError(
            'Requested number of blocks exceeds available blocks.')

    rng = numpy.random.default_rng(seed)

    blk_row = rng.choice(n_blocks, size=size_blocks, replace=False)
    blk_row = numpy.sort(blk_row)  # optional, preserves original ordering

    if paired:
        blk_col = blk_row
    else:
        blk_col = rng.choice(n_blocks, size=size_blocks, replace=False)
        blk_col = numpy.sort(blk_col)  # optional, preserves original ordering

    idx_row = (
        blk_row[:, None] * block_size + numpy.arange(block_size)[None, :]
    ).ravel()

    idx_col = (
        blk_col[:, None] * block_size + numpy.arange(block_size)[None, :]
    ).ravel()

    if haar:
        orth = scipy.stats.ortho_group.rvs(n, random_state=rng)
        matrix_conj = orth.T @ matrix @ orth
        return matrix_conj[numpy.ix_(idx_row, idx_col)]
    else:
        return matrix[numpy.ix_(idx_row, idx_col)]
