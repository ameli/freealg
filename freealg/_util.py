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

__all__ = ['resolve_complex_dtype', 'compute_eig', 'subsample_matrix']


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


# ================
# subsample matrix
# ================

def subsample_matrix(matrix, submatrix_size, seed=None):
    """
    Generate a random subsample of a larger matrix
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    n = matrix.shape[0]
    if submatrix_size > n:
        raise ValueError("Submatrix size cannot exceed matrix size")

    rng = numpy.random.default_rng(seed)
    idx = rng.choice(n, size=submatrix_size, replace=False)
    idx = numpy.sort(idx)  # optional, preserves original ordering

    return matrix[numpy.ix_(idx, idx)]
