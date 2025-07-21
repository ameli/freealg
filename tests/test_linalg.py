#! /usr/bin/env python

# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import freealg as fa
import numpy
import sys


# ===========
# test linalg
# ===========

def test_linalg():
    """
    A test for ``linalg`` function; mostly for speed.
    """

    X = numpy.random.randn(1000, 1000)
    X = (X + X.T) / (2.0**0.5)

    n = 2 * X.shape[0]
    _ = fa.eigvalsh(X, size=n, psd=False, plot=False, seed=None)
    _ = fa.cond(X, size=n, seed=None)
    _ = fa.trace(X, size=n, p=1, seed=None)
    _, _ = fa.slogdet(X, size=n, seed=None)
    _ = fa.norm(X, size=n, order=2, seed=None)
    _ = fa.norm(X, size=n, order=numpy.inf, seed=None)
    _ = fa.norm(X, size=n, order=-numpy.inf, seed=None)
    _ = fa.norm(X, size=n, order='fro', seed=None)
    _ = fa.norm(X, size=n, order='nuc', seed=None)

    print('OK')


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_linalg())
