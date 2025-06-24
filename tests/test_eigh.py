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


# =========
# test eigh
# =========

def test_eigh():
    """
    A test for ``eigh`` function; mostly for speed.
    """
    X = numpy.random.randn(1000, 1000)
    X = (X + X.T) / 2**0.5

    # Compute eigh
    fa.eigh(X)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_eigh())
