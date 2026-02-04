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

__all__ = ['poly_trim']


# =========
# poly trim
# =========

def poly_trim(p, tol):
    """
    """

    p = numpy.asarray(p, dtype=float)
    if p.size == 0:
        return p
    k = p.size - 1
    while k > 0 and abs(p[k]) <= tol:
        k -= 1
    return p[: k + 1]
