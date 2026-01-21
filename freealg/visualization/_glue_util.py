# SPDX-FileCopyrightText: Copyright 2026, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy

__all__ = ['glue_branches']


# =============
# glue branches
# =============

def glue_branches(z, m1, m2):
    """
    m12 is the mixing of m1 and m2 where it contains m1 on C^+ and m2 on C^-.
    """

    m12 = numpy.array(m2, copy=True)
    mask_p = numpy.imag(z) >= 0.0
    m12[mask_p] = m1[mask_p]

    return m12
