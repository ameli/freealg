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

__all__ = ['jackson_damping', 'lanczos_damping', 'fejer_damping',
           'exponential_damping', 'parzen_damping']


# ===============
# jackson damping
# ===============

def jackson_damping(K):
    """
    Compute Jackson damping coefficients for orders k = 0, 1, ..., K-1.
    """

    k = numpy.arange(K)
    g = ((K - k + 1) * numpy.cos(numpy.pi * k / (K + 1)) +
         numpy.sin(numpy.pi * k / (K + 1)) / numpy.tan(numpy.pi / (K + 1))) \
        / (K + 1)

    return g


# ===============
# lanczos damping
# ===============

def lanczos_damping(K):
    """
    Compute Lanczos damping coefficients for orders k = 0, 1, ..., K-1.
    """

    k = numpy.arange(K)
    sigma = numpy.sinc(k / K)

    return sigma


# =============
# fejer damping
# =============

def fejer_damping(K):
    """
    Compute Fejer damping coefficients for orders k = 0, 1, ..., K-1.
    """

    k = numpy.arange(K)
    return 1 - k / K


# ===================
# exponential damping
# ===================

def exponential_damping(K, alpha=6):
    """
    Compute exponential damping coefficients for orders k = 0, 1, ..., K-1.
    """

    k = numpy.arange(K)
    return numpy.exp(-alpha * (k / K)**2)


# ==============
# parzen damping
# ==============

def parzen_damping(K):
    """
    Compute Parzen damping coefficients for orders k = 0, 1, ..., K-1.
    """

    k = numpy.arange(K)
    return 1 - numpy.abs((k - K/2) / (K/2))**3
