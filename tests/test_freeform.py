#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2025, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import os
import sys
from freealg import FreeForm
from freealg.distributions import MarchenkoPastur
import numpy
import glob


# ===========
# remove file
# ===========

def remove_file(filename):
    """
    Remove file.
    """

    filenames = glob.glob(filename)

    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)
            print(f'Removed {filename}.', flush=True)


# =============
# test freeform
# =============

def test_freeform():
    """
    A test for :mod:`freeform` module.
    """

    latex = False
    ext = '.pdf'

    # marchenko-Pastur distribution
    mp = MarchenkoPastur(1/50)
    A = mp.matrix(2000)
    eig = numpy.linalg.eigvalsh(A)

    # Create freeform object
    ff = FreeForm(eig, support=(mp.lam_m, mp.lam_p), dtype='complex128')

    # Fit density
    _ = ff.fit(method='chebyshev', K=10, alpha=2, beta=2, reg=0,
               damp='jackson', force=True, continuation='pade', plot=True,
               latex=latex, save='qs_fit' + ext)

    # Estimate Density
    x = numpy.linspace(mp.lam_m-2, mp.lam_p+2, 300)
    rho = ff.density(x, plot=True, latex=latex, save='qs_density' + ext)

    # Hilbert transform
    _ = ff.hilbert(x, rho, plot=True, latex=latex, save='qs_hilbert' + ext)

    # Stieltjes transform
    x = numpy.linspace(mp.lam_m-1.5, mp.lam_p+1.5, 300)
    y = numpy.linspace(-1.5, 1.5, 200)
    _, _ = ff.stieltjes(x, y, plot=True, latex=latex,
                        save='qs_stieltjes' + ext)

    # Decompression
    n = 2 * A.shape[0]
    _, _ = ff.decompress(n, x=None, method='newton', max_iter=500,
                         step_size=0.1, tolerance=1e-4, plot=True, latex=latex,
                         save='qs_decompress' + ext, plot_diagnostics=False)

    # Linalg methods
    _ = ff.eigvalsh(size=n, seed=None)
    _ = ff.trace(size=n, p=1.0, seed=None)
    _ = ff.trace(size=n, p=2.0, seed=None)
    _, _ = ff.slogdet(size=n, seed=None)
    _ = ff.norm(size=n, order=2, seed=None)
    _ = ff.norm(size=n, order=numpy.inf, seed=None)
    _ = ff.norm(size=n, order=-numpy.inf, seed=None)
    _ = ff.norm(size=n, order='fro', seed=None)
    _ = ff.norm(size=n, order='nuc', seed=None)
    _ = ff.cond(size=n, seed=None)

    remove_file('qs_*.pdf')


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_freeform())
