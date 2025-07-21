#! /usr/bin/env python

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
import freealg as fa
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


# =================
# test distribution
# =================

def _test_distribution(dist):
    """
    Test each distribution.
    """

    print(f'Testing {dist.__class__.__name__}')

    ext = '.pdf'

    _ = dist.density(x=None, plot=True, latex=False, save='qs_density' + ext)
    _ = dist.hilbert(x=None, plot=True, latex=False, save='qs_hilbert' + ext)
    _, _ = dist.stieltjes(x=None, y=None, plot=True, on_disk=False,
                          latex=False, save='qs_stieltjes_1' + ext)
    _, _ = dist.stieltjes(x=None, y=None, plot=True, on_disk=True,
                          latex=False, save='qs_stieltjes_2' + ext)
    _ = dist.sample(size=100, x_min=None, x_max=None, method='qmc', seed=None,
                    plot=True, latex=False, save='qs_sample' + ext)

    # For Meixner, matrix is not implemented yet.
    if not dist.__class__.__name__.startswith('Meixner'):
        _ = dist.matrix(100, seed=None)


# ==================
# test distributions
# ==================

def test_distributions():
    """
    A test for :mod:`distributions` module.
    """

    # List of distributions
    dists = [
        fa.distributions.MarchenkoPastur(1/50),
        fa.distributions.Wigner(1),
        fa.distributions.KestenMcKay(3),
        fa.distributions.Wachter(2, 3),
        fa.distributions.Meixner(0.1, 4, 0.6),
    ]

    for dist in dists:
        _test_distribution(dist)

    remove_file('qs_*.pdf')


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(test_distributions())
