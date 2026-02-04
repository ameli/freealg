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
import matplotlib.pyplot as plt
import texplot

__all__ = ['plot_fit']


# ========
# plot fit
# ========

def plot_fit(psi, x_supp, g_supp, g_supp_approx, support, latex=False,
             save=False):
    """
    Plot fitted psi coefficients. This is a helper for
    :func:`FreeForm.fit` when ``plot=True``.
    """

    with texplot.theme(use_latex=latex):

        if g_supp is None:
            figsize = (4.5, 3)
            ncols = 1
        else:
            figsize = (9, 3)
            ncols = 2

        fig, ax = plt.subplots(figsize=figsize, ncols=ncols)

        if g_supp is None:
            ax = [ax]

        # Plot psi
        n = numpy.arange(1, 1+psi.size)
        ax[0].plot(n, psi**2, '-o', markersize=3, color='black')
        ax[0].set_xlim([n[0]-1e-3, n[-1]+1e-3])
        ax[0].set_xlabel(r'$k$')
        ax[0].set_ylabel(r'$\vert \psi_k \vert^2$')
        ax[0].set_title('Spectral Energy per Mode')
        ax[0].set_yscale('log')

        # Plot pade
        if g_supp is not None:
            lam_m, lam_p = support
            g_supp_min = numpy.min(g_supp)
            g_supp_max = numpy.max(g_supp)
            g_supp_dif = g_supp_max - g_supp_min
            g_min = g_supp_min - g_supp_dif * 1.1
            g_max = g_supp_max + g_supp_dif * 1.1

            ax[1].plot(x_supp, g_supp, color='firebrick',
                       label=r'$2 \pi \times $ Hilbert Transform')
            ax[1].plot(x_supp, g_supp_approx, color='black',
                       label='Pade estimate')
            ax[1].legend(fontsize='small')
            ax[1].set_xlim([lam_m, lam_p])
            ax[1].set_ylim([g_min, g_max])
            ax[1].set_title('Approximation of Glue Function')
            ax[1].set_xlabel(r'$x$')
            ax[1].set_ylabel(r'$G(x)$')

        plt.tight_layout()

        # Save
        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'fit.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)
