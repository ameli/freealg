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
import matplotlib.pyplot as plt
import texplot
import matplotlib.ticker as mticker
from matplotlib.ticker import NullLocator
from freealg.visualization import auto_bins, hist
# import colorcet as cc

__all__ = ['plot_flow', 'plot_mass', 'ridgeplot']


# ============
# decimal text
# ============

def _decimal_text(val, mode):
    """
    Decimal text for plot legends.
    """

    if mode == 'int':
        label = rf'$n={{{val:>0.0f}}}$K'
    elif mode == 'dec':
        label = rf'$n={{{val:>0.1f}}}$K'
    elif mode in ['pow-int']:
        if numpy.abs(val - 1.0) < 1e-8:
            label = r'$n=1$K'
        elif numpy.abs(val - 2.0) < 1e-8:
            label = r'$n=2$K'
        else:
            label = rf'$n=2^{{{numpy.log2(val):>0.0f}}}$K'
    elif mode in ['pow-dec']:
        label = rf'$n=2^{{{numpy.log2(val):>0.1f}}}$K'
    else:
        raise ValueError('"mode" is invalid.')

    return label


# =========
# plot flow
# =========

def plot_flow(sizes, x, rho, eig_init, eig_final, delta=None, ax=None,
              xlim=None, ylim=None, share_ax=False, cmap=None, c_range=None,
              hist_color=None, nbins=(80, 120), label_mode='int',
              layout='horizontal', log=False, title='Free Decompression',
              save=False, latex=False):
    """
    Plot density evolution at various decompression sizes.

    Parameters
    ----------

    sizes : array_like
        A list of matrix decompression sizes. The length of this list should
        be the same as the number of rows of ``rho``.

    x : numpy.array
        The abscissa to plot density. The length of this array should be the
        same as the number of columns of ``rho``.

    rho : numpy.ndarray
        A 2D array where the row ``rho[i, :]`` correspond to a density at the
        matrix size ``sizes[i]``.

    eig_init : numpy.array
        The empirical eigenvalues corresponding to the initial matrix at the
        size ``sizes[0]``. The histogram of this array is used to compare with
        ``rho[0, :]``.

    eig_final : numpy.array
        The empirical eigenvalues corresponding to the final matrix at the
        size ``sizes[-1]``. The histogram of this array is used to compare with
        ``rho[-1, :]``.

    delta : float, default=None
        Poisson kernel :math:`\\delta`-floor. This is only used when
        ``log=True``.

    ax : matplotlib.axes._axes.Axes
        matplotlib's axis object. If `None`, new axis and figure is created.

    xlim : tuple, default=None
        The limits ``(x_min, x_max)`` of the x axis in the plot. If `None`,
        minimum and maximum of ``x`` is used.

    ylim : tuple or scalar, default=None,
        This sets the lower and upper limit of the y axis as follows:

        * If scalar such as ``ylim=y_max``, the y limit of all three axes are
          set to ``[0, y_max]``. This should not be used for the log-scale
          plots (``plot=True``) when the lower y limit cannot be zero.
        * If a tuple of length 2, such as ``ylim= (y_min, y_max)``, the y limit
          of all three axes are set to these bounds.
        * If a tuple of length three, such as ``ylim = (a, b, c)``, the y limit
          of three axes are respectively set to ``(0, a)``, ``(0, b)``, and
          ``(0, c)``, independently.
        * If `None`, y limit is automatically set.

    share_ax : bool, default=False
        If `True`, the x axis (in vertical layout) or y axis (in horizontal
        layput) for all axes is shared. See ``layout``.

    cmap : matplotlib.cm.cmap, default=None
        The colormap used for the color progression for each density curve.

    c_range : default=(0, 1)
        The range of the ``cmap`` to clip the color range of the colormap.

    hist_color : str, default=None
        Color name of the histograms

    nbins : tuple or scalar, default=(80, 120)
        Number of bins for the histograms. It can be given as a tuple of length
        two, in which it sets the number of bins for both histograms, or it
        can be given as be scalar, which is used both both histograms.

    label_mode : {``'int'``, ``'dec'``, ``'pow-int'``, ``'pow-dec'``}, \
            default= ``'int'``
        Decimal representation in legend labels:

        * ``'int'``: integer.
        * ``'dec'``: decimal with only one decimal fraction digit.
        * ``'pow-int'``: Integer powers of base 2.
        * ``'pow-dec'``: Decimal powers of base 2 with one decimal power.

    layout : {``'horizontal'``, ``'vertical'``}, default = ``'horizontal'``
        The layout of three axis: vertical is better for log-scale plots,
        otherwise use or horizontal.

    log : bool, default=False
        If `True`, the x and y axis are shown in logarithmic scale.

    title : str, default=``'Free Decompression'``
        Title of the center plot.

    save : bool or str, default=False
        If `False`, the plot is not saved. If `True`, the plot is saved with a
        default filename. If string, the plot is saved with the full-path
        filename and file extension given by the string.

    latex : bool, default=False
        If `True`, the plot is rendered using LaTeX.

    See Also
    --------

    plot_mass
    ridgeplot
    """

    num_plots = rho.shape[0]

    if cmap is None:
        cmap = plt.get_cmap('gist_heat')
        c_range_ = (0, 0.7)

        # cmap = plt.get_cmap('ocean')
        # c_range_ = (0.3, 0.75)
        #
        # cmap = cc.cm.CET_CBL2
        # c_range_ = (0, 0.5)

        if c_range is None:
            c_range = c_range_

    elif c_range is None:
        c_range = (0, 1)

    if numpy.isscalar(nbins):
        nbins = (nbins, nbins)

    colors = cmap(numpy.linspace(c_range[0], c_range[1], num_plots))

    if hist_color is None:
        hist_color = 'lightsteelblue'

    with texplot.theme(use_latex=latex):

        if ax is None:
            if layout == 'horizontal':
                nrows = 1
                ncols = 3
                figsize = (10, 3.1)
                sharex = False
                sharey = share_ax
            elif layout == 'vertical':
                nrows = 3
                ncols = 1
                figsize = (7, 7)
                sharex = share_ax
                sharey = False
            else:
                raise ValueError('"layout" is invalid.')

            _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                 sharex=sharex, sharey=sharey)

        # Left axis
        ax[0].plot(x, rho[0], color=colors[0], label='Fitted', zorder=1)

        supp_init = [numpy.min(eig_init), numpy.max(eig_init)]
        if log:
            bins = numpy.geomspace(supp_init[0], supp_init[1], nbins[0])
            vals, edges = numpy.histogram(eig_init, bins=bins, density=True)
        else:
            edges, vals = hist(eig_init, nbins[0], atoms=[0],
                               support=supp_init)

        ax[0].stairs(vals, edges, fill=True, zorder=-1, alpha=1.0,
                     color=hist_color, label='Empirical Histogram')

        # Center axis
        for i in range(rho.shape[0]):
            label = _decimal_text(sizes[i] / 1000.0, label_mode)
            ax[1].plot(x, rho[i], color=colors[i], label=label)

        # Right axis
        ax[2].plot(x, rho[-1], color=colors[-1], label='Prediction', zorder=1)
        supp_final = [numpy.min(eig_final), numpy.max(eig_final)]

        if log:
            bins = numpy.geomspace(supp_final[0], supp_final[1], nbins[1])
            vals, edges = numpy.histogram(eig_final, bins=bins, density=True)
        else:
            edges, vals = hist(eig_final, nbins[1], atoms=[0],
                               support=supp_final)

        ax[2].stairs(vals, edges, fill=True, zorder=-1, alpha=1.0,
                     color=hist_color, label='Empirical Histogram')

        if (log is True) and (delta is not None):
            # Baseline
            lower_bound = (1.0 / numpy.pi) * delta / (x**2 + delta**2)
            for i in [0, 1, 2]:
                if i != 1:
                    label = r'Poisson kernel $\delta$-floor'
                else:
                    label = ''
                ax[i].plot(x, lower_bound, '--', color='gray', zorder=0,
                           label=label)

        # y limits
        if ylim is None:
            if log:
                ylim = numpy.array([
                    0.5 * float(numpy.min(rho)),
                    2.0 * float(numpy.max(rho))])
            else:
                ylim = numpy.array([1.1 * float(numpy.max(rho))])
        else:
            ylim = numpy.atleast_1d(ylim)

        if ylim.size == 1:
            # ylim is an array of length 1: max of y for all axis
            for i in range(len(ax)):
                ax[i].set_ylim([0, ylim[0]])
        elif ylim.size == len(ax):
            # ylim is an array of length 3:  max of y per each axis
            for i in range(len(ax)):
                ax[i].set_ylim([0, ylim[i]])
        elif ylim.size == 2:
            # ylim is an array of of length 2: min and max for all axis
            for i in range(len(ax)):
                ax[i].set_ylim([ylim[0], ylim[1]])
        else:
            raise ValueError('Size of "ylim" is invalid.')

        # X limits
        if xlim is None:
            xlim = [numpy.min(x), numpy.max(x)]

        for i in range(len(ax)):
            ax[i].set_xlim(xlim)
            ax[i].set_xlabel(r'$\lambda$')
            ax[i].set_ylabel(r'$\rho(\lambda)$')
            ax[i].legend(fontsize='x-small', facecolor='none')

            if log:
                ax[i].set_xscale('log')
                ax[i].set_yscale('log')

        if layout == 'horizontal':
            for i in range(1, len(ax)):
                ax[i].set_ylabel('')
                if share_ax:
                    ax[i].tick_params(left=False, labelleft=False)
        elif layout == 'vertical':
            for i in range(0, len(ax)-1):
                ax[i].set_xlabel('')
                if share_ax:
                    ax[i].tick_params(axis='x', which='both', bottom=False,
                                      labelbottom=False)
        else:
            raise ValueError('"layout" is invalid.')

        ax[0].set_title(
            rf'(a) Initial Density ($n={{{sizes[0]/1000:>0.0f}}}$K)')
        ax[1].set_title(r'(b) ' + title)
        ax[2].set_title(
            rf'(c) Final Density ($n={{{sizes[-1]/1000:>0.0f}}}$K)')

        plt.tight_layout()

        fig = ax[0].get_figure()
        if share_ax:
            fig.subplots_adjust(wspace=0.05)
        else:
            fig.subplots_adjust(wspace=0.18)

        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'flow.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)


# ================
# k pow2 formatter
# ================

def _k_pow2_formatter(val, pos):
    """
    Format only ticks that are (approximately) 1000 * 2^n
    """

    if val <= 0:
        return ""
    n = numpy.log2(val / 1000.0)
    if numpy.isclose(n, round(n), atol=1e-10):
        return rf"$2^{{{int(round(n))}}}\mathrm{{K}}$"
    return ""


# =========
# plot mass
# =========

def plot_mass(sizes, x, rho, atoms=None, ax=None, log_x=False, gap=0.7,
              save=False, latex=False):
    """
    Plot mass of the absolutely continuous and atoms of spectral density.

    Parameters
    ----------

    sizes : array_like
        A list of matrix decompression sizes. The length of this list should
        be the same as the number of rows of ``rho``.

    x : numpy.array
        The abscissa of density to compute its mass. The length of this array
        should be the same as the number of columns of ``rho``.

    rho : numpy.ndarray
        A 2D array where the row ``rho[i, :]`` correspond to a density at the
        matrix size ``sizes[i]``.

    atoms : list of tuples, default=None
        A list of tuples ``(atom_location, atom_weight)``.

    ax : matplotlib.axes._axes.Axes
        matplotlib's axis object. If `None`, new axis and figure is created.

    log_x : bool, default=False
        If `True`, the x axis is shown on base of powers of two times thousand,
        such as 1K, 2K, 4K, 8K, ..., etc.

    gap : float, default=0.7
        The gap (from 0 to 1) as percentage between bar plots.

    save : bool or str, default=False
        If `False`, the plot is not saved. If `True`, the plot is saved with a
        default filename. If string, the plot is saved with the full-path
        filename and file extension given by the string.

    latex : bool, default=False
        If `True`, the plot is rendered using LaTeX.

    See Also
    --------

    plot_flow
    ridgeplot
    """

    if (atoms is not None) and (len(atoms) > 0):
        atom_mass = atoms[0][1]
    else:
        atom_mass = 0.0

    ac_mass = numpy.trapz(rho, x)
    dx = numpy.diff(sizes).min()

    if log_x:
        logx = numpy.log2(sizes)
        d = numpy.diff(logx).min()

        left = 2**(logx - gap*d/2.0)
        right = 2**(logx + gap*d/2.0)
        center = 2**(logx)
        width = right - left
    else:
        center = sizes
        width = gap * dx

    with texplot.theme(use_latex=latex):

        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 3))

        atom_mass_full = numpy.full_like(sizes, atom_mass, dtype=float)
        ac_mass_full = numpy.full_like(sizes, ac_mass, dtype=float)

        ax.bar(center, ac_mass_full, bottom=atom_mass_full, width=width,
               align='center', color='darkgoldenrod', label='Bulk mass')

        if atoms is not None:
            ax.bar(center, atom_mass_full, width=width,
                   align='center', color='maroon', label='Atom mass')

        # ax.set_xticks(sizes)
        ax.set_xlabel('Sizes')
        ax.set_ylabel('Mass')

        if atoms is None:
            ax.set_title('Mass of Bulks')
        else:
            ax.set_title('Mass Transfer Between Atoms and Bulks')
        # ax.set_ylim([0, 1])
        ax.grid(axis='y')
        ax.legend(fontsize='small', loc='lower right')
        # ax.spines[['top', 'right']].set_visible(False)

        if log_x:
            ax.set_xscale('log', base=2)
            # ax.xaxis.set_major_locator(mticker.LogLocator(base=2))
            ax.xaxis.set_major_locator(
                mticker.LogLocator(base=2, subs=(1000/512.0,)))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(_k_pow2_formatter))

        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'mass.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)


# ======
# darker
# ======

def _darker(color, factor=0.85):
    """
    Return a darker version of a matplotlib color.
    factor < 1 darkens, factor > 1 lightens (roughly).
    Preserves alpha if present.
    """

    r, g, b, *rest = color
    a = rest[0] if rest else None
    out = (r * factor, g * factor, b * factor)
    return (*out, a) if a is not None else out


# =========
# ridgeplot
# =========

def ridgeplot(sizes, x=None, rho=None, eigs=None, ax=None, figsize=None,
              xlim=None, ylim_factor=1.0, scaley=True, log=False,
              text_side='left', atom_tol=0.0, cmap=None, c_range=None,
              hspace=-0.3, nbins=None, bin_factor=10, label_mode='int',
              rho_color=None, save=False, latex=False):
    """
    Rideplot of a cascade of spectral density functions.

    Parameters
    ----------

    sizes : array_like
        A list of matrix decompression sizes. The length of this list should
        be the same as the number of rows of ``rho``.

    x : numpy.array, default=None
        The abscissa to plot density. The length of this array should be the
        same as the number of columns of ``rho``.

    rho : numpy.ndarray, default=None
        A 2D array where the row ``rho[i, :]`` correspond to a density at the
        matrix size ``sizes[i]``. If this argument is provided, also ``x``
        should be given.

    eigs : list
        The list of arrays of empirical eigenvalues corresponding to the
        submatrices at various sizes. The histogram of ``eigs[i]`` is used to
        compare against ``rho[i, :]``.

    ax : matplotlib.axes._axes.Axes
        matplotlib's axis object. This should be a list (or array) of axis
        objects of the length of the size of ``sizes``. If `None`, new axis and
        figure is created.

    figsize : tuple, default=None
        Figure size as (width, height).

    xlim : tuple, default=None
        The limits ``(x_min, x_max)`` of the x axis in the plot. If `None`,
        minimum and maximum of ``x`` is used.

    ylim_factor : float, default=1.0
        The upper limit of y axis is automatically set, but can be changed by
        this factor. This is effective only if ``scaley=False``.

    scaley : bool, default=True
        If `True`, the density on each axis is scaled independently to fit the
        axis height. If `False`, the plots in all axes will the same scale,
        showing their true comparable scale.

    log : bool, default=False
        If `True`, both x and y axis will be in log scale of base 10.

    text_side: {``'left'``, ``'right'``}, default=``'left'``
        The placement of the text for the submatrix sizes on each plot.

    atom_tol : flaot, default=0.0
        A threshold to remove an atom. Values below this value is removed from
        the histogram generation.

    cmap : matplotlib.cm.cmap, default=None
        The colormap used for the color progression for each density curve.

    c_range : default=(0, 1)
        The range of the ``cmap`` to clip the color range of the colormap.

    hspace : float, default=-0.3
        The vertical gap space between axes. Negative makes axes get closer.

    nbins : int, default=None
        Number of histogram bins. If `None`, it is automatically chosen.

    bin_factor : int, default=10
        A factor to increase the number of bins in the histograms.

    label_mode : {``'int'``, ``'dec'``, ``'pow-int'``, ``'pow-dec'``}, \
            default= ``'int'``
        Decimal representation in legend labels:

        * ``'int'``: integer.
        * ``'dec'``: decimal with only one decimal fraction digit.
        * ``'pow-int'``: Integer powers of base 2.
        * ``'pow-dec'``: Decimal powers of base 2 with one decimal power.

    rho_color : str, default=None
        The color of ``rho`` curves.

    save : bool or str, default=False
        If `False`, the plot is not saved. If `True`, the plot is saved with a
        default filename. If string, the plot is saved with the full-path
        filename and file extension given by the string.

    latex : bool, default=False
        If `True`, the plot is rendered using LaTeX.

    See Also
    --------

    plot_flow
    plot_mass
    """

    # Check inputs
    if (rho is not None) and (x is None):
        raise ValueError('When "rho" is provided, "x" should also be given.')
    elif (x is not None) and (rho is None):
        raise ValueError('When "x" is provided, "rho" should also be given.')
    elif (rho is not None) and (x is not None) and (rho.shape[1] != x.size):
        raise ValueError('Number of columns of "rho" should be the size of '
                         '"x".')

    if (rho is not None) and (eigs is not None) and \
            (rho.shape[0] != len(eigs)):
        raise ValueError('Number of rows of "rho" should be the length of '
                         '"eigs"')

    sizes = numpy.array(sizes)
    if (rho is not None) and (sizes.size != rho.shape[0]):
        raise ValueError('The length of "sizes" should be the same as the '
                         'number of rows of "rho".')
    if (eigs is not None) and (sizes.size != len(eigs)):
        raise ValueError('The length of "sizes" should be the same as the '
                         'length of "eigs".')

    # Initialize min and max of y axis (used when scaley is False)
    max_y = 0.0
    min_y = numpy.inf

    fontsize = 11
    num_plots = sizes.size
    if cmap is None:
        cmap = plt.get_cmap('gist_heat')
        c_range_ = (0, 0.8)

        # cmap = plt.get_cmap('ocean')
        # c_range_ = (0.3, 0.75)
        #
        # cmap = cc.cm.CET_R4
        # c_range_ = (0, 1.0)
        #
        # cmap = cc.cm.CET_CBL2
        # c_range_ = (0, 0.5)

        if c_range is None:
            c_range = c_range_

    elif c_range is None:
        c_range = (0, 1)

    colors = cmap(numpy.linspace(c_range[0], c_range[1], num_plots))

    if text_side == 'left':
        text_x = 0.0001
        ha = 'left'
    elif text_side == 'right':
        text_x = 0.9999
        ha = 'right'
    else:
        raise ValueError('"text_side" can be "left" or "right".')

    with texplot.theme(use_latex=latex):
        if ax is None:
            if figsize is None:
                figsize = (7, 4.5)
            _, ax = plt.subplots(figsize=figsize, nrows=num_plots, sharex=True)

        fig = ax[0].get_figure()
        fig.patch.set_alpha(0)
        for a in ax:
            a.set_facecolor('none')

        for i in range(len(ax)):

            if eigs is not None:
                array = eigs[i]

                if atom_tol:
                    array = array[numpy.abs(array) >= atom_tol]

                if nbins is None:
                    nbins = auto_bins(array, factor=bin_factor)

                x_min = numpy.min(array)
                x_max = numpy.max(array)

                if log:
                    bins = numpy.geomspace(x_min, x_max, nbins)
                else:
                    bins = numpy.linspace(x_min, x_max, nbins)

                counts, edges = numpy.histogram(array, bins=bins, density=True)
                ax[i].fill_between(edges[:-1], counts, step="post",
                                   color=colors[i], linewidth=0, zorder=1)

                # white "top curve" on top
                if log:
                    mask = numpy.ones_like(counts).astype(bool)
                else:
                    mask = counts > 0.1 * numpy.max(counts)
                ax[i].step(edges[:-1][mask], counts[mask], where="post",
                           color="white", linewidth=0.5, zorder=3)

                # Update min and max of y
                min_y = numpy.min([min_y, numpy.nanmin(counts[counts > 0])])
                max_y = numpy.max([max_y, numpy.nanmax(counts)])

            elif (x is not None) and (rho is not None):
                ax[i].fill_between(x, y1=rho[i], y2=0, alpha=1,
                                   color=colors[i])

            if (x is not None) and (rho is not None):
                if rho_color is None:
                    # rho_color =_darker(colors[i], factor=0.6))
                    rho_color = 'gray'
                ax[i].plot(x, rho[i], linewidth=1, zorder=20, color=rho_color)

            # Text showing the submatrix sizes
            label = _decimal_text(sizes[i] / 1000.0, label_mode)
            ax[i].text(text_x, 0.07, label, color='black',
                       transform=ax[i].transAxes, fontsize=fontsize-1, ha=ha)

            ax[i].set_yticks([])

            ax[i].spines[['left', 'right', 'top']].set_visible(False)
            ax[i].spines['bottom'].set_color(colors[i])

            if i < len(ax) - 1:
                ax[i].tick_params(axis='x', bottom=False, labelbottom=False)
            else:
                ax[i].tick_params(axis='x', bottom=False, labelbottom=True,
                                  labelsize=fontsize)

            if (xlim is None) and (x is not None):
                xlim = [numpy.min(x), numpy.max(x)]
            ax[i].set_xlim(xlim)

            if log:
                ax[i].set_xscale('log')
                ax[i].set_yscale('log')
                ax[i].tick_params(axis='x', which='both', bottom=False)

                ax[i].set_ylabel('')
                ax[i].yaxis.set_major_locator(NullLocator())
                ax[i].yaxis.set_minor_locator(NullLocator())
                ax[i].tick_params(axis='y', which='both', bottom=False,
                                  labelbottom=False)

        if rho is not None:
            max_y = numpy.max([max_y, numpy.max(rho)])
            min_y = numpy.min([min_y, numpy.min(rho)])

        # max of y
        if log:
            log_max_y = numpy.log10(max_y)
            log_min_y = numpy.log10(min_y)
            log_center_y = 0.5 * (log_max_y + log_min_y)
            log_radius_y = 0.5 * (log_max_y - log_min_y)
            max_y_ = 10.0**(log_center_y + (log_radius_y * 1.05))
            if rho is None:
                min_y_ = 10.0**(log_center_y - (log_radius_y * 1.2))
            else:
                min_y_ = 10.0**(log_center_y - (log_radius_y * 1.04))
        else:
            center_y = 0.5 * (max_y + min_y)
            radius_y = 0.5 * (max_y - min_y)
            max_y_ = center_y + (radius_y * 1.04)
            min_y_ = 0.0

        for i in range(len(ax)):
            # When scaley is True, we let matplotlib to decide max of y
            if scaley is False:
                # Set the same upper limit for all y axes
                ax[i].set_ylim(top=max_y_ * ylim_factor)

            # Set lower limit of y
            ax[i].set_ylim(bottom=min_y_)

        ax[-1].set_xlabel(r'$\lambda$', fontsize=fontsize)
        fig.subplots_adjust(hspace=hspace)

        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'ridgeplot.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)
