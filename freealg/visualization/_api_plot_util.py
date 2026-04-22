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
import matplotlib
import matplotlib.ticker as mticker
from matplotlib.ticker import NullLocator
from matplotlib.collections import PolyCollection
from freealg.visualization import auto_bins, hist
# import colorcet as cc

__all__ = ['plot_flow', 'plot_mass', 'ridgeplot', 'plot_edges']


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
              figsize=None, xlim=None, ylim=None, share_ax=False,
              plot_middle=True, plot_floor=True, cmap=None, c_range=None,
              hist_color=None, nbins=(80, 120), label_mode='int',
              layout='horizontal', log=False, title='Free Decompression',
              inset_ax=None, inset_pos=None, inset_lims=None,
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

    figsize : tuple, default=None
        Figure size as (width, height).

    xlim : tuple, default=None
        The limits ``(x_min, x_max)`` of the x axis in the plot. If `None`,
        minimum and maximum of ``x`` is used.

    ylim : tuple or list of tuples, default=None,
        if a tuple ``(ymin, ymax)`` is given, the y limit for all axes is set
        as so. If a list of tuples ``[(ymin1, ymax1), ..., (ymin3, ymax3)]`` is
        given, each tuple in the list is used for each axis respectively. If
        `None`, an automatic y limit for all axes is set.

    share_ax : bool, default=False
        If `True`, the x axis (in vertical layout) or y axis (in horizontal
        layout) for all axes is shared. See ``layout``.

    plot_middle : bool, default=True
        If  `True`, three axes are plotted, where the first axis is the density
        at the initial time, the second axis is the flow from initial to final
        time, and the third axis is the final time. If `False`, the second axis
        is not plotted.

    plot_floor : bool, default=True
        If `True`, it plots Poisson's kernel :math:`\\delta` floor curve, which
        is defined by :math:`P(x) = (\\delta / \\pi) / (x^2 + \\delta^2)`.

    cmap : matplotlib.cm.cmap, default=None
        The colormap used for the color progression for each density curve.

    c_range : default=(0, 1)
        The range of the ``cmap`` to clip the color range of the colormap.

    hist_color : str or list, default=None
        Color name of both histograms. If a string, the color is used for both
        histograms. If a list of two strings, each string is used for a
        histogram.

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

    inset_ax : list, default=None
        List of indices of axes to inset inset axis. For example, to include
        inset to the middle and last axes, use ``[1, 2]``.

    inset_pos : list, default=None
        The relative position of an inset to each axis frame as
        ``[left_x, left_y, width, height]``.

    inset_lims : list, default=None
        List of two tuples ``[(x_min, x_max), (y_min, y_max)]`` as the x-lim
        and y-lim of the inset axes.

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
                if plot_middle:
                    ncols = 3
                    if figsize is None:
                        figsize = (10, 3.1)
                else:
                    ncols = 2
                    if figsize is None:
                        figsize = (6.7, 3.1)
                sharex = False
                sharey = share_ax
            elif layout == 'vertical':
                ncols = 1
                if plot_middle:
                    nrows = 3
                    if figsize is None:
                        figsize = (7, 7)
                else:
                    nrows = 2
                    if figsize is None:
                        figsize = (7, 4.6)
                sharex = share_ax
                sharey = False
            else:
                raise ValueError('"layout" is invalid.')

            _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                 sharex=sharex, sharey=sharey)

        # Inset
        in_ax = [None] * len(ax)
        if inset_ax is not None:
            for ax_id in inset_ax:
                in_ax[ax_id] = ax[ax_id].inset_axes(inset_pos)
                if log:
                    in_ax[ax_id].set_xscale('log')
                    in_ax[ax_id].set_yscale('log')
                ax[ax_id].indicate_inset_zoom(in_ax[ax_id])
                in_ax[ax_id].set_xlim(inset_lims[0])
                in_ax[ax_id].set_ylim(inset_lims[1])
                in_ax_ylim = inset_lims[1]
                in_ax[ax_id].set_xticks([])
                in_ax[ax_id].set_yticks([in_ax_ylim[0], in_ax_ylim[1]])
                in_ax[ax_id].minorticks_off()
                in_ax[ax_id].tick_params(axis='x', which='both', bottom=False,
                                         top=False, labelbottom=False)
                in_ax[ax_id].tick_params(axis='y', which='major',
                                         left=True, right=False,
                                         labelleft=True)
                in_ax[ax_id].tick_params(axis='y', which='minor', left=False,
                                         right=False)
                in_ax[ax_id].tick_params(axis='y', labelsize=9)
                in_ax[ax_id].set_facecolor('floralwhite')

        # Duplicate for each histogram
        if isinstance(hist_color, str):
            hist_color = [hist_color, hist_color]

        # Left axis
        ax[0].plot(x, rho[0], color=colors[0], label='Fitted', zorder=1)
        if in_ax[0] is not None:
            in_ax[0].plot(x, rho[0], color=colors[0], zorder=1)

        supp_init = [numpy.nanmin(eig_init), numpy.nanmax(eig_init)]
        if log:
            bins = numpy.geomspace(supp_init[0], supp_init[1], nbins[0])
            vals, edges = numpy.histogram(eig_init, bins=bins, density=True)
        else:
            edges, vals = hist(eig_init, nbins[0], atoms=[0],
                               support=supp_init)

        ax[0].stairs(vals, edges, fill=True, zorder=-1, alpha=1.0,
                     color=hist_color[0], label='Empirical Histogram')
        if in_ax[0] is not None:
            in_ax[0].stairs(vals, edges, fill=True, zorder=-1, alpha=1.0,
                            color=hist_color[0])

        # Center axis
        if plot_middle:
            for i in range(rho.shape[0]):
                label = _decimal_text(sizes[i] / 1000.0, label_mode)
                ax[1].plot(x, rho[i], color=colors[i], label=label)
                if in_ax[1] is not None:
                    in_ax[1].plot(x, rho[i], color=colors[i])

        # Right axis
        ax[-1].plot(x, rho[-1], color=colors[-1], label='Prediction', zorder=1)
        if in_ax[-1] is not None:
            in_ax[-1].plot(x, rho[-1], color=colors[-1], zorder=1)
        supp_final = [numpy.nanmin(eig_final), numpy.nanmax(eig_final)]

        if log:
            bins = numpy.geomspace(supp_final[0], supp_final[1], nbins[1])
            vals, edges = numpy.histogram(eig_final, bins=bins, density=True)
        else:
            edges, vals = hist(eig_final, nbins[1], atoms=[0],
                               support=supp_final)

        ax[-1].stairs(vals, edges, fill=True, zorder=-1, alpha=1.0,
                      color=hist_color[1], label='Empirical Histogram')
        if in_ax[-1] is not None:
            in_ax[-1].stairs(vals, edges, fill=True, zorder=-1, alpha=1.0,
                             color=hist_color[1])

        if plot_floor:
            if (log is True) and (delta is not None):
                # Baseline
                poisson_floor = (1.0 / numpy.pi) * delta / (x**2 + delta**2)
                for i in [0, 1, 2]:
                    if i != 1:
                        label = r'Poisson kernel $\delta$-floor'
                    else:
                        label = ''
                    ax[i].plot(x, poisson_floor, '--', color='gray', zorder=0,
                               label=label)

        # y limits
        if ylim is None:
            if log:
                ylim = (0.5 * float(numpy.nanmin(rho)),
                        2.0 * float(numpy.nanmax(rho)))
            else:
                ylim = (0.0, 1.1 * float(numpy.max(rho)))

        if isinstance(ylim, tuple):
            ylim = [ylim] * len(ax)
        elif isinstance(ylim, list) and (len(ylim) != len(ax)):
            raise ValueError('If "ylim" is given as a list, the length of '
                             '"ylim" list should match the number of axes.')

        for i in range(len(ax)):
            ax[i].set_ylim(ylim[i])

        # X limits
        if xlim is None:
            xlim = [numpy.nanmin(x), numpy.nanmax(x)]

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

        if plot_middle:
            ax[1].set_title(r'(b) ' + title)

        ax_letter = 'c' if plot_middle is True else 'b'
        ax[-1].set_title(
            rf'({ax_letter}) Final Density ($n={{{sizes[-1]/1000:>0.0f}}}$K)')

        plt.tight_layout()

        fig = ax[0].get_figure()
        if share_ax:
            fig.subplots_adjust(wspace=0.05)
        else:
            fig.subplots_adjust(wspace=0.18)

        fig.patch.set_alpha(0)

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
                                  transparent_background=False, dpi=200,
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

def plot_mass(sizes, x, rho, x0=None, rho0=None, atoms=None, ax=None,
              figsize=None, log_x=False, gap=0.7, save=False, latex=False):
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

    x0 : numpy.array, default=None
        Abscissa, together with ``rho0``, used to compute mass at initial
        density at higher resolution grid. If `None`, only ``x`` and ``rho``
        is used.

    rho0 : numpy.array, default=None
        A 1D density at initial time, used to compute higher accuracy initial
        density mass on the grid ``x0``. If `None`, the density ``rho[0, :]``
        is used. This is useful when the initial density has a spike-like
        behavior at the initial time (usually near the origin) hence a higher
        resolution grid is needed to accurately compute mass.

    atoms : list of tuples, default=None
        A list of tuples ``(atom_location, atom_weight)``.

    ax : matplotlib.axes._axes.Axes
        matplotlib's axis object. If `None`, new axis and figure is created.

    figsize : tuple, default=None
        Size of figure. If `None`, a default size is used.

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

    # Higher resolution density for t=0
    if (x0 is not None) and (rho0 is not None):
        ac_mass[0] = numpy.trapz(rho0, x0)

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
            if figsize is None:
                figsize = (5.5, 3)
            _, ax = plt.subplots(figsize=figsize)

        atom_mass_full = numpy.full_like(sizes, atom_mass, dtype=float)
        ac_mass_full = numpy.full_like(sizes, ac_mass, dtype=float)

        ax.bar(center, ac_mass_full, bottom=atom_mass_full, width=width,
               align='center', color='darkgoldenrod', label='Bulk mass')

        if atoms is not None:
            ax.bar(center, atom_mass_full, width=width,
                   align='center', color='maroon', label='Atom mass')

        # ax.set_xticks(sizes)
        ax.set_xlabel(r'Matrix size $n$')
        ax.set_ylabel(r'Mass')

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
              xlim=None, ylim=None, ylim_factor=1.0, scaley=True, log=False,
              text_side='left', atom_tol=0.0, cmap=None, c_range=None,
              hspace=-0.3, nbins=None, bin_factor=10, label_mode='int',
              rho_color=None, title='', save=False, latex=False):
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

    ylim : tuple, default=None
        The limits ``(y_min, y_max)`` of the y axis in the plot. If `None`,
        the limits are automatically chosen.

    ylim_factor : float, default=1.0
        When ``ylim`` is set to `None`, the y axis limits are automatically
        set. The upper limit can be changed by this factor. This is effective
        only if ``scaley=False`` and ``ylim=None``.

    scaley : bool, default=True
        If `True`, the density on each axis is scaled independently to fit the
        axis height. If `False`, the plots in all axes will the same scale,
        showing their true comparable scale. This option is effective if
        ``ylim=None``.

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

    title : str, default=''
        Title of the plot.

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

                x_min = numpy.nanmin(array)
                x_max = numpy.nanmax(array)

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
                min_y = numpy.nanmin([min_y, numpy.nanmin(counts[counts > 0])])
                max_y = numpy.nanmax([max_y, numpy.nanmax(counts)])

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
                xlim = [numpy.nanmin(x), numpy.nanmax(x)]
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
            max_y = numpy.nanmax([max_y, numpy.nanmax(rho)])
            min_y = numpy.nanmin([min_y, numpy.nanmin(rho)])

        # max of y
        if log:
            log_max_y = numpy.log10(max_y)
            log_min_y = numpy.nanmax([numpy.log10(min_y), numpy.log10(1e-16)])
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
            if ylim is not None:
                ax[i].set_ylim(ylim)
            else:
                # When scaley is True, we let matplotlib to decide max of y
                if scaley is False:
                    # Set the same upper limit for all y axes
                    ax[i].set_ylim(top=max_y_ * ylim_factor)

                # Set lower limit of y
                ax[i].set_ylim(bottom=min_y_)

        ax[-1].set_xlabel(r'$\lambda$', fontsize=fontsize)
        fig.subplots_adjust(hspace=hspace)

        if title != '':
            bbox = ax[0].get_position()
            fig.suptitle(title, fontsize=fontsize,
                         x=(bbox.x0 + bbox.x1) / 2, y=bbox.y1 + 0.025)

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


# ==========
# fill bulks
# ==========

def _fill_bulks(ax, t, edges, color, alpha=0.05, zorder=-2, width_tol=1e-12):
    """
    Fill bulk intervals when the number of active edges may change by 2 at cusp
    times.

    Assumptions:

    - At each time, active bulk intervals are obtained by sorting finite edges
      and pairing adjacent entries.
    - Topology changes only at sampled cusp times.
    - Between consecutive samples, the number of finite edges changes by at
      most two.
    """

    t = numpy.asarray(t, dtype=float)
    E = numpy.asarray(edges, dtype=float)

    polys = []

    def add_poly(verts):
        verts = [(float(x), float(y)) for x, y in verts
                 if numpy.isfinite(x) and numpy.isfinite(y)]
        if len(verts) >= 3:
            polys.append(verts)

    def add_strip_or_triangle(y0, y1, xL0, xR0, xL1, xR1):
        w0 = xR0 - xL0
        w1 = xR1 - xL1

        if not (numpy.isfinite(xL0) and numpy.isfinite(xR0) and
                numpy.isfinite(xL1) and numpy.isfinite(xR1)):
            return

        good0 = w0 > width_tol
        good1 = w1 > width_tol

        if good0 and good1:
            # ordinary quad
            add_poly([(xL0, y0), (xR0, y0), (xR1, y1), (xL1, y1)])
        elif good0 and not good1:
            # collapse to a point at y1
            xc = 0.5 * (xL1 + xR1)
            add_poly([(xL0, y0), (xR0, y0), (xc, y1)])
        elif not good0 and good1:
            # born from a point at y0
            xc = 0.5 * (xL0 + xR0)
            add_poly([(xc, y0), (xR1, y1), (xL1, y1)])
        # else: both zero-width -> nothing

    def best_insertion_index(shorter, longer, width_tol=1e-12):
        """
        longer is shorter with one extra adjacent pair inserted.
        Return insertion position p such that removing longer[p:p+2]
        best matches shorter.

        Structural rule:
          - internal split positions must be odd
          - p=0 and p=len(shorter) are allowed for births at far left/right
        """
        m = len(shorter)
        best_p = None
        best_score = None

        # Only structurally valid places:
        #   far-left birth: p = 0
        #   internal split: p odd
        #   far-right birth: p = m
        candidates = [0] + [p for p in range(1, m, 2)] + [m]

        for p in candidates:
            trial = numpy.concatenate([longer[:p], longer[p+2:]])
            if len(trial) != m:
                continue

            err = numpy.max(numpy.abs(trial - shorter))

            # Tie-breaker: prefer removing a narrower pair
            u, v = longer[p], longer[p+1]
            w = v - u

            score = (err, w, p)

            if best_score is None or score < best_score:
                best_score = score
                best_p = p

        return best_p

    for i in range(len(t) - 1):
        y0, y1 = t[i], t[i + 1]

        f0 = numpy.sort(E[i, numpy.isfinite(E[i])])
        f1 = numpy.sort(E[i + 1, numpy.isfinite(E[i + 1])])

        if (len(f0) % 2) or (len(f1) % 2):
            continue

        n0, n1 = len(f0), len(f1)

        if n0 == n1:
            # same number of active edges: pair adjacent sorted edges
            for j in range(n0 // 2):
                xL0, xR0 = f0[2*j], f0[2*j + 1]
                xL1, xR1 = f1[2*j], f1[2*j + 1]
                add_strip_or_triangle(y0, y1, xL0, xR0, xL1, xR1)

        elif n1 == n0 + 2:
            # one bulk splits into two at y1
            p = best_insertion_index(f0, f1)
            if p is None:
                continue

            # unchanged pieces before insertion
            for j in range(0, p // 2):
                add_strip_or_triangle(
                    y0, y1,
                    f0[2*j], f0[2*j + 1],
                    f1[2*j], f1[2*j + 1]
                )

            # unchanged pieces after insertion
            for j in range((p + 1) // 2, n0 // 2):
                k = 2*j
                add_strip_or_triangle(
                    y0, y1,
                    f0[k], f0[k + 1],
                    f1[k + 2], f1[k + 3]
                )

            # single non-overlapping bridge polygon across the cusp surrounding
            # old interval in f0 is [f0[p-1], f0[p]] if internal split
            if 0 < p < n0:
                xL0, xR0 = f0[p - 1], f0[p]
                xL1 = f1[p - 1]
                xc = 0.5 * (f1[p] + f1[p + 1])  # cusp location
                xR1 = f1[p + 2]
                add_poly([(xL0, y0), (xR0, y0), (xR1, y1), (xc, y1),
                          (xL1, y1)])

            elif p == 0:
                # birth at far left
                xc = 0.5 * (f1[0] + f1[1])
                add_poly([(xc, y0), (f1[2], y1), (f1[1], y1), (f1[0], y1)])

            elif p == n0:
                # birth at far right
                xc = 0.5 * (f1[p] + f1[p + 1])
                add_poly([(f0[-1], y0), (xc, y0), (f1[p + 1], y1),
                          (f1[p], y1)])

        elif n0 == n1 + 2:
            # one bulk merges into one at y1: do reverse case symmetrically
            p = best_insertion_index(f1, f0)
            if p is None:
                continue

            for j in range(0, p // 2):
                add_strip_or_triangle(
                    y0, y1,
                    f0[2*j], f0[2*j + 1],
                    f1[2*j], f1[2*j + 1]
                )

            for j in range((p + 1) // 2, n1 // 2):
                k = 2*j
                add_strip_or_triangle(
                    y0, y1,
                    f0[k + 2], f0[k + 3],
                    f1[k], f1[k + 1]
                )

            if 0 < p < n1:
                xL0 = f0[p - 1]
                xc = 0.5 * (f0[p] + f0[p + 1])
                xR0 = f0[p + 2]
                xL1, xR1 = f1[p - 1], f1[p]
                add_poly([(xL0, y0), (xc, y0), (xR0, y0), (xR1, y1),
                          (xL1, y1)])

        else:
            # too large a topology jump for this routine
            continue

    if not polys:
        return

    coll = PolyCollection(
        polys,
        facecolors=[color],
        edgecolors='none',   # avoid dark overlap/edge seams
        linewidths=0.0,
        antialiaseds=False,
        closed=True,
        alpha=alpha,
        zorder=zorder)

    coll.set_rasterized(True)

    try:
        coll.set_snap(True)
    except Exception:
        pass

    ax.add_collection(coll)


# ==========
# plot edges
# ==========

def plot_edges(t, complex_edges, real_merged_edges, cusps=None, sizes=None,
               edge_color='royalblue', alpha=0.1, fill_color='royal_blue',
               figsize=None, annotate=False, xlim=None, log_x=False,
               log_y=False, flip_y=False, save=False, latex=False):
    """
    """

    if (cusps is not None) and (len(cusps) > 0):
        x_cusps = numpy.zeros((len(cusps),), dtype=float)
        t_cusps = numpy.zeros((len(cusps),), dtype=float)

        for i in range(len(cusps)):
            x_cusps[i] = float(cusps[i][0])
            t_cusps[i] = float(cusps[i][1])
    else:
        x_cusps, t_cusps = None, None

    k = real_merged_edges.shape[1] // 2

    if sizes is None:
        t_ = t
        t_cusps_ = t_cusps
    else:
        t_ = sizes[0] * numpy.exp(t)
        t_cusps_ = sizes[0] * numpy.exp(t_cusps) if t_cusps is not None else \
            None

    with texplot.theme(use_latex=latex):

        if figsize is None:
            figsize = (6.5, 3.5)

        fig, ax = plt.subplots(figsize=figsize)

        for j in range(k):
            a_r = real_merged_edges[:, 2*j + 0]
            b_r = real_merged_edges[:, 2*j + 1]

            # a_c = complex_edges[:, 2*j + 0].real
            # b_c = complex_edges[:, 2*j + 1].real

            # Plot spectral edges with solid lines
            label = 'Spectral edge' if j == 0 else ''
            ax.plot(a_r, t_, color=edge_color, label=label)
            ax.plot(b_r, t_, color=edge_color)

            # Plot ghost edges with dashed lines
            # m_a = numpy.isnan(a_r)
            # m_b = numpy.isnan(b_r)
            # ax.plot(a_c[m_a], t[m_a], '--', color=colors[j], alpha=0.25,
            #         zorder=-1)
            # ax.plot(b_c[m_b], t[m_b], '--', color=colors[j], alpha=0.25,
            #         zorder=-1)
            #
            # Fill between (does not work with bifurcation,
            #               use fill_split_bulk)
            # ax.fill_betweenx(t, a_c, b_c, color=colors[j], alpha=0.05,
            #                  zorder=-2)

            # -----------
            # Plot I_j(t)
            # -----------

            if annotate:
                if log_y:
                    t_mid = numpy.sqrt(t_[-1] * t_cusps_[-1]) \
                            if t_cusps_ is not None else t_[-1]**0.75
                    j_mid = int(numpy.argmin(numpy.abs(t_ - t_mid)))
                    text_t = t_[j_mid] + 0.03 * (t_[-1] - t_[0])
                else:
                    t_mid = 0.5 * (t_[-1] + t_cusps_[-1]) \
                            if t_cusps_ is not None else 0.75 * t_[-1]
                    j_mid = int(numpy.argmin(numpy.abs(t_ - t_mid)))
                    text_t = t_[j_mid] * (t_[-1] / t_[0])**0.02

                ax.annotate('', xy=(float(b_r[j_mid]), t_[j_mid]),
                            xytext=(float(a_r[j_mid]), t_[j_mid]),
                            arrowprops=dict(arrowstyle='<->', color='gray',
                                            lw=1.2))

                ax.text(0.5 * (float(a_r[j_mid]) + float(b_r[j_mid])), text_t,
                        fr'$I_{{{j+1}}}(\tau)$',
                        color='gray', ha='center', va='bottom', fontsize=11)

        # Fill between edges including the bifurcated edges
        _fill_bulks(ax, t_, real_merged_edges, color=fill_color, alpha=alpha,
                    zorder=-2)

        # ----------
        # Plot I_(t)
        # ----------

        if annotate:
            a_all = real_merged_edges[:, 0]
            b_all = real_merged_edges[:, -1]

            if log_y:
                t_mid2 = (min(t_cusps_[0], t_[-1]) * t_[0])**0.5 \
                        if t_cusps_ is not None else 0.45 * t_[-1]
                j_mid2 = int(numpy.argmin(numpy.abs(t_ - t_mid2)))
                text_t = t_[j_mid2] * (t_[-1] / t_[0])**0.02
            else:
                t_mid2 = 0.45 * min(t_cusps_[0], t_[-1]) + t_[0] \
                        if t_cusps_ is not None else 0.45 * t_[-1]
                j_mid2 = int(numpy.argmin(numpy.abs(t_ - t_mid2)))
                text_t = t_[j_mid2] + 0.03 * (t_[-1] - t_[0])

            ax.annotate('', xy=(b_all[j_mid2], t_[j_mid2]),
                        xytext=(a_all[j_mid2], t_[j_mid2]),
                        arrowprops=dict(arrowstyle='<->', color='gray',
                                        lw=1.2))

            ax.text(0.52 * (a_all[j_mid2] + b_all[j_mid2]), text_t,
                    # r'$I_1(t) \cup I_2(t)$',
                    r'$I(\tau), \quad \tau < \tau_{\ast}$',
                    color='gray', ha='center', va='bottom', fontsize=11)

        # Initial edges
        # if af.est_supp is not None:
        #     for edge_a, edge_b in af.est_supp:
        #         plt.plot(edge_a, t_[0], '|', color='maroon', markersize=20)
        #         plt.plot(edge_b, t_[0], '|', color='maroon', markersize=20)

        # Cusp
        if t_cusps_ is not None:
            for i in range(t_cusps_.size):
                label = 'Cusp point' if i == 0 else ''
                ax.plot(x_cusps[i], t_cusps_[i], 'o', color='navy',
                        markersize=2, label=label)
                if (annotate is True):
                    if log_y:
                        text_t = t_cusps_[i] / 1.05
                    else:
                        text_t = t_cusps_[i] - 0.05
                    if i == 0:
                        ax.text(x_cusps[i] + 0.12, text_t,
                                r'$(x_{\ast}, \tau_{\ast})$', fontsize=11)

        h, l = ax.get_legend_handles_labels()
        h.append(matplotlib.patches.Patch(
            facecolor=fill_color, alpha=alpha, edgecolor='none'))
        l.append('Bulk interval')
        ax.legend(h, l, fontsize='small')

        ax.set_xlabel(r'$\lambda$')
        ax.set_title(r'Evolution of Spectral Edges')

        if sizes is not None:
            ax.set_ylabel(r'$n(\tau)$')
        else:
            ax.set_ylabel(r'$t$')

        if log_x:
            ax.set_xscale('log')

        if log_y:
            ax.set_yscale('log', base=2)

        if sizes is not None:
            ax.yaxis.set_major_locator(mticker.LogLocator(base=2,
                                                          subs=(1000/512.0,)))
            ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(_k_pow2_formatter))

        ax.set_ylim([t_[0], t_[-1]])
        if xlim is not None:
            ax.set_xlim(xlim)

        if flip_y:
            ax.invert_yaxis()

        plt.tight_layout()

        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'edge.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)
