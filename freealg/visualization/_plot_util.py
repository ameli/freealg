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
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from ._domain_coloring import domain_coloring
from ._hist_util import auto_bins
# from ._hist_util import hist
from ._glue_util import glue_branches
from ._sheets_util import infer_m1_partners_on_cuts, build_sheets_from_roots

__all__ = ['plot_density', 'plot_hilbert', 'plot_stieltjes',
           'plot_stieltjes_on_disk', 'plot_samples', 'plot_branches']


# =====================
# Handler Line 2D Arrow
# =====================

class HandlerLine2DArrow(HandlerLine2D):
    """
    Used for arrow-style object in legend.
    """

    # =============
    # create artist
    # =============

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        # draw a horizontal line
        line = Line2D([xdescent, xdescent + width],
                      [ydescent + height/2, ydescent + height/2],
                      lw=orig_handle.get_linewidth(),
                      color=orig_handle.get_color())
        line.set_transform(trans)

        # draw arrowhead at the right end
        head = Line2D([xdescent + width], [ydescent + height/2],
                      marker=orig_handle.get_marker(),
                      markersize=orig_handle.get_markersize(),
                      color=orig_handle.get_color(),
                      linestyle='None')
        head.set_transform(trans)
        return [line, head]


# ============
# plot density
# ============

def plot_density(x, rho, eig=None, atoms=None, support=None, label='',
                 title='Spectral Density', log=False, latex=False, save=False):
    """
    Parameters
    ----------

    x : numpy.array, default=None
        The abscissa to plot density.

    rho : numpy.ndarray, default=None
        A 1D array of density.

    eig : numpy.array, default=None
        If provided, the empirical histogram of the eigenvalues is also shown
        to compare with ``rho``.

    atoms : list of tuples, default=None
        A list such as ``[(t1, w1), ..., (tk, wk)]`` where ``ti`` are the atom
        locations and ``wi`` are their weight. The sum of the weights should be
        one. If this is given, each atom is shown with a arrow, with the height
        equals its weight corresponding to the right ordinate axis.

    support : list of tuples, default=None
        If provided, the histogram bins are adjusted to the spectral edges.

    label : str, default= ``''``
        Label of the plot.

    title : str, default= ``'Spectral density'``
        Title of the plot

    log : bool, default=False
        If `True`. x and y axis are shown in log scale.

    save : bool or str, default=False
        If `False`, the plot is not saved. If `True`, the plot is saved with a
        default filename. If string, the plot is saved with the full-path
        filename and file extension given by the string.

    latex : bool, default=False
        If `True`, the plot is rendered using LaTeX.
    """

    with texplot.theme(use_latex=latex):

        fig, ax = plt.subplots(figsize=(6, 2.5))

        ax.plot(x, rho, color='black', label=label, zorder=3)

        # Remove zero rho for the plot in log-scale
        if log:
            rho[rho == 0.0] = numpy.nan

        if log:
            l_max = numpy.log10(numpy.nanmax(rho))
            l_min = numpy.log10(numpy.nanmin(rho))
            l_cen = 0.5 * (l_max + l_min)
            l_rad = 0.5 * (l_max - l_min)
            max_y = 10.0**(l_cen + 1.1 * l_rad)
            min_y = numpy.nanmax([10.0**(l_cen - 1.1 * l_rad), 1e-16])
            ax.set_ylim([min_y, max_y])
        else:
            min_y = 0.0
            ax.set_ylim(bottom=min_y)

        ax.set_xlim([x[0], x[-1]])

        # Lock y autoscaling so hist won't change it if there is an atom
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        ax.set_autoscaley_on(False)

        if eig is not None:
            if support is not None:
                if len(support) == 2 and \
                        not isinstance(support[0], (list, tuple)):
                    # Single-interval support in the format [a1, b1]
                    support = [(float(support[0]), float(support[1]))]

                else:
                    # Multi-interval support in format [(a1, b1),..., (ak, bk)]
                    support = [(float(a), float(b)) for a, b in support]

            #     lam_m = support[0][0]
            #     lam_p = support[-1][1]
            # else:
            #     lam_m, lam_p = min(eig), max(eig)

            # Option 1: Use matplotlib's hist
            lam_m, lam_p = min(eig), max(eig)

            if log:
                nbins = auto_bins(numpy.log10(eig))
                bins = numpy.geomspace(lam_m, lam_p, nbins)
            else:
                nbins = auto_bins(eig)
                bins = numpy.linspace(lam_m, lam_p, nbins)
            _ = ax.hist(eig, bins, density=True, color='silver',
                        edgecolor='none', label='Empirical Histogram')

            # Option 2: Use freealg.visualization.hist
            # nbins = auto_bins(eig, factor=2)
            # edges, vals = hist(eig, bins=nbins, m=8, density=True,
            #                    support=support, atoms=atoms)
            # ax.stairs(vals, edges, fill=True, color='silver', alpha=1.0,
            #           label='Empirical Histogram')
        else:
            plt.fill_between(x, y1=rho, y2=min_y, color='silver', zorder=-1)

        arrow_handle = None
        if (atoms is not None) and (len(atoms) > 0):
            ax2 = ax.twinx()
            ax2.set_ylim([0, 1])
            ax2.set_ylabel(r'Atom weight $w$')

            for atom_loc, atom_w in atoms:

                # Plot atom only if within x range (xlim)
                if (atom_loc >= x[0]) and (atom_loc <= x[-1]):
                    ax2.annotate('', xy=(atom_loc, atom_w),
                                 xytext=(atom_loc, 0.0), zorder=10,
                                 annotation_clip=False,
                                 arrowprops={
                                    'arrowstyle': '-|>',
                                    'linewidth': 1.4,
                                    'color': 'black',
                                    'shrinkA': 0.0,
                                    'shrinkB': 0.0,
                                    'mutation_scale': 9})

                    arrow_handle = Line2D([0, 0.96], [0, 0], color='black',
                                          lw=1.4, marker='>', markevery=[1],
                                          markersize=4)

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\rho(\lambda)$''')
        ax.set_title(title)

        if label != '':
            h, ell = ax.get_legend_handles_labels()

            if arrow_handle is not None:
                insert_at = 1 if len(h) >= 1 else 0
                h = h[:insert_at] + [arrow_handle] + h[insert_at:]
                ell = ell[:insert_at] + ['Atom weight'] + ell[insert_at:]

                ax.legend(h, ell, loc='best', fontsize='small',
                          handler_map={arrow_handle: HandlerLine2DArrow()})
            else:
                ax.legend(h, ell, loc='best', fontsize='x-small')

        # Save
        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'density.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)


# ============
# plot hilbert
# ============

def plot_hilbert(x, hilb, support=None, log=False, latex=False, save=False):
    """
    """

    with texplot.theme(use_latex=latex):

        fig, ax = plt.subplots(figsize=(6, 2.7))

        if support is not None:
            lam_m, lam_p = support
            ax.plot(x, hilb, color='black', zorder=3)
            ax.axvline(lam_m, linestyle='--', linewidth=1, color='darkgray',
                       label=r'$\lambda_{\pm}$')
            ax.axvline(lam_p, linestyle='--', linewidth=1, color='darkgray')

        ax.axhline(0, linestyle='--', linewidth=0.5, color='gray', zorder=-1)

        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\mathcal{H}[\rho](x)$')
        ax.set_title('Hilbert Transform')
        ax.legend(fontsize='small')

        # Make sure y=0 is in the y ticks.
        yt = list(ax.get_yticks())
        if 0 not in yt:
            yt.append(0)
        yt = sorted(yt)
        ax.set_yticks(yt)

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        # Save
        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'hilbert.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)


# ===============
# value formatter
# ===============

def _value_formatter(v, pos):
    """
    """

    # v is the normalized "value" channel: v = 1 - exp(-|m(z)|)
    # Invert the mapping: |m(z)| = -ln(1-v)
    if v >= 1:
        return r'$\infty$'
    elif v == 0:
        return r'0'
    else:
        m_val = -numpy.log(1 - v)
        return f"{m_val:.1f}"


# ==============
# plot stieltjes
# ==============

def plot_stieltjes(x, y, m1, m2, support, latex=False, save=False, **kwargs):
    """
    """

    lam_m, lam_p = support
    x_min = numpy.min(x)
    x_max = numpy.max(x)
    y_min = numpy.min(y)
    y_max = numpy.max(y)
    n_y = y.size

    with texplot.theme(use_latex=latex):

        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.2], wspace=0.3)

        eps = 2 / n_y

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(domain_coloring(m1, **kwargs),
                   extent=[x_min, x_max, y_min, y_max], origin='lower',
                   interpolation='gaussian', rasterized=True)
        ax0.plot([lam_m, lam_p], [eps, eps], 'o', markersize=1.5,
                 color='black')
        ax0.plot([lam_m, lam_p], [eps, eps], '-', linewidth=1, color='black')
        ax0.set_xlabel(r'$\mathrm{Re}(z)$')
        ax0.set_ylabel(r'$\mathrm{Im}(z)$')
        ax0.set_title(r'(a) Principal Branch on $\mathbb{C}^{+}$ and ' +
                      r'$\mathbb{C}^{-}$')
        ax0.set_yticks(numpy.arange(y_min, y_max+0.01, 0.5))
        ax0.set_xlim([x_min, x_max])
        ax0.set_ylim([y_min, y_max])

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(domain_coloring(m2, **kwargs),
                   extent=[x_min, x_max, y_min, y_max], origin='lower',
                   interpolation='gaussian', rasterized=True)
        ax1.plot([lam_m, lam_p], [eps, eps], 'o', markersize=1.5,
                 color='black')
        ax1.plot([x_min, lam_m], [eps, eps], '-', linewidth=1, color='black')
        ax1.plot([lam_p, x_max], [eps, eps], '-', linewidth=1, color='black')
        ax1.set_xlabel(r'$\mathrm{Re}(z)$')
        ax1.set_ylabel(r'$\mathrm{Im}(z)$')
        ax1.set_title(r'(b) Principal Branch on $\mathbb{C}^{+}$, Secondary ' +
                      r'Branch on $\mathbb{C}^{-}$')
        ax1.set_yticks(numpy.arange(y_min, y_max+0.01, 0.5))
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])

        pos = ax1.get_position()
        cbar_width = 0.013
        pad = 0.013

        # gs_cb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2],
        #                                          hspace=0.4)
        gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2],
                                         width_ratios=[1, 1], wspace=0.05)

        # Create two separate axes for colorbars using make_axes_locatable:
        # divider = make_axes_locatable(ax[1])
        # cax_hue = divider.append_axes("right", size="4%", pad=0.12)
        # cax_value = divider.append_axes("right", size="4%", pad=0.7)

        # cax_hue = fig.add_subplot(gs_cb[0])
        cax_hue = fig.add_axes([pos.x1 + pad, pos.y0, cbar_width, pos.height])
        norm_hue = matplotlib.colors.Normalize(vmin=-numpy.pi, vmax=numpy.pi)
        cmap_hue = plt.get_cmap('hsv')
        sm_hue = plt.cm.ScalarMappable(norm=norm_hue, cmap=cmap_hue)
        sm_hue.set_array([])
        cb_hue = fig.colorbar(sm_hue, cax=cax_hue)
        cb_hue.set_label(r'$\mathrm{Arg}(m(z))$', labelpad=-6)
        cb_hue.set_ticks([-numpy.pi, 0, numpy.pi])
        cb_hue.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

        # cax_value = fig.add_subplot(gs_cb[1])
        cax_value = fig.add_axes([pos.x1 + 4.4*pad + cbar_width, pos.y0,
                                  cbar_width, pos.height])
        norm_value = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap_value = plt.get_cmap('gray')
        sm_value = plt.cm.ScalarMappable(norm=norm_value, cmap=cmap_value)
        sm_value.set_array([])
        cb_value = fig.colorbar(sm_value, cax=cax_value)
        ticks_norm = [0, 1 - numpy.exp(-0.5), 1 - numpy.exp(-1),
                      1 - numpy.exp(-2), 1]
        cb_value.set_ticks(ticks_norm)
        cb_value.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(_value_formatter))
        cb_value.set_ticklabels(["0", r"$\frac{1}{2}$", "1", "2", r"$\infty$"])
        cb_value.set_label(r'$|m(z)|$', labelpad=0)

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
                save_filename = 'stieltjes.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)


# ======================
# plot stieltjes on disk
# ======================

def plot_stieltjes_on_disk(r, t, m1_D, m2_D, support, latex=False, save=False,
                           **kwargs):
    """
    """

    grid_r, grid_t = numpy.meshgrid(r, t)

    # Inverse Cayley
    lam_m, lam_p = support
    lam_p_z = (lam_p - 1j) / (lam_p + 1j)
    lam_m_z = (lam_m - 1j) / (lam_m + 1j)
    theta_p = numpy.angle(lam_p_z)
    theta_n = numpy.angle(lam_m_z)

    if theta_n < 0:
        theta_n += 2.0 * numpy.pi
    if theta_p < 0:
        theta_p += 2.0 * numpy.pi

    theta_branch = numpy.linspace(theta_n, theta_p, 100)
    theta_alt_branch = numpy.linspace(theta_p, theta_n + 2*numpy.pi, 100)
    r_branch = numpy.ones_like(theta_branch)

    with texplot.theme(use_latex=latex):

        fig = plt.figure(figsize=(12, 4))
        # gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.2], wspace=0.3)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.2], wspace=-0.75)

        ax0 = fig.add_subplot(gs[0], projection='polar')
        ax0.pcolormesh(grid_t, grid_r, domain_coloring(m1_D, **kwargs),
                       shading='auto', rasterized=True)
        ax0.plot(theta_branch, r_branch, '-', linewidth=1, color='black')
        ax0.plot(theta_n, 0.994, 'o', markersize=1.5, color='black')
        ax0.plot(theta_p, 0.994, 'o', markersize=1.5, color='black')
        ax0.set_theta_zero_location("E")  # zero on left
        ax0.set_theta_direction(1)        # angles increase anti-clockwise
        # ax.set_rticks([1, r_max])
        ax0.set_rticks([])
        ax0.set_thetagrids(
            angles=[0, 90, 180, 270],
            labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
        ax0.tick_params(axis='x', pad=-2)
        ax0.grid(False)
        # ax0.set_xlabel(r'$\mathrm{Re}(z)$')
        # ax0.set_ylabel(r'$\mathrm{Im}(z)$')
        ax0.set_title(
            r'(a) Principal Branch on $\mathbb{D}$ and $\mathbb{D}^{c}$',
            pad=25)

        ax1 = fig.add_subplot(gs[1], projection='polar')
        ax1.pcolormesh(grid_t, grid_r, domain_coloring(m2_D, **kwargs),
                       shading='auto', rasterized=True)
        ax1.plot(theta_alt_branch, r_branch, '-', linewidth=1, color='black')
        ax1.plot(theta_n, 0.994, 'o', markersize=1.5, color='black')
        ax1.plot(theta_p, 0.994, 'o', markersize=1.5, color='black')
        ax1.set_theta_zero_location("E")  # zero on left
        ax1.set_theta_direction(1)        # angles increase anti-clockwise
        # ax.set_rticks([1, r_max])
        ax1.set_rticks([])
        ax1.set_thetagrids(
            angles=[0, 90, 180, 270],
            labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
        ax1.tick_params(axis='x', pad=-2)
        ax1.grid(False)
        # ax0.set_xlabel(r'$\mathrm{Re}(z)$')
        # ax0.set_ylabel(r'$\mathrm{Im}(z)$')
        ax1.set_title(r'(b) Principal Branch on $\mathbb{D}$, Secondary ' +
                      r'Branch on $\mathbb{D}^{c}$', pad=25)

        pos = ax1.get_position()
        cbar_width = 0.013
        pad = 0.013

        # gs_cb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2],
        #                                          hspace=0.4)
        gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2],
                                         width_ratios=[1, 1], wspace=0.08)

        # Create two separate axes for colorbars using make_axes_locatable:
        # divider = make_axes_locatable(ax[1])
        # cax_hue = divider.append_axes("right", size="4%", pad=0.12)
        # cax_value = divider.append_axes("right", size="4%", pad=0.7)

        # cax_hue = fig.add_subplot(gs_cb[0])
        cax_hue = fig.add_axes(
            [pos.x1 + 2.5*pad, pos.y0, cbar_width, pos.height])
        norm_hue = matplotlib.colors.Normalize(vmin=-numpy.pi, vmax=numpy.pi)
        cmap_hue = plt.get_cmap('hsv')
        sm_hue = plt.cm.ScalarMappable(norm=norm_hue, cmap=cmap_hue)
        sm_hue.set_array([])
        cb_hue = fig.colorbar(sm_hue, cax=cax_hue)
        cb_hue.set_label(r'$\mathrm{Arg}(m(\zeta))$', labelpad=-6)
        cb_hue.set_ticks([-numpy.pi, 0, numpy.pi])
        cb_hue.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

        # cax_value = fig.add_subplot(gs_cb[1])
        cax_value = fig.add_axes([pos.x1 + (4.4+2.5)*pad + cbar_width, pos.y0,
                                  cbar_width, pos.height])
        norm_value = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap_value = plt.get_cmap('gray')
        sm_value = plt.cm.ScalarMappable(norm=norm_value, cmap=cmap_value)
        sm_value.set_array([])
        cb_value = fig.colorbar(sm_value, cax=cax_value)
        ticks_norm = [0, 1 - numpy.exp(-0.5), 1 - numpy.exp(-1),
                      1 - numpy.exp(-2), 1]
        cb_value.set_ticks(ticks_norm)
        cb_value.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                                              _value_formatter))
        cb_value.set_ticklabels(["0", r"$\frac{1}{2}$", "1", "2", r"$\infty$"])
        cb_value.set_label(r'$|m(\zeta)|$', labelpad=0)

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
                save_filename = 'stieltjes_disk.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)


# ============
# plot samples
# ============

def plot_samples(x, rho, x_min, x_max, samples, latex=False, save=False):
    """
    """

    with texplot.theme(use_latex=latex):

        fig, ax = plt.subplots(figsize=(6, 3))

        bins = numpy.linspace(x_min, x_max, auto_bins(samples))
        _ = ax.hist(samples, bins, density=True, color='silver',
                    edgecolor='none', label='Samples histogram')
        ax.plot(x, rho, color='black', label='Exact density')
        ax.legend(fontsize='small')
        ax.set_ylim(bottom=0)
        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\rho(\lambda)$''')
        ax.set_title('Histogram of Samples from Distribution')

        # Save
        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'samples.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)


# ===========
# plot branch
# ===========

def _plot_branch(ax, img, extent, support, x_ax=True, y_ax=True, bc=None,
                 bc_complement=False, title='', **kwargs):
    """
    Helper for plot_branches. This plots each branch on a given axis.

    Parameters
    ----------

    ax : matplotlib.axes._axes.Axes
        Matplotlib axis.

    img : numpy.ndarray
        Data to be plotted, as domain-coloring RGB code.

    extent : list
        The extend of the domain

    support : list
        Support of the distribution

    x_ax : bool, default=True
        If `True, x-label is included.

    y_ax : bool, default=True
        If `True, y-label is included.

    bc : int or list of int or string
        If negative integer and i = -bc, the complement of the i-th branch cut
        is covered with a solid line. If list [i, j], the branch cuts i and j
        are covered with solid lines.

    title : str
        Title of the axis.

    **kwargs : dict
        Parameters to pass to :func:`freealg.domain_coloring`.
    """

    ax.imshow(domain_coloring(img, **kwargs), extent=extent, origin='lower',
              interpolation='gaussian', rasterized=True)

    n_y = img.shape[0]
    eps = 2 / n_y
    x_min, x_max, y_min, y_max = extent

    bc_color = 'darkgray'

    if bc is None or bc == 0:
        pass

    elif isinstance(bc, list) and len(bc) > 0:

        segs = [(float(a), float(b)) for a, b in bc]
        segs = [(min(a, b), max(a, b)) for a, b in segs]

        # clip + sort
        segs = [(max(x_min, a), min(x_max, b)) for a, b in segs if b > a]
        segs.sort(key=lambda t: t[0])

        # merge overlaps
        merged = []
        for a, b in segs:
            if len(merged) == 0 or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)

        if bc_complement:
            # draw complement of the UNION
            cur = x_min
            for a, b in merged:
                if a > cur:
                    ax.plot([cur, a], [eps, eps], '-', linewidth=1,
                            color=bc_color)
                cur = max(cur, b)
            if cur < x_max:
                ax.plot([cur, x_max], [eps, eps], '-', linewidth=1,
                        color=bc_color)

            # endpoints markers for the union (optional)
            for a, b in merged:
                ax.plot([a, b], [eps, eps], 'o', markersize=1.5,
                        color=bc_color)

        else:
            for a, b in merged:
                ax.plot([a, b], [eps, eps], '-', linewidth=1, color=bc_color)
                ax.plot([a, b], [eps, eps], 'o', markersize=1.5,
                        color=bc_color)

    ax.set_title(title)

    if x_ax is True:
        ax.set_xlabel(r'$\mathrm{Re}(z)$')
    if y_ax is True:
        ax.set_ylabel(r'$\mathrm{Im}(z)$')

    if x_ax is False:
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    if y_ax is False:
        ax.tick_params(axis='y', left=False, right=False, labelleft=False)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


# =============
# plot branches
# =============

def plot_branches(z, m1, roots, support, latex=False, save=False, **kwargs):
    """
    Plot branches of the spectral curve of Stieltjes transform.

    Parameters
    ----------

    x : numpy.array, default=None
        The x axis of the grid where the Stieltjes transform is evaluated.

    y : numpy.array, default=None
        The y axis of the grid where the Stieltjes transform is evaluated.

    latex : bool, default=False
        If `True`, the plot is rendered using LaTeX. This option is
        relevant only if ``plot=True``.

    save : bool, default=False
        If not `False`, the plot is saved. If a string is given, it is
        assumed to the save filename (with the file extension). This option
        is relevant only if ``plot=True``.

    **kwargs : dict
        Parameters to pass to :func:`freealg.visualization.domain_coloring`.
    """

    # Defaults to pass to domain_coloring function
    kwargs.setdefault('n_mod', 18)
    kwargs.setdefault('n_ph', 18)
    kwargs.setdefault('vmin', 0.35)
    kwargs.setdefault('vmax', 1.0)
    kwargs.setdefault('tile_gamma', 0.9)
    kwargs.setdefault('tile_mix', 1.0)
    kwargs.setdefault('shift', 0.0)

    if z.shape[0] % 2 != 0:
        raise ValueError('Size of along "y" axis should be even.')

    sheets, _ = build_sheets_from_roots(z, roots, m1, cuts=support)
    m1 = sheets[0]
    n_sheets = len(sheets)

    if n_sheets < 1:
        raise ValueError('No sheets were constructed from roots.')

    # Partners lists, for each cut I_i, the non-physical sheet index that pairs
    # with m1 across that cut. Reorder non-physical sheets to follow cut order.
    partners = infer_m1_partners_on_cuts(z, sheets, support)
    ordered_nonphys = []
    for k in partners:
        if k not in ordered_nonphys:
            ordered_nonphys.append(k)
    for k in range(1, n_sheets):
        if k not in ordered_nonphys:
            ordered_nonphys.append(k)
    sheets = [sheets[0]] + [sheets[k] for k in ordered_nonphys]
    m1 = sheets[0]

    # Remap partner indices after reordering.
    old_to_new = {0: 0}
    for new_k, old_k in enumerate(ordered_nonphys, start=1):
        old_to_new[old_k] = new_k
    partners = [old_to_new[k] for k in partners]

    # Compute cut segments for each non-physical sheet on each support interval
    xline = z[0, :].real

    ycol = z[:, 0].imag
    pos = numpy.where(ycol > 0.0)[0]
    neg = numpy.where(ycol < 0.0)[0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError('Grid must include both positive and negative Im(z).')

    i_up = int(pos[0])     # closest row above real axis
    i_dn = int(neg[-1])    # closest row below real axis

    cut_segments = {k: [] for k in range(1, n_sheets)}

    for (a, b) in support:
        mask = (xline >= float(a)) & (xline <= float(b))
        idx = numpy.where(mask)[0]
        if idx.size == 0:
            continue

        # pick partner of m1 along this interval (piecewise)
        D = []
        for k in range(1, n_sheets):
            Dk = (numpy.abs(sheets[0][i_up, idx] - sheets[k][i_dn, idx]) +
                  numpy.abs(sheets[0][i_dn, idx] - sheets[k][i_up, idx]))
            D.append(Dk)

        D = numpy.vstack(D)  # (n_sheets-1, n_idx)
        K = numpy.argmin(D, axis=0) + 1  # partner in {1..n_sheets-1}

        # compress consecutive equal partners into segments
        seg_start = idx[0]
        for t in range(1, idx.size):
            if K[t] != K[t - 1]:
                k_partner = int(K[t - 1])
                cut_segments[k_partner].append(
                    (float(xline[seg_start]), float(xline[idx[t - 1]])))
                seg_start = idx[t]

        k_partner = int(K[-1])
        cut_segments[k_partner].append(
            (float(xline[seg_start]), float(xline[idx[-1]])))

    # For each non-physical sheet, list the cuts where it partners with m1.
    partner_cuts = {k: [] for k in range(1, n_sheets)}
    for i_cut, k in enumerate(partners, start=1):
        partner_cuts[k].append(i_cut)

    # Extent
    x_min = numpy.min(z[0, :].real)
    x_max = numpy.max(z[0, :].real)
    y_min = numpy.min(z[:, 0].imag)
    y_max = numpy.max(z[:, 0].imag)
    extent = [x_min, x_max, y_min, y_max]

    letters = 'abcdefghijklmnopqrstuvwxyz'
    sheet_letters = letters[:n_sheets]  # a,b,c,...

    ncols = n_sheets
    width = max(9.0, 3.2 * ncols)

    with texplot.theme(use_latex=latex):
        fig, ax = plt.subplots(nrows=3, ncols=ncols, figsize=(width, 7.2),
                               sharey=True)

        if ncols == 1:
            ax = numpy.asarray(ax).reshape(3, 1)

        # Row 1: (a) m1, (b) m2, ..., (s) m_s
        for k in range(n_sheets):
            if k == 0:
                bc = [(float(a), float(b)) for (a, b) in support]
            else:
                bc = cut_segments[k]

            title = fr'({sheet_letters[k]}) $m_{k+1}$ on $\mathbb{{C}}$'
            _plot_branch(ax[0, k], sheets[k], extent, support, x_ax=(k == 0),
                         y_ax=(k == 0), bc=bc, bc_complement=False,
                         title=title, **kwargs)

        # Row 2: NONE, (ab) m1/m2, (ac) m1/m3, ...
        ax[1, 0].axis('off')
        for k in range(1, n_sheets):
            bc_comp = cut_segments[k]
            title = (fr'({sheet_letters[0]}{sheet_letters[k]}) '
                     fr'$m_1$ on $\mathbb{{C}}^+$ glued to '
                     fr'$m_{k+1}$ on $\mathbb{{C}}^-$')

            _plot_branch(ax[1, k], glue_branches(z, m1, sheets[k]), extent,
                         support, x_ax=False, y_ax=(k == 1), bc=bc_comp,
                         bc_complement=True, title=title, **kwargs)

        # Row 3: NONE, (ba) m2/m1, (ca) m3/m1, ...
        ax[2, 0].axis('off')
        for k in range(1, n_sheets):
            bc_comp = cut_segments[k]
            title = (fr'({sheet_letters[k]}{sheet_letters[0]}) '
                     fr'$m_1$ on $\mathbb{{C}}^-$ glued to '
                     fr'$m_{k+1}$ on $\mathbb{{C}}^+$')

            _plot_branch(ax[2, k], glue_branches(z, sheets[k], m1), extent,
                         support, x_ax=True, y_ax=(k == 1), bc=bc_comp,
                         bc_complement=True, title=title, **kwargs)

        # Fixing sharey=True and axis('off') that hides y tick labels on col=1
        for r in (1, 2):
            a = ax[r, 1]  # the first visible axis in that row
            a.tick_params(axis='y', left=True, labelleft=True)

            # also copy formatter/locator from the main y-axis owner (ax[0,0])
            a.yaxis.set_major_locator(ax[0, 0].yaxis.get_major_locator())
            a.yaxis.set_major_formatter(ax[0, 0].yaxis.get_major_formatter())

            # sometimes matplotlib keeps them invisible; force visible
            for lab in a.get_yticklabels():
                lab.set_visible(True)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.05, hspace=0.05)

        # Save
        if save is False:
            save_status = False
            save_filename = ''
        else:
            save_status = True
            if isinstance(save, str):
                save_filename = save
            else:
                save_filename = 'branches.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)
