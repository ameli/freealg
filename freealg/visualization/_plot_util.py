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
from ..visualization import domain_coloring

__all__ = ['plot_density', 'plot_hilbert', 'plot_stieltjes',
           'plot_stieltjes_on_disk', 'plot_samples']


# =========
# auto bins
# =========

def _auto_bins(array, method='scott', factor=5):
    """
    Automatic choice for the number of bins for the histogram of an array.

    Parameters
    ----------

    array : numpy.array
        An array for histogram.

    method : {``'freedman'``, ``'scott'``, ``'sturges'``}, default= ``'scott'``
        Method of choosing number of bins.

    Returns
    -------

    num_bins : int
        Number of bins for histogram.
    """

    if method == 'freedman':

        q75, q25 = numpy.percentile(array, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(array) ** (1/3))

        if bin_width == 0:
            # Fallback default
            return
            num_bins = 100
        else:
            num_bins = int(numpy.ceil((array.max() - array.min()) / bin_width))

    elif method == 'scott':

        std = numpy.std(array)
        bin_width = 3.5 * std / (len(array) ** (1/3))
        num_bins = int(numpy.ceil((array.max() - array.min()) / bin_width))

    elif method == 'sturges':

        num_bins = int(numpy.ceil(numpy.log2(len(array)) + 1))

    else:
        raise NotImplementedError('"method" is invalid.')

    return num_bins * factor


# ========
# hist ash
# ========

def hist_ash(eig, bins, m=8, hist_range=None, density=True):
    """
    Averaged shifted histogram (ASH) from raw samples.

    Parameters
    ----------

    eig : array_like
        1D samples (e.g. eigenvalues).

    bins : int or array_like
        Base histogram bins. If int, uses numpy.histogram_bin_edges(eig, bins,
        range=hist_range). If array_like, treated as base bin edges (uniform).

    m : int, default=8
        Number of shifts (smoothing strength). Typical: 8, 16.

    hist_range : tuple(float, float) or None
        Range used only when bins is int.

    density : bool, default=True
        If True, returns density. If False, returns counts.

    Returns
    -------

    edges : numpy.ndarray
        Base bin edges, length K+1.

    values : numpy.ndarray
        ASH density (or counts) per base bin, length K.
    """

    x = numpy.asarray(eig, dtype=float).ravel()
    x = x[numpy.isfinite(x)]
    if x.size == 0:
        raise ValueError("hist_ash: empty or non-finite input.")

    if isinstance(bins, (int, numpy.integer)):
        edges = numpy.histogram_bin_edges(x, bins=int(bins), range=hist_range)
    else:
        edges = numpy.asarray(bins, dtype=float).ravel()
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("hist_ash: bins must be int or 1D edges array.")

    if m < 1:
        raise ValueError("hist_ash: m must be >= 1.")

    d = numpy.diff(edges)
    h = float(d[0])
    if not numpy.allclose(d, h, rtol=0.0, atol=1.0e-12 * max(1.0, abs(h))):
        raise ValueError("hist_ash: base bin edges must be uniformly spaced.")

    k = edges.size - 1
    a = float(edges[0])
    b = float(edges[-1])

    fine_edges = numpy.linspace(a, b, k * m + 1)
    fine_counts, _ = numpy.histogram(x, bins=fine_edges)

    w = numpy.concatenate(
        (numpy.arange(1, m + 1), numpy.arange(m - 1, 0, -1))).astype(float)
    fine_pad = numpy.pad(fine_counts, (0, 2 * m - 2), mode="constant")

    ash_counts = numpy.empty(k, dtype=float)
    for i in range(k):
        seg = fine_pad[i*m:i*m+(2*m-1)]
        ash_counts[i] = (seg @ w) / float(m)

    if density:
        values = ash_counts / (x.size * h)
    else:
        values = ash_counts

    return edges, values


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
                 title='Spectral Density', latex=False, save=False):
    """
    Parameters
    ----------

    atoms : list of tuples, default=None
        A list such as ``[(t1, w1), ..., (tk, wk)]`` where ``ti`` are the atom
        locations and ``wi`` are their weight. The sum of the weights should be
        one. If this is given, each atom is shown with a arrow, with the height
        equals its weight corresponding to the right ordinate axis.
    """

    with texplot.theme(use_latex=latex):

        fig, ax = plt.subplots(figsize=(6, 2.5))

        ax.plot(x, rho, color='black', label=label, zorder=3)

        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim(bottom=0)

        # Lock y autoscaling so hist won't change it if there is an atom
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        ax.set_autoscaley_on(False)

        if eig is not None:
            if support is not None:
                lam_m, lam_p = support
            else:
                lam_m, lam_p = min(eig), max(eig)

            # bins = numpy.linspace(lam_m, lam_p, _auto_bins(eig))
            # _ = ax.hist(eig, bins, density=True, color='silver',
            #             edgecolor='none', label='Empirical Histogram')

            nbins = _auto_bins(eig)
            edges, vals = hist_ash(eig, bins=nbins, m=4,
                                   hist_range=(lam_m, lam_p), density=True)
            ax.stairs(vals, edges, fill=True, color='silver', alpha=1.0,
                      label='Empirical Histogram')
        else:
            plt.fill_between(x, y1=rho, y2=0, color='silver', zorder=-1)

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

def plot_hilbert(x, hilb, support=None, latex=False, save=False):
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

def plot_stieltjes(x, y, m1, m2, support, latex=False, save=False):
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
        shift = 0.0

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(domain_coloring(m1, shift=shift),
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
        ax1.imshow(domain_coloring(m2, shift=shift),
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

def plot_stieltjes_on_disk(r, t, m1_D, m2_D, support, latex=False, save=False):
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

        shift = 0.0

        ax0 = fig.add_subplot(gs[0], projection='polar')
        ax0.pcolormesh(grid_t, grid_r, domain_coloring(m1_D, shift=shift),
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
        ax1.pcolormesh(grid_t, grid_r, domain_coloring(m2_D, shift=shift),
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

        bins = numpy.linspace(x_min, x_max, _auto_bins(samples))
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
