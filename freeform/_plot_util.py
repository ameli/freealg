# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
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
import colorsys
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

__all__ = ['plot_coeff', 'plot_density', 'plot_hilbert', 'plot_stieltjes']


# ==========
# plot coeff
# ==========

def plot_coeff(psi, latex=False, save=False):
    """
    """

    with texplot.theme(use_latex=latex):

        fig, ax = plt.subplots(figsize=(6, 3))
        n = numpy.arange(1, 1+psi.size)
        ax.plot(n, psi**2, '-o', markersize=3, color='black')
        ax.set_xlim([n[0], n[-1]])
        ax.set_xlabel(r'$i$')
        ax.set_ylabel(r'$\psi_i^2$')
        ax.set_title('Spectral Energy per Mode')
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
                save_filename = 'energy.pdf'

        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)


# ============
# plot density
# ============

def plot_density(x, rho, eig, support, latex=False, save=False):
    """
    """

    with texplot.theme(use_latex=latex):

        fig, ax = plt.subplots(figsize=(6, 2.7))

        lam_m, lam_p = support
        bins = numpy.linspace(lam_m, lam_p, 250)
        _ = ax.hist(eig, bins, density=True, color='silver',
                    edgecolor='none', label='Histogram')
        ax.plot(x, rho, color='black', label='Estimate', zorder=3)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\rho(\lambda)$''')
        ax.legend(fontsize='small')
        ax.set_xlim([min(x[0], lam_m), max(x[-1], lam_p)])
        ax.set_ylim(bottom=0)
        ax.set_title('Empirical Spectral Density')

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
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)


# ============
# plot hilbert
# ============

def plot_hilbert(x, hilb, support, latex=False, save=False):
    """
    """

    with texplot.theme(use_latex=latex):

        fig, ax = plt.subplots(figsize=(6, 2.7))

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
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)


# =======
# rgb hsv
# =======

def _rgb_hsv(c, shift=0.0, thresh=numpy.inf):
    """
    Convert complex field c to rgb through hsv channel.
    """

    hue = (numpy.angle(c) + numpy.pi) / (2.0 * numpy.pi)
    hue = (hue + shift) % 1.0
    saturation = numpy.ones_like(hue)

    # max_val = numpy.max(numpy.abs(c))
    # max_val = numpy.min([thresh, max_val])
    # value = numpy.abs(c) / max_val
    # value = numpy.where(value > 1, V, value)
    value = 1.0 - numpy.exp(-numpy.abs(c))

    hsv = numpy.stack((hue, saturation, value), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    return rgb


# =======
# rgb hsl
# =======

def _rgb_hsl(c, shift=0.0):
    """
    Convert complex field to rgb though hsl channel.
    """

    # Use constant lightness to avoid everything becoming white.
    hue = (numpy.angle(c) + numpy.pi) / (2.0 * numpy.pi)
    hue = (hue + shift) % 1.0
    lightness = 0.5 * numpy.ones_like(hue)
    saturation = numpy.ones_like(hue)
    f = numpy.vectorize(lambda h_, l_, s_: colorsys.hls_to_rgb(h_, l_, s_))
    r, g, b = f(hue, lightness, saturation)
    rgb = numpy.stack((r, g, b), axis=-1)

    return rgb


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
        ax0.imshow(_rgb_hsv(m1, shift=shift),
                   extent=[x_min, x_max, y_min, y_max], origin='lower',
                   interpolation='gaussian', rasterized=True)
        ax0.plot([lam_m, lam_p], [eps, eps], 'o', markersize=1.5,
                 color='black')
        ax0.plot([lam_m, lam_p], [eps, eps], '-', linewidth=1, color='black')
        ax0.set_xlabel(r'$\mathrm{Re}(z)$')
        ax0.set_ylabel(r'$\mathrm{Im}(z)$')
        ax0.set_title(r'(a) Principal Branch on $\mathbb{H}$ and ' +
                      r'$\mathbb{H}^{-}$')
        ax0.set_yticks(numpy.arange(y_min, y_max+0.01, 0.5))
        ax0.set_xlim([x_min, x_max])
        ax0.set_ylim([y_min, y_max])

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(_rgb_hsv(m2, shift=shift),
                   extent=[x_min, x_max, y_min, y_max], origin='lower',
                   interpolation='gaussian', rasterized=True)
        ax1.plot([lam_m, lam_p], [eps, eps], 'o', markersize=1.5,
                 color='black')
        ax1.plot([x_min, lam_m], [eps, eps], '-', linewidth=1, color='black')
        ax1.plot([lam_p, x_max], [eps, eps], '-', linewidth=1, color='black')
        ax1.set_xlabel(r'$\mathrm{Re}(z)$')
        ax1.set_ylabel(r'$\mathrm{Im}(z)$')
        ax1.set_title(r'(b) Principal Branch on $\mathbb{H}$, Secondary ' +
                      r'Branch on $\mathbb{H}^{-}$')
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
                                  transparent_background=True, dpi=200,
                                  show_and_save=save_status, verbose=True)
