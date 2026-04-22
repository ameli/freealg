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

from ._decompress_coeffs2 import decompress_coeffs
from ._continuation_algebraic import eval_roots
from ._decompress_util import build_time_grid

__all__ = ['plot_decompress_vs_candidates']


# ===============
# poly coeff in w
# ===============

def poly_coeffs_in_w(coeffs, z):
    """
    For fixed z, convert bivariate coeffs[i,j] of
        P(z,w) = sum_{i,j} coeffs[i,j] z^i w^j
    into univariate coefficients a[j] of
        p(w) = sum_j a[j] w^j.
    """

    coeffs = numpy.asarray(coeffs)
    deg_z, deg_w = coeffs.shape
    a = numpy.zeros(deg_w, dtype=complex)
    z_pows = numpy.array([z**i for i in range(deg_z)], dtype=complex)
    for j in range(deg_w):
        a[j] = numpy.sum(coeffs[:, j] * z_pows)
    return a


# ========
# p_w eval
# ========

def p_w_eval(a, w):
    """
    Evaluate derivative p'(w) for p(w)=sum_j a[j] w^j.
    """

    if len(a) <= 1:
        return 0.0j
    j = numpy.arange(1, len(a))
    return numpy.sum(j * a[1:] * (w ** (j - 1)))


# =======================
# normalized pw indicator
# =======================

def normalized_pw_indicator(a, w):
    """
    eta(w) = |p'(w)| / sum_{j>=1} j |a_j| |w|^{j-1}
    for p(w) = sum_j a[j] w^j
    """

    if len(a) <= 1:
        return 1.0

    # numerator = |p'(w)|
    j = numpy.arange(1, len(a))
    pm = numpy.sum(j * a[1:] * (w ** (j - 1)))
    num = abs(pm)

    # denominator
    aw = abs(w)
    denom = 0.0
    pw = 1.0
    for jj in range(1, len(a)):
        denom += jj * abs(a[jj]) * pw
        pw *= aw

    if (not numpy.isfinite(denom)) or denom <= 0.0:
        return 1.0

    return num / denom


# =============================
# plot decompress vs candidates
# =============================

def plot_decompress_vs_candidates(af, decompress_newton, sizes, x,
                                  min_n_times=100, newton_opt=None, t_lim=None,
                                  re_lim=None, im_lim=None, pm_lim=None):
    """
    """

    if newton_opt is None:
        newton_opt = {}

    # Requested times exactly as af.decompress sees them
    sizes = numpy.asarray(sizes, dtype=float)
    t_all, idx_req = build_time_grid(
        sizes, af.n, min_n_times=min_n_times)

    # # Use the smallest delta branch as the raw geometric branch to overlay
    delta_ladder = numpy.asarray(af.delta_ladder, dtype=float)
    delta = delta_ladder[0]

    x = numpy.array([float(x)], dtype=float)

    if af.inv_stieltjes_opt['z_query_delta'] == 'linear':
        z_query = x.astype(complex) * (1.0 + 1j * delta)
    elif af.inv_stieltjes_opt['z_query_delta'] == 'const':
        z_query = x.astype(complex) + 1j * delta
    else:
        raise ValueError('z_query_delta is invalid.')

    w0_list = af._stieltjes_poly(z_query)
    w, ok = decompress_newton(z_query, t_all, af.coeffs, w0_list=w0_list,
                              **newton_opt)

    # Candidate roots at the SAME z_query as the raw branch
    roots = []
    for tt in t_all:
        coeffs_t = decompress_coeffs(af.coeffs, float(tt))
        roots_at_t = eval_roots(z_query, coeffs_t)[0]
        roots.append(numpy.asarray(roots_at_t, dtype=complex).ravel())

    # pad if first row has fewer roots
    max_n = max(r.size for r in roots)
    roots_pad = numpy.full((len(roots), max_n), 0.0 + 1j * 0.0)
    for i in range(len(roots)):
        roots_pad[i, :roots[i].size] = roots[i]

    # Normalized eta for all roots
    eta = []
    for tt, roots_t in zip(t_all, roots):
        coeffs_t = decompress_coeffs(af.coeffs, float(tt))
        a_w = poly_coeffs_in_w(coeffs_t, z_query[0])

        vals = numpy.empty(roots_t.size, dtype=float)
        for j in range(roots_t.size):
            vals[j] = normalized_pw_indicator(a_w, roots_t[j])
        eta.append(vals)

    # Pad eta
    eta_pad = numpy.full((len(roots), max_n), 0.0)
    for i in range(len(roots)):
        eta_pad[i, :eta[i].size] = eta[i]

    idx = numpy.argsort(roots_pad.real, axis=1)
    roots_pad = numpy.take_along_axis(roots_pad, idx, axis=1)
    eta_pad = numpy.take_along_axis(eta_pad, idx, axis=1)
    color_idx = numpy.broadcast_to(numpy.arange(roots_pad.shape[1]),
                                   roots_pad.shape)

    fig, ax = plt.subplots(nrows=3, figsize=(8, 12), sharex=True)

    ms = 1.5
    lw = 4
    cmap = plt.cm.tab10

    # Candidate roots
    tt = numpy.broadcast_to(t_all[:, None], roots_pad.shape)
    ax[0].scatter(tt.ravel(), roots_pad.real.ravel(), c=color_idx.ravel(),
                  s=ms, cmap=cmap)
    ax[1].scatter(tt.ravel(), roots_pad.imag.ravel(), c=color_idx.ravel(),
                  s=ms, cmap=cmap)
    ax[2].scatter(tt.ravel(), eta_pad.ravel(), c=color_idx.ravel(), s=ms,
                  cmap=cmap)

    ax[2].axhline(newton_opt.get("pair_enter_eta", 1e-4), linestyle='--',
                  color="black", lw=1, label="pair_enter_eta")

    # Tracked root using decompress
    ax[0].plot(t_all, w.real, "-o", color="black", lw=lw, zorder=-1,
               label="Re(m)")
    ax[1].plot(t_all, w.imag, "-o", color="black", lw=lw, zorder=-1,
               label="Im(m)")

    # Poisson floor for the smallest delta level
    floor_im = delta / (x**2 + delta**2)
    ax[1].axhline(floor_im, linestyle='--', color="black", lw=1,
                  label="Poisson floor")

    for i in range(len(ax)):
        ax[i].legend(fontsize='x-small', loc='lower left')
        ax[i].grid(True)
        if i == len(ax) - 1:
            ax[i].set_xlabel("t")
        else:
            ax[i].tick_params(axis='x', which='both', bottom=False,
                              labelbottom=False)
        if t_lim is not None:
            ax[i].set_xlim(t_lim)

    ax[0].set_ylabel(r'$\mathrm{Re}(m)$')
    ax[1].set_ylabel(r'$\mathrm{Im}(m)$')
    ax[2].set_ylabel(r'$|\eta(m)|$')

    # ax[0].set_yscale("symlog", linthresh=1e-6)
    # ax[1].set_yscale("log")
    ax[1].set_yscale("symlog", linthresh=1e-6)
    ax[2].set_yscale('log')

    if re_lim is not None:
        ax[0].set_ylim(re_lim)

    if im_lim is not None:
        ax[1].set_ylim(im_lim)
    # else:
    #     ax[1].set_ylim(bottom=1e-9)

    if pm_lim is not None:
        ax[2].set_ylim(pm_lim)

    fig.tight_layout(h_pad=0.1)
    plt.show()

    return t_all, roots, w, eta_pad
