# =======
# Imports
# =======

import math
import numpy
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import texplot
from ._continuation_algebraic import _normalize_coefficients, eval_roots
from ..visualization._hist_util import auto_bins

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(func):
            return func
        return deco

__all__ = ['decompress_coeffs', 'plot_decompress_candidates']

numba_cache = False


# ====
# comb
# ====

@njit(cache=numba_cache)
def comb(n, k):
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    kk = k
    if kk > n - kk:
        kk = n - kk
    out = 1.0
    for i in range(1, kk + 1):
        out = out * (n - kk + i) / i
    return out


# ======================
# decompress coeffs core
# ======================

@njit(cache=numba_cache)
def _decompress_coeffs_core(a, t):
    l_degree = a.shape[0] - 1
    k_degree = a.shape[1] - 1

    c = 1.0 - math.exp(-t)
    a_out = numpy.zeros((l_degree + 1, l_degree + k_degree + 1),
                        dtype=numpy.complex128)

    for j in range(l_degree + 1):
        for k in range(k_degree + 1):
            coeff = a[j, k] * math.exp(k * t)
            if coeff == 0.0:
                continue

            base0 = l_degree - j + k
            for r in range(j + 1):
                weight = comb(j, r) * (c ** (j - r))
                if weight == 0.0:
                    continue
                a_out[r, base0 + r] += weight * coeff

    return a_out


# =================
# decompress_coeffs
# =================

def decompress_coeffs(a, t, normalize=True):
    """
    Compute the decompressed coefficients A[r, s](t) induced by
    the transform Q_t(z, m) = m^L P(z + (1 - e^{-t}) / m, e^t m).

    Parameters
    ----------
    a : array_like of float, shape (L+1, K+1)
        Coefficients defining P(z, m) in the monomial basis:
            P(z, m) = sum_{j=0..L} sum_{k=0..K} a[j, k] z^j m^k.
    t : float
        Time parameter.

    Returns
    -------
    A : ndarray, shape (L+1, L+K+1)
        Coefficients A[r, s](t) such that
            sum_{r=0..L} sum_{s=0..L+K} A[r, s](t) z^r m^s = 0,
        normalized by normalize_coefficients.
    """

    a = numpy.asarray(a)
    if a.ndim != 2:
        raise ValueError("a must be a 2D array-like of shape (L+1, K+1).")

    a = numpy.array(a, dtype=numpy.complex128, copy=True)
    a[-1, 0] = 0.0

    a_out = _decompress_coeffs_core(a, float(t))

    if normalize:
        return _normalize_coefficients(a_out)

    return a_out


# ==========================
# plot decompress candidates
# ==========================

def plot_decompress_candidates(a, x, ax=None, eig=None, delta=1e-4, size=None,
                               log=False, markersize=3, ylim=None,
                               latex=False, verbose=False):
    """
    Plot candidate roots.
    """

    if not (isinstance(delta, (float, int)) and delta > 0):
        raise ValueError("delta must be a positive scalar.")

    x = numpy.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1D NumPy array.")

    a = numpy.asarray(a)
    if a.ndim != 2:
        raise ValueError("a must be a 2D NumPy array with a[i, j] coeffs.")
    if not numpy.issubdtype(a.dtype, numpy.number):
        raise ValueError("a must be numeric.")

    xs = []
    ys = []
    max_ys = numpy.zeros_like(x, dtype=float)

    z = x.astype(complex) + 1j * float(delta)
    roots = eval_roots(z, a, dtype=complex)

    for idx in range(x.size):
        im = numpy.imag(roots[idx])
        mask = im > 0

        if numpy.any(mask):
            dens = im[mask] / numpy.pi
            xs.append(numpy.full(mask.sum(), float(x[idx])))
            ys.append(dens)
            max_ys[idx] = numpy.max(dens)

    if verbose:
        max_density = numpy.trapezoid(max_ys, x)
        print("Max density: {}".format(max_density))

    if xs:
        xs = numpy.concatenate(xs)
        ys = numpy.concatenate(ys)
    else:
        xs = numpy.array([], dtype=float)
        ys = numpy.array([], dtype=float)

    with texplot.theme(use_latex=latex):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 2.7))
            external_ax = False
        else:
            fig = ax.get_figure()
            external_ax = True

        ax.scatter(xs, ys, s=markersize, alpha=1, linewidths=0, c='k',
                   zorder=2, label='Roots', rasterized=True)

        ax.set_xlim([x[0], x[-1]])

        if (eig is not None):
            lam_m, lam_p = min(eig), max(eig)

            if log:
                nbins = auto_bins(numpy.log10(eig))
                bins = numpy.geomspace(lam_m, lam_p, nbins)
            else:
                nbins = auto_bins(eig)
                bins = numpy.linspace(lam_m, lam_p, nbins)
            _ = ax.hist(eig, bins, density=True, color='lightsteelblue',
                        # alpha=0.5,
                        edgecolor='none', label='Empirical Histogram',
                        zorder=-1, rasterized=True)

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            if ys.size > 0:
                max_y = numpy.quantile(ys, 0.999)
                if eig is not None:
                    max_y = max(max_y, numpy.quantile(eig, 0.999))

                if log:
                    min_y = numpy.quantile(ys, 0.001)
                    min_y = numpy.max([min_y, 1e-16])
                    ax.set_ylim([min_y, 5.0 * max_y])
                else:
                    ax.set_ylim([0, 1.1 * max_y])
            else:
                if log:
                    ax.set_ylim(bottom=1e-16)
                else:
                    ax.set_ylim([0, 1])

        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\rho(\lambda)$')
        ax.set_title("Candidate Density Cloud")

        if log:
            ax.legend(fontsize='x-small', markerscale=4.0, loc='lower left')
        else:
            ax.legend(fontsize='x-small', markerscale=4.0, loc='upper right')

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            # ax.grid(which='both', axis='x')

        if not external_ax:

            # Zero pad on left, right, and top of canvas
            fig.canvas.draw()
            bbox = fig.get_tightbbox(fig.canvas.get_renderer())
            pad = 0.75 / 72.0
            bbox = mtransforms.Bbox.from_extents(bbox.x0-pad, bbox.y0-pad,
                                                 bbox.x1+pad, bbox.y1+pad)

            # Save
            if size is not None:
                ax.set_title(
                    rf'Candidate Density Cloud ($n = {{{int(size/1000)}}}$K)')
            save_status = False
            save_filename = ''
            texplot.show_or_save_plot(plt, default_filename=save_filename,
                                      transparent_background=True, dpi=400,
                                      bbox_inches=bbox,
                                      show_and_save=save_status, verbose=True)
