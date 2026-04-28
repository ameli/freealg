# =======
# Imports
# =======

import math
import numpy
import matplotlib.pyplot as plt
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

__all__ = ['deform_coeffs', 'plot_deform_candidates']

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


# ==================
# deform coeffs core
# ==================

@njit(cache=numba_cache)
def _deform_coeffs_core(a, t, c0):
    l_degree = a.shape[0] - 1
    k_degree = a.shape[1] - 1

    tau = math.exp(t)
    a_shift = 1.0 - tau
    b_shift = 1.0 - tau * c0

    z_degree = l_degree + k_degree
    eta_degree = l_degree + 2 * k_degree
    a_out = numpy.zeros((z_degree + 1, eta_degree + 1),
                        dtype=numpy.complex128)

    for j in range(l_degree + 1):
        tau_weight = tau ** (-j)
        for k in range(k_degree + 1):
            coeff = a[j, k] * tau_weight * (c0 ** (-k))
            if coeff == 0.0:
                continue

            # The cleared deformation polynomial is
            #
            #   eta^L (z eta + 1 - tau)^K
            #       P((z eta + 1 - tau) / (tau eta),
            #         eta (z eta + 1 - tau c0)
            #             / (c0 (z eta + 1 - tau))).
            #
            # Therefore each monomial a[j,k] zeta^j m^k contributes
            #
            #   a[j,k] tau^(-j) c0^(-k)
            #   eta^(L + k - j)
            #   (z eta + 1 - tau)^(j + K - k)
            #   (z eta + 1 - tau c0)^k.
            p_degree = j + k_degree - k
            eta_base = l_degree + k - j

            for r in range(p_degree + 1):
                weight_r = comb(p_degree, r) * \
                    (a_shift ** (p_degree - r))
                if weight_r == 0.0:
                    continue

                for q in range(k + 1):
                    weight_q = comb(k, q) * (b_shift ** (k - q))
                    if weight_q == 0.0:
                        continue

                    z_pow = r + q
                    eta_pow = eta_base + r + q
                    a_out[z_pow, eta_pow] += coeff * weight_r * weight_q

    return a_out


# =============
# deform_coeffs
# =============

def deform_coeffs(a, t, c0, normalize=True):
    """
    Compute the deformed coefficients A[r, s](t) induced by the
    companion-coordinate deformation transform.

    The input polynomial is

        P(z, m) = sum_{j=0..L} sum_{k=0..K} a[j, k] z^j m^k.

    The deformation relation uses the target companion variable eta and
    the initial aspect ratio c0:

        zeta = (z eta + 1 - tau) / (tau eta),
        m0   = eta (z eta + 1 - tau c0)
               / (c0 (z eta + 1 - tau)),
        tau  = exp(t).

    The returned polynomial is the denominator-cleared relation

        Q_t(z, eta)
            = eta^L (z eta + 1 - tau)^K P(zeta, m0),

    whose roots in eta are candidate companion Stieltjes branches at the
    target aspect ratio c = tau c0.

    Parameters
    ----------
    a : array_like of float, shape (L+1, K+1)
        Coefficients defining P(z, m) in the monomial basis:
            P(z, m) = sum_{j=0..L} sum_{k=0..K} a[j, k] z^j m^k.
    t : float
        Time parameter, with tau = exp(t).
    c0 : float
        Initial aspect ratio. Must be positive.
    normalize : bool, default=True
        If True, normalize the output coefficients using
        ``_normalize_coefficients``.

    Returns
    -------
    A : ndarray, shape (L+K+1, L+2K+1)
        Coefficients A[r, s](t) such that
            sum_{r=0..L+K} sum_{s=0..L+2K} A[r, s](t) z^r eta^s = 0,
        normalized by normalize_coefficients when ``normalize`` is True.
    """

    a = numpy.asarray(a)
    if a.ndim != 2:
        raise ValueError("a must be a 2D array-like of shape (L+1, K+1).")

    if not (isinstance(c0, (float, int)) and c0 > 0):
        raise ValueError("c0 must be a positive scalar.")

    a = numpy.array(a, dtype=numpy.complex128, copy=True)
    a[-1, 0] = 0.0

    a_out = _deform_coeffs_core(a, float(t), float(c0))

    if normalize:
        return _normalize_coefficients(a_out)

    return a_out


# ======================
# plot deform candidates
# ======================

def plot_deform_candidates(a, x, c=1.0, eig=None, delta=1e-4, size=None,
                           log=False, markersize=3, ylim=None, latex=False,
                           verbose=False):
    """
    Plot candidate roots.
    """

    if not (isinstance(delta, (float, int)) and delta > 0):
        raise ValueError("delta must be a positive scalar.")

    if not (isinstance(c, (float, int)) and c > 0):
        raise ValueError("c must be a positive scalar.")

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
        # Roots are companion branches eta. Convert to the target
        # non-companion Stieltjes branch m = (eta - (c - 1) / z) / c
        # before plotting densities.
        m_roots = (roots[idx] - (float(c) - 1.0) / z[idx]) / float(c)
        im = numpy.imag(m_roots)
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
        fig, ax = plt.subplots(figsize=(6, 2.7))
        ax.scatter(xs, ys, s=markersize, alpha=1, linewidths=0, c='k',
                   zorder=2, label='Roots')

        ax.set_xlim([x[0], x[-1]])

        if (eig is not None):
            lam_m, lam_p = min(eig), max(eig)

            if log:
                nbins = auto_bins(numpy.log10(eig))
                bins = numpy.geomspace(lam_m, lam_p, nbins)
            else:
                nbins = auto_bins(eig)
                bins = numpy.linspace(lam_m, lam_p, nbins)
            _ = ax.hist(eig, bins, density=True, color='royalblue', alpha=0.5,
                        edgecolor='none', label='Empirical Histogram',
                        zorder=1)

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
            ax.grid(which='both', axis='x')

        # Save
        if size is not None:
            ax.set_title("Candidate Density Cloud (size = {})".format(size))
        save_status = False
        save_filename = ''
        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)
