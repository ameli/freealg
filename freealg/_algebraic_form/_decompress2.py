# =======
# Imports
# =======

import numpy
import matplotlib.pyplot as plt
from scipy.special import comb
import texplot
from ._continuation_algebraic import _normalize_coefficients

__all__ = ['decompress_coeffs']


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
    a[-1, 0] = 0.0
    if a.ndim != 2:
        raise ValueError("a must be a 2D array-like of shape (L+1, K+1).")

    l_degree = a.shape[0] - 1
    k_degree = a.shape[1] - 1

    c = 1.0 - numpy.exp(-t)

    # Scale columns of a by e^{t k}: scaled[j, k] = a[j, k] e^{t k}.
    exp_factors = numpy.exp(numpy.arange(k_degree + 1) * t)
    scaled = a * exp_factors

    # Output coefficients.
    out_dtype = numpy.result_type(a, float)
    a_out = numpy.zeros((l_degree + 1, l_degree + k_degree + 1),
                        dtype=out_dtype)

    # Precompute binomial(j, r) * c^{j-r} for all j, r (lower-triangular).
    j_inds = numpy.arange(l_degree + 1)[:, None]
    r_inds = numpy.arange(l_degree + 1)[None, :]
    mask = r_inds <= j_inds

    binom_weights = numpy.zeros((l_degree + 1, l_degree + 1), dtype=float)
    binom_weights[mask] = comb(j_inds, r_inds, exact=False)[mask]
    binom_weights[mask] *= (c ** (j_inds - r_inds))[mask]

    # Main accumulation:
    # For fixed j and r, add:
    #   A[r, (L - j + r) + k] += binom_weights[j, r] * scaled[j, k],
    # for k = 0..K.
    for j in range(l_degree + 1):
        row_scaled = scaled[j]
        if numpy.all(row_scaled == 0):
            continue

        base0 = l_degree - j
        row_b = binom_weights[j]

        for r in range(j + 1):
            coeff = row_b[r]
            if coeff == 0:
                continue

            start = base0 + r
            a_out[r, start:start + (k_degree + 1)] += coeff * row_scaled

    if normalize:
        return _normalize_coefficients(a_out)

    return a_out


def plot_candidates(a, x, delta=1e-4, size=None, latex=False, verbose=False):
    """
    Visualize candidate densities implied by an algebraic Stieltjes-transform
    relation:
        P(z, m) = sum_{i=0..I} sum_{j=0..J} a[i, j] z^i m^j,
    where m(z) is defined implicitly by P(z, m(z)) = 0.

    For each grid point x_k, set z = x_k + i * delta, form the polynomial in m
    given by P(z, m) = 0, solve for its roots, and plot the cloud of candidate
    densities:
        (1 / pi) * Im(m_root),
    keeping only roots with Im(m_root) > 0 (roots are not tracked/paired across
    x-values).

    Parameters
    ----------
    a : array_like of complex or float, shape (I+1, J+1)
        Coefficients defining P(z, m) in the monomial basis:
            P(z, m) = sum_{i=0..I} sum_{j=0..J} a[i, j] z^i m^j.
    x : array_like of float, shape (N,)
        1D array of real x-values (evaluation grid).
    delta : float, optional
        Small positive imaginary offset used to evaluate m(x + i * delta).
    size : integer, optional
        For labelling purposes, the size of the corresponding matrix can
        be provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes the scatter plot was drawn on.
    """

    if not (isinstance(delta, (float, int)) and delta > 0):
        raise ValueError("delta must be a positive scalar.")

    x = numpy.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1D NumPy array.")

    a = numpy.asarray(a)
    if a.ndim != 2:
        raise ValueError("a must be a 2D NumPy array with a[i, j] coefficients.")
    if not numpy.issubdtype(a.dtype, numpy.number):
        raise ValueError("a must be numeric.")

    i_degree = a.shape[0] - 1

    xs = []
    ys = []
    max_ys = numpy.zeros_like(x)

    # Precompute i-powers indices to avoid repeated arange creation.
    i_idx = numpy.arange(i_degree + 1)

    for idx, xk in enumerate(x):
        z = complex(float(xk), float(delta))  # x + i * delta

        # b[j] = sum_i a[i, j] * z^i  => polynomial in m:
        #   sum_{j=0..J} b[j] m^j = 0
        z_pows = z ** i_idx  # length I+1
        b = (z_pows[:, None] * a).sum(axis=0)  # length J+1, low->high in m

        # Trim trailing (highest-degree) coefficients near zero to avoid
        # numerical issues in numpy.roots. b is low->high, so trim from end.
        tol = 1e-14
        b_trim = b.copy()
        while b_trim.size > 1 and abs(b_trim[-1]) < tol:
            b_trim = b_trim[:-1]

        # If constant polynomial (no roots), skip.
        if b_trim.size <= 1:
            continue

        # numpy.roots expects highest degree first.
        coeffs_high_to_low = b_trim[::-1]
        roots = numpy.roots(coeffs_high_to_low)

        # Keep only roots with positive imaginary part.
        im = numpy.imag(roots)
        mask = im > 0
        if numpy.any(mask):
            xs.append(numpy.full(mask.sum(), float(xk)))
            ys.append(im[mask] / numpy.pi)
            max_ys[idx] = max(ys[-1])

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
        ax.scatter(xs, ys, s=8, alpha=1, linewidths=0, c='k')

        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\rho(\lambda)$''')
        ax.set_title("Candidate Density Cloud")
        if size is not None:
            ax.set_title("Candidate Density Cloud (size = {})".format(size))
        ax.grid(True, alpha=1)
        save_status = False
        save_filename = ''
        texplot.show_or_save_plot(plt, default_filename=save_filename,
                                  transparent_background=True, dpi=400,
                                  show_and_save=save_status, verbose=True)
