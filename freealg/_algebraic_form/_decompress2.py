# =======
# Imports
# =======

import numpy
from scipy.special import comb
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
