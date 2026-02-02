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
# import numba
from scipy.stats import gaussian_kde

__all__ = ['support_from_density', 'supp']


# ====================
# support from density
# ====================

# @numba.njit(numba.types.UniTuple(numba.types.int64, 2)(
#     numba.types.float64,
#     numba.types.float64[::1]
# ))
def support_from_density(dx, density):
    """
    Estimates the support from a collection of noisy observations of a
    density over a grid of x-values with mesh spacing dx.
    """

    n = density.shape[0]
    target = 1.0 / dx

    # compute total_sum once
    total_sum = 0.0
    for t in range(n):
        total_sum += density[t]

    # set up our "best-so-far" trackers
    large = 1e300
    best_nonneg_sum = large
    best_nonneg_idx = -1
    best_nonpos_sum = -large
    best_nonpos_idx = -1

    # seed with first element (i.e. prefix_sum for k=1)
    prefix_sum = density[0]
    if prefix_sum >= 0.0:
        best_nonneg_sum, best_nonneg_idx = prefix_sum, 1
    else:
        best_nonpos_sum, best_nonpos_idx = prefix_sum, 1

    # sweep j from 2, ..., n-1, updating prefix_sum on the fly
    optimal_i, optimal_j = 1, 2
    minimal_cost = large

    for j in range(2, n):
        # extend prefix_sum to cover density[0]...density[j-1]
        prefix_sum += density[j-1]

        # cost for [0...i], [i...j]
        diff_mid = prefix_sum - target
        if diff_mid >= 0.0 and best_nonneg_sum <= diff_mid:
            cost12 = diff_mid
            i_cand = best_nonneg_idx
        elif diff_mid < 0.0 and best_nonpos_sum >= diff_mid:
            cost12 = -diff_mid
            i_cand = best_nonpos_idx
        else:
            cost_using_nonpos = diff_mid - 2.0 * best_nonpos_sum
            cost_using_nonneg = 2.0 * best_nonneg_sum - diff_mid
            if cost_using_nonpos < cost_using_nonneg:
                cost12, i_cand = cost_using_nonpos, best_nonpos_idx
            else:
                cost12, i_cand = cost_using_nonneg, best_nonneg_idx

        # cost for [j...n]
        cost3 = total_sum - prefix_sum
        if cost3 < 0.0:
            cost3 = -cost3

        # total and maybe update best split
        total_cost = cost12 + cost3
        if total_cost < minimal_cost:
            minimal_cost = total_cost
            optimal_i, optimal_j = i_cand, j

        # update our prefix-sum trackers
        if prefix_sum >= 0.0:
            if prefix_sum < best_nonneg_sum:
                best_nonneg_sum, best_nonneg_idx = prefix_sum, j
        else:
            if prefix_sum > best_nonpos_sum:
                best_nonpos_sum, best_nonpos_idx = prefix_sum, j

    return optimal_i, optimal_j


# ====
# supp
# ====

def supp(eigs, method='asymp', k=None, p=0.001):
    """
    Estimates the support of the eigenvalue density.

    Parameters
    ----------

    method : {``'range'``, ``'asymp'``, ``'jackknife'``, ``'regression'``, \
            ``'interior'``, ``'interior_smooth'``}, default= ``'asymp'``
        The method of support estimation:

        * ``'range'``: no estimation; the support is the range of the
          eigenvalues.
        * ``'asymp'``: assume the relative error in the min/max estimator is
          :math:`1/n`.
        * ``'jackknife'``: estimates the support using Quenouille's [1]_
          jackknife estimator. Fast and simple, more accurate than the range.
        * ``'regression'``: estimates the support by performing a regression
          under the assumption that the edge behavior is of square-root type.
          Often most accurate.
        * ``'interior'``: estimates a support assuming the range overestimates;
          uses quantiles :math:`(p, 1-p)`.
        * ``'interior_smooth'``: same as ``'interior'`` but using kernel
          density estimation, from [2]_.

    k : int, default = None
        Number of extreme order statistics to use for ``method='regression'``.

    p : float, default=0.001
        The edges of the support of the distribution is detected by the
        :math:`p`-quantile on the left and :math:`(1-p)`-quantile on the right
        where ``method='interior'`` or ``method='interior_smooth'``.
        This value should be between 0 and 1, ideally a small number close to
        zero.

    Returns
    -------

    lam_m : float
        Lower end of support interval :math:`[\\lambda_{-}, \\lambda_{+}]`.

    lam_p : float
        Upper end of support interval :math:`[\\lambda_{-}, \\lambda_{+}]`.

    See Also
    --------

    freealg.sample
    freealg.kde

    References
    ----------

    .. [1] Quenouille, M. H. (1949). Approximate tests of correlation in
           time-series. In Mathematical Proceedings of the Cambridge
           Philosophical Society (Vol. 45, No. 3, pp. 483-484). Cambridge
           University Press.

    .. [2] Cuevas, A., & Fraiman, R. (1997). A plug-in approach to support
           estimation. The Annals of Statistics, 2300-2312.
    """

    if method == 'range':
        lam_m = eigs.min()
        lam_p = eigs.max()

    elif method == 'asymp':
        lam_m = eigs.min() - abs(eigs.min()) / len(eigs)
        lam_p = eigs.max() + abs(eigs.max()) / len(eigs)

    elif method == 'jackknife':
        x, n = numpy.sort(eigs), len(eigs)
        lam_m = x[0] - (n - 1)/n * (x[1] - x[0])
        lam_p = x[-1] + (n - 1)/n * (x[-1] - x[-2])

    elif method == 'regression':
        x, n = numpy.sort(eigs), len(eigs)
        if k is None:
            k = int(round(n ** (2/3)))
            k = max(5, min(k, n // 2))

        # The theoretical cdf near the edge behaves like const*(x - a)^{3/2},
        # so (i/n) ~ (x - a)^{3/2}  ->  x ~ a + const*(i/n)^{2/3}.
        y = ((numpy.arange(1, k + 1) - 0.5) / n) ** (2 / 3)

        # Left edge: regress x_{(i)} on y
        _, lam_m = numpy.polyfit(y, x[:k], 1)

        # Right edge: regress x_{(n-i+1)} on y
        _, lam_p = numpy.polyfit(y, x[-k:][::-1], 1)

    elif method == 'interior':
        lam_m, lam_p = numpy.quantile(eigs, [p, 1-p])

    elif method == 'interior_smooth':
        kde = gaussian_kde(eigs)
        xs = numpy.linspace(eigs.min(), eigs.max(), 1000)
        fs = kde(xs)

        cdf = numpy.cumsum(fs)
        cdf /= cdf[-1]

        lam_m = numpy.interp(p, cdf, xs)
        lam_p = numpy.interp(1-p, cdf, xs)

    else:
        raise NotImplementedError("Unknown method")

    return lam_m, lam_p
