import numpy
from scipy.stats import gaussian_kde

def detect_support(eigs, method='interior_smooth', k = None, p = 0.001, **kwargs):
    """
    Estimates the support of the eigenvalue density.

    Parameters
    ----------
    method : {``'range'``, ``'jackknife'``, ``'regression'``, ``'interior'``, 
                ``'interior_smooth'``}, \
            default= ``'jackknife'``
        The method of support estimation:

        * ``'range'``: no estimation; the support is the range of the eigenvalues
        * ``'jackknife'``: estimates the support using Quenouille's [1]
            jackknife estimator. Fast and simple, more accurate than the range.
        * ``'regression'``: estimates the support by performing a regression under
            the assumption that the edge behavior is of square-root type. Often
            most accurate.
        * ``'interior'``: estimates a support assuming the range overestimates;
            uses quantiles (p, 1-p).
        * ``'interior_smooth'``: same as ``'interior'`` but using kernel density
            estimation.

    k : int, default = None
        Number of extreme order statistics to use for ``method='regression'``. 
    
    p : float, default=0.001
        The edges of the support of the distribution is detected by the
        :math:`p`-quantile on the left and :math:`(1-p)`-quantile on the right
        where ``method='interior'`` or ``method='interior_smooth'``.
        This value should be between 0 and 1, ideally a small number close to
        zero.

    References
    ----------

    .. [1] Quenouille, M. H. (1949, July). Approximate tests of correlation in time-series. 
        In Mathematical Proceedings of the Cambridge Philosophical Society (Vol. 45, No. 3, 
        pp. 483-484). Cambridge University Press.
    """

    if method=='range':
        lam_m = eigs.min()
        lam_p = eigs.max()

    elif method=='jackknife':
        x, n = numpy.sort(eigs), len(eigs)
        lam_m = x[0]  - (n - 1)/n * (x[1]  - x[0])
        lam_p = x[-1] + (n - 1)/n * (x[-1] - x[-2])

    elif method=='regression':
        x, n = numpy.sort(eigs), len(eigs)
        if k is None:
            k = int(round(n ** (2/3)))
            k = max(5, min(k, n // 2))

        # The theoretical cdf near the edge behaves like const*(x - a)^{3/2},
        # so (i/n) ≈ (x - a)^{3/2}  ⇒  x ≈ a + const*(i/n)^{2/3}.
        y = ((numpy.arange(1, k + 1) - 0.5) / n) ** (2 / 3)

        # Left edge: regress x_{(i)} on y
        _, lam_m = numpy.polyfit(y, x[:k], 1)

        # Right edge: regress x_{(n-i+1)} on y
        _, lam_p = numpy.polyfit(y, x[-k:][::-1], 1)

    elif method=='interior':
        lam_m, lam_p = numpy.quantile(eigs, [p, 1-p])
    
    elif method=='interior_smooth':
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
