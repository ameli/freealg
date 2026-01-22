import numpy


# =========
# Moments
# =========

class Moments(object):
    """
    Moments :math:`\\mu_n(t)` generated from eigenvalues, under
    free decompression, where

    .. math::

        m_n = \\mu_n(0) = \\mathbb{E}[\\lambda^n],

    and :math:`\\lambda` denotes an eigenvalue sample.

    Parameters
    ----------

    eig : array_like
        1D array of eigenvalues (or samples). Internally it is converted to a
        floating-point :class:`numpy.ndarray`.

    Attributes
    ----------

    eig : numpy.ndarray
        Eigenvalue samples.

    Methods
    -------

    m
        Compute the raw moment :math:`m_n = \\mathbb{E}[\\lambda^n]`.

    coeffs
        Compute the coefficient vector :math:`a_n`.

    __call__
        Evaluate :math:`\\mu_n(t)` for a given :math:`n` and :math:`t`.

    Notes
    -----

    The recursion memoizes:

    * Moments ``_m[n] = m_n``.
    * Coefficients ``_a[n] = a_n`` where ``a_n`` has length ``n`` and contains
      :math:`(a_{n,0}, \\dots, a_{n,n-1})`.

    The coefficient row :math:`a_n` is computed using an intermediate quantity
    :math:`R_{n,k}` formed via discrete convolutions of previous rows.

    Examples
    --------

    .. code-block:: python

        >>> import numpy as np
        >>> eig = np.array([1.0, 2.0, 3.0])
        >>> mu = MuFamily(eig)
        >>> mu(3, t=0.0)   # equals m_3
        12.0
        >>> mu(3, t=0.1)
        14.203...
    """

    # ====
    # init
    # ====

    def __init__(self, eig):
        """
        Initialization.
        """

        self.eig = numpy.asarray(eig, dtype=float)

        # Memoized moments m_n
        self._m = {0: 1.0}

        # Memoized coefficients a[n] = array of length n
        # (a_{n,0},...,a_{n,n-1})
        self._a = {0: numpy.array([1.0])}

    # ----------
    # moments
    # ----------

    def m(self, n):
        """
        Compute raw moment :math:`m_n`.

        Parameters
        ----------

        n : int
            Order of the moment.

        Returns
        -------

        m_n : float
            The raw moment :math:`m_n = \\mathbb{E}[\\lambda^n]`, estimated by
            the sample mean of ``eig**n``.
        """

        if n not in self._m:
            self._m[n] = numpy.mean(self.eig ** n)
        return self._m[n]

    # -------------
    # coefficients
    # -------------

    def coeffs(self, n):
        """
        Get coefficients :math:`a_n` for :math:`\\mu_n(t)`.

        Parameters
        ----------

        n : int
            Order of :math:`\\mu_n(t)`.

        Returns
        -------

        a_n : numpy.ndarray
            Array of shape ``(n,)`` containing :math:`(a_{n,0}, \\dots, a_{n,n-1})`.
        """

        if n in self._a:
            return self._a[n]

        # Ensure previous rows exist
        for r in range(1, n):
            if r not in self._a:
                self._compute_row(r)

        self._compute_row(n)
        return self._a[n]

    def _compute_row(self, n):
        """
        Compute and memoize the coefficient row :math:`a_n`.

        Parameters
        ----------

        n : int
            Row index to compute.

        Notes
        -----

        For :math:`n=1`, the row is

        .. math::

            a_{1,0} = m_1.

        For :math:`n \\ge 2`, let :math:`R_n` be a length ``n-1`` array defined
        by convolution of previous rows:

        .. math::

            R_n = \\sum_{i=1}^{n-1} (a_i * a_{n-i})\\big|_{0:(n-2)}.

        Then for :math:`k = 0, \\dots, n-2`,

        .. math::

            a_{n,k} = \\frac{1 + k/2}{(n-1-k)} R_{n,k},

        and the last coefficient is chosen so that :math:`\\mu_n(0)=m_n`:

        .. math::

            a_{n,n-1} = m_n - \\sum_{k=0}^{n-2} a_{n,k}.
        """

        if n in self._a:
            return

        if n == 1:
            self._a[1] = numpy.array([self.m(1)])
            return

        # Ensure all smaller rows exist
        for r in range(1, n):
            if r not in self._a:
                self._compute_row(r)

        a_n = numpy.zeros(n, dtype=float)

        # Compute R_{n,k} via convolutions:
        # R_n = sum_{i=1}^{n-1} convolve(a[i], a[n-i]) truncated to length n-1
        R = numpy.zeros(n - 1, dtype=float)
        for i in range(1, n):
            conv = numpy.convolve(self._a[i], self._a[n - i])
            R += conv[: n - 1]

        k = numpy.arange(n - 1, dtype=float)
        factors = (1.0 + 0.5 * k) / (n - 1 - k)
        a_n[: n - 1] = factors * R

        # k = n-1 from the initial condition mu_n(0) = m_n
        a_n[n - 1] = self.m(n) - a_n[: n - 1].sum()

        self._a[n] = a_n

    # ----------
    # evaluate
    # ----------

    def __call__(self, n, t=0.0):
        """
        Evaluate :math:`\\mu_n(t)`.

        Parameters
        ----------

        n : int
            Order of :math:`\\mu_n(t)`.

        t : float, default=0.0
            Deformation parameter.

        Returns
        -------

        mu_n : float
            The value of :math:`\\mu_n(t)`.

        Notes
        -----

        This function evaluates

        .. math::

            \\mu_n(t) = \\sum_{k=0}^{n-1} a_{n,k} \\, e^{k t}.

        For ``n == 0``, it returns ``1.0``.
        """

        if n == 0:
            return 1.0

        a_n = self.coeffs(n)
        k = numpy.arange(n, dtype=float)
        return numpy.dot(a_n, numpy.exp(k * t))
