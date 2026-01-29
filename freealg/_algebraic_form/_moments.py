# =======
# Imports
# =======

import numpy


# =======
# Moments
# =======

class Moments(object):
    """
    Moments :math:`\\mu_n(t)` generated from eigenvalues, under
    free decompression, where

    .. math::

        m_n = \\mu_n(0) = \\mathbb{E}[\\lambda^n],

    and :math:`\\lambda` denotes an eigenvalue sample.

    Parameters
    ----------

    source : array_like or callable
        Either

        * a 1D array of eigenvalues (or samples), or
        * a callable returning the raw moments at zero, ``source(n) = m_n``.

        If an array is provided, moments are estimated via sample averages.
        If a callable is provided, it is assumed to return exact values of
        :math:`m_n`.

    Attributes
    ----------

    eig : numpy.ndarray or None
        Eigenvalue samples, if provided.

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
    """

    # ====
    # init
    # ====

    def __init__(self, source):
        """
        Initialization.
        """
        self.eig = None
        self._moment_fn = None

        if callable(source):
            self._moment_fn = source
        else:
            self.eig = numpy.asarray(source, dtype=float)

        # Memoized moments m_n
        self._m = {0: 1.0}

        # Memoized coefficients a[n] = array of length n
        # (a_{n,0},...,a_{n,n-1})
        self._a = {0: numpy.array([1.0])}

    # =
    # m
    # =

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
            The raw moment :math:`m_n = \\mathbb{E}[\\lambda^n]`.

        Notes
        -----

        If the instance was initialized with eigenvalue samples, the moment is
        estimated by the sample mean of ``eig**n``. If initialized with a
        callable, the callable is used directly.
        """
        n = int(n)
        if n < 0:
            raise ValueError("Moment order n must be >= 0.")

        if n not in self._m:
            if self._moment_fn is not None:
                self._m[n] = float(self._moment_fn(n))
            else:
                self._m[n] = float(numpy.mean(self.eig ** n))

        return self._m[n]

    # ======
    # coeffs
    # ======

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
            Array of shape ``(n,)`` containing :math:`(a_{n,0},
            \\dots, a_{n,n-1})`.
        """
        n = int(n)
        if n < 0:
            raise ValueError("Order n must be >= 0.")

        if n in self._a:
            return self._a[n]

        # Ensure previous rows exist
        for r in range(1, n):
            if r not in self._a:
                self._compute_row(r)

        self._compute_row(n)
        return self._a[n]

    # ===========
    # compute row
    # ===========

    def _compute_row(self, n):
        """
        Compute and memoize the coefficient row :math:`a_n`.

        Parameters
        ----------

        n : int
            Row index to compute.
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

        # Compute R_{n,k} via convolutions
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

    # --------
    # evaluate
    # --------

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
        n = int(n)
        if n < 0:
            raise ValueError("Order n must be >= 0.")
        if n == 0:
            return 1.0

        a_n = self.coeffs(n)
        k = numpy.arange(n, dtype=float)
        return float(numpy.dot(a_n, numpy.exp(k * t)))


# ===========================
# Algebraic Stieltjes Moments
# ===========================

class AlgebraicStieltjesMoments(object):
    """
    Given coefficients a[i,j] for P(z,m)=sum_{i,j} a[i,j] z^i m^j,
    compute the large-|z| branch
        m(z) = sum_{k>=0} mu_series[k] / z^{k+1}.

    Convention here: choose mu0 (the leading coefficient) by solving the
    leading-diagonal equation and (by default) picking the root closest
    to -1, i.e. m(z) ~ -1/z.

    The returned 'moments(N)' are normalized density moments:
        mu_density[k] = mu_series[k] / mu_series[0]
    so mu_density[0] = 1.
    """

    def __init__(self, a, mu0=None):
        self.a = numpy.asarray(a)
        # Ensure valid
        self.a[-1, 0] = 0.0
        if self.a.ndim != 2:
            raise ValueError("a must be a 2D NumPy array with a[i,j]=a_{ij}.")

        self.I = self.a.shape[0] - 1                               # noqa: E741
        self.J = self.a.shape[1] - 1

        nz = numpy.argwhere(self.a != 0)
        if nz.size == 0:
            raise ValueError("All coefficients are zero.")

        # r = max(i-j) over nonzero terms
        self.r = int(numpy.max(nz[:, 0] - nz[:, 1]))

        # Group coefficients by diagonal offset s = r - (i-j) >= 0
        # diag[s] is list of (j, a_ij) for which i-j = r-s
        self.diag = {}
        for i, j in nz:
            i = int(i)
            j = int(j)
            coeff = self.a[i, j]
            s = self.r - (i - j)
            if s >= 0:
                self.diag.setdefault(int(s), []).append((j, coeff))

        # Choose mu0 (series leading coefficient). This should be
        # -1 for m(z) ~ -1/z, but it may only hold approximately.
        if mu0 is None:
            self.mu0 = self._solve_mu0()
        else:
            self.mu0 = mu0

        # Precompute mu0^p up to p=J
        self.mu0pow = [1]
        for _ in range(self.J):
            self.mu0pow.append(self.mu0pow[-1] * self.mu0)

        # Linear coefficient A0 = sum_{i-j=r} j a_ij mu0^{j-1}
        self.A0 = 0
        for j, coeff in self.diag.get(0, []):
            if j > 0:
                self.A0 += j * coeff * self.mu0pow[j - 1]
        if self.A0 == 0:
            raise ValueError("A0 is zero for this mu0; the sequential " +
                             "recursion is degenerate.")

        # Stored series moments mu_series[0..]
        self._mu = [self.mu0]

        # Convolution table c[j][n] = coefficient of w^n in (S(w))^j,
        # where S(w) = sum_{t>=0} mu_series[t] w^t and m(z)=w S(w), w=1/z.
        #
        # We store c as lists growing in n: c[j][n] for j=0..J.
        self._c = [[0] for _ in range(self.J + 1)]
        self._c[0][0] = 1
        for j in range(1, self.J + 1):
            self._c[j][0] = self.mu0pow[j]

    def _solve_mu0(self):
        # Leading diagonal polynomial L(m) = sum_{i-j=r} a_ij m^j.
        # That means i = j + r, so coefficient is a[j+r, j] if in bounds.
        coeffs = numpy.zeros(self.J + 1, dtype=numpy.complex128)
        for j in range(self.J + 1):
            i = j + self.r
            if 0 <= i <= self.I:
                coeffs[j] = self.a[i, j]

        if not numpy.any(coeffs != 0):
            raise ValueError("Leading diagonal polynomial is identically " +
                             "zero; cannot determine mu0.")

        deg = int(numpy.max(numpy.nonzero(coeffs)[0]))

        # descending powers for numpy.roots
        roots = numpy.roots(coeffs[:deg + 1][::-1])

        # Targetting mu0 = -1 for ~ -1/z asymptotics
        mu0 = roots[numpy.argmin(numpy.abs(roots + 1))]

        if abs(mu0.imag) < 1e-12:
            mu0 = mu0.real
        return mu0

    def _ensure(self, N):
        # Compute mu_series up to index N (inclusive)
        while len(self._mu) <= N:
            k = len(self._mu)  # compute mu_k

            # Compute f[j] = coefficient of w^k in (S_trunc(w))^j,
            # where S_trunc uses mu_0..mu_{k-1} only (i.e. mu_k treated as 0).
            # Key fact: in the true c[j,k], mu_k can only appear linearly as
            # j*mu_k*mu0^{j-1}.
            f = [0] * (self.J + 1)
            f[0] = 0
            for j in range(1, self.J + 1):
                ssum = 0
                # sum_{t=1..k-1} mu_t * c[j-1, k-t]
                for t in range(1, k):
                    ssum += self._mu[t] * self._c[j - 1][k - t]
                # recurrence: c[j,k] = mu0*c[j-1,k] + sum_{t=1..k-1}
                # mu_t*c[j-1,k-t] + mu_k*c[j-1,0] with mu_k=0 for f,
                # and c[j-1,k]=f[j-1]
                f[j] = self.mu0 * f[j - 1] + ssum

            # Build the linear equation for mu_k:
            # A0*mu_k + rest = 0
            rest = 0

            # s=0 diagonal contributes coeff*(f[j]) (the mu_k-free part)
            for j, coeff in self.diag.get(0, []):
                if j == 0:
                    # only affects k=0, but we never come here with k=0
                    continue
                rest += coeff * f[j]

            # lower diagonals s=1..k contribute coeff*c[j,k-s] (already known
            # since k-s < k)
            for s in range(1, k + 1):
                entries = self.diag.get(s)
                if not entries:
                    continue
                n = k - s
                for j, coeff in entries:
                    if j == 0:
                        if n == 0:
                            rest += coeff
                    else:
                        rest += coeff * self._c[j][n]

            mu_k = -rest / self.A0
            self._mu.append(mu_k)

            # Now append the new column k to c using the full convolution
            # recurrence:
            # c[j,k] = sum_{t=0..k} mu_t * c[j-1,k-t]
            for j in range(self.J + 1):
                self._c[j].append(0)

            self._c[0][k] = 0
            for j in range(1, self.J + 1):
                val = 0
                for t in range(0, k + 1):
                    val += self._mu[t] * self._c[j - 1][k - t]
                self._c[j][k] = val

    # --- API ---

    def __call__(self, k):
        self._ensure(k)
        return self._mu[k] / self._mu[0]

    def moments(self, N):
        # normalized density moments so moment 0 is 1
        self._ensure(N)
        mu0 = self._mu[0]
        return numpy.array([self._mu[k] / mu0 for k in range(N + 1)])

    def radius(self, N):
        # Estimate the radius of convergence of the Stieltjes
        # series
        if N < 3:
            raise RuntimeError("N is too small, choose a larger value.")
        self._ensure(N)
        return max([numpy.abs(self._mu[j] / self._mu[j-1])
                    for j in range(2, N+1)])

    def stieltjes(self, z, N):
        # Estimate Stieltjes transform (root) using moment
        # expansion
        z = numpy.asarray(z)
        mu = self.moments(N)
        return -numpy.sum(z[..., numpy.newaxis]**(-numpy.arange(N+1)-1) * mu,
                          axis=-1)
