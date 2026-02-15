# SPDX-FileCopyrightText: Copyright 2025, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import numpy
from scipy.interpolate import interp1d
from ..visualization._plot_util import plot_samples, plot_branches


try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import qmc


# =================
# Base Distribution
# =================

class BaseDistribution(object):
    """
    Base class for distributions.
    """

    # ====
    # init
    # ====

    def __init__(self):
        """
        """

        self.lam_m = None
        self.lam_p = None
        self.num_roots = None

    # ======
    # sample
    # ======

    def sample(self, size, x_min=None, x_max=None, method='qmc', seed=None,
               plot=False, latex=False, save=False):
        """
        Sample from distribution.

        Parameters
        ----------

        size : int
            Size of sample.

        x_min : float, default=None
            Minimum of sample values. If `None`, the left edge of the support
            is used.

        x_max : float, default=None
            Maximum of sample values. If `None`, the right edge of the support
            is used.

        method : {``'mc'``, ``'qmc'``}, default= ``'qmc'``
            Method of drawing samples from uniform distribution:

            * ``'mc'``: Monte Carlo
            * ``'qmc'``: Quasi Monte Carlo

        seed : int, default=None,
            Seed for random number generator.

        plot : bool, default=False
            If `True`, samples histogram is plotted.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        Returns
        -------

        s : numpy.ndarray
            Samples.

        Notes
        -----

        This method uses inverse transform sampling.

        Examples
        --------

        .. code-block::python

            >>> from freealg.distributions import KestenMcKay
            >>> km = KestenMcKay(3)
            >>> s = km.sample(2000)

        .. image:: ../_static/images/plots/km_samples.png
            :align: center
            :class: custom-dark
        """

        if x_min is None:
            x_min = self.lam_m

        if x_max is None:
            x_max = self.lam_p

        # Grid and PDF
        xs = numpy.linspace(x_min, x_max, size)
        pdf = self.density(xs)

        # CDF (using cumulative trapezoidal rule)
        cdf = cumtrapz(pdf, xs, initial=0)
        cdf /= cdf[-1]  # normalize CDF to 1

        # Inverse CDF interpolator
        inv_cdf = interp1d(cdf, xs, bounds_error=False,
                           fill_value=(x_min, x_max))

        # Random generator
        rng = numpy.random.default_rng(seed)

        # Draw from uniform distribution
        if method == 'mc':
            u = rng.random(size)

        elif method == 'qmc':
            try:
                engine = qmc.Halton(d=1, scramble=True, rng=rng)
            except TypeError:
                # Older scipy versions
                engine = qmc.Halton(d=1, scramble=True, seed=rng)
            u = engine.random(size).ravel()

        else:
            raise NotImplementedError('"method" is invalid.')

        # Draw from distribution by mapping from inverse CDF
        samples = inv_cdf(u).ravel()

        if plot:
            radius = 0.5 * (self.lam_p - self.lam_m)
            center = 0.5 * (self.lam_p + self.lam_m)
            scale = 1.25
            x_min = numpy.floor(center - radius * scale)
            x_max = numpy.ceil(center + radius * scale)
            x = numpy.linspace(x_min, x_max, 500)
            rho = self.density(x)
            plot_samples(x, rho, x_min, x_max, samples, latex=latex, save=save)

        return samples

    # =============
    # plot branches
    # =============

    def plot_branches(self, x=None, y=None, latex=False, save=False, **kwargs):
        """
        Plot branches of the spectral curve of Stieltjes transform.

        Parameters
        ----------

        x : numpy.array, default=None
            The x axis of the grid where the Stieltjes transform is evaluated.
            If `None`, an interval slightly larger than the support interval of
            the spectral density is used.

        y : numpy.array, default=None
            The y axis of the grid where the Stieltjes transform is evaluated.
            If `None`, a grid on the interval ``[-1, 1]`` is used.

        latex : bool, default=False
            If `True`, the plot is rendered using LaTeX. This option is
            relevant only if ``plot=True``.

        save : bool, default=False
            If not `False`, the plot is saved. If a string is given, it is
            assumed to the save filename (with the file extension). This option
            is relevant only if ``plot=True``.

        **kwargs : dict
            Parameters to pass to
            :func:`freealg.visualization.domain_coloring`.

        See Also
        --------

        stieltjes

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 11

            >>> import numpy
            >>> from freealg.distributions import FreeLevy

            >>> # Create an object of the class
            >>> fl = FreeLevy(t=[2.0, 5.5], w=[0.75, 1-0.75], lam=0.1, a=0,
            ...               sigma=0.9)

            >>> # Plot on a grid
            >>> x = numpy.linspace(0, 4)
            >>> y = numpy.linspace(-1, 1)
            >>> fl.plot_branches(x, y)
        """

        # Create x if not given
        if x is None:
            radius = 0.5 * (self.lam_ub - self.lam_lb)
            center = 0.5 * (self.lam_ub + self.lam_lb)
            scale = 2.0
            x_min = numpy.floor(
                2.0 * (center - 2.0 * radius * scale)) / 2.0
            x_max = numpy.ceil(
                2.0 * (center + 2.0 * radius * scale)) / 2.0
            x = numpy.linspace(x_min, x_max, 400)

        # Create y if not given
        if y is None:
            y = numpy.linspace(-1, 1, 400)

        if y.size % 2 != 0:
            raise ValueError('Size of "y" should be even.')

        X, Y = numpy.meshgrid(x, y)
        z = X + 1j * Y

        m1 = self.stieltjes(z)
        roots_ = self.roots(z.ravel())
        support = self.support()

        plot_branches(z, m1, roots_, support, latex=latex, save=save, **kwargs)
