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
import matplotlib

__all__ = ['rgb_hsv']


# =======
# rgb hsv
# =======

def rgb_hsv(c, shift=0.0, thresh=numpy.inf, n_mod=12.0, n_ph=12.0, vmin=0.35,
            vmax=1.0, tile_gamma=1.0, tile_mix=1.0):
    """
    Convert complex field c to RGB via HSV domain coloring.

    Parameters
    ----------
    c : array_like of complex
        Complex field.

    shift : float, default 0.0
        Phase offset in turns (1.0 = full 2*pi rotation). Applied to hue.

    thresh : float, default numpy.inf
        Optional cap on |c| used for magnitude-related terms. Use to prevent
        very large magnitudes from dominating the encoding.

    n_mod : float, default 12.0
        Number of modulus steps per 2*pi in log(|c|). Higher -> more concentric
        rings. Set to 0.0 to disable modulus stepping.

    n_ph : float, default 12.0
        Number of phase steps per 2*pi in arg(c). Higher -> more angular
        sectors. Set to 0.0 to disable phase stepping.

    vmin : float, default 0.35
        Minimum brightness for the tiling shading (darkest parts of tiles).

    vmax : float, default 1.0
        Maximum brightness for the tiling shading (brightest parts of tiles).
        Lowering vmax (e.g. 0.8-0.9) can reduce the "neon" look.

    tile_gamma : float, default 1.0
        Shapes the within-tile ramp. 1.0 = linear sawtooth. >1.0 makes tiles
        stay darker longer and brighten sharply near boundaries. <1.0 brightens
        earlier.

    tile_mix : float in [0, 1], default 1.0
        Mix between original magnitude brightness and tiling shading:
          0.0 -> value = 1 - exp(-|c|) (your original, no tiling influence)
          1.0 -> value = tiling shading only (Wegert-style tiling look)
        Intermediate values overlay tiling onto the original magnitude shading.

    Notes
    -----

    The coloring technique is inspired from [1]_.

    References
    ----------

    [1] Wegert, E. (2015) "Visual Complex Functions: An Introduction +with
        Phase Portraits", Springer.
        doi: https://doi.org/10.1007/978-3-0348-0180-5
    """

    hue = (numpy.angle(c) + numpy.pi) / (2.0 * numpy.pi)

    hue = (hue + shift) % 1.0

    r = numpy.abs(c)
    if numpy.isfinite(thresh):
        r = numpy.minimum(r, thresh)

    value0 = 1.0 - numpy.exp(-r)

    eps = 1e-300
    tau = 2.0 * numpy.pi

    g = numpy.ones_like(hue)

    if n_mod and n_mod > 0.0:
        x_mod = (n_mod / tau) * numpy.log(r + eps)
        g_mod = numpy.ceil(x_mod) - x_mod
        g = g * g_mod

    if n_ph and n_ph > 0.0:
        theta = (numpy.angle(c) + numpy.pi) % tau

        x_ph = (n_ph / tau) * theta
        g_ph = numpy.ceil(x_ph) - x_ph
        g = g * g_ph

    g = numpy.clip(g, 0.0, 1.0)
    if tile_gamma and tile_gamma != 1.0:
        g = g ** float(tile_gamma)

    vmin = float(numpy.clip(vmin, 0.0, 1.0))
    vmax = float(numpy.clip(vmax, 0.0, 1.0))
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    value_tile = vmin + (vmax - vmin) * g

    tile_mix = float(numpy.clip(tile_mix, 0.0, 1.0))
    value = (1.0 - tile_mix) * value0 + tile_mix * value_tile

    saturation = numpy.ones_like(hue)
    hsv = numpy.stack((hue, saturation, numpy.clip(value, 0.0, 1.0)), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    return rgb
