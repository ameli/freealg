# SPDX-FileCopyrightText: Copyright 2025, Siavash Ameli <sameli@berkeley.edu>
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

__all__ = ['auto_bins', 'hist']


# =========
# auto bins
# =========

def auto_bins(array, method='scott', factor=4):
    """
    Automatic choice for the number of bins for the histogram of an array.

    Parameters
    ----------

    array : numpy.array
        An array for histogram.

    method : {``'freedman'``, ``'scott'``, ``'sturges'``}, default= ``'scott'``
        Method of choosing number of bins.

    factor : int, default=3
        Multiply the number of bins buy a factor

    Returns
    -------

    num_bins : int
        Number of bins for histogram.

    See Also
    --------

    freealg.visualization.hist
    """

    if method == 'freedman':

        q75, q25 = numpy.percentile(array, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(array) ** (1/3))

        if bin_width == 0:
            # Fallback default
            return
            num_bins = 100
        else:
            num_bins = int(numpy.ceil((array.max() - array.min()) / bin_width))

    elif method == 'scott':

        std = numpy.std(array)
        bin_width = 3.5 * std / (len(array) ** (1/3))
        num_bins = int(numpy.ceil((array.max() - array.min()) / bin_width))

    elif method == 'sturges':

        num_bins = int(numpy.ceil(numpy.log2(len(array)) + 1))

    else:
        raise NotImplementedError('"method" is invalid.')

    return num_bins * factor


# ====
# hist
# ====

def hist(array, bins=None, m=8, density=True, support=None, atoms=None,
         edge_tol=1.0e-3, detect_bins=512, trim_q=0.01, smooth_w=7,
         merge_gap_bins=2, min_interval_bins=3, atom_exclude_sigma=3.0,
         return_support=False):
    """
    Histogram (optionally ASH-smoothed) with detected multi-interval support
    and atom-centered bins.

    This function produces histogram for empirical data with three features:

    * It can perform averaged shift histogram (ASH) to reduce the sensitivity
      of histogram values to bin selection.

    * It can detect multi-interval densities (separate bulks) and snap the bin
      edges to the edges of the density bulks for crisper histograms on the
      edges and reduce the smoothing artifact at sharp edges.

    * It ensures the bins on the atoms are centered so the spikes are shown
      right at the atom location.

    Parameters
    ----------

    array : array_like
        One-dimensional samples.

    bins : int, default=None
        Number of bins used for the absolutely-continuous (AC) part. Atom bins
        (if any) are added in addition to these bins. If `None`, number of bins
        are automatically detected using :func:`auto_bins`.

    m : int, default=8
        ASH smoothing parameter. Use m=1 for a plain histogram.

    density : bool, default=True
        If True, normalize counts by (n * bin_width) per bin.

    support : list of (float, float) or None, default=None
        AC support intervals [(a1, b1), ..., (ak, bk)]. If None, the AC support
        is detected from the samples using edge_tol and related parameters.

    atoms : list of float or None, default=None
        Atom locations only. A centered bin is created at each atom location
        (and atoms strictly inside AC intervals may also be carved into the AC
        binning).

    edge_tol : float, default=1e-3
        Relative threshold for AC support detection, as a fraction of the
        maximum coarse density.

    detect_bins : int, default=512
        Number of coarse bins used to detect the AC support.

    trim_q : float, default=0.01
        Quantile trimming fraction for robust detection range, using
        [trim_q, 1-trim_q].

    smooth_w : int, default=7
        Moving-average window size (in coarse bins) for detection smoothing.

    merge_gap_bins : int, default=2
        Merge detected intervals if the gap between them is at most this many
        coarse bins.

    min_interval_bins : int, default=3
        Drop detected intervals shorter than this many coarse bins.

    atom_exclude_sigma : float, default=3.0
        When detecting AC support, exclude a neighborhood around each atom
        location with radius atom_exclude_sigma * (coarse_bin_width).

    return_support : bool, default=False
        If `True`, the detected support is also returned.

    Returns
    -------

    edges : numpy.ndarray
        Bin edges of length (n_bins + 1), suitable for matplotlib stairs.

    vals : numpy.ndarray
        Bin heights of length n_bins. If density=True, vals integrates to the
        empirical mass within the plotted bins (up to smoothing/
        discretization).

    support : numpy.array
        If ``return_support=True``, this array is also returned.

    See Also
    --------

    freealg.visualization.auto_bins
    """

    x = numpy.asarray(array, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("array is empty.")

    if bins is None:
        bins = auto_bins(array, factor=2)
    else:
        bins = int(bins)
    if bins < 1:
        raise ValueError("bins must be >= 1.")

    m = int(m)
    if m < 1:
        m = 1

    # -------------------------
    # atoms: unique float locs
    # -------------------------

    # atoms can be either:
    #   - list/array of locations: [t1, t2, ...]
    #   - list/array of (location, weight): [(t1, w1), ...]
    if atoms is not None:
        if isinstance(atoms, tuple) and len(atoms) == 2:
            # Treat as a single (location, weight) atom.
            atoms = [atoms]
        elif isinstance(atoms, numpy.ndarray) and atoms.ndim == 1 and \
                atoms.size == 2:
            atoms = [tuple(atoms.tolist())]

    atom_locs = []
    if atoms is not None and len(atoms) > 0:
        a0 = atoms[0]
        if isinstance(a0, (list, tuple)) and len(a0) >= 1:
            atom_locs = sorted({float(loc) for loc, *_ in atoms})
        else:
            atom_locs = sorted({float(t) for t in atoms})

    # ------------------------
    # normalize support format
    # ------------------------

    if support is not None:
        if len(support) == 2 and not isinstance(support[0], (list, tuple)):
            support = [(float(support[0]), float(support[1]))]
        else:
            support = [(float(a), float(b)) for a, b in support]

    def _ash_uniform_segment(x_all, edges):

        edges = numpy.asarray(edges, dtype=float)
        if edges.size < 2:
            raise ValueError("segment edges too short.")
        h = float(edges[1] - edges[0])
        if not numpy.allclose(numpy.diff(edges), h, rtol=0.0, atol=1.0e-12):
            raise ValueError("ASH requires uniform bin widths within a "
                             "segment.")

        if m <= 1:
            c, _ = numpy.histogram(x_all, bins=edges)
            if density:
                v = c.astype(float) / (x.size * h)
            else:
                v = c.astype(float)
            return edges, v

        k = edges.size - 1
        fine_edges = edges[0] + (h / float(m)) * \
            numpy.arange(k * m + 1, dtype=float)
        fine_counts, _ = numpy.histogram(x_all, bins=fine_edges)

        w = numpy.concatenate(
            (numpy.arange(1, m + 1), numpy.arange(m - 1, 0, -1))).astype(float)
        fine_pad = numpy.pad(fine_counts, (0, 2 * m - 2), mode="constant")

        ash_counts = numpy.empty(k, dtype=float)
        for i in range(k):
            seg = fine_pad[i*m:i*m+(2*m-1)]
            ash_counts[i] = (seg @ w) / float(m)

        if density:
            v = ash_counts / (x.size * h)
        else:
            v = ash_counts
        return edges, v

    # -------------------------------------
    # empirical support detection (AC only)
    # -------------------------------------

    def _detect_support(x_in):

        # robust range
        q = float(trim_q)
        q = max(0.0, min(0.25, q))
        qlo = float(numpy.quantile(x_in, q))
        qhi = float(numpy.quantile(x_in, 1.0 - q))
        if not (qlo < qhi):
            qlo = float(numpy.min(x_in))
            qhi = float(numpy.max(x_in))
            if not (qlo < qhi):
                qlo -= 0.5
                qhi += 0.5

        nb = max(64, int(detect_bins))
        edges0 = numpy.linspace(qlo, qhi, nb + 1, dtype=float)
        h0 = float(edges0[1] - edges0[0])

        c0, _ = numpy.histogram(x_in, bins=edges0)
        d0 = c0.astype(float) / (x_in.size * h0)

        w = int(smooth_w)
        if w > 1:
            w = min(w, nb)
            ker = numpy.ones(w, dtype=float) / float(w)
            d1 = numpy.convolve(d0, ker, mode="same")
        else:
            d1 = d0

        mx = float(numpy.max(d1))
        if mx <= 0.0:
            return [(qlo, qhi)]

        thr = float(edge_tol) * mx
        mask = d1 > thr

        # intervals in coarse-bin indices [i, j)
        intervals = []
        i = 0
        n = mask.size
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= int(min_interval_bins):
                intervals.append((i, j))
            i = j

        if len(intervals) == 0:
            return [(qlo, qhi)]

        # merge small gaps
        merged = [intervals[0]]
        gap = int(merge_gap_bins)
        for i, j in intervals[1:]:
            pi, pj = merged[-1]
            if i - pj <= gap:
                merged[-1] = (pi, j)
            else:
                merged.append((i, j))

        # refine edges using actual samples within coarse window
        out = []
        for i, j in merged:
            peak = float(numpy.max(d1[i:j])) if j > i else 0.0
            thr_loc = float(edge_tol) * peak

            # use a looser threshold for edge expansion
            thr_expand = 0.25 * thr_loc   # try 0.1 if still too strict

            ii = i
            while ii > 0 and d1[ii - 1] > thr_expand:
                ii -= 1

            jj = j
            n = d1.size
            while jj < n and d1[jj] > thr_expand:
                jj += 1

            if peak <= 0.0:
                ii, jj = i, j
            jj = min(jj, d1.size)

            a0 = edges0[ii]
            b0 = edges0[jj]

            xx = x_in[(x_in >= a0) & (x_in <= b0)]
            if xx.size == 0:
                out.append((float(a0), float(b0)))
            else:
                out.append((float(xx.min()), float(xx.max())))
        return out

    # --------------------------
    # build AC support intervals
    # --------------------------

    if support is None:
        x_detect = x

        # exclude neighborhoods around atom locations so detection is AC-driven
        if atom_locs:
            # estimate a scale for exclusion width from a coarse bin width
            qlo = float(numpy.quantile(
                x_detect, max(0.0, min(0.25, float(trim_q)))))
            qhi = float(numpy.quantile(
                x_detect, 1.0 - max(0.0, min(0.25, float(trim_q)))))
            if qhi <= qlo:
                qlo = float(numpy.min(x_detect))
                qhi = float(numpy.max(x_detect))
            h0 = (qhi - qlo) / float(max(64, int(detect_bins)))
            # exclude +/- atom_exclude_sigma * h0
            rad = float(atom_exclude_sigma) * float(abs(h0))
            if rad <= 0.0:
                rad = 1e-12 * max(1.0, float(numpy.max(numpy.abs(x_detect))))
            for t in atom_locs:
                x_detect = x_detect[~((x_detect >= t - rad) &
                                      (x_detect <= t + rad))]

        # if everything got excluded, fall back
        if x_detect.size < 10:
            x_detect = x

        supp = _detect_support(x_detect)
    else:
        supp = list(support)

    # clean + sort + merge overlaps/touching
    supp2 = []
    for a, b in supp:
        a = float(a)
        b = float(b)
        if b <= a:
            continue
        supp2.append((a, b))
    supp2.sort(key=lambda t: t[0])
    if len(supp2) == 0:
        a0 = float(numpy.min(x))
        b0 = float(numpy.max(x))
        if not (a0 < b0):
            a0 -= 0.5
            b0 += 0.5
        supp2 = [(a0, b0)]

    merged = [supp2[0]]
    for a, b in supp2[1:]:
        pa, pb = merged[-1]
        if a <= pb:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    supp = merged

    # --------------------------
    # choose a base AC bin width
    # --------------------------

    total_len = sum(b - a for a, b in supp)
    if total_len <= 0.0:
        raise ValueError("support total length is zero.")
    # bins apply to AC only
    h_ac = total_len / float(bins)

    # or 3*h_ac; key idea: merge gaps smaller than a few final bin widths
    h_merge = 2.0 * h_ac

    # IMPORTANT: only merge close bulks if support was *detected*.
    # If user provided support, trust it and do NOT merge.
    if support is None:
        supp_merged = [supp[0]]
        for a, b in supp[1:]:
            pa, pb = supp_merged[-1]
            if a - pb <= h_merge:
                supp_merged[-1] = (pa, max(pb, b))
            else:
                supp_merged.append((a, b))
        supp = supp_merged

    # ------------------------------
    # prepare atom bins (standalone)
    # ------------------------------

    # atom bin width: match AC bin width
    w_atom = float(h_ac)

    atom_bins = []
    for t in atom_locs:
        left = t - 0.5 * w_atom
        right = t + 0.5 * w_atom
        atom_bins.append((left, right))
    atom_bins.sort(key=lambda p: p[0])

    # merge overlapping atom bins (rare, but safe)
    atom_bins2 = []
    for left, right in atom_bins:
        if not atom_bins2:
            atom_bins2.append((left, right))
        else:
            pl, pr = atom_bins2[-1]
            if left <= pr:
                atom_bins2[-1] = (pl, max(pr, right))
            else:
                atom_bins2.append((left, right))
    atom_bins = atom_bins2

    # -------------
    # build AC bins
    # -------------

    # optionally carve atom bins strictly inside interval. We carve only if
    # atom is well inside (a+0.5w, b-0.5w) and the carved bin doesn't overlap
    # another carved bin.
    # list of (left,right, kind) where kind in {"ac","atom"}
    ac_bins = []
    for a, b in supp:
        L = b - a
        n_int = max(1, int(round(L / h_ac)))
        # ensure interval edges snapped exactly
        edges_int = numpy.linspace(a, b, n_int + 1, dtype=float)

        # atoms that are strictly inside this interval and not near edges
        inside = []
        for t in atom_locs:
            if (t > a + 0.5 * w_atom) and (t < b - 0.5 * w_atom):
                inside.append(t)
        inside.sort()

        # carve bins: start with continuous pieces from edges_int, then insert
        # atom bins We do it by splitting [a,b] at atom-bin boundaries and then
        # re-binning AC pieces uniformly.
        cut_points = [a]
        carved_bins = []
        prev_right = -numpy.inf
        for t in inside:
            left = t - 0.5 * w_atom
            right = t + 0.5 * w_atom
            # avoid overlap among carved atom bins
            if left <= prev_right:
                continue
            carved_bins.append((left, right))
            cut_points.extend([left, right])
            prev_right = right
        cut_points.append(b)
        cut_points = sorted(cut_points)

        # continuous segments are the gaps between carved atom bins
        cont = []
        i = 0
        while i < len(cut_points) - 1:
            aa = cut_points[i]
            bb = cut_points[i + 1]
            if bb > aa:
                cont.append((aa, bb))
            i += 1

        # But cont includes atom bins too; label them:
        # We'll add atom bins explicitly, and AC bins for the remaining
        # segments. First, mark carved atom bins as bins (kind="atom_in_ac")
        # to preserve exact width/centering.
        carved_set = {(l, r) for (l, r) in carved_bins}

        # Allocate AC bins within this interval excluding carved atom widths
        L_ac = L - sum(r - ell for ell, r in carved_bins)
        n_ac = max(1, int(round(L_ac / h_ac))) if L_ac > 0 else 0

        # Collect AC segments (excluding carved atom segments) and distribute
        # bins by length
        ac_segs = []
        for aa, bb in cont:
            if (aa, bb) in carved_set:
                continue
            if bb > aa:
                ac_segs.append((aa, bb))

        if n_ac > 0 and ac_segs:
            seg_len = numpy.array([bb - aa for aa, bb in ac_segs], dtype=float)
            raw = n_ac * (seg_len / float(seg_len.sum()))
            nseg = numpy.floor(raw).astype(int)
            rem = n_ac - int(nseg.sum())
            if rem > 0:
                order = numpy.argsort(-(raw - nseg))
                for j in order[:rem]:
                    nseg[j] += 1
            # ensure each segment with length gets at least 1 if possible
            for j in range(len(ac_segs)):
                if (ac_segs[j][1] - ac_segs[j][0]) > 0 and nseg[j] == 0 and \
                        n_ac > 0:
                    k = int(numpy.argmax(nseg))
                    if nseg[k] > 1:
                        nseg[k] -= 1
                        nseg[j] = 1
        else:
            nseg = numpy.zeros(len(ac_segs), dtype=int)

        # Build bins in order within [a,b]
        pieces = []
        for (l, r) in carved_bins:
            pieces.append((l, r, "atom"))
        for (aa, bb), ns in zip(ac_segs, list(nseg) if ac_segs else []):
            if ns <= 0:
                continue
            e = numpy.linspace(aa, bb, int(ns) + 1, dtype=float)
            for j in range(e.size - 1):
                pieces.append((float(e[j]), float(e[j + 1]), "ac"))
        pieces.sort(key=lambda p: p[0])

        # If no carving happened, fall back to straight uniform bins for this
        # interval
        if not pieces:
            e = edges_int
            for j in range(e.size - 1):
                ac_bins.append((float(e[j]), float(e[j + 1]), "ac"))
        else:
            # Also ensure we didn't miss left/right ends due to rounding; fill
            # gaps with AC bins
            cur = a
            for left, right, kind in pieces:
                if left > cur + 1e-15:
                    # fill gap with AC bins (at least 1)
                    ng = max(1, int(round((left - cur) / h_ac)))
                    eg = numpy.linspace(cur, left, ng + 1, dtype=float)
                    for j in range(eg.size - 1):
                        ac_bins.append((float(eg[j]), float(eg[j + 1]), "ac"))
                ac_bins.append((left, right, "atom"))
                cur = right
            if b > cur + 1e-15:
                ng = max(1, int(round((b - cur) / h_ac)))
                eg = numpy.linspace(cur, b, ng + 1, dtype=float)
                for j in range(eg.size - 1):
                    ac_bins.append((float(eg[j]), float(eg[j + 1]), "ac"))

    # -----------------
    # combine atom bins
    # -----------------

    # (standalone) + AC bins, then insert zero-gap bins
    all_bins = []

    # standalone atoms first (always included, centered)
    for left, right in atom_bins:
        all_bins.append((float(left), float(right), "atom"))

    # AC bins (already snapped to detected support edges)
    for left, right, kind in ac_bins:
        all_bins.append((float(left), float(right), kind))

    all_bins.sort(key=lambda p: p[0])

    # merge/resolve overlaps by priority: atom bins override (keep), AC bins
    # trimmed
    resolved = []
    for left, right, kind in all_bins:
        if right <= left:
            continue
        if not resolved:
            resolved.append([left, right, kind])
            continue
        pl, pr, pk = resolved[-1]
        if left >= pr:
            resolved.append([left, right, kind])
            continue
        # overlap
        if pk.startswith("atom"):
            # trim current to start at pr
            if right > pr:
                resolved.append([pr, right, kind])
        else:
            if kind.startswith("atom"):
                # trim previous to end at left, then place atom
                resolved[-1][1] = min(pr, left)
                if resolved[-1][1] <= resolved[-1][0]:
                    resolved.pop()
                resolved.append([left, right, kind])
            else:
                # both AC: merge by extending right
                resolved[-1][1] = max(pr, right)

    # Ensure full data range is covered (never truncate histogram)
    x_min = float(numpy.min(x))
    x_max = float(numpy.max(x))

    # Base width: use h_ac if available; otherwise fallback to global width
    try:
        h_fill = float(h_ac)
    except NameError:
        h_fill = (x_max - x_min) / float(max(1, bins))

    # Build anchors: min/max, support edges, and atom-bin boundaries
    anchors = [float(x_min), float(x_max)]

    # support edges: use supp intervals (already normalized earlier)
    for a, b in supp:
        anchors.append(float(a))
        anchors.append(float(b))

    # atom-bin boundaries: include both standalone atom bins and carved atom
    # bins (atom bins are those with kind starting with "atom" in resolved/
    # all_bins paths) We already have atom_bins (standalone) and also "atom"/
    # "atom_in_ac" pieces merged into resolved. Use resolved if available; else
    # fallback to atom_bins.
    try:
        _bins_for_atoms = resolved
    except NameError:
        _bins_for_atoms = []

    for left, right, kind in _bins_for_atoms:
        if str(kind).startswith("atom"):
            anchors.append(float(left))
            anchors.append(float(right))
    for left, right in atom_bins:
        anchors.append(float(left))
        anchors.append(float(right))

    # uniquify/sort anchors and drop near-duplicates
    anchors = sorted(set(anchors))
    eps = 1e-12 * max(1.0, abs(x_max - x_min), abs(x_min), abs(x_max))
    anchors2 = [anchors[0]]
    for a in anchors[1:]:
        if a > anchors2[-1] + eps:
            anchors2.append(a)
    anchors = anchors2

    # Force atom-centered bins: add [t-h/2, t+h/2] explicitly and
    # prevent any anchor from splitting inside these atom bins.
    atom_intervals = []
    for t in atom_locs:
        ell = float(t) - 0.5 * float(h_fill)
        r = float(t) + 0.5 * float(h_fill)
        atom_intervals.append((ell, r))

    # add atom boundaries as anchors
    for ell, r in atom_intervals:
        anchors.append(ell)
        anchors.append(r)

    anchors = sorted(set(anchors))

    # remove anchors strictly inside any atom interval (so atom bin won't be
    # split)
    anchors2 = []
    for a in anchors:
        inside = False
        for ell, r in atom_intervals:
            if (a > ell + eps) and (a < r - eps):
                inside = True
                break
        if not inside:
            anchors2.append(a)
    anchors = anchors2

    # If an atom-centered bin extends beyond the AC range endpoint(s), do not
    # keep the AC endpoint anchor there; otherwise we create a tiny "sliver"
    # bin between the endpoint and the atom-bin boundary.
    if atom_intervals:
        for ell, r in atom_intervals:
            # atom bin starts to the right of x_max (AC max)
            if (ell > x_max + eps) and ((ell - x_max) < 0.8 * h_fill):
                anchors = [aa for aa in anchors
                           if not numpy.isclose(aa, x_max, atol=eps, rtol=0.0)]
            # atom bin ends to the left of x_min (AC min)
            if (r < x_min - eps) and ((x_min - r) < 0.8 * h_fill):
                anchors = [aa for aa in anchors
                           if not numpy.isclose(aa, x_min, atol=eps, rtol=0.0)]
        anchors = sorted(set(anchors))

    # Helper: check if [a,b] is an atom bin we must keep as-is
    _atom_set = set()
    for left, right in atom_bins:
        _atom_set.add((float(left), float(right)))
    for left, right, kind in _bins_for_atoms:
        if str(kind).startswith("atom"):
            _atom_set.add((float(left), float(right)))

    # also force the atom-centered intervals [t-h_fill/2, t+h_fill/2]
    for left, right in atom_intervals:
        _atom_set.add((float(left), float(right)))

    # Rebuild bins piecewise-uniform between anchors
    bins_with_gaps = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b <= a + eps:
            continue

        if (a, b) in _atom_set:
            bins_with_gaps.append((float(a), float(b), "atom"))
            continue

        nseg = max(1, int(numpy.round((b - a) / h_fill)))
        e = numpy.linspace(a, b, nseg + 1, dtype=float)
        for j in range(e.size - 1):
            bins_with_gaps.append((float(e[j]), float(e[j + 1]), "ac"))

    # ---------------------------------------------------------
    # Remove tiny "sliver" AC bins (can happen near data extrema)
    # ---------------------------------------------------------
    # Typical AC bin width
    _ac_widths = [float(r - l) for (l, r, k) in bins_with_gaps
                  if str(k).startswith("ac") and (r - l) > 0.0]
    if len(_ac_widths) > 0:
        _h_typ = float(numpy.median(_ac_widths))
        # Anything below this is considered a sliver
        _h_min = 0.50 * _h_typ

        _i = 0
        while _i < len(bins_with_gaps):
            l, r, k = bins_with_gaps[_i]
            w = float(r - l)

            if (str(k).startswith("ac")) and (w > 0.0) and (w < _h_min):
                # Prefer merging with left neighbor if it is also AC
                if _i > 0 and str(bins_with_gaps[_i - 1][2]).startswith("ac"):
                    pl, pr, pk = bins_with_gaps[_i - 1]
                    bins_with_gaps[_i - 1] = (float(pl), float(r), "ac")
                    del bins_with_gaps[_i]
                    # step back one to allow cascading merges
                    _i = max(_i - 1, 0)
                    continue

                # Otherwise merge into right neighbor if it is AC
                if _i + 1 < len(bins_with_gaps) and \
                        str(bins_with_gaps[_i + 1][2]).startswith("ac"):
                    nl, nr, nk = bins_with_gaps[_i + 1]
                    bins_with_gaps[_i + 1] = (float(l), float(nr), "ac")
                    del bins_with_gaps[_i]
                    _i = max(_i - 1, 0)
                    continue

            _i += 1

    # ----------------------
    # compute values per bin
    # ----------------------

    # (ASH only on AC bins with uniform consecutive edges) Build edges/vals by
    # grouping consecutive AC bins that are uniform width
    edges_out = [bins_with_gaps[0][0]]
    vals_out = []

    def _append_bin_value(left, right, val):

        if not numpy.isclose(edges_out[-1], left, atol=1e-12, rtol=0.0):
            edges_out.append(left)
        edges_out.append(right)
        vals_out.append(val)

    i = 0
    while i < len(bins_with_gaps):
        left, right, kind = bins_with_gaps[i]
        if kind.startswith("atom"):
            c, _ = numpy.histogram(x, bins=[left, right])
            w = right - left
            if density and w > 0:
                v = float(c[0]) / (x.size * w)
            else:
                v = float(c[0])
            _append_bin_value(left, right, v)
            i += 1
            continue

        # AC run: collect consecutive AC bins with equal width
        run = [(left, right)]
        h = right - left
        j = i + 1
        while j < len(bins_with_gaps):
            l2, r2, k2 = bins_with_gaps[j]
            if k2 != "ac":
                break
            if not numpy.isclose((r2 - l2), h, atol=1e-12, rtol=0.0):
                break
            if not numpy.isclose(run[-1][1], l2, atol=1e-12, rtol=0.0):
                break
            run.append((l2, r2))
            j += 1

        # ASH on this uniform run
        e = numpy.array([run[0][0]] + [r for _, r in run], dtype=float)
        e2, v2 = _ash_uniform_segment(x, e)

        # append run results
        for k in range(v2.size):
            _append_bin_value(float(e2[k]), float(e2[k + 1]), float(v2[k]))

        i = j

    edges = numpy.asarray(edges_out, dtype=float)
    vals = numpy.asarray(vals_out, dtype=float)

    # sanity
    if edges.size != vals.size + 1:
        raise RuntimeError("internal error: edges/vals mismatch.")
    if not numpy.all(numpy.diff(edges) >= 0.0):
        raise RuntimeError("edges are not monotone.")

    if return_support:
        return edges, vals, supp
    else:
        return edges, vals
